#!/usr/bin/env python3
import asyncio
import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import numpy as np
from monarch.actor import Actor, endpoint

from .utils import (
    extract_choice_from_continuation,
    find_block_list,
    load_model_and_tokenizer,
    load_steer_vector,
    model_slug,
    patch_tqdm_disable,
    resolve_tasks,
    task_scores_to_dict,
)

# --- Make logs safe for hyperactor_mesh (avoid Unicode tqdm bars, HF datasets bars, etc.) ---
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

# DeepEval telemetry/tracing off (best-effort)
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("OTEL_TRACES_EXPORTER", "none")
os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")
os.environ.setdefault("OTEL_LOGS_EXPORTER", "none")


patch_tqdm_disable()


@dataclass
class MMLUEvalConfig:
    dtype: str = "float32"
    seed: int = 42
    alpha_start: float = -10.0
    alpha_end: float = 10.0
    alpha_steps: int = 200
    tasks: Optional[Sequence[str]] = None  # None/[]/"all" => all tasks
    n_shots: int = 5
    apply_last_token_only: bool = False
    max_new_tokens: int = 8
    temperature: float = 0.0
    batch_size: int = 64
    use_chat_template: bool = True
    progress_every: int = 1  # mailbox yield every N alphas


class MMLUActor(Actor):
    """One actor per GPU. Computes MMLU scores over a grid of alpha for specified layers."""

    def __init__(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        self.current_model_name: Optional[str] = None
        self.current_dtype: Optional[str] = None
        self.tokenizer = None
        self.model = None

    def _ensure_model(self, model_name: str, dtype_str: str):
        if self.model is not None and self.current_model_name == model_name and self.current_dtype == dtype_str:
            return
        self.tokenizer = None
        self.model = None
        torch.cuda.empty_cache()
        self.tokenizer, self.model = load_model_and_tokenizer(model_name, dtype_str)
        self.current_model_name = model_name
        self.current_dtype = dtype_str

    @endpoint
    async def compute_mmlu(
        self,
        model_name: str,
        concept_slug: str,
        concept_label: str,
        steer_dir: str,
        save_dir: str,
        block_idx_to_steer,  # list[int] or [None] for all
        layer_path: Optional[str] = None,
        cfg_dict: Optional[dict] = None,
        rank_hint: int = 0,
    ):
        cfg = MMLUEvalConfig(**(cfg_dict or {}))
        cfg.n_shots = int(min(max(cfg.n_shots, 0), 5))
        cfg.batch_size = int(max(cfg.batch_size, 1))

        torch.manual_seed(cfg.seed + int(rank_hint))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed + int(rank_hint))

        # Import DeepEval only after env/patching above
        try:
            from deepeval.benchmarks import MMLU
            from deepeval.benchmarks.mmlu.task import MMLUTask
            try:
                from deepeval.models import DeepEvalBaseLLM
            except Exception:
                from deepeval.models.base_model import DeepEvalBaseLLM
        except Exception as e:
            return {"ok": False, "error": f"Failed to import deepeval: {e}"}

        self._ensure_model(model_name, cfg.dtype)
        tokenizer, model = self.tokenizer, self.model
        model.eval()
        device = next(model.parameters()).device

        blocks = find_block_list(model, override_path=layer_path)
        n_blocks = len(blocks)
        if block_idx_to_steer == None:
            block_idx_to_steer = list(range(n_blocks))
        else:
            # Interpret integer input as "sample this many layers uniformly across depth".
            block_idx_to_steer = np.linspace(0, n_blocks-1, num = block_idx_to_steer, dtype = int).tolist()


        # α grid (ensure 0 present)
        alphas = torch.linspace(cfg.alpha_start, cfg.alpha_end, steps=cfg.alpha_steps, dtype=torch.float32)
        if not (alphas == 0.0).any():
            alphas = torch.sort(torch.cat([alphas, torch.tensor([0.0], dtype=alphas.dtype)]))[0]
        alpha_list = [float(a.item()) for a in alphas]

        tasks = resolve_tasks(cfg.tasks, MMLUTask)
        # DeepEval prints task_scores with lowercase task ids (example in docs),
        # so prefer Enum.value when available.
        task_ids = [str(getattr(t, "value", t.name)).strip() for t in tasks]

        save_root = Path(save_dir) / model_slug(model_name) / concept_slug
        save_root.mkdir(parents=True, exist_ok=True)

        steer_dir_path = Path(steer_dir)
        saved_paths: list[str] = []
        results: list[dict[str, Any]] = []

        progress_mod = max(1, int(cfg.progress_every))
        eval_counter = 0

        # ---- DeepEval model wrapper (must return a SINGLE LETTER for MMLU exact-match scoring) ----
        class _SteeredLLM(DeepEvalBaseLLM):
            def __init__(
                self,
                *,
                model,
                tokenizer,
                block,
                steer_vec: torch.Tensor,
                alpha: float,
                apply_last_token_only: bool,
                max_new_tokens: int,
                temperature: float,
                use_chat_template: bool,
                name: str,
            ):
                self._model = model
                self._tokenizer = tokenizer
                self._block = block
                self._steer_vec = steer_vec
                self._alpha = float(alpha)
                self._apply_last_token_only = bool(apply_last_token_only)
                self._max_new_tokens = int(max_new_tokens)
                self._temperature = float(temperature)
                self._use_chat_template = bool(use_chat_template)
                self._name = name
                self._device = next(model.parameters()).device

            def load_model(self):
                return self._model

            def get_model_name(self):
                return self._name

            def _maybe_chat_wrap(self, prompt: str) -> str:
                if not self._use_chat_template:
                    return prompt
                if hasattr(self._tokenizer, "apply_chat_template"):
                    try:
                        return self._tokenizer.apply_chat_template(
                            [{"role": "user", "content": prompt}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    except Exception:
                        return prompt
                return prompt

            def _make_hook(self):
                alpha_vec = (self._alpha * self._steer_vec).to(self._device)

                def hook(module, inputs, output):
                    x = output[0] if isinstance(output, (tuple, list)) else output  # [B,T,H]
                    add = alpha_vec.to(dtype=x.dtype)[None, None, :]                # [1,1,H]
                    if self._apply_last_token_only:
                        x2 = x.clone()
                        x2[:, -1, :] = x2[:, -1, :] + add[0, 0, :]
                        x_steered = x2
                    else:
                        x_steered = x + add

                    if isinstance(output, (tuple, list)):
                        out = list(output)
                        out[0] = x_steered
                        return tuple(out)
                    return x_steered

                return hook

            def _generate_letters(self, prompts: list[str]) -> list[str]:
                prompts = [self._maybe_chat_wrap(p) for p in prompts]

                old_side = getattr(self._tokenizer, "padding_side", "right")
                self._tokenizer.padding_side = "left"
                try:
                    inputs = self._tokenizer(prompts, return_tensors="pt", padding=True).to(self._device)
                finally:
                    self._tokenizer.padding_side = old_side

                pad_id = self._tokenizer.pad_token_id
                if pad_id is None:
                    pad_id = self._tokenizer.eos_token_id

                gen_kwargs = dict(
                    max_new_tokens=self._max_new_tokens,
                    do_sample=self._temperature > 0,
                    pad_token_id=pad_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
                if self._temperature > 0:
                    gen_kwargs["temperature"] = self._temperature

                # IMPORTANT: slice generated tokens after the (padded) prompt length
                prompt_len = inputs["input_ids"].shape[1]

                handle = self._block.register_forward_hook(self._make_hook())
                try:
                    with torch.inference_mode():
                        out_ids = self._model.generate(**inputs, **gen_kwargs)
                finally:
                    handle.remove()

                letters: list[str] = []
                for i in range(out_ids.shape[0]):
                    gen_part = out_ids[i, prompt_len:]
                    cont = self._tokenizer.decode(gen_part, skip_special_tokens=True)
                    choice = extract_choice_from_continuation(cont) or "A"
                    letters.append(choice)
                return letters

            # DeepEval calls these
            def generate(self, prompt: str, schema=None) -> str:
                # schema is ignored for MMLU but kept for API compatibility
                return self._generate_letters([prompt])[0]

            async def a_generate(self, prompt: str, schema=None) -> str:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, self.generate, prompt)

            def batch_generate(self, prompts: list[str]) -> list[str]:
                return self._generate_letters(prompts)

        for block_idx in block_idx_to_steer:
            block_idx = int(block_idx)

            steer_vec_cpu = load_steer_vector(steer_dir_path, model_name, concept_slug, block_idx)
            steer_vec = steer_vec_cpu.to(device, non_blocking=True)

            overall_scores: list[Optional[float]] = []
            task_scores_by_task: dict[str, list[Optional[float]]] = {tid: [] for tid in task_ids}
            prediction_counts: list[Optional[dict[str, int]]] = []
            errors: list[Optional[str]] = []

            for alpha in alpha_list:
                wrapped = _SteeredLLM(
                    model=model,
                    tokenizer=tokenizer,
                    block=blocks[block_idx],
                    steer_vec=steer_vec,
                    alpha=alpha,
                    apply_last_token_only=cfg.apply_last_token_only,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    use_chat_template=cfg.use_chat_template,
                    name=f"{model_name}|{concept_label}|layer={block_idx}|alpha={alpha:g}",
                )

                def run_eval():
                    # DeepEval internally drives prompt generation; wrapper injects steering via hook.
                    benchmark = MMLU(tasks=tasks, n_shots=int(cfg.n_shots))

                    # Newer DeepEval: evaluate() returns score; older: sets .overall_score
                    # Suppress noisy internal printing.
                    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                        eval_res = benchmark.evaluate(model=wrapped, batch_size=int(cfg.batch_size))

                    overall: Optional[float] = None

                    # 1) Legacy style: attribute on benchmark
                    val = getattr(benchmark, "overall_score", None)
                    if isinstance(val, (int, float)):
                        overall = float(val)

                    # 2) New style: evaluate() returns raw float
                    if overall is None and isinstance(eval_res, (int, float)):
                        overall = float(eval_res)

                    # 3) Hybrid style: result object with .overall_score
                    if overall is None and hasattr(eval_res, "overall_score"):
                        val = getattr(eval_res, "overall_score")
                        if isinstance(val, (int, float)):
                            overall = float(val)

                    # Per-task scores
                    task_scores_raw = getattr(benchmark, "task_scores", None)
                    if task_scores_raw is None and hasattr(eval_res, "task_scores"):
                        task_scores_raw = getattr(eval_res, "task_scores")
                    per_task = task_scores_to_dict(task_scores_raw)

                    # Debug: distribution of predictions
                    pred_counts = None
                    try:
                        preds = getattr(benchmark, "predictions", None)
                        if preds is None and hasattr(eval_res, "predictions"):
                            preds = getattr(eval_res, "predictions")
                        if preds is not None and hasattr(preds, "columns") and "Prediction" in preds.columns:
                            vc = preds["Prediction"].value_counts().to_dict()
                            pred_counts = {str(k): int(v) for k, v in vc.items()}
                    except Exception:
                        pred_counts = None

                    return (float(overall) if overall is not None else None), per_task, pred_counts

                try:
                    overall, per_task, pred_counts = await asyncio.to_thread(run_eval)
                    errors.append(None)
                except Exception as e:
                    overall, per_task, pred_counts = None, None, None
                    errors.append(f"{type(e).__name__}: {e}")

                overall_scores.append(overall)
                prediction_counts.append(pred_counts)

                for tid in task_ids:
                    task_scores_by_task[tid].append(per_task.get(tid) if per_task else None)

                eval_counter += 1
                if eval_counter % progress_mod == 0:
                    await asyncio.sleep(0)

            payload: dict[str, Any] = {
                "model": model_name,
                "concept": concept_label,
                "concept_slug": concept_slug,
                "layer_idx": block_idx,
                "alphas": alpha_list,
                "overall_scores": overall_scores,
                "task_scores": task_scores_by_task,
                "prediction_counts": prediction_counts,
                "tasks": task_ids,
                "n_shots": int(cfg.n_shots),
                "apply_last_token_only": bool(cfg.apply_last_token_only),
                "max_new_tokens": int(cfg.max_new_tokens),
                "temperature": float(cfg.temperature),
                "batch_size": int(cfg.batch_size),
                "use_chat_template": bool(cfg.use_chat_template),
                "dtype": cfg.dtype,
                "errors": errors,
            }

            out_path = save_root / f"layer_{block_idx}_mmlu.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True, indent=2)
            saved_paths.append(str(out_path))
            results.append(payload)

        torch.cuda.empty_cache()
        return {"ok": True, "saved": saved_paths, "results": results}
