"""Continuous concept-probability scoring with a separate judge LLM."""

from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from monarch.actor import Actor, endpoint

from .utils import (
    chunked_with_bounds,
    count_negative_prompts,
    ensure_pad_token,
    find_block_list,
    load_contexts_for_concept,
    load_model_and_tokenizer,
    load_steer_vector,
    maybe_apply_chat_template,
    model_slug,
    set_left_padding,
)


_SCALAR_PATTERN = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$"
)


def parse_unit_interval_score(text: str) -> float:
    """Read one numeric score from the first nonblank output line."""
    if not isinstance(text, str):
        raise ValueError("Judge output must be text.")

    first_line = next((line.strip() for line in text.splitlines() if line.strip()), None)
    if first_line is None:
        raise ValueError("Judge output does not contain a score.")
    if _SCALAR_PATTERN.fullmatch(first_line) is None:
        raise ValueError(
            "The first nonblank judge-output line must contain only one numeric "
            f"score, got {first_line!r}."
        )

    score = float(first_line)
    if not math.isfinite(score):
        raise ValueError(f"Judge score must be finite, got {first_line!r}.")
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"Judge score must lie in [0, 1], got {score!r}.")
    return score


def mean_completion_score(scores: Sequence[float] | torch.Tensor) -> float:
    """Return the arithmetic mean of finite scores in [0,1]."""
    if torch.is_tensor(scores):
        if scores.ndim != 1:
            raise ValueError("scores must be one-dimensional.")
        values = [float(value) for value in scores.detach().cpu().tolist()]
    else:
        values = [float(value) for value in scores]

    if not values:
        raise ValueError("At least one completion score is required.")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("Completion scores must be finite values in [0, 1].")
    return math.fsum(values) / len(values)


@dataclass
class BehaviorConfig:
    generator_dtype: str = "bfloat16"
    judge_dtype: str = "bfloat16"
    judge_model_name: str = ""
    seed: int = 0

    alpha_start: float = -40.0
    alpha_end: float = 40.0
    alpha_steps: int = 41

    normalize: bool = False
    apply_last_token_only: bool = False

    n_samples_per_context: int = 12
    gen_context_batch_size: int = 12
    max_prompt_length: int = 512
    max_new_tokens: int = 100
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.95
    generator_prompt_suffix: str = ""

    judge_use_chat_template: bool = True
    judge_system_prompt: str = ""
    judge_question_template: str = (
        "Evaluate how strongly the RESPONSE expresses the CONCEPT.\n\n"
        "CONCEPT: {concept}\n\nRESPONSE:\n{completion}\n\n"
        "Return a real-valued score from 0 to 1 inclusive. Use 0 when the "
        "concept is absent, 1 when it is unmistakably expressed, and an "
        "intermediate value for partial, indirect, or ambiguous evidence.\n"
        "Your first nonblank output line must contain only the numeric score. "
        "Do not put a label, markdown, or reasoning before the score."
    )
    judge_max_prompt_length: int = 4000
    judge_batch_size: int = 16
    judge_max_new_tokens: int = 16
    judge_max_completion_chars: int = 4000

    progress_every: int = 1


class BehaviorActor(Actor):
    """Generate steered completions and judge each one with a score in [0,1]."""

    def __init__(self):
        torch.backends.cuda.matmul.allow_tf32 = True

        self._gen_name: Optional[str] = None
        self._gen_dtype: Optional[str] = None
        self._gen_tok = None
        self._gen_model = None

        self._judge_name: Optional[str] = None
        self._judge_dtype: Optional[str] = None
        self._judge_tok = None
        self._judge_model = None

    def _ensure_generator(self, model_name: str, dtype_str: str):
        if (
            self._gen_model is not None
            and self._gen_name == model_name
            and self._gen_dtype == dtype_str
        ):
            return
        self._gen_tok = None
        self._gen_model = None
        torch.cuda.empty_cache()
        self._gen_tok, self._gen_model = load_model_and_tokenizer(model_name, dtype_str)
        set_left_padding(self._gen_tok)
        ensure_pad_token(self._gen_tok, self._gen_model)
        self._gen_name = model_name
        self._gen_dtype = dtype_str

    def _ensure_judge(self, model_name: str, dtype_str: str):
        if (
            self._judge_model is not None
            and self._judge_name == model_name
            and self._judge_dtype == dtype_str
        ):
            return
        self._judge_tok = None
        self._judge_model = None
        torch.cuda.empty_cache()
        self._judge_tok, self._judge_model = load_model_and_tokenizer(model_name, dtype_str)
        set_left_padding(self._judge_tok)
        ensure_pad_token(self._judge_tok, self._judge_model)
        self._judge_name = model_name
        self._judge_dtype = dtype_str

    def _judge_completion_scores(
        self,
        samples: List[str],
        concept: str,
        cfg: BehaviorConfig,
    ) -> torch.Tensor:
        """Judge every completion independently and preserve its position."""
        assert self._judge_tok is not None and self._judge_model is not None
        if not samples:
            raise ValueError("At least one completion is required for judging.")
        if cfg.judge_batch_size <= 0 or cfg.judge_max_new_tokens <= 0:
            raise ValueError("Judge batch size and max-new-token count must be positive.")

        prompts = []
        for sample in samples:
            completion = (sample or "")[: int(cfg.judge_max_completion_chars)]
            user = cfg.judge_question_template.format(
                concept=concept,
                completion=completion,
            )
            prompts.append(
                maybe_apply_chat_template(
                    self._judge_tok,
                    cfg.judge_system_prompt,
                    user,
                    cfg.judge_use_chat_template,
                )
            )

        pad_id = self._judge_tok.pad_token_id
        if pad_id is None:
            pad_id = self._judge_tok.eos_token_id
        if pad_id is None:
            pad_id = self._judge_tok.bos_token_id
        if pad_id is None:
            raise ValueError("Judge tokenizer must define PAD, EOS, or BOS.")

        gen_kwargs = {
            "max_new_tokens": int(cfg.judge_max_new_tokens),
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": int(pad_id),
        }
        if self._judge_tok.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = int(self._judge_tok.eos_token_id)

        scores: List[float] = []
        result_device = torch.device("cuda")
        amp_judge = torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=cfg.judge_dtype.lower() == "bfloat16",
        )
        with torch.inference_mode(), amp_judge:
            for start in range(0, len(prompts), int(cfg.judge_batch_size)):
                prompt_batch = prompts[start : start + int(cfg.judge_batch_size)]
                enc = self._judge_tok(
                    prompt_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(cfg.judge_max_prompt_length),
                )
                input_ids = enc["input_ids"].to("cuda", non_blocking=True)
                result_device = input_ids.device
                attention_mask = enc.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, device=input_ids.device)
                else:
                    attention_mask = attention_mask.to(input_ids.device, non_blocking=True)

                input_len = int(input_ids.shape[1])
                out_ids = self._judge_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
                answers = self._judge_tok.batch_decode(
                    out_ids[:, input_len:],
                    skip_special_tokens=True,
                )
                if len(answers) != len(prompt_batch):
                    raise RuntimeError(
                        f"Judge returned {len(answers)} outputs for "
                        f"{len(prompt_batch)} prompts."
                    )
                scores.extend(parse_unit_interval_score(answer) for answer in answers)

        return torch.tensor(scores, device=result_device, dtype=torch.float32)

    def _generate_completions(
        self,
        prompts: List[str],
        cfg: BehaviorConfig,
    ) -> List[List[str]]:
        assert self._gen_tok is not None and self._gen_model is not None
        enc = self._gen_tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(cfg.max_prompt_length),
        )
        input_ids = enc["input_ids"].to("cuda", non_blocking=True)
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        else:
            attention_mask = attention_mask.to(input_ids.device, non_blocking=True)

        input_len = int(input_ids.shape[1])
        pad_id = self._gen_tok.pad_token_id
        if pad_id is None:
            pad_id = self._gen_tok.eos_token_id
        if pad_id is None:
            pad_id = self._gen_tok.bos_token_id
        if pad_id is None:
            raise ValueError("Generator tokenizer must define PAD, EOS, or BOS.")

        gen_kwargs = {
            "max_new_tokens": int(cfg.max_new_tokens),
            "do_sample": True,
            "temperature": float(cfg.temperature),
            "top_k": int(cfg.top_k),
            "top_p": float(cfg.top_p),
            "num_return_sequences": int(cfg.n_samples_per_context),
            "use_cache": True,
            "pad_token_id": int(pad_id),
        }
        if self._gen_tok.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = int(self._gen_tok.eos_token_id)

        with torch.inference_mode():
            out_ids = self._gen_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        texts = self._gen_tok.batch_decode(
            out_ids[:, input_len:],
            skip_special_tokens=True,
        )
        sample_count = int(cfg.n_samples_per_context)
        expected = len(prompts) * sample_count
        if len(texts) != expected:
            raise RuntimeError(
                f"Generator returned {len(texts)} outputs; expected {expected}."
            )
        return [
            [text.strip() for text in texts[start : start + sample_count]]
            for start in range(0, expected, sample_count)
        ]

    @staticmethod
    def _make_steer_hook(
        alpha: float,
        steer_vec: torch.Tensor,
        last_token_only: bool,
    ):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(hidden):
                return output

            addition = (steer_vec * float(alpha)).to(
                dtype=hidden.dtype,
                device=hidden.device,
            )
            if last_token_only:
                steered = hidden.clone()
                steered[:, -1, :] = steered[:, -1, :] + addition
            else:
                steered = hidden + addition.view(1, 1, -1)

            if isinstance(output, (tuple, list)):
                parts = list(output)
                parts[0] = steered
                return tuple(parts)
            return steered

        return hook

    @endpoint
    async def compute_behavior_curves(
        self,
        model_name: str,
        concept_slug: str,
        concept_label: str,
        block_idx_to_steer,
        contexts_file: str,
        steer_dir: str,
        save_dir: str,
        layer_path: Optional[str] = None,
        cfg_dict: Optional[Dict[str, Any]] = None,
        rank_hint: int = 0,
    ) -> Dict[str, Any]:
        cfg = BehaviorConfig(**(cfg_dict or {}))
        if not cfg.judge_model_name:
            raise ValueError("BehaviorConfig.judge_model_name must be set.")

        torch.manual_seed(int(cfg.seed) + int(rank_hint))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(cfg.seed) + int(rank_hint))

        self._ensure_generator(model_name, cfg.generator_dtype)
        self._ensure_judge(cfg.judge_model_name, cfg.judge_dtype)
        assert self._gen_model is not None

        blocks = find_block_list(self._gen_model, override_path=layer_path)
        if block_idx_to_steer is None:
            layer_indices = list(range(len(blocks)))
        else:
            layer_indices = np.linspace(
                0,
                len(blocks) - 1,
                num=int(block_idx_to_steer),
                dtype=int,
            ).tolist()
        if not layer_indices:
            return {"error": "No valid layer indices."}

        try:
            contexts, context_source_lines = load_contexts_for_concept(
                contexts_file,
                concept_slug=concept_slug,
                concept_label=concept_label,
            )
        except Exception as exc:
            return {
                "error": f"Failed to load contexts for concept '{concept_slug}': {exc}"
            }
        if not contexts:
            return {"error": f"No contexts in {contexts_file} for '{concept_slug}'"}

        negative_count = count_negative_prompts(contexts_file)
        if negative_count is None:
            ctx_is_positive = np.full(len(contexts), -1, dtype=np.int8)
        else:
            negative_count = min(negative_count, len(contexts))
            ctx_is_positive = np.zeros(len(contexts), dtype=np.int8)
            ctx_is_positive[negative_count:] = 1

        alphas = torch.linspace(
            cfg.alpha_start,
            cfg.alpha_end,
            steps=int(cfg.alpha_steps),
            dtype=torch.float32,
        )
        if not bool((alphas == 0.0).any()):
            alphas = torch.sort(
                torch.cat([alphas, torch.tensor([0.0], dtype=torch.float32)])
            )[0]
        alphas_np = alphas.cpu().numpy().astype(np.float32)

        save_root = Path(save_dir) / model_slug(model_name) / concept_slug
        save_root.mkdir(parents=True, exist_ok=True)
        steer_dir_path = Path(steer_dir)
        results: List[Dict[str, Any]] = []

        for layer_idx in layer_indices:
            steer_vec_cpu = load_steer_vector(
                steer_dir_path,
                model_name,
                concept_slug,
                int(layer_idx),
            )
            if cfg.normalize:
                steer_vec_cpu = steer_vec_cpu / torch.norm(steer_vec_cpu).clamp_min(1e-8)
            steer_vec = steer_vec_cpu.to("cuda", non_blocking=True).to(torch.float32)

            context_count = len(contexts)
            alpha_count = int(alphas.numel())
            sample_count = int(cfg.n_samples_per_context)
            concept_scores_by_ctx = np.full(
                (context_count, alpha_count),
                np.nan,
                dtype=np.float32,
            )
            completion_scores_by_ctx = np.full(
                (context_count, alpha_count, sample_count),
                np.nan,
                dtype=np.float32,
            )

            for alpha_index, alpha in enumerate(alphas.tolist()):
                handle = blocks[int(layer_idx)].register_forward_hook(
                    self._make_steer_hook(
                        alpha=float(alpha),
                        steer_vec=steer_vec,
                        last_token_only=cfg.apply_last_token_only,
                    )
                )
                try:
                    for start, end, context_batch in chunked_with_bounds(
                        contexts,
                        int(cfg.gen_context_batch_size),
                    ):
                        prompts = []
                        for context in context_batch:
                            prompt = (context or "").rstrip()
                            if cfg.generator_prompt_suffix:
                                prompt += cfg.generator_prompt_suffix
                            prompts.append(prompt)

                        grouped = self._generate_completions(prompts, cfg)
                        context_means: List[float] = []
                        for offset, samples in enumerate(grouped):
                            scores = self._judge_completion_scores(
                                samples,
                                concept=concept_label,
                                cfg=cfg,
                            )
                            scores_cpu = scores.detach().to(
                                device="cpu",
                                dtype=torch.float32,
                            ).reshape(-1)
                            if scores_cpu.numel() != sample_count:
                                raise RuntimeError(
                                    "Judge returned an unexpected number of scores."
                                )
                            completion_scores_by_ctx[
                                start + offset,
                                alpha_index,
                                :,
                            ] = scores_cpu.numpy()
                            context_means.append(mean_completion_score(scores_cpu))

                        concept_scores_by_ctx[start:end, alpha_index] = np.asarray(
                            context_means,
                            dtype=np.float32,
                        )
                        if cfg.progress_every:
                            await asyncio.sleep(0)
                finally:
                    handle.remove()

                if cfg.progress_every and alpha_index % int(cfg.progress_every) == 0:
                    await asyncio.sleep(0)

            mean_all = np.nanmean(concept_scores_by_ctx, axis=0).astype(np.float32)
            if (ctx_is_positive >= 0).all():
                negative = concept_scores_by_ctx[ctx_is_positive == 0]
                positive = concept_scores_by_ctx[ctx_is_positive == 1]
                mean_negative = (
                    np.nanmean(negative, axis=0).astype(np.float32)
                    if negative.size
                    else np.full(alpha_count, np.nan, dtype=np.float32)
                )
                mean_positive = (
                    np.nanmean(positive, axis=0).astype(np.float32)
                    if positive.size
                    else np.full(alpha_count, np.nan, dtype=np.float32)
                )
                match = np.empty_like(concept_scores_by_ctx)
                match[ctx_is_positive == 1] = concept_scores_by_ctx[
                    ctx_is_positive == 1
                ]
                match[ctx_is_positive == 0] = 1.0 - concept_scores_by_ctx[
                    ctx_is_positive == 0
                ]
                mean_match = np.nanmean(match, axis=0).astype(np.float32)
            else:
                mean_negative = np.full(alpha_count, np.nan, dtype=np.float32)
                mean_positive = np.full(alpha_count, np.nan, dtype=np.float32)
                mean_match = np.full(alpha_count, np.nan, dtype=np.float32)

            out_path = save_root / f"layer_{int(layer_idx)}_behavior.npz"
            meta = {
                "model": model_name,
                "concept_slug": concept_slug,
                "concept": concept_label,
                "judge_model": cfg.judge_model_name,
                "layer_idx": int(layer_idx),
                "alphas": {
                    "start": float(alphas_np[0]),
                    "end": float(alphas_np[-1]),
                    "steps": alpha_count,
                },
                "n_contexts": context_count,
                "n_samples_per_context": sample_count,
                "judge": {
                    "method": "generated_scalar",
                    "range": [0.0, 1.0],
                    "batch_size": int(cfg.judge_batch_size),
                    "max_new_tokens": int(cfg.judge_max_new_tokens),
                    "aggregation": "arithmetic mean over completion scores",
                },
                "contexts_file": str(contexts_file),
            }
            np.savez_compressed(
                out_path,
                alphas=alphas_np,
                concept_scores_by_ctx=concept_scores_by_ctx,
                p1_by_ctx=concept_scores_by_ctx,
                completion_concept_scores_by_ctx=completion_scores_by_ctx,
                ctx_texts=np.array(contexts, dtype=object),
                ctx_source_lines=np.array(context_source_lines, dtype=np.int32),
                ctx_is_positive=ctx_is_positive,
                mean_all=mean_all,
                mean_negative=mean_negative,
                mean_positive=mean_positive,
                mean_match=mean_match,
                meta=json.dumps(meta),
            )
            results.append({"layer_idx": int(layer_idx), "file": str(out_path)})
            await asyncio.sleep(0)

        torch.cuda.empty_cache()
        return {"ok": True, "results": results}
