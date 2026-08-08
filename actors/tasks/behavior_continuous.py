"""Continuous concept-probability scoring with a separate judge LLM."""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from monarch.actor import Actor, endpoint

from steering.batching import chunked_with_bounds
from steering.data import count_negative_prompts, load_contexts_for_concept
from steering.modeling import find_block_list, load_steering_vector
from steering.naming import model_slug

from .rubric_judge import (
    DEFAULT_CONCEPT_RUBRIC,
    RubricJudgeDetails,
    rubric_completion_scores,
)


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

    judge_max_prompt_length: int = 4000
    judge_batch_size: int = 16
    judge_max_new_tokens: int = 16
    judge_rubric_max_new_tokens: int = 512
    judge_rubric_template: str = DEFAULT_CONCEPT_RUBRIC

    progress_every: int = 1


class BehaviorActor(Actor):
    """Generate steered completions and judge each one with a score in [0,1]."""

    def _ensure_generator(self, model_name: str, dtype_str: str):
        if (
            self._gen_name == model_name
            and self._gen_dtype == dtype_str
        ):
            return
        raise RuntimeError("Distributed actor was initialized for a different generator")

    def _ensure_judge(self, model_name: str, dtype_str: str):
        if (
            self._judge_name == model_name
            and self._judge_dtype == dtype_str
        ):
            return
        raise RuntimeError("Distributed actor was initialized for a different judge")

    def _judge_completion_scores(
        self,
        samples: List[str],
        concept: str,
        cfg: BehaviorConfig,
        instruction: str | None = None,
    ) -> torch.Tensor:
        """Return sigmoid(z_true-z_false) for every completion."""
        assert self._judge_tok is not None and self._judge_model is not None
        details = rubric_completion_scores(
            self._judge_tok,
            self._judge_model,
            samples,
            concept,
            cfg,
            # Concept presence is a property of the generated continuation only.
            # The source context must not affect the verdict.
            instructions=None,
            return_details=True,
            tp_mesh=getattr(self, "tp_mesh", None),
        )
        if not isinstance(details, RubricJudgeDetails):
            raise RuntimeError("Rubric judge returned no scoring diagnostics.")
        for row_index, raw_json in enumerate(details.raw_json):
            parsed = json.loads(raw_json)
            if tuple(parsed) != ("explanation_1", "criteria_met_1"):
                raise ValueError(
                    f"Rubric judge row {row_index} must contain exactly "
                    "explanation_1 and criteria_met_1."
                )
        return details.scores

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
        input_ids = enc["input_ids"].to(self.device, non_blocking=True)
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
        exact_layer_idx: Optional[int] = None,
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
        if exact_layer_idx is not None:
            exact_layer_idx = int(exact_layer_idx)
            if not 0 <= exact_layer_idx < len(blocks):
                raise ValueError(
                    f"Layer {exact_layer_idx} is outside [0, {len(blocks) - 1}]."
                )
            layer_indices = [exact_layer_idx]
        elif block_idx_to_steer is None:
            layer_indices = list(range(len(blocks)))
        else:
            layer_indices = np.linspace(
                0,
                len(blocks) - 1,
                num=int(block_idx_to_steer),
                dtype=int,
            ).tolist()
        if not layer_indices:
            result = {"error": "No valid layer indices."}
            return result if self.is_leader else None

        try:
            contexts, context_source_lines = load_contexts_for_concept(
                contexts_file,
                concept_slug=concept_slug,
                concept_label=concept_label,
            )
        except Exception as exc:
            result = {"error": f"Failed to load contexts for concept '{concept_slug}': {exc}"}
            return result if self.is_leader else None
        if not contexts:
            result = {"error": f"No contexts in {contexts_file} for '{concept_slug}'"}
            return result if self.is_leader else None

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
        near_zero = torch.isclose(
            alphas,
            torch.tensor(0.0, dtype=alphas.dtype),
            atol=1e-6,
            rtol=0.0,
        )
        if bool(near_zero.any()):
            alphas[near_zero] = 0.0
        else:
            alphas = torch.sort(
                torch.cat([alphas, torch.tensor([0.0], dtype=torch.float32)])
            )[0]
        alphas_np = alphas.cpu().numpy().astype(np.float32)

        save_root = Path(save_dir) / model_slug(model_name) / concept_slug
        if self.is_leader:
            save_root.mkdir(parents=True, exist_ok=True)
        steer_dir_path = Path(steer_dir)
        results: List[Dict[str, Any]] = []

        for layer_idx in layer_indices:
            steer_vec_cpu = load_steering_vector(
                steer_dir_path,
                model_name,
                concept_slug,
                int(layer_idx),
            )
            if cfg.normalize:
                steer_vec_cpu = steer_vec_cpu / torch.norm(steer_vec_cpu).clamp_min(1e-8)
            steer_vec = steer_vec_cpu.to(self.device, non_blocking=True).to(torch.float32)

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
            completion_texts_by_ctx = np.full(
                (context_count, alpha_count, sample_count),
                "",
                dtype=object,
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
                            if len(samples) != sample_count:
                                raise RuntimeError(
                                    "Generator returned an unexpected number of "
                                    "completions."
                                )
                            completion_texts_by_ctx[
                                start + offset,
                                alpha_index,
                                :,
                            ] = np.asarray(samples, dtype=object)
                            scores = self._judge_completion_scores(
                                samples,
                                concept=concept_label,
                                cfg=cfg,
                                instruction=context_batch[offset],
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

            if not self.is_leader:
                await asyncio.sleep(0)
                continue

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
                    "method": "rubric_boolean_probability",
                    "formula": "sigmoid(z_true - z_false)",
                    "range": [0.0, 1.0],
                    "batch_size": int(cfg.judge_batch_size),
                    "max_new_tokens": int(cfg.judge_rubric_max_new_tokens),
                    "rubric_template": cfg.judge_rubric_template,
                    "aggregation": "arithmetic mean over completion scores",
                },
                "contexts_file": str(contexts_file),
                "tensor_parallel_size": getattr(self, "gpus_per_actor", 1),
            }
            np.savez_compressed(
                out_path,
                alphas=alphas_np,
                concept_scores_by_ctx=concept_scores_by_ctx,
                p1_by_ctx=concept_scores_by_ctx,
                completion_concept_scores_by_ctx=completion_scores_by_ctx,
                completion_texts_by_ctx=completion_texts_by_ctx,
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

        if not self.is_leader:
            torch.cuda.empty_cache()
            return None

        torch.cuda.empty_cache()
        return {"ok": True, "results": results}
