"""Recompute concept probabilities for saved completions."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
from monarch.actor import Actor, endpoint
from utils.batch_size import critical_batch_size


from .concept_probs_continuous import ConceptProbsConfig
from .rubric_judge import RubricJudgeDetails, rubric_completion_scores


def _means(scores: np.ndarray, ctx_is_positive: np.ndarray):
    """Compute overall, positive, negative, and matched score means."""
    by_context = np.mean(scores, axis=2, dtype=np.float64).astype(np.float32)
    mean_all = np.mean(by_context, axis=0, dtype=np.float64).astype(np.float32)
    if not bool((ctx_is_positive >= 0).all()):
        missing = np.full(by_context.shape[1], np.nan, dtype=np.float32)
        return by_context, mean_all, missing, missing.copy(), missing.copy()

    negative = by_context[ctx_is_positive == 0]
    positive = by_context[ctx_is_positive == 1]
    mean_negative = np.mean(negative, axis=0, dtype=np.float64).astype(np.float32)
    mean_positive = np.mean(positive, axis=0, dtype=np.float64).astype(np.float32)
    match = by_context.copy()
    match[ctx_is_positive == 0] = 1.0 - match[ctx_is_positive == 0]
    mean_match = np.mean(match, axis=0, dtype=np.float64).astype(np.float32)
    return by_context, mean_all, mean_negative, mean_positive, mean_match


class RescoreActor(Actor):
    """Rescore saved completions with a judge model."""

    @endpoint
    async def rescore_file(
        self,
        path: str,
        input_root: str,
        output_root: str,
        cfg_dict: dict,
    ) -> dict | None:
        """Score saved completions again and write a new result file."""
        cfg = ConceptProbsConfig(**cfg_dict)
        source = Path(path)
        with np.load(source, allow_pickle=True) as data:
            payload = {key: data[key] for key in data.files}

        meta = json.loads(str(payload["meta"]))
        texts = np.asarray(payload["completion_texts_by_ctx"], dtype=object)
        if texts.ndim != 3 or texts.shape[-1] == 0:
            raise ValueError("Saved completion texts must have shape [context, alpha, sample]")
        cfg.judge_batch_size = critical_batch_size(
            self.model,
            kind="rubric",
            prompt_tokens=int(cfg.judge_max_prompt_length),
            new_tokens=int(cfg.judge_rubric_max_new_tokens),
            limit=int(texts.shape[-1]),
            requested=int(cfg.judge_batch_size),
            dtype=cfg.judge_dtype,
            tp_mesh=getattr(self, "tp_mesh", None),
        ).batch_size

        # Match the original run's one-context, one-alpha judge batches. Different
        # padding widths can otherwise cause small bfloat16 score changes.
        grouped_texts = texts.reshape(-1, texts.shape[-1])
        grouped_scores = np.empty(grouped_texts.shape, dtype=np.float32)
        grouped_raw_json = np.empty(grouped_texts.shape, dtype=object)
        grouped_true_logits = np.empty(grouped_texts.shape, dtype=np.float32)
        grouped_false_logits = np.empty(grouped_texts.shape, dtype=np.float32)
        grouped_pair_mass = np.empty(grouped_texts.shape, dtype=np.float32)
        for group_index, group in enumerate(grouped_texts):
            details = rubric_completion_scores(
                self.tokenizer,
                self.model,
                [str(value) for value in group],
                str(meta["concept"]),
                cfg,
                instructions=None,
                return_details=True,
                tp_mesh=getattr(self, "tp_mesh", None),
            )
            if not isinstance(details, RubricJudgeDetails):
                raise RuntimeError("Rubric judge returned no scoring diagnostics")
            if details.scores.numel() != group.size:
                raise RuntimeError("Rubric judge returned an unexpected number of scores")
            grouped_scores[group_index] = details.scores.numpy()
            grouped_raw_json[group_index] = np.asarray(details.raw_json, dtype=object)
            grouped_true_logits[group_index] = details.true_logits.numpy()
            grouped_false_logits[group_index] = details.false_logits.numpy()
            grouped_pair_mass[group_index] = details.pair_mass.numpy()
            await asyncio.sleep(0)

        if not self.is_leader:
            return None

        scores = grouped_scores.reshape(texts.shape)
        ctx_is_positive = np.asarray(payload["ctx_is_positive"], dtype=np.int8)
        by_context, mean_all, mean_negative, mean_positive, mean_match = _means(
            scores, ctx_is_positive
        )
        payload.update(
            completion_concept_scores_by_ctx=scores,
            concept_scores_by_ctx=by_context,
            p1_by_ctx=by_context,
            mean_all=mean_all,
            mean_negative=mean_negative,
            mean_positive=mean_positive,
            mean_match=mean_match,
            judge_raw_json=grouped_raw_json.reshape(texts.shape),
            judge_true_logits=grouped_true_logits.reshape(texts.shape),
            judge_false_logits=grouped_false_logits.reshape(texts.shape),
            judge_pair_mass=grouped_pair_mass.reshape(texts.shape),
        )
        meta["judge"] = {
            "method": "rubric_boolean_probability",
            "formula": "sigmoid(z_true - z_false)",
            "range": [0.0, 1.0],
            "batch_size": int(cfg.judge_batch_size),
            "max_new_tokens": int(cfg.judge_rubric_max_new_tokens),
            "rubric_template": cfg.judge_rubric_template,
            "evidence_source": "generated completion only",
            "repetition_counts": True,
            "quality_ignored": True,
            "aggregation": "arithmetic mean over completion scores",
        }
        meta["rescored_from"] = str(source)
        meta["tensor_parallel_size"] = getattr(self, "gpus_per_actor", 1)
        payload["meta"] = json.dumps(meta)

        destination = Path(output_root) / source.relative_to(Path(input_root))
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **payload)
        return {"ok": True, "file": str(destination)}
