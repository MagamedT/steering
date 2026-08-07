"""Rescore saved completions with the current concept-presence rubric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from actors.concept_probs_continuous_actor import BehaviorConfig
from actors.rubric_judge import RubricJudgeDetails, rubric_completion_scores
from actors.utils import ensure_pad_token, load_model_and_tokenizer, set_left_padding


def _means(scores: np.ndarray, ctx_is_positive: np.ndarray):
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


def rescore_file(path: Path, input_root: Path, output_root: Path, tokenizer, model, cfg: BehaviorConfig) -> Path:
    with np.load(path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}

    meta = json.loads(str(payload["meta"]))
    texts = np.asarray(payload["completion_texts_by_ctx"], dtype=object)
    flat_texts = [str(value) for value in texts.reshape(-1)]
    details = rubric_completion_scores(
        tokenizer,
        model,
        flat_texts,
        str(meta["concept"]),
        cfg,
        instructions=None,
        return_details=True,
    )
    if not isinstance(details, RubricJudgeDetails):
        raise RuntimeError("Rubric judge returned no scoring diagnostics.")

    scores = details.scores.numpy().reshape(texts.shape).astype(np.float32)
    ctx_is_positive = np.asarray(payload["ctx_is_positive"], dtype=np.int8)
    by_context, mean_all, mean_negative, mean_positive, mean_match = _means(
        scores,
        ctx_is_positive,
    )
    payload.update(
        completion_concept_scores_by_ctx=scores,
        concept_scores_by_ctx=by_context,
        p1_by_ctx=by_context,
        mean_all=mean_all,
        mean_negative=mean_negative,
        mean_positive=mean_positive,
        mean_match=mean_match,
        judge_raw_json=np.asarray(details.raw_json, dtype=object).reshape(texts.shape),
        judge_true_logits=details.true_logits.numpy().reshape(texts.shape),
        judge_false_logits=details.false_logits.numpy().reshape(texts.shape),
        judge_pair_mass=details.pair_mass.numpy().reshape(texts.shape),
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
    meta["rescored_from"] = str(path)
    payload["meta"] = json.dumps(meta)

    output = output_root / path.relative_to(input_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--judge_model", required=True)
    parser.add_argument("--judge_batch_size", type=int, default=8)
    parser.add_argument("--judge_max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    files = sorted(input_root.glob("**/layer_*_behavior.npz"))
    if not files:
        raise RuntimeError(f"No behavior NPZ files found under {input_root}")

    tokenizer, model = load_model_and_tokenizer(args.judge_model, "bfloat16")
    set_left_padding(tokenizer)
    ensure_pad_token(tokenizer, model)
    cfg = BehaviorConfig(
        judge_model_name=args.judge_model,
        judge_batch_size=args.judge_batch_size,
        judge_rubric_max_new_tokens=args.judge_max_new_tokens,
    )
    for path in files:
        output = rescore_file(path, input_root, output_root, tokenizer, model, cfg)
        print(output, flush=True)


if __name__ == "__main__":
    main()
