#!/usr/bin/env python3
"""Manual CUDA stress check for the analytical batch-size estimator.

Example:
    python tests/stress_batch_size_cuda.py --local-files-only --max-batch 32

The script performs one estimate and one workload per model; it never searches
for a passing batch size.  Results are emitted as one JSON object per model.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.batch_size import critical_batch_size  # noqa: E402


DEFAULT_MODELS = (
    "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-360M",
    "HuggingFaceTB/SmolLM2-1.7B",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one analytically selected CUDA batch per causal LM."
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--kind",
        choices=("cross_entropy", "forward", "generate", "rubric"),
        default="cross_entropy",
    )
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--max-batch",
        type=int,
        default=None,
        help="Optional transparent cap on the analytically predicted batch.",
    )
    parser.add_argument("--safety", type=float, default=0.80)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only models already present in the Hugging Face cache.",
    )
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("--tokens must be positive")
    if args.new_tokens < 1:
        parser.error("--new-tokens must be positive")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.max_batch is not None and args.max_batch < 1:
        parser.error("--max-batch must be positive")
    if not 0 < args.safety <= 1:
        parser.error("--safety must be in (0, 1]")
    return args


def _run(model_id: str, args: argparse.Namespace, device: torch.device) -> dict:
    from transformers import AutoModelForCausalLM

    record = {
        "model": model_id,
        "workload": args.kind,
        "params": None,
        "predicted_batch_size": None,
        "selected_batch_size": None,
        "max_batch_cap": args.max_batch,
        "predicted_peak_bytes": None,
        "actual_peak_bytes": None,
        "max_memory_reserved_bytes": None,
        "free_bytes": None,
        "headroom_bytes": None,
        "prediction_covers_actual_peak": None,
        "success": False,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": device},
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
            local_files_only=args.local_files_only,
        ).eval()
        record["params"] = sum(parameter.numel() for parameter in model.parameters())

        estimate = critical_batch_size(
            model,
            kind=args.kind,
            prompt_tokens=args.tokens,
            new_tokens=(
                args.new_tokens if args.kind in {"generate", "rubric"} else 0
            ),
            requested=args.max_batch or 0,
            dtype=torch.bfloat16,
            use_cache=args.kind in {"forward", "generate", "rubric"},
            safety=args.safety,
        )
        batch = estimate.batch_size
        baseline = torch.cuda.memory_allocated(device)
        predicted_increment = batch * estimate.bytes_per_sequence
        record.update(
            predicted_batch_size=estimate.critical_batch_size,
            selected_batch_size=batch,
            predicted_peak_bytes=baseline + predicted_increment,
            free_bytes=estimate.free_bytes,
        )

        torch.cuda.reset_peak_memory_stats(device)
        input_ids = torch.randint(
            int(model.config.vocab_size),
            (batch, args.tokens),
            dtype=torch.long,
            device=device,
        )
        with torch.inference_mode():
            if args.kind == "cross_entropy":
                output = model(input_ids=input_ids, use_cache=False)
                labels = torch.randint_like(input_ids, int(model.config.vocab_size))
                logits = output.logits.float()
                result = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
                )
            elif args.kind == "forward":
                output = model(input_ids=input_ids, use_cache=True)
                logits = output.logits[:, -1].float()
                result = logits.softmax(dim=-1)
            else:
                pad_id = getattr(model.config, "pad_token_id", None)
                if pad_id is None:
                    pad_id = getattr(model.config, "eos_token_id", None)
                attention_mask = torch.ones_like(input_ids)
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.new_tokens,
                    do_sample=args.kind == "generate",
                    temperature=0.8 if args.kind == "generate" else 1.0,
                    top_p=args.top_p if args.kind == "generate" else 1.0,
                    use_cache=True,
                    eos_token_id=None,
                    pad_token_id=0 if pad_id is None else int(pad_id),
                )
                if args.kind == "rubric":
                    output = model(
                        input_ids=generated,
                        use_cache=False,
                        logits_to_keep=1,
                    )
                    logits = output.logits[:, -1].float()
                    result = logits.logsumexp(dim=-1)
                else:
                    result = generated

        torch.cuda.synchronize(device)
        actual_peak = torch.cuda.max_memory_allocated(device)
        predicted_peak = record["predicted_peak_bytes"]
        covers_peak = predicted_peak >= actual_peak
        record.update(
            actual_peak_bytes=actual_peak,
            max_memory_reserved_bytes=torch.cuda.max_memory_reserved(device),
            headroom_bytes=torch.cuda.mem_get_info(device)[0],
            prediction_covers_actual_peak=covers_peak,
            success=covers_peak,
        )
        if not covers_peak:
            record["error"] = (
                f"analytical peak {predicted_peak} is below observed peak "
                f"{actual_peak}"
            )
        # Keep the measured workload alive until after the peak and headroom reads.
        del model
    except Exception as exc:  # Keep later model sizes independent of one failure.
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this manual stress check.")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("The current CUDA device does not support bfloat16.")

    device = torch.device("cuda", torch.cuda.current_device())
    failed = False
    for model_id in args.models:
        gc.collect()
        torch.cuda.empty_cache()
        record = _run(model_id, args, device)
        print(json.dumps(record, sort_keys=True), flush=True)
        failed |= not record["success"]
        gc.collect()
        torch.cuda.empty_cache()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
