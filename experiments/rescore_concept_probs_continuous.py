"""Rescore saved completions with the current concept-presence rubric."""

import argparse
import asyncio
from pathlib import Path

from monarch.actor import shutdown_context

from actors.concept_probs_continuous_actor import BehaviorConfig
from experiments.distributed_runtime import add_distributed_args
from experiments.unified_runs import run_rescore


async def main_async(args):
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    files = sorted(input_root.glob("**/layer_*_behavior.npz"))
    if not files:
        raise RuntimeError(f"No behavior NPZ files found under {input_root}")
    config = BehaviorConfig(
        judge_model_name=args.judge_model,
        judge_dtype=args.dtype,
        judge_batch_size=args.judge_batch_size,
        judge_rubric_max_new_tokens=args.judge_max_new_tokens,
    )
    await run_rescore(args, config, files, input_root, output_root)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--judge_model", required=True)
    parser.add_argument("--judge_batch_size", type=int, default=8)
    parser.add_argument("--judge_max_new_tokens", type=int, default=256)
    add_distributed_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(main_async(parse_args()))
    finally:
        shutdown_context().get()
