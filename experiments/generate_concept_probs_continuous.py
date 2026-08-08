import argparse
import asyncio
from pathlib import Path

from monarch.actor import shutdown_context

from actors.concept_probs_continuous_actor import BehaviorConfig
from actors.utils import discover_jobs
from experiments.distributed_runtime import add_distributed_args
from experiments.unified_runs import run_behavior


async def main_async(args):
    config = BehaviorConfig(
        judge_model_name=args.judge_model,
        generator_dtype=args.dtype,
        judge_dtype=args.dtype,
        alpha_start=args.alpha_start,
        alpha_end=args.alpha_end,
        alpha_steps=args.alpha_steps,
        seed=args.seed,
        normalize=args.normalize,
        n_samples_per_context=args.samples_per_context,
        gen_context_batch_size=args.context_batch_size,
        max_new_tokens=args.max_new_tokens,
        judge_batch_size=args.judge_batch_size,
        judge_rubric_max_new_tokens=args.judge_max_new_tokens,
    )
    steer_dir = Path(args.steer_dir)
    out_dir = Path(args.out_dir)
    contexts_file = Path(args.contexts_file)
    if not steer_dir.exists():
        raise RuntimeError(f"--steer_dir {str(steer_dir)!r} does not exist")
    if not contexts_file.exists():
        raise RuntimeError(f"--contexts_file {str(contexts_file)!r} does not exist")
    jobs = discover_jobs(steer_dir, list(args.models))
    if not jobs:
        raise RuntimeError(f"No model/concept pairs found under {steer_dir}")
    await run_behavior(
        args,
        config,
        jobs,
        steer_dir,
        out_dir,
        contexts_file,
        continuous=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--judge_model", required=True)
    parser.add_argument("--steer_dir", default="steering_vectors")
    parser.add_argument("--contexts_file", default="data/contexts.jsonl")
    parser.add_argument("--out_dir", default="behavior_data_continuous")
    parser.add_argument("--alpha_start", type=float, default=-40.0)
    parser.add_argument("--alpha_end", type=float, default=40.0)
    parser.add_argument("--alpha_steps", type=int, default=41)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples_per_context", type=int, default=12)
    parser.add_argument("--context_batch_size", type=int, default=12)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--judge_batch_size", type=int, default=16)
    parser.add_argument("--judge_max_new_tokens", type=int, default=512)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--layer_path", default=None)
    add_distributed_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(main_async(parse_args()))
    finally:
        shutdown_context().get()
