"""Command-line entry point for next-token probability curves."""

import argparse
import asyncio
from pathlib import Path

from monarch.actor import shutdown_context

from actors.tasks.next_token_probs import TokenPlotConfig
from utils.data import discover_steering_jobs
from utils.runtime.pool import add_distributed_args
from experiments.runners import run_next_token_probs


async def main_async(args):
    """Run the experiment from parsed command-line arguments."""
    config = TokenPlotConfig(
        dtype=args.dtype,
        seed=args.seed,
        batch_size=args.batch_size,
        alpha_start=args.alpha_start,
        alpha_end=args.alpha_end,
        alpha_steps=args.alpha_steps,
        max_length=args.max_length,
        apply_last_token_only=args.apply_last_token_only,
        normalize=args.normalize,
        top_k=args.top_k,
        progress_every=args.progress_every,
    )
    steer_dir = Path(args.steer_dir)
    out_dir = Path(args.out_dir)
    contexts_file = Path(args.contexts_file)
    if not steer_dir.exists():
        raise RuntimeError(f"--steer_dir {str(steer_dir)!r} does not exist")
    if not contexts_file.exists():
        raise RuntimeError(f"--contexts_file {str(contexts_file)!r} does not exist")
    jobs = discover_steering_jobs(steer_dir, list(args.models))
    if not jobs:
        raise RuntimeError(f"No model/concept pairs found under {steer_dir}")
    await run_next_token_probs(
        args, config, jobs, steer_dir, out_dir, contexts_file
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--steer_dir", default="steering_vectors")
    parser.add_argument("--contexts_file", default="data/contexts.jsonl")
    parser.add_argument("--out_dir", default="plot_data")
    parser.add_argument("--layers", nargs="+", default=[None])
    parser.add_argument("--layer_path", default=None)
    parser.add_argument("--batch_size", type=int, default=TokenPlotConfig.batch_size)
    parser.add_argument("--alpha_start", type=float, default=TokenPlotConfig.alpha_start)
    parser.add_argument("--alpha_end", type=float, default=TokenPlotConfig.alpha_end)
    parser.add_argument("--alpha_steps", type=int, default=TokenPlotConfig.alpha_steps)
    parser.add_argument("--max_length", type=int, default=TokenPlotConfig.max_length)
    parser.add_argument("--top_k", type=int, default=TokenPlotConfig.top_k)
    parser.add_argument("--progress_every", type=int, default=TokenPlotConfig.progress_every)
    parser.add_argument("--apply_last_token_only", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    add_distributed_args(parser, default_dtype=TokenPlotConfig.dtype)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(main_async(parse_args()))
    finally:
        shutdown_context().get()
