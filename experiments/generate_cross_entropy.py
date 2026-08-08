import argparse
import asyncio
from pathlib import Path

from monarch.actor import shutdown_context

from actors.cross_entropy_actor import CrossEntropyPlotConfig
from actors.utils import discover_jobs
from experiments.distributed_runtime import add_distributed_args
from experiments.unified_runs import run_cross_entropy


async def main_async(args):
    config = CrossEntropyPlotConfig(seed=args.seed, dtype=args.dtype)
    steer_dir = Path(args.steer_dir)
    out_dir = Path(args.out_dir)
    eval_parquet = Path(args.eval_parquet)
    if not steer_dir.exists():
        raise RuntimeError(f"--steer_dir {str(steer_dir)!r} does not exist")
    if not eval_parquet.exists():
        raise RuntimeError(f"--eval_parquet {str(eval_parquet)!r} does not exist")
    jobs = discover_jobs(steer_dir, list(args.models))
    if not jobs:
        raise RuntimeError(f"No model/concept pairs found under {steer_dir}")
    await run_cross_entropy(
        args, config, jobs, steer_dir, out_dir, eval_parquet
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--steer_dir", default="steering_vectors")
    parser.add_argument("--eval_parquet", required=True)
    parser.add_argument("--out_dir", default="cross_entropy")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--layer_path", default=None)
    parser.add_argument("--seed", type=int, default=0)
    add_distributed_args(parser, default_dtype=CrossEntropyPlotConfig.dtype)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(main_async(parse_args()))
    finally:
        shutdown_context().get()
