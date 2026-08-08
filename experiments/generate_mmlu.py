"""Command-line entry point for MMLU evaluation."""

import os

os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("OTEL_TRACES_EXPORTER", "none")
os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")
os.environ.setdefault("OTEL_LOGS_EXPORTER", "none")

import argparse
import asyncio
from pathlib import Path

from monarch.actor import shutdown_context

from actors.tasks.mmlu import MMLUEvalConfig
from utils.data import discover_steering_jobs
from utils.runtime.pool import add_distributed_args
from experiments.runners import run_mmlu


async def main_async(args):
    """Run the experiment from parsed command-line arguments."""
    tasks = args.tasks
    if tasks and len(tasks) == 1 and tasks[0].lower() == "all":
        tasks = None
    max_problems = getattr(args, "max_problems_per_task", None)
    if max_problems is not None and max_problems < 1:
        raise ValueError("--max_problems_per_task must be positive")
    config = MMLUEvalConfig(
        dtype=args.dtype,
        seed=args.seed,
        tasks=tasks,
        max_problems_per_task=max_problems,
    )
    steer_dir = Path(args.steer_dir)
    out_dir = Path(args.out_dir)
    if not steer_dir.exists():
        raise RuntimeError(f"--steer_dir {str(steer_dir)!r} does not exist")
    jobs = discover_steering_jobs(steer_dir, list(args.models))
    if not jobs:
        raise RuntimeError(f"No model/concept pairs found under {steer_dir}")
    await run_mmlu(args, config, jobs, steer_dir, out_dir)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--steer_dir", default="steering_vectors")
    parser.add_argument("--out_dir", default="mmlu")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--max_problems_per_task", type=int, default=None)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--layer_path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    add_distributed_args(parser, default_dtype=MMLUEvalConfig.dtype)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(main_async(parse_args()))
    finally:
        shutdown_context().get()
