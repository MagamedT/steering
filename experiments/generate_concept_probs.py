"""Command-line entry point for binary concept-probability curves."""

import argparse
import asyncio
from pathlib import Path

from monarch.actor import shutdown_context

from actors.tasks.concept_probs import ConceptProbsConfig
from utils.data import discover_steering_jobs
from utils.runtime.pool import add_distributed_args
from experiments.runners import run_concept_probs


async def main_async(args):
    """Run the experiment from parsed command-line arguments."""
    config = ConceptProbsConfig(
        judge_model_name=args.judge_model,
        generator_dtype=args.dtype,
        judge_dtype=args.dtype,
        seed=getattr(args, "seed", 0),
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
    await run_concept_probs(
        args,
        config,
        jobs,
        steer_dir,
        out_dir,
        contexts_file,
        continuous=False,
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--judge_model", required=True)
    parser.add_argument("--steer_dir", default="steering_vectors")
    parser.add_argument("--contexts_file", default="data/contexts.jsonl")
    parser.add_argument("--out_dir", default="concept_probs_data")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--layer_path", default=None)
    parser.add_argument("--seed", type=int, default=ConceptProbsConfig.seed)
    add_distributed_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(main_async(parse_args()))
    finally:
        shutdown_context().get()
