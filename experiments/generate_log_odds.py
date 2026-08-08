import argparse
import asyncio
import json
from pathlib import Path

from monarch.actor import shutdown_context

from actors.log_odds_actor import LogOddsConfig
from actors.utils import model_slug, slugify
from experiments.distributed_runtime import add_distributed_args
from experiments.unified_runs import run_log_odds


def discover_jobs(prompts_dir: Path, models: list[str]):
    positives = {
        path.name[: -len("_positive.jsonl")]: path
        for path in prompts_dir.glob("*_positive.jsonl")
    }
    jobs = []
    for model_name in models:
        for slug, positive_path in positives.items():
            negative_path = prompts_dir / f"{slug}_{model_slug(model_name)}_negative.jsonl"
            if not negative_path.exists():
                continue
            label = slug.replace("-", " ").replace("_", " ")
            try:
                with positive_path.open("r", encoding="utf-8") as handle:
                    row = json.loads(next(handle))
                if isinstance(row, dict) and isinstance(row.get("concept"), str):
                    label = row["concept"].strip()
            except Exception:
                pass
            jobs.append((model_name, slug, label))
    return jobs


async def main_async(args):
    prompts_path = Path(args.prompts_dir)
    out_dir = Path(args.out_dir)
    if not prompts_path.exists():
        raise RuntimeError(f"--prompts_dir {str(prompts_path)!r} does not exist")
    jobs = discover_jobs(prompts_path, list(args.models))
    if args.concepts:
        allowed = {slugify(concept) for concept in args.concepts}
        jobs = [job for job in jobs if job[1] in allowed]
    if not jobs:
        raise RuntimeError(f"No model/concept pairs found under {prompts_path}")
    await run_log_odds(
        args,
        LogOddsConfig(dtype=args.dtype),
        jobs,
        prompts_path,
        out_dir,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--prompts_dir", default="prompts")
    parser.add_argument("--out_dir", default="log_odds")
    parser.add_argument("--concepts", nargs="*", default=None)
    add_distributed_args(parser, default_dtype=LogOddsConfig.dtype)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(main_async(parse_args()))
    finally:
        shutdown_context().get()
