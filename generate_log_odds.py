#!/usr/bin/env python3
import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import torch
from monarch.actor import this_host

from actors.log_odds_actor import LogOddsActor, LogOddsConfig
from actors.utils import model_slug, slugify


def discover_jobs(prompts_dir: Path, models: list[str]) -> list[tuple[str, str, str]]:
    """Return (model_name, concept_slug, concept_label) for concepts with positive and model-specific negative JSONLs."""
    positives = {p.name[:-len("_positive.jsonl")]: p for p in prompts_dir.glob("*_positive.jsonl")}
    jobs: list[tuple[str, str, str]] = []
    for model_name in models:
        mslug = model_slug(model_name)
        for slug, pos_path in positives.items():
            # Log-odds comparison is model-specific, so negatives must match the current model slug.
            neg_path = prompts_dir / f"{slug}_{mslug}_negative.jsonl"
            if not neg_path.exists():
                continue
            label = slug.replace("-", " ").replace("_", " ")
            try:
                with pos_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        if isinstance(row, dict) and isinstance(row.get("concept"), str):
                            label = row["concept"].strip()
                        break
            except Exception:
                pass
            jobs.append((model_name, slug, label))
    return jobs


async def main_async(args):
    cfg = LogOddsConfig()
    prompts_path = Path(args.prompts_dir)
    out_dir = Path(args.out_dir)

    if not prompts_path.exists():
        raise RuntimeError(f"--prompts_dir '{prompts_path}' does not exist")

    models = list(args.models)
    jobs = discover_jobs(prompts_path, models)

    if args.concepts:
        allowed = {slugify(c) for c in args.concepts}
        jobs = [(m, slug, label) for (m, slug, label) in jobs if slug in allowed]

    if not jobs:
        raise RuntimeError(f"No (model, concept) pairs discovered under {prompts_path} for given models.")

    visible = torch.cuda.device_count()
    if visible < 1:
        raise RuntimeError("No CUDA devices visible.")
    use_gpus = min(visible, len(jobs))
    if args.max_gpus and args.max_gpus > 0:
        use_gpus = min(use_gpus, args.max_gpus)

    mesh = this_host().spawn_procs(per_host={args.dim: use_gpus})
    print(mesh.to_table(), flush=True)

    workers = mesh.spawn("log_odds", LogOddsActor)

    def actor_for(rank: int):
        return workers.slice(**{args.dim: rank})

    async def run_one(rank: int, model_name: str, concept_slug: str, concept_label: str):
        return await actor_for(rank).compute_log_odds.call_one(
            model_name=model_name,
            concept_slug=concept_slug,
            concept_label=concept_label,
            prompts_dir=str(prompts_path),
            save_dir=str(out_dir),
            cfg_dict=asdict(cfg),
            rank_hint=rank,
        )

    next_idx = 0
    in_flight: dict[asyncio.Task, int] = {}
    for r in range(min(use_gpus, len(jobs))):
        m, slug, label = jobs[next_idx]
        next_idx += 1
        print(f"→ [gpu {r}] start model='{m}' concept='{label}' (slug={slug})", flush=True)
        task = asyncio.create_task(run_one(r, m, slug, label))
        in_flight[task] = r

    while in_flight:
        done, _ = await asyncio.wait(in_flight.keys(), return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            rank = in_flight.pop(t)
            res = await t
            print(f"[gpu {rank}] finished: {res}", flush=True)
            if next_idx < len(jobs):
                m, slug, label = jobs[next_idx]
                next_idx += 1
                print(f"→ [gpu {rank}] start model='{m}' concept='{label}' (slug={slug})", flush=True)
                task = asyncio.create_task(run_one(rank, m, slug, label))
                in_flight[task] = rank


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="HF model ids or local paths to score.")
    p.add_argument("--prompts_dir", default="prompts", help="Directory containing <concept>_positive.jsonl and <concept>_{model}_negative.jsonl files.")
    p.add_argument("--out_dir", default="log_odds", help="Where to write .npz log-odds files.")
    p.add_argument("--concepts", nargs="*", default=None, help="Optional subset of concept slugs.")
    p.add_argument("--dim", default="gpu")
    p.add_argument("--max_gpus", type=int, default=0, help="Limit visible GPUs (0 = all).")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
