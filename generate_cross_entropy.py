#!/usr/bin/env python3
# generate_plot_data.py
#
# Monarch launcher for cross-entropy-vs-α steering curves.

import argparse
import asyncio
from pathlib import Path
from dataclasses import asdict

import torch
from monarch.actor import this_host

from actors.cross_entropy_actor import CrossEntropyActor, CrossEntropyPlotConfig, model_slug


def discover_jobs(steer_dir: Path, models: list[str]) -> list[tuple[str, str, str]]:
    jobs = []
    for model_name in models:
        mslug = model_slug(model_name)
        base = steer_dir / mslug
        if not base.exists():
            continue
        for concept_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
            slug = concept_dir.name
            label = slug.replace("_", " ")
            if not any(concept_dir.glob("layer_*.pt")):
                continue
            jobs.append((model_name, slug, label))
    return jobs


async def main_async(args):
    cfg = CrossEntropyPlotConfig(seed=args.seed)    
    steer_dir = Path(args.steer_dir)
    out_dir = Path(args.out_dir)
    eval_parquet = Path(args.eval_parquet)

    if not steer_dir.exists():
        raise RuntimeError(f"--steer_dir '{steer_dir}' does not exist")
    if not eval_parquet.exists():
        raise RuntimeError(f"--eval_parquet '{eval_parquet}' does not exist")

    jobs = discover_jobs(steer_dir, list(args.models))
    if not jobs:
        raise RuntimeError(f"No (model, concept) pairs discovered under {steer_dir} for given models.")

    visible = torch.cuda.device_count()
    if visible < 1:
        raise RuntimeError("No CUDA devices visible.")
    use_gpus = min(visible, len(jobs))
    if args.max_gpus and args.max_gpus > 0:
        use_gpus = min(use_gpus, args.max_gpus)

    mesh = this_host().spawn_procs(per_host={args.dim: use_gpus})
    print(mesh.to_table(), flush=True)

    workers = mesh.spawn("plot", CrossEntropyActor)

    def actor_for(rank: int):
        return workers.slice(**{args.dim: rank})

    async def run_one(rank: int, model_name: str, concept_slug: str, concept_label: str):
        return await actor_for(rank).compute_cross_entropy_curves.call_one(
            model_name=model_name,
            concept_slug=concept_slug,
            concept_label=concept_label,
            # 0 => all layers; positive int => evenly sample that many layers.
            block_idx_to_steer=(None if args.layers == 0 else int(args.layers)),
            eval_parquet=str(eval_parquet),
            steer_dir=str(steer_dir),
            save_dir=str(out_dir),
            layer_path=args.layer_path,
            cfg_dict=asdict(cfg),
            rank_hint=rank,
        )

    # Simple as-completed scheduler across GPUs.
    next_idx = 0
    in_flight: dict[asyncio.Task, int] = {}

    for r in range(min(use_gpus, len(jobs))):
        m, slug, label = jobs[next_idx]; next_idx += 1
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
                m, slug, label = jobs[next_idx]; next_idx += 1
                print(f"→ [gpu {rank}] start model='{m}' concept='{label}' (slug={slug})", flush=True)
                task = asyncio.create_task(run_one(rank, m, slug, label))
                in_flight[task] = rank


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--steer_dir", default="steering_vectors")
    p.add_argument("--eval_parquet", required=True)
    p.add_argument("--out_dir", default="cross_entropy")

    p.add_argument("--layers", default=4)
    p.add_argument("--layer_path", default=None)


    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dim", default="gpu")
    p.add_argument("--max_gpus", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main_async(args))
