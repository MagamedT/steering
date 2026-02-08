#!/usr/bin/env python3
# Launcher for MMLU-vs-α steering sweeps.

import os

# Keep logs ASCII-only to avoid hyperactor_mesh UTF-8 boundary panics and noisy bars.
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

# DeepEval telemetry/tracing off (best-effort)
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("OTEL_TRACES_EXPORTER", "none")
os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")
os.environ.setdefault("OTEL_LOGS_EXPORTER", "none")

import argparse
import asyncio
from dataclasses import asdict
from pathlib import Path

import torch
from monarch.actor import this_host

from actors.mmlu_actor import MMLUActor, MMLUEvalConfig
from actors.utils import model_slug


def discover_jobs(steer_dir: Path, models: list[str]) -> list[tuple[str, str, str]]:
    jobs: list[tuple[str, str, str]] = []
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
    tasks_arg = None
    # Accept "all" to let DeepEval evaluate every MMLU subject.
    if args.tasks and not (len(args.tasks) == 1 and args.tasks[0].lower() == "all"):
        tasks_arg = args.tasks

    cfg = MMLUEvalConfig(
        seed=args.seed,
        tasks=tasks_arg,
    )

    steer_dir = Path(args.steer_dir)
    out_dir = Path(args.out_dir)

    if not steer_dir.exists():
        raise RuntimeError(f"--steer_dir '{steer_dir}' does not exist")

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

    workers = mesh.spawn("mmlu", MMLUActor)

    def actor_for(rank: int):
        return workers.slice(**{args.dim: rank})

    async def run_one(rank: int, model_name: str, concept_slug: str, concept_label: str):
        return await actor_for(rank).compute_mmlu.call_one(
            model_name=model_name,
            concept_slug=concept_slug,
            concept_label=concept_label,
            steer_dir=str(steer_dir),
            save_dir=str(out_dir),
            # 0 => all layers; positive int => evenly sample that many layers.
            block_idx_to_steer=(None if args.layers == 0 else int(args.layers)),
            layer_path=args.layer_path,
            cfg_dict=asdict(cfg),
            rank_hint=rank,
        )

    next_idx = 0
    in_flight: dict[asyncio.Task, int] = {}

    # Prime each GPU with one job
    for r in range(min(use_gpus, len(jobs))):
        model_name, slug, label = jobs[next_idx]
        next_idx += 1
        print(f"-> [gpu {r}] start model='{model_name}' concept='{label}' (slug={slug})", flush=True)
        task = asyncio.create_task(run_one(r, model_name, slug, label))
        in_flight[task] = r

    # Simple work-stealing loop
    while in_flight:
        done, _ = await asyncio.wait(in_flight.keys(), return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            rank = in_flight.pop(t)
            try:
                res = await t
                print(f"[gpu {rank}] finished: {res}", flush=True)
            except Exception as e:
                print(f"[gpu {rank}] FAILED: {type(e).__name__}: {e}", flush=True)

            if next_idx < len(jobs):
                model_name, slug, label = jobs[next_idx]
                next_idx += 1
                print(f"-> [gpu {rank}] start model='{model_name}' concept='{label}' (slug={slug})", flush=True)
                task = asyncio.create_task(run_one(rank, model_name, slug, label))
                in_flight[task] = rank


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--steer_dir", default="steering_vectors")
    p.add_argument("--out_dir", default="mmlu")
    p.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="e.g. HIGH_SCHOOL_COMPUTER_SCIENCE ASTRONOMY, or 'all'",
    )

    p.add_argument(
        "--layers",
        default=4,
        help="Which layer indices to steer; omit/None => all layers.",
    )
    p.add_argument("--layer_path", default=None)

    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--dim", default="gpu")
    p.add_argument("--max_gpus", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main_async(args))
