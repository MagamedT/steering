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

from ..actors.mmlu_actor import MMLUActor, MMLUEvalConfig
from ..actors.utils import discover_jobs
from .launcher_utils import run_ranked_jobs


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

    # As-completed scheduling keeps all GPUs busy during variable-length jobs.
    async for rank, res in run_ranked_jobs(jobs, use_gpus, run_one):
        if isinstance(res, Exception):
            print(f"[gpu {rank}] FAILED: {type(res).__name__}: {res}", flush=True)
            continue
        print(f"[gpu {rank}] finished: {res}", flush=True)


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
