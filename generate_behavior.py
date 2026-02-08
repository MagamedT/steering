import argparse
import asyncio
from dataclasses import asdict
from pathlib import Path

import torch
from monarch.actor import this_host

from actors.behavior_score_actor import BehaviorActor, BehaviorConfig
from actors.steering_plot_actor import model_slug


def discover_jobs(steer_dir: Path, models: list[str]) -> list[tuple[str, str, str]]:
    """Return list of (model_name, concept_slug, concept_label)."""
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
    # Behavior settings come from actor defaults.
    cfg = BehaviorConfig(judge_model_name=args.judge_model)

    steer_dir = Path(args.steer_dir)
    out_dir = Path(args.out_dir)
    contexts_file = Path(args.contexts_file)

    if not steer_dir.exists():
        raise RuntimeError(f"--steer_dir '{steer_dir}' does not exist")
    if not contexts_file.exists():
        raise RuntimeError(f"--contexts_file '{contexts_file}' does not exist")

    models = list(args.models)
    jobs = discover_jobs(steer_dir, models)
    if not jobs:
        raise RuntimeError(
            f"No (model, concept) pairs discovered under {steer_dir} for given models."
        )

    visible = torch.cuda.device_count()
    if visible < 1:
        raise RuntimeError("No CUDA devices visible.")

    use_gpus = min(visible, len(jobs))
    if args.max_gpus and args.max_gpus > 0:
        use_gpus = min(use_gpus, args.max_gpus)

    mesh = this_host().spawn_procs(per_host={args.dim: use_gpus})
    print(mesh.to_table(), flush=True)

    workers = mesh.spawn("behavior", BehaviorActor)

    def actor_for(rank: int):
        return workers.slice(**{args.dim: rank})

    async def run_one(rank: int, model_name: str, concept_slug: str, concept_label: str):
        return await actor_for(rank).compute_behavior_curves.call_one(
            model_name=model_name,
            concept_slug=concept_slug,
            concept_label=concept_label,
            # 0 => all layers; positive int => evenly sample that many layers.
            block_idx_to_steer=(None if args.layers == 0 else int(args.layers)),
            contexts_file=str(contexts_file),
            steer_dir=str(steer_dir),
            save_dir=str(out_dir),
            layer_path=args.layer_path,
            cfg_dict=asdict(cfg),
            rank_hint=rank,
        )

    # Dynamic scheduler (as-completed)
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
            try:
                res = await t
                if isinstance(res, dict) and res.get("ok"):
                    files = [rinfo.get("file") for rinfo in (res.get("results") or []) if rinfo.get("file")]
                    msg = files[0] if files else "(no files)"
                    print(f"[gpu {rank}] finished -> {msg}", flush=True)
                else:
                    print(f"[gpu {rank}] unexpected result: {res}", flush=True)
            except Exception as e:
                print(f"[gpu {rank}] EXCEPTION: {e}", flush=True)
                raise

            if next_idx < len(jobs):
                m, slug, label = jobs[next_idx]
                next_idx += 1
                print(f"→ [gpu {rank}] start model='{m}' concept='{label}' (slug={slug})", flush=True)
                task = asyncio.create_task(run_one(rank, m, slug, label))
                in_flight[task] = rank


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--models", nargs="+", required=True, help="HF model ids/paths for which steer vectors exist.")
    p.add_argument("--judge_model", required=True, help="HF model id/path for the binary judge model.")

    p.add_argument("--steer_dir", default="steering_vectors", help="Root: model_slug/concept_slug/layer_*.pt")
    p.add_argument("--contexts_file", default="contexts.jsonl", help="JSONL contexts (negatives + per-concept positives)")
    p.add_argument("--out_dir", default="behavior_data", help="Where to write .npz behavior curve files")

    p.add_argument(
        "--layers",
        type=int,
        default=4,
        help="0 => all layers; positive int => evenly sample that many layers.",
    )
    p.add_argument(
        "--layer_path",
        default=None,
        help="Override path to transformer block list (e.g., 'model.layers').",
    )

    # scheduling
    p.add_argument("--dim", default="gpu", help="Mesh dimension name (use 'gpu' if your env uses that).")
    p.add_argument("--max_gpus", type=int, default=0, help="Limit number of GPUs (0=auto)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main_async(args))
