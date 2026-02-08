import argparse
import asyncio
from dataclasses import asdict
from pathlib import Path
import json
import torch
from monarch.actor import this_host

from actors.steering_vector_actor import SteeringActor, SteeringConfig
from actors.utils import discover_concepts

def pair_jobs(models, concepts, mode="product"):
    """
    Build (model, concept_slug, concept_label) tuples.

    modes:
    - "product": cartesian product of all models with all (slug, label) concepts.
    - "zip": pair elements one-to-one up to the shorter list length.
    - "zip_cycle": pair across the longer list length, cycling through the shorter one.

    Returns a list of (model, slug, label) tuples.
    """
    jobs = []
    if mode == "product":
        for model in models:
            for slug, label in concepts:
                jobs.append((model, slug, label))
    elif mode == "zip":
        for idx, model in enumerate(models[:len(concepts)]):
            slug, label = concepts[idx]
            jobs.append((model, slug, label))
    else:  # zip_cycle
        L = max(len(models), len(concepts))
        for i in range(L):
            model = models[i % len(models)]
            slug, label = concepts[i % len(concepts)]
            jobs.append((model, slug, label))
    return jobs

async def main_async(args):
    cfg = SteeringConfig(seed=args.seed)

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        raise RuntimeError(f"--in_dir '{in_dir}' does not exist")

    concepts = discover_concepts(in_dir)
    if not concepts:
        raise RuntimeError(f"No concept pairs (*.related.jsonl & *.unrelated.jsonl) found under {in_dir}")

    models = list(args.models)
    # Pairing controls whether we run full cartesian product or a zipped pairing.
    jobs = pair_jobs(models, concepts, mode=args.pairing)

    visible = torch.cuda.device_count()
    if visible < 1:
        raise RuntimeError("No CUDA devices visible.")
    use_gpus = min(visible, len(jobs))
    if args.max_gpus and args.max_gpus > 0:
        use_gpus = min(use_gpus, args.max_gpus)

    mesh = this_host().spawn_procs(per_host={args.dim: use_gpus})
    print(mesh.to_table(), flush=True)
    steerer = mesh.spawn("steer", SteeringActor)

    def actor_for(rank: int):
        return steerer.slice(**{args.dim: rank})

    async def run_one(rank: int, model_name: str, concept_slug: str, concept_label: str):
        return await actor_for(rank).compute_for.call_one(
            model_name,
            concept_slug,
            concept_label,
            # [None] means "all layers" and is expanded by the actor.
            [None] if args.layers == [None] else [int(i) for i in args.layers],
            asdict(cfg),
            str(in_dir),
            args.save_dir,
            args.layer_path,
            rank,  # rank_hint
        )

    # Dynamic scheduler: each GPU immediately pulls the next pending job.
    next_idx = 0
    in_flight = {}

    # Kick off initial tasks
    for r in range(min(use_gpus, len(jobs))):
        m, slug, label = jobs[next_idx]; next_idx += 1
        print(f"→ [gpu {r}] start model='{m}' concept='{label}' (slug={slug})", flush=True)
        task = asyncio.create_task(run_one(r, m, slug, label))
        in_flight[task] = r

    while in_flight:
        done, _ = await asyncio.wait(in_flight.keys(), return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            rank = in_flight.pop(t)
            try:
                res = await t
                if isinstance(res, dict) and "error" in res:
                    print(f"[gpu {rank}] ERROR: {res['error']}", flush=True)
                else:
                    print(
                        f"[gpu {res['rank']}] model='{res['model']}' concept='{res['concept']}' "
                        f"layers={len(res['layers'])} saved={len(res['saved'])} files",
                        flush=True,
                    )
            except Exception as e:
                print(f"[gpu {rank}] EXCEPTION: {e}", flush=True)
                raise

            if next_idx < len(jobs):
                m, slug, label = jobs[next_idx]; next_idx += 1
                print(f"→ [gpu {rank}] start model='{m}' concept='{label}' (slug={slug})", flush=True)
                task = asyncio.create_task(run_one(rank, m, slug, label))
                in_flight[task] = rank


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True,
                   help="HF model ids or local paths (space-separated)")
    p.add_argument("--in_dir", default="prompts",
                   help="Directory containing <slug>.related.jsonl and <slug>.unrelated.jsonl")
    p.add_argument("--save_dir", default="steering_vectors",
                   help="Where to save layer_<i>.pt steering vectors")
    p.add_argument("--layers", nargs="+", default=[None],
                   help="Layer indices (e.g., 5 10 15) or [None] to compute for all blocks. No over input are valid")
    p.add_argument("--layer_path", default=None,
                   help="Optional override to the block ModuleList (e.g., 'model.layers').")

    # Pairing & scheduling
    p.add_argument("--pairing", choices=["product", "zip", "zip_cycle"], default="product",
                   help="How to pair models × discovered concepts (default: product).")
    p.add_argument("--dim", default="gpu",
                   help="Mesh dimension name (use 'gpu' if your env shows that).")
    p.add_argument("--max_gpus", type=int, default=0,
                   help="Limit number of GPUs to use on this host (0 = all visible).")

    # Tokenization / compute knobs
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main_async(args))
