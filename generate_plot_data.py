import argparse
import asyncio
from pathlib import Path
from dataclasses import asdict

import torch
from monarch.actor import this_host

from actors.steering_plot_actor import TokenActor, TokenPlotConfig, model_slug


def discover_jobs(steer_dir: Path, models: list[str]) -> list[tuple[str, str, str]]:
    """Return list of (model_name, concept_slug, concept_label). Label is slug with spaces."""
    jobs = []
    for model_name in models:
        mslug = model_slug(model_name)
        base = steer_dir / mslug
        if not base.exists():
            continue
        for concept_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
            slug = concept_dir.name
            # crude label
            label = slug.replace("_", " ")
            # require at least one layer file
            if not any(concept_dir.glob("layer_*.pt")):
                continue
            jobs.append((model_name, slug, label))
    return jobs


async def main_async(args):
    # Keep CLI simple: we only override seed here and rely on TokenPlotConfig defaults.
    cfg = TokenPlotConfig(seed=args.seed)

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
        raise RuntimeError(f"No (model, concept) pairs discovered under {steer_dir} for given models.")

    visible = torch.cuda.device_count()
    if visible < 1:
        raise RuntimeError("No CUDA devices visible.")
    use_gpus = min(visible, len(jobs))
    if args.max_gpus and args.max_gpus > 0:
        use_gpus = min(use_gpus, args.max_gpus)

    mesh = this_host().spawn_procs(per_host={args.dim: use_gpus})
    print(mesh.to_table(), flush=True)

    workers = mesh.spawn("plot", TokenActor)

    def actor_for(rank: int):
        return workers.slice(**{args.dim: rank})

    async def run_one(rank: int, model_name: str, concept_slug: str, concept_label: str):
        return await actor_for(rank).compute_plot_curves.call_one(
            model_name=model_name,
            concept_slug=concept_slug,
            concept_label=concept_label,
            # [None] means "all available transformer blocks".
            block_idx_to_steer=([None] if args.layers == [None] else [int(i) for i in args.layers]),
            contexts_file=str(contexts_file),
            steer_dir=str(steer_dir),
            save_dir=str(out_dir),
            layer_path=args.layer_path,
            cfg_dict=asdict(cfg),
            rank_hint=rank,
        )

    # As-completed scheduling keeps all GPUs busy during variable-length jobs.
    next_idx = 0
    in_flight: dict[asyncio.Task, int] = {}

    # kick initial
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
                if isinstance(res, dict) and "ok" in res:
                    print(f"[gpu {rank}] finished", flush=True)
                else:
                    print(f"[gpu {rank}] unexpected result: {res}", flush=True)
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
                   help="HF model ids or local paths for which steer vectors exist.")
    p.add_argument("--steer_dir", default="steering_vectors",
                   help="Root directory containing steering vectors (model_slug/concept_slug/layer_*.pt).")
    p.add_argument("--contexts_file", default="contexts.jsonl",
                   help="Text file with one input context per line.")
    p.add_argument("--out_dir", default="plot_data",
                   help="Where to write .npz curve files.")
    p.add_argument("--layers", nargs="+", default=[None],
                   help="Layer indices (e.g., 5 10 15) or all if kept to None. You cannot pass text, only [None] or spaced indexis")
    p.add_argument("--layer_path", default=None,
                   help="Override path to block list (e.g., 'model.layers').")

    # α grid / top-k / tokenization
    # scheduling
    p.add_argument("--dim", default="gpu", help="Mesh dimension name (use 'gpu' if your env shows that).")
    p.add_argument("--max_gpus", type=int, default=0, help="Limit number of GPUs (0 = all visible).")

    # misc
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main_async(args))
