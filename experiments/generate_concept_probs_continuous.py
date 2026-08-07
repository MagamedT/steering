import argparse
import asyncio
from dataclasses import asdict
from pathlib import Path

import torch
from monarch.actor import shutdown_context, this_host

from actors.concept_probs_continuous_actor import BehaviorActor, BehaviorConfig
from actors.utils import discover_jobs
from experiments.launcher_utils import run_ranked_jobs


async def main_async(args):
    cfg = BehaviorConfig(
        judge_model_name=args.judge_model,
        alpha_start=args.alpha_start,
        alpha_end=args.alpha_end,
        alpha_steps=args.alpha_steps,
        seed=args.seed,
        normalize=bool(getattr(args, "normalize", False)),
        n_samples_per_context=getattr(args, "samples_per_context", 12),
        gen_context_batch_size=getattr(args, "context_batch_size", 12),
        max_new_tokens=getattr(args, "max_new_tokens", 100),
        judge_batch_size=getattr(args, "judge_batch_size", 16),
        judge_rubric_max_new_tokens=getattr(args, "judge_max_new_tokens", 512),
    )

    steer_dir = Path(args.steer_dir)
    out_dir = Path(args.out_dir)
    contexts_file = Path(args.contexts_file)
    if not steer_dir.exists():
        raise RuntimeError(f"--steer_dir '{steer_dir}' does not exist")
    if not contexts_file.exists():
        raise RuntimeError(f"--contexts_file '{contexts_file}' does not exist")

    jobs = discover_jobs(steer_dir, list(args.models))
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
    workers = mesh.spawn("concept_probs_continuous", BehaviorActor)

    def actor_for(rank: int):
        return workers.slice(**{args.dim: rank})

    async def run_one(
        rank: int,
        model_name: str,
        concept_slug: str,
        concept_label: str,
    ):
        return await actor_for(rank).compute_behavior_curves.call_one(
            model_name=model_name,
            concept_slug=concept_slug,
            concept_label=concept_label,
            block_idx_to_steer=None if args.layers == 0 else int(args.layers),
            contexts_file=str(contexts_file),
            steer_dir=str(steer_dir),
            save_dir=str(out_dir),
            layer_path=args.layer_path,
            cfg_dict=asdict(cfg),
            rank_hint=rank,
            exact_layer_idx=getattr(args, "layer", None),
        )

    async for rank, result in run_ranked_jobs(jobs, use_gpus, run_one):
        if isinstance(result, Exception):
            print(f"[gpu {rank}] EXCEPTION: {result}", flush=True)
            raise result
        if isinstance(result, dict) and result.get("ok"):
            files = [
                item.get("file")
                for item in result.get("results", [])
                if item.get("file")
            ]
            print(
                f"[gpu {rank}] finished -> {files[0] if files else '(no files)'}",
                flush=True,
            )
        else:
            print(f"[gpu {rank}] unexpected result: {result}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="HF model ids/paths for which steering vectors exist.",
    )
    parser.add_argument(
        "--judge_model",
        required=True,
        help=(
            "Rubric judge whose true/false logits score each completion with "
            "sigmoid(z_true-z_false); scores are averaged per context."
        ),
    )
    parser.add_argument("--steer_dir", default="steering_vectors")
    parser.add_argument("--contexts_file", default="data/contexts.jsonl")
    parser.add_argument("--out_dir", default="behavior_data_continuous")
    parser.add_argument("--alpha_start", type=float, default=-40.0)
    parser.add_argument("--alpha_end", type=float, default=40.0)
    parser.add_argument("--alpha_steps", type=int, default=41)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples_per_context", type=int, default=12)
    parser.add_argument("--context_batch_size", type=int, default=12)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--judge_batch_size", type=int, default=16)
    parser.add_argument("--judge_max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize each steering vector to unit norm before applying alpha.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=4,
        help="0 means all layers; a positive value samples that many layers.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Run one exact layer; overrides --layers.",
    )
    parser.add_argument("--layer_path", default=None)
    parser.add_argument("--dim", default="gpu")
    parser.add_argument("--max_gpus", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    finally:
        shutdown_context().get()
