from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from monarch.actor import this_host
from monarch.spmd import setup_torch_elastic_env_async

from actors.model_placement import (
    discover_gpu_memory,
    estimate_model,
    format_bytes,
    plan_actor_mesh,
)
from actors.tensor_parallel_steering_actor import (
    TensorParallelSteeringActor,
    TensorParallelSteeringConfig,
)
from actors.utils import discover_concepts
from experiments.launcher_utils import run_ranked_jobs


def group_bounds(logical_rank: int, gpus_per_actor: int) -> tuple[int, int]:
    start = logical_rank * gpus_per_actor
    return start, start + gpus_per_actor


def leader_result(value_mesh):
    results = [value for _point, value in value_mesh.items() if value is not None]
    if len(results) != 1:
        raise RuntimeError(f"Expected one logical-actor leader result, got {len(results)}")
    return results[0]


async def run_dynamic(args) -> None:
    if len(args.models) != 1:
        raise ValueError(
            "--model_parallel_size auto currently requires exactly one model."
        )
    if args.pairing != "product":
        raise ValueError(
            "--model_parallel_size auto currently supports --pairing product only."
        )
    model_name = args.models[0]
    prompt_root = Path(args.in_dir)
    if not prompt_root.exists():
        raise RuntimeError(f"--in_dir {str(prompt_root)!r} does not exist")
    concepts = discover_concepts(prompt_root)
    if not concepts:
        raise RuntimeError(f"No positive/negative concept pairs found under {prompt_root}")

    gpu_memory = discover_gpu_memory(args.max_gpus)
    estimate = estimate_model(
        model_name,
        args.dtype,
        max_tp_size=len(gpu_memory),
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    plan = plan_actor_mesh(
        estimate,
        gpu_memory,
        desired_actors=len(concepts),
        gpu_utilization=args.gpu_utilization,
        inference_headroom=args.inference_headroom,
    )
    print(json.dumps(plan.to_dict(), indent=2), flush=True)
    print(
        "experimental model-parallel plan: "
        f"weights={format_bytes(plan.weight_bytes)}, "
        f"planned={format_bytes(plan.required_bytes)}, "
        f"{plan.logical_actors} logical actor(s) x {plan.gpus_per_actor} GPU(s)",
        flush=True,
    )
    if args.plan_only:
        return

    cfg = TensorParallelSteeringConfig(
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
        n_positive=args.n_positive,
        n_negative=args.n_negative,
        contrastive=args.contrastive,
        block_per_pass=args.block_per_pass,
        progress_every=args.progress_every,
    )
    proc_mesh = this_host().spawn_procs(per_host={args.dim: plan.total_gpus})
    actors = None
    try:
        await setup_torch_elastic_env_async(proc_mesh)
        actors = proc_mesh.spawn(
            "dynamic_tp_steering",
            TensorParallelSteeringActor,
            model_name,
            args.dtype,
            plan.logical_actors,
            plan.gpus_per_actor,
            args.local_files_only,
            args.trust_remote_code,
        )

        def actor_group(logical_rank: int):
            start, stop = group_bounds(logical_rank, plan.gpus_per_actor)
            return actors.slice(**{args.dim: slice(start, stop)})

        async def run_one(logical_rank: int, concept_slug: str, concept_label: str):
            values = await actor_group(logical_rank).compute_for.call(
                concept_slug,
                concept_label,
                [None] if args.layers == [None] else [int(index) for index in args.layers],
                asdict(cfg),
                str(prompt_root),
                args.save_dir,
                args.layer_path,
                logical_rank,
            )
            return leader_result(values)

        jobs = [(slug, label) for slug, label in concepts]
        async for logical_rank, result in run_ranked_jobs(
            jobs, plan.logical_actors, run_one
        ):
            if isinstance(result, Exception):
                raise result
            if "error" in result:
                raise RuntimeError(result["error"])
            print(
                f"[logical actor {logical_rank}; {plan.gpus_per_actor} GPU(s)] "
                f"concept={result['concept']!r} saved={len(result['saved'])}",
                flush=True,
            )
    finally:
        if actors is not None:
            await actors.close.call()
        await proc_mesh.stop()
