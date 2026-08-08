from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
from typing import Any, Sequence

from monarch.actor import this_host
from monarch.spmd import setup_torch_elastic_env_async

from .placement import (
    ActorPlan,
    combine_model_estimates,
    discover_gpu_memory,
    estimate_model,
    format_bytes,
    plan_actor_mesh,
    plan_single_gpu_actors,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    dtype: str


def group_bounds(logical_rank: int, gpus_per_actor: int) -> tuple[int, int]:
    start = logical_rank * gpus_per_actor
    return start, start + gpus_per_actor


def leader_result(value_mesh):
    results = [value for _point, value in value_mesh.items() if value is not None]
    if len(results) != 1:
        raise RuntimeError(
            f"Expected one logical-actor leader result, got {len(results)}"
        )
    return results[0]


def plan_models(
    args,
    model_specs: Sequence[ModelSpec],
    *,
    desired_actors: int,
) -> ActorPlan:
    if not model_specs:
        raise ValueError("At least one model is required")
    gpu_memory = discover_gpu_memory(getattr(args, "max_gpus", 0))
    mode = getattr(args, "model_parallel_size", "auto")
    if mode == "1":
        return plan_single_gpu_actors(
            " + ".join(spec.name for spec in model_specs),
            " + ".join(spec.dtype for spec in model_specs),
            gpu_memory,
            desired_actors=desired_actors,
        )
    if mode != "auto":
        raise ValueError("--model_parallel_size must be '1' or 'auto'")

    estimates = [
        estimate_model(
            spec.name,
            spec.dtype,
            max_tp_size=len(gpu_memory),
            local_files_only=getattr(args, "local_files_only", False),
            trust_remote_code=getattr(args, "trust_remote_code", False),
        )
        for spec in model_specs
    ]
    estimate = (
        estimates[0] if len(estimates) == 1 else combine_model_estimates(estimates)
    )
    return plan_actor_mesh(
        estimate,
        gpu_memory,
        desired_actors=desired_actors,
        gpu_utilization=getattr(args, "gpu_utilization", 0.90),
        inference_headroom=getattr(args, "inference_headroom", 1.20),
    )


def print_plan(
    plan: ActorPlan,
    *,
    stage: str,
    models: Sequence[ModelSpec],
    **metadata: Any,
) -> None:
    payload = plan.to_dict()
    payload.update(
        stage=stage,
        models=[{"name": model.name, "dtype": model.dtype} for model in models],
        **metadata,
    )
    print(json.dumps(payload, indent=2), flush=True)
    details = (
        f", weights={format_bytes(plan.weight_bytes)}, "
        f"planned={format_bytes(plan.required_bytes)}"
        if plan.weight_bytes
        else ""
    )
    print(
        f"{stage} plan: {plan.logical_actors} logical actor(s) x "
        f"{plan.gpus_per_actor} GPU(s){details}",
        flush=True,
    )


class LogicalActorPool:
    def __init__(self, actors, dim: str, plan: ActorPlan) -> None:
        self.actors = actors
        self.dim = dim
        self.plan = plan

    def group(self, logical_rank: int):
        start, stop = group_bounds(logical_rank, self.plan.gpus_per_actor)
        return self.actors.slice(**{self.dim: slice(start, stop)})

    async def call(self, logical_rank: int, endpoint_name: str, *args, **kwargs):
        endpoint = getattr(self.group(logical_rank), endpoint_name)
        values = await endpoint.call(*args, **kwargs)
        return leader_result(values)


@asynccontextmanager
async def actor_pool(args, plan: ActorPlan, name: str, actor_cls, *actor_args):
    mesh = this_host().spawn_procs(per_host={getattr(args, "dim", "gpu"): plan.total_gpus})
    print(mesh.to_table(), flush=True)
    actors = None
    try:
        await setup_torch_elastic_env_async(mesh)
        actors = mesh.spawn(name, actor_cls, *actor_args)
        yield LogicalActorPool(actors, getattr(args, "dim", "gpu"), plan)
    finally:
        if actors is not None:
            await actors.close.call()
        await mesh.stop()


def add_distributed_args(parser, *, default_dtype: str = "bfloat16") -> None:
    parser.add_argument("--dim", default="gpu", help="Monarch process-mesh dimension.")
    parser.add_argument(
        "--max_gpus", type=int, default=0, help="Maximum visible GPUs to use (0=all)."
    )
    parser.add_argument(
        "--model_parallel_size",
        choices=("1", "auto"),
        default="auto",
        help=(
            "GPUs per logical actor: '1' uses one rank; 'auto' chooses a safe "
            "tensor-parallel group and packs replicas onto the remaining GPUs."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default=default_dtype,
    )
    parser.add_argument("--gpu_utilization", type=float, default=0.90)
    parser.add_argument("--inference_headroom", type=float, default=1.20)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--plan_only",
        action="store_true",
        help="Print the planned topology without loading model weights.",
    )
