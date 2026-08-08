"""Estimate model memory and choose a safe GPU layout."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Sequence

import torch


_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class GpuMemory:
    """Free and total memory for one visible GPU."""
    index: int
    free_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class ModelEstimate:
    """Estimated model size and supported parallel layouts."""
    model_name: str
    dtype: str
    weight_bytes: int
    largest_layer_bytes: int
    supports_tensor_parallel: bool
    tensor_parallel_divisors: tuple[int, ...]


@dataclass(frozen=True)
class ActorPlan:
    """GPU and worker counts chosen for one run."""
    model_name: str
    dtype: str
    weight_bytes: int
    required_bytes: int
    largest_layer_bytes: int
    gpu_budget_bytes: int
    gpus_per_actor: int
    logical_actors: int
    total_gpus: int
    visible_gpus: int
    tensor_parallel: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the plan as a plain dictionary."""
        return asdict(self)


def dtype_from_name(dtype: str) -> torch.dtype:
    """Map a command-line dtype name to a PyTorch dtype."""
    try:
        return _DTYPES[dtype]
    except KeyError as exc:
        choices = ", ".join(sorted(_DTYPES))
        raise ValueError(f"Unsupported dtype {dtype!r}; choose one of: {choices}") from exc


def discover_gpu_memory(max_gpus: int = 0) -> list[GpuMemory]:
    """Read free and total memory for visible GPUs."""
    visible = torch.cuda.device_count()
    if max_gpus > 0:
        visible = min(visible, max_gpus)
    memories: list[GpuMemory] = []
    for index in range(visible):
        with torch.cuda.device(index):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        memories.append(
            GpuMemory(
                index=index,
                free_bytes=int(free_bytes),
                total_bytes=int(total_bytes),
            )
        )
    return memories


def _positive_ints(values: Iterable[Any]) -> tuple[int, ...]:
    """Keep only positive integer values."""
    result = []
    for value in values:
        if isinstance(value, int) and value > 0:
            result.append(value)
    return tuple(result)


def _tp_divisors(config: Any, max_size: int) -> tuple[int, ...]:
    """Return tensor-parallel sizes that divide each model width."""
    text_config = getattr(config, "text_config", config)
    dimensions = _positive_ints(
        getattr(text_config, name, None)
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
        )
    )
    if not dimensions:
        return tuple(range(1, max_size + 1))
    return tuple(
        size
        for size in range(1, max_size + 1)
        if all(dimension % size == 0 for dimension in dimensions)
    )


def estimate_model(
    model_name: str,
    dtype: str = "bfloat16",
    *,
    max_tp_size: int = 1,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
) -> ModelEstimate:
    """Estimate model weight memory without loading the weights."""
    from accelerate import init_empty_weights
    from accelerate.utils import calculate_maximum_sizes
    from transformers import AutoConfig, AutoModelForCausalLM

    torch_dtype = dtype_from_name(dtype)
    config = AutoConfig.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
    total_bytes, largest_layer = calculate_maximum_sizes(model)
    if isinstance(largest_layer, tuple):
        largest_layer_bytes = int(largest_layer[0])
    else:
        largest_layer_bytes = int(largest_layer)
    text_config = getattr(config, "text_config", config)
    tp_plan = getattr(text_config, "base_model_tp_plan", None)
    return ModelEstimate(
        model_name=model_name,
        dtype=dtype,
        weight_bytes=int(total_bytes),
        largest_layer_bytes=largest_layer_bytes,
        supports_tensor_parallel=bool(tp_plan),
        tensor_parallel_divisors=_tp_divisors(text_config, max_tp_size),
    )


def plan_actor_mesh(
    estimate: ModelEstimate,
    gpu_memory: Sequence[GpuMemory],
    *,
    desired_actors: int,
    gpu_utilization: float = 0.90,
    inference_headroom: float = 1.20,
) -> ActorPlan:
    """Choose GPUs per model and how many models can run at once."""
    if not gpu_memory:
        raise RuntimeError("No CUDA devices are available to the model-placement planner.")
    if desired_actors < 1:
        raise ValueError("desired_actors must be at least 1")
    if not 0 < gpu_utilization <= 1:
        raise ValueError("gpu_utilization must be in (0, 1]")
    if inference_headroom < 1:
        raise ValueError("inference_headroom must be at least 1")

    # Every replica has the same topology, so size against the least-free GPU.
    gpu_budget_bytes = int(
        min(memory.free_bytes for memory in gpu_memory) * gpu_utilization
    )
    if gpu_budget_bytes <= 0:
        raise RuntimeError("No usable GPU memory remains after applying gpu_utilization.")
    if estimate.largest_layer_bytes > gpu_budget_bytes:
        raise RuntimeError(
            "The model's largest layer cannot fit on one GPU at the requested "
            "utilization; automatic tensor parallelism may replicate that layer."
        )

    required_bytes = math.ceil(estimate.weight_bytes * inference_headroom)
    minimum_size = max(1, math.ceil(required_bytes / gpu_budget_bytes))
    if minimum_size == 1:
        gpus_per_actor = 1
    else:
        if not estimate.supports_tensor_parallel:
            raise RuntimeError(
                f"{estimate.model_name!r} needs about {minimum_size} GPUs but its "
                "Transformers config does not define base_model_tp_plan."
            )
        valid_sizes = [
            size
            for size in estimate.tensor_parallel_divisors
            if minimum_size <= size <= len(gpu_memory)
        ]
        if not valid_sizes:
            raise RuntimeError(
                f"{estimate.model_name!r} needs at least {minimum_size} GPUs per "
                f"model copy, but no compatible TP size fits in {len(gpu_memory)} GPUs."
            )
        gpus_per_actor = min(valid_sizes)

    logical_actors = min(desired_actors, len(gpu_memory) // gpus_per_actor)
    if logical_actors < 1:
        raise RuntimeError(
            f"Need {gpus_per_actor} GPUs for one model copy, but only "
            f"{len(gpu_memory)} are available."
        )
    return ActorPlan(
        model_name=estimate.model_name,
        dtype=estimate.dtype,
        weight_bytes=estimate.weight_bytes,
        required_bytes=required_bytes,
        largest_layer_bytes=estimate.largest_layer_bytes,
        gpu_budget_bytes=gpu_budget_bytes,
        gpus_per_actor=gpus_per_actor,
        logical_actors=logical_actors,
        total_gpus=logical_actors * gpus_per_actor,
        visible_gpus=len(gpu_memory),
        tensor_parallel=gpus_per_actor > 1,
    )


def combine_model_estimates(estimates: Sequence[ModelEstimate]) -> ModelEstimate:
    """Combine memory estimates for models loaded together."""
    if not estimates:
        raise ValueError("At least one model estimate is required")
    common_divisors = set(estimates[0].tensor_parallel_divisors)
    for estimate in estimates[1:]:
        common_divisors.intersection_update(estimate.tensor_parallel_divisors)
    return ModelEstimate(
        model_name=" + ".join(estimate.model_name for estimate in estimates),
        dtype=" + ".join(estimate.dtype for estimate in estimates),
        weight_bytes=sum(estimate.weight_bytes for estimate in estimates),
        largest_layer_bytes=max(
            estimate.largest_layer_bytes for estimate in estimates
        ),
        supports_tensor_parallel=all(
            estimate.supports_tensor_parallel for estimate in estimates
        ),
        tensor_parallel_divisors=tuple(sorted(common_divisors)),
    )


def plan_single_gpu_actors(
    model_name: str,
    dtype: str,
    gpu_memory: Sequence[GpuMemory],
    *,
    desired_actors: int,
) -> ActorPlan:
    """Use one GPU for each running model without estimating memory."""
    if not gpu_memory:
        raise RuntimeError("No CUDA devices are available to the actor planner.")
    if desired_actors < 1:
        raise ValueError("desired_actors must be at least 1")
    logical_actors = min(desired_actors, len(gpu_memory))
    return ActorPlan(
        model_name=model_name,
        dtype=dtype,
        weight_bytes=0,
        required_bytes=0,
        largest_layer_bytes=0,
        gpu_budget_bytes=min(memory.free_bytes for memory in gpu_memory),
        gpus_per_actor=1,
        logical_actors=logical_actors,
        total_gpus=logical_actors,
        visible_gpus=len(gpu_memory),
        tensor_parallel=False,
    )


def format_bytes(value: int) -> str:
    """Format bytes as gibibytes."""
    return f"{value / (1024**3):.2f} GiB"
