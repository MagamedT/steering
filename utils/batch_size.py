"""Compute safe inference batch sizes from model and GPU memory geometry."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Literal, Sequence

import torch
import torch.distributed as dist


Workload = Literal["hidden", "forward", "cross_entropy", "generate", "rubric"]

_MIB = 1024**2
_DTYPE_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
}


@dataclass(frozen=True)
class BatchEstimate:
    """A selected batch size and the memory arithmetic used to choose it."""

    batch_size: int
    critical_batch_size: int
    effective_batch_size: int
    bytes_per_sequence: int
    free_bytes: int
    usable_bytes: int
    reserve_bytes: int


@dataclass(frozen=True)
class _ModelShape:
    layers: int
    hidden: int
    intermediate: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    dtype_bytes: int
    active_experts: int
    fused_attention: bool
    context_limit: int | None


def _first_int(source: Any, names: Sequence[str], default: int | None = None) -> int:
    for name in names:
        value = getattr(source, name, None)
        if isinstance(value, int) and value > 0:
            return value
    if default is not None:
        return default
    raise ValueError(f"Model config is missing {', '.join(names)}")


def _dtype_size(model: Any, dtype: torch.dtype | str | None) -> int:
    if isinstance(dtype, str):
        try:
            return _DTYPE_BYTES[dtype]
        except KeyError as exc:
            raise ValueError(f"Unsupported compute dtype: {dtype!r}") from exc
    if isinstance(dtype, torch.dtype):
        return torch.empty((), dtype=dtype).element_size()
    model_dtype = getattr(model, "dtype", None)
    if isinstance(model_dtype, torch.dtype):
        return torch.empty((), dtype=model_dtype).element_size()
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.element_size()
    return 4


def _torch_dtype(model: Any, dtype: torch.dtype | str | None) -> torch.dtype:
    if isinstance(dtype, str):
        try:
            return {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }[dtype]
        except KeyError as exc:
            raise ValueError(f"Unsupported compute dtype: {dtype!r}") from exc
    if isinstance(dtype, torch.dtype):
        return dtype
    model_dtype = getattr(model, "dtype", None)
    if isinstance(model_dtype, torch.dtype):
        return model_dtype
    for parameter in model.parameters():
        parameter_dtype = getattr(parameter, "dtype", None)
        if isinstance(parameter_dtype, torch.dtype) and parameter.is_floating_point():
            return parameter_dtype
    return torch.float32


def _attention_implementation(model: Any) -> str:
    config = model.config
    text = getattr(config, "text_config", config)
    return str(
        getattr(text, "_attn_implementation", None)
        or getattr(config, "_attn_implementation", None)
        or ""
    ).lower()


def _model_shape(model: Any, dtype: torch.dtype | str | None) -> _ModelShape:
    config = model.config
    text = getattr(config, "text_config", config)
    hidden = _first_int(text, ("hidden_size", "n_embd", "d_model"))
    heads = _first_int(text, ("num_attention_heads", "n_head"))
    intermediate = _first_int(
        text,
        ("intermediate_size", "ffn_dim", "n_inner"),
        default=4 * hidden,
    )
    vocab = getattr(text, "vocab_size", None)
    if not isinstance(vocab, int) or vocab < 1:
        vocab = _first_int(config, ("vocab_size",))
    else:
        vocab = int(vocab)
    implementation = _attention_implementation(model)
    context_limit = None
    for name in ("max_position_embeddings", "n_positions", "n_ctx"):
        value = getattr(text, name, None)
        if isinstance(value, int) and value > 0:
            context_limit = value
            break
    return _ModelShape(
        layers=_first_int(text, ("num_hidden_layers", "n_layer", "num_layers")),
        hidden=hidden,
        intermediate=intermediate,
        heads=heads,
        kv_heads=_first_int(text, ("num_key_value_heads",), default=heads),
        head_dim=_first_int(text, ("head_dim",), default=hidden // heads),
        vocab=vocab,
        dtype_bytes=_dtype_size(model, dtype),
        active_experts=_first_int(
            text,
            ("num_experts_per_tok", "num_selected_experts"),
            default=1,
        ),
        # "sdpa" is a dispatcher and may fall back to the quadratic math
        # kernel for a concrete mask/dtype/shape. Only an explicitly selected
        # flash implementation is safe to model as linear-memory attention.
        fused_attention=implementation in {
            "flash_attention_2",
            "flash_attention_3",
            "flash_attention",
        },
        context_limit=context_limit,
    )


def _sdpa_uses_linear_memory(
    shape: _ModelShape,
    device: torch.device,
    dtype: torch.dtype,
    query_tokens: int,
    key_tokens: int,
) -> bool:
    """Ask PyTorch whether this masked SDPA geometry has a fused backend."""
    try:
        from torch.nn.attention import (
            SDPAParams,
            can_use_efficient_attention,
            can_use_flash_attention,
        )

        base = torch.empty(
            (1, shape.heads, 1, shape.head_dim),
            device=device,
            dtype=dtype,
        )
        query = base.expand(1, shape.heads, query_tokens, shape.head_dim)
        key = base.expand(1, shape.heads, key_tokens, shape.head_dim)
        mask = torch.empty(
            (1, 1, 1, key_tokens),
            device=device,
            dtype=dtype,
        )
        params = SDPAParams(query, key, key, mask, 0.0, False, False)
        return bool(
            (
                torch.backends.cuda.flash_sdp_enabled()
                and can_use_flash_attention(params)
            )
            or (
                torch.backends.cuda.mem_efficient_sdp_enabled()
                and can_use_efficient_attention(params)
            )
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _activation_bytes(
    shape: _ModelShape, tokens: int, key_tokens: int | None = None
) -> int:
    kv_width = shape.kv_heads * shape.head_dim
    layer_width = 10 * shape.hidden + max(
        3 * shape.intermediate * shape.active_experts,
        shape.hidden + 2 * kv_width,
    )
    result = tokens * shape.dtype_bytes * layer_width
    if not shape.fused_attention:
        # Protect unknown/eager attention backends that materialize score/probability matrices.
        keys = tokens if key_tokens is None else key_tokens
        result += 2 * shape.heads * tokens * keys * max(shape.dtype_bytes, 4)
    return result


def _kv_bytes(shape: _ModelShape, tokens: int) -> int:
    kv_width = shape.kv_heads * shape.head_dim
    return 2 * shape.layers * tokens * kv_width * shape.dtype_bytes


def _tp_gather_bytes(shape: _ModelShape, positions: int, tp_size: int) -> int:
    """Workspace for reconstructing tensor-parallel vocabulary logits."""
    if tp_size <= 1:
        return 0
    full = positions * shape.vocab * shape.dtype_bytes
    local = math.ceil(full / tp_size)
    # Manual all_gather owns a full gathered list and a full concatenation at
    # once, alongside the original and contiguous local shards. This also
    # covers DTensor redistribution workspace conservatively.
    return full + 2 * local


def _supports_last_logit(model: Any) -> bool:
    supported = getattr(model, "_supports_logits_to_keep", None)
    if callable(supported):
        try:
            return bool(supported())
        except (TypeError, ValueError):
            pass
    return False


def _sequence_bytes(
    model: Any,
    shape: _ModelShape,
    *,
    kind: Workload,
    prompt_tokens: int,
    new_tokens: int,
    use_cache: bool,
    tp_size: int = 1,
) -> int:
    activation = _activation_bytes(shape, prompt_tokens)
    cache = _kv_bytes(shape, prompt_tokens) if use_cache else 0
    if kind == "hidden":
        return activation + cache
    if kind == "forward":
        logits = prompt_tokens * shape.vocab * shape.dtype_bytes
        # The tasks select/cast the final-token logits after the full LM output exists.
        return (
            activation
            + cache
            + logits
            + _tp_gather_bytes(shape, prompt_tokens, tp_size)
            + 32 * shape.vocab
        )
    if kind == "cross_entropy":
        # Original logits, their float32 copy, and cross-entropy/log-softmax workspace.
        logits = prompt_tokens * shape.vocab * (shape.dtype_bytes + 8)
        return (
            activation
            + logits
            + _tp_gather_bytes(shape, prompt_tokens, tp_size)
        )
    if kind in {"generate", "rubric"}:
        total_tokens = prompt_tokens + new_tokens
        logits_positions = 1 if _supports_last_logit(model) else prompt_tokens
        prefill = (
            activation
            + _kv_bytes(shape, prompt_tokens)
            + logits_positions * shape.vocab * shape.dtype_bytes
            + _tp_gather_bytes(shape, logits_positions, tp_size)
        )
        decode = (
            _activation_bytes(shape, 1, total_tokens)
            + _kv_bytes(shape, total_tokens)
            # Transformers sampling can simultaneously own float32 logits,
            # sorted logits, int64 sort indices, cumulative probabilities,
            # masks, processed scores, and multinomial probabilities.
            + 80 * shape.vocab
            + _tp_gather_bytes(shape, 1, tp_size)
        )
        peaks = [prefill, decode]
        if kind == "rubric":
            # Rubric judging follows generation with a no-cache scoring forward.
            peaks.append(
                _activation_bytes(shape, total_tokens)
                # One-position logits plus LM-head, cast, and reduction workspace.
                + 72 * shape.vocab
                + _tp_gather_bytes(shape, 1, tp_size)
            )
        return max(peaks)
    raise ValueError(f"Unknown workload kind: {kind!r}")


def _process_group(tp_mesh: Any):
    if tp_mesh is None:
        return None
    get_group = getattr(tp_mesh, "get_group", None)
    return get_group() if callable(get_group) else tp_mesh


def _tp_size(tp_mesh: Any) -> int:
    if tp_mesh is None:
        return 1
    size = getattr(tp_mesh, "size", None)
    if callable(size):
        return max(1, int(size()))
    group = _process_group(tp_mesh)
    if dist.is_available() and dist.is_initialized():
        return max(1, int(dist.get_world_size(group=group)))
    return 1


def _group_min(value: int, device: torch.device, tp_mesh: Any) -> int:
    group = _process_group(tp_mesh)
    if group is None or not dist.is_available() or not dist.is_initialized():
        return value
    result = torch.tensor(value, dtype=torch.int64, device=device)
    dist.all_reduce(result, op=dist.ReduceOp.MIN, group=group)
    return int(result.item())


def critical_batch_size(
    model: Any,
    *,
    kind: Workload,
    prompt_tokens: int,
    new_tokens: int = 0,
    multiplier: int = 1,
    limit: int | None = None,
    requested: int = 0,
    dtype: torch.dtype | str | None = None,
    use_cache: bool = False,
    tp_mesh: Any = None,
    safety: float = 0.80,
    reserve_fraction: float = 0.05,
    minimum_reserve_bytes: int = 512 * _MIB,
) -> BatchEstimate:
    """Compute a safe batch size without executing trial model batches.

    ``multiplier`` converts one logical item into effective model sequences; for
    example, context generation uses the number of samples requested per context.
    A positive ``requested`` value is treated as an upper bound, while zero means
    use the computed critical size.
    """
    prompt_tokens = int(prompt_tokens)
    new_tokens = int(new_tokens)
    multiplier = int(multiplier)
    requested = int(requested)
    if kind not in {"hidden", "forward", "cross_entropy", "generate", "rubric"}:
        raise ValueError(f"Unknown workload kind: {kind!r}")
    if prompt_tokens < 1 or new_tokens < 0:
        raise ValueError("prompt_tokens must be positive and new_tokens nonnegative")
    if multiplier < 1:
        raise ValueError("multiplier must be positive")
    if requested < 0:
        raise ValueError("requested batch size cannot be negative")
    if limit is not None and int(limit) < 1:
        raise ValueError("batch-size limit must be positive")
    if not 0 < safety <= 1 or not 0 <= reserve_fraction < 1:
        raise ValueError("safety must be in (0, 1] and reserve_fraction in [0, 1)")

    if not torch.cuda.is_available():
        fallback = requested or 1
        selected = min(fallback, int(limit)) if limit is not None else fallback
        return BatchEstimate(selected, selected, selected * multiplier, 0, 0, 0, 0)

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type != "cuda":
        fallback = requested or 1
        selected = min(fallback, int(limit)) if limit is not None else fallback
        return BatchEstimate(selected, selected, selected * multiplier, 0, 0, 0, 0)

    shape = _model_shape(model, dtype)
    total_tokens = prompt_tokens + (
        new_tokens if kind in {"generate", "rubric"} else 0
    )
    if shape.context_limit is not None and total_tokens > shape.context_limit:
        raise ValueError(
            f"Workload needs {total_tokens} tokens but the model context limit is "
            f"{shape.context_limit}."
        )

    if _attention_implementation(model) == "sdpa":
        attention_geometries = [(prompt_tokens, prompt_tokens)]
        if kind in {"generate", "rubric"}:
            attention_geometries.append((1, total_tokens))
        if kind == "rubric":
            attention_geometries.append((total_tokens, total_tokens))
        compute_dtype = _torch_dtype(model, dtype)
        if all(
            _sdpa_uses_linear_memory(
                shape, device, compute_dtype, query_tokens, key_tokens
            )
            for query_tokens, key_tokens in attention_geometries
        ):
            shape = replace(shape, fused_attention=True)

    per_sequence = _sequence_bytes(
        model,
        shape,
        kind=kind,
        prompt_tokens=prompt_tokens,
        new_tokens=new_tokens,
        use_cache=use_cache,
        tp_size=_tp_size(tp_mesh),
    )
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    reserve_bytes = max(
        int(minimum_reserve_bytes),
        math.ceil(int(total_bytes) * reserve_fraction),
    )
    usable_bytes = math.floor(max(0, int(free_bytes) - reserve_bytes) * safety)
    effective_capacity = usable_bytes // max(1, per_sequence)
    logical_capacity = effective_capacity // multiplier
    if limit is not None:
        logical_capacity = min(logical_capacity, int(limit))
    logical_capacity = _group_min(logical_capacity, device, tp_mesh)
    if logical_capacity < 1:
        needed = per_sequence * multiplier
        raise RuntimeError(
            "One logical batch item is estimated to need "
            f"{needed / _MIB:.1f} MiB, but only {usable_bytes / _MIB:.1f} MiB "
            "is available after GPU safety reserves."
        )

    selected = min(requested, logical_capacity) if requested else logical_capacity
    return BatchEstimate(
        batch_size=selected,
        critical_batch_size=logical_capacity,
        effective_batch_size=selected * multiplier,
        bytes_per_sequence=per_sequence,
        free_bytes=int(free_bytes),
        usable_bytes=usable_bytes,
        reserve_bytes=reserve_bytes,
    )


def factor_batch_size(
    capacity: int,
    *,
    first_limit: int,
    second_limit: int,
    first_hint: int = 1,
    second_hint: int = 1,
) -> tuple[int, int]:
    """Factor an effective capacity into two bounded, balanced batch dimensions."""
    capacity = int(capacity)
    first_limit = int(first_limit)
    second_limit = int(second_limit)
    if min(capacity, first_limit, second_limit, first_hint, second_hint) < 1:
        raise ValueError("capacity, limits, and hints must be positive")
    target_ratio = first_hint / second_hint
    best: tuple[int, float, int, int] | None = None
    for first in range(1, min(capacity, first_limit) + 1):
        second = min(second_limit, capacity // first)
        if second < 1:
            continue
        product = first * second
        ratio_error = abs(math.log((first / second) / target_ratio))
        candidate = (product, -ratio_error, first, second)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise RuntimeError("No valid batch-size factors fit the requested capacity")
    return best[2], best[3]
