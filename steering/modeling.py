from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .naming import model_slug


def find_block_list(
    model: nn.Module, override_path: Optional[str] = None
) -> nn.ModuleList:
    """Locate a causal LM's transformer block list."""
    if override_path:
        obj = model
        for attr in override_path.split("."):
            if not hasattr(obj, attr):
                raise ValueError(f"layer_path '{override_path}' not found at '{attr}'")
            obj = getattr(obj, attr)
        if not isinstance(obj, nn.ModuleList):
            raise ValueError(f"layer_path '{override_path}' is not a ModuleList")
        return obj

    candidates = [
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "h"),
        ("transformer", "layers"),
        ("gpt_neox", "layers"),
        ("model", "encoder", "layers"),
        ("model", "language_model", "layers"),
    ]
    for path in candidates:
        obj = model
        for attr in path:
            if not hasattr(obj, attr):
                break
            obj = getattr(obj, attr)
        else:
            if isinstance(obj, nn.ModuleList):
                return obj

    for name in ("layers", "h", "blocks", "block"):
        value = getattr(model, name, None)
        if isinstance(value, nn.ModuleList):
            return value

    raise ValueError("Could not locate transformer block ModuleList; provide --layer_path.")


def load_steering_vector(
    steer_dir: Path,
    model_name: str,
    concept_slug: str,
    layer_idx: int,
) -> torch.Tensor:
    path = steer_dir / model_slug(model_name) / concept_slug / f"layer_{layer_idx}.pt"
    payload = torch.load(path, map_location="cpu")
    vector = payload["steering_vector"]
    if not isinstance(vector, torch.Tensor):
        vector = torch.tensor(vector, dtype=torch.float32)
    return vector


def set_left_padding(tokenizer) -> None:
    """Configure decoder-only batched generation for left padding/truncation."""
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"


def ensure_pad_token(tokenizer, model=None) -> None:
    """Ensure a tokenizer has a padding token without resizing when possible."""
    if tokenizer.pad_token_id is not None:
        return
    if getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return
    if getattr(tokenizer, "bos_token", None) is not None:
        tokenizer.pad_token = tokenizer.bos_token
        return
    if hasattr(tokenizer, "add_special_tokens"):
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        if model is not None and hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(len(tokenizer))
        return
    raise ValueError("Tokenizer has no pad/eos/bos token and cannot add special tokens.")


def maybe_apply_chat_template(
    tokenizer, system: str, user: str, use_chat: bool
) -> str:
    if use_chat and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"{system}\n\n{user}".strip()


def one_token_ids(tokenizer, variants: list[str]) -> list[int]:
    ids: list[int] = []
    for variant in variants:
        try:
            encoded = tokenizer.encode(variant, add_special_tokens=False)
        except Exception:
            continue
        if isinstance(encoded, list) and len(encoded) == 1:
            ids.append(int(encoded[0]))
    return sorted(set(ids))
