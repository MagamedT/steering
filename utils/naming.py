"""Create stable names for concepts and models."""

from __future__ import annotations

import re


def slugify(value: str) -> str:
    """Return a stable filesystem-safe slug."""
    return re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower()).strip("-") or "concept"


def model_slug(model_name: str) -> str:
    """Create a stable directory name for a model."""
    return slugify(model_name.replace("/", "-"))
