"""Provide small helpers for MMLU task selection and scoring."""

from __future__ import annotations

from functools import partialmethod
import re
from typing import Any, Optional, Sequence


def disable_tqdm() -> None:
    """Disable progress bars when tqdm is available."""
    try:
        from tqdm import tqdm

        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    except Exception:
        pass


def resolve_tasks(task_names: Optional[Sequence[str]], task_enum):
    """Resolve requested task names to MMLU enum values."""
    if not task_names or (
        len(task_names) == 1 and str(task_names[0]).lower() == "all"
    ):
        return list(task_enum)
    resolved = []
    for name in task_names:
        key = str(name).upper().replace("-", "_").replace(" ", "_")
        if not hasattr(task_enum, key):
            raise ValueError(f"MMLUTask '{name}' not found (resolved key='{key}')")
        resolved.append(getattr(task_enum, key))
    return resolved


def extract_choice(text: str) -> Optional[str]:
    """Extract the first multiple-choice letter from text."""
    if not text:
        return None
    value = text.strip()
    if value and value[0].upper() in ("A", "B", "C", "D"):
        return value[0].upper()
    match = re.search(r"\b([A-D])\b", value, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-D])\s*[\).]", value, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def task_scores_to_dict(task_scores: Any) -> Optional[dict[str, float]]:
    """Convert supported score tables into a plain dictionary."""
    if task_scores is None:
        return None
    if isinstance(task_scores, dict):
        output: dict[str, float] = {}
        for key, value in task_scores.items():
            try:
                output[str(key)] = float(value)
            except Exception:
                pass
        return output or None

    try:
        import pandas as pd

        if isinstance(task_scores, pd.Series):
            return {str(key): float(value) for key, value in task_scores.items()}
        if isinstance(task_scores, pd.DataFrame):
            if "Score" in task_scores.columns and task_scores.index is not None:
                return {
                    str(key): float(value)
                    for key, value in task_scores["Score"].items()
                }
            if "Task" in task_scores.columns and "Score" in task_scores.columns:
                series = task_scores.set_index("Task")["Score"]
                return {str(key): float(value) for key, value in series.items()}
            if task_scores.shape[1] == 1:
                series = task_scores.iloc[:, 0]
                return {str(key): float(value) for key, value in series.items()}
    except Exception:
        pass
    return None
