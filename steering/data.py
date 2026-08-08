from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
from datasets import load_dataset

from .naming import model_slug


def discover_concepts(prompt_dir: Path) -> list[tuple[str, str]]:
    """Discover concepts that have both positive and negative prompt files."""
    positive = {
        path.name[: -len("_positive.jsonl")]
        for path in prompt_dir.glob("*_positive.jsonl")
    }
    negative = set()
    for path in prompt_dir.glob("*_negative.jsonl"):
        stem = path.name[: -len("_negative.jsonl")]
        negative.add(stem.rsplit("_", 1)[0])

    concepts: list[tuple[str, str]] = []
    for slug in sorted(positive & negative):
        label = None
        probe = prompt_dir / f"{slug}_positive.jsonl"
        try:
            with probe.open("r", encoding="utf-8") as handle:
                first = next(handle, None)
            row = json.loads(first) if first else None
            if isinstance(row, dict) and isinstance(row.get("concept"), str):
                label = row["concept"].strip()
        except (OSError, ValueError, TypeError):
            pass
        concepts.append((slug, label or slug.replace("_", " ")))
    return concepts


def discover_steering_jobs(
    steer_dir: Path, models: list[str]
) -> list[tuple[str, str, str]]:
    """Return ``(model_name, concept_slug, concept_label)`` jobs."""
    jobs: list[tuple[str, str, str]] = []
    for model_name in models:
        base = steer_dir / model_slug(model_name)
        if not base.exists():
            continue
        for concept_dir in sorted(path for path in base.iterdir() if path.is_dir()):
            if any(concept_dir.glob("layer_*.pt")):
                jobs.append(
                    (model_name, concept_dir.name, concept_dir.name.replace("_", " "))
                )
    return jobs


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def read_jsonl_texts(
    path: Path, n_prompts: Optional[int] = None, text_key: str = "text"
) -> list[str]:
    texts: list[str] = []
    if not path.exists():
        return texts
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if n_prompts is not None and index >= n_prompts:
                break
            try:
                text = json.loads(line).get(text_key, "")
            except (ValueError, AttributeError):
                continue
            if isinstance(text, str):
                texts.append(text)
    return texts


def load_contexts_for_concept(
    contexts_file: str, concept_slug: str, concept_label: str
) -> tuple[list[str], list[int]]:
    """Load shared negatives followed by concept-specific positive contexts."""
    path = Path(contexts_file)
    if path.suffix != ".jsonl":
        contexts = read_lines(path)
        return contexts, [-1] * len(contexts)

    negatives: list[str] = []
    negative_lines: list[int] = []
    positives: list[str] = []
    positive_lines: list[int] = []
    concept_keys = {key for key in (concept_slug, concept_label) if key}

    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            values = row.get("negative")
            if isinstance(values, list):
                negatives.extend(values)
                negative_lines.extend([line_index] * len(values))
            for key in concept_keys:
                values = row.get(key)
                if isinstance(values, list):
                    positives.extend(values)
                    positive_lines.extend([line_index] * len(values))

    if not positives or not negatives:
        raise ValueError("positive or negative contexts are empty")
    return negatives + positives, negative_lines + positive_lines


def count_negative_prompts(contexts_file: str) -> Optional[int]:
    path = Path(contexts_file)
    if path.suffix != ".jsonl":
        return None
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                values = json.loads(line).get("negative")
            except (ValueError, AttributeError):
                continue
            if isinstance(values, list):
                count += len(values)
    return count


def _eos_token_id(tokenizer) -> Optional[int]:
    for attr in ("eos_token_id", "sep_token_id", "pad_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if isinstance(token_id, int):
            return token_id
    return None


def iter_eval_blocks_from_parquet(
    tokenizer,
    parquet_path: str,
    cfg: Any,
    batch_size: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield fixed-size ``(input_ids, labels)`` CPU batches from parquet text."""
    required_tokens = int(cfg.eval_seq_len) + 1
    stride = int(cfg.eval_stride) if cfg.eval_stride else int(cfg.eval_seq_len)
    eos_id = _eos_token_id(tokenizer)
    max_blocks = (
        int(cfg.eval_max_blocks) if getattr(cfg, "eval_max_blocks", None) else None
    )
    if max_blocks is not None and max_blocks <= 0:
        max_blocks = None

    read_rows = 2048
    tokenize_batch_size = 64
    buffer: list[list[int]] = []
    emitted = 0

    def flush() -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        nonlocal buffer
        if not buffer:
            return None
        batch = torch.tensor(buffer, dtype=torch.long)
        if torch.cuda.is_available():
            try:
                batch = batch.pin_memory()
            except Exception:
                pass
        buffer = []
        return batch[:, :-1], batch[:, 1:]

    def from_token_ids(token_ids: list[int]):
        nonlocal emitted
        if cfg.add_eos_between_docs and eos_id is not None:
            token_ids = token_ids + [eos_id]
        if len(token_ids) < required_tokens:
            return
        for start in range(0, len(token_ids) - required_tokens + 1, stride):
            buffer.append(token_ids[start : start + required_tokens])
            emitted += 1
            if len(buffer) >= batch_size:
                output = flush()
                if output is not None:
                    yield output
            if max_blocks is not None and emitted >= max_blocks:
                break

    def from_texts(texts: list[str]):
        usable = [text for text in texts if isinstance(text, str) and text.strip()]
        if not usable:
            return
        encoded = tokenizer(
            usable,
            add_special_tokens=False,
            truncation=bool(getattr(cfg, "max_doc_tokens", None)),
            max_length=(
                int(cfg.max_doc_tokens)
                if getattr(cfg, "max_doc_tokens", None)
                else None
            ),
            return_attention_mask=False,
        )
        for token_ids in encoded.get("input_ids", []):
            yield from from_token_ids(token_ids)
            if max_blocks is not None and emitted >= max_blocks:
                break

    path = Path(parquet_path)
    used_pyarrow = False
    if path.exists() and path.suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(str(path))
            used_pyarrow = True
            for row_batch in parquet.iter_batches(
                batch_size=read_rows, columns=[cfg.text_field]
            ):
                texts = row_batch.column(0).to_pylist()
                for start in range(0, len(texts), tokenize_batch_size):
                    yield from from_texts(texts[start : start + tokenize_batch_size])
                    if max_blocks is not None and emitted >= max_blocks:
                        break
                if max_blocks is not None and emitted >= max_blocks:
                    break
        except Exception:
            used_pyarrow = False

    if not used_pyarrow:
        dataset = load_dataset(
            "parquet", data_files=str(parquet_path), split="train", streaming=True
        )
        text_buffer: list[str] = []
        for sample in dataset:
            text = sample.get(cfg.text_field)
            if isinstance(text, str) and text.strip():
                text_buffer.append(text)
            if len(text_buffer) >= tokenize_batch_size:
                yield from from_texts(text_buffer)
                text_buffer = []
            if max_blocks is not None and emitted >= max_blocks:
                break
        if text_buffer and (max_blocks is None or emitted < max_blocks):
            yield from from_texts(text_buffer)

    output = flush()
    if output is not None:
        yield output
