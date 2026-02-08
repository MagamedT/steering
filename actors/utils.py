from typing import Any, Iterable, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING
import json
import re
from functools import partialmethod
from pathlib import Path
import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

if TYPE_CHECKING:
    from .cross_entropy_actor import CrossEntropyPlotConfig



def discover_concepts(in_dir: Path):
    # Slugs for which we have positive/related data
    related = {
        p.name[:-len("_positive.jsonl")]
        for p in in_dir.glob("*_positive.jsonl")
    }

    # Slugs for which we have at least one negative/unrelated file
    # Filenames look like: "<slug>_<model_slug>_negative.jsonl"
    unrelated = set()
    for p in in_dir.glob("*_negative.jsonl"):
        # Strip the suffix
        base = p.name[:-len("_negative.jsonl")]
        # Remove the trailing model slug to recover concept slug.
        concept_slug = base.rsplit("_", 1)[0]
        unrelated.add(concept_slug)

    # Only keep concepts that have both related and unrelated data
    slugs = sorted(related & unrelated)
    concepts = []
    for s in slugs:
        label = None
        probe = in_dir / f"{s}_positive.jsonl"
        if probe.exists():
            try:
                with open(probe, "r", encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        if (
                            isinstance(row, dict)
                            and "concept" in row
                            and isinstance(row["concept"], str)
                        ):
                            label = row["concept"].strip()
                        break
            except Exception:
                pass
        if not label:
            label = s.replace("_", " ")
        concepts.append((s, label))
    return concepts

def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s.strip().lower()).strip("-") or "concept"

def model_slug(name: str) -> str:
    return slugify(name.replace("/", "-"))

def read_lines(path: Path) -> List[str]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(s)
    return rows


def read_jsonl_texts(path: Path, n_prompts=None, text_key="text"):
    out = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if n_prompts is not None and i >= n_prompts:
                break
            try:
                row = json.loads(line)
                txt = row.get(text_key, "")
                if isinstance(txt, str):           # keep empty strings too
                    out.append(txt)
            except Exception:
                continue
    return out


def load_contexts_for_concept(contexts_file: str, concept_slug: str, concept_label: str):
    """
    contexts_file is a JSONL with:
      - one line like: {"negative": ["neg1", "neg2", ...]}
      - one line per concept, e.g.: {"depression": ["p1", "p2", ...]}

    Negative prompts are shared across all concepts.
    Positive prompts are looked up by concept key.

    Returns:
        contexts:           list[str]  (negatives first, then positives)
        source_line_indices:list[int]  same length, each is the JSONL line index (0-based)
    """
    path = Path(contexts_file)
    contexts: List[str] = []
    source_line_indices: List[int] = []

    # Fallback: non-JSONL = old behavior (txt file, one prompt per line)
    if path.suffix != ".jsonl":
        from .utils import read_lines
        contexts = read_lines(path)
        # -1 = "unknown source line"
        source_line_indices = [-1] * len(contexts)
        return contexts, source_line_indices

    negatives: List[str] = []
    negatives_src: List[int] = []
    positives: List[str] = []
    positives_src: List[int] = []

    # dedupe keys
    concept_keys = list({k for k in (concept_slug, concept_label) if k})

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # Shared negatives (one line, but we'll accept multiple)
            if "negative" in obj and isinstance(obj["negative"], list):
                for prompt in obj["negative"]:
                    negatives.append(prompt)
                    negatives_src.append(line_idx)

            # Concept-specific positives (by slug or label)
            for key in concept_keys:
                if key and key in obj and isinstance(obj[key], list):
                    for prompt in obj[key]:
                        positives.append(prompt)
                        positives_src.append(line_idx)
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError('positive or negatives is empty')
    
    contexts = negatives + positives
    source_line_indices = negatives_src + positives_src
    return contexts, source_line_indices


def _from_pretrained_with_dtype(cls, model_name: str, *, dtype, **kwargs):
    try:
        return cls.from_pretrained(model_name, dtype=dtype, **kwargs)
    except TypeError:
        return cls.from_pretrained(model_name, torch_dtype=dtype, **kwargs)

def find_block_list(model: nn.Module, override_path: Optional[str] = None) -> nn.ModuleList:
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
        ("model", "layers"),                 # LLaMA/Mistral/Qwen
        ("model", "decoder", "layers"),
        ("transformer", "h"),                # GPT-2/OPT
        ("transformer", "layers"),
        ("gpt_neox", "layers"),
        ("model", "encoder", "layers"),
        ("model", "language_model", "layers") # for multimodal gemma 3 (parameter count >= 4B) 
    ]
    for path in candidates:
        obj = model
        ok = True
        for attr in path:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and isinstance(obj, nn.ModuleList):
            return obj

    for name in ("layers", "h", "blocks", "block"):
        if hasattr(model, name) and isinstance(getattr(model, name), nn.ModuleList):
            return getattr(model, name)

    raise ValueError("Could not locate transformer block ModuleList; provide --layer_path.")


def load_model_and_tokenizer(model_name: str, dtype_str: str = "float32") -> Tuple[AutoTokenizer, nn.Module]:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.bos_token

    dtype = torch.float32 if dtype_str == "float32" else torch.bfloat16
    common = dict(low_cpu_mem_usage=True, device_map={"": 0})
    try:
        model = _from_pretrained_with_dtype(
            AutoModelForCausalLM, model_name, dtype=dtype,
            attn_implementation="flash_attention_2", **common
        )
    except Exception:
        model = _from_pretrained_with_dtype(
            AutoModelForCausalLM, model_name, dtype=dtype,
            attn_implementation="sdpa", **common
        )
    model.eval()
    model.generation_config.pad_token_id = tok.pad_token_id
    if tok.eos_token_id is not None:
        model.generation_config.eos_token_id = tok.eos_token_id
    return tok, model

def load_steer_vector(steer_dir: Path, model_name: str, concept_slug: str, layer_idx: int) -> torch.Tensor:
    mslug = model_slug(model_name)
    path = steer_dir / mslug / concept_slug / f"layer_{layer_idx}.pt"
    data = torch.load(path, map_location="cpu")
    vec = data["steering_vector"]  # [H], float32
    if not isinstance(vec, torch.Tensor):
        vec = torch.tensor(vec, dtype=torch.float32)
    return vec

def chunked(seq, n):
    # generator which yield batches of size n from a sequence
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def chunked_with_bounds(xs: List[Any], n: int) -> Iterable[Tuple[int, int, List[Any]]]:
    """Yield (start_idx, end_idx, batch) chunks."""
    if n <= 0:
        n = len(xs) if xs else 1
    for i in range(0, len(xs), n):
        j = min(len(xs), i + n)
        yield i, j, xs[i:j]


def count_negative_prompts(contexts_file: str) -> Optional[int]:
    """For JSONL contexts_file, count prompts under any 'negative' list(s)."""
    path = Path(contexts_file)
    if path.suffix != ".jsonl":
        return None
    n_neg = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("negative"), list):
                n_neg += len(obj["negative"])
    return n_neg


def set_left_padding(tok) -> None:
    """Make decoder-only batching safer: left padding + (usually) left truncation."""
    try:
        tok.padding_side = "left"
    except Exception:
        pass
    try:
        tok.truncation_side = "left"
    except Exception:
        pass


def ensure_pad_token(tok, model=None) -> None:
    """Ensure tokenizer has a PAD token; prefer EOS/BOS reuse to avoid resize."""
    if tok.pad_token_id is not None:
        return

    if getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
        return
    if getattr(tok, "bos_token", None) is not None:
        tok.pad_token = tok.bos_token
        return

    if hasattr(tok, "add_special_tokens"):
        tok.add_special_tokens({"pad_token": "[PAD]"})
        if model is not None and hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(len(tok))
        return

    raise ValueError("Tokenizer has no pad/eos/bos token and cannot add special tokens.")


def maybe_apply_chat_template(tok, system: str, user: str, use_chat: bool) -> str:
    """Apply tokenizer chat template when available, else return plain-text fallback."""
    has_chat = bool(getattr(tok, "chat_template", None))
    if use_chat and has_chat and hasattr(tok, "apply_chat_template"):
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{user}".strip()


def one_token_ids(tok, variants: List[str]) -> List[int]:
    """Return unique token ids for string variants that tokenize to exactly one token."""
    ids: List[int] = []
    for v in variants:
        try:
            enc = tok.encode(v, add_special_tokens=False)
        except Exception:
            continue
        if isinstance(enc, list) and len(enc) == 1:
            ids.append(int(enc[0]))
    return sorted(set(ids))


def patch_tqdm_disable() -> None:
    """Disable tqdm globally (useful for libraries that create noisy progress bars)."""
    try:
        from tqdm import tqdm as _tqdm
        _tqdm.__init__ = partialmethod(_tqdm.__init__, disable=True)
    except Exception:
        pass


def resolve_tasks(task_names: Optional[Sequence[str]], task_enum):
    """Resolve user task names into entries of the provided task enum."""
    if task_names is None or len(task_names) == 0:
        return list(task_enum)
    if len(task_names) == 1 and str(task_names[0]).lower() == "all":
        return list(task_enum)
    resolved = []
    for name in task_names:
        key = str(name).upper().replace("-", "_").replace(" ", "_")
        if not hasattr(task_enum, key):
            raise ValueError(f"MMLUTask '{name}' not found (resolved key='{key}')")
        resolved.append(getattr(task_enum, key))
    return resolved


def extract_choice_from_continuation(text: str) -> Optional[str]:
    """Extract A/B/C/D from a generated continuation string."""
    if not text:
        return None
    s = text.strip()
    if s and s[0].upper() in ("A", "B", "C", "D"):
        return s[0].upper()
    m = re.search(r"\b([A-D])\b", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-D])\s*[\).]", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def task_scores_to_dict(task_scores: Any) -> Optional[dict[str, float]]:
    """Normalize task scores into a plain {task_id: score} dict."""
    if task_scores is None:
        return None
    if isinstance(task_scores, dict):
        out: dict[str, float] = {}
        for k, v in task_scores.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                pass
        return out or None

    try:
        import pandas as pd

        if isinstance(task_scores, pd.Series):
            return {str(k): float(v) for k, v in task_scores.to_dict().items()}

        if isinstance(task_scores, pd.DataFrame):
            df = task_scores
            if "Score" in df.columns and df.index is not None:
                return {str(k): float(v) for k, v in df["Score"].to_dict().items()}
            if "Task" in df.columns and "Score" in df.columns:
                s = df.set_index("Task")["Score"]
                return {str(k): float(v) for k, v in s.to_dict().items()}
            if df.shape[1] == 1:
                s = df.iloc[:, 0]
                return {str(k): float(v) for k, v in s.to_dict().items()}
    except Exception:
        pass

    return None


### utils for cross entropy computation 

def _get_eos_id(tokenizer) -> Optional[int]:
    for attr in ("eos_token_id", "sep_token_id", "pad_token_id"):
        tid = getattr(tokenizer, attr, None)
        if isinstance(tid, int):
            return tid
    return None



def iter_eval_blocks_from_parquet(
    tokenizer,
    parquet_path: str,
    cfg: "CrossEntropyPlotConfig",
    batch_size: int,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:

    """Yield (input_ids, labels) CPU tensors of shape [B, T] from a parquet file.

    Behavior:
      - If cfg.eval_max_blocks is 0/None, we iterate the *full parquet* (can be very large).
      - Uses a fast local-file path via `pyarrow` + batched tokenization.
      - Falls back to HF `datasets` streaming when pyarrow isn't available/usable.

    This function intentionally does *not* pad: it yields fixed-size contiguous blocks
    from each document with stride `cfg.eval_stride`.
    """

    required_amount = int(cfg.eval_seq_len) + 1
    stride = int(cfg.eval_stride) if cfg.eval_stride else int(cfg.eval_seq_len)
    eos_id = _get_eos_id(tokenizer)

    # 0/None => full parquet scan
    max_blocks = int(cfg.eval_max_blocks) if getattr(cfg, "eval_max_blocks", None) else None
    if max_blocks is not None and max_blocks <= 0:
        max_blocks = None

    # Tuning knobs (safe defaults). Kept local to avoid changing your config/CLI.
    read_rows = 2048          # parquet rows per IO batch
    tokenize_batch_size = 64  # texts per tokenizer call

    buffer: List[List[int]] = []
    emitted = 0

    def _flush_buffer() -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        nonlocal buffer
        if not buffer:
            return None
        batch = torch.tensor(buffer, dtype=torch.long)
        # pin memory so .to(..., non_blocking=True) can actually be non-blocking
        if torch.cuda.is_available():
            try:
                batch = batch.pin_memory()
            except Exception:
                pass
        buffer = []
        return batch[:, :-1], batch[:, 1:]

    def _yield_from_token_ids(token_ids: List[int]):
        nonlocal emitted, buffer
        if not token_ids:
            return
        if cfg.add_eos_between_docs and eos_id is not None:
            token_ids = token_ids + [int(eos_id)]
        if len(token_ids) < required_amount:
            return

        for i in range(0, len(token_ids) - required_amount + 1, stride):
            buffer.append(token_ids[i : i + required_amount])
            emitted += 1

            if len(buffer) >= batch_size:
                out = _flush_buffer()
                if out is not None:
                    yield out

            if max_blocks is not None and emitted >= max_blocks:
                break

    def _yield_from_texts(texts: List[str]):
        # Filter empties to avoid wasted tokenizer work.
        good = [t for t in texts if isinstance(t, str) and t.strip()]
        if not good:
            return

        enc = tokenizer(
            good,
            add_special_tokens=False,
            truncation=bool(getattr(cfg, "max_doc_tokens", None)),
            max_length=int(cfg.max_doc_tokens) if getattr(cfg, "max_doc_tokens", None) else None,
            return_attention_mask=False,
        )
        for token_ids in enc.get("input_ids", []):
            for out in _yield_from_token_ids(token_ids):
                yield out
            if max_blocks is not None and emitted >= max_blocks:
                break

    # Fast path: local parquet via pyarrow row batches
    path = Path(parquet_path)
    used_pyarrow = False
    if path.exists() and path.suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq  # type: ignore

            pf = pq.ParquetFile(str(path))
            used_pyarrow = True

            for rb in pf.iter_batches(batch_size=int(read_rows), columns=[cfg.text_field]):
                texts = rb.column(0).to_pylist()

                # Tokenize in smaller chunks to bound peak memory.
                for j in range(0, len(texts), int(tokenize_batch_size)):
                    chunk = texts[j : j + int(tokenize_batch_size)]
                    for out in _yield_from_texts(chunk):
                        yield out
                    if max_blocks is not None and emitted >= max_blocks:
                        break

                if max_blocks is not None and emitted >= max_blocks:
                    break

        except Exception:
            used_pyarrow = False

    # Fallback: HF datasets streaming
    if not used_pyarrow:
        dataset = load_dataset("parquet", data_files=str(parquet_path), split="train", streaming=True)
        text_buf: List[str] = []
        for sample in dataset:
            text = sample.get(cfg.text_field, None)
            if isinstance(text, str) and text.strip():
                text_buf.append(text)

            if len(text_buf) >= int(tokenize_batch_size):
                for out in _yield_from_texts(text_buf):
                    yield out
                text_buf = []
                if max_blocks is not None and emitted >= max_blocks:
                    break

            if max_blocks is not None and emitted >= max_blocks:
                break

        if text_buf and (max_blocks is None or emitted < max_blocks):
            for out in _yield_from_texts(text_buf):
                yield out

    out = _flush_buffer()
    if out is not None:
        yield out

#############################################

class TempFp32LayerWrapper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        storage_dtype: torch.dtype = torch.bfloat16,
        compute_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.module = module
        self.storage_dtype = storage_dtype
        self.compute_dtype = compute_dtype

        # Ensure params are stored in storage_dtype initially
        with torch.no_grad():
            for p in self.module.parameters():
                p.data = p.data.to(storage_dtype)
                p.requires_grad = False

    # NEW: attribute forwarding so things like .attention_type still work
    def __getattr__(self, name: str):
        # First let nn.Module handle parameters/submodules/etc.
        try:
            return super().__getattr__(name)
        except AttributeError:
            # If it's not on the wrapper, forward to the inner module
            try:
                return getattr(self.module, name)
            except AttributeError:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object and its inner module "
                    f"have no attribute '{name}'"
                )

    def _cast_tensors(self, x: Any, dtype: torch.dtype):
        if torch.is_tensor(x):
            return x.to(dtype)
        if isinstance(x, (list, tuple)):
            return type(x)(self._cast_tensors(v, dtype) for v in x)
        if isinstance(x, dict):
            return {k: self._cast_tensors(v, dtype) for k, v in x.items()}
        return x

    def forward(self, *args, **kwargs):
        # 1) upcast params to compute_dtype
        with torch.no_grad():
            for p in self.module.parameters():
                if p.dtype is not self.compute_dtype:
                    p.data = p.data.to(self.compute_dtype)

        # 2) upcast inputs
        args_fp32 = self._cast_tensors(args, self.compute_dtype)
        kwargs_fp32 = self._cast_tensors(kwargs, self.compute_dtype)

        # 3) run the real block in fp32
        out = self.module(*args_fp32, **kwargs_fp32)

        # 4) downcast outputs back to storage_dtype
        out = self._cast_tensors(out, self.storage_dtype)

        # 5) restore params to storage_dtype
        with torch.no_grad():
            for p in self.module.parameters():
                if p.dtype is not self.storage_dtype:
                    p.data = p.data.to(self.storage_dtype)

        return out

def wrap_blocks_with_temp_fp32(
    model: nn.Module,
    layer_indices=None,
    *,
    override_path: Optional[str] = None,
    storage_dtype: torch.dtype = torch.bfloat16,
    compute_dtype: torch.dtype = torch.float32,
) -> nn.Module:
    blocks = find_block_list(model, override_path=override_path)
    n_layers = len(blocks)

    if layer_indices is None:
        idxs = list(range(n_layers))
    else:
        idxs = []
        for i in layer_indices:
            if i < 0:
                i = n_layers + i
            if not (0 <= i < n_layers):
                raise IndexError(f"layer index {i} out of range [0, {n_layers-1}]")
            idxs.append(i)

    for i in idxs:
        blocks[i] = TempFp32LayerWrapper(
            blocks[i],
            storage_dtype=storage_dtype,
            compute_dtype=compute_dtype,
        )

    for p in model.parameters():
        p.requires_grad_(False)

    return model
