import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import asyncio
import numpy as np
import torch
from monarch.actor import Actor, endpoint

from .utils import (
    read_jsonl_texts,
    load_model_and_tokenizer,
    model_slug,
    chunked,
)

@dataclass
class LogOddsConfig:
    dtype: str = "float32"
    seed: int = 42
    batch_size: int = 100
    max_length: int = 100
    top_k: int = -1
    progress_every: int = 10  # mailbox yield

class LogOddsActor(Actor):
    """Computes token-level log-odds (no steering) and saves top-k."""

    def __init__(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        self.current_model_name: Optional[str] = None
        self.current_dtype: Optional[str] = None
        self.tokenizer = None
        self.model = None

    def _ensure_model(self, model_name: str, dtype_str: str):
        if self.model is not None and self.current_model_name == model_name and self.current_dtype == dtype_str:
            return
        self.tokenizer = None
        self.model = None
        torch.cuda.empty_cache()
        self.tokenizer, self.model = load_model_and_tokenizer(model_name, dtype_str)
        self.current_model_name = model_name
        self.current_dtype = dtype_str

    @endpoint
    async def compute_log_odds(
        self,
        model_name: str,
        concept_slug: str,
        concept_label: str,
        prompts_dir: str,
        save_dir: str,
        cfg_dict: Optional[dict] = None,
        rank_hint: int = 0,
    ):
        cfg = LogOddsConfig(**(cfg_dict or {}))
        torch.manual_seed(cfg.seed + int(rank_hint))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed + int(rank_hint))

        self._ensure_model(model_name, cfg.dtype)
        tokenizer, model = self.tokenizer, self.model
        model.eval()

        prompts_root = Path(prompts_dir)
        pos_path = prompts_root / f"{concept_slug}_positive.jsonl"
        neg_path = prompts_root / f"{concept_slug}_{model_slug(model_name)}_negative.jsonl"

        concept_prompts = read_jsonl_texts(pos_path)
        negative_prompts = read_jsonl_texts(neg_path)
        missing: List[str] = []
        if not concept_prompts:
            missing.append(str(pos_path))
        if not negative_prompts:
            missing.append(str(neg_path))
        if missing:
            return {"error": f"Missing or empty prompt files for '{concept_slug}': {', '.join(missing)}"}

        device = next(model.parameters()).device
        save_root = Path(save_dir) / model_slug(model_name) / concept_slug
        save_root.mkdir(parents=True, exist_ok=True)
        progress_mod = max(1, int(cfg.progress_every))

        async def accumulate_log_probs(texts: List[str]) -> Tuple[torch.Tensor, int]:
            sum_log_probs = None
            total = 0
            for step, batch in enumerate(chunked(texts, int(cfg.batch_size))):
                enc = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(cfg.max_length),
                )
                input_ids = enc["input_ids"].to(device, non_blocking=True)
                attn_mask = enc["attention_mask"].to(device, non_blocking=True)
                last_idx = torch.clamp(attn_mask.sum(dim=1) - 1, min=0)

                with torch.no_grad():
                    logits = model(input_ids=input_ids, attention_mask=attn_mask).logits  # [B,T,V]
                row = torch.arange(logits.shape[0], device=logits.device)
                last_logits = logits[row, last_idx, :].to(torch.float32)
                log_probs = torch.log_softmax(last_logits, dim=-1).to(torch.float64)

                if sum_log_probs is None:
                    sum_log_probs = torch.zeros_like(log_probs[0], dtype=torch.float64)
                sum_log_probs += log_probs.sum(dim=0)
                total += log_probs.shape[0]

                if step % progress_mod == 0:
                    await asyncio.sleep(0)
            return sum_log_probs, total

        concept_sum, concept_count = await accumulate_log_probs(concept_prompts)
        nonconcept_sum, nonconcept_count = await accumulate_log_probs(negative_prompts)
        if concept_count == 0 or nonconcept_count == 0:
            return {"error": "Empty prompts after tokenization"}

        log_odds = (concept_sum / concept_count) - (nonconcept_sum / nonconcept_count)
        vocab_size = log_odds.numel()
        if int(cfg.top_k) == -1:
            k = vocab_size
        else:
            k = min(int(cfg.top_k), vocab_size)
        top_vals, top_ids = torch.topk(log_odds, k=k)
        token_strs = [tokenizer.decode([int(t)]) for t in top_ids.tolist()]

        out_path = save_root / "log_odds_topk.npz"
        meta = {
            "model": model_name,
            "concept": concept_label,
            "concept_slug": concept_slug,
            "prompts_dir": str(prompts_root),
            "positive_file": str(pos_path),
            "negative_file": str(neg_path),
            "concept_prompt_count": int(concept_count),
            "negative_prompt_count": int(nonconcept_count),
            "top_k": int(cfg.top_k),  # -1 means full vocab saved
            "saved_tokens": int(k),
            "max_length": int(cfg.max_length),
            "batch_size": int(cfg.batch_size),
            "dtype": cfg.dtype,
            "baseline_alpha": 0.0,
        }
        np.savez_compressed(
            out_path,
            token_ids=top_ids.to(torch.int32).cpu().numpy(),
            token_strs=np.array(token_strs, dtype=object),
            log_odds=top_vals.to(torch.float32).cpu().numpy(),
            meta=json.dumps(meta),
        )

        torch.cuda.empty_cache()
        return {"ok": True, "saved": str(out_path)}
