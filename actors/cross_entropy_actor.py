import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import asyncio

import numpy as np
import torch
import torch.nn.functional as F
from monarch.actor import Actor, endpoint

from .utils import find_block_list, model_slug, load_model_and_tokenizer, load_steer_vector, iter_eval_blocks_from_parquet


@dataclass
class CrossEntropyPlotConfig:
    # compute / reproducibility
    dtype: str = "float32"
    seed: int = 42

    # α grid
    alpha_start: float = -100.0
    alpha_end: float = 100.0
    alpha_steps: int = 128
    alpha_batch_size: int = 16  # number of α values per forward

    # evaluation tokenization / batching
    eval_seq_len: int = 128        # input length T; we score T next-token predictions using chunks of length T+1
    eval_stride: int = 128         # stride when chunking within a document
    eval_max_blocks: int = 8192     # how many (T+1)-token chunks to score
    eval_batch_size: int = 16       # how many chunks per forward (before α replication)
    # eval_seq_len * eval_max_blocks next-token prediction will be evaluated

    text_field: str = "text"
    max_doc_tokens: int = 4096     # per-doc token cap before chunking (to bound tokenizer time)
    add_eos_between_docs: bool = True

    # steering application
    apply_last_token_only: bool = False

    # cooperative scheduling
    progress_every: int = 10       # mailbox yield every N forwards

class CrossEntropyActor(Actor):
    """
    One actor per GPU. Caches last loaded (model, dtype).
    Exposes endpoint to compute cross-entropy-vs-alpha curves for many layers.
    """
    def __init__(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        self.current_model_name = None
        self.current_dtype = None
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
    async def compute_cross_entropy_curves(
        self,
        model_name: str,
        concept_slug: str,
        concept_label: str,
        block_idx_to_steer,     # int or None for all
        eval_parquet: str,      # local parquet path
        steer_dir: str,         # root where steering vectors live
        save_dir: str,          # root where .npz curve files go
        layer_path: Optional[str] = None,
        cfg_dict: Optional[dict] = None,
        rank_hint: int = 0,
    ):
        cfg = CrossEntropyPlotConfig(**(cfg_dict or {}))
        torch.manual_seed(cfg.seed + int(rank_hint))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed + int(rank_hint))

        self._ensure_model(model_name, cfg.dtype)
        tokenizer, model = self.tokenizer, self.model
        model.eval()

        blocks = find_block_list(model, override_path=layer_path)
        n_blocks = len(blocks)
        if block_idx_to_steer == None:
            block_idx_to_steer = list(range(n_blocks))
        else:
            # Interpret integer input as "sample this many layers uniformly across depth".
            block_idx_to_steer = np.linspace(0, n_blocks-1, num = block_idx_to_steer, dtype = int).tolist()

        # α grid (ensure 0 present)
        alphas = torch.linspace(cfg.alpha_start, cfg.alpha_end, steps=cfg.alpha_steps, dtype=torch.float32)
        if not (alphas == 0.0).any():
            alphas = torch.sort(torch.cat([alphas, torch.tensor([0.0], dtype=alphas.dtype)]))[0]
        alpha_amount = int(alphas.numel())
        alpha_batch_size = int(cfg.alpha_batch_size)

        save_root = Path(save_dir) / model_slug(model_name) / concept_slug
        save_root.mkdir(parents=True, exist_ok=True)

        steer_dir_path = Path(steer_dir)
        forward_counter = 0
        progress_mod = max(1, int(cfg.progress_every))
        eval_seq_len = int(cfg.eval_seq_len)

        # Pre-load all steering vectors once (so we can scan the parquet once).
        block_idx_to_steer = [int(i) for i in block_idx_to_steer]
        steer_vecs_gpu = {
            int(b): load_steer_vector(steer_dir_path, model_name, concept_slug, int(b)).to("cuda", non_blocking=True)
            for b in block_idx_to_steer
        }

        # Accumulators per layer
        # Each entry stores total NLL for one alpha over all processed tokens.
        nll_per_alpha = {int(b): torch.zeros((alpha_amount,), dtype=torch.float64) for b in block_idx_to_steer}
        token_amount = 0.0
        blocks_seen = 0

        # Keep α grid resident on GPU
        alphas_cuda = alphas.to("cuda", non_blocking=True)

        def make_hook(alpha_per_sample, steer_vec, mask):
            def _hook(module, inputs, output):
                x = output[0] if isinstance(output, (tuple, list)) else output  # [B,T,H]
                add = alpha_per_sample[:, None, None] * steer_vec[None, None, :]
                if mask is not None:
                    add = add * mask
                add = add.to(dtype=x.dtype)  # avoid promoting the whole forward to fp32
                x_steered = x + add
                if isinstance(output, (tuple, list)):
                    out = list(output)
                    out[0] = x_steered
                    return tuple(out)
                return x_steered
            return _hook
        idx_batch = 0
        # Scan the parquet ONCE, score all requested layers on the same token blocks.
        for input_tokens, label_tokens in iter_eval_blocks_from_parquet(
            tokenizer, eval_parquet, cfg, batch_size=int(cfg.eval_batch_size)
        ):
            print(idx_batch)
            idx_batch+=1
            B, T = input_tokens.shape
            blocks_seen += B
            token_amount += float(B * T)

            input_tokens = input_tokens.to("cuda", non_blocking=True)
            label_tokens = label_tokens.to("cuda", non_blocking=True)

            for alpha_0 in range(0, alpha_amount, alpha_batch_size):
                alpha_1 = min(alpha_amount, alpha_0 + alpha_batch_size)
                alpha_batch = alphas_cuda[alpha_0:alpha_1]
                alpha_in_batch_amount = int(alpha_batch.numel())

                # Repeat inputs once per α-batch; reused across all steered layers.
                input_rep = input_tokens.repeat(alpha_in_batch_amount, 1)    # [A*B,T]
                labels_rep = label_tokens.repeat(alpha_in_batch_amount, 1)   # [A*B,T]

                # Same α for each of the B blocks
                alpha_per_sample = alpha_batch.repeat_interleave(B)          # [A*B]

                if cfg.apply_last_token_only:
                    last_mask = torch.zeros(
                        (alpha_in_batch_amount * B, T, 1),
                        device=input_rep.device,
                        dtype=torch.float32,
                    )
                    last_mask[:, -1, 0] = 1.0
                else:
                    last_mask = None

                # Run one forward per layer (hooked at that layer)
                for block_idx in block_idx_to_steer:
                    handle = blocks[int(block_idx)].register_forward_hook(
                        make_hook(alpha_per_sample, steer_vecs_gpu[int(block_idx)], last_mask)
                    )

                    with torch.inference_mode():
                        # IMPORTANT: use_cache=False and no attention_mask (no padding).
                        logits = model(input_ids=input_rep, use_cache=False).logits.to(torch.float32)
                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.shape[-1]),
                            labels_rep.reshape(-1),
                            reduction="none",
                        ).reshape(alpha_in_batch_amount, B, T)
                        nll = loss.sum(dim=(1, 2)).detach().cpu().to(torch.float64)

                    handle.remove()
                    nll_per_alpha[int(block_idx)][alpha_0:alpha_1] += nll

                    forward_counter += 1
                    if (forward_counter % progress_mod) == 0:
                        await asyncio.sleep(0)

                del input_rep, labels_rep, alpha_per_sample, last_mask

            del input_tokens, label_tokens

        if blocks_seen == 0:
            raise RuntimeError(
                f"No usable {cfg.text_field!r} rows found in parquet={eval_parquet}, "
                f"or not enough tokens to form blocks of length {eval_seq_len + 1}."
            )

        # Save one curve per layer
        for block_idx in block_idx_to_steer:
            cross_entropy = (nll_per_alpha[int(block_idx)] / token_amount).numpy()
            perplexity = np.exp(cross_entropy)

            zero_idx = int((alphas == 0.0).nonzero(as_tuple=False).view(-1)[0].item())
            cross_entropy0 = float(cross_entropy[zero_idx])
            delta_cross_entropy = (cross_entropy - cross_entropy0).astype(np.float32)

            out_path = save_root / f"layer_{int(block_idx)}_cross_entropy.npz"
            meta = {
                "model": model_name,
                "concept": concept_label,
                "concept_slug": concept_slug,
                "layer_idx": int(block_idx),
                "eval_parquet": str(eval_parquet),
                "text_field": cfg.text_field,
                "eval_blocks": int(blocks_seen),
                "seq_len": eval_seq_len,
                "tokens_scored_per_alpha": int(blocks_seen * eval_seq_len),
                "apply_last_token_only": bool(cfg.apply_last_token_only),
                "alphas": {
                    "start": float(alphas[0].item()),
                    "end": float(alphas[-1].item()),
                    "steps": int(alpha_amount),
                    "alpha_batch_size": int(cfg.alpha_batch_size),
                },
                "baseline_alpha": 0.0,
                "baseline_cross_entropy": cross_entropy0,
            }
            np.savez_compressed(
                out_path,
                alphas=alphas.cpu().numpy().astype(np.float32),
                cross_entropy=cross_entropy.astype(np.float32),
                perplexity=perplexity.astype(np.float32),
                delta_cross_entropy=delta_cross_entropy,
                meta=json.dumps(meta),
            )

        torch.cuda.empty_cache()
        return {"ok": True}
