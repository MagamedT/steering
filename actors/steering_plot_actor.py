import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import asyncio

import numpy as np
import torch
from monarch.actor import Actor, endpoint

from .utils import find_block_list, model_slug, load_model_and_tokenizer, load_steer_vector, load_contexts_for_concept


# -----------------------------
# Actor config (sent as dict)
# -----------------------------

@dataclass
class TokenPlotConfig:
    dtype: str = "float32"
    seed: int = 42
    batch_size: int = 256
    alpha_start: float = -200
    alpha_end: float = 200
    alpha_steps: int = 1_024
    max_length: int = 100
    apply_last_token_only: bool = False
    normalize: bool = False
    top_k: int = 100
    progress_every: int = 5


# -----------------------------
# Actor (one GPU)
# -----------------------------

class TokenActor(Actor):
    """
    One actor per GPU. Caches last loaded (model, dtype).
    Exposes endpoint to compute batched-alpha probability curves for many layers/contexts.
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
    async def compute_plot_curves(
        self,
        model_name,       # str
        concept_slug,     # str
        concept_label,    # str (for metadata)
        block_idx_to_steer,       # list[int] or "all"
        contexts_file,    # str (path to text file with one context per line)
        steer_dir,        # str (root where steering vectors live)
        save_dir,         # str (root where .npz curve files go)
        layer_path=None,  # optional str for block list path
        cfg_dict=None,    # dict (PlotConfig)
        rank_hint=0,      # int
    ):
        cfg = TokenPlotConfig(**(cfg_dict or {}))
        torch.manual_seed(cfg.seed + int(rank_hint))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed + int(rank_hint))

        # Load model/tokenizer
        self._ensure_model(model_name, cfg.dtype)
        tokenizer, model = self.tokenizer, self.model
        # Resolve blocks & layers
        blocks = find_block_list(model, override_path=layer_path)
        n_blocks = len(blocks)
        if block_idx_to_steer == [None]:
            block_idx_to_steer = list(range(n_blocks))

        # Read contexts (one per line)
        # Read contexts for this concept:
        # - shared negative prompts
        # - concept-specific positive prompts
        contexts, context_source_lines = load_contexts_for_concept(
            contexts_file,
            concept_slug=concept_slug,
            concept_label=concept_label,
        )
        if not contexts:
            return {"error": f"No contexts in {contexts_file} for concept '{concept_slug}'"}


        # Prepare α grid (ensure 0 present)
        alphas = torch.linspace(cfg.alpha_start, cfg.alpha_end, steps=cfg.alpha_steps, dtype=torch.float32)
        if (alphas == 0.0).any() == False:
            alphas = torch.sort(torch.cat([alphas, torch.tensor([0.0])]))[0]
        alpha_amount = alphas.numel()
        batch_size = alpha_amount if cfg.batch_size == 0 else cfg.batch_size
        # Save root
        save_root = Path(save_dir) / model_slug(model_name) / concept_slug
        save_root.mkdir(parents=True, exist_ok=True)

        # Main loops: layers × contexts
        steer_dir_path = Path(steer_dir)
        progress_mod = max(1, int(cfg.progress_every))

        for block_idx in block_idx_to_steer:
            # Load steering vector for this (model, concept, layer)
            steer_vec_cpu = load_steer_vector(steer_dir_path, model_name, concept_slug, block_idx)  # [H], float32 on CPU
            if cfg.normalize:
                steer_vec_cpu = steer_vec_cpu / torch.norm(steer_vec_cpu).clamp_min(1e-8)
            for ctx_idx, context in enumerate(contexts):
                # Tokenize once
                ctx_source_line = context_source_lines[ctx_idx]
                enc = tokenizer(context, return_tensors="pt", truncation=True, max_length=cfg.max_length)
                input_ids = enc["input_ids"]           # [1, T]
                attn_mask = enc["attention_mask"]      # [1, T]
                token_amount = int(input_ids.shape[1])

                # Repeat for all α
                input_ids = input_ids.repeat(batch_size, 1).to("cuda", non_blocking=True)             # [alpha_amount, token_amount]
                attn_mask = attn_mask.repeat(batch_size, 1).to("cuda", non_blocking=True)            # [alpha_amount, token_amount]

                # Precompute last-token mask (if requested)
                if cfg.apply_last_token_only:
                    last_mask = torch.zeros((batch_size, token_amount, 1), device=input_ids.device) # [alpha_amount, token_amount, 1 = corresponds to the residual path dimension]
                    last_mask[:, -1, 0] = 1
                else:
                    last_mask = None

                # Hook: add alpha * steer_vec to block output
                steer_vec_gpu = steer_vec_cpu.to(input_ids.device).to(torch.float32)  # [H]
                def make_hook(alpha, steer_vec, mask):
                    def _hook(module, inputs, output):
                        x = output[0] if isinstance(output, (tuple, list)) else output  # [A,T,H]
                        add = alpha[:, None, None] * steer_vec[None, None, :]
                        if mask is not None:
                            add = add * mask
                        x_steered = x + add
                        # optionally cast back to bf16 for the rest of the network:
                        if isinstance(output, (tuple, list)):
                            out = list(output)
                            out[0] = x_steered
                            return tuple(out)
                        return x_steered
                    return _hook
                
                probs_list = []
                # Split alpha grid to keep VRAM bounded on long sweeps.
                for alpha_batch in torch.split(alphas, batch_size):
                    # α batch on GPU
                    alpha_batch = alpha_batch.to(input_ids.device)
                    current_batch_size = alpha_batch.shape[0]
                    if current_batch_size < batch_size:
                        # trim for the last batch if smaller size than batch_size
                        input_ids = input_ids[:current_batch_size]            # [alpha_amount, token_amount]
                        attn_mask = attn_mask[:current_batch_size]            # [alpha_amount, token_amount]
                        if cfg.apply_last_token_only:
                            last_mask = last_mask[:current_batch_size] # [alpha_amount, token_amount, 1 = corresponds to the residual path dimension]

                    handle = blocks[block_idx].register_forward_hook(make_hook(alpha_batch, steer_vec_gpu, last_mask))

                    # Forward once for all α
                    with torch.inference_mode():
                        out = model(input_ids=input_ids, attention_mask=attn_mask)
                        logits = out.logits  # [A, T, V]
                        query_token_logits = logits[:, -1, :].to(torch.float32)  # [A, V]
                        probs_list.append(torch.softmax(query_token_logits, dim=-1).cpu())  # fp32 softmax

                    handle.remove()

                # stack the probs_list along alpha dim
                probs = torch.cat(probs_list, dim = 0)
                # Choose top-k tokens based probs of query token
                topk = min(int(cfg.top_k), probs.shape[-1])
                # get the topk probs for alpha_max
                idx_topk_alphamax = probs[-1].topk(topk, largest=True)
                idx_topk_alphamin = probs[0].topk(topk, largest=True)
                # Keep both ends of the alpha sweep to analyze asymmetric token shifts.
                token_ids_alphamax, token_ids_alphamin = idx_topk_alphamax.indices, idx_topk_alphamin.indices  # [K] on GPU

                # Slice curves
                probs_topk_alphamax, probs_topk_alphamin = probs[:, token_ids_alphamax], probs[:, token_ids_alphamin]                  # [A, K]

                ## clean
                # Move to numpy
                probs_topk_alphamax, probs_topk_alphamin = probs_topk_alphamax.to(torch.float32).cpu().numpy(), probs_topk_alphamin.to(torch.float32).cpu().numpy()
                token_ids_alphamax, token_ids_alphamin = token_ids_alphamax.to(torch.int32).cpu().numpy(), token_ids_alphamin.to(torch.int32).cpu().numpy()
                toks_alphamax, toks_alphamin = [tokenizer.decode([int(t)]) for t in token_ids_alphamax.tolist()], [tokenizer.decode([int(t)]) for t in token_ids_alphamin.tolist()]

                # Save .npz
                out_path = save_root / f"layer_{block_idx}_ctx_{ctx_idx}.npz"
                meta = {
                    "model": model_name,
                    "concept": concept_label,
                    "concept_slug": concept_slug,
                    "context": context,
                    "context_source_line": int(ctx_source_line),
                    "layer_idx": int(block_idx),
                    "seq_len": int(token_amount),
                    "vocab_size": int(probs.shape[-1]),
                    "top_k": int(topk),
                    "apply_last_token_only": bool(cfg.apply_last_token_only),
                    "alphas": {"start": float(alphas[0].item()), "end": float(alphas[-1].item()), "steps": int(alpha_amount)},
                    "baseline_alpha": 0,
                }
                np.savez_compressed(
                    out_path,
                    alphas=alphas.cpu().numpy().astype(np.float32),     # [A]
                    probs_alphamax=probs_topk_alphamax,                                     # [A,K]
                    probs_alphamin=probs_topk_alphamin,    
                    token_alphamax=token_ids_alphamax,                                   # [K]
                    token_alphamin=token_ids_alphamin,
                    token_strs_alphamax=np.array(toks_alphamax, dtype=object),            # [K]
                    token_strs_alphamin=np.array(toks_alphamin, dtype=object),
                    meta=json.dumps(meta),
                )

                if (ctx_idx % progress_mod) == 0:
                    # Let the actor service its mailbox during long loops
                    await asyncio.sleep(0)

        torch.cuda.empty_cache()
        return {"ok": True}
