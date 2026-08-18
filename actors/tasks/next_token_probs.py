"""Measure how steering changes next-token probabilities."""

import json
from pathlib import Path
from dataclasses import dataclass

import asyncio

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from monarch.actor import Actor, endpoint

from utils.data import load_contexts_for_concept
from utils.modeling import find_block_list, load_steering_vector
from utils.naming import model_slug


def ensure_full_vocab_logits(logits, expected_vocab_size: int, tp_mesh=None):
    """Gather sharded logits so every token is present."""
    if isinstance(logits, DTensor):
        logits = logits.full_tensor()
    if logits.shape[-1] == expected_vocab_size:
        return logits
    if tp_mesh is None:
        raise RuntimeError(
            f"Expected vocabulary width {expected_vocab_size}, got {logits.shape[-1]}"
        )
    tp_size = int(tp_mesh.size())
    if logits.shape[-1] * tp_size != expected_vocab_size:
        raise RuntimeError(
            "Cannot reconstruct full-vocabulary logits: "
            f"local width={logits.shape[-1]}, TP size={tp_size}, "
            f"expected width={expected_vocab_size}"
        )
    local_logits = logits.contiguous()
    gathered = [torch.empty_like(local_logits) for _ in range(tp_size)]
    dist.all_gather(gathered, local_logits, group=tp_mesh.get_group())
    return torch.cat(gathered, dim=-1)


@dataclass
class TokenPlotConfig:
    """Settings for next-token probability curves."""
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


class TokenActor(Actor):
    """Compute next-token probability curves for a loaded model."""

    def _ensure_model(self, model_name: str, dtype_str: str):
        """Check that the requested model is already loaded."""
        if self.current_model_name == model_name and self.current_dtype == dtype_str:
            return
        raise RuntimeError("Distributed actor was initialized for a different model")

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
        context_indices=None,  # optional list[int] for replica-level sharding
    ):
        """Compute and save token-probability curves for selected contexts."""
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
            if self.is_leader:
                return {"error": f"No contexts in {contexts_file} for concept '{concept_slug}'"}
            return None

        if context_indices is None:
            selected_context_indices = list(range(len(contexts)))
        else:
            selected_context_indices = [int(index) for index in context_indices]
            invalid = [
                index
                for index in selected_context_indices
                if index < 0 or index >= len(contexts)
            ]
            if invalid:
                raise IndexError(f"Context indices out of range: {invalid}")


        # Prepare α grid (ensure 0 present)
        alphas = torch.linspace(cfg.alpha_start, cfg.alpha_end, steps=cfg.alpha_steps, dtype=torch.float32)
        if (alphas == 0.0).any() == False:
            alphas = torch.sort(torch.cat([alphas, torch.tensor([0.0])]))[0]
        alpha_amount = alphas.numel()
        batch_size = alpha_amount if cfg.batch_size == 0 else cfg.batch_size
        # Save root
        save_root = Path(save_dir) / model_slug(model_name) / concept_slug
        if self.is_leader:
            save_root.mkdir(parents=True, exist_ok=True)

        # Main loops: layers × contexts
        steer_dir_path = Path(steer_dir)
        progress_mod = max(1, int(cfg.progress_every))

        for block_idx in block_idx_to_steer:
            # Load steering vector for this (model, concept, layer)
            steer_vec_cpu = load_steering_vector(steer_dir_path, model_name, concept_slug, block_idx)  # [H], float32 on CPU
            if cfg.normalize:
                steer_vec_cpu = steer_vec_cpu / torch.norm(steer_vec_cpu).clamp_min(1e-8)
            for work_idx, ctx_idx in enumerate(selected_context_indices):
                context = contexts[ctx_idx]
                # Tokenize once
                ctx_source_line = context_source_lines[ctx_idx]
                enc = tokenizer(context, return_tensors="pt", truncation=True, max_length=cfg.max_length)
                input_ids = enc["input_ids"]           # [1, T]
                attn_mask = enc["attention_mask"]      # [1, T]
                token_amount = int(input_ids.shape[1])

                # Keep a maximum-sized input batch on device and slice it per α batch.
                # The same context is evaluated at every steering strength.
                input_ids = input_ids.repeat(batch_size, 1).to(self.device, non_blocking=True)
                attn_mask = attn_mask.repeat(batch_size, 1).to(self.device, non_blocking=True)

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
                        x_steered = (x + add).to(x.dtype)
                        if isinstance(output, (tuple, list)):
                            out = list(output)
                            out[0] = x_steered
                            return tuple(out)
                        return x_steered
                    return _hook

                def forward_alpha_batch(alpha_batch):
                    """Return full-vocabulary logits for one bounded α batch."""
                    current_batch_size = int(alpha_batch.shape[0])
                    batch_input_ids = input_ids[:current_batch_size]
                    batch_attention_mask = attn_mask[:current_batch_size]
                    batch_last_mask = (
                        last_mask[:current_batch_size]
                        if last_mask is not None
                        else None
                    )
                    handle = blocks[block_idx].register_forward_hook(
                        make_hook(alpha_batch, steer_vec_gpu, batch_last_mask)
                    )
                    try:
                        with torch.inference_mode():
                            out = model(
                                input_ids=batch_input_ids,
                                attention_mask=batch_attention_mask,
                            )
                            logits = ensure_full_vocab_logits(
                                out.logits,
                                int(model.config.vocab_size),
                                self.tp_mesh,
                            )
                            return logits[:, -1, :].to(torch.float32)
                    finally:
                        handle.remove()

                # Select the reported tokens from the two sweep endpoints first.
                # This costs two small forwards, but avoids copying the full
                # [alpha, vocabulary] probability matrix to CPU.
                alpha_min = alphas[:1].to(input_ids.device)
                alpha_max = alphas[-1:].to(input_ids.device)
                topk = min(int(cfg.top_k), int(model.config.vocab_size))
                token_ids_alphamin = torch.softmax(
                    forward_alpha_batch(alpha_min)[0], dim=-1
                ).topk(topk, largest=True).indices
                token_ids_alphamax = torch.softmax(
                    forward_alpha_batch(alpha_max)[0], dim=-1
                ).topk(topk, largest=True).indices

                probs_alphamax_batches = []
                probs_alphamin_batches = []
                # Split the α grid to keep VRAM bounded on long sweeps. Only
                # the selected curves cross the GPU/CPU boundary.
                for alpha_batch in torch.split(alphas, batch_size):
                    alpha_batch = alpha_batch.to(input_ids.device)
                    query_token_logits = forward_alpha_batch(alpha_batch)
                    probs = torch.softmax(query_token_logits, dim=-1)
                    if self.is_leader:
                        probs_alphamax_batches.append(
                            probs[:, token_ids_alphamax].cpu()
                        )
                        probs_alphamin_batches.append(
                            probs[:, token_ids_alphamin].cpu()
                        )

                if not self.is_leader:
                    if (work_idx % progress_mod) == 0:
                        await asyncio.sleep(0)
                    continue

                probs_topk_alphamax = torch.cat(probs_alphamax_batches, dim=0).numpy()
                probs_topk_alphamin = torch.cat(probs_alphamin_batches, dim=0).numpy()
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
                    "vocab_size": int(model.config.vocab_size),
                    "top_k": int(topk),
                    "apply_last_token_only": bool(cfg.apply_last_token_only),
                    "alphas": {"start": float(alphas[0].item()), "end": float(alphas[-1].item()), "steps": int(alpha_amount)},
                    "baseline_alpha": 0,
                    "tensor_parallel_size": int(getattr(self, "gpus_per_actor", 1)),
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

                if (work_idx % progress_mod) == 0:
                    # Yield so other actor calls can run.
                    await asyncio.sleep(0)

        torch.cuda.empty_cache()
        if not self.is_leader:
            return None
        return {
            "ok": True,
            "context_indices": selected_context_indices,
            "layers": [int(index) for index in block_idx_to_steer],
        }
