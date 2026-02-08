# steering_actor.py
# Monarch actor for steering-vector computation.
# - Low-VRAM hook: reduce [B,T,H] -> [H] on GPU, then send [H] to CPU
# - Optional --layers_per_pass: cap concurrent hooks
# - Caches the last loaded model per actor to avoid reloads

from pathlib import Path
import asyncio
from typing import List, Optional, Tuple, Dict

import torch
from monarch.actor import Actor, endpoint
from dataclasses import dataclass

from .utils import find_block_list, read_jsonl_texts, model_slug, chunked, load_model_and_tokenizer

# -----------------------------
# Config carried as a dict to endpoint
# -----------------------------

@dataclass
class SteeringConfig:
    batch_size: int = 50
    max_length: int = 300
    dtype: str = "float32"
    seed: int = 42
    is_padded_masked: bool = True
    n_positive: int | None = None
    n_negative: int | None = None
    contrastive: bool = False
    block_per_pass: int = 0   # 0 = all layers in one pass
    progress_every: int = 5    # yield to event loop every N micro-batches


# -----------------------------
# Steering Actor (one per GPU)
# -----------------------------

class SteeringActor(Actor):
    """
    Each actor uses exactly one GPU. The last loaded model is cached to avoid reloads.
    """
    def __init__(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        self.current_model_name = None
        self.current_dtype = None
        self.tokenizer = None
        self.model = None
        self._dtype = None

    def _ensure_model(self, model_name: str, dtype_str: str):
        if self.model is not None and self.current_model_name == model_name and self.current_dtype == dtype_str:
            return
        self.tokenizer = None
        self.model = None
        torch.cuda.empty_cache()
        self.tokenizer, self.model = load_model_and_tokenizer(model_name, dtype_str)
        self.current_model_name = model_name
        self.current_dtype = dtype_str
        self._dtype = dtype_str

    @endpoint
    async def compute_for(
        self,
        model_name,         # str
        concept_slug,       # str (e.g., "fresh_snow")
        concept_label,      # str (human label; for metadata only)
        block_idx_to_hook,         # list[int] or "all"
        cfg_dict,           # dict (SteeringConfig asdict)
        prompts_directory,             # str
        save_dir,           # str
        layer_path=None,    # optional str
        rank_hint=0,        # int
    ):
        """
        Compute steering vectors for (model_name, concept_slug) and save:
        save_dir/<model_slug>/<concept_slug>/layer_<i>.pt
        """
        # set seeds with different seed per gpu/cpu
        cfg = SteeringConfig(**cfg_dict)
        torch.manual_seed(cfg.seed + int(rank_hint))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed + int(rank_hint))

        # Load model/tokenizer
        self._ensure_model(model_name, cfg.dtype)
        tokenizer, model = self.tokenizer, self.model
        # Prefer text_config.hidden_size if it exists, for multimodal LLM you need to check text_config.
        model_config = model.config
        text_cfg = getattr(model_config, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
            hidden_size = text_cfg.hidden_size
        elif hasattr(model_config, "hidden_size"):
            hidden_size = model_config.hidden_size

        # get all the decoder-transformer blocks
        blocks = find_block_list(model, override_path=layer_path)
        n_blocks = len(blocks)
        if block_idx_to_hook == [None]:
            block_idx_to_hook = list(range(n_blocks))

        # Optional: number of block per pass
        if cfg.block_per_pass > 0:
            block_per_pass = int(cfg.block_per_pass)
            block_batches_idx = [block_idx_to_hook[i:i+block_per_pass] for i in range(0, len(block_idx_to_hook), block_per_pass)]
        else:
            # one big batch containing all the blocks to hook
            block_batches_idx = [block_idx_to_hook]

        # Load related and unrelated prompts previously generated
        prompts_directory_path = Path(prompts_directory)
        if cfg.contrastive:
            positive_path = prompts_directory_path / f"{concept_slug}_positive.jsonl"
            negative_path = prompts_directory_path / f"{concept_slug}_negative.jsonl"
        else:
            positive_path = prompts_directory_path / f"{concept_slug}_positive.jsonl"
            negative_path = prompts_directory_path / f"{concept_slug}_{model_slug(model_name)}_negative.jsonl"
        positive_texts = read_jsonl_texts(positive_path, cfg.n_positive)
        negative_texts = read_jsonl_texts(negative_path, cfg.n_negative)
        if not positive_texts or not negative_texts:
            return {
                "rank": int(rank_hint), "model": model_name, "concept": concept_label,
                "error": f"Empty/missing JSONLs for slug '{concept_slug}' in {prompts_directory_path}"
            }
        if len(positive_texts) % cfg.batch_size != 0:
            raise ValueError("batch_size must evenly divide positive prompts")
        if len(negative_texts) % cfg.batch_size != 0:
            raise ValueError("batch_size must evenly divide negative prompts")

        # Global accumulators for all blocks on RAM
        mean_related, mean_unrelated = {}, {}

        # For communication between GPU and Monarch internals
        progress_every = max(1, int(cfg.progress_every))

        # -------- Process layer chunks
        for batch_idxs in block_batches_idx:
            # Per decoder-block batch RAM accumulators (EXPLICITLY ON CPU), if only one batch, then mean_related_batch plays the role of mean_related
            mean_related_batch = {i: torch.zeros(hidden_size, dtype=torch.float32, device="cpu") for i in batch_idxs}
            mean_unrelated_batch = {i: torch.zeros(hidden_size, dtype=torch.float32, device="cpu") for i in batch_idxs}

            # Per-batch state for hooks
            
            #  GLOBAL variable for hook
            phase = "related" # says to which dictionnary to register the activations
            current_mask = None # store a mask on the padded tokens
            current_token_count = None

            # activation recording hook: reduce on GPU the activation to shape: [hidden_size], then send that to CPU
            def make_hook(block_idx: int):
                def _hook(module, inputs, output):
                    # some transformer on huggingface have activation as Tuples and not only tensor type
                    activation = output[0] if isinstance(output, (tuple, list)) else output  # [B,T,H] on GPU, with B the number of input prompts, T the prompt length, and H the hidden_size
                    B, T, H = activation.shape
                    if cfg.is_padded_masked and current_mask is not None:
                        # Mean over real tokens per prompt, then average over prompts in batch.
                        masked_activation_mean = activation * current_mask.unsqueeze(-1).to(activation.dtype)
                        masked_activation_mean = masked_activation_mean.sum(dim = 1) * 1/current_token_count.unsqueeze(-1) 
                        masked_activation_mean = masked_activation_mean.mean(dim = 0)
                        masked_activation_mean_cpu = masked_activation_mean.to(torch.float32).cpu()  # [H] on CPU
                        if phase == "related":
                            mean_related_batch[block_idx] += masked_activation_mean_cpu
                        else:
                            mean_unrelated_batch[block_idx] += masked_activation_mean_cpu
                    else:
                        # here we take into account the embedding of padding token in the mean-reduction along the dim = 1 (prompts length)
                        masked_activation_mean = activation.mean(dim=(0, 1))                # [H] on GPU
                        masked_activation_mean_cpu = masked_activation_mean.to(torch.float32).cpu()     # [H] on CPU
                        if phase == "related":
                            mean_related_batch[block_idx] += masked_activation_mean_cpu 
                        else:
                            mean_unrelated_batch[block_idx] += masked_activation_mean_cpu 
                return _hook
            
            # register hook and save handles
            handles = [blocks[i].register_forward_hook(make_hook(i)) for i in batch_idxs]

            # computation of activation mean for RELATED prompts
            phase = "related"
            with torch.inference_mode():
                for step, batch_prompts in enumerate(chunked(positive_texts, int(cfg.batch_size))):
                    tokenized_prompts = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=int(cfg.max_length))
                    input_ids = tokenized_prompts["input_ids"].to("cuda", non_blocking=True)
                    # attention mask in our case contains 0 only on padded tokens
                    attn_mask = tokenized_prompts["attention_mask"].to("cuda", non_blocking=True)
                    if cfg.is_padded_masked:
                        current_mask = attn_mask
                        current_token_count = attn_mask.sum(dim = 1)
                    else:
                        current_mask = None
                    # model inference to record the activations, we save the logits in _ for the GC of python
                    _ = model(input_ids=input_ids, attention_mask=attn_mask)
                    # every now and then, let the process sleep to let the GPU communicate with Monarch internals
                    if step % progress_every == 0:
                        await asyncio.sleep(0)
            # get number of prompts batches
            n_batches_related = step + 1
            # computation of activation mean for UNRELATED prompts
            phase = "unrelated"
            with torch.inference_mode():
                for step, batch_texts in enumerate(chunked(negative_texts, cfg.batch_size)):
                    tokenized_prompts = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=int(cfg.max_length))
                    input_ids = tokenized_prompts["input_ids"].to("cuda", non_blocking=True)
                    attn_mask = tokenized_prompts["attention_mask"].to("cuda", non_blocking=True)
                    if cfg.is_padded_masked:
                        current_mask = attn_mask
                        current_token_count = attn_mask.sum(dim = 1)
                    else:
                        current_mask = None
                    _ = model(input_ids=input_ids, attention_mask=attn_mask)
                    if step % progress_every == 0:
                        await asyncio.sleep(0)
            # final normalization for this batch of decoder blocks by the amount of prompts batches    
            n_batches_unrelated = step + 1
            mean_related_batch = {k: v / n_batches_related for k, v in mean_related_batch.items()}
            mean_unrelated_batch = {k: v / n_batches_unrelated for k, v in mean_unrelated_batch.items()}

            # kill all the hooks
            for h in handles:
                h.remove()

            # Merge this chunk into globals and create the keys in the global dicts: mean_related, mean_unrelated (on CPU)
            for i in batch_idxs:
                mean_related[i] = mean_related_batch[i]
                mean_unrelated[i] = mean_unrelated_batch[i]

        # Save steering vectors
        save_root = Path(save_dir) / model_slug(model_name) / concept_slug
        save_root.mkdir(parents=True, exist_ok=True)
        files = []
        for block_idx in sorted(block_idx_to_hook):
            # Standard steering direction: concept activation mean minus non-concept mean.
            steering_vector = (mean_related[block_idx] - mean_unrelated[block_idx]).to(torch.float32)
            out_file = save_root / f"layer_{block_idx}.pt"
            torch.save({
                "model": model_name,
                "concept": concept_label,
                "concept_slug": concept_slug,
                "layer_idx": block_idx,
                "hidden_size": hidden_size,
                "steering_vector": steering_vector,  # [H]
            }, out_file)
            files.append(str(out_file))

        torch.cuda.empty_cache()
        # GPU returns this:
        return {
            "rank": int(rank_hint),
            "model": model_name,
            "concept": concept_label,
            "concept_slug": concept_slug,
            "layers": sorted(block_idx_to_hook),
            "saved": files,
        }
