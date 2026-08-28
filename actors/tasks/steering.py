"""Build steering vectors from positive and negative activations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import torch
from monarch.actor import Actor, endpoint

from utils.batch_size import critical_batch_size
from utils.batching import chunked
from utils.data import read_jsonl_texts
from utils.modeling import find_block_list
from utils.naming import model_slug
from utils.runtime.actor import DistributedActorMixin


@dataclass
class SteeringConfig:
    """Settings for steering-vector extraction."""
    batch_size: int = 0  # 0 chooses the computed critical size
    max_length: int = 300
    dtype: str = "float32"
    seed: int = 42
    n_positive: int | None = None
    n_negative: int | None = None
    contrastive: bool = False
    block_per_pass: int = 0
    progress_every: int = 5


class DistributedSteeringActor(DistributedActorMixin, Actor):
    """Compute steering vectors on one or more GPUs."""

    _distributed_model_attrs = ("model", "tokenizer")

    def __init__(
        self,
        model_name: str,
        dtype: str,
        logical_actors: int,
        gpus_per_actor: int,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name = model_name
        self.dtype_name = dtype
        self._configure_topology(logical_actors, gpus_per_actor)
        self.tokenizer, self.model = self._load_distributed_causal_lm(
            model_name,
            dtype,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )

    @endpoint
    async def compute_for(
        self,
        concept_slug: str,
        concept_label: str,
        block_idx_to_hook: list[int | None],
        cfg_dict: dict,
        prompts_directory: str,
        save_dir: str,
        layer_path: str | None,
        logical_rank: int,
    ) -> dict | None:
        """Compute and save steering vectors for one concept."""
        cfg = SteeringConfig(**cfg_dict)
        # Ranks in one replica share inputs; different replicas use different seeds.
        torch.manual_seed(cfg.seed + int(logical_rank))
        torch.cuda.manual_seed_all(cfg.seed + int(logical_rank))

        model = self.model
        tokenizer = self.tokenizer
        activation_model = getattr(model, "base_model", model)
        model_config = model.config
        text_config = getattr(model_config, "text_config", model_config)
        hidden_size = int(text_config.hidden_size)
        blocks = find_block_list(model, override_path=layer_path)
        if block_idx_to_hook == [None]:
            block_idx_to_hook = list(range(len(blocks)))
        block_indices = [int(index) for index in block_idx_to_hook]
        if any(index < 0 or index >= len(blocks) for index in block_indices):
            raise ValueError(f"Layer index outside [0, {len(blocks) - 1}]")

        if cfg.block_per_pass > 0:
            block_batches = list(chunked(block_indices, int(cfg.block_per_pass)))
        else:
            block_batches = [block_indices]

        prompt_root = Path(prompts_directory)
        positive_path = prompt_root / f"{concept_slug}_positive.jsonl"
        if cfg.contrastive:
            negative_path = prompt_root / f"{concept_slug}_negative.jsonl"
        else:
            negative_path = prompt_root / f"{concept_slug}_{model_slug(self.model_name)}_negative.jsonl"
        positive_texts = read_jsonl_texts(positive_path, cfg.n_positive)
        negative_texts = read_jsonl_texts(negative_path, cfg.n_negative)
        if not positive_texts or not negative_texts:
            if self.is_leader:
                return {
                    "rank": int(logical_rank),
                    "model": self.model_name,
                    "concept": concept_label,
                    "error": f"Empty/missing JSONLs for slug {concept_slug!r} in {prompt_root}",
                }
            return None
        estimate = critical_batch_size(
            model,
            kind="hidden",
            prompt_tokens=int(cfg.max_length),
            limit=max(len(positive_texts), len(negative_texts)),
            requested=int(cfg.batch_size),
            dtype=cfg.dtype,
            use_cache=True,
            tp_mesh=getattr(self, "tp_mesh", None),
        )
        cfg.batch_size = estimate.batch_size

        mean_related: dict[int, torch.Tensor] = {}
        mean_unrelated: dict[int, torch.Tensor] = {}
        progress_every = max(1, int(cfg.progress_every))

        for batch_indices in block_batches:
            related_batch = {
                index: torch.zeros(hidden_size, dtype=torch.float32, device="cpu")
                for index in batch_indices
            }
            unrelated_batch = {
                index: torch.zeros(hidden_size, dtype=torch.float32, device="cpu")
                for index in batch_indices
            }
            phase = "related"
            current_mask = None
            current_token_count = None

            def make_hook(block_index: int):
                def hook(_module, _inputs, output):
                    if not self.is_leader:
                        return
                    activation = output[0] if isinstance(output, (tuple, list)) else output
                    if activation.shape[-1] != hidden_size:
                        raise RuntimeError(
                            f"Expected replicated hidden width {hidden_size}, got {activation.shape[-1]}"
                        )
                    if current_mask is None:
                        per_sample = activation.mean(dim=1)
                    else:
                        masked = activation * current_mask.unsqueeze(-1).to(activation.dtype)
                        per_sample = (
                            masked.sum(dim=1) / current_token_count.unsqueeze(-1)
                        )
                    target = related_batch if phase == "related" else unrelated_batch
                    target[block_index] += per_sample.sum(dim=0).to(torch.float32).cpu()

                return hook

            handles = [blocks[index].register_forward_hook(make_hook(index)) for index in batch_indices]
            try:
                phase = "related"
                related_steps = 0
                with torch.inference_mode():
                    for related_steps, prompts in enumerate(
                        chunked(positive_texts, int(cfg.batch_size)), start=1
                    ):
                        encoded = tokenizer(
                            prompts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=int(cfg.max_length),
                        )
                        input_ids = encoded["input_ids"].to(self.device, non_blocking=True)
                        current_mask = encoded["attention_mask"].to(self.device, non_blocking=True)
                        current_token_count = current_mask.sum(dim=1)
                        activation_model(input_ids=input_ids, attention_mask=current_mask)
                        if related_steps % progress_every == 0:
                            await asyncio.sleep(0)

                phase = "unrelated"
                unrelated_steps = 0
                with torch.inference_mode():
                    for unrelated_steps, prompts in enumerate(
                        chunked(negative_texts, int(cfg.batch_size)), start=1
                    ):
                        encoded = tokenizer(
                            prompts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=int(cfg.max_length),
                        )
                        input_ids = encoded["input_ids"].to(self.device, non_blocking=True)
                        current_mask = encoded["attention_mask"].to(self.device, non_blocking=True)
                        current_token_count = current_mask.sum(dim=1)
                        activation_model(input_ids=input_ids, attention_mask=current_mask)
                        if unrelated_steps % progress_every == 0:
                            await asyncio.sleep(0)
            finally:
                for handle in handles:
                    handle.remove()

            if self.is_leader:
                for index in batch_indices:
                    mean_related[index] = related_batch[index] / len(positive_texts)
                    mean_unrelated[index] = unrelated_batch[index] / len(negative_texts)

        if not self.is_leader:
            return None

        save_root = Path(save_dir) / model_slug(self.model_name) / concept_slug
        save_root.mkdir(parents=True, exist_ok=True)
        files = []
        for block_index in sorted(block_indices):
            output_file = save_root / f"layer_{block_index}.pt"
            torch.save(
                {
                    "model": self.model_name,
                    "concept": concept_label,
                    "concept_slug": concept_slug,
                    "layer_idx": block_index,
                    "hidden_size": hidden_size,
                    "steering_vector": (
                        mean_related[block_index] - mean_unrelated[block_index]
                    ).to(torch.float32),
                    "tensor_parallel_size": self.gpus_per_actor,
                },
                output_file,
            )
            files.append(str(output_file))
        torch.cuda.empty_cache()
        return {
            "rank": int(logical_rank),
            "model": self.model_name,
            "concept": concept_label,
            "concept_slug": concept_slug,
            "layers": sorted(block_indices),
            "saved": files,
            "gpus_per_actor": self.gpus_per_actor,
        }
