"""Load one model copy on one or more GPUs."""

from __future__ import annotations

import os
from typing import Iterable

import torch
import torch.distributed as dist
from monarch.actor import endpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

from .placement import dtype_from_name


class DistributedActorMixin:
    """Share model loading and multi-GPU setup."""

    _distributed_model_attrs: tuple[str, ...] = ()

    def _configure_topology(
        self,
        logical_actors: int,
        gpus_per_actor: int,
    ) -> None:
        """Set rank roles and initialize tensor parallelism when needed."""
        torch.backends.cuda.matmul.allow_tf32 = True
        self.logical_actors = int(logical_actors)
        self.gpus_per_actor = int(gpus_per_actor)
        if self.logical_actors < 1 or self.gpus_per_actor < 1:
            raise ValueError("logical_actors and gpus_per_actor must be positive")

        self.global_rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", str(self.global_rank)))
        self.replica_rank = self.global_rank // self.gpus_per_actor
        self.tensor_parallel_rank = self.global_rank % self.gpus_per_actor
        self.is_leader = self.tensor_parallel_rank == 0
        self.device_mesh = None
        self.tp_mesh = None

        if not torch.cuda.is_available():
            if self.logical_actors != 1 or self.gpus_per_actor != 1:
                raise RuntimeError("CPU fallback supports one model worker using one process")
            self.device = torch.device("cpu")
            return

        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        if self.gpus_per_actor > 1:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            self.device_mesh = torch.distributed.init_device_mesh(
                "cuda",
                (self.logical_actors, self.gpus_per_actor),
                mesh_dim_names=("replica", "tp"),
            )
            self.tp_mesh = self.device_mesh["tp"]

    def _load_distributed_causal_lm(
        self,
        model_name: str,
        dtype: str,
        *,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ):
        """Load a tokenizer and model for this actor GPU group."""
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token

        common = dict(
            low_cpu_mem_usage=True,
            dtype=dtype_from_name(dtype),
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            attn_implementation="sdpa",
        )
        if self.gpus_per_actor > 1:
            common.update(tp_plan="auto", device_mesh=self.device_mesh)
        elif self.device.type == "cuda":
            common.update(device_map={"": self.local_rank})

        model = AutoModelForCausalLM.from_pretrained(model_name, **common)
        model.eval()
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None:
            model.generation_config.eos_token_id = tokenizer.eos_token_id
        return tokenizer, model

    def _clear_distributed_models(self, extra_attrs: Iterable[str] = ()) -> None:
        """Release loaded model references and cached GPU memory."""
        for attr in (*self._distributed_model_attrs, *tuple(extra_attrs)):
            if hasattr(self, attr):
                setattr(self, attr, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @endpoint
    async def describe(self) -> dict:
        """Return this actor rank and model information."""
        return {
            "global_rank": self.global_rank,
            "local_rank": self.local_rank,
            "replica_rank": self.replica_rank,
            "tensor_parallel_rank": self.tensor_parallel_rank,
            "gpus_per_actor": self.gpus_per_actor,
            "model": getattr(self, "model_name", None),
        }

    @endpoint
    async def close(self) -> None:
        """Release models and close the distributed process group."""
        self._clear_distributed_models()
        if dist.is_initialized():
            dist.destroy_process_group()
