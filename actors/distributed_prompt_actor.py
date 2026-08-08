from __future__ import annotations

from .distributed import DistributedActorMixin
from .model_placement import dtype_from_name
from .prompts_actor import LLMActor


class DistributedPromptActor(DistributedActorMixin, LLMActor):
    """A prompt-generation logical actor spanning one or more GPUs."""

    _distributed_model_attrs = ("model", "tok")

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
        self.compute_dtype = dtype_from_name(dtype)
        self._configure_topology(logical_actors, gpus_per_actor)

        self.tok, self.model = self._load_distributed_causal_lm(
            model_name,
            dtype,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )

