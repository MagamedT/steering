from __future__ import annotations

from steering.modeling import ensure_pad_token, set_left_padding
from steering.runtime.actor import DistributedActorMixin
from steering.runtime.placement import dtype_from_name

from .tasks.behavior import BehaviorActor as BinaryBehaviorEndpoints
from .tasks.behavior_continuous import BehaviorActor as ContinuousBehaviorEndpoints
from .tasks.cross_entropy import CrossEntropyActor as CrossEntropyEndpoints
from .tasks.log_odds import LogOddsActor as LogOddsEndpoints
from .tasks.mmlu import MMLUActor as MMLUEndpoints
from .tasks.next_token_probs import TokenActor as TokenProbabilityEndpoints
from .tasks.prompts import LLMActor as PromptEndpoints
from .tasks.rescore import RescoreActor as RescoreEndpoints


class _SingleModelActor(DistributedActorMixin):
    """Distributed lifecycle shared by single-model task endpoints."""

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
        self._configure_topology(logical_actors, gpus_per_actor)
        self.tokenizer, self.model = self._load_distributed_causal_lm(
            model_name,
            dtype,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        self.current_model_name = model_name
        self.current_dtype = dtype


class DistributedCrossEntropyActor(_SingleModelActor, CrossEntropyEndpoints):
    pass


class DistributedLogOddsActor(_SingleModelActor, LogOddsEndpoints):
    pass


class DistributedMMLUActor(_SingleModelActor, MMLUEndpoints):
    pass


class DistributedRescoreActor(_SingleModelActor, RescoreEndpoints):
    pass


class _GeneratorJudgeActor(DistributedActorMixin):
    """Distributed lifecycle for tasks with co-resident generator and judge."""

    _distributed_model_attrs = (
        "_gen_model",
        "_gen_tok",
        "_judge_model",
        "_judge_tok",
    )

    def __init__(
        self,
        generator_model_name: str,
        generator_dtype: str,
        judge_model_name: str,
        judge_dtype: str,
        logical_actors: int,
        gpus_per_actor: int,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name = generator_model_name
        self._configure_topology(logical_actors, gpus_per_actor)

        self._gen_tok, self._gen_model = self._load_distributed_causal_lm(
            generator_model_name,
            generator_dtype,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        set_left_padding(self._gen_tok)
        ensure_pad_token(self._gen_tok, self._gen_model)
        self._gen_name = generator_model_name
        self._gen_dtype = generator_dtype

        self._judge_tok, self._judge_model = self._load_distributed_causal_lm(
            judge_model_name,
            judge_dtype,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        set_left_padding(self._judge_tok)
        ensure_pad_token(self._judge_tok, self._judge_model)
        self._judge_name = judge_model_name
        self._judge_dtype = judge_dtype
        self._judge_token_ids_10 = None


class DistributedBehaviorActor(_GeneratorJudgeActor, BinaryBehaviorEndpoints):
    pass


class DistributedContinuousBehaviorActor(
    _GeneratorJudgeActor, ContinuousBehaviorEndpoints
):
    pass


class DistributedPromptActor(DistributedActorMixin, PromptEndpoints):
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


class DistributedTokenActor(DistributedActorMixin, TokenProbabilityEndpoints):
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
        self.current_model_name = model_name
        self.current_dtype = dtype
