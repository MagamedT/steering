from __future__ import annotations

from .concept_probs_actor import BehaviorActor as BinaryBehaviorActor
from .concept_probs_continuous_actor import BehaviorActor as ContinuousBehaviorActor
from .cross_entropy_actor import CrossEntropyActor
from .distributed import DistributedActorMixin
from .log_odds_actor import LogOddsActor
from .mmlu_actor import MMLUActor
from .rescore_actor import RescoreActor
from .utils import ensure_pad_token, set_left_padding


class _DistributedSingleModelActor(DistributedActorMixin):
    _distributed_model_attrs = ("model", "tokenizer")

    def _init_single_model(
        self,
        model_name: str,
        dtype: str,
        logical_actors: int,
        gpus_per_actor: int,
        local_files_only: bool,
        trust_remote_code: bool,
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


class DistributedCrossEntropyActor(_DistributedSingleModelActor, CrossEntropyActor):
    def __init__(
        self,
        model_name: str,
        dtype: str,
        logical_actors: int,
        gpus_per_actor: int,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        self._init_single_model(
            model_name,
            dtype,
            logical_actors,
            gpus_per_actor,
            local_files_only,
            trust_remote_code,
        )


class DistributedLogOddsActor(_DistributedSingleModelActor, LogOddsActor):
    def __init__(
        self,
        model_name: str,
        dtype: str,
        logical_actors: int,
        gpus_per_actor: int,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        self._init_single_model(
            model_name,
            dtype,
            logical_actors,
            gpus_per_actor,
            local_files_only,
            trust_remote_code,
        )


class DistributedMMLUActor(_DistributedSingleModelActor, MMLUActor):
    def __init__(
        self,
        model_name: str,
        dtype: str,
        logical_actors: int,
        gpus_per_actor: int,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        self._init_single_model(
            model_name,
            dtype,
            logical_actors,
            gpus_per_actor,
            local_files_only,
            trust_remote_code,
        )

class DistributedRescoreActor(_DistributedSingleModelActor, RescoreActor):
    def __init__(
        self,
        model_name: str,
        dtype: str,
        logical_actors: int,
        gpus_per_actor: int,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        self._init_single_model(
            model_name,
            dtype,
            logical_actors,
            gpus_per_actor,
            local_files_only,
            trust_remote_code,
        )



class _DistributedBehaviorActor(DistributedActorMixin):
    _distributed_model_attrs = (
        "_gen_model",
        "_gen_tok",
        "_judge_model",
        "_judge_tok",
    )

    def _init_behavior_models(
        self,
        generator_model_name: str,
        generator_dtype: str,
        judge_model_name: str,
        judge_dtype: str,
        logical_actors: int,
        gpus_per_actor: int,
        local_files_only: bool,
        trust_remote_code: bool,
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


class DistributedBehaviorActor(_DistributedBehaviorActor, BinaryBehaviorActor):
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
        self._init_behavior_models(
            generator_model_name,
            generator_dtype,
            judge_model_name,
            judge_dtype,
            logical_actors,
            gpus_per_actor,
            local_files_only,
            trust_remote_code,
        )
        self._judge_token_ids_10 = None


class DistributedContinuousBehaviorActor(
    _DistributedBehaviorActor,
    ContinuousBehaviorActor,
):
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
        self._init_behavior_models(
            generator_model_name,
            generator_dtype,
            judge_model_name,
            judge_dtype,
            logical_actors,
            gpus_per_actor,
            local_files_only,
            trust_remote_code,
        )
