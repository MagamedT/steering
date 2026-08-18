"""Run the full experiment pipeline with small CPU-only stand-ins."""

import asyncio
import contextlib
import json
import sys
import tempfile
import types
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)

from actors.tasks import concept_probs as concept_probs_actor
from actors.tasks import concept_probs_continuous as concept_probs_continuous_actor
from actors.tasks import rubric_judge
from actors.tasks import rescore as rescore_actor
from actors.tasks import cross_entropy as cross_entropy_actor
from actors.tasks import log_odds as log_odds_actor
from actors.tasks import mmlu as mmlu_actor
from actors.tasks import prompts as prompts_actor
from actors.tasks import next_token_probs as next_token_probs_actor
from actors.tasks import steering as distributed_steering_actor
from utils.runtime.placement import GpuMemory
from utils.runtime import pool as distributed_runtime
from utils.naming import model_slug
from experiments import generate_concept_probs
from experiments import generate_concept_probs_continuous
from experiments import generate_cross_entropy
from experiments import generate_eval_dataset
from experiments import generate_log_odds
from experiments import generate_mmlu
from experiments import generate_next_token_probs
from experiments import generate_prompts
from experiments import generate_steering_vectors
from experiments import rescore_concept_probs_continuous


def make_tiny_model(path: Path) -> None:
    """Build a tiny local model and tokenizer for CPU tests."""
    vocab = {
        "[PAD]": 0,
        "[BOS]": 1,
        "[EOS]": 2,
        "[UNK]": 3,
        "joy": 4,
        "sad": 5,
        "good": 6,
        "bad": 7,
        "A": 8,
        "B": 9,
        "C": 10,
        "D": 11,
        "0": 12,
        "1": 13,
        "day": 14,
        "night": 15,
    }
    raw_tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    raw_tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(path)

    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocab),
            n_layer=1,
            n_head=1,
            n_embd=8,
            n_positions=64,
            n_ctx=64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    model.save_pretrained(path)


class LocalCall:
    """Copy the call pattern used by Monarch during CPU tests."""
    def __init__(self, actor, name):
        self.actor = actor
        self.name = name

    async def _invoke(self, *args, **kwargs):
        """Run one task on every local worker."""
        endpoint = getattr(type(self.actor), self.name)
        return await endpoint._method(self.actor, *args, **kwargs)

    async def call_one(self, *args, **kwargs):
        """Run one task and return its first result."""
        return await self._invoke(*args, **kwargs)

    async def call(self, *args, **kwargs):
        """Run one task on the local model worker."""
        return {0: await self._invoke(*args, **kwargs)}


class LocalActor:
    """Wrap one local actor for CPU tests."""
    def __init__(self, actor):
        self.actor = actor

    def __getattr__(self, name):
        return LocalCall(self.actor, name)


class LocalActorGroup:
    """Expose a slice of local actors as one logical group."""
    def __init__(self, actor):
        self.actor = actor

    def slice(self, **_rank):
        """Return the requested local actor group."""
        return LocalActor(self.actor)


    def __getattr__(self, name):
        return LocalCall(self.actor, name)

class LocalMesh:
    """Provide the process-mesh methods used by the runners."""
    def to_table(self):
        """Return a short description of the local mesh."""
        return "cpu worker"

    def spawn(self, _name, actor_class, *args):
        """Create local actor instances."""
        actor = object.__new__(actor_class)
        actor_class.__init__(actor, *args)
        return LocalActorGroup(actor)


    async def stop(self):
        """Stop the local mesh."""
        return None

class LocalHost:
    """Create local workers for CPU tests."""
    def spawn_procs(self, **_kwargs):
        """Create the requested number of local workers."""
        return LocalMesh()


def small_config(config_class, **small_values):
    """Build a fast configuration for CPU tests."""
    def build(**values):
        return config_class(**{**small_values, **values})

    return build

def runtime_args(**values):
    """Build default runner arguments for CPU tests."""
    defaults = {
        "model_parallel_size": "1",
        "dtype": "float32",
        "gpu_utilization": 0.9,
        "inference_headroom": 1.2,
        "local_files_only": True,
        "trust_remote_code": False,
        "plan_only": False,
        "dim": "gpu",
        "max_gpus": 1,
        "batch_size": 1,
        "max_length": 16,
        "max_new_tokens": 2,
        "temperature": 1.0,
        "top_k": 2,
        "top_p": 1.0,
        "alpha_start": -1.0,
        "alpha_end": 1.0,
        "alpha_steps": 3,
        "block_per_pass": 0,
        "progress_every": 1,
        "n_positive": None,
        "n_negative": None,
        "apply_last_token_only": False,
        "normalize": False,
        "samples_per_context": 1,
        "context_batch_size": 1,
        "judge_batch_size": 1,
        "judge_max_new_tokens": 2,
        "layer": None,
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)



@contextlib.contextmanager
def cpu_only(model_path: Path):
    """Redirect CUDA operations to CPU during the test."""
    original_to = torch.Tensor.to
    original_tensor = torch.tensor
    original_ones_like = torch.ones_like
    original_autocast = torch.autocast

    def cpu_to(tensor, *args, **kwargs):
        if args and str(args[0]).startswith("cuda"):
            args = (torch.device("cpu"), *args[1:])
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs["device"] = torch.device("cpu")
        return original_to(tensor, *args, **kwargs)

    def cpu_tensor(*args, **kwargs):
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs["device"] = torch.device("cpu")
        return original_tensor(*args, **kwargs)

    def cpu_ones_like(*args, **kwargs):
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs["device"] = torch.device("cpu")
        return original_ones_like(*args, **kwargs)

    def cpu_autocast(device_type, *args, **kwargs):
        if device_type == "cuda":
            return contextlib.nullcontext()
        return original_autocast(device_type, *args, **kwargs)

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(torch.Tensor, "to", cpu_to))
        stack.enter_context(patch.object(torch, "tensor", cpu_tensor))
        stack.enter_context(patch.object(torch, "ones_like", cpu_ones_like))
        stack.enter_context(patch.object(torch, "autocast", cpu_autocast))
        stack.enter_context(patch.object(torch, "set_default_device", lambda _device: None))
        stack.enter_context(patch.object(torch.cuda, "device_count", return_value=1))
        stack.enter_context(patch.object(torch.cuda, "is_available", return_value=False))
        stack.enter_context(patch.object(torch.cuda, "empty_cache", return_value=None))
        stack.enter_context(patch.object(torch.cuda, "manual_seed_all", return_value=None))

        stack.enter_context(
            patch.object(distributed_runtime, "this_host", return_value=LocalHost())
        )
        stack.enter_context(
            patch.object(
                distributed_runtime,
                "discover_gpu_memory",
                return_value=[GpuMemory(0, 16 * 1024**3, 16 * 1024**3)],
            )
        )

        async def fake_setup(_mesh):
            return None

        stack.enter_context(
            patch.object(
                distributed_runtime, "setup_torch_elastic_env_async", fake_setup
            )
        )

        stack.enter_context(
            patch.object(
                generate_prompts,
                "GenConfig",
                small_config(prompts_actor.GenConfig, max_new_tokens=2),
            )
        )
        stack.enter_context(
            patch.object(
                generate_steering_vectors,
                "SteeringConfig",
                small_config(distributed_steering_actor.SteeringConfig, batch_size=1, max_length=16),
            )
        )
        stack.enter_context(
            patch.object(
                generate_next_token_probs,
                "TokenPlotConfig",
                small_config(
                    next_token_probs_actor.TokenPlotConfig,
                    alpha_start=-1,
                    alpha_end=1,
                    alpha_steps=3,
                    batch_size=3,
                    max_length=16,
                    top_k=2,
                ),
            )
        )
        stack.enter_context(
            patch.object(
                generate_concept_probs,
                "ConceptProbsConfig",
                small_config(
                    concept_probs_actor.ConceptProbsConfig,
                    generator_dtype="float32",
                    judge_dtype="float32",
                    alpha_start=-1,
                    alpha_end=1,
                    alpha_steps=3,
                    n_samples_per_context=1,
                    gen_context_batch_size=1,
                    max_prompt_length=16,
                    max_new_tokens=1,
                    judge_max_prompt_length=16,
                ),
            )
        )
        stack.enter_context(
            patch.object(
                generate_concept_probs_continuous,
                "ConceptProbsConfig",
                small_config(
                    concept_probs_continuous_actor.ConceptProbsConfig,
                    generator_dtype="float32",
                    judge_dtype="float32",
                    alpha_start=-1,
                    alpha_end=1,
                    alpha_steps=3,
                    n_samples_per_context=1,
                    gen_context_batch_size=1,
                    max_prompt_length=16,
                    max_new_tokens=1,
                    judge_max_prompt_length=16,
                    judge_batch_size=1,
                    judge_max_new_tokens=2,
                ),
            )
        )
        stack.enter_context(
            patch.object(
                concept_probs_continuous_actor.ConceptProbsActor,
                "_judge_completion_scores",
                lambda _self, samples, concept, cfg, instruction=None: torch.full(
                    (len(samples),),
                    0.5,
                    dtype=torch.float32,
                ),
            )
        )
        stack.enter_context(
            patch.object(
                generate_log_odds,
                "LogOddsConfig",
                small_config(log_odds_actor.LogOddsConfig, batch_size=1, max_length=16, top_k=2),
            )
        )
        stack.enter_context(
            patch.object(
                generate_cross_entropy,
                "CrossEntropyPlotConfig",
                small_config(
                    cross_entropy_actor.CrossEntropyPlotConfig,
                    alpha_start=-1,
                    alpha_end=1,
                    alpha_steps=3,
                    alpha_batch_size=3,
                    eval_seq_len=2,
                    eval_stride=2,
                    eval_max_blocks=1,
                    eval_batch_size=1,
                    max_doc_tokens=16,
                ),
            )
        )
        stack.enter_context(
            patch.object(
                generate_mmlu,
                "MMLUEvalConfig",
                small_config(
                    mmlu_actor.MMLUEvalConfig,
                    alpha_start=0,
                    alpha_end=0,
                    alpha_steps=1,
                    n_shots=0,
                    max_new_tokens=1,
                    batch_size=1,
                    use_chat_template=False,
                ),
            )
        )
        yield


@contextlib.contextmanager
def fake_deepeval():
    """Provide small DeepEval stand-ins for the CPU test."""
    class MMLUTask(Enum):
        """Minimal MMLU task enum used by the test."""
        HIGH_SCHOOL_COMPUTER_SCIENCE = "high_school_computer_science"

    class DeepEvalBaseLLM:
        """Minimal base class used by the fake MMLU runner."""
        pass

    class MMLU:
        """Small MMLU stand-in that records generated answers."""
        def __init__(self, tasks, n_shots):
            self.tasks = tasks
            self.n_shots = n_shots
            self.overall_score = None
            self.task_scores = None

        def evaluate(self, model, batch_size):
            """Generate one answer and expose fixed benchmark scores."""
            model.generate("A B C D")
            self.overall_score = 1.0
            self.task_scores = {task.value: 1.0 for task in self.tasks}
            return 1.0

    modules = {
        "deepeval": types.ModuleType("deepeval"),
        "deepeval.benchmarks": types.ModuleType("deepeval.benchmarks"),
        "deepeval.benchmarks.mmlu": types.ModuleType("deepeval.benchmarks.mmlu"),
        "deepeval.benchmarks.mmlu.task": types.ModuleType("deepeval.benchmarks.mmlu.task"),
        "deepeval.models": types.ModuleType("deepeval.models"),
        "deepeval.models.base_model": types.ModuleType("deepeval.models.base_model"),
    }
    modules["deepeval.benchmarks"].MMLU = MMLU
    modules["deepeval.benchmarks.mmlu.task"].MMLUTask = MMLUTask
    modules["deepeval.models"].DeepEvalBaseLLM = DeepEvalBaseLLM
    modules["deepeval.models.base_model"].DeepEvalBaseLLM = DeepEvalBaseLLM
    with patch.dict(sys.modules, modules):
        yield


class AllGeneratorsCpuTest(unittest.IsolatedAsyncioTestCase):
    """Exercise every generator without a GPU or model download."""
    def test_continuous_pair_probability(self):
        """Check that true and false logits produce the expected probability."""
        pair = rubric_judge.pair_probability
        self.assertAlmostEqual(pair(0.0, 0.0), 0.5)
        self.assertAlmostEqual(pair(1.0986122886681098, 0.0), 0.75, places=6)
        result = pair(torch.tensor([0.0, -1000.0]), torch.tensor([0.0, 1000.0]))
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, torch.tensor([0.5, 0.0]))

    async def test_all_generators(self):
        """Run every experiment with tiny local models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tiny_model = root / "tiny-model"
            tiny_model.mkdir()
            make_tiny_model(tiny_model)
            model_name = str(tiny_model)

            prompts = root / "prompts"
            steering = root / "steering"
            contexts = root / "contexts.jsonl"
            contexts.write_text(
                json.dumps({"negative": ["bad night"]}) + "\n"
                + json.dumps({"joy": ["good day"]}) + "\n",
                encoding="utf-8",
            )
            parquet = root / "eval.parquet"
            pq.write_table(pa.table({"text": ["good day joy good"]}), parquet)

            with cpu_only(tiny_model):
                await generate_prompts.main_async(runtime_args(
                    model_generating_concept=model_name, models=[model_name], concepts=["joy"],
                    out_dir=str(prompts), seed=0, dim="gpu", max_gpus=1,
                    contrastive=False, n_related=2, n_unrelated=2, batch_size=1,
                ))
                for path in prompts.glob("*.jsonl"):
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    self.assertEqual(len(rows), 2)
                    self.assertTrue(all(row["text"].strip() for row in rows))

                await generate_steering_vectors.main_async(runtime_args(
                    models=[model_name], in_dir=str(prompts), save_dir=str(steering),
                    layers=["0"], layer_path=None, pairing="product", dim="gpu",
                    max_gpus=1, seed=0, contrastive=False,
                ))

                slug = model_slug(model_name)
                vector_path = steering / slug / "joy" / "layer_0.pt"
                vector = torch.load(vector_path, map_location="cpu")["steering_vector"]
                self.assertTrue(torch.isfinite(vector).all())

                plot_out = root / "plot"
                await generate_next_token_probs.main_async(runtime_args(
                    models=[model_name], steer_dir=str(steering), contexts_file=str(contexts),
                    out_dir=str(plot_out), layers=["0"], layer_path=None,
                    dim="gpu", max_gpus=1, seed=0,
                ))
                plot_path = next(plot_out.rglob("*.npz"), None)
                self.assertIsNotNone(plot_path)
                with np.load(plot_path, allow_pickle=True) as plot_data:
                    self.assertEqual(plot_data["probs_alphamax"].shape, (3, 2))
                    self.assertEqual(plot_data["probs_alphamin"].shape, (3, 2))
                    self.assertTrue(
                        np.all((0.0 <= plot_data["probs_alphamax"]) &
                               (plot_data["probs_alphamax"] <= 1.0))
                    )

                concept_probs_out = root / "concept_probs"
                await generate_concept_probs.main_async(runtime_args(
                    models=[model_name], judge_model=model_name, steer_dir=str(steering),
                    contexts_file=str(contexts), out_dir=str(concept_probs_out), layers=1,
                    layer_path=None, dim="gpu", max_gpus=1,
                ))
                self.assertTrue(any(concept_probs_out.rglob("*.npz")))

                continuous_out = root / "concept_probs_continuous"
                await generate_concept_probs_continuous.main_async(runtime_args(
                    models=[model_name], judge_model=model_name, steer_dir=str(steering),
                    contexts_file=str(contexts), out_dir=str(continuous_out), layers=1,
                    layer_path=None, dim="gpu", max_gpus=1, alpha_start=-1,
                    alpha_end=1, alpha_steps=3, seed=0,
                ))
                continuous_path = next(continuous_out.rglob("*.npz"))
                with np.load(continuous_path, allow_pickle=True) as continuous_data:
                    scores = continuous_data["completion_concept_scores_by_ctx"]
                    texts = continuous_data["completion_texts_by_ctx"]
                    self.assertEqual(texts.shape, scores.shape)
                    self.assertTrue(all(isinstance(text, str) for text in texts.flat))
                    self.assertTrue(np.all((0.0 <= scores) & (scores <= 1.0)))
                    self.assertTrue(np.allclose(scores, 0.5))

                rescore_batch_sizes = []

                def fake_rescore_details(
                    _tokenizer,
                    _model,
                    samples,
                    _concept,
                    _cfg,
                    **_kwargs,
                ):
                    count = len(samples)
                    rescore_batch_sizes.append(count)
                    values = torch.full((count,), 0.5, dtype=torch.float32)
                    return rescore_actor.RubricJudgeDetails(
                        scores=values,
                        raw_json=tuple(
                            '{"explanation_1":"test","criteria_met_1":true}'
                            for _ in samples
                        ),
                        criteria_met=(True,) * count,
                        true_logits=torch.zeros(count),
                        false_logits=torch.zeros(count),
                        pair_mass=torch.ones(count),
                        verdict_token_ids=(0,) * count,
                        true_token_ids=(0,) * count,
                        false_token_ids=(1,) * count,
                        verdict_positions=(0,) * count,
                    )

                rescore_out = root / "concept_probs_rescored"
                with patch.object(
                    rescore_actor,
                    "rubric_completion_scores",
                    fake_rescore_details,
                ):
                    await rescore_concept_probs_continuous.main_async(runtime_args(
                        input_dir=str(continuous_out), output_dir=str(rescore_out),
                        judge_model=model_name,
                    ))
                self.assertTrue(any(rescore_out.rglob("*.npz")))
                self.assertTrue(rescore_batch_sizes)
                self.assertEqual(
                    set(rescore_batch_sizes),
                    {scores.shape[-1]},
                )

                log_out = root / "log_odds"
                await generate_log_odds.main_async(runtime_args(
                    models=[model_name], prompts_dir=str(prompts), out_dir=str(log_out),
                    concepts=None, dim="gpu", max_gpus=1,
                ))
                self.assertTrue(any(log_out.rglob("*.npz")))

                cross_out = root / "cross_entropy"
                await generate_cross_entropy.main_async(runtime_args(
                    models=[model_name], steer_dir=str(steering), eval_parquet=str(parquet),
                    out_dir=str(cross_out), layers=1, layer_path=None,
                    seed=0, dim="gpu", max_gpus=1,
                ))
                self.assertTrue(any(cross_out.rglob("*.npz")))

                mmlu_out = root / "mmlu"
                with fake_deepeval():
                    await generate_mmlu.main_async(runtime_args(
                        models=[model_name], steer_dir=str(steering), out_dir=str(mmlu_out),
                        tasks=["HIGH_SCHOOL_COMPUTER_SCIENCE"], layers=1,
                        layer_path=None, seed=0, dim="gpu", max_gpus=1,
                    ))
                self.assertTrue(any(mmlu_out.rglob("*.json")))

            download_out = root / "download"
            remote_file = "sample-10BT/train/part.parquet"

            class FakeApi:
                """Return a fixed dataset file list for download tests."""
                def list_repo_files(self, **_kwargs):
                    """Return the fixed dataset file list."""
                    return [remote_file]

            def fake_download(**kwargs):
                path = Path(kwargs["local_dir"]) / kwargs["filename"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                return str(path)

            argv = [
                "generate_eval_dataset.py", "--dataset", "fake/data", "--out_dir", str(download_out)
            ]
            with patch.object(sys, "argv", argv), \
                 patch.object(generate_eval_dataset, "HfApi", return_value=FakeApi()), \
                 patch.object(generate_eval_dataset, "hf_hub_download", side_effect=fake_download):
                generate_eval_dataset.main()
            self.assertTrue((download_out / remote_file).exists())


if __name__ == "__main__":
    unittest.main()
