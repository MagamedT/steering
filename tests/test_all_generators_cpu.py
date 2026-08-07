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
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)

from actors import concept_probs_actor
from actors import concept_probs_continuous_actor
from actors import rubric_judge
from actors import cross_entropy_actor
from actors import log_odds_actor
from actors import mmlu_actor
from actors import prompts_actor
from actors import next_token_probs_actor
from actors import steering_vector_actor
from actors.utils import model_slug
from experiments import generate_concept_probs
from experiments import generate_concept_probs_continuous
from experiments import generate_cross_entropy
from experiments import generate_eval_dataset
from experiments import generate_log_odds
from experiments import generate_mmlu
from experiments import generate_next_token_probs
from experiments import generate_prompts
from experiments import generate_steering_vectors


def make_tiny_model(path: Path) -> None:
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
    def __init__(self, actor, name):
        self.actor = actor
        self.name = name

    async def call_one(self, *args, **kwargs):
        endpoint = type(self.actor).__dict__[self.name]
        return await endpoint._method(self.actor, *args, **kwargs)


class LocalActor:
    def __init__(self, actor):
        self.actor = actor

    def __getattr__(self, name):
        return LocalCall(self.actor, name)


class LocalActorGroup:
    def __init__(self, actor):
        self.actor = actor

    def slice(self, **_rank):
        return LocalActor(self.actor)


class LocalMesh:
    def to_table(self):
        return "cpu worker"

    def spawn(self, _name, actor_class, *args):
        actor = object.__new__(actor_class)
        actor_class.__init__(actor, *args)
        return LocalActorGroup(actor)


class LocalHost:
    def spawn_procs(self, **_kwargs):
        return LocalMesh()


def small_config(config_class, **small_values):
    def build(**values):
        return config_class(**{**small_values, **values})

    return build


@contextlib.contextmanager
def cpu_only(model_path: Path):
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

    def load_tiny(_model_name, _dtype="float32", **_kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        model.eval()
        return tokenizer, model

    actor_modules = (
        prompts_actor,
        steering_vector_actor,
        next_token_probs_actor,
        concept_probs_actor,
        concept_probs_continuous_actor,
        log_odds_actor,
        cross_entropy_actor,
        mmlu_actor,
    )
    launcher_modules = (
        generate_prompts,
        generate_steering_vectors,
        generate_next_token_probs,
        generate_concept_probs,
        generate_concept_probs_continuous,
        generate_log_odds,
        generate_cross_entropy,
        generate_mmlu,
    )

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

        for module in actor_modules:
            stack.enter_context(patch.object(module, "load_model_and_tokenizer", load_tiny))
        for module in launcher_modules:
            stack.enter_context(patch.object(module, "this_host", return_value=LocalHost()))

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
                small_config(steering_vector_actor.SteeringConfig, batch_size=1, max_length=16),
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
                "BehaviorConfig",
                small_config(
                    concept_probs_actor.BehaviorConfig,
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
                "BehaviorConfig",
                small_config(
                    concept_probs_continuous_actor.BehaviorConfig,
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
                concept_probs_continuous_actor.BehaviorActor,
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
    class MMLUTask(Enum):
        HIGH_SCHOOL_COMPUTER_SCIENCE = "high_school_computer_science"

    class DeepEvalBaseLLM:
        pass

    class MMLU:
        def __init__(self, tasks, n_shots):
            self.tasks = tasks
            self.n_shots = n_shots
            self.overall_score = None
            self.task_scores = None

        def evaluate(self, model, batch_size):
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
    def test_continuous_pair_probability(self):
        pair = rubric_judge.pair_probability
        self.assertAlmostEqual(pair(0.0, 0.0), 0.5)
        self.assertAlmostEqual(pair(1.0986122886681098, 0.0), 0.75, places=6)
        result = pair(torch.tensor([0.0, -1000.0]), torch.tensor([0.0, 1000.0]))
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, torch.tensor([0.5, 0.0]))

    async def test_all_generators(self):
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
                await generate_prompts.main_async(SimpleNamespace(
                    model_generating_concept=model_name, models=[model_name], concepts=["joy"],
                    out_dir=str(prompts), seed=0, dim="gpu", max_gpus=1,
                    contrastive=False, n_related=2, n_unrelated=2, batch_size=1,
                ))
                for path in prompts.glob("*.jsonl"):
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    self.assertEqual(len(rows), 2)
                    self.assertTrue(all(row["text"].strip() for row in rows))

                await generate_steering_vectors.main_async(SimpleNamespace(
                    models=[model_name], in_dir=str(prompts), save_dir=str(steering),
                    layers=["0"], layer_path=None, pairing="product", dim="gpu",
                    max_gpus=1, seed=0, contrastive=False,
                ))

                slug = model_slug(model_name)
                vector_path = steering / slug / "joy" / "layer_0.pt"
                vector = torch.load(vector_path, map_location="cpu")["steering_vector"]
                self.assertTrue(torch.isfinite(vector).all())

                plot_out = root / "plot"
                await generate_next_token_probs.main_async(SimpleNamespace(
                    models=[model_name], steer_dir=str(steering), contexts_file=str(contexts),
                    out_dir=str(plot_out), layers=["0"], layer_path=None,
                    dim="gpu", max_gpus=1, seed=0,
                ))
                self.assertTrue(any(plot_out.rglob("*.npz")))

                behavior_out = root / "behavior"
                await generate_concept_probs.main_async(SimpleNamespace(
                    models=[model_name], judge_model=model_name, steer_dir=str(steering),
                    contexts_file=str(contexts), out_dir=str(behavior_out), layers=1,
                    layer_path=None, dim="gpu", max_gpus=1,
                ))
                self.assertTrue(any(behavior_out.rglob("*.npz")))

                continuous_out = root / "behavior_continuous"
                await generate_concept_probs_continuous.main_async(SimpleNamespace(
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

                log_out = root / "log_odds"
                await generate_log_odds.main_async(SimpleNamespace(
                    models=[model_name], prompts_dir=str(prompts), out_dir=str(log_out),
                    concepts=None, dim="gpu", max_gpus=1,
                ))
                self.assertTrue(any(log_out.rglob("*.npz")))

                cross_out = root / "cross_entropy"
                await generate_cross_entropy.main_async(SimpleNamespace(
                    models=[model_name], steer_dir=str(steering), eval_parquet=str(parquet),
                    out_dir=str(cross_out), layers=1, layer_path=None,
                    seed=0, dim="gpu", max_gpus=1,
                ))
                self.assertTrue(any(cross_out.rglob("*.npz")))

                mmlu_out = root / "mmlu"
                with fake_deepeval():
                    await generate_mmlu.main_async(SimpleNamespace(
                        models=[model_name], steer_dir=str(steering), out_dir=str(mmlu_out),
                        tasks=["HIGH_SCHOOL_COMPUTER_SCIENCE"], layers=1,
                        layer_path=None, seed=0, dim="gpu", max_gpus=1,
                    ))
                self.assertTrue(any(mmlu_out.rglob("*.json")))

            download_out = root / "download"
            remote_file = "sample-10BT/train/part.parquet"

            class FakeApi:
                def list_repo_files(self, **_kwargs):
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
