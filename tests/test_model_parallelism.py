"""Test model placement, scheduling, and launcher defaults."""

import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

from utils.runtime.placement import GpuMemory, ModelEstimate, plan_actor_mesh
from actors.tasks.next_token_probs import ensure_full_vocab_logits
from utils.runtime.pool import group_bounds, leader_result
from experiments.runners import (
    plot_work_items,
    prompt_phase_jobs,
    prompt_phases,
)
from experiments.generate_next_token_probs import parse_args as parse_plot_args
from experiments.generate_concept_probs import parse_args as parse_concept_probs_args
from experiments.generate_concept_probs_continuous import (
    parse_args as parse_continuous_concept_probs_args,
)
from experiments.generate_cross_entropy import parse_args as parse_ce_args
from experiments.generate_log_odds import parse_args as parse_log_odds_args
from experiments.generate_mmlu import parse_args as parse_mmlu_args
from experiments.generate_prompts import parse_args as parse_prompt_args
from experiments.generate_steering_vectors import parse_args as parse_steering_args
from experiments.rescore_concept_probs_continuous import (
    parse_args as parse_rescore_args,
)


GIB = 1024**3


def estimate(*, weight_gib, tp=True, divisors=(1, 2, 4, 8)):
    """Build a model-memory estimate for planner tests."""
    return ModelEstimate(
        model_name="test/model",
        dtype="bfloat16",
        weight_bytes=int(weight_gib * GIB),
        largest_layer_bytes=1 * GIB,
        supports_tensor_parallel=tp,
        tensor_parallel_divisors=divisors,
    )


def gpus(count, free_gib=24):
    """Build equal GPU-memory records for planner tests."""
    return [
        GpuMemory(index=index, free_bytes=free_gib * GIB, total_bytes=free_gib * GIB)
        for index in range(count)
    ]


class UnifiedModelParallelismTests(unittest.TestCase):
    """Check shared placement and scheduling rules."""
    def test_automatic_model_parallelism_is_default(self):
        """Check that automatic model placement is the default."""
        with patch("sys.argv", ["generate_steering_vectors.py", "--models", "model"]):
            args = parse_steering_args()
        self.assertEqual(args.model_parallel_size, "auto")

    def test_new_launchers_use_automatic_placement_by_default(self):
        """Check that new launchers use automatic placement by default."""
        with patch(
            "sys.argv",
            [
                "generate_prompts.py",
                "--model_generating_concept",
                "model",
                "--concepts",
                "joy",
            ],
        ):
            prompt_args = parse_prompt_args()
        with patch("sys.argv", ["generate_next_token_probs.py", "--models", "model"]):
            plot_args = parse_plot_args()
        self.assertEqual(prompt_args.model_parallel_size, "auto")
        self.assertEqual(plot_args.model_parallel_size, "auto")

    def test_prompt_phases_cover_contrastive_and_classic_modes(self):
        """Check that prompt phases cover contrastive and classic modes."""
        contrastive = SimpleNamespace(
            contrastive=True,
            model_generating_concept="generator",
            models=["ignored"],
        )
        classic = SimpleNamespace(
            contrastive=False,
            model_generating_concept="generator",
            models=["target-a", "target-b"],
        )
        self.assertEqual(prompt_phases(contrastive), [("generator", "both", None)])
        self.assertEqual(
            prompt_phases(classic),
            [
                ("generator", "related", None),
                ("target-a", "unrelated", "target-a"),
                ("target-b", "unrelated", "target-b"),
            ],
        )
        self.assertEqual(
            prompt_phase_jobs(["joy"], "both"),
            [("joy", "both")],
        )

    def test_all_evaluation_launchers_expose_unified_topology(self):
        """Check that every command exposes the shared GPU options."""
        cases = [
            (
                parse_ce_args,
                ["ce.py", "--models", "model", "--eval_parquet", "eval.parquet"],
            ),
            (parse_log_odds_args, ["log.py", "--models", "model"]),
            (parse_mmlu_args, ["mmlu.py", "--models", "model"]),
            (
                parse_concept_probs_args,
                ["concept_probs.py", "--models", "model", "--judge_model", "judge"],
            ),
            (
                parse_continuous_concept_probs_args,
                ["continuous.py", "--models", "model", "--judge_model", "judge"],
            ),
            (
                parse_rescore_args,
                [
                    "rescore.py",
                    "--input_dir",
                    "input",
                    "--output_dir",
                    "output",
                    "--judge_model",
                    "judge",
                ],
            ),
        ]
        for parse, argv in cases:
            with self.subTest(parser=parse.__module__), patch("sys.argv", argv):
                args = parse()
            self.assertEqual(args.model_parallel_size, "auto")
            self.assertTrue(hasattr(args, "dtype"))
            self.assertTrue(hasattr(args, "plan_only"))

    def test_steered_evaluations_accept_exact_layer(self):
        """Check that steered evaluations accept one exact layer."""
        cases = [
            (
                parse_concept_probs_args,
                ["concept_probs.py", "--models", "model", "--judge_model", "judge"],
            ),
            (
                parse_ce_args,
                ["ce.py", "--models", "model", "--eval_parquet", "eval.parquet"],
            ),
            (parse_mmlu_args, ["mmlu.py", "--models", "model"]),
        ]
        for parse, argv in cases:
            with self.subTest(parser=parse.__module__), patch(
                "sys.argv", [*argv, "--layer", "13"]
            ):
                args = parse()
            self.assertEqual(args.layer, 13)

    def test_plot_jobs_split_by_context_and_layer(self):
        """Check that plot work is split by context and layer."""
        jobs = [("model", "joy", "Joy")]
        expected = [
            ("model", "joy", "Joy", 0, 39),
            ("model", "joy", "Joy", 0, 40),
            ("model", "joy", "Joy", 1, 39),
            ("model", "joy", "Joy", 1, 40),
        ]
        with patch(
            "experiments.runners.load_contexts_for_concept",
            return_value=(["negative", "positive"], [0, 1]),
        ):
            actual = plot_work_items(jobs, "contexts.jsonl", [39, 40])
        self.assertEqual(actual, expected)

    def test_mmlu_accepts_a_small_problem_subset(self):
        """Limit each MMLU task without replacing DeepEval at import time."""
        argv = [
            "mmlu.py",
            "--models",
            "model",
            "--max_problems_per_task",
            "8",
        ]
        with patch("sys.argv", argv):
            args = parse_mmlu_args()
        self.assertEqual(args.max_problems_per_task, 8)

    def test_leader_result_and_replicated_logits(self):
        """Check leader selection and full-vocabulary logits."""
        payload = {"ok": True}
        self.assertIs(leader_result({0: payload, 1: None}), payload)
        logits = torch.randn(2, 3, 7)
        self.assertIs(ensure_full_vocab_logits(logits, 7), logits)
        with self.assertRaisesRegex(RuntimeError, "Expected vocabulary width"):
            ensure_full_vocab_logits(logits, 8)

    def test_one_gpu_per_logical_actor_when_model_fits(self):
        """Check that fitting models use one GPU per actor."""
        plan = plan_actor_mesh(
            estimate(weight_gib=10),
            gpus(4),
            desired_actors=3,
            inference_headroom=1.2,
        )
        self.assertEqual(plan.gpus_per_actor, 1)
        self.assertEqual(plan.logical_actors, 3)
        self.assertEqual(plan.total_gpus, 3)
        self.assertFalse(plan.tensor_parallel)

    def test_rounds_up_to_compatible_tensor_parallel_size(self):
        """Check that placement chooses a compatible parallel size."""
        plan = plan_actor_mesh(
            estimate(weight_gib=50, divisors=(1, 2, 4, 8)),
            gpus(8),
            desired_actors=4,
            inference_headroom=1.2,
        )
        self.assertEqual(plan.gpus_per_actor, 4)
        self.assertEqual(plan.logical_actors, 2)
        self.assertEqual(plan.total_gpus, 8)
        self.assertTrue(plan.tensor_parallel)

    def test_qwen_72b_packs_two_two_gpu_actors_on_four_gpus(self):
        """Check that four GPUs can host two two-GPU Qwen 72B actors."""
        plan = plan_actor_mesh(
            estimate(weight_gib=135, divisors=(1, 2, 4)),
            gpus(4, free_gib=95),
            desired_actors=2,
            inference_headroom=1.2,
        )
        self.assertEqual(plan.gpus_per_actor, 2)
        self.assertEqual(plan.logical_actors, 2)
        self.assertEqual(plan.total_gpus, 4)

    def test_rejects_multi_gpu_model_without_native_tp_plan(self):
        """Reject multi-GPU models without a tensor-parallel plan."""
        with self.assertRaisesRegex(RuntimeError, "does not define base_model_tp_plan"):
            plan_actor_mesh(
                estimate(weight_gib=30, tp=False),
                gpus(2),
                desired_actors=1,
                inference_headroom=1.2,
            )

    def test_rejects_when_compatible_group_will_not_fit(self):
        """Reject a compatible GPU group when it cannot fit."""
        with self.assertRaisesRegex(RuntimeError, "no compatible TP size"):
            plan_actor_mesh(
                estimate(weight_gib=70, divisors=(1, 2, 4)),
                gpus(3),
                desired_actors=1,
                inference_headroom=1.2,
            )

    def test_logical_actor_groups_are_contiguous(self):
        """Check that each model gets adjacent GPU workers."""
        self.assertEqual(group_bounds(0, 4), (0, 4))
        self.assertEqual(group_bounds(2, 4), (8, 12))


if __name__ == "__main__":
    unittest.main()
