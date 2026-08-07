import unittest
from unittest.mock import patch

from actors.model_placement import GpuMemory, ModelEstimate, plan_actor_mesh
from experiments.dynamic_steering_vectors import group_bounds
from experiments.generate_steering_vectors import parse_args


GIB = 1024**3


def estimate(*, weight_gib, tp=True, divisors=(1, 2, 4, 8)):
    return ModelEstimate(
        model_name="test/model",
        dtype="bfloat16",
        weight_bytes=int(weight_gib * GIB),
        largest_layer_bytes=1 * GIB,
        supports_tensor_parallel=tp,
        tensor_parallel_divisors=divisors,
    )


def gpus(count, free_gib=24):
    return [
        GpuMemory(index=index, free_bytes=free_gib * GIB, total_bytes=free_gib * GIB)
        for index in range(count)
    ]


class DynamicModelParallelismTests(unittest.TestCase):
    def test_model_parallelism_is_disabled_by_default(self):
        with patch("sys.argv", ["generate_steering_vectors.py", "--models", "model"]):
            args = parse_args()
        self.assertEqual(args.model_parallel_size, "1")

    def test_one_gpu_per_logical_actor_when_model_fits(self):
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

    def test_rejects_multi_gpu_model_without_native_tp_plan(self):
        with self.assertRaisesRegex(RuntimeError, "does not define base_model_tp_plan"):
            plan_actor_mesh(
                estimate(weight_gib=30, tp=False),
                gpus(2),
                desired_actors=1,
                inference_headroom=1.2,
            )

    def test_rejects_when_compatible_group_will_not_fit(self):
        with self.assertRaisesRegex(RuntimeError, "no compatible TP size"):
            plan_actor_mesh(
                estimate(weight_gib=70, divisors=(1, 2, 4)),
                gpus(3),
                desired_actors=1,
                inference_headroom=1.2,
            )

    def test_logical_actor_groups_are_contiguous(self):
        self.assertEqual(group_bounds(0, 4), (0, 4))
        self.assertEqual(group_bounds(2, 4), (8, 12))


if __name__ == "__main__":
    unittest.main()
