import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from utils import batch_size as batch_size_utils


_GIB = 1024**3


class _FakeParameter:
    def __init__(self, device: str = "cuda:0") -> None:
        self.device = torch.device(device)

    def is_floating_point(self) -> bool:
        return True

    def element_size(self) -> int:
        return 2


class _FakeModel:
    def __init__(self, config, device: str = "cuda:0") -> None:
        self.config = config
        self.dtype = torch.float16
        self._parameter = _FakeParameter(device)

    def parameters(self):
        yield self._parameter


def _config(
    *,
    layers: int = 12,
    hidden: int = 768,
    intermediate: int = 3072,
    heads: int = 12,
    kv_heads: int = 12,
    head_dim: int = 64,
    vocab: int = 50_257,
    context: int = 2048,
):
    return SimpleNamespace(
        num_hidden_layers=layers,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        vocab_size=vocab,
        max_position_embeddings=context,
        _attn_implementation="sdpa",
    )


@contextmanager
def _mock_cuda_memory(free_bytes: int, total_bytes: int | None = None):
    total_bytes = free_bytes if total_bytes is None else total_bytes
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(batch_size_utils.torch.cuda, "is_available", return_value=True)
        )
        stack.enter_context(patch.object(batch_size_utils.torch.cuda, "synchronize"))
        stack.enter_context(patch.object(batch_size_utils.torch.cuda, "empty_cache"))
        stack.enter_context(
            patch.object(
                batch_size_utils.torch.cuda,
                "mem_get_info",
                return_value=(free_bytes, total_bytes),
            )
        )
        yield


class ModelShapeTests(unittest.TestCase):
    def test_reads_gpt_style_config_aliases(self):
        config = SimpleNamespace(
            n_layer=24,
            n_embd=1024,
            n_inner=4096,
            n_head=16,
            vocab_size=32_000,
            n_positions=4096,
            _attn_implementation="flash_attention_2",
        )

        shape = batch_size_utils._model_shape(_FakeModel(config), "float16")

        self.assertEqual(shape.layers, 24)
        self.assertEqual(shape.hidden, 1024)
        self.assertEqual(shape.intermediate, 4096)
        self.assertEqual(shape.heads, 16)
        self.assertEqual(shape.kv_heads, 16)
        self.assertEqual(shape.head_dim, 64)
        self.assertEqual(shape.vocab, 32_000)
        self.assertEqual(shape.context_limit, 4096)
        self.assertTrue(shape.fused_attention)

    def test_reads_nested_text_config_aliases_and_defaults(self):
        text_config = SimpleNamespace(
            num_layers=6,
            d_model=512,
            ffn_dim=1536,
            num_attention_heads=8,
            max_position_embeddings=1024,
        )
        config = SimpleNamespace(text_config=text_config, vocab_size=32_128)

        shape = batch_size_utils._model_shape(_FakeModel(config), "bfloat16")

        self.assertEqual(shape.layers, 6)
        self.assertEqual(shape.hidden, 512)
        self.assertEqual(shape.intermediate, 1536)
        self.assertEqual(shape.kv_heads, 8)
        self.assertEqual(shape.head_dim, 64)
        self.assertEqual(shape.vocab, 32_128)
        self.assertEqual(shape.dtype_bytes, 2)

    def test_rubric_includes_the_post_generation_scoring_forward(self):
        model = _FakeModel(_config())
        shape = batch_size_utils._model_shape(model, "float16")
        generation = batch_size_utils._sequence_bytes(
            model,
            shape,
            kind="generate",
            prompt_tokens=64,
            new_tokens=64,
            use_cache=True,
        )
        rubric = batch_size_utils._sequence_bytes(
            model,
            shape,
            kind="rubric",
            prompt_tokens=64,
            new_tokens=64,
            use_cache=True,
        )
        scoring = batch_size_utils._activation_bytes(shape, 128) + 72 * shape.vocab
        self.assertEqual(rubric, max(generation, scoring))

    def test_sdpa_dispatch_is_budgeted_for_quadratic_fallback(self):
        shape = batch_size_utils._model_shape(_FakeModel(_config()), "float16")

        self.assertFalse(shape.fused_attention)
        self.assertGreater(
            batch_size_utils._activation_bytes(shape, 512),
            batch_size_utils._activation_bytes(shape, 256) * 2,
        )

    def test_decode_attention_uses_the_full_key_length(self):
        shape = batch_size_utils._model_shape(_FakeModel(_config()), "float16")

        short = batch_size_utils._activation_bytes(shape, 1, 64)
        long = batch_size_utils._activation_bytes(shape, 1, 512)
        expected = (
            2 * shape.heads * (512 - 64) * max(shape.dtype_bytes, 4)
        )
        self.assertEqual(long - short, expected)

    def test_tp_forward_budgets_gather_and_concatenation_workspace(self):
        model = _FakeModel(_config())
        shape = batch_size_utils._model_shape(model, "float16")
        single = batch_size_utils._sequence_bytes(
            model,
            shape,
            kind="forward",
            prompt_tokens=64,
            new_tokens=0,
            use_cache=False,
        )
        parallel = batch_size_utils._sequence_bytes(
            model,
            shape,
            kind="forward",
            prompt_tokens=64,
            new_tokens=0,
            use_cache=False,
            tp_size=4,
        )

        full_logits = 64 * shape.vocab * shape.dtype_bytes
        expected_extra = full_logits + 2 * (full_logits // 4)
        self.assertEqual(parallel, single + expected_extra)


class CriticalBatchSizeTests(unittest.TestCase):
    def _estimate(self, model, **overrides):
        arguments = {
            "kind": "hidden",
            "prompt_tokens": 64,
            "dtype": "float16",
            "use_cache": True,
            "safety": 1.0,
            "reserve_fraction": 0.0,
            "minimum_reserve_bytes": 0,
        }
        arguments.update(overrides)
        return batch_size_utils.critical_batch_size(model, **arguments)

    def test_capacity_decreases_with_sequence_and_model_size(self):
        small_model = _FakeModel(_config())
        large_model = _FakeModel(
            _config(
                layers=32,
                hidden=4096,
                intermediate=11_008,
                heads=32,
                kv_heads=8,
                head_dim=128,
            )
        )
        with _mock_cuda_memory(8 * _GIB):
            short = self._estimate(small_model, prompt_tokens=64)
            long = self._estimate(small_model, prompt_tokens=512)
            large = self._estimate(large_model, prompt_tokens=64)

        self.assertGreater(long.bytes_per_sequence, short.bytes_per_sequence)
        self.assertLess(long.critical_batch_size, short.critical_batch_size)
        self.assertGreater(large.bytes_per_sequence, short.bytes_per_sequence)
        self.assertLess(large.critical_batch_size, short.critical_batch_size)

    def test_eligible_sdpa_uses_linear_memory_estimate(self):
        model = _FakeModel(_config())
        with _mock_cuda_memory(8 * _GIB), patch.object(
            batch_size_utils,
            "_sdpa_uses_linear_memory",
            return_value=True,
        ):
            fused = self._estimate(model, kind="generate", new_tokens=8)
        with _mock_cuda_memory(8 * _GIB), patch.object(
            batch_size_utils,
            "_sdpa_uses_linear_memory",
            return_value=False,
        ):
            fallback = self._estimate(model, kind="generate", new_tokens=8)

        self.assertLess(fused.bytes_per_sequence, fallback.bytes_per_sequence)

    def test_capacity_increases_with_available_memory(self):
        model = _FakeModel(_config())
        with _mock_cuda_memory(2 * _GIB):
            low_memory = self._estimate(model)
        with _mock_cuda_memory(8 * _GIB):
            high_memory = self._estimate(model)

        self.assertGreater(
            high_memory.critical_batch_size, low_memory.critical_batch_size
        )

    def test_multiplier_limit_and_requested_are_applied_in_logical_units(self):
        model = _FakeModel(_config())
        with _mock_cuda_memory(10_000), patch.object(
            batch_size_utils, "_sequence_bytes", return_value=100
        ):
            estimate = self._estimate(
                model,
                multiplier=4,
                limit=20,
                requested=7,
            )
            clamped = self._estimate(model, multiplier=4, requested=30)
            automatic = self._estimate(model, multiplier=4)

        self.assertEqual(estimate.critical_batch_size, 20)
        self.assertEqual(estimate.batch_size, 7)
        self.assertEqual(estimate.effective_batch_size, 28)
        self.assertEqual(clamped.batch_size, 25)
        self.assertEqual(automatic.batch_size, 25)
        self.assertEqual(automatic.effective_batch_size, 100)

    def test_raises_when_one_logical_item_does_not_fit(self):
        model = _FakeModel(_config())
        with _mock_cuda_memory(999), patch.object(
            batch_size_utils, "_sequence_bytes", return_value=500
        ):
            with self.assertRaisesRegex(
                RuntimeError, "One logical batch item.*1000.0 MiB|One logical batch item"
            ):
                self._estimate(model, multiplier=2)

    def test_rejects_workload_beyond_model_context(self):
        model = _FakeModel(_config(context=128))
        with patch.object(
            batch_size_utils.torch.cuda, "is_available", return_value=True
        ):
            with self.assertRaisesRegex(ValueError, "129 tokens.*context limit is 128"):
                self._estimate(
                    model,
                    kind="generate",
                    prompt_tokens=96,
                    new_tokens=33,
                )

    def test_cpu_fallback_honors_requested_and_uses_safe_auto_default(self):
        model = _FakeModel(_config(), device="cpu")
        with patch.object(
            batch_size_utils.torch.cuda, "is_available", return_value=False
        ):
            requested = self._estimate(model, multiplier=3, requested=5, limit=4)
            automatic = self._estimate(model, multiplier=3, limit=9)

        self.assertEqual(requested.batch_size, 4)
        self.assertEqual(requested.effective_batch_size, 12)
        self.assertEqual(automatic.batch_size, 1)
        self.assertEqual(automatic.effective_batch_size, 3)

    def test_rubric_rejects_combined_context_overflow(self):
        model = _FakeModel(_config(context=128))
        with patch.object(
            batch_size_utils.torch.cuda, "is_available", return_value=True
        ):
            with self.assertRaisesRegex(ValueError, "129 tokens.*context limit is 128"):
                self._estimate(
                    model,
                    kind="rubric",
                    prompt_tokens=96,
                    new_tokens=33,
                )

    def test_tensor_parallel_minimum_caps_the_local_capacity(self):
        model = _FakeModel(_config())
        mesh = object()
        with _mock_cuda_memory(1_000), patch.object(
            batch_size_utils, "_sequence_bytes", return_value=100
        ), patch.object(batch_size_utils, "_group_min", return_value=3) as group_min:
            estimate = self._estimate(model, tp_mesh=mesh)

        self.assertEqual(estimate.critical_batch_size, 3)
        self.assertEqual(estimate.batch_size, 3)
        self.assertEqual(group_min.call_args.args[0], 10)
        self.assertIs(group_min.call_args.args[2], mesh)


class DistributedMinimumTests(unittest.TestCase):
    def test_group_min_uses_all_reduce_min_for_a_mesh(self):
        group = object()
        mesh = SimpleNamespace(get_group=lambda: group)

        def reduce_to_three(tensor, *, op, group):
            self.assertIs(op, batch_size_utils.dist.ReduceOp.MIN)
            tensor.fill_(3)

        with patch.object(
            batch_size_utils.dist, "is_available", return_value=True
        ), patch.object(
            batch_size_utils.dist, "is_initialized", return_value=True
        ), patch.object(
            batch_size_utils.dist, "all_reduce", side_effect=reduce_to_three
        ) as all_reduce:
            result = batch_size_utils._group_min(7, torch.device("cpu"), mesh)

        self.assertEqual(result, 3)
        self.assertIs(all_reduce.call_args.kwargs["group"], group)



class FactorBatchSizeTests(unittest.TestCase):
    def test_maximizes_capacity_with_dimension_caps(self):
        self.assertEqual(
            batch_size_utils.factor_batch_size(
                100,
                first_limit=8,
                second_limit=20,
                first_hint=1,
                second_hint=4,
            ),
            (5, 20),
        )
        self.assertEqual(
            batch_size_utils.factor_batch_size(
                17,
                first_limit=4,
                second_limit=10,
                first_hint=1,
                second_hint=4,
            ),
            (2, 8),
        )
        self.assertEqual(
            batch_size_utils.factor_batch_size(
                64,
                first_limit=8,
                second_limit=8,
                first_hint=1,
                second_hint=1,
            ),
            (8, 8),
        )

    def test_rejects_nonpositive_capacity_limits_and_hints(self):
        invalid = (
            {"capacity": 0, "first_limit": 1, "second_limit": 1},
            {"capacity": 1, "first_limit": 0, "second_limit": 1},
            {
                "capacity": 1,
                "first_limit": 1,
                "second_limit": 1,
                "first_hint": 0,
            },
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    batch_size_utils.factor_batch_size(**arguments)


if __name__ == "__main__":
    unittest.main()
