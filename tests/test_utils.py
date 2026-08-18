import json
import tempfile
import unittest
from pathlib import Path

from utils.batching import chunked, chunked_with_bounds
from utils.data import (
    count_negative_prompts,
    discover_concepts,
    load_contexts_for_concept,
    read_jsonl_texts,
)
from utils.mmlu import extract_choice
from utils.naming import model_slug, slugify


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class SlugTests(unittest.TestCase):
    def test_slugify_normalizes_names(self):
        self.assertEqual(slugify(" Work / Life Balance! "), "work-life-balance")
        self.assertEqual(slugify("***"), "concept")

    def test_model_slug_includes_organization(self):
        self.assertEqual(model_slug("openai-community/gpt2"), "openai-community-gpt2")


class PromptDiscoveryTests(unittest.TestCase):
    def test_discovers_contrastive_and_model_specific_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "joy_positive.jsonl", [{"concept": "Joy", "text": "yes"}])
            write_jsonl(root / "joy_negative.jsonl", [{"text": "no"}])
            write_jsonl(
                root / "work-life-balance_positive.jsonl",
                [{"concept": "Work-life balance", "text": "yes"}],
            )
            write_jsonl(
                root / "work-life-balance_openai-community-gpt2_negative.jsonl",
                [{"text": "no"}],
            )
            write_jsonl(root / "positive-only_positive.jsonl", [{"text": "ignored"}])

            self.assertEqual(
                discover_concepts(root),
                [("joy", "Joy"), ("work-life-balance", "Work-life balance")],
            )

    def test_falls_back_to_slug_label_for_invalid_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "social-anxiety_positive.jsonl").write_text("not json\n", encoding="utf-8")
            write_jsonl(root / "social-anxiety_negative.jsonl", [{"text": "no"}])

            self.assertEqual(discover_concepts(root), [("social-anxiety", "social-anxiety")])


class ContextLoadingTests(unittest.TestCase):
    def test_loads_negatives_before_matching_positive_contexts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "contexts.jsonl"
            write_jsonl(
                path,
                [
                    {"negative": ["n1", "n2"]},
                    {"joy": ["p1"]},
                    {"unrelated": ["ignored"]},
                ],
            )

            contexts, source_lines = load_contexts_for_concept(str(path), "joy", "Joy")

            self.assertEqual(contexts, ["n1", "n2", "p1"])
            self.assertEqual(source_lines, [0, 0, 1])
            self.assertEqual(count_negative_prompts(str(path)), 2)

    def test_rejects_missing_positive_or_negative_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "contexts.jsonl"
            write_jsonl(path, [{"negative": ["n1"]}])

            with self.assertRaisesRegex(ValueError, "positive or negative contexts are empty"):
                load_contexts_for_concept(str(path), "joy", "Joy")

    def test_plain_text_contexts_remain_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "contexts.txt"
            path.write_text("first\n\nsecond\n", encoding="utf-8")

            self.assertEqual(
                load_contexts_for_concept(str(path), "unused", "Unused"),
                (["first", "second"], [-1, -1]),
            )


class SmallHelperTests(unittest.TestCase):
    def test_read_jsonl_texts_skips_bad_rows_and_honors_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text(
                '{"text": "first"}\nnot json\n{"text": "third"}\n',
                encoding="utf-8",
            )

            self.assertEqual(read_jsonl_texts(path), ["first", "third"])
            self.assertEqual(read_jsonl_texts(path, n_prompts=2), ["first"])

    def test_chunk_helpers_preserve_boundaries(self):
        self.assertEqual(list(chunked(range(5), 2)), [[0, 1], [2, 3], [4]])
        self.assertEqual(
            list(chunked_with_bounds(["a", "b", "c"], 2)),
            [(0, 2, ["a", "b"]), (2, 3, ["c"])],
        )

    def test_extract_choice_accepts_common_answer_formats(self):
        self.assertEqual(extract_choice("  b) because..."), "B")
        self.assertEqual(extract_choice("C) because..."), "C")
        self.assertIsNone(extract_choice("unknown"))


if __name__ == "__main__":
    unittest.main()
