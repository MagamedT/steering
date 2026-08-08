"""Check plotting for every saved result format."""

import json
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.plot import (
    detect_plot_kind,
    plot_artifact,
    plot_cross_entropy,
    plot_mmlu,
    plot_next_token_probs,
    save_figure,
)


class PlotTests(unittest.TestCase):
    """Build plots from small local result files."""

    def test_detects_and_plots_all_formats(self):
        """Auto-detect each experiment output and create a figure."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            alphas = np.asarray([-1.0, 0.0, 1.0])
            files = {
                "concept_probs": root / "concept.npz",
                "next_token_probs": root / "tokens.npz",
                "cross_entropy": root / "ce.npz",
                "log_odds": root / "odds.npz",
                "mmlu": root / "mmlu.json",
            }
            np.savez(
                files["concept_probs"],
                alphas=alphas,
                p1_by_ctx=np.zeros((2, 3)),
                mean_all=np.asarray([0.1, 0.2, 0.3]),
            )
            np.savez(
                files["next_token_probs"],
                alphas=alphas,
                probs_alphamax=np.asarray([[0.1], [0.2], [0.3]]),
                token_strs_alphamax=np.asarray([" token"], dtype=object),
            )
            np.savez(
                files["cross_entropy"],
                alphas=alphas,
                cross_entropy=np.asarray([2.0, 1.9, 1.8]),
                perplexity=np.asarray([7.4, 6.7, 6.0]),
                delta_cross_entropy=np.asarray([0.1, 0.0, -0.1]),
            )
            np.savez(
                files["log_odds"],
                log_odds=np.asarray([1.0, 2.0]),
                token_strs=np.asarray([" first", " second"], dtype=object),
            )
            files["mmlu"].write_text(
                json.dumps(
                    {
                        "alphas": alphas.tolist(),
                        "overall_scores": [0.2, 0.3, 0.4],
                        "task_scores": {"task": [0.1, 0.2, 0.3]},
                    }
                ),
                encoding="utf-8",
            )

            for expected, path in files.items():
                self.assertEqual(detect_plot_kind(path), expected)
                figure, _ = plot_artifact(path)
                self.assertIsNotNone(figure)
                plt.close(figure)

    def test_saves_figure(self):
        """Write a generated figure to a nested output path."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "concept.npz"
            np.savez(
                artifact,
                alphas=np.asarray([-1.0, 0.0, 1.0]),
                p1_by_ctx=np.zeros((1, 3)),
                mean_all=np.asarray([0.1, 0.2, 0.3]),
            )
            figure, _ = plot_artifact(artifact)
            output = save_figure(figure, root / "plots" / "concept.png")
            plt.close(figure)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_dense_curves_and_large_ranges_use_readable_styles(self):
        """Keep dense curves clean and use scales that show useful variation."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            alphas = np.linspace(-10.0, 10.0, 1001)
            tokens = root / "tokens.npz"
            np.savez(
                tokens,
                alphas=alphas,
                probs_alphamax=np.column_stack(
                    (np.linspace(0.0, 1.0, alphas.size),) * 2
                ),
                token_strs_alphamax=np.asarray([" token", " token"], dtype=object),
                token_alphamax=np.asarray([4, 5]),
            )
            figure, axis = plot_next_token_probs(tokens, top_n=2)
            self.assertTrue(
                all(line.get_marker() == "None" for line in axis.lines[:2])
            )
            legend_text = [
                text.get_text() for text in axis.get_legend().get_texts()
            ]
            self.assertIn("·token [id 4]", legend_text)
            plt.close(figure)

            ce = root / "ce.npz"
            np.savez(
                ce,
                alphas=np.asarray([-1.0, 0.0, 1.0]),
                cross_entropy=np.asarray([4.0, 3.0, 5.0]),
                perplexity=np.asarray([10.0, 100.0, 1000000.0]),
                delta_cross_entropy=np.asarray([1.0, 0.0, 2.0]),
            )
            figure, axis = plot_cross_entropy(ce, metric="perplexity")
            self.assertEqual(axis.get_yscale(), "log")
            plt.close(figure)

            mmlu = root / "mmlu.json"
            mmlu.write_text(
                json.dumps(
                    {
                        "alphas": [-1.0, 0.0, 1.0],
                        "overall_scores": [0.2, 0.25, 0.2],
                        "task_scores": {},
                        "n_shots": 5,
                        "prediction_counts": [{"A": 8}] * 3,
                    }
                ),
                encoding="utf-8",
            )
            figure, axis = plot_mmlu(mmlu)
            self.assertLess(axis.get_ylim()[1], 1.0)
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
