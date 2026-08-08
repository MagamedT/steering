"""Plot result files produced by the experiment commands.

The functions accept saved NPZ or JSON artifacts and return Matplotlib figures.
The module can also be run as a command to save a plot directly.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


PlotKind = Literal[
    "concept_probs",
    "next_token_probs",
    "mmlu",
    "cross_entropy",
    "log_odds",
]

_ALPHA_LABEL = r"Steering coefficient $\alpha$"
_TEXT = "#202124"
_MUTED = "#5F6368"
_GRID = "#DADCE0"
_COLORS = {
    "overall": "#0072B2",
    "negative": "#D55E00",
    "positive": "#009E73",
    "match": "#CC79A7",
    "metric": "#6F4C9B",
    "reference": "#6B7280",
}
_SERIES_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
)


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load one experiment NPZ into memory."""
    with np.load(Path(path), allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _load_meta(data: dict[str, np.ndarray]) -> dict:
    """Read optional JSON metadata from an NPZ payload."""
    if "meta" not in data:
        return {}
    value = np.asarray(data["meta"]).item()
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _new_axis(
    ax: Axes | None,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: tuple[float, float] = (7.2, 4.4),
) -> tuple[Figure, Axes]:
    """Create or configure an axis for one plot."""
    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
    else:
        figure = ax.figure
    figure.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_title(
        title,
        loc="left",
        color=_TEXT,
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(xlabel, color=_TEXT, fontsize=10)
    ax.set_ylabel(ylabel, color=_TEXT, fontsize=10)
    ax.tick_params(
        axis="both",
        colors=_TEXT,
        labelsize=9,
        direction="out",
        length=4,
        width=0.8,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, color=_GRID, linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    return figure, ax


def _artifact_title(prefix: str, meta: dict) -> str:
    """Build a short title from saved metadata."""
    details = [prefix]
    if meta.get("concept"):
        details.append(str(meta["concept"]))
    if meta.get("layer_idx") is not None:
        details.append(f"layer {meta['layer_idx']}")
    return " · ".join(details)


def _axis_note(ax: Axes, text: str | None) -> None:
    """Add a small note above the top-right edge of an axis."""
    if not text:
        return
    ax.text(
        1.0,
        1.02,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=_MUTED,
        fontsize=8,
        clip_on=False,
    )


def _zero_reference(ax: Axes) -> None:
    """Mark the unsteered alpha value."""
    ax.axvline(
        0.0,
        color=_COLORS["reference"],
        linestyle=(0, (4, 3)),
        linewidth=1.0,
        alpha=0.8,
        zorder=1,
    )


def _baseline_marker(
    ax: Axes,
    alphas: np.ndarray,
    values: np.ndarray,
    *,
    color: str,
) -> None:
    """Highlight a saved alpha-zero value when present."""
    indices = np.flatnonzero(alphas == 0.0)
    if indices.size != 1:
        return
    value = float(values[int(indices[0])])
    if not np.isfinite(value):
        return
    ax.scatter(
        [0.0],
        [value],
        s=32,
        facecolor="white",
        edgecolor=color,
        linewidth=1.5,
        zorder=5,
    )


def _mark_every(count: int, target: int = 12) -> int:
    """Space markers so dense curves remain readable."""
    return max(1, int(np.ceil(count / target)))


def _token_text(value: object) -> str:
    """Make whitespace in a decoded token visible."""
    text = str(value)
    if not text:
        return "empty"
    return (
        text.replace(" ", "·")
        .replace("\n", "↵")
        .replace("\t", "⇥")
        .replace("\r", "␍")
    )


def _token_labels(
    tokens: np.ndarray,
    token_ids: np.ndarray | None = None,
) -> list[str]:
    """Build readable labels and disambiguate duplicate decoded tokens."""
    labels = [_token_text(value) for value in tokens]
    counts = Counter(labels)
    if token_ids is None:
        return labels
    return [
        f"{label} [id {int(token_ids[index])}]" if counts[label] > 1 else label
        for index, label in enumerate(labels)
    ]


def _probability_ylim(ax: Axes, values: list[np.ndarray]) -> None:
    """Use the probability range without wasting vertical space."""
    finite = np.concatenate(
        [array[np.isfinite(array)] for array in values if array.size]
    )
    maximum = float(np.max(finite)) if finite.size else 1.0
    upper = 1.02 if maximum >= 0.9 else max(0.05, maximum * 1.12)
    ax.set_ylim(0.0, min(1.02, upper))


def _alphas(data: dict[str, np.ndarray]) -> np.ndarray:
    """Return and validate the saved alpha values."""
    values = np.asarray(data.get("alphas"), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("The result file must contain a nonempty 1D 'alphas' array.")
    return values


def plot_concept_probs(
    path: str | Path,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot binary or continuous concept-probability curves."""
    data = _load_npz(path)
    alphas = _alphas(data)
    meta = _load_meta(data)
    figure, ax = _new_axis(
        ax,
        title=_artifact_title("Concept probability", meta),
        xlabel=_ALPHA_LABEL,
        ylabel="Concept probability",
    )

    curves = (
        ("mean_all", "Overall", _COLORS["overall"], "-"),
        ("mean_negative", "Negative context", _COLORS["negative"], "--"),
        ("mean_positive", "Positive context", _COLORS["positive"], "-."),
        ("mean_match", "Context match", _COLORS["match"], ":"),
    )
    plotted = 0
    for key, label, color, linestyle in curves:
        if key not in data:
            continue
        values = np.asarray(data[key], dtype=np.float64)
        if values.shape != alphas.shape:
            raise ValueError(f"'{key}' must have one value per alpha.")
        ax.plot(
            alphas,
            values,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            markeredgewidth=0,
            markevery=_mark_every(alphas.size),
            label=label,
        )
        _baseline_marker(ax, alphas, values, color=color)
        plotted += 1
    if not plotted:
        raise ValueError("No concept-probability curves were found in the file.")

    _zero_reference(ax)
    ax.set_ylim(-0.02, 1.02)
    judge = str(meta.get("judge_model", "")).split("/")[-1]
    _axis_note(ax, f"Judge: {judge}" if judge else None)
    ax.legend(
        frameon=False,
        fontsize=8.5,
        ncol=2,
        loc="upper left",
        handlelength=2.7,
        columnspacing=1.2,
    )
    ax.margins(x=0.015)
    return figure, ax


def plot_next_token_probs(
    path: str | Path,
    *,
    token_set: Literal["alphamin", "alphamax"] = "alphamax",
    top_n: int = 5,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot selected next-token probabilities against alpha."""
    if token_set not in {"alphamin", "alphamax"}:
        raise ValueError("token_set must be 'alphamin' or 'alphamax'.")
    if top_n < 1:
        raise ValueError("top_n must be positive.")

    data = _load_npz(path)
    alphas = _alphas(data)
    probability_key = f"probs_{token_set}"
    token_key = f"token_strs_{token_set}"
    probabilities = np.asarray(data.get(probability_key), dtype=np.float64)
    tokens = np.asarray(data.get(token_key), dtype=object)
    token_id_key = f"token_{token_set}"
    token_ids = (
        np.asarray(data[token_id_key])
        if token_id_key in data
        else None
    )
    if probabilities.ndim != 2 or probabilities.shape[0] != alphas.size:
        raise ValueError(f"'{probability_key}' must have shape [alpha, token].")
    if tokens.ndim != 1 or tokens.size != probabilities.shape[1]:
        raise ValueError(f"'{token_key}' must contain one label per token curve.")

    count = min(int(top_n), tokens.size)
    meta = _load_meta(data)
    figure, ax = _new_axis(
        ax,
        title=_artifact_title("Next-token probability", meta),
        xlabel=_ALPHA_LABEL,
        ylabel="Next-token probability",
    )
    selected_tokens = tokens[:count]
    selected_ids = token_ids[:count] if token_ids is not None else None
    labels = _token_labels(selected_tokens, selected_ids)
    shown_values: list[np.ndarray] = []
    for index in range(count):
        values = probabilities[:, index]
        shown_values.append(values)
        ax.plot(
            alphas,
            values,
            color=_SERIES_COLORS[index % len(_SERIES_COLORS)],
            linewidth=2.0,
            label=labels[index],
        )
    _zero_reference(ax)
    _probability_ylim(ax, shown_values)
    context = str(meta.get("context", "")).strip()
    if len(context) > 58:
        context = context[:55].rstrip() + "..."
    _axis_note(ax, f"Context: {context}" if context else None)
    ax.legend(
        title="Selected tokens",
        title_fontsize=8.5,
        frameon=False,
        fontsize=8.2,
        loc="best",
        handlelength=2.4,
    )
    ax.margins(x=0.015)
    return figure, ax


def _optional_floats(values) -> np.ndarray:
    """Convert optional JSON scores to floats with NaN for missing values."""
    return np.asarray(
        [np.nan if value is None else float(value) for value in values],
        dtype=np.float64,
    )


def plot_mmlu(
    path: str | Path,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot overall and per-task MMLU accuracy against alpha."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("The MMLU result must be one JSON object.")

    alphas = np.asarray(data.get("alphas"), dtype=np.float64)
    overall = _optional_floats(data.get("overall_scores", []))
    if alphas.ndim != 1 or alphas.size == 0 or overall.shape != alphas.shape:
        raise ValueError("MMLU alphas and overall_scores must have matching lengths.")

    figure, ax = _new_axis(
        ax,
        title=_artifact_title("MMLU accuracy", data),
        xlabel=_ALPHA_LABEL,
        ylabel="Accuracy",
    )
    plotted_values = [overall]
    ax.axhline(
        0.25,
        color=_COLORS["reference"],
        linestyle=":",
        linewidth=1.2,
        alpha=0.9,
        label="Random-choice baseline",
        zorder=1,
    )
    ax.plot(
        alphas,
        overall,
        color=_COLORS["overall"],
        linewidth=2.2,
        drawstyle="steps-mid",
        label="Overall",
        zorder=3,
    )
    _baseline_marker(ax, alphas, overall, color=_COLORS["overall"])
    for task, scores in (data.get("task_scores") or {}).items():
        values = _optional_floats(scores)
        if values.shape != alphas.shape:
            raise ValueError(f"MMLU task '{task}' must have one score per alpha.")
        if np.isfinite(values).any():
            plotted_values.append(values)
            ax.plot(
                alphas,
                values,
                linewidth=1.5,
                linestyle="--",
                alpha=0.85,
                label=str(task).replace("_", " "),
            )
    _zero_reference(ax)
    finite = np.concatenate(
        [values[np.isfinite(values)] for values in plotted_values]
    )
    maximum = float(np.max(finite)) if finite.size else 1.0
    upper = min(1.0, max(0.1, np.ceil((maximum + 0.04) * 10) / 10))
    ax.set_ylim(0.0, upper)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    counts = data.get("prediction_counts") or []
    question_count = sum((counts[0] or {}).values()) if counts else None
    details = []
    if data.get("n_shots") is not None:
        details.append(f"{int(data['n_shots'])}-shot")
    if question_count:
        details.append(f"n={int(question_count)}")
    _axis_note(ax, " · ".join(details))
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    ax.margins(x=0.015)
    return figure, ax


def plot_cross_entropy(
    path: str | Path,
    *,
    metric: Literal["cross_entropy", "perplexity", "delta_cross_entropy"] = "cross_entropy",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot cross-entropy, perplexity, or loss change against alpha."""
    labels = {
        "cross_entropy": "Cross entropy",
        "perplexity": "Perplexity",
        "delta_cross_entropy": "Cross-entropy change",
    }
    if metric not in labels:
        raise ValueError(f"metric must be one of: {', '.join(labels)}")

    data = _load_npz(path)
    alphas = _alphas(data)
    values = np.asarray(data.get(metric), dtype=np.float64)
    if values.shape != alphas.shape:
        raise ValueError(f"'{metric}' must have one value per alpha.")

    meta = _load_meta(data)
    figure, ax = _new_axis(
        ax,
        title=_artifact_title(labels[metric], meta),
        xlabel=_ALPHA_LABEL,
        ylabel=(
            "Perplexity (log scale)"
            if metric == "perplexity"
            else labels[metric]
        ),
    )
    if metric == "perplexity":
        if np.any(values <= 0):
            raise ValueError("Perplexity values must be positive for log scaling.")
        ax.set_yscale("log")
    ax.plot(
        alphas,
        values,
        color=_COLORS["metric"],
        linewidth=2.2,
    )
    _baseline_marker(ax, alphas, values, color=_COLORS["metric"])
    _zero_reference(ax)
    if metric == "delta_cross_entropy":
        ax.axhline(
            0.0,
            color=_COLORS["reference"],
            linestyle=":",
            linewidth=1.0,
        )
    if metric == "perplexity":
        ax.grid(True, which="major", color=_GRID, linewidth=0.7, alpha=0.65)
    ax.margins(x=0.015)
    return figure, ax


def plot_log_odds(
    path: str | Path,
    *,
    top_n: int = 20,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot tokens with the largest positive log odds."""
    if top_n < 1:
        raise ValueError("top_n must be positive.")
    data = _load_npz(path)
    values = np.asarray(data.get("log_odds"), dtype=np.float64)
    tokens = np.asarray(data.get("token_strs"), dtype=object)
    if values.ndim != 1 or tokens.shape != values.shape or values.size == 0:
        raise ValueError("Log-odds values and token labels must be matching 1D arrays.")

    count = min(int(top_n), values.size)
    order = np.argsort(values)[-count:]
    shown_values = values[order]
    token_ids = np.asarray(data["token_ids"]) if "token_ids" in data else None
    shown_ids = token_ids[order] if token_ids is not None else None
    shown_tokens = _token_labels(tokens[order], shown_ids)
    meta = _load_meta(data)
    figure, ax = _new_axis(
        ax,
        title=_artifact_title("Token log odds", meta),
        xlabel="Log-odds ratio",
        ylabel="",
        figsize=(7.2, max(4.4, 0.29 * count + 1.5)),
    )
    positions = np.arange(count)
    colors = mpl.colormaps["Blues"](np.linspace(0.45, 0.85, count))
    bars = ax.barh(
        positions,
        shown_values,
        color=colors,
        edgecolor="none",
        height=0.72,
    )
    ax.set_yticks(positions, shown_tokens)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    ax.xaxis.grid(True, color=_GRID, linewidth=0.7, alpha=0.65)
    ax.axvline(0.0, color=_COLORS["reference"], linewidth=0.8)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=7.5, color=_TEXT)
    maximum = float(np.max(shown_values))
    ax.set_xlim(min(0.0, float(np.min(shown_values))), maximum * 1.13)
    _axis_note(ax, "Positive values favor concept prompts")
    return figure, ax


def detect_plot_kind(path: str | Path) -> PlotKind:
    """Detect which experiment produced a result file."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and {"alphas", "overall_scores"} <= set(data):
            return "mmlu"
        raise ValueError(f"Could not recognize JSON result: {path}")

    data = _load_npz(path)
    keys = set(data)
    if {"alphas", "probs_alphamax", "token_strs_alphamax"} <= keys:
        return "next_token_probs"
    if {"alphas", "cross_entropy", "perplexity"} <= keys:
        return "cross_entropy"
    if {"log_odds", "token_strs"} <= keys:
        return "log_odds"
    if "alphas" in keys and ("concept_scores_by_ctx" in keys or "p1_by_ctx" in keys):
        return "concept_probs"
    raise ValueError(f"Could not recognize result file: {path}")


def plot_artifact(
    path: str | Path,
    *,
    kind: PlotKind | Literal["auto"] = "auto",
    metric: Literal["cross_entropy", "perplexity", "delta_cross_entropy"] = "cross_entropy",
    token_set: Literal["alphamin", "alphamax"] = "alphamax",
    top_n: int = 5,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Detect and plot one saved experiment result."""
    selected = detect_plot_kind(path) if kind == "auto" else kind
    if selected == "concept_probs":
        return plot_concept_probs(path, ax=ax)
    if selected == "next_token_probs":
        return plot_next_token_probs(path, token_set=token_set, top_n=top_n, ax=ax)
    if selected == "mmlu":
        return plot_mmlu(path, ax=ax)
    if selected == "cross_entropy":
        return plot_cross_entropy(path, metric=metric, ax=ax)
    if selected == "log_odds":
        return plot_log_odds(path, top_n=top_n, ax=ax)
    raise ValueError(f"Unknown plot kind: {selected}")


def save_figure(
    figure: Figure,
    path: str | Path,
    *,
    dpi: int = 300,
) -> Path:
    """Save a figure and create its parent directory if needed."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(pad=0.7)
    with mpl.rc_context(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        figure.savefig(
            output,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
    return output


def parse_args() -> argparse.Namespace:
    """Parse plotting command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", help="NPZ or JSON result to plot.")
    parser.add_argument(
        "--kind",
        choices=("auto", "concept_probs", "next_token_probs", "mmlu", "cross_entropy", "log_odds"),
        default="auto",
    )
    parser.add_argument(
        "--metric",
        choices=("cross_entropy", "perplexity", "delta_cross_entropy"),
        default="cross_entropy",
    )
    parser.add_argument(
        "--token_set", choices=("alphamin", "alphamax"), default="alphamax"
    )
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--output", help="Output image path. Defaults to the artifact name with .png.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Plot one artifact from the command line."""
    args = parse_args()
    artifact = Path(args.artifact)
    figure, _ = plot_artifact(
        artifact,
        kind=args.kind,
        metric=args.metric,
        token_set=args.token_set,
        top_n=args.top_n,
    )
    if args.output or not args.show:
        output = Path(args.output) if args.output else artifact.with_suffix(".png")
        print(save_figure(figure, output, dpi=args.dpi), flush=True)
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
