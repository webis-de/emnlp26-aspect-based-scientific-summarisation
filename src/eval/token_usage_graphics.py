#!/usr/bin/env python3
"""
Generate a paper-quality quality-cost scatter plot for aspect-based
scientific summarization.

Example:
    python plot_quality_cost.py \
        --aclsum-tokens token_usage_table_aclsum.csv \
        --aclsum-metrics aggregated_metrics_table_aclsum.csv \
        --pmc-tokens token_usage_table_pmc.csv \
        --pmc-metrics aggregated_metrics_table_pmc.csv \
        --facetsum-tokens token_usage_table_facetsum.csv \
        --facetsum-metrics aggregated_metrics_table_facetsum.csv \
        --output quality_cost_tradeoff.pdf \
        --metric RLS \
        --also-png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


# ---------------------------------------------------------------------
# Experiment-name normalization
# ---------------------------------------------------------------------

METHOD_NORMALIZATION = {
    "zs": "ZS",
    "zs_results": "ZS",
    "zero_shot": "ZS",
    "zeroshot": "ZS",
    "zero-shot": "ZS",
    "e2a": "E2A",
    "rag": "RAG",
    "cod": "COD",
    "self_refine": "SR",
    "self-refine": "SR",
    "sr": "SR",
    "2a2s": "Agentic",
}

METHOD_ORDER = ["ZS", "RAG", "E2A", "COD", "SR", "Agentic"]


DATASET_DISPLAY_NAMES = {
    "PMC": "PMC-SA",
}


# ---------------------------------------------------------------------
# Metric aliases.
# These map CLI-friendly names to columns in your aggregated metric CSVs.
# ---------------------------------------------------------------------

METRIC_COLUMN_ALIASES = {
    "r1": "rouge_1",
    "r-1": "rouge_1",
    "rouge1": "rouge_1",
    "rouge_1": "rouge_1",
    "rouge-1": "rouge_1",

    "r2": "rouge_2",
    "r-2": "rouge_2",
    "rouge2": "rouge_2",
    "rouge_2": "rouge_2",
    "rouge-2": "rouge_2",

    "rl": "rouge_l",
    "r-l": "rouge_l",
    "rouge_l": "rouge_l",
    "rouge-l": "rouge_l",

    "rls": "rouge_l_sum",
    "r-l-s": "rouge_l_sum",
    "rouge_l_sum": "rouge_l_sum",
    "rouge-lsum": "rouge_l_sum",
    "rouge_lsum": "rouge_l_sum",

    "bs": "bertscore",
    "bertscore": "bertscore",
    "bert_score": "bertscore",
}

METRIC_LABELS = {
    "rouge_1": "ROUGE-1",
    "rouge_2": "ROUGE-2",
    "rouge_l": "ROUGE-L",
    "rouge_l_sum": "ROUGE-LSum",
    "bertscore": "BERTScore",
}


# ---------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------

METHOD_STYLE = {
    "ZS": {
        "marker": "*",
        "size": 260,
        "color": "#111111",
        "edgecolor": "#111111",
        "linewidth": 1.2,
        "zorder": 5,
    },
    "RAG": {
        "marker": "o",
        "size": 90,
        "color": "#4C78A8",
        "edgecolor": "white",
        "linewidth": 0.7,
        "zorder": 4,
    },
    "E2A": {
        "marker": "s",
        "size": 90,
        "color": "#F58518",
        "edgecolor": "white",
        "linewidth": 0.7,
        "zorder": 4,
    },
    "COD": {
        "marker": "^",
        "size": 90,
        "color": "#54A24B",
        "edgecolor": "white",
        "linewidth": 0.7,
        "zorder": 4,
    },
    "SR": {
        "marker": "D",
        "size": 90,
        "color": "#B279A2",
        "edgecolor": "white",
        "linewidth": 0.7,
        "zorder": 4,
    },
    "Agentic": {
        "marker": "P",
        "size": 100,
        "color": "#E45756",
        "edgecolor": "white",
        "linewidth": 0.7,
        "zorder": 4,
    },
}


# Manual label offsets in display points.
LABEL_OFFSETS: Dict[Tuple[str, str], Tuple[int, int]] = {
    ("ACLSum", "ZS"): (6, 6),
    ("ACLSum", "RAG"): (6, 6),
    ("ACLSum", "E2A"): (6, -10),
    ("ACLSum", "COD"): (6, -10),
    ("ACLSum", "SR"): (6, 5),
    ("ACLSum", "Agentic"): (6, -10),

    ("PMC", "ZS"): (6, 6),
    ("PMC", "RAG"): (6, -10),
    ("PMC", "E2A"): (6, -10),
    ("PMC", "COD"): (6, 6),
    ("PMC", "SR"): (6, -10),
    ("PMC", "Agentic"): (6, 6),

    ("FacetSum", "ZS"): (6, 6),
    ("FacetSum", "RAG"): (6, -10),
    ("FacetSum", "E2A"): (6, 6),
    ("FacetSum", "COD"): (6, -10),
    ("FacetSum", "SR"): (6, -10),
    ("FacetSum", "Agentic"): (6, 6),
}


def normalize_method_name(raw_name: str) -> str:
    key = str(raw_name).strip().lower()

    if key not in METHOD_NORMALIZATION:
        raise ValueError(
            f"Unknown experiment name: {raw_name!r}. "
            f"Known names: {sorted(METHOD_NORMALIZATION)}"
        )

    return METHOD_NORMALIZATION[key]


def normalize_metric_name(raw_metric: str) -> str:
    key = str(raw_metric).strip().lower()

    if key not in METRIC_COLUMN_ALIASES:
        raise ValueError(
            f"Unknown metric: {raw_metric!r}. "
            f"Valid options include: R1, R2, RL, RLS, BS, "
            f"rouge_1, rouge_2, rouge_l, rouge_l_sum, bertscore."
        )

    return METRIC_COLUMN_ALIASES[key]


def read_csv_checked(path: Path, required_columns: set[str], table_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"{table_name} at {path} is missing required columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    return df


def assert_unique_methods(df: pd.DataFrame, table_name: str, path: Path) -> None:
    duplicated = df.loc[df["method"].duplicated(), "method"].tolist()

    if duplicated:
        raise ValueError(
            f"{table_name} at {path} contains duplicated normalized methods: "
            f"{sorted(set(duplicated))}. Please aggregate or deduplicate first."
        )


def load_dataset_results(
    *,
    dataset_name: str,
    token_path: Path,
    metrics_path: Path,
    token_col: str,
    metric_col: str,
) -> pd.DataFrame:
    token_df = read_csv_checked(
        token_path,
        required_columns={"experiment", token_col},
        table_name=f"{dataset_name} token table",
    )

    metrics_df = read_csv_checked(
        metrics_path,
        required_columns={"experiment", metric_col},
        table_name=f"{dataset_name} metrics table",
    )

    token_df = token_df.copy()
    metrics_df = metrics_df.copy()

    token_df["method"] = token_df["experiment"].map(normalize_method_name)
    metrics_df["method"] = metrics_df["experiment"].map(normalize_method_name)

    assert_unique_methods(token_df, f"{dataset_name} token table", token_path)
    assert_unique_methods(metrics_df, f"{dataset_name} metrics table", metrics_path)

    token_df = token_df.rename(
        columns={
            "experiment": "token_experiment",
            token_col: "tokens",
        }
    )

    metrics_df = metrics_df.rename(
        columns={
            "experiment": "metric_experiment",
            metric_col: "score",
        }
    )

    merged = token_df.merge(
        metrics_df[["method", "metric_experiment", "score"]],
        on="method",
        how="inner",
        validate="one_to_one",
    )

    token_methods = set(token_df["method"])
    metric_methods = set(metrics_df["method"])

    missing_in_metrics = sorted(token_methods - metric_methods)
    missing_in_tokens = sorted(metric_methods - token_methods)

    if missing_in_metrics or missing_in_tokens:
        raise ValueError(
            f"Method mismatch for {dataset_name}.\n"
            f"Missing in metrics table: {missing_in_metrics}\n"
            f"Missing in token table: {missing_in_tokens}"
        )

    merged["dataset"] = dataset_name
    merged["tokens"] = merged["tokens"].astype(float)
    merged["score"] = merged["score"].astype(float)
    merged["method"] = pd.Categorical(
        merged["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    return merged.sort_values("method")


def build_plot_df(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    metric_col = normalize_metric_name(args.metric)

    frames = [
        load_dataset_results(
            dataset_name="ACLSum",
            token_path=Path(args.aclsum_tokens),
            metrics_path=Path(args.aclsum_metrics),
            token_col=args.token_column,
            metric_col=metric_col,
        ),
        load_dataset_results(
            dataset_name="PMC",
            token_path=Path(args.pmc_tokens),
            metrics_path=Path(args.pmc_metrics),
            token_col=args.token_column,
            metric_col=metric_col,
        ),
        load_dataset_results(
            dataset_name="FacetSum",
            token_path=Path(args.facetsum_tokens),
            metrics_path=Path(args.facetsum_metrics),
            token_col=args.token_column,
            metric_col=metric_col,
        ),
    ]

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["dataset", "method"])

    return df, metric_col


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def thousands_formatter(x: float, _: int) -> str:
    if x >= 1000:
        return f"{x / 1000:g}k"
    return f"{x:g}"


def plot_quality_cost(
    df: pd.DataFrame,
    *,
    metric_col: str,
    args: argparse.Namespace,
) -> None:
    configure_matplotlib()

    datasets = ["ACLSum", "PMC", "FacetSum"]
    metric_label = METRIC_LABELS[metric_col]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(7.2, 2.65),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )

    global_x_min = df["tokens"].min()
    global_x_max = df["tokens"].max()

    # Padding for log-scale x-axis.
    x_min = global_x_min / 1.25
    x_max = global_x_max * 1.25

    handles = {}

    for ax, dataset in zip(axes, datasets):
        sub = df[df["dataset"] == dataset].copy()
        display_dataset = DATASET_DISPLAY_NAMES.get(dataset, dataset)

        for method in METHOD_ORDER:
            row = sub[sub["method"] == method]

            if row.empty:
                continue

            row = row.iloc[0]
            style = METHOD_STYLE[method]

            point = ax.scatter(
                row["tokens"],
                row["score"],
                s=style["size"],
                marker=style["marker"],
                color=style["color"],
                edgecolor=style["edgecolor"],
                linewidth=style["linewidth"],
                zorder=style["zorder"],
                label=method,
            )

            handles[method] = point

            if not args.no_labels:
                dx, dy = LABEL_OFFSETS.get((dataset, method), (6, 6))
                ax.annotate(
                    method,
                    xy=(row["tokens"], row["score"]),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=7.6,
                    color="#222222",
                )

        ax.set_title(display_dataset, fontweight="bold", pad=5, fontsize=7)
        ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)

        ax.grid(True, which="major", axis="both", alpha=0.28)
        ax.grid(True, which="minor", axis="x", alpha=0.12)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(thousands_formatter))

        y_min = sub["score"].min()
        y_max = sub["score"].max()
        y_margin = max((y_max - y_min) * 0.20, 0.005)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        # Zero-shot reference: methods in the lower-right region are dominated by ZS
        # because they use more tokens and achieve lower quality.
        zs_row = sub[sub["method"] == "ZS"]

        if not zs_row.empty:
            zs_tokens = float(zs_row.iloc[0]["tokens"])
            zs_score = float(zs_row.iloc[0]["score"])

            ax.axhline(
                zs_score,
                linestyle="--",
                linewidth=0.8,
                color="#666666",
                alpha=0.55,
                zorder=1,
            )

            ax.axvline(
                zs_tokens,
                linestyle="--",
                linewidth=0.8,
                color="#666666",
                alpha=0.55,
                zorder=1,
            )

            y_low, y_high = ax.get_ylim()
            dominated_ymax = (zs_score - y_low) / (y_high - y_low)

            ax.axvspan(
                zs_tokens,
                x_max,
                ymin=0,
                ymax=dominated_ymax,
                color="#999999",
                alpha=0.10,
                zorder=0,
            )

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel(metric_label)

    fig.supxlabel(
        f"{args.token_column.replace('_', ' ').capitalize()} per instance "
        "(log scale)",
        y=-0.025,
        fontsize=9,
    )

    legend_handles = [handles[m] for m in METHOD_ORDER if m in handles]
    legend_labels = [m for m in METHOD_ORDER if m in handles]

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=len(legend_labels),
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.1,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output, bbox_inches="tight")

    if args.also_png:
        fig.savefig(output.with_suffix(".png"), bbox_inches="tight")

    if args.also_svg:
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a paper-quality quality-cost scatter plot from "
            "token-usage CSVs and aggregated metric CSVs."
        )
    )

    parser.add_argument(
        "--aclsum-tokens",
        required=True,
        help="Path to ACLSum token-usage CSV.",
    )
    parser.add_argument(
        "--aclsum-metrics",
        required=True,
        help="Path to ACLSum aggregated metric CSV.",
    )

    parser.add_argument(
        "--pmc-tokens",
        required=True,
        help="Path to PMC token-usage CSV.",
    )
    parser.add_argument(
        "--pmc-metrics",
        required=True,
        help="Path to PMC aggregated metric CSV.",
    )

    parser.add_argument(
        "--facetsum-tokens",
        required=True,
        help="Path to FacetSum token-usage CSV.",
    )
    parser.add_argument(
        "--facetsum-metrics",
        required=True,
        help="Path to FacetSum aggregated metric CSV.",
    )

    parser.add_argument(
        "--output",
        default="quality_cost_tradeoff.pdf",
        help="Output figure path. Recommended: .pdf for papers.",
    )

    parser.add_argument(
        "--metric",
        default="RLS",
        help=(
            "Metric to plot on the y-axis. Common options: "
            "R1, R2, RL, RLS, BS. "
            "Also accepts CSV column names such as rouge_l_sum or bertscore."
        ),
    )

    parser.add_argument(
        "--token-column",
        default="mean_total_tokens",
        help=(
            "Token statistic to plot on the x-axis. "
            "Examples: mean_total_tokens, median_total_tokens, "
            "mean_input_tokens, mean_output_tokens."
        ),
    )

    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Do not annotate points with method names.",
    )

    parser.add_argument(
        "--also-png",
        action="store_true",
        help="Also save a high-resolution PNG next to the main output.",
    )

    parser.add_argument(
        "--also-svg",
        action="store_true",
        help="Also save an SVG next to the main output.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df, metric_col = build_plot_df(args)
    plot_quality_cost(df, metric_col=metric_col, args=args)


if __name__ == "__main__":
    main()
    print("Plot saved successfully.")