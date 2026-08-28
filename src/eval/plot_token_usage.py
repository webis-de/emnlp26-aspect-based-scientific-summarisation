#!/usr/bin/env python3
"""Plot token-usage and summary-quality tradeoffs with seaborn.

This script merges per-dataset token usage and ROUGE/BERTScore CSV tables,
then writes a suite of plots:
1) All pairwise combinations of token metrics (x) vs quality metrics (y)
2) Per-dataset correlation heatmaps
3) Efficiency frontier plots (quality vs token cost)
4) Bubble plots (cost, quality, rollout count)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


TOKEN_METRICS = ["mean_input_tokens", "mean_output_tokens", "mean_num_rollouts"]
QUALITY_METRICS = ["rouge_1", "rouge_2", "rouge_l", "rouge_l_sum", "bertscore"]
CORR_COLUMNS = TOKEN_METRICS + QUALITY_METRICS


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Plot token usage vs ROUGE/BERTScore for multiple datasets."
	)
	parser.add_argument(
		"--token-dir",
		type=Path,
			 default=Path("results/all/token_usage"),
		help="Directory containing token_usage_table_<dataset>.csv files.",
	)
	parser.add_argument(
		"--quality-dir",
		type=Path,
			 default=Path("results/all/rouge_bertscore"),
		help="Directory containing aggregated_metrics_table_<dataset>.csv files.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
			 default=Path("results/all/plots/token_vs_quality"),
		help="Directory to write generated plots.",
	)
	parser.add_argument(
		"--datasets",
		nargs="+",
		default=["aclsum", "facetsum", "pmc"],
		help="Dataset names (suffixes used in input CSV filenames).",
	)
	parser.add_argument(
		"--dpi",
		type=int,
		default=220,
		help="PNG resolution for saved figures.",
	)
	parser.add_argument(
		"--style",
		default="whitegrid",
		help="Seaborn style preset.",
	)
	parser.add_argument(
		"--context",
		default="talk",
		help="Seaborn context preset (paper, notebook, talk, poster).",
	)
	return parser.parse_args()


def load_dataset_tables(
	token_dir: Path, quality_dir: Path, datasets: Iterable[str]
) -> pd.DataFrame:
	frames: list[pd.DataFrame] = []

	for dataset in datasets:
		token_path = token_dir / f"token_usage_table_{dataset}.csv"
		quality_path = quality_dir / f"aggregated_metrics_table_{dataset}.csv"

		if not token_path.exists():
			print(f"[warn] Missing token CSV: {token_path}")
			continue
		if not quality_path.exists():
			print(f"[warn] Missing quality CSV: {quality_path}")
			continue

		token_df = pd.read_csv(token_path)
		quality_df = pd.read_csv(quality_path)

		merged = token_df.merge(quality_df, on="experiment", how="inner")
		merged["dataset"] = dataset
		frames.append(merged)

	if not frames:
		raise FileNotFoundError(
			"No dataset tables were loaded. Check --token-dir, --quality-dir, and --datasets."
		)

	df = pd.concat(frames, ignore_index=True)

	for col in CORR_COLUMNS + ["mean_total_tokens"]:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce")

	before = len(df)
	df = df.dropna(subset=TOKEN_METRICS + QUALITY_METRICS)
	dropped = before - len(df)
	if dropped > 0:
		print(f"[info] Dropped {dropped} rows with missing metric values.")

	return df


def save_pairwise_token_quality_plots(df: pd.DataFrame, outdir: Path, dpi: int) -> None:
	pair_dir = outdir / "pairwise_token_vs_quality"
	pair_dir.mkdir(parents=True, exist_ok=True)

	for x_metric in TOKEN_METRICS:
		for y_metric in QUALITY_METRICS:
			g = sns.lmplot(
				data=df,
				x=x_metric,
				y=y_metric,
				col="dataset",
				hue="experiment",
				height=4.3,
				aspect=1.1,
				ci=None,
				scatter_kws={"s": 60, "alpha": 0.85},
				line_kws={"alpha": 0.8, "linewidth": 1.2},
				facet_kws={"sharex": False, "sharey": False},
				legend=False,
			)
			g.set_axis_labels(x_metric.replace("_", " "), y_metric.replace("_", " "))
			g.set_titles("dataset = {col_name}")
			g.figure.suptitle(
				f"{y_metric} vs {x_metric}", y=1.03, fontsize=14, fontweight="bold"
			)

			# Large legends can obscure facets; place outside the plotting area.
			g.add_legend(title="experiment", bbox_to_anchor=(1.02, 0.5), loc="center left")

			save_path = pair_dir / f"{y_metric}__vs__{x_metric}.png"
			g.figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
			plt.close(g.figure)


def save_correlation_heatmaps(df: pd.DataFrame, outdir: Path, dpi: int) -> None:
	heatmap_dir = outdir / "correlation_heatmaps"
	heatmap_dir.mkdir(parents=True, exist_ok=True)

	for dataset, subdf in df.groupby("dataset", sort=True):
		corr = subdf[CORR_COLUMNS].corr(numeric_only=True)
		plt.figure(figsize=(9.2, 7.4))
		sns.heatmap(
			corr,
			annot=True,
			fmt=".2f",
			cmap="vlag",
			center=0,
			linewidths=0.4,
			square=True,
			cbar_kws={"shrink": 0.85},
		)
		plt.title(f"Correlation matrix: {dataset}")
		plt.tight_layout()
		plt.savefig(heatmap_dir / f"corr_heatmap_{dataset}.png", dpi=dpi)
		plt.close()


def _annotate_points(ax: plt.Axes, frame: pd.DataFrame, x_col: str, y_col: str) -> None:
	for _, row in frame.iterrows():
		ax.annotate(
			row["experiment"],
			(row[x_col], row[y_col]),
			xytext=(4, 4),
			textcoords="offset points",
			fontsize=8,
			alpha=0.85,
		)


def save_efficiency_frontiers(df: pd.DataFrame, outdir: Path, dpi: int) -> None:
	frontier_dir = outdir / "efficiency_frontiers"
	frontier_dir.mkdir(parents=True, exist_ok=True)

	cost_metrics = ["mean_output_tokens", "mean_total_tokens"]
	score_metrics = ["rouge_l_sum", "bertscore"]

	for dataset, subdf in df.groupby("dataset", sort=True):
		for x_metric in cost_metrics:
			if x_metric not in subdf.columns:
				continue
			for y_metric in score_metrics:
				fig, ax = plt.subplots(figsize=(8.6, 5.9))
				sns.scatterplot(
					data=subdf,
					x=x_metric,
					y=y_metric,
					hue="experiment",
					s=120,
					ax=ax,
				)
				sns.regplot(
					data=subdf,
					x=x_metric,
					y=y_metric,
					scatter=False,
					color="black",
					line_kws={"linewidth": 1.2, "alpha": 0.8},
					ax=ax,
				)
				_annotate_points(ax, subdf, x_metric, y_metric)
				ax.set_title(f"{dataset}: {y_metric} vs {x_metric}")
				ax.set_xlabel(x_metric.replace("_", " "))
				ax.set_ylabel(y_metric.replace("_", " "))
				ax.legend(title="experiment", bbox_to_anchor=(1.02, 1), loc="upper left")
				fig.tight_layout()
				fig.savefig(
					frontier_dir / f"frontier_{dataset}_{y_metric}__vs__{x_metric}.png",
					dpi=dpi,
					bbox_inches="tight",
				)
				plt.close(fig)


def save_bubble_plots(df: pd.DataFrame, outdir: Path, dpi: int) -> None:
	bubble_dir = outdir / "bubble_plots"
	bubble_dir.mkdir(parents=True, exist_ok=True)

	for y_metric in ["rouge_l_sum", "bertscore"]:
		fig, ax = plt.subplots(figsize=(9.5, 6.2))
		sns.scatterplot(
			data=df,
			x="mean_output_tokens",
			y=y_metric,
			size="mean_num_rollouts",
			hue="dataset",
			style="dataset",
			sizes=(60, 450),
			alpha=0.82,
			ax=ax,
		)
		ax.set_title(f"Bubble tradeoff: {y_metric} vs mean_output_tokens")
		ax.set_xlabel("mean output tokens")
		ax.set_ylabel(y_metric.replace("_", " "))
		ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
		fig.tight_layout()
		fig.savefig(
			bubble_dir / f"bubble_{y_metric}__vs__mean_output_tokens.png",
			dpi=dpi,
			bbox_inches="tight",
		)
		plt.close(fig)


def main() -> None:
	args = parse_args()
	args.output_dir.mkdir(parents=True, exist_ok=True)

	sns.set_theme(style=args.style, context=args.context)

	merged_df = load_dataset_tables(args.token_dir, args.quality_dir, args.datasets)
	merged_csv = args.output_dir / "merged_token_quality_table.csv"
	merged_df.to_csv(merged_csv, index=False)

	save_pairwise_token_quality_plots(merged_df, args.output_dir, args.dpi)
	save_correlation_heatmaps(merged_df, args.output_dir, args.dpi)
	save_efficiency_frontiers(merged_df, args.output_dir, args.dpi)
	save_bubble_plots(merged_df, args.output_dir, args.dpi)

	print(f"[done] Wrote merged table: {merged_csv}")
	print(f"[done] Plots saved under: {args.output_dir}")


if __name__ == "__main__":
	main()
