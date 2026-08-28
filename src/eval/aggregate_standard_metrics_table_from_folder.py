"""
This script aggregates standard metrics (ROUGE and BERTScore) from a folder of JSON files into a single CSV or JSON table.
It generates per-dataset tables and optionally plots the top K experiments for each dataset.

Example usage:
python src/eval/aggregate_standard_metrics_table_from_folder.py
    --input-dir 2a2s/results/e2a/rebuttal_sensitivity/
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence


ROUGE_METRICS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
ROUGE_SCORE_FIELDS = ["precision", "recall", "fmeasure"]

DEFAULT_METRICS = [
    "rouge1_fmeasure",
    "rouge2_fmeasure",
    "rougeL_fmeasure",
    "rougeLsum_fmeasure",
    "bertscore",
]
DISPLAY_METRIC_NAMES = {
    "rouge1_fmeasure": "rouge_1",
    "rouge2_fmeasure": "rouge_2",
    "rougeL_fmeasure": "rouge_l",
    "rougeLsum_fmeasure": "rouge_l_sum",
    "bertscore": "bertscore",
}
DEFAULT_DATASET_OUTPUT_PREFIX = "aggregated_metrics_table"
EXTRA_AGGREGATE_COLUMNS = [
    "mean_num_rollouts",
    "max_num_rollouts",
    "mean_input_tokens",
    "max_input_tokens",
    "mean_output_tokens",
    "max_output_tokens",
]
DEFAULT_OUTPUT_STEM = "aggregated_metrics_table"
FILENAME_MARKERS = ["_results_standard_metrics_", "_standard_metrics_"]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            try:
                item = json.loads(stripped_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            if isinstance(item, dict):
                items.append(item)
    return items


def load_aggregated_json(path: Path) -> Dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    overall = payload.get("overall")
    if not isinstance(overall, dict):
        return None

    metrics = overall.get("metrics")
    if not isinstance(metrics, dict):
        return None

    return overall


def aggregate_metrics(
    items: Sequence[Dict[str, Any]],
    metrics: Sequence[str],
    strict: bool = False,
) -> Dict[str, float | int | None]:
    collectors: Dict[str, List[float]] = {metric: [] for metric in metrics}

    for item in items:
        for metric in metrics:
            value = item.get(metric)
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                collectors[metric].append(float(value))

    missing_metrics = [metric for metric, values in collectors.items() if not values]
    if strict and missing_metrics:
        joined_metrics = ", ".join(missing_metrics)
        raise ValueError(f"Missing numeric values for requested metrics: {joined_metrics}")

    row: Dict[str, float | int | None] = {"num_examples": len(items)}
    for metric, values in collectors.items():
        row[metric] = mean(values) if values else None
    return row


def get_numeric_value(item: Dict[str, Any], key: str) -> float | None:
    value = item.get(key)
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return float(value)

    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get(key)
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            return float(value)

    return None


def normalize_max_value(value: float) -> float | int:
    return int(value) if value.is_integer() else value


def aggregate_extra_columns(items: Sequence[Dict[str, Any]]) -> Dict[str, float | int | None]:
    rollout_values = [
        value for item in items if (value := get_numeric_value(item, "num_rollouts")) is not None
    ]
    input_token_values = [
        value for item in items if (value := get_numeric_value(item, "input_tokens")) is not None
    ]
    output_token_values = [
        value for item in items if (value := get_numeric_value(item, "output_tokens")) is not None
    ]

    return {
        "mean_num_rollouts": mean(rollout_values) if rollout_values else None,
        "max_num_rollouts": normalize_max_value(max(rollout_values)) if rollout_values else None,
        "mean_input_tokens": mean(input_token_values) if input_token_values else None,
        "max_input_tokens": normalize_max_value(max(input_token_values)) if input_token_values else None,
        "mean_output_tokens": mean(output_token_values) if output_token_values else None,
        "max_output_tokens": normalize_max_value(max(output_token_values)) if output_token_values else None,
    }


def resolve_output_path(input_dir: Path, output: str | None, output_format: str) -> Path:
    if output:
        return Path(output)

    suffix = ".json" if output_format == "json" else ".csv"
    return input_dir / f"{DEFAULT_OUTPUT_STEM}{suffix}"


def discover_input_files(
    input_dir: Path,
    glob_pattern: str,
    recursive: bool,
    output_path: Path,
) -> List[Path]:
    candidates = input_dir.rglob(glob_pattern) if recursive else input_dir.glob(glob_pattern)
    resolved_output = output_path.resolve()
    files = [
        path
        for path in candidates
        if path.is_file() and path.resolve() != resolved_output
    ]
    return sorted(files)


def make_file_label(path: Path, input_dir: Path, recursive: bool) -> str:
    if recursive:
        return path.relative_to(input_dir).as_posix()
    return path.name


def parse_dataset_experiment_model(path: Path) -> Dict[str, str]:
    stem = path.stem

    marker_used = None
    for marker in FILENAME_MARKERS:
        if marker in stem:
            marker_used = marker
            break

    if marker_used is None:
        return {"dataset": "", "experiment": "", "model": ""}

    left_part, model = stem.split(marker_used, maxsplit=1)
    if "_" not in left_part:
        return {"dataset": "", "experiment": "", "model": model}

    dataset, experiment = left_part.split("_", maxsplit=1)
    return {"dataset": dataset, "experiment": experiment, "model": model}


def build_table_rows(
    input_dir: Path,
    files: Sequence[Path],
    metrics: Sequence[str],
    recursive: bool,
    strict: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for path in files:
        try:
            overall = load_aggregated_json(path)
            if overall is not None:
                raw_num_examples = overall.get("num_examples")
                num_examples = raw_num_examples if isinstance(raw_num_examples, int) else None
                overall_metrics = overall.get("metrics", {})
                metric_values = {"num_examples": num_examples}
                for metric in metrics:
                    value = overall_metrics.get(metric)
                    if isinstance(value, (float, int)) and not isinstance(value, bool):
                        metric_values[metric] = float(value)
                    else:
                        metric_values[metric] = None
                if strict:
                    missing_metrics = [metric for metric in metrics if metric_values[metric] is None]
                    if missing_metrics:
                        joined_metrics = ", ".join(missing_metrics)
                        raise ValueError(
                            f"Missing numeric values for requested metrics: {joined_metrics}"
                        )
                extra_values = {column: None for column in EXTRA_AGGREGATE_COLUMNS}
            else:
                items = load_jsonl(path)
                metric_values = aggregate_metrics(items, metrics, strict=strict)
                extra_values = aggregate_extra_columns(items)
        except ValueError as exc:
            raise ValueError(f"Failed to aggregate {path}: {exc}") from exc

        row: Dict[str, Any] = {
            "file": make_file_label(path, input_dir, recursive),
            "num_examples": metric_values["num_examples"],
        }
        row.update(parse_dataset_experiment_model(path))
        row.update(extra_values)
        for metric in metrics:
            row[metric] = metric_values[metric]
        rows.append(row)

    return rows


def build_dataset_tables(
    rows: Sequence[Dict[str, Any]],
    metrics: Sequence[str],
    top_k: int | None,
    sort_metric: str,
) -> Dict[str, List[Dict[str, Any]]]:
    datasets = sorted(
        {
            row["dataset"]
            for row in rows
            if isinstance(row.get("dataset"), str) and row["dataset"]
        }
    )
    tables: Dict[str, List[Dict[str, Any]]] = {}

    sort_metric_display = DISPLAY_METRIC_NAMES.get(sort_metric, sort_metric)

    for dataset in datasets:
        dataset_rows = [row for row in rows if row.get("dataset") == dataset]

        table_rows: List[Dict[str, Any]] = []
        seen_experiments: set[str] = set()
        for row in dataset_rows:
            experiment = row.get("experiment")
            if not isinstance(experiment, str) or not experiment:
                continue
            if experiment in seen_experiments:
                raise ValueError(
                    f"Duplicate experiment {experiment!r} found for dataset {dataset!r}."
                )
            seen_experiments.add(experiment)

            table_row: Dict[str, Any] = {"experiment": experiment}
            for metric in metrics:
                table_row[DISPLAY_METRIC_NAMES.get(metric, metric)] = row.get(metric)
            table_rows.append(table_row)

        table_rows.sort(
            key=lambda row: row.get(sort_metric_display)
            if isinstance(row.get(sort_metric_display), (float, int))
            else float("-inf"),
            reverse=True,
        )
        if top_k is not None:
            table_rows = table_rows[:top_k]

        tables[dataset] = table_rows

    return tables


def write_csv(path: Path, headers: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(headers))
        writer.writeheader()
        writer.writerows(rows)


def write_dataset_csvs(
    output_path: Path,
    rows: Sequence[Dict[str, Any]],
    metrics: Sequence[str],
    top_k: int | None,
    sort_metric: str,
) -> List[Path]:
    dataset_tables = build_dataset_tables(rows, metrics, top_k=top_k, sort_metric=sort_metric)
    written_paths: List[Path] = []

    output_dir = output_path.parent
    output_stem = output_path.stem if output_path.suffix == ".csv" else DEFAULT_DATASET_OUTPUT_PREFIX
    headers = ["experiment", *[DISPLAY_METRIC_NAMES.get(metric, metric) for metric in metrics]]

    for dataset, dataset_rows in dataset_tables.items():
        dataset_output_path = output_dir / f"{output_stem}_{dataset}.csv"
        write_csv(dataset_output_path, headers, dataset_rows)
        written_paths.append(dataset_output_path)

    return written_paths


def write_dataset_plots(
    output_path: Path,
    rows: Sequence[Dict[str, Any]],
    metrics: Sequence[str],
    top_k: int | None,
    sort_metric: str,
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]
    except ImportError as exc:
        raise RuntimeError(
            "Plot generation requires matplotlib. Install it or run with --no-plot."
        ) from exc

    dataset_tables = build_dataset_tables(rows, metrics, top_k=top_k, sort_metric=sort_metric)
    written_paths: List[Path] = []

    output_dir = output_path.parent
    output_stem = output_path.stem if output_path.suffix == ".csv" else DEFAULT_DATASET_OUTPUT_PREFIX
    sort_metric_display = DISPLAY_METRIC_NAMES.get(sort_metric, sort_metric)

    for dataset, dataset_rows in dataset_tables.items():
        experiments: List[str] = []
        values: List[float] = []
        for row in dataset_rows:
            value = row.get(sort_metric_display)
            if isinstance(value, (float, int)):
                experiments.append(str(row.get("experiment", "")))
                values.append(float(value))

        if not experiments:
            continue

        figure_height = max(4.0, min(0.45 * len(experiments) + 1.0, 18.0))
        fig, ax = plt.subplots(figsize=(12, figure_height))
        ax.barh(experiments, values, color="#2E86AB")
        ax.invert_yaxis()
        ax.set_xlabel(sort_metric_display)
        ax.set_ylabel("experiment")
        ax.set_title(f"{dataset}: Top {len(experiments)} by {sort_metric_display}")
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        fig.tight_layout()

        plot_path = output_dir / f"{output_stem}_{dataset}.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        written_paths.append(plot_path)

    return written_paths


def write_json(path: Path, rows: Sequence[Dict[str, Any]], pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        if pretty:
            json.dump(list(rows), file_handle, indent=2, ensure_ascii=False)
        else:
            json.dump(list(rows), file_handle, ensure_ascii=False)
        file_handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate metric files in a folder into a result table (JSON or JSONL)."
    )
    parser.add_argument("--input-dir", required=True, help="Folder containing metric JSON/JSONL files.")
    parser.add_argument("--output", default=None, help="Output table path. Defaults inside --input-dir.")
    parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format for the aggregated table.",
    )
    parser.add_argument("--glob", default="*.json", help="Glob used to find input files.")
    parser.add_argument("--recursive", action="store_true", help="Search input files recursively.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metric keys to aggregate. Defaults to standard ROUGE fields and BERTScore.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Keep only the top K experiments per dataset in CSV/plot outputs. Use 0 to keep all.",
    )
    parser.add_argument(
        "--sort-metric",
        default=None,
        help="Metric key used to rank rows for top-k selection. Defaults to the first metric in --metrics.",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-dataset PNG bar plots for ranked outputs.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a requested metric has no numeric values in any input file.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {input_dir}")

    if args.top_k < 0:
        raise ValueError("--top-k must be >= 0")
    top_k = None if args.top_k == 0 else args.top_k

    sort_metric = args.sort_metric or args.metrics[0]
    if sort_metric not in args.metrics:
        raise ValueError("--sort-metric must be included in --metrics")

    output_path = resolve_output_path(input_dir, args.output, args.format)
    files = discover_input_files(input_dir, args.glob, args.recursive, output_path)
    if not files:
        raise ValueError(f"No files matched {args.glob!r} in {input_dir}")

    rows = build_table_rows(
        input_dir=input_dir,
        files=files,
        metrics=args.metrics,
        recursive=args.recursive,
        strict=args.strict,
    )

    if args.format == "json":
        write_json(output_path, rows, pretty=args.pretty)
        print(f"Wrote {len(rows)} rows to {output_path}")
    else:
        csv_paths = write_dataset_csvs(
            output_path,
            rows,
            args.metrics,
            top_k=top_k,
            sort_metric=sort_metric,
        )
        print(f"Wrote {len(csv_paths)} dataset CSV files to {output_path.parent}")
        if args.plot:
            plot_paths = write_dataset_plots(
                output_path,
                rows,
                args.metrics,
                top_k=top_k,
                sort_metric=sort_metric,
            )
            print(f"Wrote {len(plot_paths)} dataset plot files to {output_path.parent}")


if __name__ == "__main__":
    main()
