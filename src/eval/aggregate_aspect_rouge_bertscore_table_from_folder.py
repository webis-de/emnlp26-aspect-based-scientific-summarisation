"""
Example usage:
python src/eval/aggregate_aspect_rouge_bertscore_table_from_folder.py
    --input-dir 2a2s/results/e2a/rebuttal_sensitivity/
    
Aggregate aspect-level Rouge and BERTScore metrics from a folder of JSON files.

"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


FILENAME_MARKER = "_results_standard_metrics_"
DEFAULT_GLOB = "*_results_standard_metrics_*.json"
DEFAULT_OUTPUT_NAME = "aggregated_aspect_metric_tables.csv"
DEFAULT_METRICS = [
    "rouge1_fmeasure",
    "rouge2_fmeasure",
    "rougeL_fmeasure",
    "rougeLsum_fmeasure",
    "bertscore",
]
DEFAULT_PIPELINE_ORDER = ["zs", "e2a", "rag", "cod", "self_refine", "2a2s"]
DEFAULT_PIPELINE_LABELS = {
    "zs": "ZS",
    "e2a": "E2A",
    "rag": "RAG",
    "cod": "COD",
    "self_refine": "SR",
    "2a2s": "2A2S",
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    if not isinstance(payload, dict):
        raise ValueError("Expected top-level JSON object")
    return payload


def parse_dataset_pipeline_model(path: Path) -> Dict[str, str]:
    stem = path.stem
    if FILENAME_MARKER not in stem:
        return {
            "dataset": "",
            "pipeline": stem,
            "model": "",
        }

    left_part, model = stem.split(FILENAME_MARKER, maxsplit=1)
    if "_" not in left_part:
        return {
            "dataset": "",
            "pipeline": left_part,
            "model": model,
        }

    dataset, pipeline = left_part.split("_", maxsplit=1)
    return {
        "dataset": dataset,
        "pipeline": pipeline,
        "model": model,
    }


def derive_datasets_from_files(files: Sequence[Path]) -> List[str]:
    datasets: set[str] = set()
    for path in files:
        parsed = parse_dataset_pipeline_model(path)
        dataset = parsed.get("dataset")
        if isinstance(dataset, str) and dataset:
            datasets.add(dataset)
    return sorted(datasets)


def discover_files(input_dir: Path, glob_pattern: str, recursive: bool, output_path: Path) -> List[Path]:
    candidates = input_dir.rglob(glob_pattern) if recursive else input_dir.glob(glob_pattern)
    resolved_output = output_path.resolve()
    files = [path for path in candidates if path.is_file() and path.resolve() != resolved_output]
    return sorted(files)


def get_metric(metrics: Dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def collect_rows(
    files: Sequence[Path],
    requested_metrics: Sequence[str],
    dataset_filter: str | None,
) -> tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    aspect_order: List[str] = []
    seen_aspects: set[str] = set()

    for path in files:
        payload = load_json(path)
        by_aspect = payload.get("by_aspect")
        if not isinstance(by_aspect, dict):
            raise ValueError(f"Missing or invalid by_aspect in {path}")

        parsed_name = parse_dataset_pipeline_model(path)
        dataset = parsed_name["dataset"]
        if dataset_filter and dataset != dataset_filter:
            continue

        row: Dict[str, Any] = {
            "pipeline": parsed_name["pipeline"],
            "dataset": dataset,
            "model": parsed_name["model"],
            "file": path.name,
        }

        for aspect_name, aspect_payload in by_aspect.items():
            if not isinstance(aspect_payload, dict):
                continue
            aspect_metrics = aspect_payload.get("metrics")
            if not isinstance(aspect_metrics, dict):
                continue

            aspect = str(aspect_name)
            if aspect not in seen_aspects:
                seen_aspects.add(aspect)
                aspect_order.append(aspect)

            for metric in requested_metrics:
                row[f"{aspect}__{metric}"] = get_metric(aspect_metrics, metric)

        rows.append(row)

    return rows, aspect_order


def normalize_pipeline_key(name: str) -> str:
    key = name.strip().lower()
    key = key.replace("-", "_")
    key = key.replace(" ", "_")
    return key


def pipeline_display_name(pipeline: str, label_map: Dict[str, str]) -> str:
    normalized = normalize_pipeline_key(pipeline)
    if normalized in label_map:
        return label_map[normalized]
    return pipeline


def build_pipeline_columns(
    rows: Sequence[Dict[str, Any]],
    pipeline_order: Sequence[str],
) -> List[str]:
    seen: set[str] = set()
    discovered: List[str] = []
    for row in rows:
        raw = row.get("pipeline")
        if not isinstance(raw, str):
            continue
        normalized = normalize_pipeline_key(raw)
        if normalized not in seen:
            seen.add(normalized)
            discovered.append(normalized)

    ordered: List[str] = []
    for pipeline in pipeline_order:
        normalized = normalize_pipeline_key(pipeline)
        if normalized in seen and normalized not in ordered:
            ordered.append(normalized)

    for pipeline in sorted(discovered):
        if pipeline not in ordered:
            ordered.append(pipeline)

    return ordered


def build_metric_matrix(
    rows: Sequence[Dict[str, Any]],
    aspects: Sequence[str],
    metrics: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, float | None]]]:
    matrix: Dict[str, Dict[str, Dict[str, float | None]]] = {
        metric: {aspect: {} for aspect in aspects}
        for metric in metrics
    }

    for row in rows:
        pipeline_raw = row.get("pipeline")
        if not isinstance(pipeline_raw, str):
            continue
        pipeline = normalize_pipeline_key(pipeline_raw)

        for aspect in aspects:
            for metric in metrics:
                key = f"{aspect}__{metric}"
                value = row.get(key)
                matrix[metric][aspect][pipeline] = value if isinstance(value, float) else None

    return matrix


def format_csv_score(value: float | None, decimals: int) -> str:
    if value is None:
        return ""
    return f"{value:.{decimals}f}"


def metric_output_path(base_output_path: Path, metric: str) -> Path:
    stem = base_output_path.stem
    return base_output_path.with_name(f"{stem}_{metric}.csv")


def write_csv_tables(
    output_path: Path,
    rows: Sequence[Dict[str, Any]],
    aspects: Sequence[str],
    metrics: Sequence[str],
    pipeline_order: Sequence[str],
    label_map: Dict[str, str],
    decimals: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pipelines = build_pipeline_columns(rows, pipeline_order)
    if not pipelines:
        raise ValueError("No pipeline rows found after filtering")

    datasets: List[str] = sorted(
        {
            str(row.get("dataset"))
            for row in rows
            if isinstance(row.get("dataset"), str) and str(row.get("dataset"))
        }
    )
    if not datasets:
        datasets = [""]

    for metric in metrics:
        metric_path = metric_output_path(output_path, metric)
        headers = ["dataset", "aspect", *[pipeline_display_name(pipeline, label_map) for pipeline in pipelines]]

        with metric_path.open("w", encoding="utf-8", newline="") as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(headers)

            for dataset in datasets:
                dataset_rows = [
                    row for row in rows if str(row.get("dataset", "")) == dataset
                ]
                dataset_aspects = [
                    aspect
                    for aspect in aspects
                    if any(
                        isinstance(row.get(f"{aspect}__{metric}"), (int, float))
                        and not isinstance(row.get(f"{aspect}__{metric}"), bool)
                        for row in dataset_rows
                    )
                ]
                matrix = build_metric_matrix(dataset_rows, dataset_aspects, [metric])

                for aspect in dataset_aspects:
                    row_values = [matrix[metric][aspect].get(pipeline) for pipeline in pipelines]
                    writer.writerow(
                        [dataset, aspect, *[format_csv_score(value, decimals) for value in row_values]]
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build CSV aspect tables from aggregated standard-metrics JSON files: "
            "one table per metric with aspect rows and pipeline columns."
        )
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing aggregated metrics JSON files.")
    parser.add_argument("--output", default=None, help="Base output path used to derive per-metric CSV files.")
    parser.add_argument("--glob", default=DEFAULT_GLOB, help="Glob used to find JSON files.")
    parser.add_argument("--recursive", action="store_true", help="Search recursively in --input-dir.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset filter from filename prefix (e.g. aclsum).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to render as separate tables.",
    )
    parser.add_argument(
        "--pipeline-order",
        nargs="+",
        default=DEFAULT_PIPELINE_ORDER,
        help="Preferred pipeline order for columns.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Number of decimal places in table cells.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {input_dir}")

    output_path = Path(args.output) if args.output else input_dir / DEFAULT_OUTPUT_NAME
    files = discover_files(input_dir, args.glob, args.recursive, output_path)
    if not files:
        raise ValueError(f"No files matched {args.glob!r} in {input_dir}")

    rows, aspects = collect_rows(
        files=files,
        requested_metrics=args.metrics,
        dataset_filter=args.dataset,
    )
    if not rows:
        raise ValueError("No rows found after applying filters")
    if not aspects:
        raise ValueError("No aspect metrics found in input files")

    write_csv_tables(
        output_path=output_path,
        rows=rows,
        aspects=aspects,
        metrics=args.metrics,
        pipeline_order=args.pipeline_order,
        label_map=DEFAULT_PIPELINE_LABELS,
        decimals=args.decimals,
    )

    total_files = len(args.metrics)
    print(f"Wrote {total_files} CSV tables next to {output_path.parent}")


if __name__ == "__main__":
    main()
