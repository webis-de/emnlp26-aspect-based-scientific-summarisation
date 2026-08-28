import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from build_aspect_results_explorer import (
    build_view_model,
    discover_input_files,
    pipeline_sort_key,
    write_html,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a single-file interactive HTML explorer with embedded data, "
            "sampling a fixed number of datapoints per dataset."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing files named like dataset_pipeline_model_rouge_bs.json(l) and *_claim.json(l).",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to <input-dir>/aspect_results_explorer_sampled_25.html",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30,
        help="Number of datapoints to sample per dataset (default: 30).",
    )
    parser.add_argument(
        "--band-size",
        type=int,
        default=10,
        help="Target datapoints per quality band (low/medium/high), default: 10.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def mean(values: List[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def datapoint_quality(row: Dict[str, Any]) -> Tuple[float, float, float]:
    rouge_values: List[float] = []
    claim_values: List[float] = []
    for system in row.get("systems", []):
        scores = system.get("scores", {}) if isinstance(system, dict) else {}
        rouge_lsum = scores.get("rougeLsum_fmeasure") if isinstance(scores, dict) else None
        claim_f1 = scores.get("fact_claim_f1") if isinstance(scores, dict) else None
        if isinstance(rouge_lsum, (int, float)):
            rouge_values.append(float(rouge_lsum))
        if isinstance(claim_f1, (int, float)):
            claim_values.append(float(claim_f1))

    rouge_mean = mean(rouge_values)
    claim_mean = mean(claim_values)

    # Treat missing dimensions as 0.0 to bias toward lower-quality bins.
    rouge_component = rouge_mean if rouge_mean is not None else 0.0
    claim_component = claim_mean if claim_mean is not None else 0.0
    composite = (rouge_component + claim_component) / 2.0
    return composite, rouge_component, claim_component


def split_quality_bands(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    sorted_rows = sorted(
        rows,
        key=lambda item: (
            datapoint_quality(item)[0],
            datapoint_quality(item)[1],
            datapoint_quality(item)[2],
            str(item.get("unique_id") or ""),
            str(item.get("aspect_name") or ""),
        ),
    )
    n_rows = len(sorted_rows)
    if n_rows == 0:
        return [], [], []

    low_end = n_rows // 3
    medium_end = (2 * n_rows) // 3

    low = sorted_rows[:low_end]
    medium = sorted_rows[low_end:medium_end]
    high = sorted_rows[medium_end:]
    return low, medium, high


def take_random(rows: List[Dict[str, Any]], k: int, rng: random.Random) -> List[Dict[str, Any]]:
    if k <= 0 or not rows:
        return []
    if len(rows) <= k:
        return rows[:]
    return rng.sample(rows, k)


def sample_stratified_rows(
    rows: List[Dict[str, Any]],
    sample_size: int,
    band_size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if len(rows) <= sample_size:
        return sorted(
            rows,
            key=lambda item: (
                str(item.get("unique_id") or ""),
                str(item.get("aspect_name") or ""),
            ),
        )

    rng = random.Random(seed)
    low, medium, high = split_quality_bands(rows)

    selected: List[Dict[str, Any]] = []
    selected.extend(take_random(low, band_size, rng))
    selected.extend(take_random(medium, band_size, rng))
    selected.extend(take_random(high, band_size, rng))

    selected_keys = {
        (str(item.get("unique_id") or ""), str(item.get("aspect_name") or ""))
        for item in selected
    }
    if len(selected) < sample_size:
        pool = [
            item
            for item in rows
            if (str(item.get("unique_id") or ""), str(item.get("aspect_name") or "")) not in selected_keys
        ]
        remainder = sample_size - len(selected)
        selected.extend(take_random(pool, remainder, rng))

    if len(selected) > sample_size:
        selected = take_random(selected, sample_size, rng)

    return sorted(
        selected,
        key=lambda item: (
            str(item.get("unique_id") or ""),
            str(item.get("aspect_name") or ""),
        ),
    )


def sample_view_model(view_model: Dict[str, Any], sample_size: int, band_size: int, seed: int) -> Dict[str, Any]:
    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")
    if band_size <= 0:
        raise ValueError("band_size must be > 0")

    datapoints = view_model.get("datapoints", [])
    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for row in datapoints:
        dataset = str(row.get("dataset") or "")
        by_dataset.setdefault(dataset, []).append(row)

    sampled_rows: List[Dict[str, Any]] = []
    for dataset in sorted(by_dataset.keys()):
        rows = sorted(
            by_dataset[dataset],
            key=lambda item: (
                str(item.get("unique_id") or ""),
                str(item.get("aspect_name") or ""),
            ),
        )
        if len(rows) <= sample_size:
            sampled_rows.extend(rows)
            continue

        sampled_rows.extend(
            sample_stratified_rows(
                rows=rows,
                sample_size=sample_size,
                band_size=band_size,
                seed=seed,
            )
        )

    datasets = sorted({str(row.get("dataset") or "") for row in sampled_rows if str(row.get("dataset") or "")})
    pipelines = sorted(
        {
            str(system.get("pipeline"))
            for row in sampled_rows
            for system in row.get("systems", [])
            if isinstance(system.get("pipeline"), str) and str(system.get("pipeline"))
        },
        key=pipeline_sort_key,
    )

    return {
        "datasets": datasets,
        "pipelines": pipelines,
        "datapoints": sampled_rows,
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_html = args.output_html or input_dir / "aspect_results_explorer_sampled_30.html"

    descriptors = discover_input_files(input_dir)
    full_view_model = build_view_model(descriptors)
    sampled_view_model = sample_view_model(full_view_model, args.sample_size, args.band_size, args.seed)

    write_html(sampled_view_model, output_html)

    print(f"Explorer written to: {output_html}")
    print(f"Datasets: {len(sampled_view_model['datasets'])}")
    print(f"Datapoints: {len(sampled_view_model['datapoints'])}")
    print(f"Sample size per dataset: {args.sample_size}")
    print(f"Band size (low/medium/high): {args.band_size}")


if __name__ == "__main__":
    main()
