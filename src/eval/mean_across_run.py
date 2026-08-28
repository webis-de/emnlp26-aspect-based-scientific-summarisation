import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List

# --- CONFIG ---
# Set exactly 3 input files (one per run) and one output file.
INPUT_JSON_PATHS = [
    "results/zs/qwen3_32b/run_1/pmc/short_temp_0p1/vllm_qwen3_32b_pmc_short_sampled_results_standard_metrics_aggregated_aspect_wise.json",
    "results/zs/qwen3_32b/run_2/pmc/short_temp_0p1/vllm_qwen3_32b_pmc_short_sampled_results_standard_metrics_aggregated_aspect_wise.json",
    "results/zs/qwen3_32b/run_3/pmc/short_temp_0p1/vllm_qwen3_32b_pmc_short_sampled_results_standard_metrics_aggregated_aspect_wise.json",
]
OUTPUT_JSON_PATH = "results/zs/qwen3_32b/pmc_short_mean_std_across_runs.json"
COMPACT_JSON = False


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level object in {path}")
    return data


def _safe_stats(values: Iterable[float]) -> Dict[str, float]:
    seq = list(values)
    if not seq:
        return {"mean": 0.0, "stddev": 0.0, "count": 0}
    if len(seq) == 1:
        return {"mean": float(seq[0]), "stddev": 0.0, "count": 1}
    return {"mean": float(mean(seq)), "stddev": float(pstdev(seq)), "count": len(seq)}


def _collect_metric_values(run_items: List[Dict[str, Any]], path: List[str]) -> Dict[str, List[float]]:
    # Collect union of metric names across runs, then gather available numeric values.
    metric_names: set[str] = set()
    for item in run_items:
        metrics = item
        for key in path:
            metrics = metrics.get(key, {}) if isinstance(metrics, dict) else {}
        if isinstance(metrics, dict):
            metric_names.update(metrics.keys())

    values_by_metric: Dict[str, List[float]] = {name: [] for name in metric_names}
    for item in run_items:
        metrics = item
        for key in path:
            metrics = metrics.get(key, {}) if isinstance(metrics, dict) else {}
        if not isinstance(metrics, dict):
            continue
        for name in metric_names:
            value = metrics.get(name)
            if isinstance(value, (float, int)):
                values_by_metric[name].append(float(value))

    return values_by_metric


def aggregate_block(run_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    num_examples_values: List[float] = []
    for item in run_items:
        num_examples = item.get("num_examples") if isinstance(item, dict) else None
        if isinstance(num_examples, (float, int)):
            num_examples_values.append(float(num_examples))

    metric_values = _collect_metric_values(run_items, ["metrics"])
    metric_stats = {name: _safe_stats(values) for name, values in sorted(metric_values.items())}

    return {
        "num_runs": len(run_items),
        "num_examples": _safe_stats(num_examples_values),
        "metrics": metric_stats,
    }


def aggregate_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    overall_items = [run.get("overall", {}) for run in runs]
    by_aspect_names: set[str] = set()
    for run in runs:
        by_aspect = run.get("by_aspect", {})
        if isinstance(by_aspect, dict):
            by_aspect_names.update(str(name) for name in by_aspect.keys())

    by_aspect_agg: Dict[str, Any] = {}
    for aspect_name in sorted(by_aspect_names):
        aspect_items: List[Dict[str, Any]] = []
        for run in runs:
            by_aspect = run.get("by_aspect", {})
            if isinstance(by_aspect, dict):
                payload = by_aspect.get(aspect_name)
                if isinstance(payload, dict):
                    aspect_items.append(payload)
        by_aspect_agg[aspect_name] = aggregate_block(aspect_items)

    return {
        "num_runs": len(runs),
        "input_files": [],
        "overall": aggregate_block(overall_items),
        "by_aspect": by_aspect_agg,
    }


def main() -> None:
    if len(INPUT_JSON_PATHS) != 3:
        raise ValueError("INPUT_JSON_PATHS must contain exactly 3 file paths.")

    input_paths = [str(Path(p).expanduser().resolve()) for p in INPUT_JSON_PATHS]
    output_path = str(Path(OUTPUT_JSON_PATH).expanduser().resolve())

    runs = [load_json(path) for path in input_paths]
    aggregated = aggregate_runs(runs)
    aggregated["input_files"] = input_paths

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        if COMPACT_JSON:
            json.dump(aggregated, f, ensure_ascii=False)
        else:
            json.dump(aggregated, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
