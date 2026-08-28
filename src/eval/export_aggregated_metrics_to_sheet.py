import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

INPUT_JSON_PATH = "results/2a2s/planner_extractor_writer/qwen_qwen3_8b/scholarsum_full/long/scholarsum_full_long_test_planner_extractor_writer_final_results_standard_metrics_aggregated_aspect_wise.json"
OUTPUT_CSV_PATH = INPUT_JSON_PATH.replace(".json", ".csv")

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_aggregated_metrics(data: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    metric_keys: List[str] = []

    def add_row(scope: str, name: str, payload: Dict[str, Any]) -> None:
        nonlocal metric_keys
        metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
        if not metric_keys:
            metric_keys = sorted(metrics.keys())
        row: Dict[str, Any] = {
            "scope": scope,
            "name": name,
            "num_examples": payload.get("num_examples", 0) if isinstance(payload, dict) else 0,
        }
        for key in metric_keys:
            row[key] = metrics.get(key, 0.0)
        rows.append(row)

    overall = data.get("overall")
    if isinstance(overall, dict):
        add_row("overall", "overall", overall)

    by_aspect = data.get("by_aspect", {})
    if isinstance(by_aspect, dict):
        for aspect_name, payload in by_aspect.items():
            add_row("aspect", str(aspect_name), payload)

    headers = ["scope", "name", "num_examples", *metric_keys]
    return headers, rows


def write_csv(path: str, headers: Iterable[str], rows: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(headers))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    data = load_json(INPUT_JSON_PATH)
    headers, rows = flatten_aggregated_metrics(data)
    write_csv(OUTPUT_CSV_PATH, headers, rows)


if __name__ == "__main__":
    main()
