import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


# --- CONFIG ---
INPUT_DIR = "results/all/qwen_3_5_2b/fact"
OUTPUT_DIR = "results/all/qwen_3_5_2b/fact"
INPUT_GLOB = "*.json"

# Expected filename pattern:
# <dataset>_<experiment>_fact_score_<model>.json
# <dataset>_<experiment>_fact_scores_<model>.json
FILENAME_MARKERS = ["_fact_score_", "_fact_scores_"]
OUTPUT_PREFIX = "factuality_table"


def _to_float(value: Any) -> float | None:
	if isinstance(value, (int, float)) and not isinstance(value, bool):
		return float(value)
	return None


def _parse_dataset_experiment(path: Path) -> Dict[str, str]:
	stem = path.stem

	marker_used = None
	for marker in FILENAME_MARKERS:
		if marker in stem:
			marker_used = marker
			break

	if marker_used is None:
		return {"dataset": "", "experiment": ""}

	left_part = stem.split(marker_used, maxsplit=1)[0]
	if "_" not in left_part:
		return {"dataset": "", "experiment": left_part}

	dataset, experiment = left_part.split("_", maxsplit=1)
	return {"dataset": dataset, "experiment": experiment}


def _load_factuality_metrics(path: Path) -> Dict[str, float]:
	with path.open("r", encoding="utf-8") as f:
		payload = json.load(f)

	if not isinstance(payload, dict):
		raise ValueError(f"Top-level JSON must be an object in {path}")

	overall = payload.get("overall")
	if not isinstance(overall, dict):
		raise ValueError(f"Missing 'overall' object in {path}")

	metrics = overall.get("metrics")
	if not isinstance(metrics, dict):
		raise ValueError(f"Missing 'overall.metrics' object in {path}")

	recall = _to_float(metrics.get("fact_claim_recall"))
	precision = _to_float(metrics.get("fact_claim_precision"))
	f1 = _to_float(metrics.get("fact_claim_f1"))

	if recall is None or precision is None or f1 is None:
		raise ValueError(
			"Missing/non-numeric factuality metric(s) in "
			f"{path}: expected fact_claim_recall, fact_claim_precision, fact_claim_f1"
		)

	return {
		"claim_recall": recall,
		"claim_precision": precision,
		"claim_f1": f1,
	}


def _discover_files(input_dir: Path, glob_pattern: str) -> List[Path]:
	return sorted(path for path in input_dir.glob(glob_pattern) if path.is_file())


def _write_csv(path: Path, headers: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=list(headers))
		writer.writeheader()
		writer.writerows(rows)


def main() -> None:
	input_dir = Path(INPUT_DIR).expanduser().resolve()
	output_dir = Path(OUTPUT_DIR).expanduser().resolve()

	if not input_dir.is_dir():
		raise ValueError(f"Input directory does not exist: {input_dir}")

	files = _discover_files(input_dir, INPUT_GLOB)
	if not files:
		raise ValueError(f"No files found with pattern {INPUT_GLOB!r} in {input_dir}")

	rows: List[Dict[str, Any]] = []
	for path in files:
		parsed = _parse_dataset_experiment(path)
		if not parsed["dataset"]:
			continue

		metrics = _load_factuality_metrics(path)
		rows.append(
			{
				"dataset": parsed["dataset"],
				"experiment": parsed["experiment"],
				"claim_recall": metrics["claim_recall"],
				"claim_precision": metrics["claim_precision"],
				"claim_f1": metrics["claim_f1"],
			}
		)

	if not rows:
		raise ValueError(
			"No rows were produced. Ensure filenames follow <dataset>_<experiment>_fact_score_<model>.json"
		)

	grouped_rows: Dict[str, List[Dict[str, Any]]] = {}
	for row in rows:
		grouped_rows.setdefault(row["dataset"], []).append(row)

	headers = ["experiment", "claim_recall", "claim_precision", "claim_f1"]

	written_count = 0
	for dataset, dataset_rows in sorted(grouped_rows.items()):
		dataset_rows.sort(key=lambda row: str(row["experiment"]))
		csv_rows = [{header: row.get(header) for header in headers} for row in dataset_rows]

		output_path = output_dir / f"{OUTPUT_PREFIX}_{dataset}.csv"
		_write_csv(output_path, headers, csv_rows)
		written_count += 1

	print(f"Wrote {written_count} dataset CSV file(s) to {output_dir}")


if __name__ == "__main__":
	main()
