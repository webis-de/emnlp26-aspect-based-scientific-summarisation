import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


# --- CONFIG ---
# Folder containing aggregated token-usage JSON files.
INPUT_DIR = "results/all/olmo_3_7b/token_usage"

# Output directory for per-dataset CSV tables.
OUTPUT_DIR = "results/all/olmo_3_7b/token_usage"

# Input file pattern inside INPUT_DIR.
INPUT_GLOB = "*.json"

# Filename markers supported for metadata extraction.
# Expected examples:
#   pmc_2a2s_token_metrics_qwen_3_5_9b.json
#   aclsum_zs_results_standard_metrics_qwen_3_5_9b.json
FILENAME_MARKERS = ["_token_metrics_", "_results_standard_metrics_"]

# Prefix for output CSVs: <OUTPUT_PREFIX>_<dataset>.csv
OUTPUT_PREFIX = "token_usage_table"


REQUIRED_FIELDS = [
	"min_input_tokens",
	"max_input_tokens",
	"mean_input_tokens",
	"median_input_tokens",
	"min_output_tokens",
	"max_output_tokens",
	"mean_output_tokens",
	"median_output_tokens",
	"min_num_rollouts",
	"max_num_rollouts",
	"mean_num_rollouts",
	"median_num_rollouts",
]


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


def _load_aggregated_json(path: Path) -> Dict[str, float]:
	with path.open("r", encoding="utf-8") as f:
		payload = json.load(f)

	if not isinstance(payload, dict):
		raise ValueError(f"Top-level JSON must be an object in {path}")

	values: Dict[str, float] = {}
	for key in REQUIRED_FIELDS:
		value = _to_float(payload.get(key))
		if value is None:
			raise ValueError(f"Missing or non-numeric field {key!r} in {path}")
		values[key] = value
	return values


def _build_row(path: Path) -> Dict[str, Any]:
	parsed = _parse_dataset_experiment(path)
	values = _load_aggregated_json(path)

	# Total token usage columns are derived as input + output.
	values["min_total_tokens"] = values["min_input_tokens"] + values["min_output_tokens"]
	values["max_total_tokens"] = values["max_input_tokens"] + values["max_output_tokens"]
	values["mean_total_tokens"] = values["mean_input_tokens"] + values["mean_output_tokens"]
	values["median_total_tokens"] = values["median_input_tokens"] + values["median_output_tokens"]

	row: Dict[str, Any] = {
		"file": path.name,
		"dataset": parsed["dataset"],
		"experiment": parsed["experiment"],
	}
	row.update(values)
	return row


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

	rows = [_build_row(path) for path in files]

	grouped_rows: Dict[str, List[Dict[str, Any]]] = {}
	for row in rows:
		dataset = row.get("dataset")
		if not isinstance(dataset, str) or not dataset:
			continue
		grouped_rows.setdefault(dataset, []).append(row)

	if not grouped_rows:
		raise ValueError(
			"No dataset rows created. Ensure file names follow <dataset>_<experiment>_* pattern."
		)

	headers = [
		"experiment",
		"min_input_tokens",
		"max_input_tokens",
		"mean_input_tokens",
		"median_input_tokens",
		"min_output_tokens",
		"max_output_tokens",
		"mean_output_tokens",
		"median_output_tokens",
		"min_num_rollouts",
		"max_num_rollouts",
		"mean_num_rollouts",
		"median_num_rollouts",
		"min_total_tokens",
		"max_total_tokens",
		"mean_total_tokens",
		"median_total_tokens",
	]

	written_count = 0
	for dataset, dataset_rows in sorted(grouped_rows.items()):
		dataset_rows.sort(key=lambda row: str(row.get("experiment", "")))

		csv_rows = []
		for row in dataset_rows:
			csv_rows.append({header: row.get(header) for header in headers})

		output_path = output_dir / f"{OUTPUT_PREFIX}_{dataset}.csv"
		_write_csv(output_path, headers, csv_rows)
		written_count += 1

	print(f"Wrote {written_count} dataset CSV file(s) to {output_dir}")


if __name__ == "__main__":
	main()
