import json
from pathlib import Path
from statistics import mean, median
from typing import Dict, List


# --- CONFIG ---
# Set these paths directly before running the script.
INPUT_JSONL_PATH = "results/2a2s/full/qwen_qwen3.5_9b/pmc/long/pmc_long_test_full_final_results.jsonl"
OUTPUT_JSON_PATH = "results/all/token_usage/pmc_2a2s_token_metrics_qwen_3_5_9b.json"


def _to_number(value: object, field_name: str, line_no: int) -> float:
	if isinstance(value, (int, float)):
		return float(value)
	raise ValueError(
		f"Invalid '{field_name}' at line {line_no}: expected int/float, got {type(value).__name__}"
	)


def _get_field(row: Dict[str, object], field_name: str) -> object:
	metadata = row.get("metadata")
	if isinstance(metadata, dict) and field_name in metadata:
		return metadata[field_name]
	if field_name in row:
		return row[field_name]
	return None


def _read_values(input_jsonl: Path) -> Dict[str, List[float]]:
	input_values: List[float] = []
	output_values: List[float] = []
	rollout_values: List[float] = []

	with input_jsonl.open("r", encoding="utf-8") as f:
		for idx, raw_line in enumerate(f, start=1):
			line = raw_line.strip()
			if not line:
				continue

			try:
				row = json.loads(line)
			except json.JSONDecodeError as exc:
				raise ValueError(f"Invalid JSON at line {idx}: {exc}") from exc

			if not isinstance(row, dict):
				raise ValueError(f"Invalid row at line {idx}: expected JSON object")

			input_tokens = _get_field(row, "input_tokens")
			output_tokens = _get_field(row, "output_tokens")
			num_rollouts = _get_field(row, "num_rollouts")

			if input_tokens is None or output_tokens is None or num_rollouts is None:
				raise ValueError(
					"Missing required key(s) at line "
					f"{idx}: expected metadata.input_tokens/output_tokens/num_rollouts "
					"(or top-level input_tokens/output_tokens/num_rollouts)"
				)

			input_values.append(_to_number(input_tokens, "input_tokens", idx))
			output_values.append(_to_number(output_tokens, "output_tokens", idx))
			rollout_values.append(_to_number(num_rollouts, "num_rollouts", idx))

	if not input_values:
		raise ValueError("No valid JSONL rows found in input file.")

	return {
		"input_tokens": input_values,
		"output_tokens": output_values,
		"num_rollouts": rollout_values,
	}


def _stats(values: List[float]) -> Dict[str, float]:
	return {
		"min": float(min(values)),
		"max": float(max(values)),
		"mean": float(mean(values)),
		"median": float(median(values)),
	}


def _build_summary(values: Dict[str, List[float]]) -> Dict[str, float]:
	in_stats = _stats(values["input_tokens"])
	out_stats = _stats(values["output_tokens"])
	rollout_stats = _stats(values["num_rollouts"])

	return {
		"min_input_tokens": in_stats["min"],
		"max_input_tokens": in_stats["max"],
		"mean_input_tokens": in_stats["mean"],
		"median_input_tokens": in_stats["median"],
		"min_output_tokens": out_stats["min"],
		"max_output_tokens": out_stats["max"],
		"mean_output_tokens": out_stats["mean"],
		"median_output_tokens": out_stats["median"],
		"min_num_rollouts": rollout_stats["min"],
		"max_num_rollouts": rollout_stats["max"],
		"mean_num_rollouts": rollout_stats["mean"],
		"median_num_rollouts": rollout_stats["median"],
	}


def main() -> None:
	input_jsonl = Path(INPUT_JSONL_PATH).expanduser().resolve()
	output_json = Path(OUTPUT_JSON_PATH).expanduser().resolve()

	values = _read_values(input_jsonl)
	summary = _build_summary(values)

	output_json.parent.mkdir(parents=True, exist_ok=True)
	with output_json.open("w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2, ensure_ascii=False)
		f.write("\n")


if __name__ == "__main__":
	main()
