"""W3 rebuttal helper: extract the frozen rebuttal subsubsample's per-example
scores from an existing full-run scored file — either
`*_standard_metrics_aspect_wise.jsonl` (ROUGE/BERTScore) or
`*_fact_score_aspect_wise.jsonl` (claim recall/precision/F1).

This does NOT recompute any metrics. It only filters an already-scored full-run
file down to the same 100-document rebuttal subsubsample used for the RAG/E2A
sensitivity reruns (see data_utils.load_rebuttal_subsubsample), so that the new
sensitivity configurations (RAG top-k 20, chunk size 512, E2A evidence budget
75/6000) can be compared paired, on the identical case set, against the
original ZS / RAG / E2A / 2A2S full-run scores.

The file type (standard metrics vs fact_score) is auto-detected from the
--input filename suffix; the appropriate default metric keys and output
suffix/infix are chosen automatically. Use --metrics to override.

Matching strategy (reuses the same approach as run_geval_on_rebuttal_subsample.py
and the HASH_ALIGNMENT_FIELDS convention in paired_bootstrap_significance.py):
  1. Exact top-level `unique_id` match, if present in the row (all fact_score
     files and RAG/E2A standard_metrics files have this).
  2. Exact `metadata.unique_id` match, if present.
  3. Fallback: normalized (aspect_name, gold_aspect_summary) hash match against
     the rebuttal subsubsample. This is necessary because ZS's
     `*_standard_metrics_aspect_wise.jsonl` files do not carry unique_id at all
     (only the raw pre-metrics ZS file does).

Usage:
    python src/eval/extract_rebuttal_subsample_metrics.py \\
        --input results/zs/qwen3p5_9b/aclsum/long_temp_0p1/vllm_qwen3p5_9b_aclsum_long_full_results_standard_metrics_aspect_wise.jsonl \\
        --dataset aclsum

    python src/eval/extract_rebuttal_subsample_metrics.py \\
        --input results/zs/qwen3p5_9b/aclsum/long_temp_0p1/vllm_qwen3p5_9b_aclsum_long_full_results_fact_score_aspect_wise.jsonl \\
        --dataset aclsum
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_utils import ASPECT_NORMALIZATION, load_rebuttal_subsubsample

STANDARD_METRICS = [
    "rouge1_precision", "rouge1_recall", "rouge1_fmeasure",
    "rouge2_precision", "rouge2_recall", "rouge2_fmeasure",
    "rougeL_precision", "rougeL_recall", "rougeL_fmeasure",
    "rougeLsum_precision", "rougeLsum_recall", "rougeLsum_fmeasure",
    "bertscore",
]
FACT_SCORE_METRICS = [
    "fact_claim_recall", "fact_claim_precision", "fact_claim_f1",
]

STANDARD_METRICS_INFIX = "_standard_metrics_aspect_wise"
FACT_SCORE_INFIX = "_fact_score_aspect_wise"

OUTPUT_SUFFIX = "_rebuttal_subsample_standard_metrics_aspect_wise"
AGGREGATE_SUFFIX = "_rebuttal_subsample_standard_metrics_aggregated_aspect_wise"
FACT_OUTPUT_SUFFIX = "_rebuttal_subsample_fact_score_aspect_wise"
FACT_AGGREGATE_SUFFIX = "_rebuttal_subsample_fact_score_aggregated_aspect_wise"


def detect_file_kind(input_path: Path) -> str:
    """Return 'fact_score' or 'standard_metrics' based on the input filename."""
    name = input_path.name
    if FACT_SCORE_INFIX in name:
        return "fact_score"
    return "standard_metrics"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    cleaned = "".join(
        ch if ch in ("\n", "\t", "\r") or ord(ch) >= 32 else " " for ch in value
    ).strip()
    return " ".join(cleaned.split()).lower()


def normalize_aspect(dataset_name: str, aspect_name: Any) -> str:
    aspect = str(aspect_name or "").strip()
    normalization = ASPECT_NORMALIZATION.get(dataset_name, {})
    return normalization.get(aspect, aspect)


def hash_key(aspect_name: str, gold_summary: str) -> str:
    payload = json.dumps(
        {"aspect_name": clean_text(aspect_name), "gold_aspect_summary": clean_text(gold_summary)},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_rebuttal_index(dataset_name: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (unique_id_set, hash_key -> unique_id) for the frozen rebuttal subsubsample."""
    grouped = load_rebuttal_subsubsample(dataset_name)
    unique_ids: Dict[str, str] = {}
    hash_index: Dict[str, str] = {}
    for doc_id, records in grouped.items():
        for record in records:
            uid = str(record.get("unique_id") or "")
            if not uid:
                continue
            unique_ids[uid] = uid
            aspect_name = record.get("aspect_name")
            gold_summary = record.get("aspect_summary")
            key = hash_key(aspect_name, gold_summary)
            hash_index[key] = uid
    return unique_ids, hash_index


def row_unique_id(row: Dict[str, Any]) -> str | None:
    uid = row.get("unique_id")
    if uid:
        return str(uid)
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        uid = metadata.get("unique_id")
        if uid:
            return str(uid)
    return None


def match_row(
    row: Dict[str, Any],
    unique_ids: Dict[str, str],
    hash_index: Dict[str, str],
) -> str | None:
    uid = row_unique_id(row)
    if uid and uid in unique_ids:
        return uid
    key = hash_key(row.get("aspect_name"), row.get("gold_aspect_summary"))
    return hash_index.get(key)


def aggregate_metrics(rows: Sequence[Dict[str, Any]], metrics: Sequence[str]) -> Dict[str, Any]:
    collectors: Dict[str, List[float]] = {m: [] for m in metrics}
    for row in rows:
        for m in metrics:
            val = row.get(m)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                collectors[m].append(float(val))
    return {
        "num_examples": len(rows),
        "metrics": {m: (mean(v) if v else 0.0) for m, v in collectors.items()},
    }


def aggregate_metrics_by_aspect(rows: Sequence[Dict[str, Any]], metrics: Sequence[str]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        aspect = str(row.get("aspect_name") or "unknown")
        grouped.setdefault(aspect, []).append(row)
    return {aspect: aggregate_metrics(entries, metrics) for aspect, entries in grouped.items()}


def derive_output_path(input_path: Path, suffix: str, extension: str) -> Path:
    name = input_path.name
    if name.endswith(".jsonl"):
        name = name[: -len(".jsonl")]
    # Avoid stacking "..._standard_metrics_aspect_wise_rebuttal_subsample_..."
    # (or the fact_score equivalent) when --input already ends in the scored-file
    # infix; strip it first so the rebuttal-subsample suffix reads cleanly.
    for infix in (STANDARD_METRICS_INFIX, FACT_SCORE_INFIX):
        if name.endswith(infix):
            name = name[: -len(infix)]
            break
    return input_path.parent / f"{name}{suffix}{extension}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the frozen rebuttal subsubsample's rows from an existing full-run "
        "standard_metrics_aspect_wise.jsonl or fact_score_aspect_wise.jsonl file, without "
        "recomputing any metrics."
    )
    parser.add_argument("--input", required=True, help="Path to the full-run *_standard_metrics_aspect_wise.jsonl "
                        "or *_fact_score_aspect_wise.jsonl file.")
    parser.add_argument("--dataset", required=True, choices=["facetsum", "aclsum", "pmc"], help="Dataset name.")
    parser.add_argument("--output", default=None, help="Output per-example subset JSONL path.")
    parser.add_argument("--aggregate-output", default=None, help="Output aggregated JSON path.")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="Metric keys to aggregate. Defaults to ROUGE/BERTScore or fact_claim_* "
                        "depending on the auto-detected --input file kind.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Do not fail if some rebuttal subsubsample examples are missing from --input.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    input_path = Path(args.input)
    file_kind = detect_file_kind(input_path)
    if file_kind == "fact_score":
        default_metrics, output_suffix, aggregate_suffix = FACT_SCORE_METRICS, FACT_OUTPUT_SUFFIX, FACT_AGGREGATE_SUFFIX
    else:
        default_metrics, output_suffix, aggregate_suffix = STANDARD_METRICS, OUTPUT_SUFFIX, AGGREGATE_SUFFIX
    metrics = args.metrics if args.metrics is not None else default_metrics
    logging.info("Detected file kind=%s; using metrics=%s", file_kind, metrics)

    output_path = Path(args.output) if args.output else derive_output_path(input_path, output_suffix, ".jsonl")
    aggregate_output_path = (
        Path(args.aggregate_output) if args.aggregate_output else derive_output_path(input_path, aggregate_suffix, ".json")
    )

    logging.info("Loading rebuttal subsubsample for dataset=%s", args.dataset)
    unique_ids, hash_index = build_rebuttal_index(args.dataset)
    logging.info("Rebuttal subsubsample has %s examples", len(unique_ids))

    logging.info("Loading full-run metrics from %s", input_path)
    rows = load_jsonl(input_path)
    logging.info("Loaded %s rows from full-run file", len(rows))

    matched_rows: List[Dict[str, Any]] = []
    matched_uids: set[str] = set()
    for row in rows:
        uid = match_row(row, unique_ids, hash_index)
        if uid is None:
            continue
        if uid in matched_uids:
            continue  # keep-first on duplicate match
        matched_uids.add(uid)
        matched_rows.append(row)

    missing = sorted(set(unique_ids) - matched_uids)
    logging.info(
        "Matched %s/%s rebuttal subsubsample examples (missing=%s)",
        len(matched_rows), len(unique_ids), len(missing),
    )
    if missing and not args.allow_missing:
        raise ValueError(
            f"{len(missing)} rebuttal subsubsample examples were not found in {input_path}. "
            f"Preview of missing unique_ids: {missing[:20]}. Use --allow-missing to proceed anyway."
        )

    save_jsonl(matched_rows, output_path)
    logging.info("Saved %s per-example rows to %s", len(matched_rows), output_path)

    aggregated = {
        "overall": aggregate_metrics(matched_rows, metrics),
        "by_aspect": aggregate_metrics_by_aspect(matched_rows, metrics),
        "metadata": {
            "source_full_run_file": str(input_path),
            "file_kind": file_kind,
            "dataset": args.dataset,
            "rebuttal_subsubsample_size": len(unique_ids),
            "matched_examples": len(matched_rows),
            "missing_examples": len(missing),
            "missing_unique_ids_preview": missing[:20],
        },
    }
    save_json(aggregated, aggregate_output_path)
    logging.info("Saved aggregated metrics to %s", aggregate_output_path)


if __name__ == "__main__":
    main()
