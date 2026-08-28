import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
EVAL_DIR = SRC_DIR / "eval"
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(SRC_DIR))
sys.path.append(str(EVAL_DIR))

import g_eval_metrics as geval
from data_utils import ASPECT_NORMALIZATION, find_longest_digit_sequence, load_rebuttal_subsubsample


GEVAL_METRICS = [
    "geval_aspect_consistency",
    "geval_aspect_relevance",
]
GENERATED_SUMMARY_FIELD_CANDIDATES = [
    "generated_aspect_summary",
    "final_generated_aspect_summary",
    "refined_aspect_summary",
    "generated_aspect_summary_step3",
    "generated_aspect_summary_step2",
    "generated_aspect_summary_step1",
]


def flatten_source_text(source_text: Any) -> str:
    if isinstance(source_text, str):
        return source_text.strip()
    if isinstance(source_text, dict):
        sections = []
        for title, text in source_text.items():
            if text is None:
                continue
            title_text = str(title).strip()
            body_text = str(text).strip()
            if not body_text:
                continue
            if title_text:
                sections.append(f"## {title_text}\n{body_text}")
            else:
                sections.append(body_text)
        return "\n\n".join(sections).strip()
    if source_text is None:
        return ""
    return json.dumps(source_text, ensure_ascii=False)


def normalize_for_lookup(text: Any) -> str:
    return " ".join(flatten_source_text(text).split()).lower()


def normalize_aspect(dataset_name: str, aspect_name: Any) -> str:
    aspect = str(aspect_name or "").strip()
    normalization = ASPECT_NORMALIZATION.get(dataset_name, {})
    return normalization.get(aspect, aspect)


def get_generated_summary(record: Dict[str, Any], generated_field: str | None) -> Tuple[str, str]:
    candidates = [generated_field] if generated_field else GENERATED_SUMMARY_FIELD_CANDIDATES
    for field in candidates:
        if not field:
            continue
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip(), field
    return "", ""


def build_sample_indexes(
    sample_by_doc_id: Dict[str, List[dict]],
    dataset_name: str,
) -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[Tuple[str, str], Dict[str, Any]],
    Dict[Tuple[str, str], Dict[str, Any]],
    List[Dict[str, Any]],
]:
    exact_id_index: Dict[str, Dict[str, Any]] = {}
    doc_aspect_index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    source_aspect_index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    sample_records: List[Dict[str, Any]] = []

    for doc_id, records in sample_by_doc_id.items():
        for record in records:
            sample_record = dict(record)
            sample_record["sample_doc_id"] = doc_id
            sample_record["aspect_name"] = normalize_aspect(dataset_name, sample_record.get("aspect_name"))
            sample_records.append(sample_record)

            unique_id = sample_record.get("unique_id")
            if unique_id:
                exact_id_index[str(unique_id)] = sample_record

            aspect_key = sample_record["aspect_name"]
            doc_aspect_index[(str(doc_id), aspect_key)] = sample_record

            source_key = normalize_for_lookup(sample_record.get("source_text"))
            if source_key:
                source_aspect_index[(source_key, aspect_key)] = sample_record

    return exact_id_index, doc_aspect_index, source_aspect_index, sample_records


def find_sample_match(
    result_record: Dict[str, Any],
    dataset_name: str,
    exact_id_index: Dict[str, Dict[str, Any]],
    doc_aspect_index: Dict[Tuple[str, str], Dict[str, Any]],
    source_aspect_index: Dict[Tuple[str, str], Dict[str, Any]],
) -> Tuple[Dict[str, Any] | None, str]:
    result_unique_id = result_record.get("unique_id")
    if result_unique_id and str(result_unique_id) in exact_id_index:
        return exact_id_index[str(result_unique_id)], "unique_id"

    result_aspect = normalize_aspect(dataset_name, result_record.get("aspect_name"))
    result_doc_id = find_longest_digit_sequence(str(result_unique_id or ""))
    if result_doc_id:
        sample_record = doc_aspect_index.get((result_doc_id, result_aspect))
        if sample_record is not None:
            return sample_record, "doc_id+aspect_name"

    source_key = normalize_for_lookup(result_record.get("source_text"))
    if source_key:
        sample_record = source_aspect_index.get((source_key, result_aspect))
        if sample_record is not None:
            return sample_record, "source_text+aspect_name"

    return None, ""


def prepare_geval_input(
    result_records: Sequence[Dict[str, Any]],
    dataset_name: str,
    generated_field: str | None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sample_by_doc_id = load_rebuttal_subsubsample(dataset_name)
    exact_id_index, doc_aspect_index, source_aspect_index, sample_records = build_sample_indexes(
        sample_by_doc_id,
        dataset_name,
    )

    prepared: List[Dict[str, Any]] = []
    matched_sample_ids = set()
    skipped_without_match = 0
    skipped_without_generation = 0
    match_counts: Dict[str, int] = {}

    for result_record in result_records:
        sample_record, match_strategy = find_sample_match(
            result_record=result_record,
            dataset_name=dataset_name,
            exact_id_index=exact_id_index,
            doc_aspect_index=doc_aspect_index,
            source_aspect_index=source_aspect_index,
        )
        if sample_record is None:
            skipped_without_match += 1
            continue

        generated_summary, used_generated_field = get_generated_summary(result_record, generated_field)
        if not generated_summary:
            skipped_without_generation += 1
            continue

        reference_text = flatten_source_text(sample_record.get("source_text"))
        sample_unique_id = str(sample_record.get("unique_id") or result_record.get("unique_id") or "")
        matched_sample_ids.add(sample_unique_id)
        match_counts[match_strategy] = match_counts.get(match_strategy, 0) + 1

        metadata = dict(result_record.get("metadata") or {})
        metadata.update(
            {
                "geval_reference": "sample_full_source_text",
                "match_strategy": match_strategy,
                "generated_summary_field": used_generated_field,
                "result_unique_id": result_record.get("unique_id"),
                "sample_unique_id": sample_record.get("unique_id"),
                "sample_doc_id": sample_record.get("sample_doc_id"),
            }
        )

        prepared.append(
            {
                "unique_id": sample_unique_id,
                "dataset": dataset_name,
                "context_size": result_record.get("context_size", "long"),
                "source_type": sample_record.get("source_type") or result_record.get("source_type"),
                "aspect_name": sample_record.get("aspect_name") or normalize_aspect(dataset_name, result_record.get("aspect_name")),
                "source_text": reference_text,
                "gold_aspect_summary": reference_text,
                "result_gold_aspect_summary": result_record.get("gold_aspect_summary"),
                "generated_aspect_summary": generated_summary,
                "metadata": metadata,
            }
        )

    sample_ids = {str(record.get("unique_id") or "") for record in sample_records if record.get("unique_id")}
    unmatched_sample_ids = sorted(sample_ids - matched_sample_ids)
    stats = {
        "sample_documents": len(sample_by_doc_id),
        "sample_examples": len(sample_records),
        "input_result_examples": len(result_records),
        "prepared_examples": len(prepared),
        "skipped_without_sample_match": skipped_without_match,
        "skipped_without_generated_summary": skipped_without_generation,
        "unmatched_sample_examples": len(unmatched_sample_ids),
        "unmatched_sample_unique_ids_preview": unmatched_sample_ids[:20],
        "match_counts": match_counts,
    }
    return prepared, stats


def derive_output_path(input_path: str, suffix: str, extension: str) -> str:
    if input_path.endswith(".jsonl"):
        return input_path.replace(".jsonl", suffix + extension)
    return input_path + suffix + extension


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run G-Eval on the deterministic rebuttal subsubsample using full source text as reference."
    )
    parser.add_argument("--input", required=True, help="Experiment result JSONL path.")
    parser.add_argument("--dataset", required=True, choices=["facetsum", "pmc", "aclsum"], help="Dataset name.")
    parser.add_argument("--output", default=None, help="Per-example output JSONL path.")
    parser.add_argument("--aggregate-output", default=None, help="Aggregate output JSON path.")
    parser.add_argument("--log", default=None, help="Optional log path.")
    parser.add_argument("--mode", choices=["parallel", "batch"], default="parallel", help="G-Eval execution mode.")
    parser.add_argument("--generated-field", default=None, help="Override the generated summary field name.")
    parser.add_argument("--max-workers", type=int, default=geval.MAX_WORKERS, help="Parallel worker count.")
    parser.add_argument("--requests-per-minute", type=int, default=geval.PARALLEL_REQUESTS_PER_MINUTE)
    parser.add_argument("--save-interval", type=int, default=geval.SAVE_INTERVAL)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing checkpoint at the output path and rescore everything from scratch.",
    )
    parser.add_argument("--allow-missing", action="store_true", help="Do not fail when some sample records are missing.")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional smoke-test limit after matching.")
    parser.add_argument("--prepare-only", action="store_true", help="Only write matched G-Eval input records; do not call Azure.")
    parser.add_argument("--pretty", action="store_true", default=geval.PRETTY_PRINT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_path = args.output or derive_output_path(input_path, "_rebuttal_subsample_geval_metrics_aspect_wise", ".jsonl")
    aggregate_output_path = args.aggregate_output or derive_output_path(
        input_path,
        "_rebuttal_subsample_geval_metrics_aggregated_aspect_wise",
        ".json",
    )
    log_path = args.log or str(
        PROJECT_ROOT / "logs" / "eval" / (Path(input_path).stem + "_rebuttal_subsample_geval.log")
    )

    geval.setup_logging(log_path)
    logging.info("Loading experiment results from %s", input_path)
    result_records = geval.load_jsonl(input_path)
    if not result_records:
        raise ValueError(f"No result records found in {input_path}")

    prepared, stats = prepare_geval_input(
        result_records=result_records,
        dataset_name=args.dataset,
        generated_field=args.generated_field,
    )
    if args.max_examples is not None:
        prepared = prepared[: args.max_examples]
        stats["prepared_examples_after_max_examples"] = len(prepared)

    logging.info("Prepared %s G-Eval examples. Stats: %s", len(prepared), stats)
    if not prepared:
        raise ValueError("No records matched the rebuttal subsubsample and generated-summary field.")
    if stats["unmatched_sample_examples"] and not args.allow_missing and args.max_examples is None:
        raise ValueError(
            "Some rebuttal subsubsample examples were not found in the result file. "
            "Use --allow-missing to score only matched records. "
            f"Stats: {stats}"
        )

    if args.prepare_only:
        logging.info("Prepare-only mode: saving matched G-Eval input records to %s", output_path)
        geval.save_jsonl(prepared, output_path)
        prepare_metadata = {
            "metadata": {
                "input": input_path,
                "dataset": args.dataset,
                "mode": "prepare-only",
                "reference": "full_source_text_from_load_rebuttal_subsubsample",
                "stats": stats,
            }
        }
        logging.info("Prepare-only mode: saving metadata to %s", aggregate_output_path)
        geval.save_json(prepare_metadata, aggregate_output_path, pretty=args.pretty)
        logging.info("Done.")
        return

    geval.PARALLEL_REQUESTS_PER_MINUTE = args.requests_per_minute
    mode = args.mode.strip().lower()
    if mode == "parallel":
        processed = geval.calculate_geval_metrics_parallel(
            data=prepared,
            checkpoint_output=output_path,
            save_interval=args.save_interval,
            max_workers=args.max_workers,
            resume=not args.no_resume,
        )
    else:
        batch_requests_path = derive_output_path(input_path, "_rebuttal_subsample_geval_batch_requests", ".jsonl")
        batch_raw_output_path = derive_output_path(input_path, "_rebuttal_subsample_geval_batch_raw_output", ".jsonl")
        evaluator = geval.GEvalScientificMetric(
            azure_endpoint=geval.AZURE_ENDPOINT,
            api_key=geval.API_KEY,
            api_version=geval.API_VERSION,
            deployment_name=geval.BATCH_DEPLOYMENT_NAME,
            max_retries_on_rate_limit=geval.MAX_RETRIES_ON_RATE_LIMIT,
            retry_base_delay_seconds=geval.RETRY_BASE_DELAY_SECONDS,
            retry_max_delay_seconds=geval.RETRY_MAX_DELAY_SECONDS,
        )
        processed = geval.calculate_geval_metrics_batch(
            data=prepared,
            evaluator=evaluator,
            requests_file_path=batch_requests_path,
            raw_output_file_path=batch_raw_output_path,
        )

    logging.info("Saving per-example G-Eval metrics to %s", output_path)
    geval.save_jsonl(processed, output_path)

    aggregate = {
        "overall": geval.aggregate_metrics(processed, GEVAL_METRICS),
        "by_aspect": geval.aggregate_metrics_by_aspect(processed, GEVAL_METRICS),
        "metadata": {
            "input": input_path,
            "dataset": args.dataset,
            "mode": mode,
            "reference": "full_source_text_from_load_rebuttal_subsubsample",
            "stats": stats,
        },
    }
    logging.info("Saving aggregate G-Eval metrics to %s", aggregate_output_path)
    geval.save_json(aggregate, aggregate_output_path, pretty=args.pretty)
    logging.info("Done.")


if __name__ == "__main__":
    main()