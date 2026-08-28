"""Batched RAG (retrieval-augmented) aspect summarization runner."""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Keep legacy engine behavior for stability with current environment.
os.environ["VLLM_USE_V1"] = "0"

# -----------------------------------------------------------------------------
# SYSTEM SETUP
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_utils import load_summarization_dataset
from paths import DATA_ROOT, LOG_ROOT
import io_agents as i_o
from language_engine import LanguageEngine, VLLM_CONFIG
from prompts import WriterPrompts
from retriever import Retriever

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    "base_storage": DATA_ROOT,
    "log_dir": LOG_ROOT / "rag",
    "dataset": "pmc",
    "context_size": "long",
    "split": "test",
    "max_samples": 0,
    "batch_size": 100,
    "pipeline_mode": "rag",
    "language_engine": {
        "model_path": "Qwen/Qwen3.5-9B",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "max_tokens": 1000,
        "temperature": 0.1,
        "top_p": 0.95,
    },
    "retriever": {
        "model_name": "NovaSearch/stella_en_1.5B_v5",
        "device": "cpu",
        "chunk_size": 256,    # words per document chunk
        "top_k_chunks": 15,   # how many chunks to retrieve per aspect
        "word_budget": 512,   # max words kept after retrieval (global cap)
    },
}

SUMMARY_TARGETS = {
    "aclsum": {"sentences": 1, "words": 25},
    "facetsum": {"sentences": 2, "words": 50},
    "pmc": {"sentences": 4, "words": 75},
}

TYPE_BY_DATASET = {
    "aclsum": "full",
    "facetsum": "sampled",
    "pmc": "sampled",
}


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def setup_logging(config: Dict[str, Any], model_name: str, dataset: str, context_size: str, mode: str) -> None:
    config["log_dir"].mkdir(parents=True, exist_ok=True)
    log_file = config["log_dir"] / f"rag_{mode}_{model_name}_{dataset}_{context_size}.log"

    file_handler = logging.FileHandler(str(log_file), mode="w")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info("Initialized RAG run with config: %s", config)
    logging.info("Logging to: %s", log_file)


def clean_model_name(model_path: str) -> str:
    if not model_path:
        return "unknown_model"
    parts = [p for p in model_path.strip().split("/") if p]
    joined = "_".join(parts).lower().replace("-", "_")
    cleaned = re.sub(r"[^a-z0-9._-]+", "_", joined)
    return cleaned.strip("_") or "unknown_model"


def get_summary_targets(dataset_name: str) -> Dict[str, int]:
    key = str(dataset_name).strip().lower()
    if key not in SUMMARY_TARGETS:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Expected one of: {', '.join(SUMMARY_TARGETS)}")
    return SUMMARY_TARGETS[key]


def get_record_type(dataset_name: str) -> str:
    key = str(dataset_name).strip().lower()
    if key not in TYPE_BY_DATASET:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Expected one of: {', '.join(TYPE_BY_DATASET)}")
    return TYPE_BY_DATASET[key]


def summary_length_for_dataset(dataset_name: str) -> str:
    targets = get_summary_targets(dataset_name)
    return f"Maximum {targets['sentences']} sentences and {targets['words']} words."


def normalize_text(text: str) -> str:
    return "".join(ch if (ch == "\n" or ord(ch) >= 32) else " " for ch in text)


def parse_source_text(raw_source: Any) -> str:
    if isinstance(raw_source, dict):
        text_parts: List[str] = []
        for section_title, section_content in raw_source.items():
            if section_content:
                clean_title = normalize_text(str(section_title).upper().strip())
                clean_content = normalize_text(str(section_content).strip())
                text_parts.append(f"## {clean_title}\n{clean_content}")
        return "\n\n".join(text_parts).strip()
    return normalize_text(str(raw_source or "")).strip()


def output_root_dir(dataset_name: str, context_size: str) -> Path:
    model_name_clean = clean_model_name(CONFIG["language_engine"].get("model_path"))
    return (
        CONFIG["base_storage"]
        / "results"
        / CONFIG["pipeline_mode"]
        / model_name_clean
        / dataset_name
        / context_size
    )


def build_output_paths(dataset_name: str, context_size: str) -> Dict[str, Path]:
    run_id = f"{dataset_name}_{context_size}_{CONFIG['split']}"
    root = output_root_dir(dataset_name, context_size)
    return {
        "root": root,
        "final_results": root / f"{run_id}_rag_final_results.jsonl",
        "traces_results": root / f"{run_id}_rag_traces_results.jsonl",
        "parser_errors": root / f"{run_id}_rag_parser_errors.jsonl",
    }


def save_chunk_to_jsonl(rows: List[Dict[str, Any]], save_path: Path) -> None:
    if not rows:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_completed_samples(save_path: Path) -> int:
    if not save_path.exists():
        return 0
    with open(save_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl_rows(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_unique_id(row: Dict[str, Any]) -> str | None:
    value = row.get("unique_id")
    if value is None and isinstance(row.get("metadata"), dict):
        value = row["metadata"].get("unique_id")
    if value is None:
        return None
    return str(value)


def merge_rows_by_unique_id(
    existing_rows: List[Dict[str, Any]],
    updated_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for row in existing_rows:
        uid = row_unique_id(row)
        if uid is None:
            continue
        if uid not in merged:
            order.append(uid)
        merged[uid] = row
    for row in updated_rows:
        uid = row_unique_id(row)
        if uid is None:
            continue
        if uid not in merged:
            order.append(uid)
        merged[uid] = row
    return [merged[uid] for uid in order]


def extract_unique_id(record: Dict[str, Any], fallback_idx: int) -> str:
    uid = record.get("unique_id")
    if uid is None:
        return f"{CONFIG['dataset']}_{CONFIG['context_size']}_{CONFIG['split']}_{fallback_idx}"
    return str(uid)


def build_record_lookup_by_unique_id(dataset: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for idx, record in enumerate(dataset):
        lookup[extract_unique_id(record, idx)] = record
    return lookup


def build_retry_records(
    parser_error_rows: List[Dict[str, Any]],
    dataset_lookup: Dict[str, Dict[str, Any]],
    dataset_name: str,
) -> List[Dict[str, Any]]:
    retry_by_id: Dict[str, Dict[str, Any]] = {}
    for error_row in parser_error_rows:
        uid = row_unique_id(error_row)
        if uid is None:
            continue
        original = dataset_lookup.get(uid, {})
        retry_by_id[uid] = {
            "unique_id": uid,
            "dataset": error_row.get("dataset") or original.get("dataset") or dataset_name,
            "aspect_name": error_row.get("aspect_name") or original.get("aspect_name"),
            "aspect_summary": original.get("aspect_summary"),
            "source_text": error_row.get("source_text")
            if error_row.get("source_text") is not None
            else original.get("source_text"),
        }
    return list(retry_by_id.values())


def build_engine() -> LanguageEngine:
    engine_config = VLLM_CONFIG.copy()
    for key, value in CONFIG["language_engine"].items():
        engine_config[key] = value
    return LanguageEngine(engine_config)


def build_retriever() -> Retriever:
    cfg = CONFIG["retriever"]
    return Retriever(
        model_name=cfg["model_name"],
        device=cfg["device"],
        chunk_size=cfg["chunk_size"],
        top_k_chunks=cfg["top_k_chunks"],
        word_budget=cfg["word_budget"],
    )


# -----------------------------------------------------------------------------
# CORE BATCH PROCESSING
# -----------------------------------------------------------------------------
def run_batch(
    *,
    batch_records: List[Dict[str, Any]],
    fallback_start_idx: int,
    context_size: str,
    summary_length: str,
    retriever: Retriever,
    lang_engine: LanguageEngine,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    logging.info("Running batch with %s records (fallback_start_idx=%s)", len(batch_records), fallback_start_idx)

    batch_source_texts = [parse_source_text(record.get("source_text")) for record in batch_records]

    # --- RETRIEVAL STAGE ---
    # Encode all (document, aspect) pairs in the batch with 3 model.encode()
    # calls total rather than O(N * K) sequential calls.
    aspect_names = [str(record.get("aspect_name") or "") for record in batch_records]
    retriever_inputs: List[Dict[str, Any]] = [
        i_o.RetrieverInput(
            document_text=source_text,
            aspect_name=aspect_name,
            chunk_size=CONFIG["retriever"]["chunk_size"],
            top_k_chunks=CONFIG["retriever"]["top_k_chunks"],
            word_budget=CONFIG["retriever"]["word_budget"],
        ).model_dump()
        for source_text, aspect_name in zip(batch_source_texts, aspect_names)
    ]

    try:
        prune_results = retriever.prune_batch(list(zip(batch_source_texts, aspect_names)))
    except Exception as exc:
        logging.warning("prune_batch failed for entire batch: %s", exc)
        prune_results = [(None, None)] * len(batch_records)

    retriever_outputs: List[Dict[str, Any] | None] = []
    d_pruned_texts: List[str | None] = []
    for d_pruned, stats in prune_results:
        if d_pruned is None or stats is None:
            retriever_outputs.append(None)
            d_pruned_texts.append(None)
        else:
            retriever_outputs.append({
                "d_pruned": d_pruned,
                "num_chunks": stats["num_chunks"],
                "num_chunks_retrieved": stats.get("num_chunks_retrieved"),
                "retrieved_chunk_indices": stats.get("retrieved_chunk_indices"),
                "num_sentences_retained": stats["num_sentences_retained"],
                "num_words_retained": stats["num_words_retained"],
                "chunk_scores": stats.get("chunk_scores"),
            })
            d_pruned_texts.append(d_pruned)

    retriever_success = sum(1 for o in retriever_outputs if o is not None)
    logging.info("Retriever completed %s/%s samples", retriever_success, len(batch_records))

    # --- WRITER STAGE ---
    # Pass D_pruned sentences as evidence_set to the writer.
    writer_inputs_by_index: Dict[int, Dict[str, Any]] = {}
    writer_prompts_by_index: Dict[int, str] = {}
    writer_prompts: List[str] = []
    writer_target_indices: List[int] = []

    for local_idx, (record, d_pruned) in enumerate(zip(batch_records, d_pruned_texts)):
        if d_pruned is None:
            continue
        writer_input = i_o.WriterInput(
            aspect_name=str(record.get("aspect_name") or ""),
            summary_length=summary_length,
            d_pruned=d_pruned,
        )
        writer_inputs_by_index[local_idx] = writer_input.model_dump()
        _, writer_prompt = WriterPrompts.render(writer_input)
        writer_prompts_by_index[local_idx] = writer_prompt
        writer_prompts.append(writer_prompt)
        writer_target_indices.append(local_idx)

    writer_outputs_by_index: Dict[int, i_o.WriterOutput | None] = {}
    writer_token_metadata_by_index: Dict[int, Dict[str, int]] = {}
    if writer_prompts:
        writer_outputs, writer_token_metadata = lang_engine.generate_structured_in_batch_with_token_metadata(
            user_prompts=writer_prompts,
            system_prompt=WriterPrompts.SYSTEM,
            pydantic_model=i_o.WriterOutput,
            enable_thinking=False,
        )
        for local_idx, writer_output, token_info in zip(writer_target_indices, writer_outputs, writer_token_metadata):
            writer_outputs_by_index[local_idx] = writer_output
            writer_token_metadata_by_index[local_idx] = token_info

    writer_success = sum(1 for o in writer_outputs_by_index.values() if o is not None)
    logging.info("Writer parsed %s/%s prompted samples", writer_success, len(writer_prompts))

    # --- COLLECT RESULTS ---
    traces_rows: List[Dict[str, Any]] = []
    final_rows: List[Dict[str, Any]] = []
    parser_error_rows: List[Dict[str, Any]] = []

    for local_idx, (record, source_text, retriever_output) in enumerate(
        zip(batch_records, batch_source_texts, retriever_outputs)
    ):
        fallback_idx = fallback_start_idx + local_idx
        unique_id = extract_unique_id(record, fallback_idx)
        writer_output = writer_outputs_by_index.get(local_idx)
        wrt_tokens = writer_token_metadata_by_index.get(local_idx)

        if retriever_output is None:
            termination_reason = "retriever_error"
        elif writer_output is None:
            termination_reason = "writer_parse_error"
        else:
            termination_reason = "completed"

        writer_output_payload = writer_output.model_dump() if writer_output else None
        retriever_status = "ok" if retriever_output is not None else "error"
        if retriever_status == "error":
            writer_status = "skipped_due_to_retriever_error"
        else:
            writer_status = "ok" if writer_output_payload is not None else "parse_error"

        trace_row = {
            "unique_id": unique_id,
            "dataset": record.get("dataset"),
            "context_size": context_size,
            "pipeline_mode": CONFIG["pipeline_mode"],
            "aspect_name": record.get("aspect_name"),
            "source_text": source_text,
            "stages": {
                "retriever": {
                    "status": retriever_status,
                    "input": retriever_inputs[local_idx],
                    "output": retriever_output,
                },
                "writer": {
                    "status": writer_status,
                    "max_tokens": CONFIG["language_engine"]["max_tokens"],
                    "system_prompt": WriterPrompts.SYSTEM,
                    "input": writer_inputs_by_index.get(local_idx),
                    "user_prompt": writer_prompts_by_index.get(local_idx),
                    "output": writer_output_payload,
                    "num_rollouts": wrt_tokens["num_rollouts"] if wrt_tokens else None,
                    "input_tokens": wrt_tokens["input_tokens"] if wrt_tokens else None,
                    "output_tokens": wrt_tokens["output_tokens"] if wrt_tokens else None,
                },
            },
            "termination_reason": termination_reason,
        }
        traces_rows.append(trace_row)

        if termination_reason == "completed":
            final_rows.append({
                "unique_id": unique_id,
                "generated_aspect_summary": writer_output.summary_text,
                "gold_aspect_summary": record.get("aspect_summary"),
                "aspect_name": record.get("aspect_name"),
                "source_text": source_text,
                "d_pruned": retriever_output["d_pruned"],
                "context_size": context_size,
                "dataset": record.get("dataset"),
                "metadata": {
                    "unique_id": unique_id,
                    "pipeline_mode": CONFIG["pipeline_mode"],
                    "termination_reason": termination_reason,
                    "num_chunks": retriever_output["num_chunks"],
                    "num_chunks_retrieved": retriever_output.get("num_chunks_retrieved"),
                    "num_sentences_retained": retriever_output["num_sentences_retained"],
                    "num_words_retained": retriever_output["num_words_retained"],
                    "num_rollouts": wrt_tokens["num_rollouts"] if wrt_tokens else None,
                    "input_tokens": wrt_tokens["input_tokens"] if wrt_tokens else None,
                    "output_tokens": wrt_tokens["output_tokens"] if wrt_tokens else None,
                },
            })
        else:
            parser_error_rows.append({
                "unique_id": unique_id,
                "dataset": record.get("dataset"),
                "context_size": context_size,
                "aspect_name": record.get("aspect_name"),
                "source_text": source_text,
                "termination_reason": termination_reason,
            })

    logging.info(
        "Batch result summary: completed=%s parser_errors=%s",
        len(final_rows),
        len(parser_error_rows),
    )
    return traces_rows, final_rows, parser_error_rows


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Batched RAG aspect summarization runner.")
    parser.add_argument("--dataset", type=str, default=CONFIG["dataset"])
    parser.add_argument("--context-size", dest="context_size", type=str, default=CONFIG["context_size"])
    parser.add_argument(
        "--retry-parser-errors",
        dest="retry_parser_errors",
        action="store_true",
        help="Retry only samples listed in parser_errors and merge back by unique_id.",
    )
    args = parser.parse_args()

    CONFIG["dataset"] = args.dataset
    CONFIG["context_size"] = args.context_size

    dataset_name = CONFIG["dataset"]
    context_size = CONFIG["context_size"]
    split = CONFIG["split"]
    summary_length = summary_length_for_dataset(dataset_name)
    record_type = get_record_type(dataset_name)
    batch_size = max(1, int(CONFIG["batch_size"]))
    retry_parser_errors = bool(args.retry_parser_errors)

    output_paths = build_output_paths(dataset_name, context_size)
    mode = "retry" if retry_parser_errors else "main"
    model_name_clean = clean_model_name(CONFIG["language_engine"].get("model_path"))
    setup_logging(CONFIG, model_name_clean, dataset_name, context_size, mode)

    logging.info("Run mode: %s", mode)
    logging.info("Output paths: %s", output_paths)
    logging.info("Loading dataset...")
    dataset = load_summarization_dataset(
        split=split,
        dataset_name=dataset_name,
        type=record_type,
        context_size_type=context_size,
        prompt_format="none",
    )
    logging.info("Loaded dataset rows: %s", len(dataset))
    if CONFIG["max_samples"] > 0:
        dataset = dataset[: CONFIG["max_samples"]]
        logging.info("Applied max_samples=%s. Effective rows=%s", CONFIG["max_samples"], len(dataset))

    logging.info("Initializing retriever (model=%s, device=%s)...",
                 CONFIG["retriever"]["model_name"], CONFIG["retriever"]["device"])
    retriever = build_retriever()
    logging.info("Retriever ready.")

    logging.info("Initializing language engine (model=%s)...", CONFIG["language_engine"]["model_path"])
    lang_engine = build_engine()
    logging.info("Language engine ready.")

    if retry_parser_errors:
        parser_error_rows = load_jsonl_rows(output_paths["parser_errors"])
        if not parser_error_rows:
            logging.info("No parser error file (or empty). Nothing to retry.")
            return

        dataset_lookup = build_record_lookup_by_unique_id(dataset)
        retry_records = build_retry_records(parser_error_rows, dataset_lookup, dataset_name)
        if not retry_records:
            logging.warning("No valid parser-error records found for retry.")
            return

        logging.info("Retry mode: %s parser-error samples", len(retry_records))

        all_traces_updates: List[Dict[str, Any]] = []
        all_final_updates: List[Dict[str, Any]] = []
        all_parser_error_updates: List[Dict[str, Any]] = []

        for i in range(0, len(retry_records), batch_size):
            batch_records = retry_records[i : i + batch_size]
            t0 = time.perf_counter()
            traces_rows, final_rows, error_rows = run_batch(
                batch_records=batch_records,
                fallback_start_idx=0,
                context_size=context_size,
                summary_length=summary_length,
                retriever=retriever,
                lang_engine=lang_engine,
            )
            all_traces_updates.extend(traces_rows)
            all_final_updates.extend(final_rows)
            all_parser_error_updates.extend(error_rows)
            duration = time.perf_counter() - t0
            logging.info(
                "Retried batch %s-%s/%s | completed=%s errors=%s | %.2fs",
                i + 1, i + len(batch_records), len(retry_records),
                len(final_rows), len(error_rows), duration,
            )

        retry_ids = {row_unique_id(row) for row in retry_records}
        retry_ids.discard(None)

        existing_traces = load_jsonl_rows(output_paths["traces_results"])
        write_jsonl_rows(merge_rows_by_unique_id(existing_traces, all_traces_updates), output_paths["traces_results"])

        existing_final = load_jsonl_rows(output_paths["final_results"])
        write_jsonl_rows(merge_rows_by_unique_id(existing_final, all_final_updates), output_paths["final_results"])

        existing_errors = load_jsonl_rows(output_paths["parser_errors"])
        remaining_errors = [row for row in existing_errors if row_unique_id(row) not in retry_ids]
        write_jsonl_rows(merge_rows_by_unique_id(remaining_errors, all_parser_error_updates), output_paths["parser_errors"])

        logging.info("Retry run complete | retried=%s recovered=%s still_failing=%s",
                     len(retry_records), len(all_final_updates), len(all_parser_error_updates))
        return

    processed_count = count_completed_samples(output_paths["traces_results"])
    remaining_dataset = dataset[processed_count:]
    if not remaining_dataset:
        logging.info("No remaining samples to process. Exiting.")
        return

    logging.info(
        "Dataset loaded | total=%s completed=%s remaining=%s",
        len(dataset), processed_count, len(remaining_dataset),
    )

    total_processed = 0
    total_completed = 0
    total_errors = 0

    for i in range(0, len(remaining_dataset), batch_size):
        batch_records = remaining_dataset[i : i + batch_size]
        t0 = time.perf_counter()
        traces_rows, final_rows, parser_error_rows = run_batch(
            batch_records=batch_records,
            fallback_start_idx=processed_count + i,
            context_size=context_size,
            summary_length=summary_length,
            retriever=retriever,
            lang_engine=lang_engine,
        )

        save_chunk_to_jsonl(traces_rows, output_paths["traces_results"])
        save_chunk_to_jsonl(final_rows, output_paths["final_results"])
        save_chunk_to_jsonl(parser_error_rows, output_paths["parser_errors"])

        total_processed += len(batch_records)
        total_completed += len(final_rows)
        total_errors += len(parser_error_rows)
        duration = time.perf_counter() - t0
        logging.info(
            "Processed batch %s-%s/%s | completed=%s errors=%s | %.2fs",
            i + 1, i + len(batch_records), len(remaining_dataset),
            len(final_rows), len(parser_error_rows), duration,
        )

    logging.info("Run complete | processed=%s completed=%s errors=%s", total_processed, total_completed, total_errors)
    logging.info("Final results: %s", output_paths["final_results"])
    logging.info("Traces: %s", output_paths["traces_results"])
    logging.info("Parser errors: %s", output_paths["parser_errors"])


if __name__ == "__main__":
    main()
