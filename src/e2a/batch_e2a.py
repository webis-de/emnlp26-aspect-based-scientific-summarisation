"""Batched extractor->writer 2A2S runner (simplified)."""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Annotated, Any, Dict, List, Tuple

from pydantic import BaseModel, Field

# Keep legacy engine behavior for stability with current environment.
os.environ["VLLM_USE_V1"] = "0"

# -----------------------------------------------------------------------------
# SYSTEM SETUP
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_utils import load_summarization_dataset, load_rebuttal_subsubsample
from paths import DATA_ROOT, LOG_ROOT
import io_agents as i_o
from language_engine import LanguageEngine, VLLM_CONFIG
from prompts import ExtractorPrompts, WriterPrompts

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    "base_storage": DATA_ROOT,
    "log_dir": LOG_ROOT / "e2a",
    "dataset": "pmc",  # CLI override: --dataset
    "context_size": "long",  # CLI override: --context-size
    "split": "test",
    "max_samples": 0,  # 0 = process all
    "chunk_size": 50,
    "pipeline_mode": "e2a",
    "language_engine": {
        "model_path": "allenai/Olmo-3-7B-Instruct",
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.9,
        "max_tokens": 3000,
        "temperature": 0.1,
        "top_p": 0.95,
    },
    # Stage-specific generation budgets (extractor can be larger than writer).
    "stage_max_tokens": {
        "extractor": 4000,
        "writer": 1000,
    },
    "extractor_constraints": {
        "max_evidence_items": 50,
        "max_span_chars": 1000,
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

EXTRACTOR_MAX_EVIDENCE_ITEMS = CONFIG["extractor_constraints"]["max_evidence_items"]
EXTRACTOR_MAX_SPAN_CHARS = CONFIG["extractor_constraints"]["max_span_chars"]


def build_extractor_output_model(max_evidence_items: int, max_span_chars: int) -> type:
    """Build the E2AExtractorOutput Pydantic model with the given caps.

    The evidence-cap sensitivity run raises max_evidence_items (e.g. 50 -> 75)
    while keeping max_span_chars fixed at 1000, per the W3 rebuttal plan. This
    factory lets the model schema be rebuilt after CLI overrides are applied,
    since Pydantic model field constraints must be fixed at class-definition time.
    """
    evidence_span = Annotated[str, Field(min_length=1, max_length=max_span_chars)]

    class _E2AExtractorOutput(BaseModel):
        evidence_set: List[evidence_span] = Field(..., min_length=1, max_length=max_evidence_items)

    return _E2AExtractorOutput


# Default model; rebuilt in main() if CLI overrides change the caps.
E2AExtractorOutput = build_extractor_output_model(EXTRACTOR_MAX_EVIDENCE_ITEMS, EXTRACTOR_MAX_SPAN_CHARS)

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def setup_logging(config: Dict[str, Any], model_name: str, dataset: str, context_size: str, mode: str) -> None:
    config["log_dir"].mkdir(parents=True, exist_ok=True)
    log_file = config["log_dir"] / f"e2a_{mode}_{model_name}_{dataset}_{context_size}.log"

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

    logging.info("Initialized E2A run with config: %s", config)
    logging.info("Logging to: %s", log_file)


def clean_model_name(model_path: str) -> str:
    if not model_path:
        return "unknown_model"
    parts = [part for part in model_path.strip().split("/") if part]
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


def output_root_dir(dataset_name: str, context_size: str, config_name: str = "") -> Path:
    model_name_clean = clean_model_name(CONFIG["language_engine"].get("model_path"))
    if config_name:
        return (
            CONFIG["base_storage"]
            / "results"
            / CONFIG["pipeline_mode"]
            / "rebuttal_sensitivity"
            / config_name
            / model_name_clean
            / dataset_name
            / context_size
        )
    return (
        CONFIG["base_storage"]
        / "results"
        / CONFIG["pipeline_mode"]
        / model_name_clean
        / dataset_name
        / context_size
    )


def flatten_rebuttal_subsample(dataset_name: str) -> List[Dict[str, Any]]:
    grouped = load_rebuttal_subsubsample(dataset_name)
    flat: List[Dict[str, Any]] = []
    for doc_id in grouped:
        flat.extend(grouped[doc_id])
    return flat


def build_output_paths(dataset_name: str, context_size: str, config_name: str = "") -> Dict[str, Path]:
    run_id = f"{dataset_name}_{context_size}_{CONFIG['split']}"
    root = output_root_dir(dataset_name, context_size, config_name)
    return {
        "root": root,
        "final_results": root / f"{run_id}_{CONFIG['pipeline_mode']}_final_results.jsonl",
        "traces_results": root / f"{run_id}_{CONFIG['pipeline_mode']}_traces_results.jsonl",
        "parser_errors": root / f"{run_id}_{CONFIG['pipeline_mode']}_parser_errors.jsonl",
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


def build_engine() -> LanguageEngine:
    engine_config = VLLM_CONFIG.copy()
    for key, value in CONFIG["language_engine"].items():
        engine_config[key] = value
    return LanguageEngine(engine_config)


def set_structured_stage_max_tokens(engine: LanguageEngine, max_tokens: int) -> None:
    engine.config["max_tokens"] = max(1, int(max_tokens))


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
            # Prefer already-normalized prompt payload captured in parser_errors.
            "source_text": error_row.get("source_text")
            if error_row.get("source_text") is not None
            else original.get("source_text"),
        }
    return list(retry_by_id.values())


def run_chunk(
    *,
    batch_records: List[Dict[str, Any]],
    fallback_start_idx: int,
    context_size: str,
    summary_length: str,
    lang_engine: LanguageEngine,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    logging.info("Running chunk with %s records (fallback_start_idx=%s)", len(batch_records), fallback_start_idx)
    batch_source_texts = [parse_source_text(record.get("source_text")) for record in batch_records]

    extractor_inputs: List[Dict[str, Any]] = []
    extractor_prompts: List[str] = []
    for record, source_text in zip(batch_records, batch_source_texts):
        extractor_input = i_o.ExtractorInput(
            document_text=source_text,
            aspect_name=str(record.get("aspect_name") or ""),
            max_evidence_items=EXTRACTOR_MAX_EVIDENCE_ITEMS,
            max_span_chars=EXTRACTOR_MAX_SPAN_CHARS,
        )
        extractor_inputs.append(extractor_input.model_dump())
        _, user_prompt = ExtractorPrompts.render(extractor_input)
        extractor_prompts.append(user_prompt)

    extractor_max_tokens = CONFIG["stage_max_tokens"]["extractor"]
    set_structured_stage_max_tokens(lang_engine, extractor_max_tokens)
    extractor_outputs, extractor_token_metadata = lang_engine.generate_structured_in_batch_with_token_metadata(
        user_prompts=extractor_prompts,
        system_prompt=ExtractorPrompts.SYSTEM,
        pydantic_model=E2AExtractorOutput,
        enable_thinking=False,
    )
    extractor_success = sum(1 for output in extractor_outputs if output is not None)
    logging.info("Extractor parsed %s/%s samples", extractor_success, len(batch_records))

    writer_inputs_by_index: Dict[int, Dict[str, Any]] = {}
    writer_prompts_by_index: Dict[int, str] = {}
    writer_prompts: List[str] = []
    writer_target_indices: List[int] = []
    for local_idx, (record, extractor_output) in enumerate(zip(batch_records, extractor_outputs)):
        if extractor_output is None:
            continue
        writer_input = i_o.WriterInput(
            aspect_name=str(record.get("aspect_name") or ""),
            summary_length=summary_length,
            evidence_set=extractor_output.evidence_set,
        )
        writer_inputs_by_index[local_idx] = writer_input.model_dump()
        _, writer_prompt = WriterPrompts.render(writer_input)
        writer_prompts_by_index[local_idx] = writer_prompt
        writer_prompts.append(writer_prompt)
        writer_target_indices.append(local_idx)

    writer_outputs_by_index: Dict[int, i_o.WriterOutput | None] = {}
    writer_token_metadata_by_index: Dict[int, Dict[str, int]] = {}
    writer_max_tokens = CONFIG["stage_max_tokens"]["writer"]
    if writer_prompts:
        set_structured_stage_max_tokens(lang_engine, writer_max_tokens)
        writer_outputs, writer_token_metadata = lang_engine.generate_structured_in_batch_with_token_metadata(
            user_prompts=writer_prompts,
            system_prompt=WriterPrompts.SYSTEM,
            pydantic_model=i_o.WriterOutput,
            enable_thinking=False,
        )
        for local_idx, writer_output, token_info in zip(writer_target_indices, writer_outputs, writer_token_metadata):
            writer_outputs_by_index[local_idx] = writer_output
            writer_token_metadata_by_index[local_idx] = token_info
    writer_success = sum(1 for output in writer_outputs_by_index.values() if output is not None)
    logging.info("Writer parsed %s/%s prompted samples", writer_success, len(writer_prompts))

    traces_rows: List[Dict[str, Any]] = []
    final_rows: List[Dict[str, Any]] = []
    parser_error_rows: List[Dict[str, Any]] = []

    for local_idx, (record, source_text, extractor_output) in enumerate(
        zip(batch_records, batch_source_texts, extractor_outputs)
    ):
        fallback_idx = fallback_start_idx + local_idx
        unique_id = extract_unique_id(record, fallback_idx)
        writer_output = writer_outputs_by_index.get(local_idx)
        ext_tokens = extractor_token_metadata[local_idx]
        wrt_tokens = writer_token_metadata_by_index.get(local_idx)

        if extractor_output is None:
            termination_reason = "extractor_parse_error"
        elif writer_output is None:
            termination_reason = "writer_parse_error"
        else:
            termination_reason = "completed"

        extractor_output_payload = extractor_output.model_dump() if extractor_output else None
        writer_output_payload = writer_output.model_dump() if writer_output else None

        extractor_status = "ok" if extractor_output_payload is not None else "parse_error"
        if extractor_status == "parse_error":
            writer_status = "skipped_due_to_extractor_error"
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
                "extractor": {
                    "status": extractor_status,
                    "max_tokens": extractor_max_tokens,
                    "system_prompt": ExtractorPrompts.SYSTEM,
                    "input": extractor_inputs[local_idx],
                    "user_prompt": extractor_prompts[local_idx],
                    "output": extractor_output_payload,
                    "num_rollouts": ext_tokens["num_rollouts"],
                    "input_tokens": ext_tokens["input_tokens"],
                    "output_tokens": ext_tokens["output_tokens"],
                },
                "writer": {
                    "status": writer_status,
                    "max_tokens": writer_max_tokens,
                    "system_prompt": WriterPrompts.SYSTEM,
                    "input": writer_inputs_by_index.get(local_idx),
                    "user_prompt": writer_prompts_by_index.get(local_idx),
                    "output": writer_output_payload,
                    "num_rollouts": wrt_tokens["num_rollouts"] if wrt_tokens else None,
                    "input_tokens": wrt_tokens["input_tokens"] if wrt_tokens else None,
                    "output_tokens": wrt_tokens["output_tokens"] if wrt_tokens else None,
                },
            },
            "iteration_count": 1 if termination_reason == "completed" else 0,
            "termination_reason": termination_reason,
        }
        traces_rows.append(trace_row)

        if termination_reason == "completed":
            final_rows.append(
                {
                    "unique_id": unique_id,
                    "generated_aspect_summary": writer_output.summary_text,
                    "gold_aspect_summary": record.get("aspect_summary"),
                    "aspect_name": record.get("aspect_name"),
                    # Save source_text exactly as sent to the model.
                    "source_text": source_text,
                    "context_size": context_size,
                    "dataset": record.get("dataset"),
                    "metadata": {
                        "unique_id": unique_id,
                        "pipeline_mode": CONFIG["pipeline_mode"],
                        "termination_reason": termination_reason,
                        "iteration_count": 1,
                        "num_rollouts": ext_tokens["num_rollouts"] + (wrt_tokens["num_rollouts"] if wrt_tokens else 0),
                        "input_tokens": ext_tokens["input_tokens"] + (wrt_tokens["input_tokens"] if wrt_tokens else 0),
                        "output_tokens": ext_tokens["output_tokens"] + (wrt_tokens["output_tokens"] if wrt_tokens else 0),
                    },
                }
            )
        else:
            parser_error_rows.append(
                {
                    "unique_id": unique_id,
                    "dataset": record.get("dataset"),
                    "context_size": context_size,
                    "aspect_name": record.get("aspect_name"),
                    "source_text": source_text,
                    "termination_reason": termination_reason,
                }
            )

    logging.info(
        "Chunk result summary: completed=%s parser_errors=%s",
        len(final_rows),
        len(parser_error_rows),
    )
    return traces_rows, final_rows, parser_error_rows


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main() -> None:
    global E2AExtractorOutput, EXTRACTOR_MAX_EVIDENCE_ITEMS, EXTRACTOR_MAX_SPAN_CHARS

    parser = argparse.ArgumentParser(description="Batched extractor->writer 2A2S runner.")
    parser.add_argument("--dataset", type=str, default=CONFIG["dataset"])
    parser.add_argument("--context-size", dest="context_size", type=str, default=CONFIG["context_size"])
    parser.add_argument(
        "--retry-parser-errors",
        dest="retry_parser_errors",
        action="store_true",
        help="Retry only samples listed in parser_errors and merge back by unique_id.",
    )
    parser.add_argument("--max-evidence-items", dest="max_evidence_items", type=int, default=None,
                         help="Override extractor max_evidence_items cap (default: CONFIG value, 50).")
    parser.add_argument("--max-span-chars", dest="max_span_chars", type=int, default=None,
                         help="Override extractor max_span_chars cap (default: CONFIG value, 1000).")
    parser.add_argument("--extractor-max-tokens", dest="extractor_max_tokens", type=int, default=None,
                         help="Override extractor generation budget in tokens (default: CONFIG value, 4000).")
    parser.add_argument("--config-name", dest="config_name", type=str, default="",
                         help="Name for a non-default sensitivity config; routes output under "
                              "results/e2a/rebuttal_sensitivity/<config_name>/ instead of the main path.")
    parser.add_argument("--rebuttal-only", dest="rebuttal_only", action="store_true",
                         help="Restrict the run to the frozen rebuttal subsubsample (load_rebuttal_subsubsample).")
    parser.add_argument("--model-path", dest="model_path", type=str, default=None,
                         help="Override the extractor/writer model path (default: CONFIG value).")
    parser.add_argument("--max-samples", dest="max_samples", type=int, default=CONFIG["max_samples"],
                         help="If > 0, truncate the loaded dataset to this many records (for smoke tests).")
    args = parser.parse_args()

    CONFIG["dataset"] = args.dataset
    CONFIG["context_size"] = args.context_size
    if args.max_evidence_items is not None:
        CONFIG["extractor_constraints"]["max_evidence_items"] = args.max_evidence_items
        EXTRACTOR_MAX_EVIDENCE_ITEMS = args.max_evidence_items
    if args.max_span_chars is not None:
        CONFIG["extractor_constraints"]["max_span_chars"] = args.max_span_chars
        EXTRACTOR_MAX_SPAN_CHARS = args.max_span_chars
    if args.extractor_max_tokens is not None:
        CONFIG["stage_max_tokens"]["extractor"] = args.extractor_max_tokens
    if args.model_path is not None:
        CONFIG["language_engine"]["model_path"] = args.model_path
    E2AExtractorOutput = build_extractor_output_model(EXTRACTOR_MAX_EVIDENCE_ITEMS, EXTRACTOR_MAX_SPAN_CHARS)
    CONFIG["config_name"] = args.config_name
    CONFIG["rebuttal_only"] = args.rebuttal_only
    CONFIG["max_samples"] = args.max_samples

    dataset_name = CONFIG["dataset"]
    context_size = CONFIG["context_size"]
    split = CONFIG["split"]
    config_name = CONFIG["config_name"]
    rebuttal_only = CONFIG["rebuttal_only"]
    summary_length = summary_length_for_dataset(dataset_name)
    record_type = get_record_type(dataset_name)
    chunk_size = max(1, int(CONFIG["chunk_size"]))
    retry_parser_errors = bool(args.retry_parser_errors)

    if rebuttal_only and not config_name:
        raise ValueError("--config-name is required when --rebuttal-only is set, to avoid overwriting full-run outputs.")
    if rebuttal_only and context_size != "long":
        raise ValueError("The rebuttal subsubsample is only defined for context_size='long'.")

    output_paths = build_output_paths(dataset_name, context_size, config_name)
    mode = "retry" if retry_parser_errors else "main"
    model_name_clean = clean_model_name(CONFIG["language_engine"].get("model_path"))
    setup_logging(CONFIG, model_name_clean, dataset_name, context_size, mode)

    logging.info("Run mode: %s", mode)
    logging.info("Output paths: %s", output_paths)
    logging.info("Loading dataset...")
    if rebuttal_only:
        dataset = flatten_rebuttal_subsample(dataset_name)
        logging.info("Loaded %s records from the frozen rebuttal subsubsample", len(dataset))
    else:
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

    logging.info("Initializing language engine...")
    lang_engine = build_engine()
    logging.info("Language engine ready. model=%s", lang_engine.config.get("model_path"))

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

        for i in range(0, len(retry_records), chunk_size):
            batch_records = retry_records[i : i + chunk_size]
            t0 = time.perf_counter()
            traces_rows, final_rows, error_rows = run_chunk(
                batch_records=batch_records,
                fallback_start_idx=0,
                context_size=context_size,
                summary_length=summary_length,
                lang_engine=lang_engine,
            )
            all_traces_updates.extend(traces_rows)
            all_final_updates.extend(final_rows)
            all_parser_error_updates.extend(error_rows)
            duration = time.perf_counter() - t0
            logging.info(
                "Retried chunk %s-%s/%s | completed=%s errors=%s | %.2fs",
                i + 1,
                i + len(batch_records),
                len(retry_records),
                len(final_rows),
                len(error_rows),
                duration,
            )

        retry_ids = {row_unique_id(row) for row in retry_records}
        retry_ids.discard(None)

        existing_traces = load_jsonl_rows(output_paths["traces_results"])
        merged_traces = merge_rows_by_unique_id(existing_traces, all_traces_updates)
        write_jsonl_rows(merged_traces, output_paths["traces_results"])

        existing_final = load_jsonl_rows(output_paths["final_results"])
        merged_final = merge_rows_by_unique_id(existing_final, all_final_updates)
        write_jsonl_rows(merged_final, output_paths["final_results"])

        existing_errors = load_jsonl_rows(output_paths["parser_errors"])
        remaining_errors = [row for row in existing_errors if row_unique_id(row) not in retry_ids]
        merged_errors = merge_rows_by_unique_id(remaining_errors, all_parser_error_updates)
        write_jsonl_rows(merged_errors, output_paths["parser_errors"])

        logging.info("Retry run complete")
        logging.info("Retried: %s", len(retry_records))
        logging.info("Recovered: %s", len(all_final_updates))
        logging.info("Still failing: %s", len(all_parser_error_updates))
        logging.info("Final results: %s", output_paths["final_results"])
        logging.info("Traces: %s", output_paths["traces_results"])
        logging.info("Parser errors: %s", output_paths["parser_errors"])
        return

    processed_count = count_completed_samples(output_paths["traces_results"])
    remaining_dataset = dataset[processed_count:]
    if not remaining_dataset:
        logging.info("No remaining samples to process. Exiting.")
        return

    logging.info(
        "Dataset loaded | total=%s completed=%s remaining=%s",
        len(dataset),
        processed_count,
        len(remaining_dataset),
    )

    total_processed = 0
    total_completed = 0
    total_errors = 0

    logging.info("Starting simplified batched extractor->writer processing")

    for i in range(0, len(remaining_dataset), chunk_size):
        batch_records = remaining_dataset[i : i + chunk_size]
        t0 = time.perf_counter()
        traces_rows, final_rows, parser_error_rows = run_chunk(
            batch_records=batch_records,
            fallback_start_idx=processed_count + i,
            context_size=context_size,
            summary_length=summary_length,
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
            "Processed chunk %s-%s/%s | completed=%s errors=%s | %.2fs",
            i + 1,
            i + len(batch_records),
            len(remaining_dataset),
            len(final_rows),
            len(parser_error_rows),
            duration,
        )

    logging.info("Run complete")
    logging.info("Processed: %s", total_processed)
    logging.info("Completed: %s", total_completed)
    logging.info("Parser errors: %s", total_errors)
    logging.info("Final results: %s", output_paths["final_results"])
    logging.info("Traces: %s", output_paths["traces_results"])
    logging.info("Parser errors: %s", output_paths["parser_errors"])


if __name__ == "__main__":
    main()
