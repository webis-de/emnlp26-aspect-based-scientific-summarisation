"""RAG Step 2: read D_pruned from retrieved.jsonl, run writer LLM, save summaries.

Reads the output of batch_retrieve.py.
Does NOT load the stella model.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ["VLLM_USE_V1"] = "0"

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import DATA_ROOT, LOG_ROOT
import io_agents as i_o
from language_engine import LanguageEngine, VLLM_CONFIG
from prompts import WriterPrompts

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    "base_storage": DATA_ROOT,
    "log_dir": LOG_ROOT / "rag",
    "dataset": "aclsum",
    "context_size": "long",
    "split": "test",
    "batch_size": 100,
    "pipeline_mode": "rag",
    "language_engine": {
        "model_path": "allenai/Olmo-3-7B-Instruct",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "max_tokens": 1000,
        "temperature": 0.1,
        "top_p": 0.95,
    },
}

SUMMARY_TARGETS = {
    "aclsum": {"sentences": 1, "words": 25},
    "facetsum": {"sentences": 2, "words": 50},
    "pmc": {"sentences": 4, "words": 75},
}


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def clean_model_name(model_path: str) -> str:
    parts = [p for p in model_path.strip().split("/") if p]
    joined = "_".join(parts).lower().replace("-", "_")
    return re.sub(r"[^a-z0-9._-]+", "_", joined).strip("_") or "unknown_model"


def summary_length_for_dataset(dataset_name: str) -> str:
    t = SUMMARY_TARGETS.get(dataset_name.lower(), {"sentences": 3, "words": 60})
    return f"Maximum {t['sentences']} sentences and {t['words']} words."


def setup_logging(log_dir: Path, model_name: str, dataset: str, context_size: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"generate_{model_name}_{dataset}_{context_size}.log"
    file_handler = logging.FileHandler(str(log_file), mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def retrieved_input_path(dataset: str, context_size: str, config_name: str = "") -> Path:
    run_id = f"{dataset}_{context_size}_{CONFIG['split']}"
    if config_name:
        return (
            CONFIG["base_storage"]
            / "results" / "rag" / "rebuttal_sensitivity" / config_name / "retrieved" / dataset / context_size
            / f"{run_id}_retrieved.jsonl"
        )
    return (
        CONFIG["base_storage"]
        / "results" / "rag" / "retrieved" / dataset / context_size
        / f"{run_id}_retrieved.jsonl"
    )


def output_paths(dataset: str, context_size: str, config_name: str = "") -> Dict[str, Path]:
    model_name_clean = clean_model_name(CONFIG["language_engine"]["model_path"])
    run_id = f"{dataset}_{context_size}_{CONFIG['split']}"
    if config_name:
        root = (
            CONFIG["base_storage"] / "results" / "rag" / "rebuttal_sensitivity" / config_name
            / model_name_clean / dataset / context_size
        )
    else:
        root = CONFIG["base_storage"] / "results" / "rag" / model_name_clean / dataset / context_size
    return {
        "final_results": root / f"{run_id}_rag_final_results.jsonl",
        "traces":        root / f"{run_id}_rag_traces_results.jsonl",
        "parser_errors": root / f"{run_id}_rag_parser_errors.jsonl",
    }


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def count_completed(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def append_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_engine() -> LanguageEngine:
    engine_config = VLLM_CONFIG.copy()
    engine_config.update(CONFIG["language_engine"])
    return LanguageEngine(engine_config)


# -----------------------------------------------------------------------------
# CORE BATCH
# -----------------------------------------------------------------------------
def run_batch(
    batch: List[Dict[str, Any]],
    summary_length: str,
    lang_engine: LanguageEngine,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:

    # Build prompts for all records in batch
    writer_prompts: List[str] = []
    writer_inputs: List[Dict[str, Any]] = []
    for record in batch:
        writer_input = i_o.WriterInput(
            aspect_name=str(record.get("aspect_name") or ""),
            summary_length=summary_length,
            d_pruned=record["d_pruned"],
        )
        writer_inputs.append(writer_input.model_dump())
        _, prompt = WriterPrompts.render(writer_input)
        writer_prompts.append(prompt)

    writer_outputs, token_metadata = lang_engine.generate_structured_in_batch_with_token_metadata(
        user_prompts=writer_prompts,
        system_prompt=WriterPrompts.SYSTEM,
        pydantic_model=i_o.WriterOutput,
        enable_thinking=False,
    )

    final_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    for record, writer_input, prompt, writer_output, tokens in zip(
        batch, writer_inputs, writer_prompts, writer_outputs, token_metadata
    ):
        uid = record["unique_id"]
        retrieval_stats = record.get("retrieval_stats", {})

        if writer_output is None:
            termination_reason = "writer_parse_error"
        else:
            termination_reason = "completed"

        trace_rows.append({
            "unique_id": uid,
            "dataset": record.get("dataset"),
            "context_size": record.get("context_size"),
            "pipeline_mode": CONFIG["pipeline_mode"],
            "aspect_name": record.get("aspect_name"),
            "d_pruned": record["d_pruned"],
            "retrieval_stats": retrieval_stats,
            "stages": {
                "writer": {
                    "status": "ok" if writer_output else "parse_error",
                    "max_tokens": CONFIG["language_engine"]["max_tokens"],
                    "system_prompt": WriterPrompts.SYSTEM,
                    "input": writer_input,
                    "user_prompt": prompt,
                    "output": writer_output.model_dump() if writer_output else None,
                    "num_rollouts": tokens["num_rollouts"],
                    "input_tokens": tokens["input_tokens"],
                    "output_tokens": tokens["output_tokens"],
                },
            },
            "termination_reason": termination_reason,
        })

        if writer_output is not None:
            final_rows.append({
                "unique_id": uid,
                "generated_aspect_summary": writer_output.summary_text,
                "gold_aspect_summary": record.get("gold_aspect_summary"),
                "aspect_name": record.get("aspect_name"),
                "d_pruned": record["d_pruned"],
                "source_text": record.get("source_text"),
                "context_size": record.get("context_size"),
                "dataset": record.get("dataset"),
                "metadata": {
                    "unique_id": uid,
                    "pipeline_mode": CONFIG["pipeline_mode"],
                    "termination_reason": termination_reason,
                    "num_chunks": retrieval_stats.get("num_chunks"),
                    "num_chunks_retrieved": retrieval_stats.get("num_chunks_retrieved"),
                    "num_sentences_retained": retrieval_stats.get("num_sentences_retained"),
                    "num_words_retained": retrieval_stats.get("num_words_retained"),
                    "num_rollouts": tokens["num_rollouts"],
                    "input_tokens": tokens["input_tokens"],
                    "output_tokens": tokens["output_tokens"],
                },
            })
        else:
            error_rows.append({
                "unique_id": uid,
                "dataset": record.get("dataset"),
                "context_size": record.get("context_size"),
                "aspect_name": record.get("aspect_name"),
                "d_pruned": record["d_pruned"],
                "termination_reason": termination_reason,
            })

    return final_rows, trace_rows, error_rows


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="RAG generation step: read D_pruned, run writer LLM.")
    parser.add_argument("--dataset", type=str, default=CONFIG["dataset"])
    parser.add_argument("--context-size", dest="context_size", type=str, default=CONFIG["context_size"])
    parser.add_argument("--config-name", dest="config_name", type=str, default="",
                         help="Name of a non-default sensitivity config; must match the --config-name used "
                              "in batch_retrieve.py. Reads/writes under results/rag/rebuttal_sensitivity/<config_name>/.")
    parser.add_argument("--model-path", dest="model_path", type=str, default=None,
                         help="Override the writer model path (default: CONFIG value).")
    args = parser.parse_args()

    CONFIG["dataset"] = args.dataset
    CONFIG["context_size"] = args.context_size
    CONFIG["config_name"] = args.config_name
    if args.model_path is not None:
        CONFIG["language_engine"]["model_path"] = args.model_path
    dataset_name = CONFIG["dataset"]
    context_size = CONFIG["context_size"]
    config_name = CONFIG["config_name"]

    model_name_clean = clean_model_name(CONFIG["language_engine"]["model_path"])
    setup_logging(CONFIG["log_dir"], model_name_clean, dataset_name, context_size)
    logging.info("Generation config: %s", CONFIG)

    in_path = retrieved_input_path(dataset_name, context_size, config_name)
    if not in_path.exists():
        logging.error("Retrieved file not found: %s — run batch_retrieve.py first.", in_path)
        sys.exit(1)

    retrieved = load_jsonl(in_path)
    logging.info("Loaded %s retrieved records from %s", len(retrieved), in_path)

    paths = output_paths(dataset_name, context_size, config_name)
    processed = count_completed(paths["traces"])
    remaining = retrieved[processed:]
    if not remaining:
        logging.info("All records already generated. Exiting.")
        return
    logging.info("Completed=%s  Remaining=%s", processed, len(remaining))
    logging.info("Output paths: %s", paths)

    summary_length = summary_length_for_dataset(dataset_name)

    logging.info("Loading language engine (%s)...", CONFIG["language_engine"]["model_path"])
    lang_engine = build_engine()
    logging.info("Language engine ready.")

    batch_size = CONFIG["batch_size"]
    total_ok = 0
    total_err = 0

    for i in range(0, len(remaining), batch_size):
        batch = remaining[i : i + batch_size]
        t0 = time.perf_counter()
        final_rows, trace_rows, error_rows = run_batch(batch, summary_length, lang_engine)

        append_jsonl(final_rows, paths["final_results"])
        append_jsonl(trace_rows, paths["traces"])
        append_jsonl(error_rows, paths["parser_errors"])

        total_ok += len(final_rows)
        total_err += len(error_rows)
        duration = time.perf_counter() - t0
        logging.info(
            "Batch %s-%s/%s | ok=%s errors=%s | %.2fs",
            i + 1, i + len(batch), len(remaining), len(final_rows), len(error_rows), duration,
        )

    logging.info("Done. ok=%s  errors=%s", total_ok, total_err)
    logging.info("Final results: %s", paths["final_results"])
    logging.info("Traces: %s", paths["traces"])
    logging.info("Parser errors: %s", paths["parser_errors"])


if __name__ == "__main__":
    main()
