"""
Run batch zero-shot summarization using vLLM with Tensor Parallelism.
"""
import os

# --- CRITICAL FIX: CUDA-safe multiprocessing for vLLM workers on Linux ---
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import sys
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

import argparse

# -----------------------------------------------------------------------------
# SETUP & IMPORTS
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

ENGINE_DIR = SRC_DIR / "2a2s"
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from data_utils import load_summarization_dataset
import data_utils
from paths import DATA_ROOT, LOG_ROOT, RESULTS_ROOT
from language_engine import LanguageEngine, VLLM_CONFIG

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    "base_storage": DATA_ROOT,
    "output_storage": RESULTS_ROOT / "zs",
    "log_dir": LOG_ROOT / "zs",
    # Model/runtime config (overrides LanguageEngine defaults)
    "model_name": "qwen3p5-9b", # "olmo-3-7b" or "qwen3-8b" or "qwen3-32b" or "qwen3p5-27b" 
    "model_path": "Qwen/Qwen3.5-9B", # allenai/Olmo-3-7B-Instruct or "Qwen/Qwen3-8B" or "Qwen/Qwen3-32B" or "Qwen/Qwen3.5-27B"
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.95,
    "max_model_len": None,
    "dtype": "bfloat16",
    "quantization": None,
    "kv_cache_dtype": "auto",
    "enforce_eager": True,
    # Sampling config
    "temperature": 0.1,
    "top_p": 0.95,
    "max_tokens": 2000,
    # YaRN config
    "yarn_rope_scaling": False,
    "yarn_factor": 4.0,
    "native_ctx_length": None,
    # Checkpointing
    "max_samples": 0,  # 0 = Process all
    "chunk_size": 100,  # Save to disk every N samples
    # Thinking mode
    "enable_thinking": False,
}

# -----------------------------------------------------------------------------
# LOGGING & SAVING
# -----------------------------------------------------------------------------
def setup_logging(config: Dict[str, Any], model_name: str, dataset: str, context_size: str) -> None:
    config["log_dir"].mkdir(parents=True, exist_ok=True)
    log_file = config["log_dir"] / (
        f"vllm_{model_name}_{dataset}_{context_size}.log"
    )

    file_handler = logging.FileHandler(str(log_file), mode="w")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(file_formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info("Initialized run with config: %s", config)


def save_chunk_to_jsonl(results: List[dict], save_path: Path) -> None:
    with open(save_path, "a", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_completed_samples(save_path: Path) -> int:
    if not save_path.exists():
        return 0
    count = 0
    with open(save_path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def to_clean_model_name(model_path: str) -> str:
    return model_path.lower().replace("/", "_").replace("-", "_")


def to_temp_tag(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


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
# DATA PROCESSING
# -----------------------------------------------------------------------------
SOURCE_TYPE_DESCRIPTIONS = {
    "introduction-conclusion": "the introduction and conclusion sections of a scientific paper",
    "aspect-section": "the specific section of a scientific paper relevant to the requested aspect",
    "abstract-introduction-conclusion": "the abstract, introduction, and conclusion sections of a scientific paper",
    "full-text": "the full text of a scientific paper",
    "full-text-structured": "the full text of a scientific paper, structured by section headers",
}

SYSTEM_PROMPT = (
    "You are an expert scientific editor, you are given a scientific text and tasked to write "
    "a summary focused on a specific aspect."
    "You use the same wording and tone used in the source text, which is scientific and formal. Write concise, faithful aspect-focused summaries."
)

PROMPT_TEMPLATE = """
# CONTEXT
You will be provided with a scientific text and a specific aspect to focus on.

# TASK
Given the scientific text below and a focused aspect, which is {aspect}, write a short summary using your own words.
The summary needs to be a coherent paragraph and should include the major points.
Write in free form, avoid bullet points or numbered lists.
The summary should focus on the provided aspect only, contain only information about the aspect, and avoid adding irrelevant sentences or your own opinions and suggestions.

# INPUT
The source text is a {source_type_pretty}. This is {source_type_desc}.
FOCUSED ASPECT: {aspect}
{summary_line}
SOURCE TEXT:
{source_text}

# OUTPUT
"""

def parse_source_text(raw_source) -> str:
    if isinstance(raw_source, dict):
        text_parts = []
        for section_title, section_content in raw_source.items():
            if section_content:
                clean_title = str(section_title).upper().strip()
                clean_content = str(section_content).strip()
                part = f"## {clean_title}\n{clean_content}"
                text_parts.append(part)
        return "\n\n".join(text_parts)
    return str(raw_source).strip()


def prepare_messages(
    record: dict,
    parsed_text: str,
    summary_target_length_sentences: int,
    summary_target_length_words: int,
) -> List[Dict[str, str]]:
    st_key = record.get("source_type", "unknown")
    st_desc = SOURCE_TYPE_DESCRIPTIONS.get(st_key, "scientific text")

    if summary_target_length_sentences is None and summary_target_length_words is None:
        summary_line = ""
    else:
        sent = summary_target_length_sentences if summary_target_length_sentences is not None else ""
        words = summary_target_length_words if summary_target_length_words is not None else ""
        summary_line = f"SUMMARY TARGET LENGTH: Maximum {sent} sentences and {words} words."

    user_content = PROMPT_TEMPLATE.format(
        aspect=record.get("aspect_name", "main"),
        source_type_pretty=st_key.replace("-", " "),
        source_type_desc=st_desc,
        summary_line=summary_line,
        source_text=parsed_text,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


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


# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run zero-shot vLLM summarization")
    parser.add_argument("--dataset", type=str, default="aclsum", help="Dataset name (pmc, facetsum, aclsum)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--context-size", dest="context_size", type=str, default="short", help="Context size: short or long")
    parser.add_argument("--max-samples", type=int, default=CONFIG["max_samples"], help="Max samples to run; 0 means all")
    parser.add_argument("--chunk-size", type=int, default=CONFIG["chunk_size"], help="Generation batch/checkpoint chunk size")
    args = parser.parse_args()

    dataset_name = args.dataset
    split = args.split
    context_size = args.context_size
    max_samples = args.max_samples
    chunk_size = args.chunk_size

    summary_targets = get_summary_targets(dataset_name)
    record_type = get_record_type(dataset_name)
    summary_target_length_sentences = summary_targets["sentences"]
    summary_target_length_words = summary_targets["words"]

    # Build runtime config for LanguageEngine from this script's VLLM settings.
    engine_config = VLLM_CONFIG.copy()
    engine_config["model_path"] = CONFIG["model_path"]
    engine_config["tensor_parallel_size"] = CONFIG["tensor_parallel_size"]
    engine_config["gpu_memory_utilization"] = CONFIG["gpu_memory_utilization"]
    engine_config["max_model_len"] = CONFIG["max_model_len"]
    engine_config["dtype"] = CONFIG["dtype"]
    engine_config["quantization"] = CONFIG["quantization"]
    engine_config["kv_cache_dtype"] = CONFIG["kv_cache_dtype"]
    engine_config["enforce_eager"] = CONFIG["enforce_eager"]
    engine_config["temperature"] = CONFIG["temperature"]
    engine_config["top_p"] = CONFIG["top_p"]
    engine_config["max_tokens"] = CONFIG["max_tokens"]
    engine_config["yarn_rope_scaling"] = CONFIG["yarn_rope_scaling"]
    engine_config["yarn_factor"] = CONFIG["yarn_factor"]
    engine_config["native_ctx_length"] = CONFIG["native_ctx_length"]

    engine = LanguageEngine(engine_config)
    engine_model_path = engine.config.get("model_path", CONFIG["model_path"])
    model_name_clean = to_clean_model_name(CONFIG.get("model_name", engine_model_path))
    temperature_tag = to_temp_tag(engine.config.get("temperature", CONFIG["temperature"]))

    setup_logging(CONFIG, model_name_clean, dataset_name, context_size)

    if CONFIG["base_storage"]:
        data_utils.DEFAULT_DATA_ROOT = CONFIG["base_storage"]

    output_dir = (
        CONFIG["output_storage"]
        / model_name_clean
        / dataset_name
        / f"{context_size}_temp_{temperature_tag}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / (
        f"vllm_{model_name_clean}_{dataset_name}_{context_size}_{record_type}_results.jsonl"
    )

    # 1. LOAD DATASET
    logging.info("Loading dataset...")
    dataset = load_summarization_dataset(
        split=split,
        dataset_name=dataset_name,
        type=record_type,
        prompt_format="none",
        context_size_type=context_size,
    )
    if max_samples > 0:
        dataset = dataset[: max_samples]

    # 2. RESUME LOGIC (Count lines in JSONL)
    completed_count = count_completed_samples(save_path)
    if completed_count > 0:
        logging.info("Found existing JSONL. Resuming from sample %s.", completed_count + 1)

    remaining_dataset = dataset[completed_count:]
    if not remaining_dataset:
        logging.info("All samples processed. Exiting.")
        return

    logging.info(
        "Total: %s | Completed: %s | To Do: %s",
        len(dataset),
        completed_count,
        len(remaining_dataset),
    )

    # 3. PREPARE ALL PROMPTS FIRST (CPU Work)
    logging.info("Formatting prompts for remaining samples...")

    logging.info(
        "Initialized LanguageEngine with model=%s, tensor_parallel_size=%s",
        engine_model_path,
        engine.config.get("tensor_parallel_size"),
    )
    tokenizer = engine.tokenizer

    work_queue = []
    for record in remaining_dataset:
        source_text = parse_source_text(record.get("source_text"))
        messages = prepare_messages(
            record,
            source_text,
            summary_target_length_sentences,
            summary_target_length_words,
        )

        full_prompt_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=CONFIG.get("enable_thinking", False),
        )

        meta = {
            "record": record,
            "source_text_clean": source_text,
            "full_prompt": full_prompt_str,
        }
        work_queue.append((full_prompt_str, meta))

    # 4. CHUNKED GENERATION LOOP
    total_items = len(work_queue)
    generation_params = {
        "temperature": getattr(engine.sampling_params, "temperature", None),
        "top_p": getattr(engine.sampling_params, "top_p", None),
        "max_tokens": getattr(engine.sampling_params, "max_tokens", None),
    }

    logging.info(
        "Starting generation in chunks of %s with type=%s and summary target=%s sentences, %s words...",
        chunk_size,
        record_type,
        summary_target_length_sentences,
        summary_target_length_words,
    )

    for i in range(0, total_items, chunk_size):
        chunk = work_queue[i : i + chunk_size]
        chunk_prompts = [item[0] for item in chunk]
        chunk_meta = [item[1] for item in chunk]

        start_time = time.perf_counter()

        outputs = engine.model.generate(chunk_prompts, engine.sampling_params)

        results_to_save = []
        thinking_enabled = CONFIG.get("enable_thinking", False)
        for output, meta in zip(outputs, chunk_meta):
            raw_text = output.outputs[0].text.strip()
            generated_text = raw_text
            thinking_text = None
            if thinking_enabled:
                closing_tag = "</think>"
                lower_text = raw_text.lower()
                closing_idx = lower_text.find(closing_tag)
                if closing_idx != -1:
                    end_idx = closing_idx + len(closing_tag)
                    thinking_text = raw_text[:end_idx].strip()
                    generated_text = raw_text[end_idx:].strip()

            record = meta["record"]
            results_to_save.append(
                {
                    "dataset": record.get("dataset"),
                    "unique_id": record.get("unique_id"),
                    "source_type": record.get("source_type"),
                    "context_size": context_size,
                    "aspect_name": record.get("aspect_name", "main"),
                    "source_text": meta["source_text_clean"],
                    "gold_aspect_summary": record.get("aspect_summary"),
                    "generated_aspect_summary": generated_text,
                    "metadata": {
                        "model": engine_model_path,
                        "system_prompt": SYSTEM_PROMPT,
                        "user_prompt": meta.get("full_prompt"),
                        "generation_params": generation_params,
                        "thinking_enabled": thinking_enabled,
                        "thinking_text": thinking_text,
                        "num_rollouts": len(output.outputs),
                        "input_tokens": len(output.prompt_token_ids),
                        "output_tokens": len(output.outputs[0].token_ids),
                    },
                }
            )

        save_chunk_to_jsonl(results_to_save, save_path)

        duration = time.perf_counter() - start_time
        logging.info(
            "Saved chunk %s-%s/%s (%s samples in %.2fs)",
            i + 1,
            min(i + chunk_size, total_items),
            total_items,
            len(chunk),
            duration,
        )

    logging.info("Done. Results saved to %s", save_path)


if __name__ == "__main__":
    main()
