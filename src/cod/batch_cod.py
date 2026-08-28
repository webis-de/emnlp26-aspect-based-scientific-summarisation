"""Run batch iterative dense summarization using vLLM.

For each sample:
1) Generate an initial entity-sparse aspect summary.
2) Iteratively identify missing salient entities and rewrite a denser summary
    without increasing the configured summary length limits.
"""

import os

# Keep legacy engine behavior for stability with current environment.
os.environ["VLLM_USE_V1"] = "0"

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# -----------------------------------------------------------------------------
# SETUP & IMPORTS
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_utils import load_summarization_dataset
import data_utils
from paths import DATA_ROOT, LOG_ROOT, RESULTS_ROOT
from language_engine import LanguageEngine, VLLM_CONFIG

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    "base_storage": DATA_ROOT,
    "output_storage": RESULTS_ROOT / "cod",
    "secondary_output_storage": RESULTS_ROOT / "cod",
    "log_dir": LOG_ROOT,
    # Model/runtime config
    "model_name": "olmo-3-7b",
    "model_path": "allenai/Olmo-3-7B-Instruct",
    "tensor_parallel_size": 2,
    "gpu_memory_utilization": 0.95,
    "max_model_len": None,
    "dtype": "bfloat16",
    "quantization": None,
    "kv_cache_dtype": "auto",
    "enforce_eager": True,
    # Sampling config
    "temperature": 0.1,
    "top_p": 0.95,
    "max_tokens": 4000,
    # Run config
    "max_samples": 0,  # 0 = all
    "chunk_size": 50,
    "enable_thinking": False,
    "cod_iterations": 2,
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

COD_INITIAL_SYSTEM_PROMPT = (
    "You are an expert scientific editor, you are given a scientific text and tasked to write "
    "a summary focused on a specific aspect."
    "You use the same wording and tone used in the source text, which is scientific and formal. Write concise, faithful aspect-focused summaries."
)

COD_INITIAL_PROMPT_TEMPLATE = """
# CONTEXT
You will be provided with a scientific text and a specific aspect to focus on.

# TASK
Given the scientific text below and a focused aspect, which is {aspect}, write a short summary using your own words.
The summary needs to be a coherent paragraph and should include the major points.
Write in free form, avoid bullet points or numbered lists.
The summary should focus on the provided aspect only, contain only information about the aspect, and avoid adding irrelevant sentences or your own opinions and suggestions.
This is the initial draft, so keep it relatively high-level and leave room for adding missing details in later refinement passes.

# INPUT
The source text is the full text of a scientific paper.
FOCUSED ASPECT: {aspect}
SUMMARY TARGET LENGTH: Maximum {max_sentences} sentences and {max_words} words.
SOURCE TEXT:
{source_text}

# OUTPUT
Return only the initial summary paragraph text.
"""

COD_ITERATION_SYSTEM_PROMPT = (
    "You are an expert scientific editor, you are given a scientific text and tasked to revise "
    "a summary focused on a specific aspect."
    "You use the same wording and tone used in the source text, which is scientific and formal. Write concise, faithful aspect-focused summaries."
)

COD_ITERATION_PROMPT_TEMPLATE = """
# CONTEXT
You will be provided with a scientific text, a focused aspect, and a draft summary for that aspect.

# TASK
Revise the draft summary for aspect "{aspect}".
Compare the draft summary against the source text and identify missing high-value information.
If important information is missing, add it.
Keep the summary focused on the same aspect and faithful to the source.
Condense and remove redundancy if needed to keep the summary concise.

# MISSING ENTITY DEFINITION
A missing entity should be:
- Relevant: central to the focused aspect
- Specific: concrete name, term, model, dataset, method, metric, number, or finding
- Novel: not already present in previous summary
- Faithful: explicitly supported by source text

# REVISION METHOD
- Check for missing high-value details: named entities, methods/models, datasets, key numbers, and main findings.
- Integrate only the most important missing details.
- Rewrite the paragraph for better clarity, flow, and scientific style.
- If the draft is already complete, still rewrite for clearer wording and smoother style.
- Avoid copying the draft verbatim.

# STRICT LENGTH RULES
- Your revised summary must satisfy BOTH limits:
  - at most {max_words} words
  - at most {max_sentences} sentences
- Keep approximately the same length as the previous summary ({prev_words} words, {prev_sentences} sentences).

# STYLE RULES
- Write in the same tone and style as sentences in scientific abstracts, which is concise, logical to follow, and formal.
- Single coherent paragraph.
- No bullet points and no numbered lists.
- No unsupported claims.

# INPUT
SOURCE TYPE: full text of a scientific paper
FOCUSED ASPECT: {aspect}
ITERATION: {iteration_idx} / {total_iterations}
SUMMARY TARGET LENGTH: Maximum {max_sentences} sentences and {max_words} words.
SOURCE TEXT:
{source_text}

PREVIOUS SUMMARY:
{draft_summary}

# OUTPUT
Return exactly this format:
MISSING_ENTITIES: entity1; entity2; entity3
DENSER_SUMMARY: <single paragraph>
"""

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def setup_logging(config: Dict[str, Any], model_name: str, dataset: str, context_size: str, mode: str) -> None:
    config["log_dir"].mkdir(parents=True, exist_ok=True)
    log_file = config["log_dir"] / f"cod_{mode}_{model_name}_{dataset}_{context_size}.log"

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

    logging.info("Initialized COD run with config: %s", config)


def to_clean_model_name(model_path: str) -> str:
    return model_path.lower().replace("/", "_").replace("-", "_")


def to_temp_tag(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


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


# Convert structured sections into a plain text prompt payload.
def parse_source_text(raw_source: Any) -> str:
    if isinstance(raw_source, dict):
        text_parts: List[str] = []
        for section_title, section_content in raw_source.items():
            if section_content:
                clean_title = str(section_title).upper().strip()
                clean_content = str(section_content).strip()
                text_parts.append(f"## {clean_title}\n{clean_content}")
        return "\n\n".join(text_parts)
    return str(raw_source).strip()


# Lightweight tokenizer proxy used for length constraints.
def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


# Simple sentence splitter for guard checks and reporting.
def sentence_count(text: str) -> int:
    cleaned = (text or "").strip()
    if not cleaned:
        return 0
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", cleaned) if p.strip()]
    return len(parts) if parts else 1


# Normalize raw target config and guarantee positive limits.
def length_limits_from_targets(max_words: int, max_sentences: int) -> Dict[str, int]:
    return {
        "max_words": max(1, int(max_words)),
        "max_sentences": max(1, int(max_sentences)),
    }


# Strictly enforce both sentence and word limits.
def is_within_length_guard(candidate: str, max_words: int, max_sentences: int) -> bool:
    return (
        word_count(candidate) <= max_words
        and sentence_count(candidate) <= max_sentences
    )


def normalize_summary_for_compare(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def is_effectively_identical(candidate: str, reference: str) -> bool:
    return normalize_summary_for_compare(candidate) == normalize_summary_for_compare(reference)


def parse_cod_iteration_output(raw_text: str) -> Tuple[List[str], str]:
    text = (raw_text or "").strip()
    if not text:
        return [], ""

    entities_match = re.search(r"MISSING_ENTITIES\s*:\s*(.*?)(?:\n|$)", text, flags=re.IGNORECASE | re.DOTALL)
    summary_match = re.search(r"DENSER_SUMMARY\s*:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)

    entities: List[str] = []
    if entities_match:
        raw_entities = entities_match.group(1).strip()
        split_candidates = re.split(r"\s*;\s*|\s*,\s*", raw_entities)
        entities = [e.strip(" -\t\n\r") for e in split_candidates if e.strip()]

    if summary_match:
        summary_text = summary_match.group(1).strip()
    else:
        summary_text = text

    return entities[:3], summary_text


def build_full_prompt_from_messages(tokenizer: Any, messages: List[Dict[str, str]], enable_thinking: bool) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is None:
        lines = []
        for message in messages:
            role = str(message.get("role", "user")).strip().upper()
            content = str(message.get("content", "")).strip()
            lines.append(f"{role}: {content}")
        lines.append("ASSISTANT:")
        return "\n\n".join(lines)

    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    try:
        return apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        try:
            return apply_chat_template(messages, **kwargs)
        except TypeError:
            return apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def save_chunk_to_jsonl(rows: List[Dict[str, Any]], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_completed_samples(save_path: Path) -> int:
    if not save_path.exists():
        return 0
    with open(save_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def build_initial_prompt(
    record: Dict[str, Any],
    parsed_text: str,
    max_words: int,
    max_sentences: int,
) -> str:
    return COD_INITIAL_PROMPT_TEMPLATE.format(
        aspect=record.get("aspect_name", "main"),
        source_text=parsed_text,
        max_words=max_words,
        max_sentences=max_sentences,
    )


def build_cod_iteration_prompt(
    record: Dict[str, Any],
    parsed_text: str,
    draft_summary: str,
    max_words: int,
    max_sentences: int,
    iteration_idx: int,
    total_iterations: int,
) -> str:
    prev_words = word_count(draft_summary)
    prev_sentences = sentence_count(draft_summary)
    return COD_ITERATION_PROMPT_TEMPLATE.format(
        aspect=record.get("aspect_name", "main"),
        source_text=parsed_text,
        draft_summary=draft_summary,
        max_words=max_words,
        max_sentences=max_sentences,
        prev_words=prev_words,
        prev_sentences=prev_sentences,
        iteration_idx=iteration_idx,
        total_iterations=total_iterations,
    )


def build_output_paths(
    model_name_clean: str,
    dataset_name: str,
    context_size: str,
    record_type: str,
    temperature_tag: str,
) -> Dict[str, Path]:
    output_dir = (
        CONFIG["output_storage"]
        / model_name_clean
        / dataset_name
        / f"{context_size}_temp_{temperature_tag}"
    )
    secondary_output_dir = (
        CONFIG["secondary_output_storage"]
        / model_name_clean
        / dataset_name
        / f"{context_size}_temp_{temperature_tag}"
    )
    base_name = f"vllm_{model_name_clean}_{dataset_name}_{context_size}_{record_type}_cod"
    return {
        "output_dir": output_dir,
        "secondary_output_dir": secondary_output_dir,
        "final_results": output_dir / f"{base_name}_results.jsonl",
        "length_failures": output_dir / f"{base_name}_length_guard_failed.jsonl",
        "secondary_final_results": secondary_output_dir / f"{base_name}_results.jsonl",
        "secondary_length_failures": secondary_output_dir / f"{base_name}_length_guard_failed.jsonl",
    }


# Mirror `zero_shot_vllm.py` engine setup style for consistency.
def build_engine() -> LanguageEngine:
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
    return LanguageEngine(engine_config)


def prepare_initial_batch(
    batch_records: List[Dict[str, Any]],
    max_words: int,
    max_sentences: int,
) -> Tuple[List[str], List[str]]:
    parsed_texts: List[str] = []
    initial_prompts: List[str] = []
    for record in batch_records:
        parsed_text = parse_source_text(record.get("source_text"))
        parsed_texts.append(parsed_text)
        initial_prompts.append(
            build_initial_prompt(
                record=record,
                parsed_text=parsed_text,
                max_words=max_words,
                max_sentences=max_sentences,
            )
        )
    return parsed_texts, initial_prompts


def prepare_cod_iteration_batch(
    batch_records: List[Dict[str, Any]],
    parsed_texts: List[str],
    previous_summaries: List[str],
    max_words: int,
    max_sentences: int,
    iteration_idx: int,
    total_iterations: int,
) -> List[str]:
    iteration_prompts: List[str] = []
    for record, parsed_text, previous_summary in zip(batch_records, parsed_texts, previous_summaries):
        iteration_prompts.append(
            build_cod_iteration_prompt(
                record=record,
                parsed_text=parsed_text,
                draft_summary=(previous_summary or "").strip(),
                max_words=max_words,
                max_sentences=max_sentences,
                iteration_idx=iteration_idx,
                total_iterations=total_iterations,
            )
        )
    return iteration_prompts


def build_rows_for_batch(
    batch_records: List[Dict[str, Any]],
    parsed_texts: List[str],
    initial_user_prompts: List[str],
    initial_full_prompts: List[str],
    initial_outputs: List[str],
    rollout_stats_per_record: List[Dict[str, int]],
    cod_steps_per_record: List[List[Dict[str, Any]]],
    context_size: str,
    engine_model_path: str,
    generation_params: Dict[str, Any],
    summary_target_length_words: int,
    summary_target_length_sentences: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    limits = length_limits_from_targets(
        summary_target_length_words,
        summary_target_length_sentences,
    )
    rows_to_save: List[Dict[str, Any]] = []
    length_failures_to_save: List[Dict[str, Any]] = []

    for local_idx, record in enumerate(batch_records):
        step1_summary = (initial_outputs[local_idx] or "").strip()
        cod_steps = cod_steps_per_record[local_idx] if local_idx < len(cod_steps_per_record) else []
        final_summary = step1_summary
        if cod_steps:
            final_summary = (cod_steps[-1].get("summary") or "").strip() or step1_summary

        final_within_guard = is_within_length_guard(
            final_summary,
            summary_target_length_words,
            summary_target_length_sentences,
        )

        result_row = {
            "dataset": record.get("dataset"),
            "unique_id": record.get("unique_id"),
            "source_type": record.get("source_type"),
            "context_size": context_size,
            "aspect_name": record.get("aspect_name", "main"),
            "source_text": parsed_texts[local_idx],
            "gold_aspect_summary": record.get("aspect_summary"),
            "generated_aspect_summary_step1": step1_summary,
            "generated_aspect_summary_step2": final_summary,
            "generated_aspect_summary": final_summary,
            "metadata": {
                "model": engine_model_path,
                "system_prompt": COD_INITIAL_SYSTEM_PROMPT,
                "user_prompt": initial_user_prompts[local_idx],
                "full_prompt": initial_full_prompts[local_idx],
                "generation_params": generation_params,
                "num_rollouts": rollout_stats_per_record[local_idx].get("num_rollouts", 0),
                "input_tokens": rollout_stats_per_record[local_idx].get("input_tokens", 0),
                "output_tokens": rollout_stats_per_record[local_idx].get("output_tokens", 0),
                "cod": {
                    "iterations": len(cod_steps),
                    "steps": cod_steps,
                },
                "length_guard": {
                    "allowed_max_words": limits["max_words"],
                    "allowed_max_sentences": limits["max_sentences"],
                    "step1_words": word_count(step1_summary),
                    "step1_sentences": sentence_count(step1_summary),
                    "step2_words": word_count(final_summary),
                    "step2_sentences": sentence_count(final_summary),
                    "final_words": word_count(final_summary),
                    "final_sentences": sentence_count(final_summary),
                    "final_within_guard": final_within_guard,
                },
                "step2_quality": {
                    "step2_identical_to_step1": is_effectively_identical(final_summary, step1_summary),
                },
            },
        }
        rows_to_save.append(result_row)

        if not final_within_guard:
            failure_row = dict(result_row)
            failure_row["failure_reason"] = "strict_length_not_satisfied"
            length_failures_to_save.append(failure_row)

    return rows_to_save, length_failures_to_save


# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run iterative dense vLLM summarization")
    parser.add_argument("--dataset", type=str, default="aclsum", help="Dataset name (pmc, facetsum, aclsum)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--context-size", dest="context_size", type=str, default="short", help="Context size: short or long")
    parser.add_argument("--max-samples", type=int, default=CONFIG["max_samples"], help="Max samples to run; 0 means all")
    parser.add_argument("--chunk-size", type=int, default=CONFIG["chunk_size"], help="Generation chunk size")
    parser.add_argument("--cod-iterations", type=int, default=CONFIG["cod_iterations"], help="Number of densification iterations (max 2)")
    args = parser.parse_args()

    dataset_name = args.dataset
    split = args.split
    context_size = args.context_size
    max_samples = args.max_samples
    chunk_size = args.chunk_size
    cod_iterations = min(2, max(0, int(args.cod_iterations)))

    summary_targets = get_summary_targets(dataset_name)
    record_type = get_record_type(dataset_name)
    summary_target_length_sentences = summary_targets["sentences"]
    summary_target_length_words = summary_targets["words"]

    engine = build_engine()
    engine_model_path = engine.config.get("model_path", CONFIG["model_path"])
    model_name_clean = to_clean_model_name(CONFIG.get("model_name", engine_model_path))
    temperature_tag = to_temp_tag(engine.config.get("temperature", CONFIG["temperature"]))

    setup_logging(CONFIG, model_name_clean, dataset_name, context_size, "main")

    if CONFIG["base_storage"]:
        data_utils.DEFAULT_DATA_ROOT = CONFIG["base_storage"]

    paths = build_output_paths(
        model_name_clean=model_name_clean,
        dataset_name=dataset_name,
        context_size=context_size,
        record_type=record_type,
        temperature_tag=temperature_tag,
    )
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    paths["secondary_output_dir"].mkdir(parents=True, exist_ok=True)

    logging.info("Loading dataset...")
    dataset = load_summarization_dataset(
        split=split,
        dataset_name=dataset_name,
        type=record_type,
        prompt_format="none",
        context_size_type=context_size,
    )
    if max_samples > 0:
        dataset = dataset[:max_samples]

    completed_count = count_completed_samples(paths["final_results"])
    if completed_count > 0:
        logging.info("Found existing final results. Resuming from sample %s.", completed_count + 1)

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

    generation_params = {
        "temperature": getattr(engine.sampling_params, "temperature", None),
        "top_p": getattr(engine.sampling_params, "top_p", None),
        "max_tokens": getattr(engine.sampling_params, "max_tokens", None),
    }

    logging.info("Starting iterative dense generation in chunks of %s (iterations=%s)", chunk_size, cod_iterations)

    for i in range(0, len(remaining_dataset), chunk_size):
        batch_records = remaining_dataset[i : i + chunk_size]

        # 1) Prepare initial sparse summaries.
        parsed_texts, initial_prompts = prepare_initial_batch(
            batch_records,
            max_words=summary_target_length_words,
            max_sentences=summary_target_length_sentences,
        )

        # Build full prompts in the same style as zero-shot for token accounting.
        tokenizer = engine.tokenizer
        initial_full_prompts: List[str] = []
        for prompt in initial_prompts:
            messages = [
                {"role": "system", "content": COD_INITIAL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            initial_full_prompts.append(
                build_full_prompt_from_messages(
                    tokenizer=tokenizer,
                    messages=messages,
                    enable_thinking=CONFIG["enable_thinking"],
                )
            )

        t0 = time.perf_counter()
        # 2) Generate initial entity-sparse summaries.
        initial_outputs_raw = engine.model.generate(initial_full_prompts, engine.sampling_params)
        initial_outputs = [output.outputs[0].text.strip() for output in initial_outputs_raw]

        rollout_stats_per_record: List[Dict[str, int]] = []
        for output in initial_outputs_raw:
            rollout_stats_per_record.append(
                {
                    "num_rollouts": len(output.outputs),
                    "input_tokens": len(output.prompt_token_ids),
                    "output_tokens": len(output.outputs[0].token_ids),
                }
            )

        current_summaries: List[str] = [(text or "").strip() for text in initial_outputs]
        cod_steps_per_record: List[List[Dict[str, Any]]] = [[] for _ in batch_records]

        # 3) Iteratively densify summaries via CoD steps.
        for iteration_idx in range(1, cod_iterations + 1):
            iteration_prompts = prepare_cod_iteration_batch(
                batch_records=batch_records,
                parsed_texts=parsed_texts,
                previous_summaries=current_summaries,
                max_words=summary_target_length_words,
                max_sentences=summary_target_length_sentences,
                iteration_idx=iteration_idx,
                total_iterations=cod_iterations,
            )

            iteration_full_prompts: List[str] = []
            for prompt in iteration_prompts:
                messages = [
                    {"role": "system", "content": COD_ITERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
                iteration_full_prompts.append(
                    build_full_prompt_from_messages(
                        tokenizer=tokenizer,
                        messages=messages,
                        enable_thinking=CONFIG["enable_thinking"],
                    )
                )

            iteration_outputs_raw = engine.model.generate(iteration_full_prompts, engine.sampling_params)
            iteration_outputs = [output.outputs[0].text.strip() for output in iteration_outputs_raw]

            next_summaries: List[str] = []
            for local_idx, (output_text, raw_output) in enumerate(zip(iteration_outputs, iteration_outputs_raw)):
                entities, denser_summary = parse_cod_iteration_output(output_text)
                denser_summary = (denser_summary or "").strip()
                fallback_summary = current_summaries[local_idx]

                within_guard = is_within_length_guard(
                    denser_summary,
                    summary_target_length_words,
                    summary_target_length_sentences,
                )
                if not denser_summary or not within_guard:
                    denser_summary = fallback_summary

                cod_steps_per_record[local_idx].append(
                    {
                        "iteration": iteration_idx,
                        "system_prompt": COD_ITERATION_SYSTEM_PROMPT,
                        "user_prompt": iteration_prompts[local_idx],
                        "full_prompt": iteration_full_prompts[local_idx],
                        "missing_entities": entities,
                        "summary": denser_summary,
                        "summary_words": word_count(denser_summary),
                        "summary_sentences": sentence_count(denser_summary),
                        "num_rollouts": len(raw_output.outputs),
                        "input_tokens": len(raw_output.prompt_token_ids),
                        "output_tokens": len(raw_output.outputs[0].token_ids),
                    }
                )
                rollout_stats_per_record[local_idx]["num_rollouts"] += len(raw_output.outputs)
                rollout_stats_per_record[local_idx]["input_tokens"] += len(raw_output.prompt_token_ids)
                rollout_stats_per_record[local_idx]["output_tokens"] += len(raw_output.outputs[0].token_ids)
                next_summaries.append(denser_summary)

            current_summaries = next_summaries

        # 4) Assemble output rows + a separate file for strict-length misses.
        rows_to_save, length_failures_to_save = build_rows_for_batch(
            batch_records=batch_records,
            parsed_texts=parsed_texts,
            initial_user_prompts=initial_prompts,
            initial_full_prompts=initial_full_prompts,
            initial_outputs=initial_outputs,
            rollout_stats_per_record=rollout_stats_per_record,
            cod_steps_per_record=cod_steps_per_record,
            context_size=context_size,
            engine_model_path=engine_model_path,
            generation_params=generation_params,
            summary_target_length_words=summary_target_length_words,
            summary_target_length_sentences=summary_target_length_sentences,
        )

        # 5) Persist chunk outputs for resumability.
        save_chunk_to_jsonl(rows_to_save, paths["final_results"])
        save_chunk_to_jsonl(rows_to_save, paths["secondary_final_results"])
        if length_failures_to_save:
            save_chunk_to_jsonl(length_failures_to_save, paths["length_failures"])
            save_chunk_to_jsonl(length_failures_to_save, paths["secondary_length_failures"])

        duration = time.perf_counter() - t0
        logging.info(
            "Saved chunk %s-%s/%s (%s samples in %.2fs, length-failures=%s)",
            completed_count + i + 1,
            completed_count + min(i + chunk_size, len(remaining_dataset)),
            len(dataset),
            len(batch_records),
            duration,
            len(length_failures_to_save),
        )

    logging.info("Done. Final results saved to %s", paths["final_results"])
    logging.info("Done. Final results also saved to %s", paths["secondary_final_results"])
    logging.info("Length-guard failures saved to %s", paths["length_failures"])
    logging.info("Length-guard failures also saved to %s", paths["secondary_length_failures"])


if __name__ == "__main__":
    main()
