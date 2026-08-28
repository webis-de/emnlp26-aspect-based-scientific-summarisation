"""Batched 2A2S pipeline runner with parser-error retry support.

Pipeline order:
Planner -> Extractor -> Writer -> Verifier -> Router

Execution model:
- Planner runs batched once and is persisted.
- Extractor/Writer/Verifier run in role-batched loops using router decisions.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# vLLM must use spawn with CUDA worker processes in this environment.
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from data_utils import load_summarization_dataset
from paths import DATA_ROOT, LOG_ROOT
from language_engine import LanguageEngine, VLLM_CONFIG
from prompts import PlannerPrompts, ExtractorPrompts, WriterPrompts, VerifierPrompts
from router import RouterAgent
import io_agents as i_o

CONFIG: Dict[str, Any] = {
    "base_storage": DATA_ROOT,
    "dataset": "facetsum",
    "split": "test",
    "context_size": "long",
    "max_samples": 0,
    "max_retries": 2,
    "pipeline_mode": "full",  # full | no_planner | no_verifier
    "batch_size": 64,
    "language_engine": {
        "model_path": "Qwen/Qwen3.5-2B",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "temperature": 0.1,
        "top_p": 0.95,
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


def clean_model_name(model_path: str) -> str:
    if not model_path:
        return "unknown_model"
    return model_path.replace("/", "_").replace("-", "_").lower().strip("_")


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


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return "".join(ch if (ch in ("\n", "\t", "\r") or ord(ch) >= 32) else " " for ch in text)


def flatten_source_text(raw_source: Any) -> str:
    if isinstance(raw_source, dict):
        chunks: List[str] = []
        for section_title, section_content in raw_source.items():
            clean_title = normalize_text(section_title).strip()
            clean_content = normalize_text(section_content).strip()
            chunks.append(f"## {clean_title}\n{clean_content}")
        return "\n\n".join(chunks).strip()
    return normalize_text(raw_source).strip()


def output_root_dir(dataset_name: str, context_size: str) -> Path:
    model_name = clean_model_name(CONFIG["language_engine"].get("model_path", ""))
    return (
        CONFIG["base_storage"]
        / "results"
        / "2a2s"
        / CONFIG["pipeline_mode"]
        / model_name
        / dataset_name
        / context_size
    )


def build_output_paths(dataset_name: str, context_size: str) -> Dict[str, Path]:
    run_id = f"{dataset_name}_{context_size}_{CONFIG['split']}"
    root = output_root_dir(dataset_name, context_size)
    return {
        "root": root,
        "final_results": root / f"{run_id}_{CONFIG['pipeline_mode']}_final_results.jsonl",
        "traces_results": root / f"{run_id}_{CONFIG['pipeline_mode']}_traces_results.jsonl",
        "parser_errors": root / f"{run_id}_{CONFIG['pipeline_mode']}_parser_errors.jsonl",
        "plans": root / f"{run_id}_{CONFIG['pipeline_mode']}_plans.jsonl",
    }


def extract_unique_id(record: Dict[str, Any], fallback_idx: int) -> str:
    uid = record.get("unique_id")
    if uid is None:
        return f"{CONFIG['dataset']}_{CONFIG['context_size']}_{CONFIG['split']}_{fallback_idx}"
    return str(uid)


def row_unique_id(row: Dict[str, Any]) -> Optional[str]:
    value = row.get("unique_id")
    if value is None and isinstance(row.get("metadata"), dict):
        value = row["metadata"].get("unique_id")
    if value is None:
        return None
    return str(value)


def load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl_rows(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_rows_by_unique_id(existing_rows: List[Dict[str, Any]], updated_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


def chunk_indices(indices: List[int], batch_size: int) -> List[List[int]]:
    if not indices:
        return []
    if batch_size <= 0:
        batch_size = len(indices)
    return [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]


def stage_counts(states: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "active": sum(1 for s in states if s["termination_reason"] is None),
        "extractor": sum(1 for s in states if s["termination_reason"] is None and s["next_role"] == "extractor"),
        "writer": sum(1 for s in states if s["termination_reason"] is None and s["next_role"] == "writer"),
        "verifier": sum(1 for s in states if s["termination_reason"] is None and s["next_role"] == "verifier"),
        "stopped": sum(1 for s in states if s["termination_reason"] is not None),
    }


def log_stage_snapshot(states: List[Dict[str, Any]], label: str) -> None:
    counts = stage_counts(states)
    logging.info(
        "[%s] active=%s extractor=%s writer=%s verifier=%s stopped=%s",
        label,
        counts["active"],
        counts["extractor"],
        counts["writer"],
        counts["verifier"],
        counts["stopped"],
    )


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
            "source_type": error_row.get("source_type") or original.get("source_type"),
            "aspect_name": error_row.get("aspect_name") or original.get("aspect_name"),
            "aspect_summary": original.get("aspect_summary"),
            # Keep exactly what the model saw during the failed run.
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


def init_states(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    states: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        unique_id = extract_unique_id(record, idx)
        aspect_name = str(record.get("aspect_name") or "")
        source_type = str(record.get("source_type") or "unknown")
        source_text = flatten_source_text(record.get("source_text"))

        planner_input = i_o.PlannerInput(
            aspect_name=aspect_name,
            document_text=source_text,
            text_type=source_type,
        )

        states.append(
            {
                "unique_id": unique_id,
                "record": record,
                "planner_input": planner_input,
                "plan": None,
                "extractor_output": None,
                "writer_output": None,
                "verifier_output": None,
                "routing_history": [],
                "trace_steps": [],
                "iteration_count": 0,
                "termination_reason": None,
                "failed_stage": None,
                "correction_instruction": None,
                "next_role": "extractor",
                "latest_summary_text": None,
                "token_usage": {
                    "planner": [],
                    "extractor": [],
                    "writer": [],
                    "verifier": [],
                },
            }
        )
    return states


def load_plan_cache(path: Path) -> Dict[str, i_o.PlannerOutput]:
    rows = load_jsonl_rows(path)
    plans: Dict[str, i_o.PlannerOutput] = {}
    for row in rows:
        uid = row_unique_id(row)
        payload = row.get("plan")
        if uid is None or payload is None:
            continue
        try:
            plans[uid] = i_o.PlannerOutput.model_validate(payload)
        except Exception:
            continue
    return plans


def persist_plan_cache(path: Path, states: List[Dict[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    for state in states:
        plan = state.get("plan")
        if plan is None:
            continue
        rows.append({"unique_id": state["unique_id"], "plan": plan.model_dump()})
    write_jsonl_rows(rows, path)


def run_batched_planner(states: List[Dict[str, Any]], lang_engine: LanguageEngine, batch_size: int) -> None:
    pending = [i for i, s in enumerate(states) if s["termination_reason"] is None and s["plan"] is None]
    if not pending:
        logging.info("[planner] no pending states")
        return

    chunks = chunk_indices(pending, batch_size)
    logging.info("[planner] starting with %s pending states across %s chunks", len(pending), len(chunks))

    for chunk_idx, chunk in enumerate(chunks, start=1):
        prompts: List[str] = []
        for idx in chunk:
            planner_input: i_o.PlannerInput = states[idx]["planner_input"]
            _, user_prompt = PlannerPrompts.render(planner_input)
            prompts.append(user_prompt)

        plans, token_metadata = lang_engine.generate_structured_in_batch_with_token_metadata(
            user_prompts=prompts,
            pydantic_model=i_o.PlannerOutput,
            enable_thinking=True,
            system_prompt=PlannerPrompts.SYSTEM,
        )

        parsed_ok = sum(1 for plan in plans if plan is not None)
        logging.info(
            "[planner] chunk %s/%s size=%s parsed_ok=%s parse_failed=%s",
            chunk_idx,
            len(chunks),
            len(chunk),
            parsed_ok,
            len(chunk) - parsed_ok,
        )

        for idx, plan, token_info in zip(chunk, plans, token_metadata):
            states[idx]["token_usage"]["planner"].append(token_info)
            if plan is None:
                states[idx]["termination_reason"] = "planner_parse_error"
                states[idx]["failed_stage"] = "planner"
                states[idx]["next_role"] = "stop"
                continue
            states[idx]["plan"] = plan


def run_batched_extractor(states: List[Dict[str, Any]], lang_engine: LanguageEngine, batch_size: int) -> None:
    pending = [i for i, s in enumerate(states) if s["termination_reason"] is None and s["next_role"] == "extractor"]
    if not pending:
        logging.info("[extractor] no pending states")
        return

    chunks = chunk_indices(pending, batch_size)
    logging.info("[extractor] starting with %s pending states across %s chunks", len(pending), len(chunks))

    for chunk_idx, chunk in enumerate(chunks, start=1):
        prompts: List[str] = []
        for idx in chunk:
            state = states[idx]
            planner_input: i_o.PlannerInput = state["planner_input"]
            plan: Optional[i_o.PlannerOutput] = state["plan"]
            extractor_input = i_o.ExtractorInput(
                document_text=planner_input.document_text,
                aspect_name=planner_input.aspect_name,
                source_type=planner_input.text_type,
                aspect_definition=plan.aspect_definition if plan else None,
                extraction_cues=plan.extraction_cues if plan else None,
                correction_instruction=state["correction_instruction"],
            )
            _, user_prompt = ExtractorPrompts.render(extractor_input)
            prompts.append(user_prompt)

        outputs, token_metadata = lang_engine.generate_structured_in_batch_with_token_metadata(
            user_prompts=prompts,
            pydantic_model=i_o.ExtractorOutput,
            enable_thinking=False,
            system_prompt=ExtractorPrompts.SYSTEM,
        )

        parsed_ok = sum(1 for output in outputs if output is not None)
        logging.info(
            "[extractor] chunk %s/%s size=%s parsed_ok=%s parse_failed=%s",
            chunk_idx,
            len(chunks),
            len(chunk),
            parsed_ok,
            len(chunk) - parsed_ok,
        )

        for idx, output, token_info in zip(chunk, outputs, token_metadata):
            states[idx]["token_usage"]["extractor"].append(token_info)
            if output is None:
                states[idx]["termination_reason"] = "extractor_parse_error"
                states[idx]["failed_stage"] = "extractor"
                states[idx]["next_role"] = "stop"
                continue
            states[idx]["extractor_output"] = output
            states[idx]["next_role"] = "writer"


def run_batched_writer(states: List[Dict[str, Any]], lang_engine: LanguageEngine, batch_size: int, summary_length: str, pipeline_mode: str) -> None:
    pending = [i for i, s in enumerate(states) if s["termination_reason"] is None and s["next_role"] == "writer"]
    if not pending:
        logging.info("[writer] no pending states")
        return

    chunks = chunk_indices(pending, batch_size)
    logging.info("[writer] starting with %s pending states across %s chunks", len(pending), len(chunks))

    for chunk_idx, chunk in enumerate(chunks, start=1):
        prompts: List[str] = []
        for idx in chunk:
            state = states[idx]
            planner_input: i_o.PlannerInput = state["planner_input"]
            plan: Optional[i_o.PlannerOutput] = state["plan"]
            extractor_output: i_o.ExtractorOutput = state["extractor_output"]

            prior_summary = None
            if state["correction_instruction"] and state["latest_summary_text"]:
                prior_summary = state["latest_summary_text"]

            writer_input = i_o.WriterInput(
                aspect_name=planner_input.aspect_name,
                summary_length=summary_length,
                evidence_set=extractor_output.evidence_set,
                aspect_definition=plan.aspect_definition if plan else None,
                acceptance_criteria=plan.acceptance_criteria if plan else None,
                correction_instruction=state["correction_instruction"],
                prior_summary=prior_summary,
            )
            _, user_prompt = WriterPrompts.render(writer_input)
            prompts.append(user_prompt)

        outputs, token_metadata = lang_engine.generate_structured_in_batch_with_token_metadata(
            user_prompts=prompts,
            pydantic_model=i_o.WriterOutput,
            enable_thinking=False,
            system_prompt=WriterPrompts.SYSTEM,
        )

        parsed_ok = sum(1 for output in outputs if output is not None)
        logging.info(
            "[writer] chunk %s/%s size=%s parsed_ok=%s parse_failed=%s",
            chunk_idx,
            len(chunks),
            len(chunk),
            parsed_ok,
            len(chunk) - parsed_ok,
        )

        for idx, output, token_info in zip(chunk, outputs, token_metadata):
            state = states[idx]
            state["token_usage"]["writer"].append(token_info)
            if output is None:
                state["termination_reason"] = "writer_parse_error"
                state["failed_stage"] = "writer"
                state["next_role"] = "stop"
                continue

            state["writer_output"] = output
            state["latest_summary_text"] = output.summary_text

            if pipeline_mode == "no_verifier":
                state["iteration_count"] += 1
                state["termination_reason"] = "no_verifier"
                state["next_role"] = "stop"
                state["trace_steps"].append(
                    {
                        "iteration": state["iteration_count"],
                        "correction_instruction_in": state["correction_instruction"],
                        "extractor_output": state["extractor_output"].model_dump() if state["extractor_output"] else None,
                        "writer_output": state["writer_output"].model_dump() if state["writer_output"] else None,
                        "verifier_output": None,
                        "routing": {
                            "next_role": "stop",
                            "correction_instruction": None,
                            "termination_reason": state["termination_reason"],
                        },
                    }
                )
            else:
                state["next_role"] = "verifier"


def run_batched_verifier(states: List[Dict[str, Any]], lang_engine: LanguageEngine, batch_size: int, router: RouterAgent) -> None:
    pending = [i for i, s in enumerate(states) if s["termination_reason"] is None and s["next_role"] == "verifier"]
    if not pending:
        logging.info("[verifier] no pending states")
        return

    chunks = chunk_indices(pending, batch_size)
    logging.info("[verifier] starting with %s pending states across %s chunks", len(pending), len(chunks))

    for chunk_idx, chunk in enumerate(chunks, start=1):
        prompts: List[str] = []
        for idx in chunk:
            state = states[idx]
            planner_input: i_o.PlannerInput = state["planner_input"]
            plan: Optional[i_o.PlannerOutput] = state["plan"]
            extractor_output: i_o.ExtractorOutput = state["extractor_output"]
            writer_output: i_o.WriterOutput = state["writer_output"]

            verifier_input = i_o.VerifierInput(
                aspect_name=planner_input.aspect_name,
                summary_text=writer_output.summary_text,
                evidence_set=extractor_output.evidence_set,
                aspect_definition=plan.aspect_definition if plan else None,
                acceptance_criteria=plan.acceptance_criteria if plan else None,
            )
            _, user_prompt = VerifierPrompts.render(verifier_input)
            prompts.append(user_prompt)

        outputs, token_metadata = lang_engine.generate_structured_in_batch_with_token_metadata(
            user_prompts=prompts,
            pydantic_model=i_o.VerifierOutput,
            enable_thinking=True,
            system_prompt=VerifierPrompts.SYSTEM,
        )

        parsed_ok = sum(1 for output in outputs if output is not None)
        logging.info(
            "[verifier] chunk %s/%s size=%s parsed_ok=%s parse_failed=%s",
            chunk_idx,
            len(chunks),
            len(chunk),
            parsed_ok,
            len(chunk) - parsed_ok,
        )

        for idx, output, token_info in zip(chunk, outputs, token_metadata):
            state = states[idx]
            state["token_usage"]["verifier"].append(token_info)
            if output is None:
                state["termination_reason"] = "verifier_parse_error"
                state["failed_stage"] = "verifier"
                state["next_role"] = "stop"
                continue

            state["verifier_output"] = output
            route = router.route(output, current_iteration=state["iteration_count"])
            route_payload = route.model_dump()
            state["routing_history"].append(route_payload)
            state["trace_steps"].append(
                {
                    "iteration": state["iteration_count"],
                    "correction_instruction_in": state["correction_instruction"],
                    "extractor_output": state["extractor_output"].model_dump() if state["extractor_output"] else None,
                    "writer_output": state["writer_output"].model_dump() if state["writer_output"] else None,
                    "verifier_output": state["verifier_output"].model_dump() if state["verifier_output"] else None,
                    "routing": route_payload,
                }
            )
            state["iteration_count"] += 1

            if route.next_role == "stop":
                state["termination_reason"] = route.termination_reason or "stop"
                state["next_role"] = "stop"
            else:
                state["correction_instruction"] = route.correction_instruction
                state["next_role"] = route.next_role


def finalize_rows(states: List[Dict[str, Any]], context_size: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    traces_rows: List[Dict[str, Any]] = []
    final_rows: List[Dict[str, Any]] = []
    parser_error_rows: List[Dict[str, Any]] = []

    parse_fail_reasons = {
        "planner_parse_error",
        "extractor_parse_error",
        "writer_parse_error",
        "verifier_parse_error",
    }

    for state in states:
        planner_input: i_o.PlannerInput = state["planner_input"]
        record = state["record"]
        token_usage = state.get("token_usage", {})

        planner_usage = token_usage.get("planner", [])
        extractor_usage = token_usage.get("extractor", [])
        writer_usage = token_usage.get("writer", [])
        verifier_usage = token_usage.get("verifier", [])

        all_usage = planner_usage + extractor_usage + writer_usage + verifier_usage
        total_num_rollouts = sum(item.get("num_rollouts", 0) for item in all_usage)
        total_input_tokens = sum(item.get("input_tokens", 0) for item in all_usage)
        total_output_tokens = sum(item.get("output_tokens", 0) for item in all_usage)

        trace_row = {
            "unique_id": state["unique_id"],
            "dataset": record.get("dataset"),
            "context_size": context_size,
            "pipeline_mode": CONFIG["pipeline_mode"],
            "aspect_name": planner_input.aspect_name,
            "source_type": planner_input.text_type,
            "source_text": planner_input.document_text,
            "planner_input": planner_input.model_dump(),
            "plan": state["plan"].model_dump() if state["plan"] else None,
            "extractor_output": state["extractor_output"].model_dump() if state["extractor_output"] else None,
            "writer_output": state["writer_output"].model_dump() if state["writer_output"] else None,
            "verifier_output": state["verifier_output"].model_dump() if state["verifier_output"] else None,
            "routing_history": state["routing_history"],
            "trace_steps": state["trace_steps"],
            "iteration_count": state["iteration_count"],
            "termination_reason": state["termination_reason"],
            "failed_stage": state["failed_stage"],
            "token_usage": {
                "planner": planner_usage,
                "extractor": extractor_usage,
                "writer": writer_usage,
                "verifier": verifier_usage,
                "totals": {
                    "num_rollouts": total_num_rollouts,
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                },
            },
        }
        traces_rows.append(trace_row)

        writer_output: Optional[i_o.WriterOutput] = state["writer_output"]
        if writer_output is not None and state["termination_reason"] not in parse_fail_reasons:
            final_rows.append(
                {
                    "unique_id": state["unique_id"],
                    "generated_aspect_summary": writer_output.summary_text,
                    "gold_aspect_summary": record.get("aspect_summary"),
                    "aspect_name": planner_input.aspect_name,
                    "source_text": planner_input.document_text,
                    "context_size": context_size,
                    "dataset": record.get("dataset"),
                    "metadata": {
                        "unique_id": state["unique_id"],
                        "pipeline_mode": CONFIG["pipeline_mode"],
                        "termination_reason": state["termination_reason"],
                        "iteration_count": state["iteration_count"],
                        "num_rollouts": total_num_rollouts,
                        "input_tokens": total_input_tokens,
                        "output_tokens": total_output_tokens,
                        "stage_token_usage": {
                            "planner": planner_usage,
                            "extractor": extractor_usage,
                            "writer": writer_usage,
                            "verifier": verifier_usage,
                        },
                    },
                }
            )

        if state["termination_reason"] in parse_fail_reasons:
            parser_error_rows.append(
                {
                    "unique_id": state["unique_id"],
                    "dataset": record.get("dataset"),
                    "context_size": context_size,
                    "aspect_name": planner_input.aspect_name,
                    "source_type": planner_input.text_type,
                    "source_text": planner_input.document_text,
                    "termination_reason": state["termination_reason"],
                    "failed_stage": state["failed_stage"],
                }
            )

    return traces_rows, final_rows, parser_error_rows


def run_pipeline_batched(states: List[Dict[str, Any]], lang_engine: LanguageEngine, output_paths: Dict[str, Path], summary_length: str, pipeline_mode: str, batch_size: int, retry_parser_errors: bool) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    log_stage_snapshot(states, "pipeline-start")

    if pipeline_mode != "no_planner":
        cached_plans = load_plan_cache(output_paths["plans"])
        if cached_plans:
            for state in states:
                uid = state["unique_id"]
                if uid in cached_plans:
                    state["plan"] = cached_plans[uid]
            logging.info("Loaded %s plans from cache", len(cached_plans))

        run_batched_planner(states, lang_engine, batch_size)
        persist_plan_cache(output_paths["plans"], states)
        logging.info("Planner step completed and persisted to %s", output_paths["plans"])
        log_stage_snapshot(states, "post-planner")
    else:
        logging.info("Planner ablation enabled; skipping planning step")

    router = RouterAgent(max_retries=CONFIG["max_retries"])

    loop = 0
    while True:
        active_before = sum(1 for s in states if s["termination_reason"] is None)
        if active_before == 0:
            break

        loop += 1
        logging.info("Loop %s: active states=%s", loop, active_before)
        log_stage_snapshot(states, f"loop-{loop}-start")

        progressed = False

        before = sum(1 for s in states if s["termination_reason"] is None and s["next_role"] == "extractor")
        run_batched_extractor(states, lang_engine, batch_size)
        after = sum(1 for s in states if s["termination_reason"] is None and s["next_role"] == "extractor")
        if before != after:
            progressed = True
        log_stage_snapshot(states, f"loop-{loop}-post-extractor")

        before = sum(1 for s in states if s["termination_reason"] is None and s["next_role"] == "writer")
        run_batched_writer(states, lang_engine, batch_size, summary_length, pipeline_mode)
        after = sum(1 for s in states if s["termination_reason"] is None and s["next_role"] == "writer")
        if before != after:
            progressed = True
        log_stage_snapshot(states, f"loop-{loop}-post-writer")

        before = sum(1 for s in states if s["termination_reason"] is None and s["next_role"] == "verifier")
        if pipeline_mode != "no_verifier":
            run_batched_verifier(states, lang_engine, batch_size, router)
        after = sum(1 for s in states if s["termination_reason"] is None and s["next_role"] == "verifier")
        if before != after:
            progressed = True
        if pipeline_mode != "no_verifier":
            log_stage_snapshot(states, f"loop-{loop}-post-verifier")

        if not progressed:
            # Safety guard to prevent infinite loops when no state transitions occur.
            for state in states:
                if state["termination_reason"] is None:
                    state["termination_reason"] = "stalled_no_progress"
                    state["failed_stage"] = "router"
                    state["next_role"] = "stop"
                logging.warning("[router] no progress in loop %s, forcing stop for remaining active states", loop)
            break

            log_stage_snapshot(states, "pipeline-end")

    traces_rows, final_rows, parser_error_rows = finalize_rows(states, CONFIG["context_size"])

    if retry_parser_errors:
        existing_traces = load_jsonl_rows(output_paths["traces_results"])
        merged_traces = merge_rows_by_unique_id(existing_traces, traces_rows)
        write_jsonl_rows(merged_traces, output_paths["traces_results"])

        existing_final = load_jsonl_rows(output_paths["final_results"])
        merged_final = merge_rows_by_unique_id(existing_final, final_rows)
        write_jsonl_rows(merged_final, output_paths["final_results"])

        retry_ids = {s["unique_id"] for s in states}
        existing_errors = load_jsonl_rows(output_paths["parser_errors"])
        remaining_errors = [row for row in existing_errors if row_unique_id(row) not in retry_ids]
        merged_errors = merge_rows_by_unique_id(remaining_errors, parser_error_rows)
        write_jsonl_rows(merged_errors, output_paths["parser_errors"])
    else:
        write_jsonl_rows(traces_rows, output_paths["traces_results"])
        write_jsonl_rows(final_rows, output_paths["final_results"])
        write_jsonl_rows(parser_error_rows, output_paths["parser_errors"])

    return traces_rows, final_rows, parser_error_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batched 2A2S planner->extractor->writer->verifier->router runner.")
    parser.add_argument("--dataset", type=str, default=CONFIG["dataset"])
    parser.add_argument("--split", type=str, default=CONFIG["split"])
    parser.add_argument("--context-size", dest="context_size", type=str, default=CONFIG["context_size"])
    parser.add_argument("--max-samples", dest="max_samples", type=int, default=CONFIG["max_samples"])
    parser.add_argument("--max-retries", dest="max_retries", type=int, default=CONFIG["max_retries"])
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=CONFIG["batch_size"])
    parser.add_argument(
        "--temperature",
        dest="temperature",
        type=float,
        default=CONFIG["language_engine"]["temperature"],
        help="Sampling temperature for all agent generations.",
    )
    parser.add_argument(
        "--top-p",
        dest="top_p",
        type=float,
        default=CONFIG["language_engine"]["top_p"],
        help="Nucleus sampling top_p for all agent generations.",
    )
    parser.add_argument(
        "--pipeline-mode",
        dest="pipeline_mode",
        type=str,
        default=CONFIG["pipeline_mode"],
        choices=["full", "no_planner", "no_verifier"],
    )
    parser.add_argument(
        "--retry-parser-errors",
        dest="retry_parser_errors",
        action="store_true",
        help="Retry only records listed in parser_errors and merge updates by unique_id.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    CONFIG["dataset"] = args.dataset
    CONFIG["split"] = args.split
    CONFIG["context_size"] = args.context_size
    CONFIG["max_samples"] = args.max_samples
    CONFIG["max_retries"] = args.max_retries
    CONFIG["batch_size"] = args.batch_size
    CONFIG["pipeline_mode"] = args.pipeline_mode
    CONFIG["language_engine"]["temperature"] = args.temperature
    CONFIG["language_engine"]["top_p"] = args.top_p

    dataset_name = CONFIG["dataset"]
    context_size = CONFIG["context_size"]
    record_type = get_record_type(dataset_name)
    summary_length = summary_length_for_dataset(dataset_name)
    retry_parser_errors = bool(args.retry_parser_errors)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    output_paths = build_output_paths(dataset_name, context_size)
    logging.info("Output root: %s", output_paths["root"])
    logging.info("Run mode: %s", "retry" if retry_parser_errors else "main")
    logging.info("Dataset '%s' resolved to record type '%s'", dataset_name, record_type)
    logging.info("Summary length policy: %s", summary_length)
    logging.info("Batch size: %s", CONFIG["batch_size"])
    logging.info("Sampling settings: temperature=%s top_p=%s", CONFIG["language_engine"]["temperature"], CONFIG["language_engine"]["top_p"])

    dataset = load_summarization_dataset(
        split=CONFIG["split"],
        dataset_name=dataset_name,
        type=record_type,
        context_size_type=context_size,
        prompt_format="none",
    )
    logging.info("Loaded dataset rows: %s", len(dataset))
    if CONFIG["max_samples"] > 0:
        dataset = dataset[: CONFIG["max_samples"]]
        logging.info("Applied max_samples=%s; effective rows=%s", CONFIG["max_samples"], len(dataset))

    records: List[Dict[str, Any]]
    if retry_parser_errors:
        parser_error_rows = load_jsonl_rows(output_paths["parser_errors"])
        logging.info("Retry mode parser_error rows loaded: %s", len(parser_error_rows))
        if not parser_error_rows:
            logging.info("No parser_errors file found (or empty). Nothing to retry.")
            return
        dataset_lookup = build_record_lookup_by_unique_id(dataset)
        records = build_retry_records(parser_error_rows, dataset_lookup, dataset_name)
        if not records:
            logging.warning("No valid parser-error records found for retry.")
            return
    else:
        records = dataset

    logging.info("Records selected for this run: %s", len(records))

    states = init_states(records)

    t0 = time.perf_counter()
    lang_engine = build_engine()
    traces_rows, final_rows, parser_error_rows = run_pipeline_batched(
        states=states,
        lang_engine=lang_engine,
        output_paths=output_paths,
        summary_length=summary_length,
        pipeline_mode=CONFIG["pipeline_mode"],
        batch_size=int(CONFIG["batch_size"]),
        retry_parser_errors=retry_parser_errors,
    )
    elapsed = time.perf_counter() - t0

    logging.info("Run complete in %.2fs", elapsed)
    logging.info("Processed states: %s", len(states))
    logging.info("Completed summaries: %s", len(final_rows))
    logging.info("Parser errors: %s", len(parser_error_rows))
    termination_counts: Dict[str, int] = {}
    for state in states:
        reason = str(state.get("termination_reason"))
        termination_counts[reason] = termination_counts.get(reason, 0) + 1
    logging.info("Termination breakdown: %s", termination_counts)
    logging.info("Traces: %s", output_paths["traces_results"])
    logging.info("Final results: %s", output_paths["final_results"])
    logging.info("Parser errors: %s", output_paths["parser_errors"])
    logging.info("Plans cache: %s", output_paths["plans"])


if __name__ == "__main__":
    main()
