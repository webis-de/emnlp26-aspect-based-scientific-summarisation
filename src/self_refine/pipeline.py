"""Self-Refine dataset pipeline runner."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple


os.environ.setdefault("VLLM_USE_V1", "0")

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from data_utils import load_summarization_dataset
from io_dataclasses import GeneratedDatapoint
from language_engine import LanguageEngine, VLLM_CONFIG
from config import SUMMARY_TARGETS

try:
    from .self_refine import flatten_source_text, self_refine
except ImportError:
    from self_refine import flatten_source_text, self_refine


CONFIG = {
    "base_storage": Path("results"),
    "dataset": "aclsum",
    "split": "test",
    "type": "full",
    "context_size": "long",
    "prompt_format": "none",
    "max_samples": 0,
    "language_engine": {
        "model_path": "allenai/Olmo-3-7B-Instruct",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.95,
        "max_tokens": int(8 * 2048),
    },
}


def resolve_dataset_name(name: str) -> str:
    lowered = name.lower()
    if lowered == "alcsum":
        return "aclsum"
    return lowered


def resolve_record_type(dataset_name: str, split: str, requested_type: str) -> str:
    if split == "test":
        if dataset_name == "aclsum":
            return "full"
        return "sampled"
    return requested_type


def get_summary_targets(dataset_name: str) -> Dict[str, int]:
    key = resolve_dataset_name(dataset_name)
    if key not in SUMMARY_TARGETS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Expected one of: {', '.join(SUMMARY_TARGETS)}"
        )

    targets = SUMMARY_TARGETS[key]
    return {
        "sentences": int(targets["sentences"]),
        "words": int(targets["words"]),
    }


def prepare_dataset(config: Dict[str, object]) -> Tuple[List[dict], str, str]:
    dataset_name = resolve_dataset_name(str(config["dataset"]))
    split = str(config["split"])
    record_type = resolve_record_type(dataset_name, split, str(config["type"]))

    dataset = load_summarization_dataset(
        split=split,
        dataset_name=dataset_name,
        type=record_type,
        context_size_type="long",
        prompt_format="none",
    )

    max_samples = int(config.get("max_samples", 0))
    if max_samples > 0:
        dataset = dataset[:max_samples]

    return dataset, dataset_name, record_type


def make_unique_id(record: dict, index: int) -> str:
    if record.get("unique_id"):
        return str(record["unique_id"])
    return f"{record.get('dataset', 'unknown')}_{record.get('aspect_name', 'aspect')}_{index}"


def save_jsonl(rows: List[dict], save_path: Path) -> None:
    with save_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_pipeline(config: Dict[str, object] | None = None) -> Path:
    runtime_config = dict(CONFIG)
    if config:
        runtime_config.update(config)

    dataset, dataset_name, record_type = prepare_dataset(runtime_config)
    summary_targets = get_summary_targets(dataset_name)

    engine_config = VLLM_CONFIG.copy()
    engine_config.update(runtime_config["language_engine"])
    engine = LanguageEngine(engine_config)
    generation_params = {
        "temperature": getattr(engine.sampling_params, "temperature", None),
        "top_p": getattr(engine.sampling_params, "top_p", None),
        "max_tokens": getattr(engine.sampling_params, "max_tokens", None),
    }

    run_id = (
        f"{dataset_name}_{runtime_config['split']}_{record_type}_"
        f"{runtime_config['context_size']}_self_refine"
    )
    output_dir = Path(runtime_config["base_storage"]) / "self_refine"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{run_id}_final_results.jsonl"

    rows_to_save: List[dict] = []
    for index, record in enumerate(dataset):
        source_text = flatten_source_text(record.get("source_text", ""))
        artifacts = self_refine(
            source_text=source_text,
            aspect=record.get("aspect_name", ""),
            engine=engine,
            target_sentences=summary_targets["sentences"],
            max_words=summary_targets["words"],
            enable_thinking=bool(engine_config.get("enable_thinking", False)),
        )
        unique_id = make_unique_id(record, index)

        datapoint = GeneratedDatapoint(
            generated_aspect_summary=artifacts["refined_summary"],
            gold_aspect_summary=record.get("aspect_summary", ""),
            aspect_name=record.get("aspect_name", ""),
            source_text=source_text,
            source_type=record.get("source_type", "unknown"),
            context_size="long",
            dataset=record.get("dataset", dataset_name),
            metadata={
                "unique_id": unique_id,
                "initial_summary": artifacts["initial_summary"],
                "suggestions": artifacts["suggestions"],
                "refined_summary": artifacts["refined_summary"],
                "generation_params": dict(generation_params),
                "num_rollouts": artifacts["num_rollouts"],
                "input_tokens": artifacts["input_tokens"],
                "output_tokens": artifacts["output_tokens"],
                "model_path": engine_config.get("model_path"),
            },
        )
        rows_to_save.append(asdict(datapoint))

    save_jsonl(rows_to_save, output_file)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Self-Refine summarization pipeline")
    parser.add_argument("--dataset", type=str, default=CONFIG["dataset"], help="Dataset name: aclsum, facetsum, or pmc")
    parser.add_argument("--split", type=str, default=CONFIG["split"], help="Dataset split")
    parser.add_argument("--max-samples", type=int, default=CONFIG["max_samples"], help="Maximum number of records to run")
    parser.add_argument("--output-dir", type=str, default=str(CONFIG["base_storage"]), help="Base output directory")
    parser.add_argument("--model-path", type=str, default=CONFIG["language_engine"]["model_path"], help="Model path for LanguageEngine")
    args = parser.parse_args()

    runtime_config = dict(CONFIG)
    runtime_config["dataset"] = args.dataset
    runtime_config["split"] = args.split
    runtime_config["max_samples"] = args.max_samples
    runtime_config["base_storage"] = Path(args.output_dir)

    language_engine_config = dict(CONFIG["language_engine"])
    language_engine_config["model_path"] = args.model_path
    runtime_config["language_engine"] = language_engine_config

    output_file = run_pipeline(runtime_config)
    print(f"Saved Self-Refine results to {output_file}")


if __name__ == "__main__":
    main()