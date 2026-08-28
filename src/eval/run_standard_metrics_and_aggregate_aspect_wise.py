import argparse
import json
import logging
import os
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

import nltk
import torch
from bert_score import score as bert_score_func
from rouge_score import rouge_scorer
from tqdm import tqdm


# --- CONFIGURATION (absolute paths) ---
ROOT_DIR = "results"
DEFAULT_INPUT_SUFFIX = "2a2s/results/zs/qwen3_8b/run_3/aclsum/long_temp_0p1/vllm_qwen3_8b_aclsum_long_full_results.jsonl"
LOG_BASE = "logs/eval"

BERTSCORE_MODEL = "roberta-large"
BATCH_SIZE = 32
ROUGE_METRICS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
ROUGE_SCORE_FIELDS = ["precision", "recall", "fmeasure"]

DEFAULT_METRICS = [
    *[f"{metric}_{field}" for metric in ROUGE_METRICS for field in ROUGE_SCORE_FIELDS],
    "bertscore",
]
PRETTY_PRINT = True


def ensure_nltk_resources() -> None:
    resources = ["punkt", "punkt_tab"]
    for res in resources:
        try:
            if res == "punkt":
                nltk.data.find("tokenizers/punkt")
            else:
                nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download(res)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def save_jsonl(data: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False))
            f.write("\n")


def save_json(data: Dict[str, Any], path: str, pretty: bool = True) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            json.dump(data, f, ensure_ascii=False)


def add_newlines(text: str) -> str:
    if not text:
        return ""
    sentences = nltk.sent_tokenize(text)
    return "\n".join(sentences)


def calculate_standard_metrics(
    data: Sequence[Dict[str, Any]],
    bertscore_model: str,
    batch_size: int,
    device: str,
) -> List[Dict[str, Any]]:
    generated_list = [d.get("generated_aspect_summary", "").strip() for d in data]
    reference_list = [d.get("gold_aspect_summary", "").strip() for d in data]

    logging.info("Computing BERTScore on %s using %s...", device, bertscore_model)
    _, _, f1 = bert_score_func(
        generated_list,
        reference_list,
        model_type=bertscore_model,
        lang="en",
        verbose=True,
        device=device,
        batch_size=batch_size,
    )
    bert_f1_scores = f1.tolist()

    logging.info("Computing ROUGE scores...")
    scorer = rouge_scorer.RougeScorer(ROUGE_METRICS, use_stemmer=True)

    processed_data: List[Dict[str, Any]] = []
    for i, item in enumerate(tqdm(data, desc="ROUGE Processing")):
        gen_text = generated_list[i]
        ref_text = reference_list[i]

        gen_formatted = add_newlines(gen_text)
        ref_formatted = add_newlines(ref_text)

        scores = scorer.score(ref_formatted, gen_formatted)

        new_entry = {
            "generated_aspect_summary": item.get("generated_aspect_summary"),
            "gold_aspect_summary": item.get("gold_aspect_summary"),
            "aspect_name": item.get("aspect_name"),
            "source_text": item.get("source_text"),
            "source_type": item.get("source_type"),
            "context_size": item.get("context_size"),
            "dataset": item.get("dataset"),
            "bertscore": bert_f1_scores[i],
            "metadata": item.get("metadata", {}),
        }

        for metric in ROUGE_METRICS:
            metric_scores = scores[metric]
            new_entry[f"{metric}_precision"] = metric_scores.precision
            new_entry[f"{metric}_recall"] = metric_scores.recall
            new_entry[f"{metric}_fmeasure"] = metric_scores.fmeasure

        processed_data.append(new_entry)

    return processed_data


def aggregate_metrics(data: Sequence[Dict[str, Any]], metrics: Sequence[str]) -> Dict[str, Any]:
    collectors: Dict[str, List[float]] = {metric: [] for metric in metrics}
    valid_count = 0

    for entry in data:
        if not isinstance(entry, dict):
            continue
        valid_count += 1
        for metric in metrics:
            val = entry.get(metric)
            if isinstance(val, (float, int)):
                collectors[metric].append(float(val))

    aggregated = {metric: (mean(values) if values else 0.0) for metric, values in collectors.items()}

    return {
        "num_examples": valid_count,
        "metrics": aggregated,
    }


def aggregate_metrics_by_aspect(
    data: Sequence[Dict[str, Any]],
    metrics: Sequence[str],
    aspect_key: str = "aspect_name",
) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        aspect = entry.get(aspect_key) or "unknown"
        grouped.setdefault(str(aspect), []).append(entry)

    aggregated_by_aspect: Dict[str, Any] = {}
    for aspect, entries in grouped.items():
        aggregated_by_aspect[aspect] = aggregate_metrics(entries, metrics)

    return aggregated_by_aspect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute standard metrics and save aggregated results.")
    parser.add_argument("--root-dir", default=ROOT_DIR, help="Base root directory for relative input suffix.")
    parser.add_argument(
        "--input-suffix",
        default=DEFAULT_INPUT_SUFFIX,
        help="Input path suffix to append to --root-dir (ignored if --input is provided).",
    )
    parser.add_argument("--input", default=None, help="Optional full input .jsonl path.")
    parser.add_argument("--output", default=None, help="Path to output .jsonl metrics file.")
    parser.add_argument("--aggregate-output", default=None, help="Path to output aggregated .json file.")
    parser.add_argument("--bertscore-model", default=BERTSCORE_MODEL, help="BERTScore model name.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="BERTScore batch size.")
    parser.add_argument("--device", default=None, help="Device string. Defaults to cuda if available.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metric keys to aggregate.",
    )
    parser.add_argument("--pretty", action="store_true", default=PRETTY_PRINT, help="Pretty-print aggregated JSON.")
    parser.add_argument("--log", default=None, help="Optional log file path.")
    return parser.parse_args()


def setup_logging(log_path: str | None) -> None:
    handlers = [logging.StreamHandler()]
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8", mode="w"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def main() -> None:
    args = parse_args()

    input_path = args.input or os.path.join(args.root_dir, args.input_suffix.lstrip("/"))
    output_path = args.output or input_path.replace(".jsonl", "_standard_metrics_aspect_wise.jsonl")
    aggregate_output_path = args.aggregate_output or input_path.replace(
        ".jsonl", "_standard_metrics_aggregated_aspect_wise.json"
    )
    log_path = args.log or os.path.join(
        LOG_BASE, os.path.basename(input_path).replace(".jsonl", "_standard_metrics_aspect_wise.log")
    )

    setup_logging(log_path)
    ensure_nltk_resources()

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    logging.info("Loading data from %s...", input_path)
    data = load_jsonl(input_path)
    if not data:
        raise ValueError(f"No data found in {input_path}")

    processed = calculate_standard_metrics(
        data=data,
        bertscore_model=args.bertscore_model,
        batch_size=args.batch_size,
        device=device,
    )

    logging.info("Saving per-example metrics to %s...", output_path)
    save_jsonl(processed, output_path)

    aggregated = {
        "overall": aggregate_metrics(processed, args.metrics),
        "by_aspect": aggregate_metrics_by_aspect(processed, args.metrics),
    }
    logging.info("Saving aggregated metrics to %s...", aggregate_output_path)
    save_json(aggregated, aggregate_output_path, pretty=args.pretty)

    logging.info("Done.")


if __name__ == "__main__":
    main()
