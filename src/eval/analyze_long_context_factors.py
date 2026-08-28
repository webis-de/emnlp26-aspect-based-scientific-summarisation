#!/usr/bin/env python3
"""Analyze summary quality by document length and evidence location proxies.

Evidence annotations are not present in the result files. This script therefore
aligns each gold-summary sentence to the most similar source sentence using
within-document TF-IDF cosine similarity. Position and dispersion results must
be reported as reference-summary alignment proxies, not gold evidence spans.

Example:
    python3 src/eval/analyze_long_context_factors.py \
        --input-dir results/full \
        --output-dir results/long_context_analysis
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Sequence


DEFAULT_INPUT_DIR = Path("results/full")
DEFAULT_OUTPUT_DIR = Path("results/long_context_analysis")
DEFAULT_GLOB = "**/*standard_metrics_aspect_wise.jsonl"
DEFAULT_FACT_GLOB = "**/*fact_score_aspect_wise.jsonl"
DEFAULT_METRICS = (
    "rouge1_fmeasure",
    "rouge2_fmeasure",
    "rougeL_fmeasure",
    "rougeLsum_fmeasure",
    "bertscore",
)
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n+")
POSITION_BINS = ("early", "middle", "late")
DISPERSION_BINS = ("compact", "moderate", "dispersed")


@dataclass(frozen=True)
class EvidenceProxy:
    mean_position: float | None
    dispersion: float | None
    mean_similarity: float
    aligned_count: int
    query_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Break down per-example summarization metrics by document length and "
            "reference-summary evidence position/dispersion proxies."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--glob", default=DEFAULT_GLOB)
    parser.add_argument("--fact-glob", default=DEFAULT_FACT_GLOB)
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument(
        "--min-alignment-similarity",
        type=float,
        default=0.10,
        help="Exclude gold sentences whose best TF-IDF cosine match is below this value.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(f"{key}. {flatten_text(item)}" for key, item in value.items())
    if isinstance(value, list):
        return "\n".join(flatten_text(item) for item in value)
    return "" if value is None else str(value)


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_BOUNDARY_PATTERN.split(text) if tokenize(part)]


def stable_hash(*values: str) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def tfidf_vector(tokens: Sequence[str], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokens)
    vector = {token: count * idf.get(token, 0.0) for token, count in counts.items()}
    norm = math.sqrt(sum(value * value for value in vector.values()))
    return {token: value / norm for token, value in vector.items()} if norm else {}


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(token, 0.0) for token, value in left.items())


def align_reference_to_source(
    source_text: str,
    reference_text: str,
    min_similarity: float,
) -> EvidenceProxy:
    source_sentences = split_sentences(source_text)
    query_sentences = split_sentences(reference_text)
    if not source_sentences or not query_sentences:
        return EvidenceProxy(None, None, 0.0, 0, len(query_sentences))

    source_tokens = [tokenize(sentence) for sentence in source_sentences]
    document_frequency: Counter[str] = Counter()
    for sentence_tokens in source_tokens:
        document_frequency.update(set(sentence_tokens))
    sentence_count = len(source_tokens)
    idf = {
        token: math.log((sentence_count + 1) / (frequency + 1)) + 1.0
        for token, frequency in document_frequency.items()
    }
    source_vectors = [tfidf_vector(tokens, idf) for tokens in source_tokens]

    positions: list[float] = []
    similarities: list[float] = []
    denominator = max(sentence_count - 1, 1)
    for query_sentence in query_sentences:
        query_vector = tfidf_vector(tokenize(query_sentence), idf)
        scored = [cosine(query_vector, source_vector) for source_vector in source_vectors]
        best_index = max(range(sentence_count), key=scored.__getitem__)
        best_similarity = scored[best_index]
        similarities.append(best_similarity)
        if best_similarity >= min_similarity:
            positions.append(best_index / denominator)

    if not positions:
        return EvidenceProxy(None, None, mean(similarities), 0, len(query_sentences))
    return EvidenceProxy(
        mean_position=mean(positions),
        dispersion=max(positions) - min(positions),
        mean_similarity=mean(similarities),
        aligned_count=len(positions),
        query_count=len(query_sentences),
    )


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, value


def experiment_labels(path: Path, input_dir: Path) -> tuple[str, str]:
    relative = path.relative_to(input_dir)
    if len(relative.parts) >= 3:
        return relative.parts[0], relative.parts[1]
    if len(relative.parts) == 2:
        return input_dir.name, relative.parts[0]
    raise ValueError(
        f"Expected <model>/<method>/<file> or <method>/<file> below {input_dir}: {path}"
    )


def numeric_value(row: dict[str, Any], key: str, path: Path, line_number: int) -> float:
    value = row.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ValueError(f"{path}:{line_number}: missing numeric metric {key!r}")


def load_examples(
    files: Sequence[Path],
    input_dir: Path,
    metrics: Sequence[str],
    min_similarity: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    examples: list[dict[str, Any]] = []
    alignment_cache: dict[str, EvidenceProxy] = {}
    document_lengths: dict[str, dict[str, int]] = defaultdict(dict)

    for path in files:
        model, method = experiment_labels(path, input_dir)
        for line_number, item in read_jsonl(path):
            source_text = flatten_text(item.get("source_text"))
            reference_text = flatten_text(item.get("gold_aspect_summary"))
            if not source_text:
                raise ValueError(f"{path}:{line_number}: source_text is empty")
            dataset = str(item.get("dataset") or "unknown").lower()
            source_hash = stable_hash(source_text)
            document_lengths[dataset][source_hash] = len(tokenize(source_text))
            alignment_key = stable_hash(source_text, reference_text)
            if alignment_key not in alignment_cache:
                alignment_cache[alignment_key] = align_reference_to_source(
                    source_text, reference_text, min_similarity
                )
            proxy = alignment_cache[alignment_key]
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            example: dict[str, Any] = {
                "model": model,
                "method": method,
                "dataset": dataset,
                "unique_id": metadata.get("unique_id", ""),
                "aspect_name": item.get("aspect_name", ""),
                "document_id": source_hash[:16],
                "document_words": document_lengths[dataset][source_hash],
                "evidence_mean_position": proxy.mean_position,
                "evidence_dispersion": proxy.dispersion,
                "alignment_mean_similarity": proxy.mean_similarity,
                "aligned_reference_sentences": proxy.aligned_count,
                "reference_sentences": proxy.query_count,
            }
            example.update(
                {metric: numeric_value(item, metric, path, line_number) for metric in metrics}
            )
            examples.append(example)
    return examples, document_lengths


def load_claim_examples(files: Sequence[Path], input_dir: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for path in files:
        model, method = experiment_labels(path, input_dir)
        for line_number, item in read_jsonl(path):
            source_text = flatten_text(item.get("source_text"))
            if not source_text:
                raise ValueError(f"{path}:{line_number}: source_text is empty")
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            examples.append(
                {
                    "model": model,
                    "method": method,
                    "dataset": str(item.get("dataset") or "unknown").lower(),
                    "unique_id": item.get("unique_id") or metadata.get("unique_id", ""),
                    "aspect_name": item.get("aspect_name", ""),
                    "document_id": stable_hash(source_text)[:16],
                    "document_words": len(tokenize(source_text)),
                    "fact_claim_f1": numeric_value(item, "fact_claim_f1", path, line_number),
                }
            )
    return examples


def tertile_thresholds(values: Sequence[int]) -> tuple[int, int]:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute length bins without documents")
    return ordered[(len(ordered) - 1) // 3], ordered[(2 * (len(ordered) - 1)) // 3]


def assign_bins(
    examples: list[dict[str, Any]], document_lengths: dict[str, dict[str, int]]
) -> dict[str, dict[str, int]]:
    thresholds: dict[str, dict[str, int]] = {}
    for dataset, lengths_by_document in document_lengths.items():
        short_max, medium_max = tertile_thresholds(list(lengths_by_document.values()))
        thresholds[dataset] = {"short_max_words": short_max, "medium_max_words": medium_max}

    for example in examples:
        bounds = thresholds[example["dataset"]]
        words = example["document_words"]
        example["length_bin"] = (
            "short" if words <= bounds["short_max_words"] else
            "medium" if words <= bounds["medium_max_words"] else "long"
        )
        position = example["evidence_mean_position"]
        example["evidence_position_bin"] = (
            "unavailable" if position is None else
            POSITION_BINS[min(int(position * 3), 2)]
        )
        dispersion = example["evidence_dispersion"]
        example["evidence_dispersion_bin"] = (
            "unavailable" if dispersion is None else
            "compact" if dispersion <= 0.20 else
            "moderate" if dispersion <= 0.50 else "dispersed"
        )
    return thresholds


def assign_length_bins(
    examples: list[dict[str, Any]], thresholds: dict[str, dict[str, int]]
) -> None:
    for example in examples:
        bounds = thresholds.get(example["dataset"])
        if bounds is None:
            raise ValueError(f"No length thresholds available for dataset {example['dataset']!r}")
        words = example["document_words"]
        example["length_bin"] = (
            "short" if words <= bounds["short_max_words"] else
            "medium" if words <= bounds["medium_max_words"] else "long"
        )


def grouped_rows(
    examples: Sequence[dict[str, Any]], group_key: str, metrics: Sequence[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        key = (example["model"], example["method"], example["dataset"], example[group_key])
        groups[key].append(example)

    rows: list[dict[str, Any]] = []
    for (model, method, dataset, bin_name), group in sorted(groups.items()):
        row: dict[str, Any] = {
            "model": model,
            "method": method,
            "dataset": dataset,
            "bin": bin_name,
            "num_examples": len(group),
            "num_documents": len({item["document_id"] for item in group}),
            "mean_document_words": mean(item["document_words"] for item in group),
            "median_document_words": median(item["document_words"] for item in group),
        }
        alignment_values = [
            item["alignment_mean_similarity"]
            for item in group
            if "alignment_mean_similarity" in item
        ]
        if alignment_values:
            row["mean_alignment_similarity"] = mean(alignment_values)
        row.update({metric: mean(item[metric] for item in group) for metric in metrics})
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plots(
    output_dir: Path,
    tables: dict[str, list[dict[str, Any]]],
    metrics: Sequence[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib; rerun with --no-plots") from exc

    orders = {
        "length": ["short", "medium", "long"],
        "evidence_position": [*POSITION_BINS, "unavailable"],
        "evidence_dispersion": [*DISPERSION_BINS, "unavailable"],
    }
    for analysis_name, rows in tables.items():
        datasets = sorted({row["dataset"] for row in rows})
        for metric in metrics:
            figure, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4), squeeze=False)
            for axis, dataset in zip(axes[0], datasets):
                dataset_rows = [row for row in rows if row["dataset"] == dataset]
                for method in sorted({row["method"] for row in dataset_rows}):
                    method_rows = {row["bin"]: row for row in dataset_rows if row["method"] == method}
                    bins = [bin_name for bin_name in orders[analysis_name] if bin_name in method_rows]
                    axis.plot(bins, [method_rows[name][metric] for name in bins], marker="o", label=method)
                axis.set_title(dataset.upper())
                axis.set_xlabel(analysis_name.replace("_", " "))
                axis.set_ylabel(metric)
                axis.grid(axis="y", alpha=0.25)
                axis.tick_params(axis="x", rotation=20)
            axes[0][-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
            figure.tight_layout()
            figure.savefig(output_dir / f"{analysis_name}_{metric}.png", dpi=180, bbox_inches="tight")
            plt.close(figure)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.min_alignment_similarity <= 1.0:
        raise ValueError("--min-alignment-similarity must be between 0 and 1")
    files = sorted(args.input_dir.glob(args.glob))
    if not files:
        raise ValueError(f"No files matched {args.glob!r} below {args.input_dir}")
    fact_files = sorted(args.input_dir.glob(args.fact_glob))
    if not fact_files:
        raise ValueError(f"No files matched {args.fact_glob!r} below {args.input_dir}")

    examples, document_lengths = load_examples(
        files, args.input_dir, args.metrics, args.min_alignment_similarity
    )
    length_thresholds = assign_bins(examples, document_lengths)
    claim_examples = load_claim_examples(fact_files, args.input_dir)
    assign_length_bins(claim_examples, length_thresholds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_example.csv", examples)
    write_csv(args.output_dir / "per_example_claim_f1.csv", claim_examples)

    table_specs = {
        "length": "length_bin",
        "evidence_position": "evidence_position_bin",
        "evidence_dispersion": "evidence_dispersion_bin",
    }
    tables = {
        name: grouped_rows(examples, group_key, args.metrics)
        for name, group_key in table_specs.items()
    }
    for name, rows in tables.items():
        write_csv(args.output_dir / f"by_{name}.csv", rows)
    claim_length_rows = grouped_rows(claim_examples, "length_bin", ["fact_claim_f1"])
    write_csv(args.output_dir / "by_length_claim_f1.csv", claim_length_rows)

    aligned = [item for item in examples if item["evidence_mean_position"] is not None]
    methodology = {
        "input_dir": str(args.input_dir),
        "input_glob": args.glob,
        "num_files": len(files),
        "num_fact_files": len(fact_files),
        "num_examples": len(examples),
        "num_claim_examples": len(claim_examples),
        "num_unique_documents_by_dataset": {
            dataset: len(lengths) for dataset, lengths in sorted(document_lengths.items())
        },
        "length_bin_definition": "Dataset-specific tertiles over unique document word counts.",
        "length_thresholds_words": length_thresholds,
        "evidence_proxy_definition": (
            "Each gold-summary sentence is aligned to the source sentence with maximum "
            "within-document TF-IDF cosine similarity. Mean normalized aligned position "
            "measures evidence position; normalized min-max position range measures dispersion."
        ),
        "minimum_alignment_similarity": args.min_alignment_similarity,
        "position_bins": {"early": "[0, 1/3)", "middle": "[1/3, 2/3)", "late": "[2/3, 1]"},
        "dispersion_bins": {"compact": "[0, 0.2]", "moderate": "(0.2, 0.5]", "dispersed": "(0.5, 1]"},
        "alignment_available_examples": len(aligned),
        "alignment_coverage": len(aligned) / len(examples) if examples else 0.0,
        "mean_alignment_similarity": mean(
            item["alignment_mean_similarity"] for item in examples
        ) if examples else 0.0,
        "reporting_caution": (
            "Position and dispersion are lexical reference-summary alignment proxies because "
            "the result files do not contain gold evidence spans."
        ),
    }
    with (args.output_dir / "methodology.json").open("w", encoding="utf-8") as file_handle:
        json.dump(methodology, file_handle, indent=2)
        file_handle.write("\n")

    if not args.no_plots:
        write_plots(args.output_dir, tables, args.metrics)
        write_plots(args.output_dir, {"length": claim_length_rows}, ["fact_claim_f1"])
    print(
        f"Analyzed {len(examples)} standard-metric and {len(claim_examples)} "
        f"Claim-F1 examples; wrote {args.output_dir}"
    )


if __name__ == "__main__":
    main()