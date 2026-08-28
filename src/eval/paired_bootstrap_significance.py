#!/usr/bin/env python3
"""Compute paired bootstrap significance for ZS-vs-pipeline score differences.

Example:
    python src/eval/paired_bootstrap_significance.py \
        --pair "ACLSum: ZS vs RAG" zs_metrics.jsonl rag_metrics.jsonl \
        --pair "ACLSum: ZS vs 2A2S" zs_metrics.jsonl two_a2s_metrics.jsonl \
        --output significance_table.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence


DEFAULT_METRICS = [
    "rouge1_fmeasure",
    "rouge2_fmeasure",
    "rougeL_fmeasure",
    "rougeLsum_fmeasure",
    "bertscore",
]
CONFIDENCE_LEVEL = 0.95
MAX_COMMON_SUBSET_LENGTH_MISMATCH_RATIO = 0.01
MAX_DUPLICATE_HASH_KEY_DEDUP_RATIO = 0.01

METRIC_ALIASES = {
    "r1": "rouge1_fmeasure",
    "r-1": "rouge1_fmeasure",
    "rouge1": "rouge1_fmeasure",
    "rouge_1": "rouge1_fmeasure",
    "rouge-1": "rouge1_fmeasure",
    "rouge1_fmeasure": "rouge1_fmeasure",
    "rouge_1_fmeasure": "rouge1_fmeasure",
    "r2": "rouge2_fmeasure",
    "r-2": "rouge2_fmeasure",
    "rouge2": "rouge2_fmeasure",
    "rouge_2": "rouge2_fmeasure",
    "rouge-2": "rouge2_fmeasure",
    "rouge2_fmeasure": "rouge2_fmeasure",
    "rouge_2_fmeasure": "rouge2_fmeasure",
    "rl": "rougeL_fmeasure",
    "r-l": "rougeL_fmeasure",
    "rougel": "rougeL_fmeasure",
    "rouge_l": "rougeL_fmeasure",
    "rouge-l": "rougeL_fmeasure",
    "rougel_fmeasure": "rougeL_fmeasure",
    "rouge_l_fmeasure": "rougeL_fmeasure",
    "rls": "rougeLsum_fmeasure",
    "r-l-s": "rougeLsum_fmeasure",
    "rougelsum": "rougeLsum_fmeasure",
    "rouge_l_sum": "rougeLsum_fmeasure",
    "rouge-lsum": "rougeLsum_fmeasure",
    "rouge_lsum": "rougeLsum_fmeasure",
    "rougelsum_fmeasure": "rougeLsum_fmeasure",
    "rouge_l_sum_fmeasure": "rougeLsum_fmeasure",
    "rouge_lsum_fmeasure": "rougeLsum_fmeasure",
    "bs": "bertscore",
    "bertscore": "bertscore",
    "bert_score": "bertscore",
    "claim_recall": "fact_claim_recall",
    "claim_precision": "fact_claim_precision",
    "claim_f1": "fact_claim_f1",
    "fact_claim_recall": "fact_claim_recall",
    "fact_claim_precision": "fact_claim_precision",
    "fact_claim_f1": "fact_claim_f1",
    "fact_recall": "fact_claim_recall",
    "fact_precision": "fact_claim_precision",
    "fact_f1": "fact_claim_f1",
}

DISPLAY_METRIC_NAMES = {
    "rouge1_fmeasure": "rouge_1",
    "rouge2_fmeasure": "rouge_2",
    "rougeL_fmeasure": "rouge_l",
    "rougeLsum_fmeasure": "rouge_l_sum",
    "bertscore": "bertscore",
    "fact_claim_recall": "claim_recall",
    "fact_claim_precision": "claim_precision",
    "fact_claim_f1": "claim_f1",
}

OUTPUT_HEADERS = [
    "Metric",
    "Comparison",
    "\u0394 ZS - Pipeline",
    "95% CI",
    "Significant",
]

HASH_ALIGNMENT_FIELDS = [
    "aspect_name",
    "gold_aspect_summary",
]
HASH_ALIGNMENT_KEY_LABEL = "hash alignment key"


@dataclass(frozen=True)
class PairSpec:
    comparison: str
    zs_path: Path
    pipeline_path: Path


@dataclass(frozen=True)
class BootstrapResult:
    delta: float
    lower: float
    upper: float
    significant: bool


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        raise ValueError(
            f"{path}: expected a per-example metric JSONL file, not a CSV table. "
            "Use *_standard_metrics_aspect_wise.jsonl inputs for paired bootstrap."
        )
    if path.suffix.lower() == ".json":
        raise ValueError(
            f"{path}: expected a per-example metric JSONL file, not an aggregated JSON file. "
            "Use *_standard_metrics_aspect_wise.jsonl inputs for paired bootstrap."
        )

    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            try:
                item = json.loads(stripped_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path} line {line_number}: expected a JSON object")
            items.append(item)

    if not items:
        raise ValueError(f"No metric rows found in {path}")
    return items


def normalize_metric_name(raw_metric: str) -> str:
    key = str(raw_metric).strip().lower()
    if key not in METRIC_ALIASES:
        raise ValueError(
            f"Unknown metric: {raw_metric!r}. Valid defaults/aliases include: "
            "R1, R2, RL, RLS, BS, rouge1_fmeasure, rouge2_fmeasure, "
            "rougeL_fmeasure, rougeLsum_fmeasure, bertscore."
        )
    return METRIC_ALIASES[key]


def normalize_metrics(raw_metrics: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    seen: set[str] = set()
    for raw_metric in raw_metrics:
        metric = normalize_metric_name(raw_metric)
        if metric not in seen:
            normalized.append(metric)
            seen.add(metric)
    return normalized


def numeric_metric_value(item: Dict[str, Any], metric: str, path: Path, row_number: int) -> float:
    value = item.get(metric)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ValueError(f"{path} row {row_number}: missing numeric metric {metric!r}")


def clean_hash_component(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    cleaned = "".join(
        char if char in ("\n", "\t", "\r") or ord(char) >= 32 else " " for char in value
    ).strip()
    return cleaned.lower()


def hash_alignment_key_label() -> str:
    return f"{HASH_ALIGNMENT_KEY_LABEL} ({' + '.join(HASH_ALIGNMENT_FIELDS)})"


def required_hash_payload(
    item: Dict[str, Any],
    path: Path,
    row_number: int,
) -> Dict[str, str]:
    payload: Dict[str, str] = {}
    label = hash_alignment_key_label()
    for field in HASH_ALIGNMENT_FIELDS:
        if field not in item:
            raise ValueError(
                f"{path} row {row_number}: missing required {label} field {field!r}"
            )
        normalized_value = clean_hash_component(item.get(field))
        if normalized_value == "":
            raise ValueError(
                f"{path} row {row_number}: empty required {label} field {field!r}"
            )
        payload[field] = normalized_value
    return payload


def hash_payload(payload: Dict[str, str]) -> str:
    canonical_payload = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()


def shorten_preview(value: str, limit: int = 80) -> str:
    single_line = " ".join(value.split())
    if len(single_line) <= limit:
        return single_line
    return f"{single_line[: limit - 3]}..."


def hash_payload_preview(payload: Dict[str, str]) -> str:
    return "; ".join(
        [
            f"aspect={shorten_preview(payload['aspect_name'])!r}",
            f"gold={shorten_preview(payload['gold_aspect_summary'])!r}",
        ]
    )


def derive_row_key(
    item: Dict[str, Any],
    path: Path | None = None,
    row_number: int = 0,
) -> str:
    key_path = path or Path("<item>")
    payload = required_hash_payload(item, key_path, row_number)
    return hash_payload(payload)


def collect_hash_alignment_keys(
    items: Sequence[Dict[str, Any]],
    path: Path,
) -> tuple[List[str], Dict[str, str]]:
    keys: List[str] = []
    previews: Dict[str, str] = {}
    for row_number, item in enumerate(items, start=1):
        payload = required_hash_payload(item, path, row_number)
        key = hash_payload(payload)
        keys.append(key)
        previews.setdefault(key, hash_payload_preview(payload))
    return keys, previews


def format_key_preview(value: Any, previews: Dict[Any, str] | None = None) -> str:
    if previews and value in previews:
        return f"{str(value)[:12]} ({previews[value]})"
    return str(value)


def ensure_unique(
    values: Sequence[Any],
    path: Path,
    label: str,
    previews: Dict[Any, str] | None = None,
) -> None:
    seen: set[Any] = set()
    duplicates: set[Any] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        preview = ", ".join(
            format_key_preview(value, previews) for value in sorted(duplicates, key=str)[:5]
        )
        raise ValueError(f"{path}: duplicated {label} values prevent paired alignment: {preview}")


def remove_duplicate_hash_key_items(
    items: Sequence[Dict[str, Any]],
    path: Path,
) -> List[Dict[str, Any]]:
    key_label = hash_alignment_key_label()
    keys, previews = collect_hash_alignment_keys(items, path)

    seen: set[str] = set()
    duplicate_keys: set[str] = set()
    filtered_items: List[Dict[str, Any]] = []
    for item, key in zip(items, keys):
        if key in seen:
            duplicate_keys.add(key)
            continue
        seen.add(key)
        filtered_items.append(item)

    duplicate_count = len(items) - len(filtered_items)
    if duplicate_count == 0:
        return list(items)

    duplicate_ratio = duplicate_count / len(items)
    preview = ", ".join(
        format_key_preview(key, previews) for key in sorted(duplicate_keys, key=str)[:5]
    )
    if duplicate_ratio >= MAX_DUPLICATE_HASH_KEY_DEDUP_RATIO:
        raise ValueError(
            f"{path}: duplicated {key_label} values affect {duplicate_count}/{len(items)} "
            f"rows ({duplicate_ratio:.6f}), which is not less than 0.010000; "
            f"keep-first duplicate removal is not safe: {preview}"
        )

    print(
        f"{path}: removed {duplicate_count} duplicate rows by {key_label} with "
        f"keep-first policy ({duplicate_ratio:.6f}).",
        file=sys.stderr,
    )
    return filtered_items


def align_by_hash_key(
    zs_items: Sequence[Dict[str, Any]],
    pipeline_items: Sequence[Dict[str, Any]],
    zs_path: Path,
    pipeline_path: Path,
    comparison: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    key_label = hash_alignment_key_label()
    zs_keys, zs_previews = collect_hash_alignment_keys(zs_items, zs_path)
    pipeline_keys, pipeline_previews = collect_hash_alignment_keys(pipeline_items, pipeline_path)

    ensure_unique(zs_keys, zs_path, key_label, previews=zs_previews)
    ensure_unique(pipeline_keys, pipeline_path, key_label, previews=pipeline_previews)

    zs_key_set = set(zs_keys)
    pipeline_key_set = set(pipeline_keys)
    if zs_key_set != pipeline_key_set:
        missing_in_pipeline = [
            format_key_preview(key, zs_previews) for key in sorted(zs_key_set - pipeline_key_set)[:5]
        ]
        missing_in_zs = [
            format_key_preview(key, pipeline_previews)
            for key in sorted(pipeline_key_set - zs_key_set)[:5]
        ]
        raise ValueError(
            f"{comparison}: {key_label} values do not match for "
            f"{zs_path} and {pipeline_path}. "
            f"Missing in pipeline: {missing_in_pipeline}; missing in ZS: {missing_in_zs}"
        )

    pipeline_by_key = dict(zip(pipeline_keys, pipeline_items))
    aligned_pipeline = [pipeline_by_key[key] for key in zs_keys]
    return list(zs_items), aligned_pipeline


def align_by_common_subset(
    zs_items: Sequence[Dict[str, Any]],
    pipeline_items: Sequence[Dict[str, Any]],
    zs_path: Path,
    pipeline_path: Path,
    comparison: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    key_label = hash_alignment_key_label()
    zs_keys, zs_previews = collect_hash_alignment_keys(zs_items, zs_path)
    pipeline_keys, pipeline_previews = collect_hash_alignment_keys(pipeline_items, pipeline_path)

    ensure_unique(zs_keys, zs_path, key_label, previews=zs_previews)
    ensure_unique(pipeline_keys, pipeline_path, key_label, previews=pipeline_previews)

    common_keys = set(zs_keys) & set(pipeline_keys)
    if not common_keys:
        raise ValueError(
            f"{comparison}: no common {key_label} values found for {zs_path} and {pipeline_path}."
        )

    zs_dropped = len(zs_items) - len(common_keys)
    pipeline_dropped = len(pipeline_items) - len(common_keys)
    zs_drop_ratio = zs_dropped / len(zs_items)
    pipeline_drop_ratio = pipeline_dropped / len(pipeline_items)
    if (
        zs_drop_ratio >= MAX_COMMON_SUBSET_LENGTH_MISMATCH_RATIO
        or pipeline_drop_ratio >= MAX_COMMON_SUBSET_LENGTH_MISMATCH_RATIO
    ):
        raise ValueError(
            f"{comparison}: common {key_label} subset is too small for {zs_path} and "
            f"{pipeline_path}. Matched {len(common_keys)} rows; dropped {zs_dropped} ZS "
            f"rows ({zs_drop_ratio:.6f}) and {pipeline_dropped} pipeline rows "
            f"({pipeline_drop_ratio:.6f}), which must both be less than 0.010000."
        )

    pipeline_by_key = dict(zip(pipeline_keys, pipeline_items))
    aligned_zs: List[Dict[str, Any]] = []
    aligned_pipeline: List[Dict[str, Any]] = []
    for zs_key, zs_item in zip(zs_keys, zs_items):
        if zs_key in common_keys:
            aligned_zs.append(zs_item)
            aligned_pipeline.append(pipeline_by_key[zs_key])

    print(
        f"{comparison}: using {len(aligned_zs)} common rows by {key_label}; "
        f"dropped {zs_dropped} ZS rows and {pipeline_dropped} pipeline rows.",
        file=sys.stderr,
    )
    return aligned_zs, aligned_pipeline


def load_and_align_pair(
    pair: PairSpec,
    allow_near_length_common_subset: bool = False,
    allow_duplicate_hash_key_dedup: bool = False,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    zs_items = load_jsonl(pair.zs_path)
    pipeline_items = load_jsonl(pair.pipeline_path)

    if allow_duplicate_hash_key_dedup:
        zs_items = remove_duplicate_hash_key_items(zs_items, pair.zs_path)
        pipeline_items = remove_duplicate_hash_key_items(pipeline_items, pair.pipeline_path)

    if len(zs_items) != len(pipeline_items):
        mismatch_ratio = abs(len(zs_items) - len(pipeline_items)) / max(
            len(zs_items), len(pipeline_items)
        )
        if allow_near_length_common_subset:
            if mismatch_ratio < MAX_COMMON_SUBSET_LENGTH_MISMATCH_RATIO:
                return align_by_common_subset(
                    zs_items=zs_items,
                    pipeline_items=pipeline_items,
                    zs_path=pair.zs_path,
                    pipeline_path=pair.pipeline_path,
                    comparison=pair.comparison,
                )
            raise ValueError(
                f"{pair.comparison}: input lengths differ by {mismatch_ratio:.6f}, which is "
                "not less than the 0.010000 common-subset fallback threshold. "
                f"ZS has {len(zs_items)} rows; pipeline has {len(pipeline_items)} rows."
            )
        raise ValueError(
            f"{pair.comparison}: input lengths differ. "
            f"ZS has {len(zs_items)} rows; pipeline has {len(pipeline_items)} rows."
        )

    return align_by_hash_key(
        zs_items=zs_items,
        pipeline_items=pipeline_items,
        zs_path=pair.zs_path,
        pipeline_path=pair.pipeline_path,
        comparison=pair.comparison,
    )


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile of an empty sample")
    if probability <= 0:
        return float(sorted_values[0])
    if probability >= 1:
        return float(sorted_values[-1])

    position = probability * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return float(lower_value + (upper_value - lower_value) * fraction)


def bootstrap_mean_difference(
    differences: Sequence[float],
    n_bootstrap: int,
    confidence: float,
    rng: random.Random,
) -> BootstrapResult:
    if not differences:
        raise ValueError("Cannot bootstrap an empty difference vector")
    if n_bootstrap <= 0:
        raise ValueError("--n-bootstrap must be > 0")
    if not 0 < confidence < 1:
        raise ValueError("--confidence must be between 0 and 1")

    observed_delta = mean(differences)
    sample_size = len(differences)
    bootstrap_means: List[float] = []

    for _ in range(n_bootstrap):
        sample_total = 0.0
        for _ in range(sample_size):
            sample_total += differences[rng.randrange(sample_size)]
        bootstrap_means.append(sample_total / sample_size)

    bootstrap_means.sort()
    alpha = 1.0 - confidence
    lower = percentile(bootstrap_means, alpha / 2.0)
    upper = percentile(bootstrap_means, 1.0 - alpha / 2.0)
    significant = lower > 0.0 or upper < 0.0

    return BootstrapResult(
        delta=observed_delta,
        lower=lower,
        upper=upper,
        significant=significant,
    )


def collect_differences(
    zs_items: Sequence[Dict[str, Any]],
    pipeline_items: Sequence[Dict[str, Any]],
    metric: str,
    zs_path: Path,
    pipeline_path: Path,
) -> List[float]:
    differences: List[float] = []
    for row_number, (zs_item, pipeline_item) in enumerate(zip(zs_items, pipeline_items), start=1):
        zs_score = numeric_metric_value(zs_item, metric, zs_path, row_number)
        pipeline_score = numeric_metric_value(pipeline_item, metric, pipeline_path, row_number)
        differences.append(zs_score - pipeline_score)
    return differences


def format_number(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def format_ci(lower: float, upper: float, digits: int) -> str:
    return f"[{format_number(lower, digits)}, {format_number(upper, digits)}]"


def build_table_rows(
    pairs: Sequence[PairSpec],
    metrics: Sequence[str],
    n_bootstrap: int,
    seed: int,
    digits: int,
    allow_near_length_common_subset: bool = False,
    allow_duplicate_hash_key_dedup: bool = False,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    rng = random.Random(seed)

    for pair in pairs:
        zs_items, pipeline_items = load_and_align_pair(
            pair,
            allow_near_length_common_subset=allow_near_length_common_subset,
            allow_duplicate_hash_key_dedup=allow_duplicate_hash_key_dedup,
        )
        for metric in metrics:
            differences = collect_differences(
                zs_items=zs_items,
                pipeline_items=pipeline_items,
                metric=metric,
                zs_path=pair.zs_path,
                pipeline_path=pair.pipeline_path,
            )
            result = bootstrap_mean_difference(
                differences=differences,
                n_bootstrap=n_bootstrap,
                confidence=CONFIDENCE_LEVEL,
                rng=rng,
            )
            rows.append(
                {
                    "Metric": DISPLAY_METRIC_NAMES.get(metric, metric),
                    "Comparison": pair.comparison,
                    "\u0394 ZS - Pipeline": format_number(result.delta, digits),
                    "95% CI": format_ci(result.lower, result.upper, digits),
                    "Significant": "Yes" if result.significant else "No",
                }
            )

    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=OUTPUT_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def parse_pair_specs(raw_pairs: Sequence[Sequence[str]]) -> List[PairSpec]:
    pairs: List[PairSpec] = []
    for raw_pair in raw_pairs:
        if len(raw_pair) != 3:
            raise ValueError("Each --pair must provide LABEL ZS_JSONL PIPELINE_JSONL")
        comparison, zs_path, pipeline_path = raw_pair
        pairs.append(
            PairSpec(
                comparison=comparison,
                zs_path=Path(zs_path),
                pipeline_path=Path(pipeline_path),
            )
        )
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute paired bootstrap confidence intervals for "
            "Delta = score(ZS) - score(pipeline) from per-example metric JSONL files."
        )
    )
    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        metavar=("COMPARISON", "ZS_JSONL", "PIPELINE_JSONL"),
        required=True,
        help=(
            "Comparison label plus paired per-example metric files. Repeat for multiple "
            "comparisons, e.g. --pair 'ACLSum: ZS vs RAG' zs.jsonl rag.jsonl."
        ),
    )
    parser.add_argument("--output", required=True, help="Output CSV table path.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to test. Defaults to R1/R2/RL/RLS f-measure plus BERTScore.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of paired bootstrap resamples.",
    )
    parser.add_argument(
        "--allow-near-length-common-subset",
        action="store_true",
        help=(
            "When paired inputs differ in length by less than 1%%, align and evaluate only "
            "the common keyed subset instead of failing."
        ),
    )
    parser.add_argument(
        "--dedupe-duplicate-hash-keys",
        action="store_true",
        help=(
            "Opt in to removing duplicate hash-alignment rows with a keep-first policy when "
            "duplicates affect less than 1%% of each input file."
        ),
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed for reproducible CIs.")
    parser.add_argument("--digits", type=int, default=6, help="Decimal places for table values.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.digits < 0:
        raise ValueError("--digits must be >= 0")

    pairs = parse_pair_specs(args.pair)
    metrics = normalize_metrics(args.metrics)
    rows = build_table_rows(
        pairs=pairs,
        metrics=metrics,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        digits=args.digits,
        allow_near_length_common_subset=args.allow_near_length_common_subset,
        allow_duplicate_hash_key_dedup=args.dedupe_duplicate_hash_keys,
    )
    output_path = Path(args.output)
    write_csv(output_path, rows)
    print(f"Wrote {len(rows)} paired-bootstrap rows to {output_path}")


if __name__ == "__main__":
    main()