"""W3 rebuttal pre-check: how often does the E2A extractor approach its current
limits (50 evidence spans, 1000 chars/span, 4000-token extraction budget)?

Only analyzes the frozen rebuttal subsubsample (data_utils.load_rebuttal_subsubsample),
matching the same cases used for the RAG top-k sensitivity check, so results are
directly comparable and decisions about rerunning E2A stay scoped to that sample.

This does NOT rerun any model. It reads existing E2A trace files and reports,
per dataset and aggregated:
  - % of cases where evidence_set reached the 50-item cap
  - % of cases where at least one span reached (>=95% of) the 1000-char cap
  - % of cases where extractor output_tokens reached (>=95% of) the 4000-token budget
  - % of cases where the extractor stage failed to parse (a proxy for truncation
    against the token budget, since a cut-off JSON object cannot validate)
  - the union: % of cases that approached ANY of the three caps

Usage:
    python src/e2a/analyze_extraction_caps.py --model-name qwen_qwen3.5_9b
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_utils import load_rebuttal_subsubsample

BASE_STORAGE = Path("data")
DATASETS = ["facetsum", "aclsum", "pmc"]
CONTEXT_SIZE = "long"
SPLIT = "test"

MAX_EVIDENCE_ITEMS = 50
MAX_SPAN_CHARS = 1000
EXTRACTOR_TOKEN_BUDGET = 4000

# "Near the cap" thresholds (fraction of the hard limit).
SPAN_CHAR_NEAR_FRACTION = 0.95
TOKEN_BUDGET_NEAR_FRACTION = 0.95


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


def traces_path(model_name: str, dataset: str) -> Path:
    run_id = f"{dataset}_{CONTEXT_SIZE}_{SPLIT}"
    return (
        BASE_STORAGE / "results" / "e2a" / model_name / dataset / CONTEXT_SIZE
        / f"{run_id}_e2a_traces_results.jsonl"
    )


def rebuttal_unique_ids(dataset: str) -> set:
    grouped = load_rebuttal_subsubsample(dataset)
    ids = set()
    for doc_id, records in grouped.items():
        for record in records:
            uid = record.get("unique_id")
            if uid is not None:
                ids.add(str(uid))
    return ids


def analyze_dataset(model_name: str, dataset: str) -> Dict[str, Any]:
    path = traces_path(model_name, dataset)
    all_traces = load_jsonl(path)
    allowed_ids = rebuttal_unique_ids(dataset)

    traces = [t for t in all_traces if str(t.get("unique_id")) in allowed_ids]

    n_total = len(traces)
    n_matched = n_total
    n_extractor_parse_error = 0
    n_hit_evidence_cap = 0
    n_near_span_cap = 0
    n_near_token_budget = 0
    n_any_cap = 0

    span_char_threshold = MAX_SPAN_CHARS * SPAN_CHAR_NEAR_FRACTION
    token_threshold = EXTRACTOR_TOKEN_BUDGET * TOKEN_BUDGET_NEAR_FRACTION

    for trace in traces:
        extractor_stage = (trace.get("stages") or {}).get("extractor") or {}
        status = extractor_stage.get("status")
        output = extractor_stage.get("output")
        output_tokens = extractor_stage.get("output_tokens")

        hit_any = False

        if status != "ok" or output is None:
            n_extractor_parse_error += 1
            # A parse error is itself evidence of possible truncation against
            # the token budget (a cut-off JSON object cannot validate), so we
            # count it toward the token-budget signal but do not double count
            # it as an evidence/span cap hit since we have no parsed output.
            if isinstance(output_tokens, (int, float)) and output_tokens >= token_threshold:
                n_near_token_budget += 1
                hit_any = True
            if hit_any:
                n_any_cap += 1
            continue

        evidence_set = output.get("evidence_set") or []
        n_items = len(evidence_set)
        max_span_len = max((len(span) for span in evidence_set), default=0)

        if n_items >= MAX_EVIDENCE_ITEMS:
            n_hit_evidence_cap += 1
            hit_any = True
        if max_span_len >= span_char_threshold:
            n_near_span_cap += 1
            hit_any = True
        if isinstance(output_tokens, (int, float)) and output_tokens >= token_threshold:
            n_near_token_budget += 1
            hit_any = True
        if hit_any:
            n_any_cap += 1

    def pct(n: int) -> float:
        return 100.0 * n / n_total if n_total else 0.0

    return {
        "dataset": dataset,
        "traces_path": str(path),
        "n_rebuttal_ids": len(allowed_ids),
        "n_traces_matched": n_matched,
        "n_extractor_parse_error": n_extractor_parse_error,
        "pct_extractor_parse_error": pct(n_extractor_parse_error),
        "n_hit_evidence_cap_50": n_hit_evidence_cap,
        "pct_hit_evidence_cap_50": pct(n_hit_evidence_cap),
        "n_near_span_cap_1000chars": n_near_span_cap,
        "pct_near_span_cap_1000chars": pct(n_near_span_cap),
        "n_near_token_budget_4000": n_near_token_budget,
        "pct_near_token_budget_4000": pct(n_near_token_budget),
        "n_any_cap_approached": n_any_cap,
        "pct_any_cap_approached": pct(n_any_cap),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cap-hit analysis for E2A extractor on the rebuttal subsample.")
    parser.add_argument("--model-name", type=str, default="qwen_qwen3.5_9b",
                         help="Cleaned model directory name under results/e2a/<model_name>/... "
                              "(default matches the primary Qwen3.5-9B experiment).")
    args = parser.parse_args()

    per_dataset_results = [analyze_dataset(args.model_name, dataset) for dataset in DATASETS]

    total_traces = sum(r["n_traces_matched"] for r in per_dataset_results)
    total_any_cap = sum(r["n_any_cap_approached"] for r in per_dataset_results)
    total_evidence_cap = sum(r["n_hit_evidence_cap_50"] for r in per_dataset_results)
    total_span_cap = sum(r["n_near_span_cap_1000chars"] for r in per_dataset_results)
    total_token_cap = sum(r["n_near_token_budget_4000"] for r in per_dataset_results)
    total_parse_error = sum(r["n_extractor_parse_error"] for r in per_dataset_results)

    aggregate = {
        "dataset": "ALL",
        "n_traces_matched": total_traces,
        "n_extractor_parse_error": total_parse_error,
        "pct_extractor_parse_error": 100.0 * total_parse_error / total_traces if total_traces else 0.0,
        "n_hit_evidence_cap_50": total_evidence_cap,
        "pct_hit_evidence_cap_50": 100.0 * total_evidence_cap / total_traces if total_traces else 0.0,
        "n_near_span_cap_1000chars": total_span_cap,
        "pct_near_span_cap_1000chars": 100.0 * total_span_cap / total_traces if total_traces else 0.0,
        "n_near_token_budget_4000": total_token_cap,
        "pct_near_token_budget_4000": 100.0 * total_token_cap / total_traces if total_traces else 0.0,
        "n_any_cap_approached": total_any_cap,
        "pct_any_cap_approached": 100.0 * total_any_cap / total_traces if total_traces else 0.0,
    }

    print(json.dumps({"per_dataset": per_dataset_results, "aggregate": aggregate}, indent=2))

    print("\n--- Summary ---")
    for r in per_dataset_results + [aggregate]:
        print(
            f"{r['dataset']:>10} | n={r['n_traces_matched']:>4} | "
            f"parse_error={r['pct_extractor_parse_error']:5.1f}% | "
            f"evidence_cap50={r['pct_hit_evidence_cap_50']:5.1f}% | "
            f"span_cap1000={r['pct_near_span_cap_1000chars']:5.1f}% | "
            f"token_budget4000={r['pct_near_token_budget_4000']:5.1f}% | "
            f"any_cap={r['pct_any_cap_approached']:5.1f}%"
        )

    if aggregate["pct_any_cap_approached"] < 5.0:
        print(
            "\nDECISION: fewer than 5% of rebuttal-subsample E2A cases approach any cap. "
            "Per the W3 rebuttal plan, do NOT rerun the E2A cap sensitivity configuration; "
            "report this cap-hit analysis and note relaxing the cap is unlikely to explain the result."
        )
    else:
        print(
            "\nDECISION: >=5% of rebuttal-subsample E2A cases approach at least one cap. "
            "Proceed with the E2A sensitivity rerun (max_evidence_items 50->75, "
            "extractor budget 4000->6000 tokens, span cap fixed at 1000 chars)."
        )


if __name__ == "__main__":
    main()
