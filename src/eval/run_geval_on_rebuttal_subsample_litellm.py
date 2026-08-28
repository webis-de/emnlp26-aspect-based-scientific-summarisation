import argparse
import concurrent.futures
import logging
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
EVAL_DIR = SRC_DIR / "eval"
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(SRC_DIR))
sys.path.append(str(EVAL_DIR))

import g_eval_metrics as geval
import run_geval_on_rebuttal_subsample as rebuttal_geval

load_dotenv()

LITELLM_BASE_URL = "https://litellm.service-gateway.dev.imw.fraunhofer.de"
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
MODEL_NAME = os.getenv("LITELLM_MODEL_NAME", "gpt-4.1-mini")


class LiteLLMGEvalMetric:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        rate_limiter: geval.GlobalRateLimiter | None = None,
        max_retries_on_rate_limit: int = 8,
        retry_base_delay_seconds: float = 2.0,
        retry_max_delay_seconds: float = 70.0,
    ):
        if not api_key:
            raise ValueError("LITELLM_API_KEY is not set.")
        if not base_url:
            raise ValueError("LiteLLM base_url is not set.")
        if not model_name:
            raise ValueError("LITELLM_MODEL_NAME is not set.")

        self.model_name = model_name
        self.rate_limiter = rate_limiter
        self.max_retries_on_rate_limit = max(0, int(max_retries_on_rate_limit))
        self.retry_base_delay_seconds = max(0.1, float(retry_base_delay_seconds))
        self.retry_max_delay_seconds = max(self.retry_base_delay_seconds, float(retry_max_delay_seconds))
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    def evaluate_prompt(self, prompt: str) -> float:
        for attempt in range(self.max_retries_on_rate_limit + 1):
            if self.rate_limiter is not None:
                self.rate_limiter.acquire()

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=1,
                    logprobs=True,
                    top_logprobs=5,
                )
            except Exception as exc:
                if geval._is_rate_limit_error(exc):
                    if attempt >= self.max_retries_on_rate_limit:
                        logging.error(
                            "Rate limit persisted after %s attempts; returning 0.0",
                            self.max_retries_on_rate_limit + 1,
                        )
                        return 0.0
                    retry_after = geval._extract_retry_after_seconds(exc)
                    if retry_after is None:
                        retry_after = min(
                            self.retry_max_delay_seconds,
                            self.retry_base_delay_seconds * (2 ** attempt),
                        )
                        retry_after += random.uniform(0.0, 0.5)
                    logging.warning(
                        "Rate limited (429). Sleeping %.2fs before retry %s/%s",
                        retry_after,
                        attempt + 1,
                        self.max_retries_on_rate_limit,
                    )
                    time.sleep(retry_after)
                    continue
                logging.error("LiteLLM API error in prompt evaluation: %s", exc)
                return 0.0

            try:
                body = response.model_dump()
                score = geval.extract_score_from_completion_body(body)
                if score > 0:
                    return score
            except Exception:
                pass

            if not response.choices:
                logging.error("No completion choices returned.")
                return 0.0
            content = (response.choices[0].message.content or "").strip()
            return geval.extract_score_from_text(content)

        return 0.0

    def evaluate_aspect(self, aspect_name: str, reference_text: str, generated_summary: str, dimension: str) -> float:
        prompt = geval.build_prompt(aspect_name, reference_text, generated_summary, dimension)
        return self.evaluate_prompt(prompt)


def score_entry(item: Dict[str, Any], evaluator: LiteLLMGEvalMetric) -> Dict[str, Any]:
    if not geval.has_required_summaries(item):
        return geval.attach_scores(item, 0.0, 0.0)

    generated_summary = (item.get("generated_aspect_summary") or "").strip()
    reference_text = (item.get("gold_aspect_summary") or "").strip()
    aspect_name = str(item.get("aspect_name") or "unknown")
    consistency = evaluator.evaluate_aspect(aspect_name, reference_text, generated_summary, "consistency")
    relevance = evaluator.evaluate_aspect(aspect_name, reference_text, generated_summary, "relevance")
    return geval.attach_scores(item, consistency, relevance)


def calculate_metrics_parallel(
    data: Sequence[Dict[str, Any]],
    checkpoint_output: str,
    save_interval: int,
    max_workers: int,
    requests_per_minute: int,
    resume: bool,
) -> List[Dict[str, Any]]:
    processed: List[Dict[str, Any]] = []
    remaining = list(data)

    if resume and os.path.exists(checkpoint_output):
        processed, done_ids = geval.load_checkpoint(checkpoint_output)
        remaining = [item for item in data if str(item.get("unique_id") or "") not in done_ids]
        logging.info(
            "Resuming from checkpoint %s: %s already scored, %s remaining",
            checkpoint_output,
            len(processed),
            len(remaining),
        )

    logging.info(
        "Running LiteLLM parallel mode with model=%s, max_workers=%s, requests_per_minute=%s",
        MODEL_NAME,
        max_workers,
        requests_per_minute,
    )
    thread_local = threading.local()
    rate_limiter = geval.GlobalRateLimiter(requests_per_minute)

    def _get_thread_evaluator() -> LiteLLMGEvalMetric:
        evaluator = getattr(thread_local, "evaluator", None)
        if evaluator is None:
            evaluator = LiteLLMGEvalMetric(
                api_key=LITELLM_API_KEY,
                base_url=LITELLM_BASE_URL,
                model_name=MODEL_NAME,
                rate_limiter=rate_limiter,
                max_retries_on_rate_limit=geval.MAX_RETRIES_ON_RATE_LIMIT,
                retry_base_delay_seconds=geval.RETRY_BASE_DELAY_SECONDS,
                retry_max_delay_seconds=geval.RETRY_MAX_DELAY_SECONDS,
            )
            thread_local.evaluator = evaluator
        return evaluator

    def _worker(item: Dict[str, Any]) -> Dict[str, Any]:
        return score_entry(item, _get_thread_evaluator())

    if not remaining:
        return processed

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        iterator = executor.map(_worker, remaining)
        for index, result in enumerate(tqdm(iterator, total=len(remaining), desc="LiteLLM G-Eval"), start=1):
            processed.append(result)
            if save_interval > 0 and index % save_interval == 0:
                geval.save_jsonl(processed, checkpoint_output)
                logging.info("Checkpoint saved at item %s", index)
    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run rebuttal-subsample G-Eval through the Fraunhofer LiteLLM endpoint."
    )
    parser.add_argument("--input", required=True, help="Experiment result JSONL path.")
    parser.add_argument("--dataset", required=True, choices=["facetsum", "pmc", "aclsum"], help="Dataset name.")
    parser.add_argument("--output", default=None, help="Per-example output JSONL path.")
    parser.add_argument("--aggregate-output", default=None, help="Aggregate output JSON path.")
    parser.add_argument("--log", default=None, help="Optional log path.")
    parser.add_argument("--generated-field", default=None, help="Override the generated summary field name.")
    parser.add_argument("--max-workers", type=int, default=6, help="Parallel worker count.")
    parser.add_argument("--requests-per-minute", type=int, default=18)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--no-resume", action="store_true", help="Ignore any existing checkpoint at the output path.")
    parser.add_argument("--allow-missing", action="store_true", help="Do not fail when some sample records are missing.")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional smoke-test limit after matching.")
    parser.add_argument("--prepare-only", action="store_true", help="Only write matched G-Eval input records; do not call LiteLLM.")
    parser.add_argument("--pretty", action="store_true", default=geval.PRETTY_PRINT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_path = args.output or rebuttal_geval.derive_output_path(
        input_path,
        "_rebuttal_subsample_litellm_geval_metrics_aspect_wise",
        ".jsonl",
    )
    aggregate_output_path = args.aggregate_output or rebuttal_geval.derive_output_path(
        input_path,
        "_rebuttal_subsample_litellm_geval_metrics_aggregated_aspect_wise",
        ".json",
    )
    log_path = args.log or str(
        PROJECT_ROOT / "logs" / "eval" / (Path(input_path).stem + "_rebuttal_subsample_litellm_geval.log")
    )

    geval.setup_logging(log_path)
    logging.info("Loading experiment results from %s", input_path)
    result_records = geval.load_jsonl(input_path)
    if not result_records:
        raise ValueError(f"No result records found in {input_path}")

    prepared, stats = rebuttal_geval.prepare_geval_input(
        result_records=result_records,
        dataset_name=args.dataset,
        generated_field=args.generated_field,
    )
    if args.max_examples is not None:
        prepared = prepared[: args.max_examples]
        stats["prepared_examples_after_max_examples"] = len(prepared)

    logging.info("Prepared %s LiteLLM G-Eval examples. Stats: %s", len(prepared), stats)
    if not prepared:
        raise ValueError("No records matched the rebuttal subsubsample and generated-summary field.")
    if stats["unmatched_sample_examples"] and not args.allow_missing and args.max_examples is None:
        raise ValueError(
            "Some rebuttal subsubsample examples were not found in the result file. "
            "Use --allow-missing to score only matched records. "
            f"Stats: {stats}"
        )

    if args.prepare_only:
        logging.info("Prepare-only mode: saving matched G-Eval input records to %s", output_path)
        geval.save_jsonl(prepared, output_path)
        geval.save_json(
            {
                "metadata": {
                    "input": input_path,
                    "dataset": args.dataset,
                    "mode": "prepare-only",
                    "endpoint": LITELLM_BASE_URL,
                    "model": MODEL_NAME,
                    "reference": "full_source_text_from_load_rebuttal_subsubsample",
                    "stats": stats,
                }
            },
            aggregate_output_path,
            pretty=args.pretty,
        )
        logging.info("Done.")
        return

    processed = calculate_metrics_parallel(
        data=prepared,
        checkpoint_output=output_path,
        save_interval=args.save_interval,
        max_workers=args.max_workers,
        requests_per_minute=args.requests_per_minute,
        resume=not args.no_resume,
    )

    logging.info("Saving per-example LiteLLM G-Eval metrics to %s", output_path)
    geval.save_jsonl(processed, output_path)

    aggregate = {
        "overall": geval.aggregate_metrics(processed, rebuttal_geval.GEVAL_METRICS),
        "by_aspect": geval.aggregate_metrics_by_aspect(processed, rebuttal_geval.GEVAL_METRICS),
        "metadata": {
            "input": input_path,
            "dataset": args.dataset,
            "mode": "parallel",
            "endpoint": LITELLM_BASE_URL,
            "model": MODEL_NAME,
            "reference": "full_source_text_from_load_rebuttal_subsubsample",
            "stats": stats,
        },
    }
    logging.info("Saving aggregate LiteLLM G-Eval metrics to %s", aggregate_output_path)
    geval.save_json(aggregate, aggregate_output_path, pretty=args.pretty)
    logging.info("Done.")


if __name__ == "__main__":
    main()
