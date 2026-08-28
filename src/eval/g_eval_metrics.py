import concurrent.futures
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence, Tuple

from dotenv import load_dotenv
from openai import AzureOpenAI, BadRequestError
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from eval_prompts.GEVAL_ASP_CON import GEVAL_ASP_CON_PROMPT
from eval_prompts.GEVAL_ASP_REL import GEVAL_ASP_REL_PROMPT

load_dotenv()

# This script supports:
# 1) parallel online scoring, and
# 2) offline Batch API scoring.
# Both modes produce the same per-item keys and aggregated output structure.

# ------------------------------------------
# Configuration
# ------------------------------------------
# Set this path before running.
INPUT_FILE_PATH = "results/2a2s/e2a/qwen_qwen3_8b/facetsum_sampled/long/facetsum_sampled_long_test_e2a_final_results.jsonl"

# Select one:
# - "parallel": regular API calls with thread-based parallelism
# - "batch": asynchronous Batch API flow (cheaper on supported deployments, not realtime)
PROCESSING_MODE = "parallel"

# Optional overrides. If None, paths are derived from INPUT_FILE_PATH.
OUTPUT_FILE_PATH = None
AGGREGATE_OUTPUT_PATH = None
LOG_FILE_PATH = None

SAVE_INTERVAL = 10
PRETTY_PRINT = True
AGGREGATE_METRICS = [
    "geval_aspect_consistency",
    "geval_aspect_relevance",
]

# Parallel mode tuning
MAX_WORKERS = 12
# Global client-side throttle for parallel mode.
# Set to 0 to disable throttling.
PARALLEL_REQUESTS_PER_MINUTE = 60
# Extra retries when a 429 is returned.
MAX_RETRIES_ON_RATE_LIMIT = 8
RETRY_BASE_DELAY_SECONDS = 2.0
RETRY_MAX_DELAY_SECONDS = 70.0

# Batch mode tuning
BATCH_COMPLETION_WINDOW = "24h"
BATCH_POLL_INTERVAL_SECONDS = 20
BATCH_ENDPOINT = "/chat/completions"
BATCH_REQUESTS_FILE_PATH = None
BATCH_RAW_OUTPUT_FILE_PATH = None

# Azure OpenAI config
AZURE_ENDPOINT = os.getenv("GPT_4_1_MINI_AZURE_ENDPOINT")
API_KEY = os.getenv("GPT_4_1_MINI_AZURE_OPENAI_API_KEY")
API_VERSION = os.getenv("GPT_4_1_MINI_API_VERSION")
DEPLOYMENT_NAME = os.getenv("GPT_4_1_MINI_DEPLOYMENT_NAME")
# Batch mode must use a batch-capable Azure deployment SKU (globalbatch/datazonebatch).
# If unset, it falls back to DEPLOYMENT_NAME.
BATCH_DEPLOYMENT_NAME = os.getenv("GPT_4_1_MINI_BATCH_DEPLOYMENT_NAME", DEPLOYMENT_NAME)


def derive_output_path(input_path: str, suffix: str, extension: str) -> str:
    if input_path.endswith(".jsonl"):
        return input_path.replace(".jsonl", suffix + extension)
    base = os.path.splitext(input_path)[0]
    return base + suffix + extension


def default_log_path(input_path: str) -> str:
    project_root = Path(__file__).resolve().parents[2]
    log_dir = project_root / "logs" / "eval"
    stem = Path(input_path).stem or "geval"
    return str(log_dir / f"{stem}_g_eval_metrics.log")


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


def ensure_parent_dir(path: str) -> None:
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number} in {path}: {exc}") from exc
    return items


def save_jsonl(data: Sequence[Dict[str, Any]], path: str) -> None:
    ensure_parent_dir(path)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write("\n")
    shutil.move(tmp_path, path)


def save_json(data: Dict[str, Any], path: str, pretty: bool = True) -> None:
    ensure_parent_dir(path)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            json.dump(data, f, ensure_ascii=False)
    shutil.move(tmp_path, path)


def extract_score_from_text(content: str) -> float:
    match = re.search(r"\b([1-5])\b", content or "")
    if not match:
        return 0.0
    return float(match.group(1))


def _score_from_top_logprobs(top_logprobs: Sequence[Dict[str, Any]]) -> float:
    score_sum = 0.0
    total_prob = 0.0
    for token_data in top_logprobs:
        token_str = str(token_data.get("token", "")).strip()
        if token_str.isdigit():
            score = int(token_str)
            if 1 <= score <= 5:
                logprob = token_data.get("logprob")
                if isinstance(logprob, (float, int)):
                    prob = math.exp(float(logprob))
                    score_sum += prob * score
                    total_prob += prob
    if total_prob > 0:
        return score_sum / total_prob
    return 0.0


def extract_score_from_completion_body(body: Dict[str, Any]) -> float:
    choices = body.get("choices") or []
    if not choices:
        return 0.0
    choice = choices[0]

    # Preferred path: use token logprobs to compute expected score in [1..5].
    logprobs = choice.get("logprobs") or {}
    content_parts = logprobs.get("content") or []
    if content_parts:
        first_part = content_parts[0] if isinstance(content_parts[0], dict) else {}
        top_logprobs = first_part.get("top_logprobs") or []
        if top_logprobs:
            weighted = _score_from_top_logprobs(top_logprobs)
            if weighted > 0:
                return weighted

    # Fallback path: parse numeric content from the completion text.
    message = choice.get("message") or {}
    raw_content = message.get("content")
    if isinstance(raw_content, str):
        return extract_score_from_text(raw_content.strip())
    if isinstance(raw_content, list):
        merged = " ".join(str(block.get("text", "")) for block in raw_content if isinstance(block, dict))
        return extract_score_from_text(merged.strip())
    return 0.0


def build_prompt(aspect_name: str, gold_summary: str, generated_summary: str, dimension: str) -> str:
    if dimension == "consistency":
        template = GEVAL_ASP_CON_PROMPT
    elif dimension == "relevance":
        template = GEVAL_ASP_REL_PROMPT
    else:
        raise ValueError(f"Unknown dimension: {dimension}")
    return template.format(
        aspect=aspect_name,
        reference=gold_summary,
        summary=generated_summary,
    )


def has_required_summaries(item: Dict[str, Any]) -> bool:
    return bool((item.get("generated_aspect_summary") or "").strip()) and bool(
        (item.get("gold_aspect_summary") or "").strip()
    )


def attach_scores(item: Dict[str, Any], consistency: float, relevance: float) -> Dict[str, Any]:
    scored = dict(item)
    scored["geval_aspect_consistency"] = consistency
    scored["geval_aspect_relevance"] = relevance
    return scored


class GlobalRateLimiter:
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = max(0, requests_per_minute)
        self.interval_seconds = 0.0
        if self.requests_per_minute > 0:
            self.interval_seconds = 60.0 / float(self.requests_per_minute)
        self._lock = threading.Lock()
        self._next_allowed_time = 0.0

    def acquire(self) -> None:
        if self.interval_seconds <= 0:
            return

        # Shared pacing across worker threads to avoid burst spikes.
        while True:
            wait_time = 0.0
            with self._lock:
                now = time.monotonic()
                if now >= self._next_allowed_time:
                    self._next_allowed_time = now + self.interval_seconds
                    return
                wait_time = self._next_allowed_time - now
            if wait_time > 0:
                time.sleep(wait_time)


def _extract_retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    retry_after_ms = headers.get("retry-after-ms")
    if retry_after_ms:
        try:
            return max(0.0, float(retry_after_ms) / 1000.0)
        except (TypeError, ValueError):
            pass

    retry_after = headers.get("retry-after")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            pass

    return None


def _is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True

    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if response_status == 429:
        return True

    msg = str(exc).lower()
    return "429" in msg or "too many requests" in msg or "rate limit" in msg


class GEvalScientificMetric:
    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        api_version: str,
        deployment_name: str,
        rate_limiter: GlobalRateLimiter | None = None,
        max_retries_on_rate_limit: int = 8,
        retry_base_delay_seconds: float = 2.0,
        retry_max_delay_seconds: float = 70.0,
    ):
        if not api_key:
            raise ValueError("GPT_4_1_MINI_AZURE_OPENAI_API_KEY is not set.")
        if not azure_endpoint:
            raise ValueError("GPT_4_1_MINI_AZURE_ENDPOINT is not set.")
        if not api_version:
            raise ValueError("GPT_4_1_MINI_API_VERSION is not set.")
        if not deployment_name:
            raise ValueError("GPT_4_1_MINI_DEPLOYMENT_NAME is not set.")

        self.deployment_name = deployment_name
        self.rate_limiter = rate_limiter
        self.max_retries_on_rate_limit = max(0, int(max_retries_on_rate_limit))
        self.retry_base_delay_seconds = max(0.1, float(retry_base_delay_seconds))
        self.retry_max_delay_seconds = max(self.retry_base_delay_seconds, float(retry_max_delay_seconds))
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            max_retries=0,
        )

    def evaluate_prompt(self, prompt: str) -> float:
        for attempt in range(self.max_retries_on_rate_limit + 1):
            if self.rate_limiter is not None:
                self.rate_limiter.acquire()

            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=1,
                    logprobs=True,
                    top_logprobs=5,
                )
            except Exception as exc:
                if _is_rate_limit_error(exc):
                    if attempt >= self.max_retries_on_rate_limit:
                        logging.error(
                            "Rate limit persisted after %s attempts; returning 0.0",
                            self.max_retries_on_rate_limit + 1,
                        )
                        return 0.0

                    # Honor server-provided wait hints when present.
                    retry_after = _extract_retry_after_seconds(exc)
                    if retry_after is None:
                        # Exponential backoff + a small jitter to desynchronize workers.
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

                logging.error("API error in prompt evaluation: %s", exc)
                return 0.0

            try:
                body = response.model_dump()
                score = extract_score_from_completion_body(body)
                if score > 0:
                    return score
            except Exception:
                pass

            if not response.choices:
                logging.error("No completion choices returned.")
                return 0.0
            content = (response.choices[0].message.content or "").strip()
            return extract_score_from_text(content)

        return 0.0

    def evaluate_aspect(self, aspect_name: str, gold_summary: str, generated_summary: str, dimension: str) -> float:
        prompt = build_prompt(aspect_name, gold_summary, generated_summary, dimension)
        return self.evaluate_prompt(prompt)


def score_entry(item: Dict[str, Any], evaluator: GEvalScientificMetric) -> Dict[str, Any]:
    if not has_required_summaries(item):
        consistency = 0.0
        relevance = 0.0
    else:
        generated_summary = (item.get("generated_aspect_summary") or "").strip()
        gold_summary = (item.get("gold_aspect_summary") or "").strip()
        aspect_name = str(item.get("aspect_name") or "unknown")
        consistency = evaluator.evaluate_aspect(aspect_name, gold_summary, generated_summary, "consistency")
        relevance = evaluator.evaluate_aspect(aspect_name, gold_summary, generated_summary, "relevance")
    return attach_scores(item, consistency, relevance)


def load_checkpoint(checkpoint_output: str) -> Tuple[List[Dict[str, Any]], set]:
    existing = load_jsonl(checkpoint_output)
    done_ids = {str(item["unique_id"]) for item in existing if item.get("unique_id")}
    return existing, done_ids


def calculate_geval_metrics_parallel(
    data: Sequence[Dict[str, Any]],
    checkpoint_output: str,
    save_interval: int,
    max_workers: int,
    resume: bool = True,
) -> List[Dict[str, Any]]:
    processed: List[Dict[str, Any]] = []
    remaining = list(data)

    if resume and os.path.exists(checkpoint_output):
        processed, done_ids = load_checkpoint(checkpoint_output)
        remaining = [item for item in data if str(item.get("unique_id") or "") not in done_ids]
        logging.info(
            "Resuming from checkpoint %s: %s already scored, %s remaining",
            checkpoint_output,
            len(processed),
            len(remaining),
        )

    logging.info(
        "Running parallel mode with max_workers=%s and PARALLEL_REQUESTS_PER_MINUTE=%s",
        max_workers,
        PARALLEL_REQUESTS_PER_MINUTE,
    )
    thread_local = threading.local()
    rate_limiter = GlobalRateLimiter(PARALLEL_REQUESTS_PER_MINUTE)

    def _get_thread_evaluator() -> GEvalScientificMetric:
        evaluator = getattr(thread_local, "evaluator", None)
        if evaluator is None:
            evaluator = GEvalScientificMetric(
                azure_endpoint=AZURE_ENDPOINT,
                api_key=API_KEY,
                api_version=API_VERSION,
                deployment_name=DEPLOYMENT_NAME,
                rate_limiter=rate_limiter,
                max_retries_on_rate_limit=MAX_RETRIES_ON_RATE_LIMIT,
                retry_base_delay_seconds=RETRY_BASE_DELAY_SECONDS,
                retry_max_delay_seconds=RETRY_MAX_DELAY_SECONDS,
            )
            thread_local.evaluator = evaluator
        return evaluator

    def _worker(item: Dict[str, Any]) -> Dict[str, Any]:
        return score_entry(item, _get_thread_evaluator())

    if not remaining:
        return processed

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        iterator = executor.map(_worker, remaining)
        for index, result in enumerate(tqdm(iterator, total=len(remaining), desc="G-Eval Parallel"), start=1):
            processed.append(result)
            if save_interval > 0 and index % save_interval == 0:
                save_jsonl(processed, checkpoint_output)
                logging.info("Checkpoint saved at item %s", index)
    return processed


def make_custom_id(item_index: int, dimension: str) -> str:
    # Used to map batch responses back to original row + metric dimension.
    return f"{item_index}::{dimension}"


def build_batch_requests(data: Sequence[Dict[str, Any]], deployment_name: str) -> List[Dict[str, Any]]:
    requests: List[Dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not has_required_summaries(item):
            continue

        generated_summary = (item.get("generated_aspect_summary") or "").strip()
        gold_summary = (item.get("gold_aspect_summary") or "").strip()
        aspect_name = str(item.get("aspect_name") or "unknown")

        for dimension in ("consistency", "relevance"):
            prompt = build_prompt(aspect_name, gold_summary, generated_summary, dimension)
            requests.append(
                {
                    "custom_id": make_custom_id(idx, dimension),
                    "method": "POST",
                    "url": BATCH_ENDPOINT,
                    "body": {
                        "model": deployment_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0,
                        "max_tokens": 1,
                        "logprobs": True,
                        "top_logprobs": 5,
                    },
                }
            )
    return requests


def save_batch_requests_jsonl(requests: Sequence[Dict[str, Any]], path: str) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False))
            f.write("\n")


def _extract_file_content_text(file_content_obj: Any) -> str:
    if hasattr(file_content_obj, "text"):
        return str(file_content_obj.text)
    if hasattr(file_content_obj, "read"):
        raw = file_content_obj.read()
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        return str(raw)
    return str(file_content_obj)


def run_batch_and_get_output_text(
    client: AzureOpenAI,
    requests_path: str,
    completion_window: str,
    poll_interval_seconds: int,
    deployment_name: str,
) -> str:
    logging.info("Uploading batch requests file: %s", requests_path)
    with open(requests_path, "rb") as request_file:
        file_obj = client.files.create(file=request_file, purpose="batch")

    logging.info("Creating batch job...")
    try:
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint=BATCH_ENDPOINT,
            completion_window=completion_window,
        )
    except BadRequestError as exc:
        message = str(exc)
        lowered = message.lower()
        if "invalid_deployment_type" in lowered or "not supported for batch jobs" in lowered:
            raise ValueError(
                "Batch mode requires a batch-capable deployment (SKU: globalbatch or datazonebatch). "
                f"Current deployment='{deployment_name}'. Set GPT_4_1_MINI_BATCH_DEPLOYMENT_NAME "
                "to your batch deployment and retry."
            ) from exc
        raise
    batch_id = batch.id
    logging.info("Batch created: %s", batch_id)

    terminal_statuses = {"completed", "failed", "cancelled", "expired"}
    while True:
        current = client.batches.retrieve(batch_id)
        status = str(getattr(current, "status", "")).lower()
        logging.info("Batch %s status: %s", batch_id, status)
        if status in terminal_statuses:
            batch = current
            break
        time.sleep(poll_interval_seconds)

    final_status = str(getattr(batch, "status", "")).lower()
    if final_status != "completed":
        raise RuntimeError(f"Batch finished with non-success status: {final_status}")

    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        raise RuntimeError("Batch completed but no output_file_id was returned.")

    logging.info("Downloading batch output file: %s", output_file_id)
    content_obj = client.files.content(output_file_id)
    return _extract_file_content_text(content_obj)


def parse_batch_output_scores(output_text: str) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for line_number, raw_line in enumerate(output_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            logging.warning("Skipping malformed batch output at line %s: %s", line_number, exc)
            continue

        custom_id = record.get("custom_id")
        if not custom_id:
            continue

        if record.get("error"):
            logging.warning("Batch item failed for custom_id=%s: %s", custom_id, record.get("error"))
            continue

        response = record.get("response") or {}
        if response.get("status_code") != 200:
            logging.warning("Non-200 response for custom_id=%s: %s", custom_id, response.get("status_code"))
            continue

        # The value here is the same weighted/logprob score extraction used online.
        body = response.get("body") or {}
        scores[custom_id] = extract_score_from_completion_body(body)
    return scores


def calculate_geval_metrics_batch(
    data: Sequence[Dict[str, Any]],
    evaluator: GEvalScientificMetric,
    requests_file_path: str,
    raw_output_file_path: str | None,
) -> List[Dict[str, Any]]:
    requests = build_batch_requests(data, evaluator.deployment_name)
    if not requests:
        logging.warning("No valid request pairs found for batch mode. Returning zero-scored entries.")
        return [attach_scores(item, 0.0, 0.0) for item in data]

    save_batch_requests_jsonl(requests, requests_file_path)
    logging.info("Saved %s batch requests to %s", len(requests), requests_file_path)

    output_text = run_batch_and_get_output_text(
        client=evaluator.client,
        requests_path=requests_file_path,
        completion_window=BATCH_COMPLETION_WINDOW,
        poll_interval_seconds=BATCH_POLL_INTERVAL_SECONDS,
        deployment_name=evaluator.deployment_name,
    )

    if raw_output_file_path:
        ensure_parent_dir(raw_output_file_path)
        with open(raw_output_file_path, "w", encoding="utf-8") as f:
            f.write(output_text)
        logging.info("Saved raw batch output to %s", raw_output_file_path)

    score_map = parse_batch_output_scores(output_text)

    processed: List[Dict[str, Any]] = []
    for idx, item in enumerate(tqdm(data, desc="G-Eval Batch Postprocess")):
        # Missing/failed responses naturally fall back to 0.0.
        consistency = score_map.get(make_custom_id(idx, "consistency"), 0.0)
        relevance = score_map.get(make_custom_id(idx, "relevance"), 0.0)
        processed.append(attach_scores(item, consistency, relevance))

    return processed


def aggregate_metrics(data: Sequence[Dict[str, Any]], metrics: Sequence[str]) -> Dict[str, Any]:
    collectors: Dict[str, List[float]] = {metric: [] for metric in metrics}
    valid_count = 0

    for entry in data:
        if not isinstance(entry, dict):
            continue
        valid_count += 1
        for metric in metrics:
            value = entry.get(metric)
            if isinstance(value, (float, int)):
                collectors[metric].append(float(value))

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
        aspect = str(entry.get(aspect_key) or "unknown")
        grouped.setdefault(aspect, []).append(entry)
    return {aspect: aggregate_metrics(entries, metrics) for aspect, entries in grouped.items()}


def resolve_paths(input_path: str) -> Dict[str, str]:
    return {
        "output": OUTPUT_FILE_PATH or derive_output_path(input_path, "_geval_metrics_aspect_wise", ".jsonl"),
        "aggregate": AGGREGATE_OUTPUT_PATH
        or derive_output_path(input_path, "_geval_metrics_aggregated_aspect_wise", ".json"),
        "log": LOG_FILE_PATH or default_log_path(input_path),
        "batch_requests": BATCH_REQUESTS_FILE_PATH
        or derive_output_path(input_path, "_geval_batch_requests", ".jsonl"),
        "batch_raw_output": BATCH_RAW_OUTPUT_FILE_PATH
        or derive_output_path(input_path, "_geval_batch_raw_output", ".jsonl"),
    }


def main() -> None:
    if not INPUT_FILE_PATH or INPUT_FILE_PATH == "/path/to/input_results.jsonl":
        raise ValueError("Set INPUT_FILE_PATH at the top of src/eval/g_eval_metrics.py before running.")
    if not os.path.exists(INPUT_FILE_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE_PATH}")

    mode = PROCESSING_MODE.strip().lower()
    if mode not in {"parallel", "batch"}:
        raise ValueError("PROCESSING_MODE must be either 'parallel' or 'batch'.")

    paths = resolve_paths(INPUT_FILE_PATH)
    setup_logging(paths["log"])

    logging.info("Loading input data from %s", INPUT_FILE_PATH)
    data = load_jsonl(INPUT_FILE_PATH)
    if not data:
        raise ValueError(f"No data found in {INPUT_FILE_PATH}")

    if mode == "parallel":
        processed = calculate_geval_metrics_parallel(
            data=data,
            checkpoint_output=paths["output"],
            save_interval=SAVE_INTERVAL,
            max_workers=MAX_WORKERS,
        )
    else:
        logging.info("Batch mode using deployment=%s", BATCH_DEPLOYMENT_NAME)
        evaluator = GEvalScientificMetric(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=API_KEY,
            api_version=API_VERSION,
            deployment_name=BATCH_DEPLOYMENT_NAME,
            max_retries_on_rate_limit=MAX_RETRIES_ON_RATE_LIMIT,
            retry_base_delay_seconds=RETRY_BASE_DELAY_SECONDS,
            retry_max_delay_seconds=RETRY_MAX_DELAY_SECONDS,
        )
        processed = calculate_geval_metrics_batch(
            data=data,
            evaluator=evaluator,
            requests_file_path=paths["batch_requests"],
            raw_output_file_path=paths["batch_raw_output"],
        )

    logging.info("Saving per-example G-Eval metrics to %s", paths["output"])
    save_jsonl(processed, paths["output"])

    aggregated = {
        "total": aggregate_metrics(processed, AGGREGATE_METRICS),
        "by_aspect": aggregate_metrics_by_aspect(processed, AGGREGATE_METRICS),
    }
    logging.info("Saving aggregated metrics to %s", paths["aggregate"])
    save_json(aggregated, paths["aggregate"], pretty=PRETTY_PRINT)

    logging.info("Done. Processed %s examples using mode=%s.", len(processed), mode)


if __name__ == "__main__":
    main()
