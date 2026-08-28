import argparse
import gc
import json
import logging
import math
import re
import shutil
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterator, List, Sequence, Tuple
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
try:
    from huggingface_hub import login as hf_login
except Exception:  # pragma: no cover - optional dependency
    hf_login = None

import io_agents as i_o
from language_engine import LanguageEngine, VLLM_CONFIG
from prompts import ClaimDecompositionPrompts


CONFIG: Dict[str, Any] = {
    "root_dir": "results",
    "log_dir": "logs/eval",
    "cache_dir": ".cache/eval/fact_score",
    "decomposition": {
        "model_path": "mistralai/Mistral-Large-Instruct-2407",
        "tokenizer_mode": "mistral",
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.9,
        # Keep context bounded for memory safety; summaries are short.
        "max_model_len": 16384,
        "dtype": "bfloat16",
        "quantization": None,
        "kv_cache_dtype": "auto",
        "enforce_eager": True,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1024,
        "batch_size": 16,
    },
    "true_entailment": {
        "model_name": "google/t5_xxl_true_nli_mixture",
        "device": "auto",
        "dtype": "bfloat16",
        # Conservative defaults to reduce OOM risk on single A100.
        "batch_size": 4,
        "max_input_tokens": 1024,
        "max_new_tokens": 3,
    },
    "output": {
        "write_trace": False,
        "save_interval": 50,
        "cache_flush_interval": 200,
        "pretty_print": True,
    },
}

AGGREGATE_METRICS = [
    "fact_claim_recall",
    "fact_claim_precision",
    "fact_claim_f1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute claim-based fact metrics (CLR/CLP/F1).")
    parser.add_argument(
        "input_rel_path",
        help="Input .jsonl path relative to CONFIG['root_dir'].",
    )
    return parser.parse_args()


def setup_logging(log_path: Path) -> None:
    handlers = [logging.StreamHandler()]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(log_path, encoding="utf-8", mode="w"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def to_clean_tag(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return cleaned.strip("_") or "unknown"


def derive_output_path(input_path: Path, suffix: str, extension: str) -> Path:
    text = str(input_path)
    if text.endswith(".jsonl"):
        return Path(text.replace(".jsonl", suffix + extension))
    return Path(text + suffix + extension)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number} in {path}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected object JSON at line {line_number} in {path}")
            items.append(item)
    return items


def save_jsonl(data: Sequence[Dict[str, Any]], path: Path) -> None:
    ensure_parent_dir(path)
    tmp_path = Path(str(path) + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
    shutil.move(tmp_path, path)


def save_json(data: Dict[str, Any], path: Path, pretty: bool = True) -> None:
    ensure_parent_dir(path)
    tmp_path = Path(str(path) + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            json.dump(data, f, ensure_ascii=False)
    shutil.move(tmp_path, path)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def is_empty_or_unknown_summary(text: str) -> bool:
    normalized = normalize_text(text)
    return not normalized or normalized.lower() == "unknown"


def normalize_claims(claims: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for claim in claims:
        cleaned = normalize_text(claim)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
    return normalized


def chunked(items: Sequence[Any], chunk_size: int) -> Iterator[Sequence[Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def derive_unique_id(item: Dict[str, Any]) -> str | None:
    value = item.get("unique_id")
    if value is None:
        return None
    uid = str(value).strip()
    return uid or None


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


def harmonic_mean(a: float, b: float) -> float:
    if (a + b) == 0:
        return 0.0
    return (2 * a * b) / (a + b)


def load_gold_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return cache

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Skipping malformed gold cache line %s in %s", line_number, path)
                continue
            if not isinstance(row, dict):
                continue
            uid = row.get("unique_id")
            claims = row.get("claims")
            if not isinstance(uid, str) or not isinstance(claims, list):
                continue
            cache[uid] = {
                "gold_summary": normalize_text(str(row.get("gold_summary", ""))),
                "claims": normalize_claims([str(claim) for claim in claims]),
                "decomposition_model": row.get("decomposition_model"),
            }
    return cache


def save_gold_cache(cache: Dict[str, Dict[str, Any]], path: Path, decomposition_model: str) -> None:
    ensure_parent_dir(path)
    tmp_path = Path(str(path) + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for uid in sorted(cache.keys()):
            entry = cache[uid]
            record = {
                "unique_id": uid,
                "gold_summary": normalize_text(str(entry.get("gold_summary", ""))),
                "claims": normalize_claims(entry.get("claims", [])),
                "decomposition_model": entry.get("decomposition_model") or decomposition_model,
            }
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
    shutil.move(tmp_path, path)


class ClaimDecomposer:
    def __init__(self, cfg: Dict[str, Any]):
        engine_cfg = VLLM_CONFIG.copy()
        engine_cfg["model_path"] = cfg["model_path"]
        engine_cfg["tokenizer_mode"] = cfg.get("tokenizer_mode", "auto")
        engine_cfg["tensor_parallel_size"] = cfg["tensor_parallel_size"]
        engine_cfg["gpu_memory_utilization"] = cfg["gpu_memory_utilization"]
        engine_cfg["max_model_len"] = cfg["max_model_len"]
        engine_cfg["dtype"] = cfg["dtype"]
        engine_cfg["quantization"] = cfg["quantization"]
        engine_cfg["kv_cache_dtype"] = cfg["kv_cache_dtype"]
        engine_cfg["enforce_eager"] = cfg["enforce_eager"]
        engine_cfg["temperature"] = cfg["temperature"]
        engine_cfg["top_p"] = cfg["top_p"]
        engine_cfg["max_tokens"] = cfg["max_tokens"]
        engine_cfg["yarn_rope_scaling"] = False
        engine_cfg["native_ctx_length"] = None
        self.engine = LanguageEngine(engine_cfg)

    def decompose_batch(self, summaries: Sequence[str]) -> List[List[str]]:
        results: List[List[str]] = [[] for _ in summaries]
        active_indices: List[int] = []
        prompts: List[str] = []
        system_prompt = None

        for idx, summary in enumerate(summaries):
            if is_empty_or_unknown_summary(summary):
                continue
            model_input = i_o.ClaimDecompositionInput(summary_text=normalize_text(summary))
            current_system_prompt, user_prompt = ClaimDecompositionPrompts.render(model_input)
            if system_prompt is None:
                system_prompt = current_system_prompt
            prompts.append(user_prompt)
            active_indices.append(idx)

        if not prompts:
            return results

        outputs = self.engine.generate_structured_in_batch(
            prompts,
            pydantic_model=i_o.ClaimDecompositionOutput,
            enable_thinking=False,
            system_prompt=system_prompt,
        )
        for idx, output in zip(active_indices, outputs):
            if output is None:
                results[idx] = []
                continue
            results[idx] = normalize_claims(output.claims)

        return results


class TrueEntailmentScorer:
    def __init__(self, cfg: Dict[str, Any]):
        requested_device = str(cfg.get("device", "auto"))
        if requested_device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = requested_device

        model_kwargs: Dict[str, Any] = {}
        if self.device.startswith("cuda"):
            dtype_name = str(cfg.get("dtype", "bfloat16")).lower()
            if dtype_name == "bfloat16":
                model_kwargs["torch_dtype"] = torch.bfloat16
            elif dtype_name == "float16":
                model_kwargs["torch_dtype"] = torch.float16

        self.model_name = cfg["model_name"]
        self.batch_size = int(cfg["batch_size"])
        self.max_input_tokens = int(cfg["max_input_tokens"])
        self.max_new_tokens = int(cfg["max_new_tokens"])

        logging.info("Loading TRUE entailment model=%s on device=%s", self.model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _label_to_binary(decoded_text: str) -> int:
        text = normalize_text(decoded_text).lower()
        if not text:
            return 0
        if text[0].isdigit():
            return 1 if text[0] == "1" else 0
        if "entail" in text:
            return 1
        if text in {"true", "yes"}:
            return 1
        return 0

    def predict_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[int]:
        if not pairs:
            return []

        outputs: List[int] = []
        for pair_chunk in chunked(pairs, self.batch_size):
            model_inputs = [
                f"premise: {normalize_text(premise)} hypothesis: {normalize_text(hypothesis)}"
                for premise, hypothesis in pair_chunk
            ]
            tokenized = self.tokenizer(
                model_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
            )
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            with torch.no_grad():
                generated = self.model.generate(
                    **tokenized,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            outputs.extend(self._label_to_binary(text) for text in decoded)

        return outputs


def main() -> None:
    args = parse_args()

    input_path = Path(CONFIG["root_dir"]) / args.input_rel_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = derive_output_path(input_path, "_fact_score_aspect_wise", ".jsonl")
    aggregate_output_path = derive_output_path(input_path, "_fact_score_aggregated_aspect_wise", ".json")
    trace_output_path = derive_output_path(input_path, "_fact_score_trace_aspect_wise", ".jsonl")
    log_path = Path(CONFIG["log_dir"]) / f"{input_path.stem}_fact_score.log"

    setup_logging(log_path)
    # Load .env (if present) so HF_TOKEN can be provided via a .env file.
    try:
        from dotenv import load_dotenv

        repo_root = Path(__file__).resolve().parents[3]
        env_loaded = False
        for candidate in (Path.cwd(), repo_root):
            env_file = candidate / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                logging.info("Loaded environment variables from %s", env_file)
                env_loaded = True
                break
        if not env_loaded:
            load_dotenv()
    except Exception:
        # Fallback: simple .env parser (no dependency on python-dotenv)
        repo_root = Path(__file__).resolve().parents[3]
        for candidate in (Path.cwd(), repo_root):
            env_file = candidate / ".env"
            if env_file.exists():
                try:
                    with open(env_file, "r", encoding="utf-8") as ef:
                        for line in ef:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            if "=" not in line:
                                continue
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip('"').strip("'")
                            if k and v and k not in os.environ:
                                os.environ[k] = v
                    logging.info("Loaded .env from %s", env_file)
                    break
                except Exception as e:
                    logging.warning("Failed to read %s: %s", env_file, e)

    # Hugging Face authentication: prefer `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` env vars.
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        if hf_login is not None:
            try:
                hf_login(token=hf_token)
                logging.info("Authenticated to Hugging Face Hub via environment token")
            except Exception as exc:  # pragma: no cover - network/login issues
                logging.warning("Hugging Face login failed: %s", exc)
        else:
            logging.warning(
                "huggingface_hub not available; please `pip install huggingface-hub` or run `huggingface-cli login`",
            )
    logging.info("Loading input data from %s", input_path)
    data = load_jsonl(input_path)
    if not data:
        raise ValueError(f"No data found in {input_path}")

    dataset_tag = to_clean_tag(str(data[0].get("dataset") or "unknown"))
    context_tag = to_clean_tag(str(data[0].get("context_size") or "unknown"))
    model_tag = to_clean_tag(str(CONFIG["decomposition"]["model_path"]))
    cache_path = Path(CONFIG["cache_dir"]) / f"{dataset_tag}_{context_tag}_{model_tag}_gold_claim_cache.jsonl"

    logging.info("Using gold cache path: %s", cache_path)
    gold_cache = load_gold_cache(cache_path)
    logging.info("Loaded %s cached gold claim entries", len(gold_cache))

    decomposer = ClaimDecomposer(CONFIG["decomposition"])

    save_interval = int(CONFIG["output"]["save_interval"])
    cache_flush_interval = int(CONFIG["output"]["cache_flush_interval"])
    write_trace = bool(CONFIG["output"]["write_trace"])

    num_rows = len(data)
    gold_summaries = [normalize_text(item.get("gold_aspect_summary", "")) for item in data]
    pred_summaries = [normalize_text(item.get("generated_aspect_summary", "")) for item in data]

    gold_claims_by_index: List[List[str]] = [[] for _ in range(num_rows)]
    pred_claims_by_index: List[List[str]] = [[] for _ in range(num_rows)]

    valid_indices: List[int] = []
    pred_to_decompose_indices: List[int] = []
    pred_to_decompose_texts: List[str] = []
    gold_to_decompose_indices: List[int] = []
    gold_to_decompose_texts: List[str] = []
    gold_uid_by_index: Dict[int, str] = {}

    cache_hits = 0
    missing_unique_id_count = 0

    for idx, item in enumerate(data):
        gold_summary = gold_summaries[idx]
        pred_summary = pred_summaries[idx]
        if is_empty_or_unknown_summary(gold_summary) or is_empty_or_unknown_summary(pred_summary):
            continue

        valid_indices.append(idx)
        pred_to_decompose_indices.append(idx)
        pred_to_decompose_texts.append(pred_summary)

        uid = derive_unique_id(item)
        if uid is None:
            missing_unique_id_count += 1
            gold_to_decompose_indices.append(idx)
            gold_to_decompose_texts.append(gold_summary)
            continue

        cached_entry = gold_cache.get(uid)
        if cached_entry is not None:
            gold_claims_by_index[idx] = normalize_claims(cached_entry.get("claims", []))
            cache_hits += 1
        else:
            gold_to_decompose_indices.append(idx)
            gold_to_decompose_texts.append(gold_summary)
            gold_uid_by_index[idx] = uid

    logging.info("Rows with non-empty summaries: %s/%s", len(valid_indices), num_rows)
    logging.info("Gold cache hits: %s", cache_hits)
    if missing_unique_id_count > 0:
        logging.warning(
            "Rows without top-level unique_id (gold claims not cacheable): %s",
            missing_unique_id_count,
        )

    decomp_batch_size = int(CONFIG["decomposition"]["batch_size"])

    for text_chunk, idx_chunk in zip(
        chunked(pred_to_decompose_texts, decomp_batch_size),
        chunked(pred_to_decompose_indices, decomp_batch_size),
    ):
        claim_chunk = decomposer.decompose_batch(text_chunk)
        for row_idx, claims in zip(idx_chunk, claim_chunk):
            pred_claims_by_index[row_idx] = claims

    dirty_cache_entries = 0
    for text_chunk, idx_chunk in zip(
        chunked(gold_to_decompose_texts, decomp_batch_size),
        chunked(gold_to_decompose_indices, decomp_batch_size),
    ):
        claim_chunk = decomposer.decompose_batch(text_chunk)
        for row_idx, claims in zip(idx_chunk, claim_chunk):
            gold_claims_by_index[row_idx] = claims
            uid = gold_uid_by_index.get(row_idx)
            if uid:
                gold_cache[uid] = {
                    "gold_summary": gold_summaries[row_idx],
                    "claims": claims,
                    "decomposition_model": CONFIG["decomposition"]["model_path"],
                }
                dirty_cache_entries += 1

        if dirty_cache_entries >= cache_flush_interval:
            save_gold_cache(gold_cache, cache_path, CONFIG["decomposition"]["model_path"])
            logging.info("Flushed gold cache (%s entries)", len(gold_cache))
            dirty_cache_entries = 0

    if dirty_cache_entries > 0:
        save_gold_cache(gold_cache, cache_path, CONFIG["decomposition"]["model_path"])
        logging.info("Saved final gold cache (%s entries)", len(gold_cache))

    # Release vLLM resources before loading TRUE model to avoid GPU memory overlap.
    logging.info("Releasing decomposition engine before loading TRUE entailment model")
    del decomposer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    entailment = TrueEntailmentScorer(CONFIG["true_entailment"])

    pair_scores: Dict[Tuple[str, str], int] = {}
    pair_requests: List[Tuple[str, str]] = []

    for idx in valid_indices:
        for claim in gold_claims_by_index[idx]:
            key = (pred_summaries[idx], claim)
            if key not in pair_scores:
                pair_scores[key] = -1
                pair_requests.append(key)
        for claim in pred_claims_by_index[idx]:
            key = (gold_summaries[idx], claim)
            if key not in pair_scores:
                pair_scores[key] = -1
                pair_requests.append(key)

    if pair_requests:
        logging.info("Running TRUE entailment for %s unique premise-hypothesis pairs", len(pair_requests))
        entail_batch_size = int(CONFIG["true_entailment"]["batch_size"])
        total_batches = math.ceil(len(pair_requests) / entail_batch_size)
        for pair_chunk in tqdm(
            chunked(pair_requests, entail_batch_size),
            total=total_batches,
            desc="TRUE Entailment",
        ):
            labels = entailment.predict_pairs(pair_chunk)
            for key, label in zip(pair_chunk, labels):
                pair_scores[key] = int(label)

    processed_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []

    for processed_idx, item in enumerate(tqdm(data, desc="Fact Score"), start=1):
        row_idx = processed_idx - 1
        gold_summary = gold_summaries[row_idx]
        pred_summary = pred_summaries[row_idx]
        scored = dict(item)

        if is_empty_or_unknown_summary(gold_summary) or is_empty_or_unknown_summary(pred_summary):
            clr = 0.0
            clp = 0.0
            f1 = 0.0
            ref_claims = []
            pred_claims = []
            ref_flags: List[int] = []
            pred_flags: List[int] = []
        else:
            ref_claims = gold_claims_by_index[row_idx]
            pred_claims = pred_claims_by_index[row_idx]

            ref_flags = [pair_scores.get((pred_summary, claim), 0) for claim in ref_claims]
            pred_flags = [pair_scores.get((gold_summary, claim), 0) for claim in pred_claims]

            ref_matched = sum(ref_flags)
            pred_matched = sum(pred_flags)

            clr = (ref_matched / len(ref_claims)) if ref_claims else 0.0
            clp = (pred_matched / len(pred_claims)) if pred_claims else 0.0
            f1 = harmonic_mean(clr, clp)

        scored["fact_claim_recall"] = clr
        scored["fact_claim_precision"] = clp
        scored["fact_claim_f1"] = f1
        scored["fact_ref_claim_count"] = len(ref_claims)
        scored["fact_pred_claim_count"] = len(pred_claims)
        scored["fact_ref_claims_entailed"] = int(sum(ref_flags))
        scored["fact_pred_claims_entailed"] = int(sum(pred_flags))
        scored["fact_gold_claims"] = ref_claims
        scored["fact_generated_claims"] = pred_claims

        processed_rows.append(scored)

        if write_trace:
            trace_rows.append(
                {
                    "unique_id": item.get("unique_id"),
                    "index": row_idx,
                    "fact_claim_recall": clr,
                    "fact_claim_precision": clp,
                    "fact_claim_f1": f1,
                    "ref_claims": ref_claims,
                    "pred_claims": pred_claims,
                    "ref_claim_entailment_flags": ref_flags,
                    "pred_claim_entailment_flags": pred_flags,
                }
            )

        if save_interval > 0 and processed_idx % save_interval == 0:
            save_jsonl(processed_rows, output_path)
            if write_trace:
                save_jsonl(trace_rows, trace_output_path)
            logging.info("Checkpoint saved at row %s", processed_idx)

    save_jsonl(processed_rows, output_path)
    if write_trace:
        save_jsonl(trace_rows, trace_output_path)

    aggregated = {
        "overall": aggregate_metrics(processed_rows, AGGREGATE_METRICS),
        "by_aspect": aggregate_metrics_by_aspect(processed_rows, AGGREGATE_METRICS),
    }
    save_json(aggregated, aggregate_output_path, pretty=bool(CONFIG["output"]["pretty_print"]))

    logging.info("Saved per-example fact scores to %s", output_path)
    logging.info("Saved aggregated fact scores to %s", aggregate_output_path)
    if write_trace:
        logging.info("Saved fact trace to %s", trace_output_path)
    logging.info("Done. Processed %s rows.", len(processed_rows))


if __name__ == "__main__":
    main()
