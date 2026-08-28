"""RAG Step 1: chunk, embed, retrieve — saves D_pruned to a JSONL file.

Output feeds directly into batch_generate.py.
Does NOT load vLLM; runs stella on CPU only.
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# stella_en_1.5B_v5 ships a custom modeling_qwen.py written for transformers ~4.38-4.45.
# The vLLM container's newer transformers removed several DynamicCache methods.
# Patch all of them before any model code runs.
try:
    from transformers import DynamicCache

    if not hasattr(DynamicCache, "get_max_length"):
        def _get_max_length(self):
            return None
        DynamicCache.get_max_length = _get_max_length

    if not hasattr(DynamicCache, "get_seq_length"):
        def _get_seq_length(self, layer_idx: int = 0) -> int:
            if not hasattr(self, "key_cache") or len(self.key_cache) <= layer_idx:
                return 0
            return self.key_cache[layer_idx].shape[-2]
        DynamicCache.get_seq_length = _get_seq_length

    if not hasattr(DynamicCache, "get_usable_length"):
        def _get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
            max_length = self.get_max_length()
            previous_seq_length = self.get_seq_length(layer_idx)
            if max_length is not None and previous_seq_length + new_seq_length > max_length:
                return max_length - new_seq_length
            return previous_seq_length
        DynamicCache.get_usable_length = _get_usable_length

    if not hasattr(DynamicCache, "from_legacy_cache"):
        @classmethod
        def _from_legacy_cache(cls, past_key_values=None):
            cache = cls()
            if past_key_values is not None:
                for layer_idx, kv in enumerate(past_key_values):
                    cache.update(kv[0], kv[1], layer_idx)
            return cache
        DynamicCache.from_legacy_cache = _from_legacy_cache

    if not hasattr(DynamicCache, "to_legacy_cache"):
        def _to_legacy_cache(self):
            legacy = []
            for layer_idx in range(len(self.key_cache)):
                legacy.append((self.key_cache[layer_idx], self.value_cache[layer_idx]))
            return tuple(legacy)
        DynamicCache.to_legacy_cache = _to_legacy_cache

except Exception:
    pass

try:
    from transformers import Qwen2Config
    if not hasattr(Qwen2Config, "rope_theta"):
        Qwen2Config.rope_theta = 1000000.0
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_utils import load_summarization_dataset, load_rebuttal_subsubsample
from paths import DATA_ROOT, LOG_ROOT
from retriever import Retriever

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    "base_storage": DATA_ROOT,
    "log_dir": LOG_ROOT / "rag",
    "dataset": "aclsum",
    "context_size": "long",
    "split": "test",
    "max_samples": 0,
    "batch_size": 100,
    "retriever": {
        "model_name": "NovaSearch/stella_en_1.5B_v5",
        "device": "cuda",
        "chunk_size": 256,
        "top_k_chunks": 10,
        "word_budget": 512,
    },
}

TYPE_BY_DATASET = {
    "aclsum": "full",
    "facetsum": "sampled",
    "pmc": "sampled",
}


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def setup_logging(log_dir: Path, dataset: str, context_size: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"retrieve_{dataset}_{context_size}.log"
    file_handler = logging.FileHandler(str(log_file), mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def normalize_text(text: str) -> str:
    return "".join(ch if (ch == "\n" or ord(ch) >= 32) else " " for ch in text)


def parse_source_text(raw_source: Any) -> str:
    if isinstance(raw_source, dict):
        parts = []
        for title, content in raw_source.items():
            if content:
                parts.append(f"## {normalize_text(str(title).upper().strip())}\n{normalize_text(str(content).strip())}")
        return "\n\n".join(parts).strip()
    return normalize_text(str(raw_source or "")).strip()


def extract_unique_id(record: Dict[str, Any], fallback_idx: int, dataset: str, context_size: str, split: str) -> str:
    uid = record.get("unique_id")
    if uid is None:
        return f"{dataset}_{context_size}_{split}_{fallback_idx}"
    return str(uid)


def retrieved_output_path(dataset: str, context_size: str, config_name: str = "") -> Path:
    run_id = f"{dataset}_{context_size}_{CONFIG['split']}"
    if config_name:
        return (
            CONFIG["base_storage"]
            / "results" / "rag" / "rebuttal_sensitivity" / config_name / "retrieved" / dataset / context_size
            / f"{run_id}_retrieved.jsonl"
        )
    return (
        CONFIG["base_storage"]
        / "results" / "rag" / "retrieved" / dataset / context_size
        / f"{run_id}_retrieved.jsonl"
    )


def flatten_rebuttal_subsample(dataset_name: str) -> List[Dict[str, Any]]:
    grouped = load_rebuttal_subsubsample(dataset_name)
    flat: List[Dict[str, Any]] = []
    for doc_id in grouped:
        flat.extend(grouped[doc_id])
    return flat


def count_completed(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def append_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="RAG retrieval step: chunk, embed, retrieve → save D_pruned.")
    parser.add_argument("--dataset", type=str, default=CONFIG["dataset"])
    parser.add_argument("--context-size", dest="context_size", type=str, default=CONFIG["context_size"])
    parser.add_argument("--top-k-chunks", dest="top_k_chunks", type=int, default=None,
                         help="Override retriever top_k_chunks (default: CONFIG value).")
    parser.add_argument("--chunk-size", dest="chunk_size", type=int, default=None,
                         help="Override retriever chunk_size in words (default: CONFIG value).")
    parser.add_argument("--word-budget", dest="word_budget", type=int, default=None,
                         help="Override retriever global word_budget (default: CONFIG value).")
    parser.add_argument("--config-name", dest="config_name", type=str, default="",
                         help="Name for a non-default sensitivity config; routes output under "
                              "results/rag/rebuttal_sensitivity/<config_name>/ instead of the main path.")
    parser.add_argument("--rebuttal-only", dest="rebuttal_only", action="store_true",
                         help="Restrict retrieval to the frozen rebuttal subsubsample (load_rebuttal_subsubsample).")
    parser.add_argument("--max-samples", dest="max_samples", type=int, default=CONFIG["max_samples"],
                         help="If > 0, truncate the loaded dataset to this many records (for smoke tests).")
    args = parser.parse_args()

    CONFIG["dataset"] = args.dataset
    CONFIG["context_size"] = args.context_size
    if args.top_k_chunks is not None:
        CONFIG["retriever"]["top_k_chunks"] = args.top_k_chunks
    if args.chunk_size is not None:
        CONFIG["retriever"]["chunk_size"] = args.chunk_size
    if args.word_budget is not None:
        CONFIG["retriever"]["word_budget"] = args.word_budget
    CONFIG["config_name"] = args.config_name
    CONFIG["rebuttal_only"] = args.rebuttal_only
    CONFIG["max_samples"] = args.max_samples
    dataset_name = CONFIG["dataset"]
    context_size = CONFIG["context_size"]
    split = CONFIG["split"]

    if CONFIG["rebuttal_only"] and not CONFIG["config_name"]:
        raise ValueError("--config-name is required when --rebuttal-only is set, to avoid overwriting full-run outputs.")
    if CONFIG["rebuttal_only"] and context_size != "long":
        raise ValueError("The rebuttal subsubsample is only defined for context_size='long'.")

    setup_logging(CONFIG["log_dir"], dataset_name, context_size)
    logging.info("Retrieval config: %s", CONFIG)

    if CONFIG["rebuttal_only"]:
        dataset = flatten_rebuttal_subsample(dataset_name)
        logging.info("Loaded %s records from the frozen rebuttal subsubsample", len(dataset))
    else:
        record_type = TYPE_BY_DATASET.get(dataset_name, "sampled")
        dataset = load_summarization_dataset(
            split=split,
            dataset_name=dataset_name,
            type=record_type,
            context_size_type=context_size,
            prompt_format="none",
        )
        logging.info("Loaded %s records", len(dataset))
    if CONFIG["max_samples"] > 0:
        dataset = dataset[: CONFIG["max_samples"]]

    out_path = retrieved_output_path(dataset_name, context_size, CONFIG["config_name"])
    processed = count_completed(out_path)
    remaining = dataset[processed:]
    if not remaining:
        logging.info("All records already retrieved. Exiting.")
        return
    logging.info("Completed=%s  Remaining=%s  Output=%s", processed, len(remaining), out_path)

    cfg = CONFIG["retriever"]
    logging.info("Loading retriever (%s, device=%s)...", cfg["model_name"], cfg["device"])
    retriever = Retriever(
        model_name=cfg["model_name"],
        device=cfg["device"],
        chunk_size=cfg["chunk_size"],
        top_k_chunks=cfg["top_k_chunks"],
        word_budget=cfg["word_budget"],
    )
    logging.info("Retriever ready.")

    batch_size = CONFIG["batch_size"]
    total_ok = 0
    total_err = 0

    for i in range(0, len(remaining), batch_size):
        batch = remaining[i : i + batch_size]
        source_texts = [parse_source_text(r.get("source_text")) for r in batch]
        aspect_names = [str(r.get("aspect_name") or "") for r in batch]

        t0 = time.perf_counter()
        try:
            results = retriever.prune_batch(list(zip(source_texts, aspect_names)))
        except Exception as exc:
            logging.error("prune_batch failed for batch %s-%s: %s", i, i + len(batch), exc)
            results = [(None, None)] * len(batch)

        rows = []
        for local_idx, (record, source_text, (d_pruned, stats)) in enumerate(
            zip(batch, source_texts, results)
        ):
            uid = extract_unique_id(record, processed + i + local_idx, dataset_name, context_size, split)
            if d_pruned is None:
                total_err += 1
                logging.warning("Retrieval failed for uid=%s", uid)
                continue
            rows.append({
                "unique_id": uid,
                "dataset": record.get("dataset") or dataset_name,
                "context_size": context_size,
                "aspect_name": record.get("aspect_name"),
                "gold_aspect_summary": record.get("aspect_summary"),
                "source_text": source_text,
                "d_pruned": d_pruned,
                "retrieval_stats": {
                    "num_chunks": stats["num_chunks"],
                    "num_chunks_retrieved": stats.get("num_chunks_retrieved"),
                    "retrieved_chunk_indices": stats.get("retrieved_chunk_indices"),
                    "num_sentences_retained": stats["num_sentences_retained"],
                    "num_words_retained": stats["num_words_retained"],
                },
            })
            total_ok += 1

        append_jsonl(rows, out_path)
        duration = time.perf_counter() - t0
        logging.info(
            "Batch %s-%s/%s | ok=%s failed=%s | %.2fs",
            i + 1, i + len(batch), len(remaining), len(rows), len(batch) - len(rows), duration,
        )

    logging.info("Done. ok=%s  failed=%s  output=%s", total_ok, total_err, out_path)


if __name__ == "__main__":
    main()
