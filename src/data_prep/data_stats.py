from pathlib import Path
from paths import DATA_ROOT
from typing import List, Dict
import json
import re
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
import logging

from data_utils import load_summarization_dataset, load_data_split_from_jsonl
from data_utils import DEFAULT_PMC_PATH, DEFAULT_ACLSUM_LONG_PATH, DEFAULT_ACLSUM_SHORT_PATH, DEFAULT_FACETSUM_PATH, DEFAULT_FACETSUM_PATH_LONG

BASE_STORAGE = DATA_ROOT
OUTPUT_FILE = BASE_STORAGE / "datasets_stats.json"
LOG_FILE = BASE_STORAGE / "datasets_stats.log"

# Erase existing log file if it exists
if LOG_FILE.exists():
    LOG_FILE.unlink()
# Erase existing output file if it exists
if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

# Set up logging
logging.basicConfig(filename=LOG_FILE,
                    filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)


def build_dataset_paths(dataset_name: str, context_size_type: str, split: str, sample_type: str) -> Path:
    if context_size_type == "short":
        dataset_paths = {
            "facetsum": DEFAULT_FACETSUM_PATH / f"{split}.jsonl" if sample_type == "full" else DEFAULT_FACETSUM_PATH / f"{split}_sampled_with_ids.jsonl",
            "aclsum": DEFAULT_ACLSUM_SHORT_PATH / f"{split}_with_ids.jsonl" if split == "test" else DEFAULT_ACLSUM_SHORT_PATH / f"{split}.jsonl", # aclsum does not have sampled split
            "pmc": DEFAULT_PMC_PATH / f"{split}.jsonl" if sample_type == "full" else DEFAULT_PMC_PATH / f"{split}_sampled_with_ids.jsonl",
        }
    else:  # long context size
        dataset_paths = {
            "facetsum": DEFAULT_FACETSUM_PATH_LONG / f"{split}.jsonl" if sample_type == "full" else DEFAULT_FACETSUM_PATH_LONG / f"{split}_fixed_sampled_with_ids.jsonl",
            "pmc": DEFAULT_PMC_PATH / f"{split}.jsonl" if sample_type == "full" else DEFAULT_PMC_PATH / f"{split}_sampled_with_ids.jsonl",
            "aclsum": DEFAULT_ACLSUM_LONG_PATH / f"{split}_with_ids.jsonl" if split == "test" else DEFAULT_ACLSUM_LONG_PATH / f"{split}.jsonl",
        }
    dataset_path = dataset_paths.get(dataset_name)
    return dataset_path

def count_sentences(text: str) -> int:
    if not text:
        return 0
    parts = re.split(r"[.!?]+", text)
    return sum(1 for part in parts if part.strip())


def compute_data_stats(dataset: List[Dict]) -> Dict[str, float]:
    total_source_length = 0
    total_summary_length = 0
    total_summary_sentences = 0
    num_records = len(dataset)

    for record in dataset:
        source_text = record.get("source_text", "")
        # If source_text is dict (structured), flatten it with keys and values concatenated
        if isinstance(source_text, dict):
            flattened_parts = []
            for key, value in source_text.items():
                flattened_parts.append(f"{key}: {value}")
            source_text = " ".join(flattened_parts)
        # Get summary text
        summary_text = record['aspect_summary']
        total_source_length += len(source_text.split())
        total_summary_length += len(summary_text.split())
        total_summary_sentences += count_sentences(summary_text)

    avg_source_length = total_source_length / num_records if num_records > 0 else 0
    avg_summary_length = total_summary_length / num_records if num_records > 0 else 0
    avg_summary_sentences = total_summary_sentences / num_records if num_records > 0 else 0

    return {
        "num_formatted_records": num_records,
        "avg_source_length": avg_source_length,
        "avg_summary_length": avg_summary_length,
        "avg_summary_sentences": avg_summary_sentences,
    }


def compute_dataset_issues(dataset: List[Dict], max_examples: int = 5) -> Dict[str, object]:
    """
    Scan the dataset for severe issues such as empty summaries, empty sources,
    missing keys, and duplicated records.
    """
    empty_summary_indices = []
    empty_source_indices = []
    missing_summary_indices = []
    missing_source_indices = []
    duplicate_unique_id_indices = []
    duplicate_content_indices = []

    empty_summary_count = 0
    empty_source_count = 0
    missing_summary_count = 0
    missing_source_count = 0

    seen_unique_ids = set()
    seen_content_signatures = set()
    duplicate_unique_id_count = 0
    duplicate_content_count = 0

    for idx, record in enumerate(dataset):
        summary_text = record.get("aspect_summary")
        if summary_text is None:
            missing_summary_count += 1
            if len(missing_summary_indices) < max_examples:
                missing_summary_indices.append(idx)
            summary_text = ""

        source_text = record.get("source_text")
        if source_text is None:
            missing_source_count += 1
            if len(missing_source_indices) < max_examples:
                missing_source_indices.append(idx)
            source_text = ""

        if isinstance(source_text, dict):
            flattened_parts = []
            for key, value in source_text.items():
                flattened_parts.append(f"{key}: {value}")
            source_text = " ".join(flattened_parts)

        if not str(summary_text).strip():
            empty_summary_count += 1
            if len(empty_summary_indices) < max_examples:
                empty_summary_indices.append(idx)

        if not str(source_text).strip():
            empty_source_count += 1
            if len(empty_source_indices) < max_examples:
                empty_source_indices.append(idx)

        unique_id = record.get("unique_id")
        if unique_id is not None:
            if unique_id in seen_unique_ids:
                duplicate_unique_id_count += 1
                if len(duplicate_unique_id_indices) < max_examples:
                    duplicate_unique_id_indices.append(idx)
            seen_unique_ids.add(unique_id)

        content_signature = (str(source_text).strip(), str(summary_text).strip())
        if content_signature in seen_content_signatures:
            duplicate_content_count += 1
            if len(duplicate_content_indices) < max_examples:
                duplicate_content_indices.append(idx)
        seen_content_signatures.add(content_signature)

    return {
        "num_formatted_records": len(dataset),
        "missing_summary_count": missing_summary_count,
        "missing_source_count": missing_source_count,
        "empty_summary_count": empty_summary_count,
        "empty_source_count": empty_source_count,
        "duplicate_unique_id_count": duplicate_unique_id_count,
        "duplicate_content_count": duplicate_content_count,
        "missing_summary_examples": missing_summary_indices,
        "missing_source_examples": missing_source_indices,
        "empty_summary_examples": empty_summary_indices,
        "empty_source_examples": empty_source_indices,
        "duplicate_unique_id_examples": duplicate_unique_id_indices,
        "duplicate_content_examples": duplicate_content_indices,
    }


if __name__ == "__main__":
    # Example usage
    dataset_setups = [
        ("pmc", "short"),
        ("pmc", "long"),
        ("facetsum", "short"),
        ("facetsum", "long"),
        ("aclsum", "short"),
        ("aclsum", "long"),
    ]
    all_stats = []
    for dataset_name, context_size_type in dataset_setups:
        for split in [ "train", "val", "test"]:
            if split == "test" and (dataset_name in ["facetsum", "pmc"]):
                for dataset_sample_type in ['sampled', 'full']:
                    print(f"Computing stats for dataset: {dataset_name}, context size: {context_size_type}, split: {split}, sample type: {dataset_sample_type}")
                    file_path = build_dataset_paths(dataset_name, context_size_type, split, dataset_sample_type)
                    print(f"Loading data from: {file_path}")
                    non_formatted_dataset = load_data_split_from_jsonl(file_path=file_path)
                    print(f"Loaded {len(non_formatted_dataset)} records from {dataset_name} {split} split.")
                    
                    dataset = load_summarization_dataset(split=split, dataset_name=dataset_name, prompt_format="none", context_size_type=context_size_type, type=dataset_sample_type)
                    print(f"Avg aspects per record: {len(dataset)/len(non_formatted_dataset)}")
                    stats = compute_data_stats(dataset)
                    issues_stats = compute_dataset_issues(dataset)
                    stats.update(issues_stats)
                    print(f"Stats for {dataset_name} {split} {dataset_sample_type}: {stats}")
                    log_payload = {
                        "dataset": dataset_name,
                        "context_size": context_size_type,
                        "split": split,
                        "sample_type": dataset_sample_type,
                        "num_records": len(non_formatted_dataset),
                        "avg_aspects_per_record": len(dataset)/len(non_formatted_dataset),
                        **stats,
                    }

                    logging.info(json.dumps(log_payload))
                    print(f"Logged stats to {LOG_FILE}\n")
                    all_stats.append(log_payload)
            else:
                print(f"Computing stats for dataset: {dataset_name}, context size: {context_size_type}, split: {split}")
                file_path = build_dataset_paths(dataset_name, context_size_type, split, sample_type="full")
                print(f"Loading data from: {file_path}")
                non_formatted_dataset = load_data_split_from_jsonl(
                    file_path=file_path
                )
                print(f"Loaded {len(non_formatted_dataset)} records from {dataset_name} {split} split.")
                
                dataset = load_summarization_dataset(split=split, dataset_name=dataset_name, prompt_format="none", context_size_type=context_size_type, type="full")
                print(f"Avg aspects per record: {len(dataset)/len(non_formatted_dataset)}")
                stats = compute_data_stats(dataset)
                issues_stats = compute_dataset_issues(dataset)
                stats.update(issues_stats)
                print(f"Stats for {dataset_name} {split}: {stats}")
                log_payload = {
                    "dataset": dataset_name,
                    "context_size": context_size_type,
                    "split": split,
                    "sample_type": "full",
                    "num_records": len(non_formatted_dataset),
                    "avg_aspects_per_record": len(dataset)/len(non_formatted_dataset),
                    **stats,
                }

                logging.info(json.dumps(log_payload))
                print(f"Logged stats to {LOG_FILE}\n")
                all_stats.append(log_payload)

    # Save stats to output file once
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_stats, f, indent=2)