import argparse
import json
from pathlib import Path
from typing import List, Dict
import random
from paths import DATA_ROOT

try:
    from datasets import Dataset, DatasetDict  # type: ignore[reportMissingImports]
except ImportError:
    Dataset = None
    DatasetDict = None

try:
    from transformers import (  # type: ignore[reportMissingImports]
        AutoTokenizer,
        BartForConditionalGeneration,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
except ImportError:
    AutoTokenizer = None
    BartForConditionalGeneration = None
    DataCollatorForSeq2Seq = None
    Seq2SeqTrainer = None
    Seq2SeqTrainingArguments = None

DEFAULT_DATA_ROOT = DATA_ROOT
DEFAULT_FACETSUM_PATH = DEFAULT_DATA_ROOT / "facetsum"
DEFAULT_FACETSUM_PATH_LONG = DEFAULT_DATA_ROOT / "facetsum_long"
DEFAULT_ACLSUM_SHORT_PATH = DEFAULT_DATA_ROOT / "aclsum" / "short"
DEFAULT_ACLSUM_LONG_PATH = DEFAULT_DATA_ROOT / "aclsum" / "long"
#DEFAULT_SCHOLARSUM_PATH = DEFAULT_DATA_ROOT / "scholarsum" / "arxiv"
DEFAULT_PMC_PATH = DEFAULT_DATA_ROOT / "pmc" / "extracted" / "splits"

SOURCE_TYPE_SHORT = {
    "facetsum": "introduction-conclusion",
    "aclsum": "abstract-introduction-conclusion",
    "pmc": "aspect-section",
}
SOURCE_TYPE_LONG = {
    "facetsum": "full-text-structured",
    #"scholarsum": "full-text",
    "aclsum": "full-text",
    "pmc": "full-text-structured",
}

ASPECT_NORMALIZATION = {
    "facetsum": {
        "Purpose": "Purpose",
        "Design/methodology/approach": "Methods",
        "Findings": "Findings",
        "Originality/value": "Value",
    },
    "aclsum": {
        "challenge": "Challenge",
        "approach": "Approach",
        "outcome": "Outcome",
    },
    "pmc": {
        "background": "Background",
        "methods": "Methods",
        "results": "Results",
        "conclusions": "Conclusions",
        "objectives": "Objectives",
    },
    "scholarsum": {
        "background": "Background",
        "method": "Methods",
        "result": "Results",
        "conclusion": "Conclusions",
    },
}

def load_data_split_from_jsonl(
    file_path: str,
) -> List[dict]:
    """
    Load a dataset split from a JSONL file and retrurn as a list of dicts.

    Args:
        file_path (str): Path to the JSONL file.
    Returns:
        List[Dict]: A list of dictionaries representing the dataset.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                data.append(record)
    print(f"DATA_UTILS: Loaded {len(data)} records from {file_path}")
    return data

def format_aspect_flat_summary(
    aspect: str,
    aspect_text: str,
) -> str:
    """
    Format a single aspect summary into a flat string.

    Args:
        aspect (str): The aspect name.
        aspect_text (str): The summary text for the aspect.
    Returns:
        str: Formatted flat summary string.
    """
    return f"<{aspect.upper()}>" + "\n" + aspect_text.strip() + "\n"

def format_source_with_aspect_marker_instruction( # T5-like format
    source_text: str,
    aspect_name: str,
) -> str:
    """
    Format the source text by adding aspect marker instruction.

    Args:
        source_text (str): The original source text.
        aspect_name (str): The aspect name.
    Returns:
        str: Formatted source text with aspect instruction.
    """
    # return f"summarize the following text with focus on the aspect: {aspect_name}\n\n{source_text.strip()}"
    return f"<{aspect_name.upper()}>\n\n{source_text.strip()}"

def format_source_with_utterance_instruction( # T0-like format
    source_text: str,
    aspect_name: str,
) -> str:
    """
    Format the source text by adding utterance-style instruction.

    Args:
        source_text (str): The original source text.
        aspect_name (str): The aspect name.
    Returns:
        str: Formatted source text with utterance instruction.
    """
    templates = [
        f'{source_text.strip()}\n\n# Write a summary of the aspect {aspect_name} in the text above.',
        f'{source_text.strip()}\n\n# Given the above, write a summary of the aspect {aspect_name}.',
        f"# Article: \n\n{source_text.strip()} \n\n# Summarize the aspect: {aspect_name}.",
        f"# Read the article below \n\n\n {source_text.strip()} \n\n\n # Now summarize the aspect: {aspect_name}.",
        f"# Summarize the following text with focus on the aspect: {aspect_name}\n\n{source_text.strip()}",
        f"# Please provide a summary of the aspect {aspect_name} based on the text below.\n\n{source_text.strip()}",
        f"# Based on the text below, write a summary focusing on the aspect {aspect_name}.\n\n{source_text.strip()}",
        f"# Here is an article:\n\n{source_text.strip()}\n\n# Now, summarize the aspect: {aspect_name}.",
        f"# Below is an article:\n\n{source_text.strip()}\n\n# Please summarize the aspect: {aspect_name}.",
        f"# Article excerpt: \n\n{source_text.strip()} \n\n# Please summarize the aspect: {aspect_name}.",
        f"# Given the following article:\n\n{source_text.strip()}\n\n# Summarize the aspect: {aspect_name}.",
        f"# Please summarize the aspect {aspect_name} from the article below.\n\n{source_text.strip()}",
        f"# Summarize the aspect {aspect_name} based on the article below.\n\n{source_text.strip()}",
        f"# In the text below, identify and summarize the aspect: {aspect_name}.\n\n{source_text.strip()}",
        f"# Read the following article and summarize the aspect: {aspect_name}.\n\n{source_text.strip()}",
        f"# From the article below, provide a summary of the aspect: {aspect_name}.\n\n{source_text.strip()}",
        f"# Summarize the aspect {aspect_name} from the following text:\n\n{source_text.strip()}",
        f"# Based on the article below, provide a summary of the aspect: {aspect_name}.\n\n{source_text.strip()}",
        f"# Here is an article excerpt:\n\n{source_text.strip()}\n\n# Summarize the aspect: {aspect_name}.",
        f"# Please read the article below and summarize the aspect: {aspect_name}.\n\n{source_text.strip()}",
        f"# Summarize the aspect {aspect_name} from the article below.\n\n{source_text.strip()}",
        f"# Given the article below, summarize the aspect: {aspect_name}.\n\n{source_text.strip()}",
        f"# Article provided:\n\n{source_text.strip()}\n\n# Summarize the aspect: {aspect_name}.",
        f"# Below is an article excerpt:\n\n{source_text.strip()}\n\n# Please summarize the aspect: {aspect_name}.",
        f"# Read the article below and provide a summary of the aspect: {aspect_name}.\n\n{source_text.strip()}",
    ]
    return random.choice(templates)

def format_data_for_summarization_as_instruction(
    data: List[dict],
    dataset_name: str = "facetsum",
    prompt_format: str = "t5",
) -> List[dict]:
    """
    Format the dataset for summarization by creating a flat summary string in a T5-like format.

    Args:
        data (List[Dict]): A list of dictionaries with 'source_text' and 'summary' fields.
    Returns:
        List[Dict]: A list of dictionaries with formatted 'text' and 'summary' fields.
    """
    formatted_data = []
    if dataset_name in ["facetsum", "aclsum", "scholarsum"]:
        for record in data:
            source_text = record["source_text"]
            summary_dict = record["summary"]
            for aspect in summary_dict:
                aspect_text = summary_dict[aspect]
                if aspect_text:
                    if prompt_format == "t5":
                        source_text_with_instruction = format_source_with_aspect_marker_instruction(source_text, aspect)
                    elif prompt_format == "t0":
                        source_text_with_instruction = format_source_with_utterance_instruction(source_text, aspect)
                    else:
                        raise ValueError(f"Unknown prompt format: {prompt_format}")
                    formatted_data.append({
                        "source_text": source_text_with_instruction,
                        "aspect_summary": aspect_text.strip(),
                        "aspect_name": aspect,
                        "source_type": SOURCE_TYPE_SHORT[dataset_name],
                        "dataset": dataset_name,
                        "unique_id": f"{record['unique_id']}_{aspect}_short" if "unique_id" in record else None,
                    })
                else:
                    continue
    elif dataset_name == "pmc":
        error_count = 0
        for record in data:
            source_text_dict = record["source_text"]
            assert type(source_text_dict) is dict, "Expected source_text to be a dict. Every record should have multiple sections. The source_text field should be a dict with section titles as keys and section texts as values."
            summary_dict = record["summary"]
            try:
                # # Assert that the summary_dict and source_text_dict have the same keys
                # assert set(source_text_dict.keys()) == set(summary_dict.keys()), "Source text and summary must have the same sections."
                # Assert that all summary sections are in the source text sections
                for aspect in summary_dict:
                    assert aspect in source_text_dict, f"Summary aspect '{aspect}' not found in source text."
            except AssertionError as e:
                print(f"Skipping record due to key mismatch: {e}")
                print(f"Source text keys: {list(source_text_dict.keys())}")
                print(f"Summary keys: {list(summary_dict.keys())}")
                error_count += 1
                if error_count > 1000:
                    raise ValueError("Too many records with key mismatches between source_text and summary.")
                continue
            for aspect in summary_dict:
                aspect_text = summary_dict[aspect]
                if aspect_text:
                    if prompt_format == "t5":
                        source_text_with_instruction = format_source_with_aspect_marker_instruction(source_text_dict[aspect], aspect)
                    elif prompt_format == "t0":
                        source_text_with_instruction = format_source_with_utterance_instruction(source_text_dict[aspect], aspect)
                    else:
                        raise ValueError(f"Unknown prompt format: {prompt_format}")
                    formatted_data.append({
                        "source_text": source_text_with_instruction,
                        "aspect_summary": aspect_text.strip(),
                        "aspect_name": aspect,
                        "source_type": SOURCE_TYPE_SHORT[dataset_name],
                        "dataset": "pmc",
                        "unique_id": f"{record['unique_id']}_{aspect}_short" if "unique_id" in record else None,
                    })
                else:
                    raise ValueError(f"Missing summary for aspect '{aspect}' in record.")
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return formatted_data

def format_data_for_summarization(
    data: List[dict],
    dataset_name: str = "facetsum",
) -> List[dict]:
    """
    Format the dataset for summarization by creating a flat summary string.

    Args:
        data (List[Dict]): A list of dictionaries with 'source_text' and 'summary' fields.
    Returns:
        List[Dict]: A list of dictionaries with formatted 'text' and 'summary' fields.
    """
    formatted_data = []
    if dataset_name in ["facetsum", "aclsum", "scholarsum"]:
        for record in data:
            source_text = record["source_text"]
            summary_dict = record["summary"]
            for aspect in summary_dict:
                aspect_text = summary_dict[aspect]
                if aspect_text:
                    flat_summary = format_aspect_flat_summary(aspect, aspect_text)
                else:
                    # raise ValueError(f"Missing summary for aspect '{aspect}' in record.")
                    continue
                formatted_data.append({
                    "source_text": source_text,
                    "aspect_summary": flat_summary.strip(),
                    "aspect_name": aspect,
                    "source_type": SOURCE_TYPE_SHORT[dataset_name],
                    "dataset": dataset_name,
                    "unique_id": f"{record['unique_id']}_{aspect}_short" if "unique_id" in record else None,
                })
    elif dataset_name == "pmc":
        error_count = 0
        for record in data:
            source_text_dict = record["source_text"]
            assert type(source_text_dict) is dict, "Expected source_text to be a dict. Every record should have multiple sections. The source_text field should be a dict with section titles as keys and section texts as values."
            summary_dict = record["summary"]
            try:
                # # Assert that the summary_dict and source_text_dict have the same keys
                # assert set(source_text_dict.keys()) == set(summary_dict.keys()), "Source text and summary must have the same sections."
                # Assert that all summary sections are in the source text sections
                for aspect in summary_dict:
                    assert aspect in source_text_dict, f"Summary aspect '{aspect}' not found in source text."
            except AssertionError as e:
                print(f"Skipping record due to key mismatch: {e}")
                print(f"Source text keys: {list(source_text_dict.keys())}")
                print(f"Summary keys: {list(summary_dict.keys())}")
                error_count += 1
                if error_count > 1000:
                    raise ValueError("Too many records with key mismatches between source_text and summary.")
                continue
            for aspect in summary_dict:
                aspect_text = summary_dict[aspect]
                if aspect_text:
                    flat_summary = format_aspect_flat_summary(aspect, aspect_text)
                else:
                    raise ValueError(f"Missing summary for aspect '{aspect}' in record.")
                formatted_data.append({
                    "source_text": source_text_dict[aspect],
                    "aspect_summary": flat_summary.strip(),
                    "aspect_name": aspect,
                    "source_type": SOURCE_TYPE_SHORT[dataset_name],
                    "dataset": "pmc",
                    "unique_id": f"{record['unique_id']}_{aspect}_short" if "unique_id" in record else None,
                })
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return formatted_data

def format_data_for_summarization_raw(
    data: List[dict],
    dataset_name: str = "facetsum",
) -> List[dict]:
    """
    Format the dataset for summarization by creating documents with raw source and summary.

    Args:
        data (List[Dict]): A list of dictionaries with 'source_text' and 'summary' fields.
    Returns:
        List[Dict]: A list of dictionaries with formatted 'text' and 'summary' fields.
    """
    formatted_data = []
    if dataset_name in ["facetsum", "aclsum", "scholarsum"]:
        for record in data:
            source_text = record["source_text"]
            summary_dict = record["summary"]
            for aspect in summary_dict:
                aspect_text = summary_dict[aspect]
                if aspect_text:
                    formatted_data.append({
                        "source_text": source_text,
                        "aspect_summary": aspect_text.strip(),
                        "aspect_name": aspect,
                        "source_type": SOURCE_TYPE_SHORT[dataset_name],
                        "dataset": dataset_name,
                        "unique_id": f"{record['unique_id']}_{aspect}_short" if "unique_id" in record else None,
                    })
                else:
                    continue
    elif dataset_name == "pmc":
        error_count = 0
        for record in data:
            source_text_dict = record["source_text"]
            assert type(source_text_dict) is dict, "Expected source_text to be a dict. Every record should have multiple sections. The source_text field should be a dict with section titles as keys and section texts as values."
            summary_dict = record["summary"]
            try:
                # # Assert that the summary_dict and source_text_dict have the same keys
                # assert set(source_text_dict.keys()) == set(summary_dict.keys()), "Source text and summary must have the same sections."
                # Assert that all summary sections are in the source text sections
                for aspect in summary_dict:
                    assert aspect in source_text_dict, f"Summary aspect '{aspect}' not found in source text."
            except AssertionError as e:
                print(f"Skipping record due to key mismatch: {e}")
                print(f"Source text keys: {list(source_text_dict.keys())}")
                print(f"Summary keys: {list(summary_dict.keys())}")
                error_count += 1
                if error_count > 1000:
                    raise ValueError("Too many records with key mismatches between source_text and summary.")
                continue
            for aspect in summary_dict:
                aspect_text = summary_dict[aspect]
                if aspect_text:
                    formatted_data.append({
                        "source_text": source_text_dict[aspect],
                        "aspect_summary": aspect_text.strip(),
                        "aspect_name": aspect,
                        "source_type": SOURCE_TYPE_SHORT[dataset_name],
                        "dataset": "pmc",
                        "unique_id": f"{record['unique_id']}_{aspect}_short" if "unique_id" in record else None,
                    })
                else:
                    raise ValueError(f"Missing summary for aspect '{aspect}' in record.")
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return formatted_data

def format_data_for_summarization_long_context(
    data: List[dict],
    dataset_name: str = "facetsum",
    prompt_format: str = "facetsum",
) -> List[dict]:
    """
    Format the dataset for summarization with long context (full text).

    Args:
        data (List[Dict]): A list of dictionaries with 'source_text' and 'summary' fields.
    Returns:
        List[Dict]: A list of dictionaries with formatted 'text' and 'summary' fields.
    """
    assert dataset_name in ["aclsum", "facetsum", "pmc"], "Long context formatting is only supported for 'aclsum', 'facetsum', and 'pmc' datasets."

    def _format_prompts_for_long_context(
        source_text: str,
        aspect_summary: str,
        aspect_name: str,
        prompt_format: str = "facetsum",
    ) -> str:
        if prompt_format == "facetsum":
            return source_text, format_aspect_flat_summary(aspect_name, aspect_summary)
        elif prompt_format == "t5":
            return format_source_with_aspect_marker_instruction(source_text, aspect_name), aspect_summary.strip()
        elif prompt_format == "t0":
            return format_source_with_utterance_instruction(source_text, aspect_name), aspect_summary.strip()
        elif prompt_format == "none":
            return source_text, aspect_summary.strip()
        else:
            raise ValueError(f"Unknown prompt format: {prompt_format}")

    formatted_data = []
    for record in data:
        source_text = record["source_text"]
        if prompt_format != "none": # If not raw, flatten source_text dict to string for adding instructions
            # If source_text is a dict (as in pmc), convert to string by json dumping
            if isinstance(source_text, dict):
                source_text = json.dumps(source_text)
        summary_dict = record["summary"]
        for aspect in summary_dict:
            aspect_text = summary_dict[aspect]
            if aspect_text:
                formatted_source_text, formatted_aspect_summary = _format_prompts_for_long_context(
                    source_text,
                    aspect_text,
                    aspect,
                    prompt_format,
                )
                formatted_data.append({
                    "source_text": formatted_source_text,
                    "aspect_summary": formatted_aspect_summary,
                    "aspect_name": aspect,
                    "source_type": SOURCE_TYPE_LONG[dataset_name],
                    "dataset": dataset_name,
                    "unique_id": f"{record['unique_id']}_{aspect}_long" if "unique_id" in record else None,
                })
            else:
                continue
    return formatted_data

def normalize_formatted_data_aspect_names(
    data: List[dict],
    dataset_name: str = "facetsum",
) -> List[dict]:
    """
    Normalize aspect names in the formatted dataset to a standard set of aspect names.

    Args:
        data (List[Dict]): A list of dictionaries with 'aspect_name' fields.
        dataset_name (str): The name of the dataset, used to determine the normalization mapping.
    Returns:
        List[Dict]: A list of dictionaries with normalized 'aspect_name' fields.
    """
    if dataset_name not in ASPECT_NORMALIZATION:
        raise ValueError(f"No aspect normalization mapping found for dataset: {dataset_name}")
    
    normalization_mapping = ASPECT_NORMALIZATION[dataset_name]
    normalized_data = []
    for record in data:
        aspect_name = record["aspect_name"]
        if aspect_name not in normalization_mapping:
            continue  # Skip records with aspect names that are not in the normalization mapping
        normalized_aspect_name = normalization_mapping.get(aspect_name, aspect_name)  # Default to original if not found in mapping
        normalized_record = record.copy()
        normalized_record["aspect_name"] = normalized_aspect_name
        normalized_data.append(normalized_record)
    
    return normalized_data

def load_summarization_dataset(
    split: str = "train",
    dataset_name: str = "facetsum",
    type: str = "full", # or "sampled"
    prompt_format: str = "facetsum", # or "t5" or "none" or "t0"
    context_size_type: str = "short", # or "long"
) -> List[dict]:
    """
    Load and format a dataset split from a JSONL file with a disk cache.

    Args:
        split (str): Name of the split (e.g., 'train', 'validation', 'test').
        dataset_name (str): Name of the dataset (e.g., 'facetsum', 'aclsum', 'scholarsum', 'pmc').
        prompt_format (str): Format of the prompts ('facetsum', 't5', 't0', 'none').
        context_size_type (str): Type of context size ('short' or 'long').
    Returns:
        List[Dict]: A list of formatted dictionaries representing the dataset. One element of the list corresponds to the tuple {'text': ..., 'summary': ..., 'aspect': ..., 'dataset': ...}.
    """
    if type not in ["full", "sampled"]:
        raise ValueError("Type must be either 'full' or 'sampled'.")
    if type == "sampled" and split != "test":
        raise ValueError("Sampled type is only supported for the 'test' split.")
    if type == "sampled" and dataset_name not in ["facetsum", "pmc"]:
        raise ValueError("Sampled type is only supported for 'facetsum' and 'pmc' (test) datasets.")

    if context_size_type == "short" and dataset_name not in ["facetsum", "aclsum", "pmc"]:
        raise ValueError(f"Short context size type is only supported for 'facetsum', 'aclsum', and 'pmc' datasets.")
    if context_size_type == "long" and dataset_name not in ["aclsum", "pmc", "facetsum"]:
        raise ValueError(f"Long context size type is only supported for 'aclsum', 'pmc' and 'facetsum' datasets.")
    
    print("----" * 10)
    print('DATA_UTILS: sanity check passed')
    print(f'DATA_UTILS: Loading {type} {dataset_name} dataset with {context_size_type} context...')
    
    if context_size_type == "short":
        dataset_paths = {
            "facetsum": DEFAULT_FACETSUM_PATH / f"{split}.jsonl" if type == "full" else DEFAULT_FACETSUM_PATH / f"{split}_sampled_with_ids.jsonl",
            "aclsum": DEFAULT_ACLSUM_SHORT_PATH / f"{split}_with_ids.jsonl" if split == "test" else DEFAULT_ACLSUM_SHORT_PATH / f"{split}.jsonl", # aclsum does not have sampled split
            "pmc": DEFAULT_PMC_PATH / f"{split}.jsonl" if type == "full" else DEFAULT_PMC_PATH / f"{split}_sampled_with_ids.jsonl",
        }
    else:  # long context size
        dataset_paths = {
            "facetsum": DEFAULT_FACETSUM_PATH_LONG / f"{split}.jsonl" if type == "full" else DEFAULT_FACETSUM_PATH_LONG / f"{split}_fixed_sampled_with_ids.jsonl",
            "aclsum": DEFAULT_ACLSUM_LONG_PATH / f"{split}_with_ids.jsonl" if split == "test" else DEFAULT_ACLSUM_LONG_PATH / f"{split}.jsonl", # aclsum does not have sampled split
            "pmc": DEFAULT_PMC_PATH / f"{split}.jsonl" if type == "full" else DEFAULT_PMC_PATH / f"{split}_sampled_with_ids.jsonl",
            #"scholarsum": DEFAULT_SCHOLARSUM_PATH / f"{split}.jsonl",
        }
    dataset_path = dataset_paths.get(dataset_name)
    if dataset_path is None:
        raise ValueError(f"Unknown dataset name: {dataset_name}")   
    
    print(f"DATA_UTILS: Loading dataset from: {dataset_path}")
    if dataset_name == "pmc" and context_size_type == "long":
        print("DATA_UTILS: Note: PMC long context and PMC short context are created from the same file.")

    # Build dataset and write cache
    raw_data = load_data_split_from_jsonl(str(dataset_path))

    if context_size_type == "long":
        formatted_data = format_data_for_summarization_long_context(raw_data, dataset_name, prompt_format)
    else:  # short context size
        if prompt_format == "facetsum":
            formatted_data = format_data_for_summarization(raw_data, dataset_name)
        elif prompt_format == "t5" or prompt_format == "t0":
            formatted_data = format_data_for_summarization_as_instruction(raw_data, dataset_name, prompt_format)
        elif prompt_format == "none":
            formatted_data = format_data_for_summarization_raw(raw_data, dataset_name)
        else:
            raise ValueError(f"Unknown format: {prompt_format}")

    print(f"DATA_UTILS: Loaded {len(formatted_data)} formatted records from {dataset_path}")

    if dataset_name in ASPECT_NORMALIZATION:
        print("DATA_UTILS: Normalizing aspect names in the formatted dataset...")
        formatted_data = normalize_formatted_data_aspect_names(formatted_data, dataset_name)
        print(f"DATA_UTILS: Loaded {len(formatted_data)} normalized formatted records from {dataset_path}")

    print("----" * 10)

    return formatted_data

def load_document_subsubsample(
    dataset_name: str,
    sample_size: int = 100,
    seed: int = 42,
) -> Dict[str, List[dict]]:
    """
    Load the document subsubsample dataset for a given dataset name.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'facetsum', 'aclsum', 'pmc').
        sample_size (int): Number of documents to sample.
        seed (int): Seed used for deterministic document sampling.
    Returns:
        Dict[str, List[Dict]]: A dictionary where keys are unique document IDs and values are lists of records corresponding to that document ID.
    """

    def group_record_by_doc_id(records: List[dict]) -> Dict[str, List[dict]]:
        """
        Group records by their unique document ID.

        Args:
            records (List[Dict]): A list of dictionaries representing the dataset.
        Returns:
            Dict[str, List[Dict]]: A dictionary where keys are unique document IDs and values are lists of records corresponding to that document ID.
        """
        grouped_records = dict()
        for record in records:
            unique_doc_id = find_longest_digit_sequence(record.get("unique_id", None))
            if unique_doc_id is not None:
                if unique_doc_id not in grouped_records:
                    grouped_records[unique_doc_id] = []
                grouped_records[unique_doc_id].append(record)
            else:
                raise ValueError("Record does not contain 'unique_id' field.")
        return grouped_records

    if dataset_name == "facetsum":
        formatted_data = load_summarization_dataset(split="test", dataset_name=dataset_name, prompt_format="none", context_size_type="long", type="sampled")
    elif dataset_name == "pmc":
        formatted_data = load_summarization_dataset(split="test", dataset_name=dataset_name, prompt_format="none", context_size_type="long", type="sampled")
    elif dataset_name == "aclsum":
        formatted_data = load_summarization_dataset(split="test", dataset_name=dataset_name, prompt_format="none", context_size_type="long", type="full")
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    document_records_sets = group_record_by_doc_id(formatted_data)

    if sample_size is not None:
        if sample_size < 0:
            raise ValueError("sample_size must be non-negative.")
        if sample_size > len(document_records_sets):
            raise ValueError(
                f"sample_size ({sample_size}) cannot exceed the number of documents "
                f"({len(document_records_sets)})."
            )
        sampled_doc_ids = random.Random(seed).sample(
            list(document_records_sets), sample_size
        )
        document_records_sets = {
            doc_id: document_records_sets[doc_id] for doc_id in sampled_doc_ids
        }

    # Sanity checks
    ## Ensure that no two records in the same document have the same aspect_name and aspect_summary
    for doc_id, records in document_records_sets.items():
        seen_aspects = set()
        for record in records:
            aspect_tuple = (record["aspect_name"], record["aspect_summary"])
            if aspect_tuple in seen_aspects:
                raise ValueError(f"Duplicate aspect_name and aspect_summary found in document {doc_id}.")
            seen_aspects.add(aspect_tuple)
    ## Ensure that all records in the same document have the same dataset name
    for doc_id, records in document_records_sets.items():
        dataset_names = {record["dataset"] for record in records}
        if len(dataset_names) > 1:
            raise ValueError(f"Records in document {doc_id} have different dataset names: {dataset_names}.")
    ## Ensure that all records in the same document have the same source_type
    for doc_id, records in document_records_sets.items():
        source_types = {record["source_type"] for record in records}
        if len(source_types) > 1:
            raise ValueError(f"Records in document {doc_id} have different source types: {source_types}.")
    ## Esure that all records in the same document have different aspect_name and aspect_summary
    for doc_id, records in document_records_sets.items():
        aspect_tuples = [(record["aspect_name"], record["aspect_summary"]) for record in records]
        if len(aspect_tuples) != len(set(aspect_tuples)):
            raise ValueError(f"Records in document {doc_id} have duplicate aspect_name and aspect_summary pairs.")
    ## Ensure that there are no more than 5 records in the same document (since there are only 5 aspects in the datasets)
    for doc_id, records in document_records_sets.items():
        if len(records) > 5:
            raise ValueError(f"Document {doc_id} has more than 5 records: {len(records)}.")

    return document_records_sets

def load_rebuttal_subsubsample(
    dataset_name: str
    ) -> Dict[str, List[dict]]:
    """
    Load the rebuttal subsubsample dataset for a given dataset name.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'facetsum', 'aclsum', 'pmc').
    Returns:
        Dict[str, List[Dict]]: A dictionary where keys are unique document IDs and values are lists of records corresponding to that document ID.
    """
    return load_document_subsubsample(dataset_name=dataset_name, sample_size=100, seed=42)

def find_longest_digit_sequence(s: str) -> str:
    """
    Find the longest sequence of digits in a string.

    Args:
        s (str): Input string.
    Returns:
        str: The longest sequence of digits found in the string. If no digits are found, returns an empty string.
    """
    import re
    digit_sequences = re.findall(r'\d+', s)
    if not digit_sequences:
        return ""
    return max(digit_sequences, key=len)

if __name__ == "__main__":
    # Example usage
    # data = load_summarization_dataset(split="train", dataset_name="pmc", prompt_format="none", context_size_type="long")
    # print(f"Loaded {len(data)} records.")
    # # data = load_summarization_dataset(split="train", dataset_name="pmc", prompt_format="none")
    # # print(f"Loaded {len(data)} records from facetsum train split.")
    # # print("Sample record:")
    # print(json.dumps(data[0], indent=2))

    # for dataset_name in ['pmc', 'facetsum']:
    #     for context_size_type in ['short', 'long']:
    #         data = load_summarization_dataset(split="train", dataset_name=dataset_name, prompt_format="none", context_size_type=context_size_type, type="full")
    #         pass
    
    # for dataset_name in ['pmc', 'facetsum']:
    #     for context_size_type in ['short', 'long']:
    #         data = load_summarization_dataset(split="val", dataset_name=dataset_name, prompt_format="none", context_size_type=context_size_type, type="full")
    #         pass

    # for dataset_name in ['pmc', 'facetsum']:
    #     for context_size_type in ['short', 'long']:
    #         data = load_summarization_dataset(split="test", dataset_name=dataset_name, prompt_format="none", context_size_type=context_size_type, type="full")
    #         pass

    # for dataset_name in ['pmc', 'facetsum']:
    #     for context_size_type in ['short', 'long']:
    #         data = load_summarization_dataset(split="test", dataset_name=dataset_name, prompt_format="none", context_size_type=context_size_type, type="sampled")
    #         pass

    # assert len(facetsum_long_sampled) == len(facetsum_short_sampled), "Sampled long and short datasets should have the same number of records."
    # for long_record, short_record in zip(facetsum_long_sampled, facetsum_short_sampled):
    #     assert long_record["aspect_name"] == short_record["aspect_name"], "Aspect names should match between long and short sampled datasets."
    #     assert long_record["aspect_summary"] == short_record["aspect_summary"], "Aspect summaries should match between long and short sampled datasets."
    #     assert long_record["dataset"] == short_record["dataset"], "Dataset names should match between long and short sampled datasets."
    #     # Note: source_text and aspect_summary will be different between long and short datasets, so we do not assert their equality.

    # test loading aclsum long dataset
    aclsum_long = load_summarization_dataset(split="test", dataset_name="aclsum", prompt_format="none", context_size_type="long", type="full")
    print(f"Loaded {len(aclsum_long)} records from ACLSum long dataset.")
    # Show keys of first record of aclsum long dataset
    print("Keys of first record from ACLSum long dataset:")
    print(aclsum_long[0].keys())
    # Show first record of aclsum long dataset in pretty json format
    # print("Sample record from ACLSum long dataset:")
    # print(json.dumps(aclsum_long[0], indent=2))
    # Reconstruct unique document IDs from ACLSum long dataset by extracting the longest digit sequence from the unique_id field
    doc_ids = set()
    for record in aclsum_long:
        longest_digit_sequence = find_longest_digit_sequence(record.get("unique_id", ""))
        doc_ids.add(longest_digit_sequence)
    print(f"Number of unique document IDs in ACLSum long dataset: {len(doc_ids)}")

    # test loading pmc long dataset
    pmc_long = load_summarization_dataset(split="test", dataset_name="pmc", prompt_format="none", context_size_type="long", type="sampled")
    print(f"Loaded {len(pmc_long)} records from PMC long dataset.")
    # Show keys of first record of pmc long dataset
    print("Keys of first record from PMC long dataset:")
    print(pmc_long[0].keys())
    # # Show first record of pmc long dataset in pretty json format
    # print("Sample record from PMC long dataset:")
    # print(json.dumps(pmc_long[0], indent=2))
    # Reconstruct unique document IDs from PMC long dataset by extracting the longest digit sequence from the unique_id field
    doc_ids = set()
    for record in pmc_long:
        longest_digit_sequence = find_longest_digit_sequence(record.get("unique_id", ""))
        doc_ids.add(longest_digit_sequence)
    print(f"Number of unique document IDs in PMC long dataset: {len(doc_ids)}")

    # test loading facetsum long dataset
    facetsum_long = load_summarization_dataset(split="test", dataset_name="facetsum", prompt_format="none", context_size_type="long", type="sampled")
    print(f"Loaded {len(facetsum_long)} records from FacetSum long dataset.")
    # Show keys of first record of facetsum long dataset
    print("Keys of first record from FacetSum long dataset:")
    print(facetsum_long[0].keys())
    # # Show first record of facetsum long dataset in pretty json format
    # print("Sample record from FacetSum long dataset:")
    # print(json.dumps(facetsum_long[0], indent=2))
    # Reconstruct unique document IDs from FacetSum long dataset by extracting the longest digit sequence from the unique_id field
    doc_ids = set()
    for record in facetsum_long:
        longest_digit_sequence = find_longest_digit_sequence(record.get("unique_id", ""))
        doc_ids.add(longest_digit_sequence)
    print(f"Number of unique document IDs in FacetSum long dataset: {len(doc_ids)}")

    # Load rebuttal subsubsample for facetsum
    facetsum_rebuttal_subsubsample_1 = load_rebuttal_subsubsample("facetsum")
    facetsum_rebuttal_subsubsample_2 = load_rebuttal_subsubsample("facetsum")
    # Check that the two samples are identical
    assert facetsum_rebuttal_subsubsample_1.keys() == facetsum_rebuttal_subsubsample_2.keys(), "Two samples of the rebuttal subsubsample for FacetSum should have the same document IDs."
    for doc_id in facetsum_rebuttal_subsubsample_1.keys():
        records_1 = facetsum_rebuttal_subsubsample_1[doc_id]
        records_2 = facetsum_rebuttal_subsubsample_2[doc_id]
        assert len(records_1) == len(records_2), f"Two samples of the rebuttal subsubsample for FacetSum should have the same number of records for document ID {doc_id}."
        for r1, r2 in zip(records_1, records_2):
            assert r1 == r2, f"Two samples of the rebuttal subsubsample for FacetSum should have identical records for document ID {doc_id}."
    print(f"Loaded {len(facetsum_rebuttal_subsubsample_1)} unique document IDs from FacetSum rebuttal subsubsample.")
    # Load rebuttal subsubsample for pmc
    pmc_rebuttal_subsubsample_1 = load_rebuttal_subsubsample("pmc")
    pmc_rebuttal_subsubsample_2 = load_rebuttal_subsubsample("pmc")
    # Check that the two samples are identical
    assert pmc_rebuttal_subsubsample_1.keys() == pmc_rebuttal_subsubsample_2.keys(), "Two samples of the rebuttal subsubsample for PMC should have the same document IDs."
    for doc_id in pmc_rebuttal_subsubsample_1.keys():
        records_1 = pmc_rebuttal_subsubsample_1[doc_id]
        records_2 = pmc_rebuttal_subsubsample_2[doc_id]
        assert len(records_1) == len(records_2), f"Two samples of the rebuttal subsubsample for PMC should have the same number of records for document ID {doc_id}."
        for r1, r2 in zip(records_1, records_2):
            assert r1 == r2, f"Two samples of the rebuttal subsubsample for PMC should have identical records for document ID {doc_id}."
    print(f"Loaded {len(pmc_rebuttal_subsubsample_1)} unique document IDs from PMC rebuttal subsubsample.")
    # Load rebuttal subsubsample for aclsum
    aclsum_rebuttal_subsubsample_1 = load_rebuttal_subsubsample("aclsum")
    aclsum_rebuttal_subsubsample_2 = load_rebuttal_subsubsample("aclsum")
    # Check that the two samples are identical
    assert aclsum_rebuttal_subsubsample_1.keys() == aclsum_rebuttal_subsubsample_2.keys(), "Two samples of the rebuttal subsubsample for ACLSum should have the same document IDs."
    for doc_id in aclsum_rebuttal_subsubsample_1.keys():
        records_1 = aclsum_rebuttal_subsubsample_1[doc_id]
        records_2 = aclsum_rebuttal_subsubsample_2[doc_id]
        assert len(records_1) == len(records_2), f"Two samples of the rebuttal subsubsample for ACLSum should have the same number of records for document ID {doc_id}."
        for r1, r2 in zip(records_1, records_2):
            assert r1 == r2, f"Two samples of the rebuttal subsubsample for ACLSum should have identical records for document ID {doc_id}."
    print(f"Loaded {len(aclsum_rebuttal_subsubsample_1)} unique document IDs from ACLSum rebuttal subsubsample.")