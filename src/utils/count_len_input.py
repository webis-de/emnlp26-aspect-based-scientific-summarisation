"""
Calculate token length statistics for specific dataset/context combinations.
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_utils import load_summarization_dataset
import data_utils
from paths import DATA_ROOT

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
MODEL_PATH = "Qwen/Qwen3-32B" 
BASE_STORAGE = DATA_ROOT
OUTPUT_FILE = BASE_STORAGE / "dataset_token_stats.json"
SPLIT = "test"

# Explicitly define which contexts to check for which dataset
# Format: (dataset_name, context_size)
RUN_CONFIG: List[Tuple[str, str]] = [
    ("facetsum", "short"),
    ("facetsum", "long"),
    ("pmc", "short"),
    ("pmc", "long"),
    ("aclsum", "short"),
    ("scholarsum", "long"),
]

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

SOURCE_TYPE_DESCRIPTIONS = {
    "introduction-conclusion": "the introduction and conclusion sections of a scientific paper",
    "aspect-section": "the specific section of a scientific paper relevant to the requested aspect",
    "abstract-introduction-conclusion": "the abstract, introduction, and conclusion sections of a scientific paper",
    "full-text": "the full text of a scientific paper",
    "full-text-structured": "the full text of a scientific paper, structured by section headers",
}

SYSTEM_PROMPT = "You are an expert scientific editor, you are given a scientific text and tasked to write a summary focused on a specific aspect."

PROMPT_TEMPLATE = """
Given the scientific text below and a focused aspect, which is {aspect}, create a short summary using your own words. 
The summary needs to be a coherent paragraph and should include the major points. 
Write in free form, avoid bullet points or numbered lists. 
The summary should focus on the provided aspect only, contain only information about the aspect, and avoid adding irrelevant sentences or your own opinions and suggestions.

The source text is a {source_type_pretty}. This is {source_type_desc}.
FOCUSED ASPECT: {aspect}
SUMMARY TARGET LENGTH: approximately {target_length} words
SOURCE TEXT:
{source_text}
Provide the summary below:
"""

def parse_source_text(raw_source) -> str:
    if isinstance(raw_source, dict):
        text_parts = []
        for section_title, section_content in raw_source.items():
            if section_content:
                clean_title = str(section_title).upper().strip()
                clean_content = str(section_content).strip()
                part = f"## {clean_title}\n{clean_content}"
                text_parts.append(part)
        return "\n\n".join(text_parts)
    return str(raw_source).strip()

def build_full_prompt(record: dict, parsed_text: str) -> List[Dict[str, str]]:
    reference_summary = record.get("aspect_summary", "").strip()
    summary_len = len(reference_summary.split())
    source_len = len(parsed_text.split())
    target_length = summary_len if summary_len > 0 else max(40, source_len // 10)
    
    st_key = record.get("source_type", "unknown")
    st_desc = SOURCE_TYPE_DESCRIPTIONS.get(st_key, "scientific text")
    
    user_content = PROMPT_TEMPLATE.format(
        aspect=record.get("aspect_name", "main"),
        source_type_pretty=st_key.replace("-", " "),
        source_type_desc=st_desc,
        source_text=parsed_text,
        target_length=target_length,
    )
    
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    if BASE_STORAGE.exists():
        data_utils.DEFAULT_DATA_ROOT = BASE_STORAGE
        data_utils.DEFAULT_FACETSUM_PATH = BASE_STORAGE / "facetsum"
        data_utils.DEFAULT_ACLSUM_PATH = BASE_STORAGE / "aclsum"
        data_utils.DEFAULT_SCHOLARSUM_PATH = BASE_STORAGE / "scholarsum" / "arxiv"
        data_utils.DEFAULT_PMC_PATH = BASE_STORAGE / "pmc" / "extracted" / "splits"

    print(f"Loading Tokenizer: {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    results = {}

    # Iterate over the explicit config list
    for dataset_name, context_size in RUN_CONFIG:
        run_key = f"{dataset_name}_{context_size}" # e.g. facetsum_short
        print(f"\nProcessing {run_key}...")
        
        try:
            dataset = load_summarization_dataset(
                split=SPLIT,
                dataset_name=dataset_name,
                prompt_format='none',
                context_size_type=context_size
            )
        except Exception as e:
            print(f"Skipping {run_key}: {e}")
            continue

        if not dataset:
            print(f"Dataset {run_key} empty.")
            continue
            
        token_counts = []
        
        for record in tqdm(dataset, desc=f"Tokenizing {run_key}"):
            source_text = parse_source_text(record.get("source_text"))
            messages = build_full_prompt(record, source_text)
            
            full_prompt_str = tokenizer.apply_chat_template(messages, tokenize=False)
            tokens = tokenizer(full_prompt_str, add_special_tokens=False)["input_ids"]
            
            token_counts.append(len(tokens))

        if not token_counts:
            continue
            
        # Calculate stats
        max_len = int(np.max(token_counts))
        # Add buffer for generation (e.g. 512 tokens)
        rec_len = max_len + 512
        
        stats = {
            "dataset": dataset_name,
            "context": context_size,
            "count": len(token_counts),
            "min": int(np.min(token_counts)),
            "max": max_len,
            "avg": int(np.mean(token_counts)),
            "p95": int(np.percentile(token_counts, 95)),
            "p99": int(np.percentile(token_counts, 99)),
            "recommended_max_model_len": rec_len
        }
        
        results[run_key] = stats
        print(f"  -> Max: {stats['max']} | P99: {stats['p99']} | Rec: {rec_len}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved stats to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()