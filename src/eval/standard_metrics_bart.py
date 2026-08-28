import json
import os
import torch
import logging
from typing import List, Dict, Any
from rouge_score import rouge_scorer
from bert_score import score as bert_score_func
from tqdm import tqdm 
import nltk

# --- CONFIGURATION ---
ROOT_PATH = "data"
MODEL_NAME = "bart"

# --- NLTK SETUP ---
# Required for sentence splitting to distinguish rougeLsum from rougeL
def ensure_nltk_resources():
    resources = ['punkt', 'punkt_tab']
    for res in resources:
        try:
            # Try to find the resource
            if res == 'punkt':
                nltk.data.find('tokenizers/punkt')
            else:
                nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            # If not found, download it
            logging.info(f"Downloading NLTK resource: {res}...")
            nltk.download(res)

# Call the function to ensure resources are available
ensure_nltk_resources()

# --- HARDWARE SETUP ---
if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

# --- MODEL SETTINGS ---
# roberta-large is standard for BERTScore (correlation with human judgment).
BERTSCORE_MODEL = "roberta-large" 
BATCH_SIZE = 32 

# --- DATA FIXING ---
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

def configure_logger(log_file_path: str):
    """Configure root logger to write to the provided path."""
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logger = logging.getLogger()
    logger.handlers.clear()
    handler = logging.FileHandler(log_file_path, encoding="utf-8", mode="w")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def load_data(path: str) -> List[Dict[str, Any]]:
    """Load JSONL data (one JSON object per line) from a file."""
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_data(data: List[Dict[str, Any]], path: str):
    """Save data as JSONL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def add_newlines(text: str) -> str:
    """
    Tokenizes text into sentences and joins them with newlines (\n).
    
    Why?
    rouge_scorer calculates 'rougeLsum' by splitting on newlines. 
    If we don't do this, rougeLsum treats the whole text as one sentence, 
    making it identical to rougeL.
    """
    if not text:
        return ""
    # sent_tokenize handles punctuation like "Dr. Smith" correctly.
    sentences = nltk.sent_tokenize(text)
    return "\n".join(sentences)

def strip_aspect_prefix(texts: List[str]) -> List[str]:
    """
    Remove leading aspect prefixes enclosed in angle brackets from a list of strings.

    This function iterates over each input string and, if the string begins with `<` and
    contains a closing `>`, it strips off the first occurrence of an angle-bracketed
    prefix and any surrounding whitespace. Strings without such a prefix are returned
    unchanged.

    Args:
        texts (List[str]): A list of strings that may start with an angle-bracketed prefix.

    Returns:
        List[str]: A new list of strings with any leading `<...>` prefixes removed.
    """
    stripped = []
    for text in texts:
        prefix_window = text[:50]
        closing_idx = prefix_window.rfind(">")
        if text.startswith("<") and closing_idx != -1:
            stripped.append(text[closing_idx + 1 :].strip())
        else:
            stripped.append(text)
    return stripped

def strip_aspect_prefix_with_aspect_name(texts: List[str], aspect_names: List[str]) -> List[str]:
    """
    Remove a specific uppercase `<ASPECT>` prefix from each string if present.

    For each (text, aspect) pair, this function constructs the literal
    `f"<{aspect.upper()}>"` token and, if it appears anywhere in the text,
    splits on its first occurrence and returns the remainder with surrounding
    whitespace stripped. If the token is not present, the original text is returned.

    Args:
        texts (List[str]): A list of strings that may contain `<ASPECT>` tokens.
        aspect_names (List[str]): A list of aspect names corresponding to each text.

    Returns:
        List[str]: A new list of strings with the first `<ASPECT>` token removed when present.
    """
    stripped = []
    for text, aspect in zip(texts, aspect_names):
        aspect_prefix = f"<{aspect.upper()}>"
        stripped_text = text.split(aspect_prefix, 1)[-1].strip() if aspect_prefix in text else text
        stripped.append(stripped_text)
    return stripped

def calculate_deterministic_metrics(
    input_file_path: str,
    output_file_path: str,
    bertscore_model: str,
    device: str,
    batch_size: int,
):
    """Calculate ROUGE and BERTScore metrics."""
    logging.info(f"Loading data from {input_file_path}...")
    print(f"Loading data from {input_file_path}...")
    data = load_data(input_file_path)
    
    # 1. Prepare Data
    generated_list = [d["generated_aspect_summary"].strip() for d in data]
    reference_list = [d["gold_aspect_summary"].strip() for d in data]

    # Normalize aspect names based on dataset
    aspect_filtered_data = []
    for _d in data:
        _dataset = _d.get("dataset")
        _aspect = _d.get("aspect_name")
        if _dataset in ASPECT_NORMALIZATION and _aspect.upper() in [_k.upper() for _k in ASPECT_NORMALIZATION[_dataset].keys()]:
            aspect_filtered_data.append(_d)
    data = aspect_filtered_data
    logging.info(f"Filtered data to {len(data)} entries based on aspect normalization.")

    # Optionally strip aspect prefixes
    generated_list = strip_aspect_prefix_with_aspect_name(generated_list, [d["aspect_name"] for d in data])
    reference_list = strip_aspect_prefix_with_aspect_name(reference_list, [d["aspect_name"] for d in data])

    # Show how many empty summaries we have after stripping (for debugging)
    num_empty_generated = sum(1 for gen in generated_list if not gen)
    num_empty_reference = sum(1 for ref in reference_list if not ref)
    logging.info(f"Number of empty generated summaries after stripping: {num_empty_generated}")
    logging.info(f"Number of empty reference summaries after stripping: {num_empty_reference}")
    for _gen_as, _gold_as, _striped_gen, _strip_ref in zip(
        [d["generated_aspect_summary"] for d in data],
        [d["gold_aspect_summary"] for d in data],
        generated_list,
        reference_list,
    ):
        if not _striped_gen or not _strip_ref:
            logging.info(f"Original Generated: {_gen_as}")
            logging.info(f"Original Reference: {_gold_as}")
            logging.info(f"Stripped Generated: {_striped_gen}")
            logging.info(f"Stripped Reference: {_strip_ref}")
    
    # Remove entries where either generated or reference is empty after stripping, as they can cause issues with metrics.
    filtered_data = []
    filtered_generated = []
    filtered_reference = []
    for item, gen, ref in zip(data, generated_list, reference_list):
        if gen and ref:
            filtered_data.append(item)
            filtered_generated.append(gen)
            filtered_reference.append(ref)
        else:
            logging.info(f"Skipping entry due to empty generated or reference summary after stripping. Generated: '{gen}', Reference: '{ref}'")
    generated_list = filtered_generated
    reference_list = filtered_reference
    data = filtered_data
    logging.info(f"Number of entries after filtering empty summaries: {len(generated_list)} out of {len(data)}")

    # 2. Compute BERTScore (Batch Mode)
    # Batch processing is crucial for speed. BERTScore handles its own tokenization.
    logging.info(f"Computing BERTScores on {device} using {bertscore_model}...")
    
    _, _, F1 = bert_score_func(
        generated_list, 
        reference_list, 
        model_type=bertscore_model, 
        lang="en", 
        verbose=True, 
        device=device,
        batch_size=batch_size
    )
    bert_f1_scores = F1.tolist()

    # 3. Compute ROUGE & Assemble Output
    logging.info("Computing ROUGE scores and assembling output...")
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    
    processed_data = []

    for i, item in enumerate(tqdm(data, desc="ROUGE Processing")):
        gen_text = generated_list[i]
        ref_text = reference_list[i]
        
        # PRE-PROCESSING FOR ROUGE
        # We inject newlines solely for the scorer to recognize sentence boundaries.
        gen_formatted = add_newlines(gen_text)
        ref_formatted = add_newlines(ref_text)
        
        scores = scorer.score(ref_formatted, gen_formatted)
        
        new_entry = {
            "generated_aspect_summary": item.get("generated_aspect_summary"),
            "gold_aspect_summary": item.get("gold_aspect_summary"),
            "aspect_name": item.get("aspect_name"),
            "source_text": item.get("source_text"),
            "source_type": item.get("source_type"),
            "context_size": item.get("context_size"),
            "dataset": item.get("dataset"),
            
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            
            # RougeL: Longest Common Subsequence based on the raw text stream.
            # Does not respect sentence boundaries (ignores newlines).
            "rougeL": scores['rougeL'].fmeasure,
            
            # RougeLSum: Summary-level LCS.
            # 1. Splits text by newlines (\n).
            # 2. Computes LCS for each sentence pair.
            # 3. Aggregates the union of these scores.
            "rougeLsum": scores['rougeLsum'].fmeasure, 
            
            "bertscore": bert_f1_scores[i],
            
            "metadata": item.get("metadata", {})
        }
        
        processed_data.append(new_entry)

    logging.info(f"Saving results to {output_file_path}...")
    save_data(processed_data, output_file_path)

    if processed_data:
        metric_keys = ["rouge1", "rouge2", "rougeL", "rougeLsum", "bertscore"]
        aggregated_metrics = {
            key: float(sum(item[key] for item in processed_data) / len(processed_data))
            for key in metric_keys
        }
        aggregated_payload = {
            "num_examples": len(processed_data),
            "metrics": aggregated_metrics,
        }
        aggregated_file_path = os.path.splitext(output_file_path)[0] + "_aggregated.json"
        logging.info(f"Saving aggregate metrics to {aggregated_file_path}...")
        with open(aggregated_file_path, "w", encoding="utf-8") as f:
            json.dump(aggregated_payload, f, ensure_ascii=False, indent=2)
    else:
        logging.warning("No entries found; skipping aggregate metrics file.")

    logging.info("Done.")

if __name__ == "__main__":
    DATASET_TYPE = "sampled"  # or "full"

    if DATASET_TYPE == "full":
        for context_size in ["short", "long"]:
            for dataset_name in ["aclsum", "pmc", "facetsum"]:
                if dataset_name == "aclsum" and context_size == "long":
                    continue  # Skip invalid combination
                
                if context_size == "long" and dataset_name == "facetsum":
                    dataset_name = "facetsum_long"

                input_file_path = f"{ROOT_PATH}{dataset_name}/{MODEL_NAME}_{context_size}_context_results.jsonl"   
                output_file_path = f"{ROOT_PATH}{dataset_name}/{MODEL_NAME}_{context_size}_context_standard_metrics.jsonl"
                log_file_path = f"logs/eval/{dataset_name}_{MODEL_NAME}_{context_size}_standard_metrics.log"

                # --- LOGGING SETUP ---
                configure_logger(log_file_path)
                calculate_deterministic_metrics(
                    input_file_path=input_file_path,
                    output_file_path=output_file_path,
                    bertscore_model=BERTSCORE_MODEL,
                    device=DEVICE,
                    batch_size=BATCH_SIZE,
                )
    elif DATASET_TYPE == "sampled":
        for context_size in ["short", "long"]:
            for dataset_name in ["pmc", "facetsum"]:
                
                if context_size == "long" and dataset_name == "facetsum":
                    dataset_name = "facetsum_long"

                input_file_path = f"{ROOT_PATH}/{dataset_name}/{MODEL_NAME}/{MODEL_NAME}_{context_size}_context_sampled_results.jsonl"   
                output_file_path = f"{ROOT_PATH}/{dataset_name}/{MODEL_NAME}/{MODEL_NAME}_{context_size}_context_sampled_standard_metrics.jsonl"
                log_file_path = f"logs/eval/{dataset_name}_{MODEL_NAME}_{context_size}_sampled_standard_metrics.log"

                # --- LOGGING SETUP ---
                configure_logger(log_file_path)
                calculate_deterministic_metrics(
                    input_file_path=input_file_path,
                    output_file_path=output_file_path,
                    bertscore_model=BERTSCORE_MODEL,
                    device=DEVICE,
                    batch_size=BATCH_SIZE,
                )
    else:
        logging.error(f"Invalid DATASET_TYPE: {DATASET_TYPE}. Must be 'full' or 'sampled'.")