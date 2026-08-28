import json
from typing import List, Dict, Any

# --- CONFIGURATION ---
ROOT_PATH = "data"
MODEL_NAME = "bart"

# --- GLOBAL VARIABLES ---
ASPECT_MAPPING = {
    "Purpose": "Purpose",
    "Design/methodology/approach": "Methods",
    "Findings": "Findings",
    "Originality/value": "Value"
}

# --- FUNCTIONS ---

def load_data(path: str) -> List[Dict[str, Any]]:
    """Load JSONL data (one JSON object per line) from a file."""
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def filter_results(full_results: List[Dict[str, Any]], sampled_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter full results to only include entries that match the sampled data."""
    filtered_results = []
    for sample in sampled_data:
        source_text = sample["source_text"]
        # If source_text is a dictionary, convert it to a string for matching
        if isinstance(source_text, dict):
            source_text = json.dumps(source_text, ensure_ascii=False)
        aspect_names = set(sample["summary"].keys())  # Get aspect names from the summary keys
        for aspect in aspect_names:
            if aspect in ASPECT_MAPPING:
                for result in full_results:
                    if result["source_text"] == source_text and result["aspect_name"].upper() == aspect.upper():
                        filtered_results.append(result)
                        break  # Stop searching after finding the first match
    return filtered_results

def filter_results_pmc(full_results: List[Dict[str, Any]], sampled_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter full results to only include entries that match the sampled data for PMC dataset."""
    filtered_results = []
    for sample in sampled_data:
        source_text = sample["source_text"]
        # If source_text is a dictionary, convert it to a string for matching
        if isinstance(source_text, dict):
            source_text = json.dumps(source_text, ensure_ascii=False)
        aspect_names = set(sample["summary"].keys())  # Get aspect names from the summary keys
        for aspect in aspect_names:
            for result in full_results:
                if result['context_size'] == 'short' and result["source_text"] == sample["source_text"][aspect]:
                    filtered_results.append(result)
                    break  # Stop searching after finding the first match
                if result['context_size'] == 'long' and (sample["summary"][aspect]) in result["gold_aspect_summary"]:
                    filtered_results.append(result)
                    break  # Stop searching after finding the first match
    return filtered_results

def save_data(data: List[Dict[str, Any]], path: str) -> None:
    """Save a list of dictionaries to a JSONL file (one JSON object per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")

def create_sampled_results_file(results_file, sampled_data_file, output_file, dataset='facetsum'):
    """Create a new results file that only includes entries matching the sampled data."""
    print('---' * 10)
    print(f"Loading full results from: {results_file}")
    full_results = load_data(results_file)
    print(f"Total full results entries: {len(full_results)}")
    print(f"Loading sampled data from: {sampled_data_file}")
    sampled_data = load_data(sampled_data_file)
    print(f"Total sampled data entries: {len(sampled_data)}")
    if dataset == 'facetsum':
        filtered_results = filter_results(full_results, sampled_data)
    elif dataset == 'pmc':
        filtered_results = filter_results_pmc(full_results, sampled_data)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    print(f"Total filtered results: {len(filtered_results)}")
    save_data(filtered_results, output_file)
    print(f"Filtered results saved to: {output_file}")
    print('---' * 10)

if __name__ == "__main__":
    print("##################################################################")
    print("Starting to create sampled results files...")

    # full_results_file = f"{ROOT_PATH}/facetsum/bart/bart_short_context_results.jsonl"
    # sampled_data_file = f"{ROOT_PATH}/facetsum/test_sampled_with_ids.jsonl"
    # output_file = f"{ROOT_PATH}/facetsum/bart/bart_short_context_sampled_results.jsonl"
    # create_sampled_results_file(full_results_file, sampled_data_file, output_file)
    
    # full_results_file = f"{ROOT_PATH}/facetsum_long/bart/bart_long_context_results.jsonl"
    # sampled_data_file = f"{ROOT_PATH}/facetsum_long/test_fixed_sampled_with_ids.jsonl"
    # output_file = f"{ROOT_PATH}/facetsum_long/bart/bart_long_context_sampled_results.jsonl"
    # create_sampled_results_file(full_results_file, sampled_data_file, output_file)
    
    # full_results_file = f"{ROOT_PATH}/pmc/bart/bart_long_context_results.jsonl"
    # sampled_data_file = f"{ROOT_PATH}/pmc/extracted/splits/test_sampled_with_ids.jsonl"
    # output_file = f"{ROOT_PATH}/pmc/bart/bart_long_context_sampled_results.jsonl"
    # create_sampled_results_file(full_results_file, sampled_data_file, output_file, dataset='pmc')

    full_results_file = f"{ROOT_PATH}/pmc/bart/bart_short_context_results.jsonl"
    sampled_data_file = f"{ROOT_PATH}/pmc/extracted/splits/test_sampled_with_ids.jsonl"
    output_file = f"{ROOT_PATH}/pmc/bart/bart_short_context_sampled_results.jsonl"
    create_sampled_results_file(full_results_file, sampled_data_file, output_file, dataset='pmc')