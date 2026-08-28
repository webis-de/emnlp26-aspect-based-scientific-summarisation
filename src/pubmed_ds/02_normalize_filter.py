import json
import os
import re
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_ROOT = Path(os.getenv("ABSS_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
PMC_EXTRACTED = DATA_ROOT / "pmc" / "extracted"
INPUT_FILE = PMC_EXTRACTED / "pmc_raw_sampled.jsonl"
OUTPUT_FILE = PMC_EXTRACTED / "pmc_normalized_filtered.jsonl"
STATS_FILE = PMC_EXTRACTED / "distribution_stats.json"
NLM_MAPPING_FILE = DATA_ROOT / "pmc" / "nlm_mapping.txt"

# Constraints (Section 3.2 of the paper)
MIN_SUMM_WORDS = 10
MAX_SUMM_WORDS = 2500
MIN_DOC_WORDS = 150
MAX_DOC_WORDS = 80000
MIN_COMPRESSION = 5.0
MAX_COMPRESSION = 500.0

# --- MANUAL OVERRIDES ---
# Note: No need for "methods:" because clean_header() strips colons automatically.
MANUAL_OVERRIDES = {
    # Methods
    "method": "methods",
    "methodology": "methods",
    "experimental design": "methods",
    "materials and methods": "methods",
    "study design": "methods",
    
    # Results
    "result": "results",
    "experimental results": "results",
    "findings": "results",
    
    # Conclusion
    "conclusion": "conclusions",
    "concluding remarks": "conclusions",
    "discussion": "conclusions", 
    
    # Intro / Background
    "objective": "objectives",
    "aim": "objectives",
    "aims": "objectives",
    "purpose": "objectives",
    "introduction": "background",
    "background": "background"
}

CATEGORY_BRIDGE = {
    "INTRODUCTION": "background", "BACKGROUND": "background",
    "OBJECTIVE": "objectives", "METHODS": "methods",
    "RESULTS": "results",
    "CONCLUSIONS": "conclusions", "DISCUSSION": "conclusions"
}

def load_nlm_mapping(file_path):
    mapping = {}
    if not os.path.exists(file_path):
        print(f"Warning: NLM Mapping file not found at {file_path}. Using Manual Overrides only.")
        return {}

    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                variant = parts[0].lower().strip()
                nlm_category = parts[1].upper().strip()
                if nlm_category in CATEGORY_BRIDGE:
                    mapping[variant] = CATEGORY_BRIDGE[nlm_category]
    return mapping

def clean_header(text):
    """
    Standardizes string: lowercase, strips numbering (1. / 2.1), strips colons.
    Example: "1. Methods:" -> "methods"
    """
    text = text.lower().strip()
    text = re.sub(r'^(section|chapter)\s*\d+[:\.]?\s*', '', text) # "Section 1"
    text = re.sub(r'^([0-9]+\.)+\s*', '', text) # "1.2."
    text = re.sub(r'^[ivx]+\.\s*', '', text)    # "IV."
    return text.strip(':').strip() # <--- Removes the colons here

def normalize_sections(raw_dict, nlm_map):
    normalized = {}
    word_count = 0
    
    for raw_key, text in raw_dict.items():
        clean_key = clean_header(raw_key)
        
        # 1. Manual Override
        norm_key = MANUAL_OVERRIDES.get(clean_key)
        
        # 2. NLM Map
        if not norm_key:
            norm_key = nlm_map.get(clean_key)
        
        # 3. Heuristics (Substring matching)
        if not norm_key:
            if "method" in clean_key: norm_key = "methods"
            elif "result" in clean_key: norm_key = "results"
            elif "conclusion" in clean_key: norm_key = "conclusions"
            elif "background" in clean_key: norm_key = "background"
            elif "aim" in clean_key or "objective" in clean_key: norm_key = "objectives"

        if norm_key:
            if norm_key in normalized:
                normalized[norm_key] += "\n" + text
            else:
                normalized[norm_key] = text
            word_count += len(text.split())
            
    return normalized, word_count

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found: {INPUT_FILE}")
        return

    print("Loading NLM Mapping...")
    header_map = load_nlm_mapping(NLM_MAPPING_FILE)

    # Statistics Counters
    raw_summ_dist = Counter()
    raw_doc_dist = Counter()
    norm_summ_dist = Counter()
    norm_doc_dist = Counter()
    
    kept_count = 0
    total_scanned = 0
    
    print(f"Processing...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in):
            try:
                data = json.loads(line)
                total_scanned += 1
                
                # Capture Raw Stats
                for k in data.get('summary', {}): raw_summ_dist[clean_header(k)] += 1
                for k in data.get('source_text', {}): raw_doc_dist[clean_header(k)] += 1
                
                # Normalize
                norm_summary, summ_words = normalize_sections(data['summary'], header_map)
                norm_source, doc_words = normalize_sections(data['source_text'], header_map)
                
                # --- FILTERING ---
                
                # 1. Structure: Abstract must have >= 3 mapped sections
                if len(norm_summary) < 3: continue
                
                # 2. Structure: Must have Ending Label
                if not ('results' in norm_summary or 'conclusions' in norm_summary): continue
                
                # 3. Length Constraints
                if not (MIN_SUMM_WORDS <= summ_words <= MAX_SUMM_WORDS): continue
                if not (MIN_DOC_WORDS <= doc_words <= MAX_DOC_WORDS): continue
                
                # 4. Compression Ratio
                ratio = doc_words / summ_words if summ_words > 0 else 0
                if not (MIN_COMPRESSION <= ratio <= MAX_COMPRESSION): continue
                
                # Capture Normalized Stats
                for k in norm_summary: norm_summ_dist[k] += 1
                for k in norm_source: norm_doc_dist[k] += 1
                
                # Save
                data['summary'] = norm_summary
                data['source_text'] = norm_source
                data['stats'] = {"summ_words": summ_words, "doc_words": doc_words, "ratio": round(ratio, 2)}
                
                f_out.write(json.dumps(data) + "\n")
                kept_count += 1

            except json.JSONDecodeError:
                continue

    # Save Stats
    print(f"Scanned {total_scanned} papers.")
    print(f"Saving distribution stats to {STATS_FILE}...")
    stats_out = {
        "raw_abstract": dict(raw_summ_dist.most_common(50)),
        "raw_fulltext": dict(raw_doc_dist.most_common(50)),
        "norm_abstract": dict(norm_summ_dist),
        "norm_fulltext": dict(norm_doc_dist)
    }
    with open(STATS_FILE, 'w') as f:
        json.dump(stats_out, f, indent=2)
        
    print(f"Done. Kept {kept_count} papers.")

if __name__ == "__main__":
    main()