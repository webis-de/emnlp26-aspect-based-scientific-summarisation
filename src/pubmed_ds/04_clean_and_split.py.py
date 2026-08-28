import json
import os
import random
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_ROOT = Path(os.getenv("ABSS_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
PMC_EXTRACTED = DATA_ROOT / "pmc" / "extracted"
INPUT_FILE = PMC_EXTRACTED / "pmc_normalized_filtered.jsonl"
OUTPUT_DIR = PMC_EXTRACTED / "splits"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42

def validate_strict_no_fallback(data):
    """
    STRICT FILTER (NO FALLBACK):
    1. Iterates through EVERY section in the Abstract.
    2. Checks if the EXACT corresponding section key exists in Full Text.
    3. If ANY abstract section is missing in source -> DROP PAPER.
    """
    summary = data.get('summary', {})
    source = data.get('source_text', {})
    
    # --- 1. STRICT MATCH CHECK ---
    for section_name in summary.keys():
        
        # Check if the exact key exists in the source
        if section_name not in source:
            # STRICT RULE: Abstract has a section that Source lacks.
            # We do NOT fallback to background. We DROP the paper.
            return None

    # --- 2. STRUCTURE CHECK ---
    # Must still have >= 3 sections and ending labels (Results/Conclusions)
    if len(summary) < 3:
        return None
    if not ('results' in summary or 'conclusions' in summary):
        return None

    # --- 3. CONSTRUCT RECORD ---
    # If we get here, the paper is perfect.
    new_doc = {
        "id": data.get('id'),
        "summary": summary,
        "source_type": "full_text",
        "source_text": source 
    }
    
    return new_doc

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    valid_docs = []
    dropped_count = 0
    
    print(f"Strict filtering (NO FALLBACK) in progress: {INPUT_FILE}...")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            try:
                raw_doc = json.loads(line)
                clean_doc = validate_strict_no_fallback(raw_doc)
                
                if clean_doc:
                    valid_docs.append(clean_doc)
                else:
                    dropped_count += 1
                    
            except json.JSONDecodeError:
                continue
                
    print(f"Strict Filter Complete.")
    print(f"  Kept:    {len(valid_docs)}")
    print(f"  Dropped: {dropped_count} (Abstract sections missing in Full Text)")

    if len(valid_docs) == 0:
        print("CRITICAL: No documents passed the filter. Check your normalization.")
        return

    # --- SPLITTING ---
    print("Shuffling and splitting...")
    random.seed(SEED)
    random.shuffle(valid_docs)
    
    total = len(valid_docs)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    
    splits = {
        "train": valid_docs[:train_end],
        "val":   valid_docs[train_end:val_end],
        "test":  valid_docs[val_end:]
    }
    
    for name, docs in splits.items():
        path = os.path.join(OUTPUT_DIR, f"{name}.jsonl")
        print(f"  Writing {len(docs)} docs to {name}.jsonl")
        with open(path, 'w', encoding='utf-8') as f_out:
            for doc in docs:
                f_out.write(json.dumps(doc) + "\n")

    print("\nDONE. Splits created.")

if __name__ == "__main__":
    main()