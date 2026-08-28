import json
import os
import glob
from pathlib import Path
from paths import DATA_ROOT
from tqdm import tqdm

# --- CONFIGURATION ---
METADATA_FILE = DATA_ROOT / "aclsum" / "short" / "test_with_ids.jsonl"
CLEAN_TEXT_DIR = DATA_ROOT / "aclsum" / "aclsum_cleaned" / "test"
OUTPUT_FILE = DATA_ROOT / "aclsum" / "long" / "test_with_ids.jsonl"

def build_full_text_index(directory):
    """
    Reads all .txt files in the directory and maps normalized IDs to text content.
    """
    index = {}
    print(f"Indexing full-text files from {directory}...")
    file_paths = glob.glob(os.path.join(directory, "*.txt"))
    
    for path in tqdm(file_paths, desc="Loading texts"):
        # Extract filename without extension (e.g., 'E09-1056.txt' -> 'E09-1056')
        file_id = os.path.basename(path).replace(".txt", "")
        
        with open(path, 'r', encoding='utf-8') as f:
            # We strip() to ensure no trailing newlines from the file read
            index[file_id.upper()] = f.read().strip()
            
    return index

def generate_long_jsonl():
    # 1. Index the cleaned full-text files
    text_index = build_full_text_index(CLEAN_TEXT_DIR)
    
    stats = {"matched": 0, "fallback": 0}
    
    # 2. Count total lines for progress bar
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    # 3. Process and Join
    print(f"Merging metadata with full text...")
    with open(METADATA_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, total=total_lines, desc="Processing JSONL"):
            data = json.loads(line)
            doc_id = data.get("doc_id", "").upper()
            
            # Match logic: Use normalized uppercase ID
            if doc_id in text_index:
                data["source_text"] = text_index[doc_id]
                data["source_type"] = "full_text"
                stats["matched"] += 1
            else:
                # Fallback: Keep original abstract/intro/conclusion if no full text found
                stats["fallback"] += 1
            
            # Write the updated record
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    print("\n--- Processing Complete ---")
    print(f"Successfully joined: {stats['matched']} papers")
    print(f"Fallback (short text): {stats['fallback']} papers")
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_long_jsonl()