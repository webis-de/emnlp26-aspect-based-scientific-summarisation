import json
import os
import requests
import time
from pathlib import Path
from paths import DATA_ROOT
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILE = DATA_ROOT / "aclsum" / "short" / "train.jsonl"
SAVE_DIR = "./data/aclsum_pdfs/train/"
DELAY_BETWEEN_DOWNLOADS = 1.0
HEADERS = {"User-Agent": "ACL-Anthology-Downloader/1.0 (Research Purpose)"}

def download_acl_papers():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory: {SAVE_DIR}")

    # Read lines into memory once to avoid double file-handle overhead
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if not lines:
        print(f"Failure: {INPUT_FILE} is empty.")
        return

    for line in tqdm(lines, desc="Downloading PDFs"):
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
            paper_id = data.get("doc_id")
            
            if not paper_id:
                print("Warning: Line missing 'id' key.")
                continue

            pdf_url = f"https://aclanthology.org/{paper_id}.pdf"
            file_path = os.path.join(SAVE_DIR, f"{paper_id}.pdf")

            # Check if file exists AND has content (prevents skip on failed 0kb downloads)
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                continue

            response = requests.get(pdf_url, headers=HEADERS, stream=True, timeout=10)
            
            if response.status_code == 200:
                with open(file_path, 'wb') as pdf_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        pdf_file.write(chunk)
                time.sleep(DELAY_BETWEEN_DOWNLOADS)
            else:
                print(f"\nFailed {paper_id}: HTTP {response.status_code}")

        except json.JSONDecodeError as e:
            print(f"\nJSON Error: {e}")
        except Exception as e:
            print(f"\nUnexpected error for {paper_id if 'paper_id' in locals() else 'unknown'}: {e}")

if __name__ == "__main__":
    download_acl_papers()