import os
import glob
import json
import tarfile
import logging
from pathlib import Path
from lxml import etree
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_ROOT = Path(os.getenv("ABSS_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
INPUT_DIR = DATA_ROOT / "pmc" / "raw"
OUTPUT_DIR = DATA_ROOT / "pmc" / "extracted"
OUTPUT_FILE = OUTPUT_DIR / "pmc_raw_sampled.jsonl"

# Stop after collecting this many raw papers 
MAX_PAPERS = 200000 

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

def process_xml_raw(file_content):
    # [Same XML processing function as before...]
    try:
        tree = etree.fromstring(file_content)
    except:
        return None
    
    try:
        pmc_id_node = tree.xpath(".//*[local-name()='article-id'][@pub-id-type='pmc']")
        if not pmc_id_node: return None
        pmc_id = pmc_id_node[0].text
    except:
        return None

    # Abstract
    summary_data = {}
    abstract_sections = tree.xpath(".//*[local-name()='abstract']//*[local-name()='sec']")
    if not abstract_sections: return None 

    for sec in abstract_sections:
        title_node = sec.xpath(".//*[local-name()='title']")
        if title_node:
            raw_title = "".join(title_node[0].itertext()).lower().strip()
            text_parts = [t for t in sec.itertext() if t != "".join(title_node[0].itertext())]
            text = " ".join(" ".join(text_parts).split())
            if raw_title and text:
                summary_data.setdefault(raw_title, []).append(text)

    if len(summary_data) < 3: return None
    summary_data = {k: " ".join(v) for k, v in summary_data.items()}

    # Full Text
    source_text = {}
    body_sections = tree.xpath(".//*[local-name()='body']//*[local-name()='sec']")
    for sec in body_sections:
        title_node = sec.xpath(".//*[local-name()='title']")
        if title_node:
            raw_title = "".join(title_node[0].itertext()).lower().strip()
            text_parts = [t for t in sec.itertext() if t != "".join(title_node[0].itertext())]
            text = " ".join(" ".join(text_parts).split())
            if raw_title and text:
                source_text.setdefault(raw_title, []).append(text)

    if not source_text: return None 
    source_text = {k: " ".join(v) for k, v in source_text.items()}

    return {"id": pmc_id, "summary": summary_data, "source_type": "full_text", "source_text": source_text}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tar_files = glob.glob(os.path.join(INPUT_DIR, "*.tar.gz"))
    tar_files.sort() # Systematic order (PMC000 -> PMC009)

    logging.info(f"Found {len(tar_files)} archives. Target: {MAX_PAPERS} papers.")
    
    total_papers = 0
    
    with open(OUTPUT_FILE, 'w') as f_out:
        for tar_path in tar_files:
            if total_papers >= MAX_PAPERS:
                break
                
            logging.info(f"Processing {tar_path}...")
            try:
                # stream extraction (r|gz)
                with tarfile.open(tar_path, "r|gz") as tar:
                    for member in tar:
                        if total_papers >= MAX_PAPERS:
                            break
                            
                        if member.isfile() and (member.name.endswith('.xml') or member.name.endswith('.nxml')):
                            try:
                                f_obj = tar.extractfile(member)
                                if f_obj:
                                    content = f_obj.read()
                                    record = process_xml_raw(content)
                                    if record:
                                        f_out.write(json.dumps(record) + "\n")
                                        total_papers += 1
                                        if total_papers % 1000 == 0:
                                            print(f"Collected {total_papers}...", end='\r')
                            except Exception:
                                continue
            except Exception as e:
                logging.error(f"Error reading {tar_path}: {e}")

    logging.info(f"Done. Collected {total_papers} raw papers.")

if __name__ == "__main__":
    main()