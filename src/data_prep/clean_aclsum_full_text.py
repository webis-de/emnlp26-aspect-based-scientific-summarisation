import re
import html
import os
from pathlib import Path
from paths import DATA_ROOT
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# --- CONFIGURATION ---
INPUT_DIR = DATA_ROOT / "aclsum" / "aclsum_md" / "val"
OUTPUT_DIR = DATA_ROOT / "aclsum" / "aclsum_cleaned" / "val"
STRIP_TABLES = False # Keep table content by default to minimize information loss.


def _truncate_at_first_heading(text, heading_pattern):
    """Return text up to the first markdown heading that matches heading_pattern."""
    match = re.search(
        rf"(?im)^\s{{0,3}}#{{1,6}}\s*(?:\d+[\.\)]\s*)?(?:{heading_pattern})\s*$",
        text,
    )
    if not match:
        return text
    return text[:match.start()]


def _remove_front_matter_between_title_and_abstract(text):
    """
    Remove only the likely author/affiliation span between top title and Abstract.
    Conservative guards keep this from deleting body text when format is unusual.
    """
    lines = text.splitlines()
    title_idx = next(
        (i for i, line in enumerate(lines) if re.match(r"^\s*#\s+\S", line)),
        None,
    )
    if title_idx is None:
        return text

    abstract_idx = next(
        (
            i
            for i, line in enumerate(lines)
            if i > title_idx
            and i - title_idx <= 120
            and re.match(r"^\s*#{1,6}\s*(?:\d+[\.\)]\s*)?abstract\b", line, re.IGNORECASE)
        ),
        None,
    )
    if abstract_idx is None:
        return text

    # Skip stripping if another major body header appears before Abstract.
    for i in range(title_idx + 1, abstract_idx):
        if re.match(
            r"^\s*#{1,6}\s*(?:\d+[\.\)]\s*)?(?:introduction|background|method|methods|related work)\b",
            lines[i],
            re.IGNORECASE,
        ):
            return text

    kept = lines[: title_idx + 1] + lines[abstract_idx:]
    return "\n".join(kept)


def _is_structural_line(line):
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith(("#", "-", "*", "+", "|", "```", ">", "![", "\\begin{", "\\end{")):
        return True
    if re.match(r"^\d+\.\s", stripped):
        return True
    return False


def _join_wrapped_lines_preserving_structure(text):
    """
    Merge OCR/PDF line wraps inside prose paragraphs while preserving markdown structure.
    """
    blocks = re.split(r"\n\s*\n+", text)
    out_blocks = []

    for block in blocks:
        raw_lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not raw_lines:
            continue

        if any(_is_structural_line(ln) for ln in raw_lines):
            out_blocks.append("\n".join(raw_lines))
        else:
            out_blocks.append(" ".join(raw_lines))

    return "\n\n".join(out_blocks)

def get_core_clean_text(md_text, strip_tables=STRIP_TABLES):
    """
    Performs semantic cleaning of Markdown text extracted from PDFs.

    Core logic:
    1. Truncation: Hard-slices the document at 'References', 'Appendices', or 
       'Acknowledgements' to eliminate trailing noise.
    2. Metadata Purge: Deletes author names, affiliations, and emails by 
       identifying the span between the Title (#) and the Abstract header.
    3. Vision Denoising: Removes HTML containers, <img> tags, and OCR-generated 
       'Extracted text' descriptions of figures.
    4. Token Repair: Normalizes LaTeX math and repairs line-break hyphenation 
       (e.g., 'dis- \n course' -> 'discourse').
    5. Density Normalization: Collapses excessive whitespace and joins lines.

    Args:
        md_text (str): Raw Markdown content from a PDF parser.
        strip_tables (bool): If True, removes all HTML <table> blocks. Set to 
            False to retain raw tabular data (note: this increases token noise).

    Returns:
        str: A cleaned, dense text string with semantic paragraph boundaries.
    """

    # --- 1. HTML DECODING ---
    # Convert entities like &quot; or &amp; into standard characters for the tokenizer.
    text = html.unescape(md_text)

    # --- 2. SECTION TRUNCATION (References/Appendix) ---
    # Hard split at the bibliography header. Requires a newline and header marker (#)
    # to avoid false positives if 'references' is mentioned in the prose.
    # Logic: Splits the string and takes index [0] (everything before the split).
    text = _truncate_at_first_heading(text, r"references|bibliography")
    text = _truncate_at_first_heading(text, r"appendices?|supplement(?:ary materials?)?")
    text = _truncate_at_first_heading(text, r"acknowledg(?:e)?ments?")

    # --- 3. METADATA REMOVAL (Title to Abstract) ---
    # Deletes the author/affiliation block.
    # Pattern: Finds the first level 1-3 header (Title), skips all text in between, 
    # and restarts at 'Abstract'. The \d+\s+ handles numbered headers like '1 Abstract'.
    text = _remove_front_matter_between_title_and_abstract(text)

    # --- 4. NOISE PURGE (OCR/Vision/HTML) ---
    # Remove all <div> containers—these usually hold image descriptions or layout artifacts.
    text = re.sub(r'<div.*?>.*?</div>', '', text, flags=re.DOTALL)
    
    # Conditionally remove HTML tables to save significant token space.
    if strip_tables:
        text = re.sub(r'<table.*?>.*?</table>', '', text, flags=re.DOTALL)
    
    # Purge remaining <img> tags and typical figure/table captions.
    text = re.sub(r'<img.*?>', '', text, flags=re.DOTALL)
    text = re.sub(r'!\[[^\]]*]\([^)]*\)', '', text)
    # Remove standalone caption lines, but avoid deleting prose like "Table 2 lists ...".
    text = re.sub(r'(?im)^\s*(?:Figure|Table)\s+\d+[A-Za-z]?\s*[:\.-]\s+.*$', '', text)
    
    # Remove 'Extracted text:' artifacts often produced by multimodal parsers (e.g., Nougat/Marker).
    text = re.sub(r'Extracted text:.*?\n', '', text, flags=re.IGNORECASE)

    # --- 5. SEMANTIC REPAIR & LATEX STRIP ---
    # Remove author-marker LaTeX noise like ^{*1}.
    text = re.sub(r'\$ \^{.*?\} \$', '', text)
    
    # Remove standard math delimiters ($x$ -> x) while preserving the inner content.
    text = re.sub(r'\$(.*?)\$', r'\1', text)
    
    # Repair words split by line-break hyphens. 
    # Example: 'evalu- \n ation' becomes 'evaluation'.
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

    # --- 6. WHITESPACE NORMALIZATION ---
    # Collapse multiple horizontal spaces or tabs into a single space.
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Normalize per-line whitespace and collapse extreme blank runs.
    lines = [line.strip() for line in text.split('\n')]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Reconstruct natural prose paragraphs from line-wrapped PDF text.
    return _join_wrapped_lines_preserving_structure(text).strip()

# --- EXECUTION ---
if __name__ == "__main__":

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".md")]

    for filename in tqdm(files, desc="Processing files"):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename.replace(".md", ".txt"))
        
        with open(input_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
        
        cleaned = get_core_clean_text(raw_content, strip_tables=STRIP_TABLES)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned)
                
    print(f"Done. Processed files saved to {OUTPUT_DIR}")
