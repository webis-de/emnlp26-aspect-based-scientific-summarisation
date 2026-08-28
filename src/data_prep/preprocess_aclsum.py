import argparse
import json
import hashlib
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from tqdm.auto import tqdm

from io_dataclasses import AspectSummaryDatapoint
from paths import DATA_ROOT

DEFAULT_INPUT_DIR = DATA_ROOT / "aclsum" / "raw_split"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "aclsum"
ASPECT_KEYS: tuple[str, ...] = ("challenge", "approach", "outcome")
SECTION_ORDER: tuple[str, ...] = ("abstract", "introduction", "conclusion")


def load_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def build_source_text(entry: Dict, include_title=False) -> str:
    if include_title:
        title = (entry.get("title") or "").strip()
    sections = entry.get("sentences") or {}
    if not isinstance(sections, dict):
        sections = {}
    ordered_keys: List[str] = []
    ordered_sections: List[str] = []
    # Preserve canonical order, then append any remaining sections as they appear.
    for key in SECTION_ORDER:
        if key in sections:
            ordered_keys.append(key)
    for key in sections.keys():
        if key not in ordered_keys:
            ordered_keys.append(key)
    for key in ordered_keys:
        sents = sections.get(key) or []
        if isinstance(sents, list):
            section_body = " ".join(sent.strip() for sent in sents if isinstance(sent, str)).strip()
        else:
            section_body = str(sents).strip()
        if not section_body:
            continue
        header = f"{key.capitalize()}\n" if key else ""
        ordered_sections.append(f"{header}{section_body}")
    if include_title:
        full_text_parts = [part for part in [title, *ordered_sections] if part]
    else:
        full_text_parts = [part for part in ordered_sections if part]
    return "\n\n".join(full_text_parts).strip()


def extract_summary(entry: Dict) -> Dict[str, str]:
    summary = entry.get("summary") or {}
    if not isinstance(summary, dict):
        summary = {}
    normalized: Dict[str, str] = {}
    for aspect in ASPECT_KEYS:
        value = summary.get(aspect)
        normalized[aspect] = value.strip() if isinstance(value, str) else ""
    return normalized


def format_datapoint(entry: Dict) -> AspectSummaryDatapoint:
    source_text = build_source_text(entry)
    if not source_text:
        raise ValueError("Empty source_text")
    summary = extract_summary(entry)
    doc_id = entry.get("id") or hashlib.md5(source_text.encode("utf-8")).hexdigest()
    doc_id = str(doc_id)
    return AspectSummaryDatapoint(
        source_text=source_text,
        summary=summary,
        source_type="abstract_intro_conclusion",
        doc_id=doc_id,
    )


def convert_split(split_name: str, input_dir: Path, output_dir: Path) -> None:
    input_path = input_dir / f"{split_name}.jsonl"
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing split file: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split_name}.jsonl"
    total = 0
    written = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for entry in tqdm(load_jsonl(input_path), desc=f"Processing {split_name}"):
            total += 1
            try:
                datapoint = format_datapoint(entry)
            except ValueError:
                skipped += 1
                continue
            payload = {
                "doc_id": datapoint.doc_id,
                "source_type": datapoint.source_type,
                "source_text": datapoint.source_text,
                "summary": datapoint.summary,
            }
            out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
    print(
        f"{split_name}: wrote {written} records, skipped {skipped}, out of {total} entries. Saved to {output_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess ACLSum jsonl splits into 2a2s format.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing raw ACLSum jsonl splits.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store processed jsonl splits.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "val", "test"),
        help="Split names to process (files are <split>.jsonl).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split_name in args.splits:
        convert_split(split_name, args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
