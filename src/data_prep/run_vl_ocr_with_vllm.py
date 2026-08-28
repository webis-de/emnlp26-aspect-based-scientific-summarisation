"""Generate Markdown (with image descriptions) using PaddleOCR-VL + vLLM."""

from __future__ import annotations

import argparse
import html
import re
import shutil
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
from paddleocr import PaddleOCR, PaddleOCRVL
from PIL import Image, ImageOps


DEFAULT_DESCRIPTION_PROMPT = (
    "Carefully read this figure, chart, or embedded document image. Provide a one-sentence summary,"
    " then transcribe every visible heading, label, bullet, or table row in reading order so nothing is omitted."
)

DEFAULT_DESCRIPTION_SETTINGS: Dict[str, Any] = {
    "temperature": 0.15,
    "top_p": 0.85,
    "repetition_penalty": 1.08,
    "max_new_tokens": 640,
    "use_cache": True,
    "skip_special_tokens": True,
}

FALLBACK_DESCRIPTION_PROMPT = (
    "Transcribe the entire figure verbatim. List every heading, bullet, axis label, arrow label,"
    " and sentence exactly as printed so the reader can reconstruct the page without seeing it."
    " Never say that description is impossible; if text is uncertain, note it as '(unclear text)'"
    " but still describe its approximate wording and placement."
)
FALLBACK_FAILURE_SNIPPETS = [
    "not possible",
    "cannot",
    "too blurry",
    "graphic design",
    "does not contain",
]

REPEATED_TOKEN_LIMIT = 4

DEFAULT_DESCRIPTION_TILE_WIDTH = 600
DEFAULT_DESCRIPTION_TILE_OVERLAP = 80
DEFAULT_DESCRIPTION_BORDER = 16
DEFAULT_DESCRIPTION_MIN_WIDTH = 2000

_TEXT_OCR: PaddleOCR | None = None

HTML_IMG_PATTERN = re.compile(r'(<img\b[^>]*src="(?P<src>[^"]+)"[^>]*>)', re.IGNORECASE)
MARKDOWN_IMG_PATTERN = re.compile(r"(!\[[^\]]*]\((?P<src>[^)]+)\))")
WHITESPACE_PATTERN = re.compile(r"\s+")


def _normalize_image_key(key: str) -> str:
    return key.lstrip("./")


def _sanitize_description(text: str) -> str:
    cleaned = WHITESPACE_PATTERN.sub(" ", text.strip())
    return cleaned.strip('"')


def _normalize_multiline_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _has_repeated_tokens(text: str) -> bool:
    if not text:
        return False
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not tokens:
        return False
    streak = 1
    prev = tokens[0]
    for token in tokens[1:]:
        if token == prev:
            streak += 1
            if streak >= REPEATED_TOKEN_LIMIT:
                return True
        else:
            streak = 1
            prev = token
    return False


def _needs_fallback(description: str | None) -> bool:
    if not description:
        return True
    lowered = description.lower()
    if any(snippet in lowered for snippet in FALLBACK_FAILURE_SNIPPETS):
        return True
    return _has_repeated_tokens(description)


def _ensure_alt_text(img_tag: str, description: str) -> str:
    escaped = html.escape(description, quote=True)
    if re.search(r'alt="[^"]*"', img_tag, flags=re.IGNORECASE):
        return re.sub(
            r'alt="[^"]*"',
            lambda _match: f'alt="{escaped}"',
            img_tag,
            count=1,
            flags=re.IGNORECASE,
        )

    if "/>" in img_tag:
        insert_at = img_tag.rfind("/>")
        return f'{img_tag[:insert_at].rstrip()} alt="{escaped}" />'

    insert_at = img_tag.rfind(">")
    if insert_at == -1:
        return img_tag
    return f'{img_tag[:insert_at]} alt="{escaped}">'


def _split_image_for_description(
    image: Image.Image,
    tile_width: int | None,
    overlap: int,
    border: int,
) -> List[Tuple[Image.Image, Tuple[int, int]]]:
    """Return a list of (tile, (start_px, end_px)) tuples for description queries."""

    processed = image
    if border > 0:
        processed = ImageOps.expand(processed, border=border, fill="white")

    width = processed.width
    if tile_width is None or tile_width <= 0 or width <= tile_width:
        return [(processed, (0, width))]

    overlap = max(0, overlap)
    step = max(1, tile_width - overlap)
    tiles: List[Tuple[Image.Image, Tuple[int, int]]] = []
    start = 0
    while start < width:
        end = min(width, start + tile_width)
        tiles.append((processed.crop((start, 0, end, processed.height)), (start, end)))
        if end == width:
            break
        start += step

    return tiles


def _prepare_tile_image(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    if DEFAULT_DESCRIPTION_MIN_WIDTH and rgb.width < DEFAULT_DESCRIPTION_MIN_WIDTH:
        scale = DEFAULT_DESCRIPTION_MIN_WIDTH / rgb.width
        new_size = (
            int(rgb.width * scale),
            int(rgb.height * scale),
        )
        rgb = rgb.resize(new_size, Image.Resampling.LANCZOS)
    rgb = ImageOps.autocontrast(rgb)
    rgb = ImageOps.equalize(rgb)
    return rgb


def _get_text_ocr() -> PaddleOCR:
    global _TEXT_OCR
    if _TEXT_OCR is None:
        _TEXT_OCR = PaddleOCR(lang="en", use_angle_cls=True)
    return _TEXT_OCR


def _describe_with_standard_ocr(image: Image.Image) -> str:
    ocr = _get_text_ocr()
    bgr_image = np.asarray(image.convert("RGB"))[:, :, ::-1]
    results = ocr.predict(bgr_image, use_textline_orientation=True)
    lines: List[str] = []
    for result in results:
        texts = result.get("rec_texts", [])
        for text in texts:
            cleaned = text.strip()
            if cleaned:
                lines.append(cleaned)
    return "\n".join(lines)


def describe_markdown_images(
    pipeline: PaddleOCRVL,
    markdown_images: List[Dict[str, Image.Image]],
    *,
    prompt: str = DEFAULT_DESCRIPTION_PROMPT,
    vlm_kwargs: Mapping[str, Any] | None = None,
    tile_width: int | None = DEFAULT_DESCRIPTION_TILE_WIDTH,
) -> Dict[str, str]:
    if not markdown_images:
        return {}

    paddlex_pipeline = getattr(pipeline, "paddlex_pipeline", None)
    if paddlex_pipeline is None or not hasattr(paddlex_pipeline, "vl_rec_model"):
        raise RuntimeError(
            "Cannot locate vl_rec_model on the PaddleOCRVL pipeline instance."
        )

    vl_rec_model = paddlex_pipeline.vl_rec_model

    unique_images: List[tuple[str, Image.Image]] = []
    seen = set()
    for page_images in markdown_images:
        for rel_path, image in page_images.items():
            normalized = _normalize_image_key(rel_path)
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_images.append((normalized, image))

    if not unique_images:
        return {}

    image_lookup = dict(unique_images)
    text_ocr_cache: Dict[str, str] = {}

    model_kwargs: Dict[str, Any] = dict(DEFAULT_DESCRIPTION_SETTINGS)
    if vlm_kwargs:
        model_kwargs.update(vlm_kwargs)

    queries: List[Dict[str, Any]] = []
    query_meta: List[Dict[str, Any]] = []
    for image_key, image in unique_images:
        tiles = _split_image_for_description(
            image,
            tile_width=tile_width,
            overlap=DEFAULT_DESCRIPTION_TILE_OVERLAP,
            border=DEFAULT_DESCRIPTION_BORDER,
        )
        tile_total = len(tiles)
        for tile_index, (tile_img, (start_px, end_px)) in enumerate(tiles, start=1):
            prepped_image = _prepare_tile_image(tile_img)
            bgr_array = np.asarray(prepped_image)[:, :, ::-1]
            tile_prompt = prompt
            if tile_total > 1:
                tile_prompt = (
                    f"{prompt} This is segment {tile_index} of {tile_total}, covering columns {start_px}-{end_px}px. "
                    "Describe only the content visible in this slice."
                )
            queries.append({"image": bgr_array, "query": tile_prompt})
            query_meta.append(
                {
                    "image_key": image_key,
                    "tile_index": tile_index,
                    "tile_total": tile_total,
                }
            )

    responses = list(vl_rec_model.predict(queries, **model_kwargs))

    aggregated: Dict[str, List[str]] = defaultdict(list)
    for meta, response in zip(query_meta, responses):
        raw_text = response.get("result", "") if isinstance(response, dict) else ""
        description = _sanitize_description(raw_text)
        if not description:
            continue
        if meta["tile_total"] > 1:
            prefix = f"Segment {meta['tile_index']}/{meta['tile_total']}: "
            aggregated[meta["image_key"]].append(prefix + description)
        else:
            aggregated[meta["image_key"]].append(description)

    descriptions: Dict[str, str] = {
        key: "\n".join(parts) for key, parts in aggregated.items() if parts
    }

    fallback_queries: List[Dict[str, Any]] = []
    fallback_meta: List[str] = []
    for image_key, image in unique_images:
        if _needs_fallback(descriptions.get(image_key)):
            prepped = _prepare_tile_image(image)
            bgr_array = np.asarray(prepped)[:, :, ::-1]
            fallback_queries.append(
                {"image": bgr_array, "query": FALLBACK_DESCRIPTION_PROMPT}
            )
            fallback_meta.append(image_key)

    if fallback_queries:
        fallback_responses = list(
            vl_rec_model.predict(fallback_queries, **model_kwargs)
        )
        for image_key, response in zip(fallback_meta, fallback_responses):
            text = response.get("result", "") if isinstance(response, dict) else ""
            description = _sanitize_description(text)
            if description:
                descriptions[image_key] = description

    for image_key, image in image_lookup.items():
        if _needs_fallback(descriptions.get(image_key)):
            text_result = text_ocr_cache.get(image_key)
            if text_result is None:
                text_result = _describe_with_standard_ocr(image)
                text_ocr_cache[image_key] = text_result
            description = _sanitize_description(text_result)
            if description:
                descriptions[image_key] = description

    for image_key, image in image_lookup.items():
        text_result = text_ocr_cache.get(image_key)
        if text_result is None:
            text_result = _describe_with_standard_ocr(image)
            text_ocr_cache[image_key] = text_result
        text_block = _normalize_multiline_text(text_result)
        if not text_block:
            continue
        prefix = f"Extracted text:\n{text_block}"
        existing = descriptions.get(image_key)
        if not existing or _needs_fallback(existing):
            descriptions[image_key] = prefix
        else:
            descriptions[image_key] = f"{existing}\n\n{prefix}"

    return descriptions


def inject_image_descriptions(
    markdown_text: str, descriptions: Mapping[str, str]
) -> str:
    if not descriptions:
        return markdown_text

    used_sources = set()

    def replace_html(match: re.Match[str]) -> str:
        tag = match.group(1)
        src = _normalize_image_key(match.group("src"))
        description = descriptions.get(src)
        if not description:
            return tag
        used_sources.add(src)
        updated_tag = _ensure_alt_text(tag, description)
        caption = f"\n<p><em>{html.escape(description)}</em></p>"
        return updated_tag + caption

    updated = HTML_IMG_PATTERN.sub(replace_html, markdown_text)

    def replace_markdown(match: re.Match[str]) -> str:
        src_raw = match.group("src")
        normalized = _normalize_image_key(src_raw)
        description = descriptions.get(normalized)
        if not description:
            return match.group(0)
        image_markup = f"![{description}]({match.group('src')})"
        if normalized in used_sources:
            return image_markup
        used_sources.add(normalized)
        return f"{image_markup}\n\n*{description}*"

    return MARKDOWN_IMG_PATTERN.sub(replace_markdown, updated)


def save_markdown_images(
    markdown_images: List[Dict[str, Image.Image]], output_dir: Path
) -> List[Path]:
    created_dirs: List[Path] = []
    for images in markdown_images:
        for rel_path, image in images.items():
            file_path = output_dir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(file_path)
            created_dirs.append(file_path.parent)
    return created_dirs


def cleanup_image_directories(directories: List[Path]) -> None:
    seen = set()
    for directory in sorted(directories, key=lambda d: len(d.parts), reverse=True):
        if directory in seen:
            continue
        seen.add(directory)
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)


def _create_default_pipeline() -> PaddleOCRVL:
    return PaddleOCRVL(
        vl_rec_backend="vllm-server",
        vl_rec_server_url="http://127.0.0.1:8118/v1",
        use_chart_recognition=True,
        format_block_content=True,
    )


def ocr_pdf_to_markdown_with_images(
    input_file: str,
    output_dir: str = "./output",
    *,
    description_prompt: str = DEFAULT_DESCRIPTION_PROMPT,
    description_kwargs: Mapping[str, Any] | None = None,
    description_tile_width: int | None = DEFAULT_DESCRIPTION_TILE_WIDTH,
    pipeline: PaddleOCRVL | None = None,
) -> Path:
    """Convert a PDF into Markdown and enrich embedded images with VLM descriptions."""

    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pipeline_instance = pipeline or _create_default_pipeline()

    start = time.time()
    output = pipeline_instance.predict(
        input=str(input_path),
        use_chart_recognition=True,
        format_block_content=True,
        use_queues=True,
    )
    end = time.time()
    print(f"OCR finished in {end - start:.2f} seconds")

    markdown_list: List[Dict[str, Any]] = []
    markdown_images: List[Dict[str, Image.Image]] = []

    for res in output:
        md_info = res.markdown
        markdown_list.append(md_info)
        markdown_images.append(md_info.get("markdown_images", {}))

    merged_markdown = pipeline_instance.concatenate_markdown_pages(markdown_list)
    descriptions = describe_markdown_images(
        pipeline_instance,
        markdown_images,
        prompt=description_prompt,
        vlm_kwargs=description_kwargs,
        tile_width=description_tile_width,
    )
    enriched_markdown = inject_image_descriptions(merged_markdown, descriptions)

    mkd_file_path = output_path / f"{input_path.stem}.md"
    mkd_file_path.write_text(enriched_markdown, encoding="utf-8")

    image_dirs = save_markdown_images(markdown_images, output_path)
    if image_dirs:
        cleanup_image_directories(image_dirs)
        print("Temporary image assets cleaned up after processing.")

    print(f"Markdown saved to: {mkd_file_path}")
    if descriptions:
        print(
            f"Annotated {len(descriptions)} image(s) with natural language descriptions."
        )

    return mkd_file_path


def process_pdfs_in_directory(
    input_dir: str,
    output_dir: str,
    *,
    description_prompt: str = DEFAULT_DESCRIPTION_PROMPT,
    description_kwargs: Mapping[str, Any] | None = None,
    description_tile_width: int | None = DEFAULT_DESCRIPTION_TILE_WIDTH,
    pipeline: PaddleOCRVL | None = None,
) -> List[Path]:
    """Run OCR over every PDF under ``input_dir`` while mirroring its structure."""

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_path}")

    pdf_files = sorted(p for p in input_path.rglob("*.pdf") if p.is_file())
    if not pdf_files:
        print(f"No PDF files found under {input_path}")
        return []

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # ERROR HANDLING: Define log file path
    failure_log = output_root / "failed_files.log"

    pipeline_instance = pipeline or _create_default_pipeline()
    created_markdowns: List[Path] = []
    total = len(pdf_files)

    for index, pdf_path in enumerate(pdf_files, start=1):
        relative_parent = pdf_path.parent.relative_to(input_path)
        target_dir = (
            output_root
            if relative_parent == Path(".")
            else output_root / relative_parent
        )
        
        # ------------------------------------------------------------------
        # SKIPPING LOGIC: Check if output file already exists
        # ------------------------------------------------------------------
        expected_md_path = target_dir / f"{pdf_path.stem}.md"
        if expected_md_path.exists():
            print(f"[{index}/{total}] ⏭️  Skipping {pdf_path.name} (Already exists: {expected_md_path})")
            continue

        print(f"[{index}/{total}] Processing {pdf_path}")

        # ERROR HANDLING: Try/Except block for individual PDF
        try:
            markdown_path = ocr_pdf_to_markdown_with_images(
                input_file=str(pdf_path),
                output_dir=str(target_dir),
                description_prompt=description_prompt,
                description_kwargs=description_kwargs,
                description_tile_width=description_tile_width,
                pipeline=pipeline_instance,
            )
            created_markdowns.append(markdown_path)
            print(f"[{index}/{total}] Finished {pdf_path.name} -> {markdown_path}")

        except Exception as e:
            # Catch ANY error during processing of this specific file
            error_msg = f"❌ [ERROR] Skipped {pdf_path.name}: {e}"
            print(error_msg)
            
            # Log to file with timestamp and stack trace for debugging
            with open(failure_log, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*40}\n")
                f.write(f"TIMESTAMP: {time.ctime()}\n")
                f.write(f"FILE: {pdf_path}\n")
                f.write(f"ERROR: {str(e)}\n")
                f.write("TRACEBACK:\n")
                f.write(traceback.format_exc())
                f.write(f"{'='*40}\n")
            
            # Continue to the next file in the loop
            continue

    print(
        f"Processed {total} PDF file(s) from {input_path} into {output_root} with mirrored structure."
    )
    if failure_log.exists():
        print(f"⚠️  Errors were encountered. Check log at: {failure_log}")

    return created_markdowns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR-VL + vLLM over PDFs to produce Markdown output."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to a single PDF to convert.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing PDFs to process recursively.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output_md",
        help="Directory where Markdown (and images) will be written.",
    )
    parser.add_argument(
        "--description-prompt",
        type=str,
        default=DEFAULT_DESCRIPTION_PROMPT,
        help="Custom prompt for describing images (optional).",
    )
    parser.add_argument(
        "--description-tile-width",
        type=int,
        default=DEFAULT_DESCRIPTION_TILE_WIDTH,
        help="Tile width for splitting wide images when describing them.",
    )

    args = parser.parse_args()

    if args.input_dir:
        process_pdfs_in_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            description_prompt=args.description_prompt,
            description_tile_width=args.description_tile_width,
        )
    elif args.input_file:
        ocr_pdf_to_markdown_with_images(
            input_file=args.input_file,
            output_dir=args.output_dir,
            description_prompt=args.description_prompt,
            description_tile_width=args.description_tile_width,
        )
    else:
        parser.error(
            "Please supply --input-dir for batch processing or --input-file for a single PDF."
        )
