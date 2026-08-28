import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PIPELINE_ALIASES = {"2a2s": "agentic"}
PIPELINE_ORDER = ["zs", "e2a", "rag", "cod", "self_refine", "agentic"]
EXCLUDED_UNIQUE_IDS = {"auto_e089f2de6a2a09d4"}
GOLD_SUMMARY_KEYS = ["gold_aspect_summary", "aspect_summary", "gold_summary", "reference_summary"]
GENERATED_SUMMARY_KEYS = [
  "generated_aspect_summary",
  "generated_aspect_summary_step2",
  "generated_aspect_summary_step1",
  "generated_summary",
  "summary",
  "final_summary",
]


@dataclass(frozen=True)
class FileDescriptor:
    path: Path
    dataset: str
    pipeline: str
    model: str
    metric_kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an interactive HTML viewer for per-datapoint summaries and metrics "
            "from rouge/bertscore and claim JSONL files."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing files named like dataset_pipeline_model_rouge_bs.json(l) and *_claim.json(l).",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to <input-dir>/aspect_results_explorer.html",
    )
    return parser.parse_args()


def normalize_slug(text: str) -> str:
    return text.strip().lower().replace("-", "_").replace(" ", "_")


def normalize_pipeline_name(pipeline_name: str) -> str:
    normalized = normalize_slug(pipeline_name)
    return PIPELINE_ALIASES.get(normalized, normalized)


def pick_first_nonempty_str(row: Dict[str, Any], keys: List[str]) -> str:
  for key in keys:
    value = row.get(key)
    if isinstance(value, str) and value.strip():
      return value
  return ""


def parse_filename(path: Path) -> FileDescriptor | None:
    stem = path.stem
    tokens = stem.split("_")
    if len(tokens) < 4:
        return None

    lower_tokens = [token.lower() for token in tokens]
    metric_kind = ""
    base_tokens: List[str] = tokens[:]

    if lower_tokens[-1] == "claim":
        metric_kind = "claim"
        base_tokens = tokens[:-1]
    elif len(tokens) >= 2 and lower_tokens[-2:] in (["rouge", "bs"], ["rougr", "bs"]):
        metric_kind = "rouge_bs"
        base_tokens = tokens[:-2]
    elif len(tokens) >= 2 and lower_tokens[-2:] in (["rouge", "bertscore"], ["rougr", "bertscore"]):
        metric_kind = "rouge_bs"
        base_tokens = tokens[:-2]
    elif "claim" in lower_tokens:
        claim_index = lower_tokens.index("claim")
        metric_kind = "claim"
        base_tokens = tokens[:claim_index]
    elif "rouge" in lower_tokens or "rougr" in lower_tokens or "bertscore" in lower_tokens:
        metric_kind = "rouge_bs"
        cut_index = len(tokens)
        for index, token in enumerate(lower_tokens):
            if token in {"rouge", "rougr", "bertscore", "bs"}:
                cut_index = index
                break
        base_tokens = tokens[:cut_index]

    if metric_kind == "" or len(base_tokens) < 3:
        return None

    dataset = base_tokens[0]
    pipeline = normalize_pipeline_name(base_tokens[1])
    model = "_".join(base_tokens[2:])
    return FileDescriptor(
        path=path,
        dataset=dataset,
        pipeline=pipeline,
        model=model,
        metric_kind=metric_kind,
    )


def discover_input_files(input_dir: Path) -> List[FileDescriptor]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    descriptors: List[FileDescriptor] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        descriptor = parse_filename(path)
        if descriptor is not None:
            descriptors.append(descriptor)

    if not descriptors:
        raise ValueError(
            "No matching files found. Expected names like dataset_pipeline_model_rouge_bs.jsonl and *_claim.jsonl."
        )

    return descriptors


def load_records(path: Path) -> List[Dict[str, Any]]:
  # JSONL is the primary format for this pipeline; stream it line-by-line for scale.
  if path.suffix.lower() == ".jsonl":
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_handle:
      for line_number, line in enumerate(file_handle, start=1):
        stripped = line.strip()
        if not stripped:
          continue
        try:
          obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
          raise ValueError(f"Invalid JSON line in {path.name}:{line_number}: {exc}") from exc
        if isinstance(obj, dict):
          records.append(obj)
    return records

  text = path.read_text(encoding="utf-8").strip()
  if not text:
    return []

  try:
    payload = json.loads(text)
    if isinstance(payload, list):
      return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
      return [payload]
  except json.JSONDecodeError:
    # Graceful fallback: allow line-delimited JSON even with .json extension.
    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
      stripped = line.strip()
      if not stripped:
        continue
      try:
        obj = json.loads(stripped)
      except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON line in {path.name}:{line_number}: {exc}") from exc
      if isinstance(obj, dict):
        records.append(obj)
    return records

  return []


def normalize_text_for_match(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    collapsed = re.sub(r"\s+", " ", value).strip().lower()
    return collapsed


def metric_join_id(row: Dict[str, Any], descriptor: FileDescriptor) -> str:
  aspect = normalize_text_for_match(row.get("aspect_name"))
  gold = normalize_text_for_match(pick_first_nonempty_str(row, GOLD_SUMMARY_KEYS))
  source_text = normalize_text_for_match(row.get("source_text"))
  source_excerpt = source_text[:4000] if source_text else ""

  if aspect or gold or source_excerpt:
    text_signature = {
      "dataset": normalize_text_for_match(row.get("dataset")) or normalize_text_for_match(descriptor.dataset),
      "aspect": aspect,
      "gold": gold,
      "source_excerpt": source_excerpt,
    }
    digest = hashlib.sha1(json.dumps(text_signature, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return f"txt_{digest[:20]}"

  unique_id = row.get("unique_id")
  if isinstance(unique_id, str) and unique_id.strip():
    return f"uid_{unique_id.strip()}"

  fallback = {
    "dataset": row.get("dataset", descriptor.dataset),
    "aspect_name": row.get("aspect_name", ""),
    "gold_aspect_summary": pick_first_nonempty_str(row, GOLD_SUMMARY_KEYS),
    "generated_aspect_summary": pick_first_nonempty_str(row, GENERATED_SUMMARY_KEYS),
  }
  digest = hashlib.sha1(json.dumps(fallback, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
  return f"auto_{digest[:16]}"


def pick_float(row: Dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def pick_int(row: Dict[str, Any], key: str) -> int | None:
    value = row.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def pick_str_list(row: Dict[str, Any], key: str) -> List[str]:
    value = row.get(key)
    if not isinstance(value, list):
        return []
    items: List[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            items.append(item.strip())
    return items


def extract_metrics(row: Dict[str, Any], metric_kind: str) -> Dict[str, float | None]:
    if metric_kind == "rouge_bs":
        return {
            "bertscore": pick_float(row, "bertscore"),
            "rouge1_fmeasure": pick_float(row, "rouge1_fmeasure"),
            "rouge2_fmeasure": pick_float(row, "rouge2_fmeasure"),
            "rougeL_fmeasure": pick_float(row, "rougeL_fmeasure"),
            "rougeLsum_fmeasure": pick_float(row, "rougeLsum_fmeasure"),
        }
    return {
        "fact_claim_recall": pick_float(row, "fact_claim_recall"),
        "fact_claim_precision": pick_float(row, "fact_claim_precision"),
        "fact_claim_f1": pick_float(row, "fact_claim_f1"),
    }


def extract_claim_details(row: Dict[str, Any], metric_kind: str) -> Dict[str, Any]:
    if metric_kind != "claim":
        return {}

    return {
        "fact_ref_claim_count": pick_int(row, "fact_ref_claim_count"),
        "fact_pred_claim_count": pick_int(row, "fact_pred_claim_count"),
        "fact_ref_claims_entailed": pick_int(row, "fact_ref_claims_entailed"),
        "fact_pred_claims_entailed": pick_int(row, "fact_pred_claims_entailed"),
        "fact_gold_claims": pick_str_list(row, "fact_gold_claims"),
        "fact_generated_claims": pick_str_list(row, "fact_generated_claims"),
    }


def pipeline_sort_key(pipeline_name: str) -> Tuple[int, str]:
    normalized = normalize_pipeline_name(pipeline_name)
    if normalized in PIPELINE_ORDER:
        return (PIPELINE_ORDER.index(normalized), normalized)
    return (len(PIPELINE_ORDER), normalized)


def system_key(pipeline_name: str, model_name: str) -> Tuple[str, str]:
    return (normalize_pipeline_name(pipeline_name), normalize_slug(model_name))


def build_view_model(descriptors: Iterable[FileDescriptor]) -> Dict[str, Any]:
    merged: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}
    expected_systems_by_dataset: Dict[str, set[Tuple[str, str]]] = {}

    for descriptor in descriptors:
        dataset_slug = normalize_slug(descriptor.dataset)
        if dataset_slug not in expected_systems_by_dataset:
            expected_systems_by_dataset[dataset_slug] = set()
        expected_systems_by_dataset[dataset_slug].add(system_key(descriptor.pipeline, descriptor.model))

    for descriptor in descriptors:
        rows = load_records(descriptor.path)
        for row in rows:
            if not isinstance(row, dict):
                continue

            dataset = str(row.get("dataset") or descriptor.dataset)
            aspect_name = str(row.get("aspect_name") or "")
            join_id = metric_join_id(row, descriptor)
            key = (dataset, join_id, descriptor.pipeline, descriptor.model, aspect_name)

            if key not in merged:
                merged[key] = {
                    "dataset": dataset,
                    "unique_id": join_id,
                    "pipeline": descriptor.pipeline,
                    "model": descriptor.model,
                    "aspect_name": aspect_name,
                "gold_aspect_summary": pick_first_nonempty_str(row, GOLD_SUMMARY_KEYS),
                "generated_aspect_summary": pick_first_nonempty_str(row, GENERATED_SUMMARY_KEYS),
                    "source_type": row.get("source_type", ""),
                    "context_size": row.get("context_size", ""),
                    "scores": {},
                    "claims": {},
                }

            existing = merged[key]
            row_gold_summary = pick_first_nonempty_str(row, GOLD_SUMMARY_KEYS)
            row_generated_summary = pick_first_nonempty_str(row, GENERATED_SUMMARY_KEYS)
            if not existing.get("gold_aspect_summary") and row_gold_summary:
              existing["gold_aspect_summary"] = row_gold_summary
            if not existing.get("generated_aspect_summary") and row_generated_summary:
              existing["generated_aspect_summary"] = row_generated_summary
            if not existing.get("source_type") and row.get("source_type"):
                existing["source_type"] = row.get("source_type")
            if not existing.get("context_size") and row.get("context_size"):
                existing["context_size"] = row.get("context_size")

            existing["scores"].update(extract_metrics(row, descriptor.metric_kind))
            existing["claims"].update(extract_claim_details(row, descriptor.metric_kind))

    datapoints: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for record in merged.values():
        dp_key = (record["dataset"], record["unique_id"], record["aspect_name"])
        if dp_key not in datapoints:
            datapoints[dp_key] = {
                "dataset": record["dataset"],
                "unique_id": record["unique_id"],
                "aspect_name": record["aspect_name"],
                "gold_aspect_summary": record["gold_aspect_summary"],
                "source_type": record["source_type"],
                "context_size": record["context_size"],
                "systems": [],
            }

        data_entry = datapoints[dp_key]
        if not data_entry.get("gold_aspect_summary") and record.get("gold_aspect_summary"):
            data_entry["gold_aspect_summary"] = record["gold_aspect_summary"]
        if not data_entry.get("source_type") and record.get("source_type"):
            data_entry["source_type"] = record["source_type"]
        if not data_entry.get("context_size") and record.get("context_size"):
            data_entry["context_size"] = record["context_size"]

        data_entry["systems"].append(
            {
                "pipeline": record["pipeline"],
                "model": record["model"],
                "generated_aspect_summary": record.get("generated_aspect_summary", ""),
                "scores": record.get("scores", {}),
            "claims": record.get("claims", {}),
            }
        )

    rows = sorted(datapoints.values(), key=lambda item: (item["dataset"], item["unique_id"], item["aspect_name"]))
    for row in rows:
        row["systems"].sort(key=lambda item: (pipeline_sort_key(item["pipeline"]), normalize_slug(item["model"])))

    filtered_rows: List[Dict[str, Any]] = []
    for row in rows:
        unique_id = str(row.get("unique_id") or "")
        if unique_id in EXCLUDED_UNIQUE_IDS:
            continue

        dataset_slug = normalize_slug(str(row.get("dataset") or ""))
        expected_systems = expected_systems_by_dataset.get(dataset_slug, set())
        if not expected_systems:
            continue

        present_systems = {
            system_key(str(system.get("pipeline") or ""), str(system.get("model") or ""))
            for system in row.get("systems", [])
            if isinstance(system.get("generated_aspect_summary"), str)
            and system.get("generated_aspect_summary", "").strip()
        }

        if expected_systems.issubset(present_systems):
            filtered_rows.append(row)

    datasets = sorted({row["dataset"] for row in filtered_rows})
    pipelines = sorted(
        {
            str(system.get("pipeline"))
        for row in filtered_rows
            for system in row.get("systems", [])
            if isinstance(system.get("pipeline"), str) and str(system.get("pipeline"))
        },
        key=pipeline_sort_key,
    )
    return {
        "datasets": datasets,
        "pipelines": pipelines,
        "datapoints": filtered_rows,
    }


def html_template(serialized_data: str) -> str:
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Aspect Results Explorer</title>
  <style>
    :root {{
      --bg: #f5f8f7;
      --ink: #11211e;
      --muted: #53645f;
      --card: #ffffff;
      --line: #d7e1dc;
      --accent: #0f766e;
      --accent-soft: #e6f4f2;
      --gold: #f7b500;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at 15% 5%, #e8f5f2 0%, transparent 35%),
        radial-gradient(circle at 85% 0%, #fff3d9 0%, transparent 30%),
        var(--bg);
      color: var(--ink);
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    .wrap {{ max-width: 1320px; margin: 0 auto; padding: 20px; }}
    .hero {{
      background: linear-gradient(135deg, #0f766e 0%, #14532d 80%);
      color: #f5fffd;
      border-radius: 16px;
      padding: 20px;
      box-shadow: 0 8px 30px rgba(16, 65, 60, 0.28);
      margin-bottom: 16px;
    }}
    .hero h1 {{ margin: 0; font-size: 1.5rem; }}
    .hero p {{ margin: 8px 0 0; color: #d4f5ee; }}
    .toolbar {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .field {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px;
    }}
    .field label {{ display: block; font-size: 0.85rem; color: var(--muted); margin-bottom: 6px; }}
    select {{ width: 100%; padding: 9px; border-radius: 8px; border: 1px solid var(--line); background: #fff; }}
    .meta {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      margin-bottom: 12px;
    }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }}
    .meta-key {{ font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}
    .meta-value {{ font-weight: 600; }}
    .summary {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      margin-bottom: 12px;
    }}
    .summary h3 {{ margin: 0 0 8px; font-size: 1rem; }}
    .summary p {{ margin: 0; white-space: pre-wrap; }}
    .systems {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
      gap: 12px;
    }}
    .agreement {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      margin-bottom: 12px;
      overflow-x: auto;
    }}
    .agreement h3 {{ margin: 0 0 8px; font-size: 1rem; }}
    .agreement p {{ margin: 0 0 10px; color: var(--muted); font-size: 0.88rem; }}
    .agreement-legend {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 0 0 10px;
      font-size: 0.82rem;
      color: var(--muted);
    }}
    .agreement-gradient {{
      width: 180px;
      height: 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: linear-gradient(90deg, #f6dde4 0%, #f2ecce 50%, #c7ecd6 100%);
    }}
    .agreement table {{ border-collapse: collapse; width: 100%; min-width: 420px; }}
    .agreement th, .agreement td {{ border: 1px solid var(--line); padding: 6px 8px; text-align: center; font-size: 0.85rem; }}
    .agreement th {{ background: #f0f6f4; color: var(--muted); font-weight: 600; }}
    .agreement td {{ font-variant-numeric: tabular-nums; font-weight: 600; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      min-height: 240px;
    }}
    .badge {{
      display: inline-block;
      font-size: 0.75rem;
      border-radius: 999px;
      padding: 4px 8px;
      background: var(--accent-soft);
      color: var(--accent);
      border: 1px solid #bee7df;
      font-weight: 600;
      width: fit-content;
    }}
    .model {{ font-size: 0.85rem; color: var(--muted); }}
    .gen {{ margin: 0; white-space: pre-wrap; }}
    .scores {{
      border-top: 1px dashed var(--line);
      padding-top: 8px;
      display: grid;
      grid-template-columns: repeat(2, minmax(120px, 1fr));
      gap: 4px 10px;
      font-size: 0.85rem;
    }}
    .score-k {{ color: var(--muted); }}
    .score-v {{ font-variant-numeric: tabular-nums; font-weight: 600; }}
    .claims {{
      border-top: 1px dashed var(--line);
      padding-top: 8px;
      display: grid;
      gap: 8px;
      margin-top: 4px;
    }}
    .claims-title {{ font-size: 0.85rem; font-weight: 700; color: var(--muted); }}
    .claim-meta {{ display: grid; grid-template-columns: repeat(2, minmax(120px, 1fr)); gap: 4px 8px; font-size: 0.82rem; }}
    .claim-list {{ margin: 0; padding-left: 18px; display: grid; gap: 4px; }}
    .claim-item {{ font-size: 0.84rem; }}
    .claim-match {{ color: #14532d; }}
    .claim-miss {{ color: #9f1239; }}
    .empty {{
      background: #fff9e9;
      border: 1px solid #f1dfae;
      color: #6c5512;
      border-radius: 10px;
      padding: 10px;
    }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <section class=\"hero\">
      <h1>Aspect Results Explorer</h1>
      <p>Compare one datapoint across datasets, pipelines, and models using ROUGE/BERTScore and claim-level factual metrics.</p>
    </section>

    <section class=\"toolbar\">
      <div class=\"field\">
        <label for=\"datasetSelect\">Dataset</label>
        <select id=\"datasetSelect\"></select>
      </div>
      <div class=\"field\">
        <label for=\"pipelineSelect\">Pipeline</label>
        <select id=\"pipelineSelect\"></select>
      </div>
      <div class=\"field\">
        <label for=\"datapointSelect\">Datapoint</label>
        <select id=\"datapointSelect\"></select>
      </div>
    </section>

    <section id=\"content\"></section>
  </div>

  <script>
    const APP_DATA = {serialized_data};
    const datasetSelect = document.getElementById("datasetSelect");
    const pipelineSelect = document.getElementById("pipelineSelect");
    const datapointSelect = document.getElementById("datapointSelect");
    const content = document.getElementById("content");

    function formatScore(value) {{
      if (value === null || value === undefined || Number.isNaN(value)) return "-";
      return Number(value).toFixed(4);
    }}

    function createScoreRow(key, value) {{
      const k = document.createElement("div");
      k.className = "score-k";
      k.textContent = key;
      const v = document.createElement("div");
      v.className = "score-v";
      v.textContent = formatScore(value);
      return [k, v];
    }}

    function datapointLabel(dp) {{
      const aspect = dp.aspect_name || "(no aspect)";
      return `${{dp.unique_id}} | ${{aspect}}`;
    }}

    function tokenizeSummary(text) {{
      if (!text) return new Set();
      const tokens = text.toLowerCase().match(/[a-z0-9]+/g) || [];
      return new Set(tokens);
    }}

    function tokenJaccard(a, b) {{
      if (a.size === 0 && b.size === 0) return 1;
      if (a.size === 0 || b.size === 0) return 0;
      let intersection = 0;
      a.forEach((token) => {{ if (b.has(token)) intersection += 1; }});
      const union = a.size + b.size - intersection;
      if (union <= 0) return 0;
      return intersection / union;
    }}

    function normalizeClaim(text) {{
      return String(text || "").toLowerCase().replace(/\\s+/g, " ").trim();
    }}

    function heatmapColorForSimilarity(sim) {{
      const clamped = Math.max(0, Math.min(1, Number(sim) || 0));
      const hue = 345 + (135 * clamped);   // red-ish -> green-ish
      const sat = 55;
      const light = 90 - (18 * clamped);
      return `hsl(${{hue}}, ${{sat}}%, ${{light}}%)`;
    }}

    function textColorForSimilarity(sim) {{
      const clamped = Math.max(0, Math.min(1, Number(sim) || 0));
      return clamped >= 0.65 ? "#0f3d2f" : "#5a2234";
    }}

    function claimTokens(text) {{
      const normalized = normalizeClaim(text);
      const tokens = normalized.match(/[a-z0-9]+/g) || [];
      return new Set(tokens);
    }}

    function claimSimilarity(a, b) {{
      const aNorm = normalizeClaim(a);
      const bNorm = normalizeClaim(b);
      if (!aNorm || !bNorm) return 0;
      if (aNorm === bNorm) return 1;
      if (aNorm.includes(bNorm) || bNorm.includes(aNorm)) return 0.95;

      const aSet = claimTokens(aNorm);
      const bSet = claimTokens(bNorm);
      return tokenJaccard(aSet, bSet);
    }}

    function bestClaimSimilarity(claim, counterpartClaims) {{
      if (!counterpartClaims || counterpartClaims.length === 0) return 0;
      let best = 0;
      counterpartClaims.forEach((candidate) => {{
        const sim = claimSimilarity(claim, candidate);
        if (sim > best) best = sim;
      }});
      return best;
    }}

    function createClaimList(items, counterpartClaims) {{
      const ul = document.createElement("ul");
      ul.className = "claim-list";
      if (!items || items.length === 0) {{
        const li = document.createElement("li");
        li.className = "claim-item";
        li.textContent = "(none)";
        ul.appendChild(li);
        return ul;
      }}

      items.forEach((claim) => {{
        const li = document.createElement("li");
        const bestSim = bestClaimSimilarity(claim, counterpartClaims);
        const matched = bestSim >= 0.25;
        li.className = `claim-item ${{matched ? "claim-match" : "claim-miss"}}`;
        li.textContent = claim;
        li.title = `Best lexical similarity: ${{bestSim.toFixed(3)}}`;
        ul.appendChild(li);
      }});
      return ul;
    }}

    function renderAgreement(visibleSystems) {{
      const section = document.createElement("section");
      section.className = "agreement";

      const title = document.createElement("h3");
      title.textContent = "Cross-System Summary Similarity";
      section.appendChild(title);

      const subtitle = document.createElement("p");
      subtitle.textContent = "Heatmap of token-level Jaccard similarity between generated summaries for this datapoint.";
      section.appendChild(subtitle);

      const legend = document.createElement("div");
      legend.className = "agreement-legend";
      const low = document.createElement("span");
      low.textContent = "Low";
      const grad = document.createElement("div");
      grad.className = "agreement-gradient";
      const high = document.createElement("span");
      high.textContent = "High";
      legend.appendChild(low);
      legend.appendChild(grad);
      legend.appendChild(high);
      section.appendChild(legend);

      if (!visibleSystems || visibleSystems.length === 0) {{
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "No systems available for agreement view.";
        section.appendChild(empty);
        return section;
      }}

      const labels = visibleSystems.map((sys, index) => `${{index + 1}}. ${{sys.pipeline}} | ${{sys.model}}`);
      const tokenSets = visibleSystems.map((sys) => tokenizeSummary(sys.generated_aspect_summary || ""));

      const table = document.createElement("table");
      const thead = document.createElement("thead");
      const headerRow = document.createElement("tr");
      const topLeft = document.createElement("th");
      topLeft.textContent = "System";
      headerRow.appendChild(topLeft);
      labels.forEach((label) => {{
        const th = document.createElement("th");
        th.textContent = label;
        headerRow.appendChild(th);
      }});
      thead.appendChild(headerRow);
      table.appendChild(thead);

      const tbody = document.createElement("tbody");
      labels.forEach((rowLabel, rowIndex) => {{
        const tr = document.createElement("tr");
        const rowHead = document.createElement("th");
        rowHead.textContent = rowLabel;
        tr.appendChild(rowHead);

        labels.forEach((_, colIndex) => {{
          const td = document.createElement("td");
          const sim = tokenJaccard(tokenSets[rowIndex], tokenSets[colIndex]);
          td.textContent = sim.toFixed(3);
          td.style.background = heatmapColorForSimilarity(sim);
          td.style.color = textColorForSimilarity(sim);
          td.title = `${{rowLabel}} vs ${{labels[colIndex]}}: ${{sim.toFixed(3)}}`;
          tr.appendChild(td);
        }});
        tbody.appendChild(tr);
      }});
      table.appendChild(tbody);
      section.appendChild(table);
      return section;
    }}

    function getFilteredRows(dataset, pipeline) {{
      return APP_DATA.datapoints.filter((row) => {{
        if (row.dataset !== dataset) return false;
        if (!pipeline || pipeline === "__all__") return true;
        return (row.systems || []).some((sys) => sys.pipeline === pipeline);
      }});
    }}

    function populateDatasets() {{
      datasetSelect.innerHTML = "";
      APP_DATA.datasets.forEach((dataset) => {{
        const option = document.createElement("option");
        option.value = dataset;
        option.textContent = dataset;
        datasetSelect.appendChild(option);
      }});
    }}

    function populatePipelines() {{
      pipelineSelect.innerHTML = "";
      const allOption = document.createElement("option");
      allOption.value = "__all__";
      allOption.textContent = "All";
      pipelineSelect.appendChild(allOption);

      (APP_DATA.pipelines || []).forEach((pipeline) => {{
        const option = document.createElement("option");
        option.value = pipeline;
        option.textContent = pipeline;
        pipelineSelect.appendChild(option);
      }});
      pipelineSelect.value = "__all__";
    }}

    function populateDatapoints(dataset, pipeline) {{
      const rows = getFilteredRows(dataset, pipeline);
      datapointSelect.innerHTML = "";
      rows.forEach((dp, index) => {{
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = datapointLabel(dp);
        datapointSelect.appendChild(option);
      }});
      if (rows.length === 0) {{
        renderEmpty(`No datapoints found for dataset '${{dataset}}' with pipeline '${{pipeline === "__all__" ? "All" : pipeline}}'.`);
        return;
      }}
      datapointSelect.value = "0";
      renderDatapoint(rows[0], pipeline);
    }}

    function renderEmpty(message) {{
      content.innerHTML = "";
      const div = document.createElement("div");
      div.className = "empty";
      div.textContent = message;
      content.appendChild(div);
    }}

    function renderDatapoint(dp, pipelineFilter) {{
      content.innerHTML = "";

      const visibleSystems = (dp.systems || []).filter((sys) => {{
        if (!pipelineFilter || pipelineFilter === "__all__") return true;
        return sys.pipeline === pipelineFilter;
      }});

      const meta = document.createElement("section");
      meta.className = "meta";
      const grid = document.createElement("div");
      grid.className = "meta-grid";

      const metaItems = [
        ["Unique ID", dp.unique_id],
        ["Aspect", dp.aspect_name || "-"],
        ["Source Type", dp.source_type || "-"],
        ["Context", dp.context_size || "-"],
        ["Systems", String(visibleSystems.length)],
      ];

      metaItems.forEach(([key, value]) => {{
        const cell = document.createElement("div");
        const k = document.createElement("div");
        k.className = "meta-key";
        k.textContent = key;
        const v = document.createElement("div");
        v.className = "meta-value";
        v.textContent = value;
        cell.appendChild(k);
        cell.appendChild(v);
        grid.appendChild(cell);
      }});
      meta.appendChild(grid);

      const gold = document.createElement("section");
      gold.className = "summary";
      const h = document.createElement("h3");
      h.textContent = "Gold Summary";
      const p = document.createElement("p");
      p.textContent = dp.gold_aspect_summary || "(missing)";
      gold.appendChild(h);
      gold.appendChild(p);

      const systems = document.createElement("section");
      systems.className = "systems";

      content.appendChild(meta);
      content.appendChild(gold);
      content.appendChild(renderAgreement(visibleSystems));

      visibleSystems.forEach((sys) => {{
        const card = document.createElement("article");
        card.className = "card";

        const badge = document.createElement("span");
        badge.className = "badge";
        badge.textContent = sys.pipeline;

        const model = document.createElement("div");
        model.className = "model";
        model.textContent = `Model: ${{sys.model}}`;

        const gen = document.createElement("p");
        gen.className = "gen";
        gen.textContent = sys.generated_aspect_summary || "(missing summary)";

        const scores = document.createElement("div");
        scores.className = "scores";
        const orderedScores = [
          ["BERTScore", sys.scores?.bertscore],
          ["ROUGE-1 F", sys.scores?.rouge1_fmeasure],
          ["ROUGE-2 F", sys.scores?.rouge2_fmeasure],
          ["ROUGE-L F", sys.scores?.rougeL_fmeasure],
          ["ROUGE-Lsum F", sys.scores?.rougeLsum_fmeasure],
          ["Claim Precision", sys.scores?.fact_claim_precision],
          ["Claim Recall", sys.scores?.fact_claim_recall],
          ["Claim F1", sys.scores?.fact_claim_f1],
        ];

        orderedScores.forEach(([k, v]) => {{
          const [keyNode, valNode] = createScoreRow(k, v);
          scores.appendChild(keyNode);
          scores.appendChild(valNode);
        }});

        const claims = document.createElement("div");
        claims.className = "claims";
        const claimsTitle = document.createElement("div");
        claimsTitle.className = "claims-title";
        claimsTitle.textContent = "Claim Inspector (fuzzy lexical match)";

        const goldClaims = sys.claims?.fact_gold_claims || [];
        const generatedClaims = sys.claims?.fact_generated_claims || [];

        const claimMeta = document.createElement("div");
        claimMeta.className = "claim-meta";
        const claimRows = [
          ["Gold claims", sys.claims?.fact_ref_claim_count ?? goldClaims.length],
          ["Generated claims", sys.claims?.fact_pred_claim_count ?? generatedClaims.length],
          ["Gold entailed", sys.claims?.fact_ref_claims_entailed ?? "-"],
          ["Generated entailed", sys.claims?.fact_pred_claims_entailed ?? "-"],
        ];
        claimRows.forEach(([key, value]) => {{
          const keyNode = document.createElement("div");
          keyNode.className = "score-k";
          keyNode.textContent = key;
          const valNode = document.createElement("div");
          valNode.className = "score-v";
          valNode.textContent = String(value);
          claimMeta.appendChild(keyNode);
          claimMeta.appendChild(valNode);
        }});

        const goldClaimsTitle = document.createElement("div");
        goldClaimsTitle.className = "score-k";
        goldClaimsTitle.textContent = "Gold claims (green if matched)";
        const generatedClaimsTitle = document.createElement("div");
        generatedClaimsTitle.className = "score-k";
        generatedClaimsTitle.textContent = "Generated claims (green if matched)";

        claims.appendChild(claimsTitle);
        claims.appendChild(claimMeta);
        claims.appendChild(goldClaimsTitle);
        claims.appendChild(createClaimList(goldClaims, generatedClaims));
        claims.appendChild(generatedClaimsTitle);
        claims.appendChild(createClaimList(generatedClaims, goldClaims));

        card.appendChild(badge);
        card.appendChild(model);
        card.appendChild(gen);
        card.appendChild(scores);
        card.appendChild(claims);
        systems.appendChild(card);
      }});

      content.appendChild(systems);
    }}

    datasetSelect.addEventListener("change", () => {{
      populateDatapoints(datasetSelect.value, pipelineSelect.value);
    }});

    pipelineSelect.addEventListener("change", () => {{
      populateDatapoints(datasetSelect.value, pipelineSelect.value);
    }});

    datapointSelect.addEventListener("change", () => {{
      const dataset = datasetSelect.value;
      const pipeline = pipelineSelect.value;
      const rows = getFilteredRows(dataset, pipeline);
      const selected = rows[Number(datapointSelect.value)] || null;
      if (!selected) {{
        renderEmpty("No datapoint selected.");
        return;
      }}
      renderDatapoint(selected, pipeline);
    }});

    if (!APP_DATA.datasets || APP_DATA.datasets.length === 0) {{
      renderEmpty("No data available.");
    }} else {{
      populateDatasets();
      populatePipelines();
      datasetSelect.value = APP_DATA.datasets[0];
      populateDatapoints(APP_DATA.datasets[0], pipelineSelect.value);
    }}
  </script>
</body>
</html>
"""


def write_html(view_model: Dict[str, Any], output_html: Path) -> None:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(view_model, ensure_ascii=False).replace("</script>", "<\\/script>")
    output_html.write_text(html_template(serialized), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_html = args.output_html or input_dir / "aspect_results_explorer.html"

    descriptors = discover_input_files(input_dir)
    view_model = build_view_model(descriptors)
    write_html(view_model, output_html)

    print(f"Explorer written to: {output_html}")
    print(f"Datasets: {len(view_model['datasets'])}")
    print(f"Datapoints: {len(view_model['datapoints'])}")


if __name__ == "__main__":
    main()
