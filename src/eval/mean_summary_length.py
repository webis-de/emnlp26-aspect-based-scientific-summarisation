import json
import os
from statistics import mean

import spacy
from spacy.cli import download as spacy_download

INPUT_FILE_PATH = "results/2a2s/planner_extractor_writer/qwen_qwen3_8b/scholarsum_full/long/scholarsum_full_long_test_planner_extractor_writer_final_results.jsonl"
OUTPUT_FILE_PATH = INPUT_FILE_PATH.replace(".jsonl", "_mean_summary_length.json")


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


SPACY_MODEL_NAME = "en_core_web_sm"


def load_spacy_model():
    try:
        return spacy.load(SPACY_MODEL_NAME)
    except OSError as exc:
        spacy_download(SPACY_MODEL_NAME)
        try:
            return spacy.load(SPACY_MODEL_NAME)
        except OSError as exc_inner:
            raise RuntimeError(
                f"spaCy model '{SPACY_MODEL_NAME}' could not be loaded even after attempting "
                "to download it."
            ) from exc_inner


def unique_entities_from_text(text, nlp):
    if not text:
        return set()
    doc = nlp(text)
    return {ent.text.strip().lower() for ent in doc.ents if ent.text.strip()}


def compute_entity_stats(texts, word_counts, nlp):
    counts = []
    densities = []
    for text, word_count in zip(texts, word_counts):
        unique_entities = unique_entities_from_text(text, nlp)
        unique_count = len(unique_entities)
        counts.append(unique_count)
        densities.append(unique_count / word_count if word_count > 0 else 0.0)
    return counts, densities


def main():
    nlp = load_spacy_model()
    data = load_jsonl(INPUT_FILE_PATH)
    gold_texts = [str(d.get("gold_aspect_summary") or "") for d in data]
    gen_texts = [str(d.get("generated_aspect_summary") or "") for d in data]
    gold_lengths = [len(text) for text in gold_texts]
    gen_lengths = [len(text) for text in gen_texts]
    gold_word_lengths = [len(text.split()) for text in gold_texts]
    gen_word_lengths = [len(text.split()) for text in gen_texts]

    mean_gold_chars = mean(gold_lengths)
    mean_gen_chars = mean(gen_lengths)
    mean_gold_words = mean(gold_word_lengths)
    mean_gen_words = mean(gen_word_lengths)

    gold_entity_counts, gold_entity_densities = compute_entity_stats(
        gold_texts, gold_word_lengths, nlp
    )
    gen_entity_counts, gen_entity_densities = compute_entity_stats(
        gen_texts, gen_word_lengths, nlp
    )

    mean_gold_unique_entities = mean(gold_entity_counts)
    mean_gen_unique_entities = mean(gen_entity_counts)
    mean_gold_entity_density = mean(gold_entity_densities)
    mean_gen_entity_density = mean(gen_entity_densities)

    ratio_chars = round(mean_gen_chars / mean_gold_chars, 2) if mean_gold_chars > 0 else None
    ratio_words = round(mean_gen_words / mean_gold_words, 2) if mean_gold_words > 0 else None
    ratio_entity_density = (
        round(mean_gen_entity_density / mean_gold_entity_density, 2)
        if mean_gold_entity_density > 0
        else None
    )
    ratio_unique_entities = (
        round(mean_gen_unique_entities / mean_gold_unique_entities, 2)
        if mean_gold_unique_entities > 0
        else None
    )

    result = {
        "mean_gold_summary_length_chars": round(mean_gold_chars, 2), # Mean character length of gold summaries
        "mean_generated_summary_length_chars": round(mean_gen_chars, 2), # Mean character length of generated summaries
        "mean_gold_summary_length_words": round(mean_gold_words, 2), # Mean word count of gold summaries
        "mean_generated_summary_length_words": round(mean_gen_words, 2), # Mean word count of generated summaries
        "mean_gold_unique_entities": round(mean_gold_unique_entities, 2), # Mean count of unique entities in gold summaries
        "mean_generated_unique_entities": round(mean_gen_unique_entities, 2), # Mean count of unique entities in generated summaries
        "mean_gold_entity_density": round(mean_gold_entity_density, 4), # Mean unique entity density (unique entities per word) in gold summaries
        "mean_generated_entity_density": round(mean_gen_entity_density, 4), # Mean unique entity density (unique entities per word) in generated summaries
        "ratio_generated_to_gold_chars": ratio_chars,  # Ratio of mean generated summary length in characters to mean gold summary length in characters
        "ratio_generated_to_gold_words": ratio_words, # Ratio of mean generated summary length in words to mean gold summary length in words 
        "ratio_entity_density_generated_to_gold": ratio_entity_density, # Ratio of mean unique entity density in generated summaries to mean unique entity density in gold summaries
        "ratio_unique_entities_generated_to_gold": ratio_unique_entities, # Ratio of mean unique entities in generated summaries to mean unique entities in gold summaries
        "num_examples": len(data),
    }
    save_json(result, OUTPUT_FILE_PATH)


if __name__ == "__main__":
    main()
