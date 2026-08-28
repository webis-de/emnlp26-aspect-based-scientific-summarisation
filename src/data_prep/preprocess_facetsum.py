from datasets import load_dataset, load_from_disk
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from tqdm.auto import tqdm
import hashlib
from io_dataclasses import AspectSummaryDatapoint
from paths import DATA_ROOT
import uuid

def reconstruct_document(facetsum_datapoint: Dict, include_title=False, flatten_document=False) -> str:
    """
    Reconstruct the original document from the FacetSum datapoint.
    """
    if include_title:
        title = facetsum_datapoint.get('title', '')
    section_names: List[str] = facetsum_datapoint.get('section_names', [])
    section_sents: List[List[str]] = facetsum_datapoint.get('sections', [])
    assert len(section_names) == len(section_sents), "Section names and texts length mismatch."
    sections = []
    if flatten_document:
        for name, sents in zip(section_names, section_sents):
            section_header = f"{name}\n" if name else ""
            section_body = " ".join(sents)
            sections.append(f"{section_header}{section_body}")
        if include_title:
            full_text = f"{title}\n\n" + "\n\n".join(sections)
        else:
            full_text = "\n\n".join(sections)
        return full_text.strip()
    else:
        full_text: Dict[str, str] = {}
        for name, sents in zip(section_names, section_sents):
            section_body = " ".join(sents)
            full_text[name] = section_body
        if include_title:
            full_text = {"Title": title, **full_text}
        return full_text

def just_intro_and_conclusion(facetsum_datapoint: Dict, include_title=False) -> str:
    """
    Reconstruct a document containing only the introduction and conclusion sections.
    """
    if include_title:
        title = facetsum_datapoint.get('title', '')
    section_names: List[str] = facetsum_datapoint.get('section_names', [])
    section_sents: List[List[str]] = facetsum_datapoint.get('sections', [])
    assert len(section_names) == len(section_sents), "Section names and texts length mismatch."

    intro_concl_sections = []
    intro_found = False
    concl_found = False
    for name, sents in zip(section_names, section_sents):
        if any([_keyword in name.lower() for _keyword in ['intro', 'purpose']]): # Table 4 in FacetSum paper
            section_header = f"{name}\n"
            section_body = " ".join(sents)
            intro_concl_sections.append(f"{section_header}{section_body}")
            intro_found = True
        if any([_keyword in name.lower() for _keyword in ['conclu', 'future']]): # Table 4 in FacetSum paper
            section_header = f"{name}\n"
            section_body = " ".join(sents)
            intro_concl_sections.append(f"{section_header}{section_body}")
            concl_found = True
    
    if not intro_found or not concl_found:
        return ""
    else:
        if include_title:
            context_text = f"{title}\n\n" + "\n\n".join(intro_concl_sections)
        else:
            context_text = "\n\n".join(intro_concl_sections)
        return context_text.strip()

def format_aspect_summary_datapoint(facetsum_datapoint: Dict, source_text_type: str='intro_and_conclusion', include_full_text: bool=True) -> AspectSummaryDatapoint:
    """
    Convert a FacetSum datapoint into an AspectSummaryDatapoint.
    """
    if source_text_type == 'intro_and_conclusion':
        doc_text = just_intro_and_conclusion(facetsum_datapoint)
        if include_full_text:
            full_text = reconstruct_document(facetsum_datapoint)
    elif source_text_type == 'full_text':
        doc_text = reconstruct_document(facetsum_datapoint)
    else:
        raise ValueError(f"Unknown source_text_type: {source_text_type}")

    abstract_section_names: List[str] = facetsum_datapoint.get('abstract_sections_names', [])
    abstract_section_sents: List[List[str]] = facetsum_datapoint.get('abstract_sections', [])
    assert len(abstract_section_names) == len(abstract_section_sents), "Abstract section names and texts length mismatch."

    summary_dict: Dict[str, str] = {}
    for name, sents in zip(abstract_section_names, abstract_section_sents):
        summary_dict[name] = " ".join(sents)

    if source_text_type == 'intro_and_conclusion':
        if include_full_text:
            return AspectSummaryDatapoint(source_text=doc_text, summary=summary_dict, source_type=source_text_type, doc_id=hashlib.md5(doc_text.encode('utf-8')).hexdigest(), full_text=full_text)
        else:
            return AspectSummaryDatapoint(source_text=doc_text, summary=summary_dict, source_type=source_text_type, doc_id=hashlib.md5(doc_text.encode('utf-8')).hexdigest())
    else:
        return AspectSummaryDatapoint(source_text=doc_text, summary=summary_dict, source_type=source_text_type, doc_id=uuid.uuid4().hex)

def inspect_facetsum_dataset(ds):
    print(ds['train'][0].keys())  # Example access to first training example

    # Pretty-print an example
    import pprint
    pprint.pprint(ds['train'][0])

    # Convert to AspectSummaryDatapoint format
    example_datapoint = format_aspect_summary_datapoint(ds['train'][0])
    print("\nReconstructed Document:\n", example_datapoint.source_text)
    print("\nAspect Summaries:")
    for aspect, summary in example_datapoint.summary.items():
        print(f"Aspect: {aspect}\nSummary: {summary}\n")

def save_facetsum_dataset_locally(ds) -> None:
    # Save to local disk
    raw_path = DATA_ROOT / "facetsum" / "raw" / "facetsum"
    ds.save_to_disk(str(raw_path))
    # Print confirmation
    print(f"FacetSum dataset saved locally at {raw_path}")

def prepare_facetsum_data_for_2a2s(ds, source_text_type) -> None:
    if source_text_type not in ['intro_and_conclusion', 'full_text']:
        raise ValueError(f"Unknown source_text_type: {source_text_type}")
    if source_text_type == 'full_text':
        include_full_text = False
    else:
        include_full_text = True

    # Now parse through the entire dataset and convert
    converted_dataset_train: List[AspectSummaryDatapoint] = [
        format_aspect_summary_datapoint(dp, source_text_type=source_text_type, include_full_text=include_full_text)
        for dp in tqdm(ds['train'], desc="Converting train", total=len(ds['train']))
    ]
    converted_dataset_validation: List[AspectSummaryDatapoint] = [
        format_aspect_summary_datapoint(dp, source_text_type=source_text_type, include_full_text=include_full_text)
        for dp in tqdm(ds['validation'], desc="Converting validation", total=len(ds['validation']))
    ]
    converted_dataset_test: List[AspectSummaryDatapoint] = [
        format_aspect_summary_datapoint(dp, source_text_type=source_text_type, include_full_text=include_full_text)
        for dp in tqdm(ds['test'], desc="Converting test", total=len(ds['test']))
    ]

    # Compute statistics
    num_train = len(converted_dataset_train)
    num_validation = len(converted_dataset_validation)
    num_test = len(converted_dataset_test)
    print(f"Converted dataset sizes - Train: {num_train}, Validation: {num_validation}, Test: {num_test}")
    # Remove datapoints with empty source_text
    if source_text_type == 'intro_and_conclusion':
        converted_dataset_train = [dp for dp in converted_dataset_train if dp.source_text.strip()]
        converted_dataset_validation = [dp for dp in converted_dataset_validation if dp.source_text.strip()]
        converted_dataset_test = [dp for dp in converted_dataset_test if dp.source_text.strip()]
    else:
        # For full_text source type, source_text is never empty
        pass
    print(f"After removing empty source_text - Train: {len(converted_dataset_train)}, Validation: {len(converted_dataset_validation)}, Test: {len(converted_dataset_test)}")
    # Compute lost datapoints in percentage
    lost_train = num_train - len(converted_dataset_train)
    lost_validation = num_validation - len(converted_dataset_validation)
    lost_test = num_test - len(converted_dataset_test)
    print(f"Lost datapoints - Train: {lost_train} ({(lost_train/num_train)*100:.2f}%), "
          f"Validation: {lost_validation} ({(lost_validation/num_validation)*100:.2f}%), "
          f"Test: {lost_test} ({(lost_test/num_test)*100:.2f}%)")

    # Save converted datasets to disk as jsonl files
    import json
    def save_aspect_summary_dataset(dataset: List[AspectSummaryDatapoint], filepath: str) -> None:
        with open(filepath, 'w', encoding='utf-8') as f:
            for datapoint in dataset:
                if include_full_text:
                    json_line = json.dumps({
                        'source_text': datapoint.source_text,
                        'source_type': datapoint.source_type,
                        'full_text': datapoint.full_text,
                        'summary': datapoint.summary
                    }, ensure_ascii=False)
                else:
                    json_line = json.dumps({
                        'source_text': datapoint.source_text,
                        'source_type': datapoint.source_type,
                        'summary': datapoint.summary
                    }, ensure_ascii=False)
                f.write(json_line + '\n')
    output_dir = DATA_ROOT / "facetsum_long"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_aspect_summary_dataset(converted_dataset_train, str(output_dir / "train.jsonl"))
    save_aspect_summary_dataset(converted_dataset_validation, str(output_dir / "validation.jsonl"))
    save_aspect_summary_dataset(converted_dataset_test, str(output_dir / "test.jsonl"))

    # Print summary
    print(f"Saved {len(converted_dataset_train)} training examples, "
          f"{len(converted_dataset_validation)} validation examples, "
          f"and {len(converted_dataset_test)} test examples to disk.")
    print(f"Files saved to {output_dir}")

if __name__ == "__main__":
    # Load the FacetSum dataset
    # # Login using e.g. `huggingface-cli login` to access this dataset
    dataset = load_dataset("memray/FacetSum")
    # # # Alternatively, load from local disk if already saved
    # dataset = load_from_disk(DATA_ROOT / "facetsum" / "raw" / "facetsum")

    # # Inspect the dataset (optional)
    # inspect_facetsum_dataset(dataset)

    # # Save the original dataset locally (optional)
    # save_facetsum_dataset_locally(dataset)

    # Prepare and save the dataset in 2a2s format
    prepare_facetsum_data_for_2a2s(dataset, source_text_type='full_text')