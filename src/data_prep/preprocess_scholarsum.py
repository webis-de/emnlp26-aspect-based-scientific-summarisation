import json
import os
import random

def process_file(input_path, output_path, mode='w'):
    if mode not in {'w', 'a', 'x'}:
        raise ValueError(f"Unsupported file mode: {mode}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, mode, encoding='utf-8') as output_file:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if not isinstance(entry, dict):
                    continue
                
                # Get source text and summary
                source_text = entry.get('article')
                if 'human_aspect' not in entry and 'human' not in entry:
                    raise KeyError("Missing 'human_aspect' field in the dataset entry.")
                human_aspect = entry.get('human_aspect', None) or entry.get('human', None)
                assert human_aspect is not None, "human_aspect should not be None here"
                
                if source_text is not None and human_aspect is not None:
                    try:
                        # Summary should be a dict as JSON string
                        summary = json.loads(human_aspect)
                    except Exception:
                        # Fallback: the dictionary is indirectly provided with "key : value"
                        # Split the string human_aspect in four segments with separators: "background :", "methods :", "results :", "conclusions :"
                        summary = {
                            "background": "",
                            "method": "",
                            "result": "",
                            "conclusion": ""
                        }
                        segments_sets = [
                            ['background :', 'methods :', 'results :', 'conclusions :'],
                            ['background .', 'methods .', 'results .', 'conclusions .'],
                        ]
                        for segments in segments_sets:
                            current_aspect = None
                            current_text = []
                            for part in human_aspect.splitlines():
                                part = part.strip()
                                if any(part.lower().startswith(seg) for seg in segments):
                                    if current_aspect is not None:
                                        summary[current_aspect] = ' '.join(current_text).strip()
                                    for seg in segments:
                                        if part.lower().startswith(seg):
                                            current_aspect = seg[:-2].lower()  # remove " :"
                                            current_text = [part[len(seg):].strip()]
                                            break
                                else:
                                    if current_aspect is not None:
                                        current_text.append(part)
                            if current_aspect is not None:
                                summary[current_aspect] = ' '.join(current_text).strip()
                    # Now summary is a dict
                    
                    # Print dict keys for debugging
                    print("Summary keys:", summary.keys())
                    # Print lengths of each aspect for debugging
                    for k, v in summary.items():
                        print(f"  {k}: {len(v)} characters")

                    # Remove aspects with empty strings with less than 6 characters
                    summary = {k: v for k, v in summary.items() if len(v.strip()) > 6}

                    # Ensure all 4 aspects are present
                    aspects = ['background', 'method', 'result', 'conclusion']
                    summary_out = {aspect: summary.get(aspect, "") for aspect in aspects}
                    # Use id if present, else fallback to hash of abstract
                    doc_id = entry.get('id', None)
                    if not doc_id:
                        import hashlib
                        doc_id = hashlib.md5(source_text.encode('utf-8')).hexdigest()
                    output = {
                        'doc_id': doc_id,
                        'summary': summary_out,
                        'source_type': 'abstract',
                        'source_text': source_text
                    }
                    output_file.write(json.dumps(output, ensure_ascii=False) + '\n')


def split_processed_file(processed_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, shuffle=True):
    ratios_sum = train_ratio + val_ratio + test_ratio
    if ratios_sum <= 0 or abs(ratios_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0 and be positive")
    if not os.path.isfile(processed_path):
        raise FileNotFoundError(f"Processed file not found: {processed_path}")

    with open(processed_path, 'r', encoding='utf-8') as f:
        records = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(records)

    total = len(records)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    base_dir = os.path.dirname(processed_path)
    os.makedirs(base_dir, exist_ok=True)

    splits = [
        (records[:train_end], os.path.join(base_dir, 'train.jsonl')),
        (records[train_end:val_end], os.path.join(base_dir, 'val.jsonl')),
        (records[val_end:], os.path.join(base_dir, 'test.jsonl')),
    ]

    for split_records, path in splits:
        with open(path, 'w', encoding='utf-8') as out_f:
            for record in split_records:
                out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    # Print sizes of each split
    for split_name, (split_records, path) in zip(['train', 'val', 'test'], splits):
        print(f"{split_name} size: {len(split_records)} records")

if __name__ == '__main__':
    # Process arxiv
    print("Processing arxiv dataset...")
    inpput_file = 'data/scholarsum/raw/ScholarSum/dataset/arxiv.json'
    output_path = 'data/scholarsum/arxiv/processed.jsonl'
    process_file(inpput_file, output_path, mode='w')
    split_processed_file(output_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, shuffle=True)

    # # Process pubmed
    # print("Processing pubmed dataset...")
    # inpput_file = 'data/scholarsum/raw/ScholarSum/dataset/pubmed.json'
    # output_path = 'data/scholarsum/pubmed/processed.jsonl'
    # process_file(inpput_file, output_path, mode='w')
    # split_processed_file(output_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, shuffle=True)

    # # Combine both datasets into one file
    # print("Combining datasets...")
    # input_files = [
    #     'data/scholarsum/raw/ScholarSum/dataset/arxiv.json',
    #     'data/scholarsum/raw/ScholarSum/dataset/pubmed.json',
    # ]
    # output_path = 'data/scholarsum/combined/processed.jsonl'
    # mode = 'w'  # 'w' to overwrite, 'a' to append
    # # Use the requested mode for the first file; append for subsequent files to avoid unintended overwrites.
    # write_modes = [mode] + ['a'] * (len(input_files) - 1) if mode in {'w', 'x'} else [mode] * len(input_files)
    # for write_mode, input_path in zip(write_modes, input_files):
    #     process_file(input_path, output_path, write_mode)
    # # Split combined dataset
    # split_processed_file(output_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, shuffle=True)