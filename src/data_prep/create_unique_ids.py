import argparse
from pathlib import Path
from paths import DATA_ROOT
import pandas as pd


DEFAULT_DATA_ROOT = DATA_ROOT


def build_paths(split: str, data_root: Path):
    facetsum = data_root / "facetsum"
    facetsum_long = data_root / "facetsum_long"
    pmc = data_root / "pmc" / "extracted" / "splits"
    aclsum = data_root / "aclsum"

    dataset_paths = {
        "facetsum_long": facetsum_long / f"{split}_sampled.jsonl",
        "facetsum": facetsum / f"{split}_sampled.jsonl",
        "pmc": pmc / f"{split}_sampled.jsonl",
        "aclsum": aclsum / f"{split}.jsonl", # aclsum does not have sampled split
    }

    output_paths = {
        name: path.parent / f"{split}_sampled_with_ids.jsonl" if "sampled" in path.name else path.parent / f"{split}_with_ids.jsonl"
        for name, path in dataset_paths.items()
    }

    return dataset_paths, output_paths


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_json(path, orient="records", lines=True)


def main():
    dataset_paths, output_paths = build_paths("test", DEFAULT_DATA_ROOT)

    datasets = {name: load_dataset(path) for name, path in dataset_paths.items()}

    for name in datasets:
        print(f"{name} size: {len(datasets[name])}")
    
    print("Creating unique IDs for each dataset...")

    sampled = {}
    for name, df in datasets.items():
        df = df.copy()
        if name == "facetsum":
            df["unique_id"] = [f"{name}_short_sampled_test_{i}" for i in range(len(df))]
        else:
            df["unique_id"] = [f"{name}_sampled_test_{i}" for i in range(len(df))]
        sampled[name] = df

    for name, df in sampled.items():
        output_path = output_paths[name]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output_path, orient="records", lines=True)
        print(f"Saved {name} with unique IDs to {output_path}")

if __name__ == "__main__":
    main()