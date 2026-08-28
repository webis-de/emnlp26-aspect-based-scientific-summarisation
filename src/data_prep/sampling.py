import argparse
from pathlib import Path
from paths import DATA_ROOT

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DATA_ROOT = DATA_ROOT


def parse_args():
    parser = argparse.ArgumentParser(description="Stratified sampling for FacetSum/PMC splits.")
    parser.add_argument("--split", default="test", help="Dataset split to sample (e.g., train, val, test).")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of rows to keep per dataset.")
    parser.add_argument("--column", default="summary", help="Column containing aspect dictionaries for stratification.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing dataset folders.",
    )
    return parser.parse_args()


def build_paths(split: str, data_root: Path):
    facetsum = data_root / "facetsum"
    facetsum_long = data_root / "facetsum_long"
    pmc = data_root / "pmc" / "extracted" / "splits"

    dataset_paths = {
        "facetsum_long": facetsum_long / f"{split}.jsonl",
        "facetsum": facetsum / f"{split}.jsonl",
        "pmc": pmc / f"{split}.jsonl",
    }

    output_paths = {
        name: path.parent / f"{split}_sampled.jsonl"
        for name, path in dataset_paths.items()
    }

    return dataset_paths, output_paths


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_json(path, orient="records", lines=True)


def get_stratified_sample(df: pd.DataFrame, sample_size: int = 1000, column: str = "summary") -> pd.DataFrame:
    if sample_size <= 0:
        raise ValueError("sample_size must be a positive integer.")
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataframe columns: {df.columns.tolist()}")
    if df.empty:
        raise ValueError("Dataframe is empty; cannot sample.")

    working_df = df.copy()
    working_df["_stratify_key"] = working_df[column].apply(
        lambda x: "|".join(sorted(x.keys())) if isinstance(x, dict) else "none"
    )

    counts = working_df["_stratify_key"].value_counts()
    rare_keys = counts[counts < 2].index
    working_df.loc[working_df["_stratify_key"].isin(rare_keys), "_stratify_key"] = "rare_combination"

    if sample_size >= len(working_df):
        return working_df.drop(columns=["_stratify_key"])

    sampled_df, _ = train_test_split(
        working_df,
        train_size=sample_size,
        stratify=working_df["_stratify_key"],
        random_state=42,
    )

    return sampled_df.drop(columns=["_stratify_key"])


def main():
    args = parse_args()
    dataset_paths, output_paths = build_paths(args.split, args.data_root)

    datasets = {name: load_dataset(path) for name, path in dataset_paths.items()}
    sampled = {
        name: get_stratified_sample(df, sample_size=args.sample_size, column=args.column)
        for name, df in datasets.items()
    }

    for name in datasets:
        print(f"New {name} size: {len(sampled[name])}")
    for name in datasets:
        print(
            f"{name} columns sampled: {sampled[name].columns.tolist()}, {name} columns before: {datasets[name].columns.tolist()}"
        )

    for name, df in sampled.items():
        output_path = output_paths[name]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output_path, orient="records", lines=True)


if __name__ == "__main__":
    main()