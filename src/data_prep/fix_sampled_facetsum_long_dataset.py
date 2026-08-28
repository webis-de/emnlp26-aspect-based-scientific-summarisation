from pathlib import Path
from paths import DATA_ROOT
import pandas as pd
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# from preprocess_facetsum import just_intro_and_conclusion


DEFAULT_DATA_ROOT = DATA_ROOT


def build_paths(data_root: Path):
    split = "test"
    
    facetsum = data_root / "facetsum"
    facetsum_long = data_root / "facetsum_long"

    dataset_paths = dict()
    
    dataset_paths['sampled'] = {
        "facetsum_long": facetsum_long / f"{split}_sampled_with_ids.jsonl",
        "facetsum": facetsum / f"{split}_sampled_with_ids.jsonl",
    }
    dataset_paths['full'] = {
        "facetsum_long": facetsum_long / f"{split}.jsonl",
        "facetsum": facetsum / f"{split}.jsonl",
    }

    output_path = data_root / "facetsum_long" / f"{split}_fixed_sampled_with_ids.jsonl"

    return dataset_paths, output_path


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_json(path, orient="records", lines=True)

def check_overlap(origin_df: pd.DataFrame, target_df: pd.DataFrame) -> bool:
    n_multiple_matches: int = 0
    n_no_matches: int = 0
    for datapoint in origin_df.to_dict(orient="records"):
        summary_dict = datapoint["summary"]
        matching_rows = target_df[target_df["summary"] == summary_dict]
        if len(matching_rows) == 0:
            n_no_matches += 1
            continue
        elif len(matching_rows) > 1:
            n_multiple_matches += 1
            continue
    return n_multiple_matches, n_no_matches

def find_matching_rows(origin_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    matching_rows_list = []
    for datapoint in origin_df.to_dict(orient="records"):
        summary_dict = datapoint["summary"]
        matching_rows = target_df[target_df["summary"] == summary_dict]
        if len(matching_rows) == 0:
            continue
        elif len(matching_rows) > 1:
            for _, row in matching_rows.iterrows():
                text_snippets_to_check = []
                for key in row['source_text'].keys():
                    key_lower = key.lower()
                    if 'intro' in key_lower or 'purpose' in key_lower: # Table 4 in FacetSum paper
                        text_snippets_to_check.append(row['source_text'][key])
                    if 'conclu' in key_lower or 'future' in key_lower: # Table 4 in FacetSum paper
                        text_snippets_to_check.append(row['source_text'][key])
                if all(snippet in datapoint['source_text'] for snippet in text_snippets_to_check):
                    matching_rows_list.append(row)
                    break
            continue
        else:
            matching_rows_list.append(matching_rows.iloc[0])
    return pd.DataFrame(matching_rows_list)

def main():
    dataset_paths, output_path = build_paths(DEFAULT_DATA_ROOT)

    print("Sampled facetsum long")
    sampled_facetsum_long = load_dataset(dataset_paths['sampled']['facetsum_long'])
    print(sampled_facetsum_long[:2])
    print("Full facetsum long")
    full_facetsum_long = load_dataset(dataset_paths['full']['facetsum_long'])
    print(full_facetsum_long[:2])

    print("Sampled facetsum short")
    sampled_facetsum_short = load_dataset(dataset_paths['sampled']['facetsum'])
    print(sampled_facetsum_short[:2])
    print("Full facetsum short")
    full_facetsum_short = load_dataset(dataset_paths['full']['facetsum'])
    print(full_facetsum_short[:2])

    print("Checking overlap from sampled facetsum long and full facetsum long...") 
    # No matches should be 0 since the sampled dataset is supposed to be a subset of the full dataset. 
    # Multiple matches can be > 0 if there are duplicate summaries in the full dataset, but should not be too high.
    n_multiple_matches, n_no_matches = check_overlap(sampled_facetsum_long, full_facetsum_long)
    print(f"Number of summaries in sampled facetsum long with multiple matches in full facetsum long: {n_multiple_matches}")
    print(f"Number of summaries in sampled facetsum long with no matches in full facetsum long: {n_no_matches}")

    print("Checking overlap from sampled facetsum short and full facetsum short...")
    n_multiple_matches, n_no_matches = check_overlap(sampled_facetsum_short, full_facetsum_short)
    # No matches should be 0 since the sampled dataset is supposed to be a subset of the full dataset. 
    # Multiple matches can be > 0 if there are duplicate summaries in the full dataset, but should not be too high.
    print(f"Number of summaries in sampled facetsum short with multiple matches in full facetsum short: {n_multiple_matches}")
    print(f"Number of summaries in sampled facetsum short with no matches in full facetsum short: {n_no_matches}")

    print("Checking overlap from full facetsum short and full facetsum long...")
    n_multiple_matches, n_no_matches = check_overlap(full_facetsum_short, full_facetsum_long)
    # No matches should be 0 since the full facetsum short dataset is supposed to be a subset of the full facetsum long dataset.
    # Multiple matches can be > 0 if there are duplicate summaries in the full facetsum long dataset, but should not be too high.
    print(f"Number of summaries in full facetsum short with multiple matches in full facetsum long: {n_multiple_matches}")
    print(f"Number of summaries in full facetsum short with no matches in full facetsum long: {n_no_matches}")

    print("Checking overlap from sampled facetsum short and full facetsum long...")
    n_multiple_matches, n_no_matches = check_overlap(sampled_facetsum_short, full_facetsum_long)
    # No matches should be 0 since the sampled facetsum short dataset is supposed to be a subset of the full facetsum long dataset.
    # Multiple matches can be > 0 if there are duplicate summaries in the full facetsum long dataset, but should not be too high.
    print(f"Number of summaries in sampled facetsum short with multiple matches in full facetsum long: {n_multiple_matches}")
    print(f"Number of summaries in sampled facetsum short with no matches in full facetsum long: {n_no_matches}")

    # Find the matching rows in the full facetsum long dataset for the sampled facetsum short dataset. This will be the fixed sampled facetsum long dataset that we will use for evaluation.
    print("Finding matching rows in full facetsum long for sampled facetsum short...")
    print("Size of sampled facetsum short: ", len(sampled_facetsum_short))
    fixed_sampled_facetsum_long = find_matching_rows(sampled_facetsum_short, full_facetsum_long)
    print(f"Size of fixed sampled facetsum long: {len(fixed_sampled_facetsum_long)}")

    # Add unique ids to the fixed sampled facetsum long dataset. The unique ids will be in the format "facetsum_long_sampled_test_{i}" where i is the index of the row in the fixed sampled facetsum long dataset.
    fixed_sampled_facetsum_long = fixed_sampled_facetsum_long.copy()
    fixed_sampled_facetsum_long["unique_id"] = [f"facetsum_long_sampled_test_{i}" for i in range(len(fixed_sampled_facetsum_long))]

    print(f"Saving fixed sampled facetsum long to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_sampled_facetsum_long.to_json(output_path, orient="records", lines=True)

if __name__ == "__main__":
    main()