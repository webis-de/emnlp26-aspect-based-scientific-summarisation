

import sys
import os
from paths import DATA_ROOT
import json
from collections import Counter, defaultdict

# Ensure parent src directory is in sys.path for import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_utils import load_summarization_dataset

def get_all_dataset_configs():
	# Define all dataset/split/context combinations to check
	configs = []
	# (dataset_name, context_size_type, type, splits)
	configs += [
		("facetsum", "short", "sampled", ["test"]),
		("facetsum", "long", "sampled", ["test"]),
		("aclsum", "short", "full", ["test"]),
		("aclsum", "long", "full", ["test"]),
		("pmc", "short", "sampled", ["test"]),
		("pmc", "long", "sampled", ["test"]),
	]
	return configs


def main():
	stats = defaultdict(Counter)
	configs = get_all_dataset_configs()
	for dataset_name, context_size_type, type_, splits in configs:
		for split in splits:
			try:
				data = load_summarization_dataset(
					split=split,
					dataset_name=dataset_name,
					type=type_,
					prompt_format="none",
					context_size_type=context_size_type,
				)
			except Exception as e:
				print(f"[WARN] Could not load {dataset_name} {context_size_type} {type_} {split}: {e}")
				continue
			for record in data:
				aspect = record.get("aspect_name")
				if aspect:
					stats[(dataset_name, context_size_type, type_, split)][aspect] += 1

	# Print summary
	print("\nAspect statistics by dataset, context, type, split:")
	for key, aspect_counter in stats.items():
		dataset_name, context_size_type, type_, split = key
		print(f"\nDataset: {dataset_name}, Context: {context_size_type}, Type: {type_}, Split: {split}")
		for aspect, count in aspect_counter.most_common():
			print(f"  {aspect}: {count}")

	# Save as JSON
	output = {}
	for key, aspect_counter in stats.items():
		dataset_name, context_size_type, type_, split = key
		key_str = f"{dataset_name}|{context_size_type}|{type_}|{split}"
		output[key_str] = dict(aspect_counter)

	out_dir = str(DATA_ROOT)
	os.makedirs(out_dir, exist_ok=True)
	out_path = os.path.join(out_dir, "aspect_stats.json")
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(output, f, indent=2, ensure_ascii=False)
	print(f"\nSaved aspect statistics to {out_path}")

if __name__ == "__main__":
	main()
