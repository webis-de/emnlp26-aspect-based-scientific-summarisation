#!/usr/bin/env python3
"""Filter specific fields from a JSONL file into a new JSONL file."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep only selected fields from each JSONL record.")
    parser.add_argument("input", type=Path, help="Path to input JSONL file")
    parser.add_argument("output", type=Path, help="Path to output JSONL file")
    parser.add_argument(
        "fields",
        nargs="+",
        help="Field names to keep in each record (space separated)",
    )
    return parser.parse_args(list(argv))


def filter_jsonl(input_path: Path, output_path: Path, fields: List[str]) -> None:
    missing_counts: Counter[str] = Counter()
    total = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                sys.stderr.write(f"Skipping invalid JSON on line {line_no}\n")
                continue

            total += 1
            filtered = {}
            for field in fields:
                if field in record:
                    filtered[field] = record[field]
                else:
                    missing_counts[field] += 1
            json.dump(filtered, fout, ensure_ascii=False)
            fout.write("\n")

    if missing_counts:
        sys.stderr.write(
            "Missing field counts (records where field was absent):\n"
        )
        for field in fields:
            count = missing_counts.get(field, 0)
            sys.stderr.write(f"  {field}: {count} of {total}\n")


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    filter_jsonl(args.input, args.output, args.fields)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
