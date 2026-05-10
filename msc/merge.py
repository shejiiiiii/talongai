"""
merge_csv.py
------------
Appends one or more eggplant spectral data CSV files into a single output file.

Usage:
    # Append one file into another
    python merge_csv.py eggplant_spectral_data_v5.csv eggplant_spectral_data_v6.csv

    # Merge many files into a new output file
    python merge_csv.py file1.csv file2.csv file3.csv --output merged_dataset.csv

    # Merge with duplicate removal (by Label + Eggplant_ID + Timestamp)
    python merge_csv.py file1.csv file2.csv --dedupe
"""

import argparse
import pandas as pd
import sys
from pathlib import Path


EXPECTED_COLUMNS = [
    "Label", "Eggplant_ID", "Timestamp",
    *[f"Ch_{i}" for i in range(1, 19)]
]


def load_and_validate(filepath: str) -> pd.DataFrame:
    """Load a CSV and verify it has the expected spectral columns."""
    path = Path(filepath)
    if not path.exists():
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)

    df = pd.read_csv(filepath)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        print(f"[ERROR] '{filepath}' is missing columns: {missing}")
        sys.exit(1)

    # Keep only the expected columns, in the correct order
    df = df[EXPECTED_COLUMNS]

    # Normalise Label capitalisation (e.g. "healthy" -> "Healthy")
    df["Label"] = df["Label"].str.strip().str.title()

    print(f"  Loaded  {len(df):>5} rows  |  {filepath}")
    return df


def merge(input_files: list[str], output_file: str, dedupe: bool) -> None:
    print(f"\n{'='*55}")
    print(f"  Merging {len(input_files)} file(s) -> {output_file}")
    print(f"{'='*55}\n")

    frames = [load_and_validate(f) for f in input_files]

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)

    if dedupe:
        combined.drop_duplicates(
            subset=["Label", "Eggplant_ID", "Timestamp"],
            keep="first",
            inplace=True
        )
        removed = before - len(combined)
        print(f"\n  Duplicate rows removed : {removed}")

    # Drop rows where any spectral channel is exactly 0 (bad readings)
    ch_cols = [f"Ch_{i}" for i in range(1, 19)]
    zero_mask = (combined[ch_cols] == 0.0).any(axis=1)
    n_zero = zero_mask.sum()
    combined = combined[~zero_mask].reset_index(drop=True)
    if n_zero:
        print(f"  Zero-value rows dropped: {n_zero}")

    combined.to_csv(output_file, index=False)

    # Summary
    print(f"\n{'='*55}")
    print(f"  Output file : {output_file}")
    print(f"  Total rows  : {len(combined)}")
    print()
    for label, group in combined.groupby("Label"):
        ids = group["Eggplant_ID"].str.replace(r"_S\d$", "", regex=True).nunique()
        print(f"    {label:<10} {len(group):>5} scans  |  {ids} eggplants")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge eggplant spectral data CSV files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Two or more CSV files to merge (first file is the base)."
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help=(
            "Output filename. "
            "Defaults to the first input file (appends in-place)."
        )
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Remove duplicate rows matched on Label + Eggplant_ID + Timestamp."
    )

    args = parser.parse_args()

    if len(args.inputs) < 2:
        print("[ERROR] Provide at least two input files.")
        sys.exit(1)

    output = args.output or args.inputs[0]
    merge(args.inputs, output, args.dedupe)
