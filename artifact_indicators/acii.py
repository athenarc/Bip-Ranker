#!/usr/bin/env python3
"""
Compute Artefact Composite Impact Indicator (ACII) from ICC and IM scores.

ACII aggregates Indirect Citation Count (ICC) and In-text Mentions (IM)
using a weighted combination of min-max normalized scores:

    norm_icc = (icc - min(icc)) / (max(icc) - min(icc))
    norm_im  = (im  - min(im))  / (max(im)  - min(im))
    acii     = weight_icc * norm_icc + weight_im * norm_im

Each indicator is normalized to [0, 1] across all artifacts before weighting.
If all values for an indicator are equal, normalized scores are set to 0.

**Input Files Required (Tab-Separated):**

1. **icc**: Output from indirect_citations.py
   - Required columns: `artifact_id`, `indirect_citations`

2. **im**: Output from mentions.py
   - Required columns: `artifact_id`, `mention_count`

**Output (Tab-Separated):**

- Columns: `artifact_id`, `indirect_citations`, `mention_count`, `acii`
- One row per artifact, sorted by `acii` descending

**Usage:**
    python acii.py \\
        --icc <icc_output_csv> \\
        --im <im_output_csv> \\
        [--weight-icc 0.5] \\
        [--weight-im 0.5] \\
        [--output <output_csv>]
"""

import pandas as pd
import sys
import argparse


def min_max_normalize(series):
    """Normalize a numeric series to [0, 1]. Returns zeros if all values are equal."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.0, index=series.index)
    return (series - min_val) / (max_val - min_val)


def load_indicator_file(path, value_column):
    """Load an indicator file and validate required columns."""
    df = pd.read_csv(path, sep='\t')
    if 'artifact_id' not in df.columns or value_column not in df.columns:
        raise ValueError(f"File must have 'artifact_id' and '{value_column}' columns: {path}")
    return df[['artifact_id', value_column]].drop_duplicates('artifact_id')


def compute_acii(icc_df, im_df, weight_icc, weight_im):
    """
    Merge ICC and IM scores and compute ACII.

    Artifacts present in only one input file get 0 for the missing indicator.
    """
    merged = pd.merge(icc_df, im_df, on='artifact_id', how='outer')
    merged['indirect_citations'] = merged['indirect_citations'].fillna(0).astype(int)
    merged['mention_count'] = merged['mention_count'].fillna(0).astype(int)

    norm_icc = min_max_normalize(merged['indirect_citations'])
    norm_im = min_max_normalize(merged['mention_count'])

    merged['acii'] = weight_icc * norm_icc + weight_im * norm_im
    merged = merged.sort_values('acii', ascending=False)

    return merged


def main():
    parser = argparse.ArgumentParser(
        description='Compute Artefact Composite Impact Indicator (ACII) from ICC and IM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--icc', required=True,
                        help='Path to ICC output from indirect_citations.py (artifact_id, indirect_citations)')
    parser.add_argument('--im', required=True,
                        help='Path to IM output from mentions.py (artifact_id, mention_count)')
    parser.add_argument('--weight-icc', type=float, default=0.5,
                        help='Weight for normalized ICC in ACII (default: 0.5)')
    parser.add_argument('--weight-im', type=float, default=0.5,
                        help='Weight for normalized IM in ACII (default: 0.5)')
    parser.add_argument('--output', default='acii_output.csv',
                        help='Output path (default: acii_output.csv)')

    args = parser.parse_args()

    if args.weight_icc < 0 or args.weight_im < 0:
        print("ERROR: Weights must be non-negative")
        sys.exit(1)
    if args.weight_icc + args.weight_im == 0:
        print("ERROR: At least one weight must be greater than 0")
        sys.exit(1)

    print("=" * 80)
    print("ARTEFACT COMPOSITE IMPACT INDICATOR (ACII)")
    print("=" * 80)
    print(f"\nICC file: {args.icc}")
    print(f"IM file: {args.im}")
    print(f"Weights: ICC={args.weight_icc}, IM={args.weight_im}")
    print(f"Output: {args.output}")
    print("=" * 80)

    print("\nSTEP 1: Loading ICC and IM files...")
    try:
        icc_df = load_indicator_file(args.icc, 'indirect_citations')
        im_df = load_indicator_file(args.im, 'mention_count')
        print(f"  -> Loaded {len(icc_df)} artifacts with ICC")
        print(f"  -> Loaded {len(im_df)} artifacts with IM")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("\nSTEP 2: Computing ACII...")
    result = compute_acii(icc_df, im_df, args.weight_icc, args.weight_im)

    print("\nSTEP 3: Saving output...")
    result.to_csv(args.output, index=False, sep='\t')
    print(f"  -> Saved to: {args.output}")
    print(f"  -> Total artifacts: {len(result)}")


if __name__ == "__main__":
    main()
