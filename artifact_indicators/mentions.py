#!/usr/bin/env python3
"""
Aggregate mentions for artifacts (software and datasets).

This script counts how many papers mention each artifact by:
- Reading artifact-to-paper mentions (artifact_id, paper_id)
- Aggregating mention counts per artifact
- Outputting enriched file with mention counts

**Input Files Required (Tab-Separated):**

1. **artifacts-to-mentions**: Papers mentioning artifacts
   - Required columns: `artifact_id`, `paper_id`
   - Optional: other metadata columns (preserved in output)

**Output Files (Tab-Separated):**

1. **output**: Unique artifacts file with mention count column
   - One row per unique artifact
   - Columns: `artifact_id`, `mention_count`, plus any other metadata columns from input

**Note:** All input and output files must be tab-separated (TSV format).

**Usage:**
    python mentions.py \\
        --artifacts-to-mentions <artifacts_to_mentions_csv> \\
        [--output <output_csv>]

**Example:**
    python mentions.py \\
        --artifacts-to-mentions artifacts_to_mentions.csv \\
        --output enriched_output.csv

**Note:** All files are tab-separated (TSV format), even though they use .csv extension.
"""

import pandas as pd
import sys
import argparse


def aggregate_mentions(mentions_df):
    """
    Aggregate mention counts per artifact.
    
    Counts how many unique papers mention each artifact.
    
    Args:
        mentions_df: DataFrame with columns: artifact_id, paper_id
        
    Returns:
        dict: {artifact_id: mention_count}
    """
    print("\nAggregating mentions...")
    
    # Count unique paper_ids per artifact_id
    mention_counts = mentions_df.groupby('artifact_id')['paper_id'].nunique().to_dict()
    
    print(f"  -> Found mentions for {len(mention_counts)} artifacts")
    print(f"  -> Total mention records: {len(mentions_df)}")
    
    return mention_counts


def main():
    """Main function to orchestrate the aggregation process."""
    parser = argparse.ArgumentParser(
        description='Aggregate mentions for artifacts from CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--artifacts-to-mentions', required=True,
                        help='Path to artifacts-to-mentions CSV (columns: artifact_id, paper_id)')
    parser.add_argument('--output', default='enriched_output.csv',
                        help='Output path for enriched tab-separated file (default: enriched_output.csv)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ARTIFACTS MENTIONS AGGREGATION SCRIPT")
    print("=" * 80)
    print(f"\nArtifacts-to-mentions file: {args.artifacts_to_mentions}")
    print(f"Output: {args.output}")
    print("=" * 80)
    
    # STEP 1: Load artifacts-to-mentions file
    print("\nSTEP 1: Loading artifacts-to-mentions file...")
    try:
        mentions_df = pd.read_csv(args.artifacts_to_mentions, sep='\t')
        print(f"  -> Loaded {len(mentions_df)} mention entries")
        
        if 'artifact_id' not in mentions_df.columns or 'paper_id' not in mentions_df.columns:
            raise ValueError("Artifacts-to-mentions file must have 'artifact_id' and 'paper_id' columns")
    except Exception as e:
        print(f"ERROR: Failed to load artifacts-to-mentions file: {e}")
        sys.exit(1)
    
    # STEP 2: Aggregate mentions
    print("\nSTEP 2: Processing mentions...")
    mention_counts = aggregate_mentions(mentions_df)
    
    # STEP 3: Create enriched output with unique artifacts
    print("\nSTEP 3: Creating enriched output...")
    
    # Get unique artifacts - take first row for each artifact_id to preserve metadata
    unique_artifacts = mentions_df.groupby('artifact_id').first().reset_index()
    
    # Add mention counts
    unique_artifacts['mention_count'] = unique_artifacts['artifact_id'].map(mention_counts)
    
    # Save enriched CSV
    print(f"\nSTEP 4: Saving enriched output file...")
    unique_artifacts.to_csv(args.output, index=False, sep='\t')
    print(f"  -> Saved to: {args.output}")
    print(f"  -> Total unique artifacts: {len(unique_artifacts)}")


if __name__ == "__main__":
    main()
