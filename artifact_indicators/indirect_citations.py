#!/usr/bin/env python3
"""
Enrich artifacts (software and datasets) with indirect citations.

This script calculates indirect citation metrics for artifacts by:
- Reading publications associated with artifacts via DOIs
- Matching DOIs with paper information (doi, paper_id, openaire_id, citation_count)
- Summing citation counts for each artifact's publications, counting each openaire_id only once

**Input Files Required (Tab-Separated):**

1. **artifacts-to-publications**: Publications associated with artifacts
   - Required columns: `artifact_id`, `doi`
   - Optional: other artifact metadata columns
   
2. **publications**: Paper information with DOI mapping and citation counts
   - Required columns: `doi`, `paper_id`, `openaire_id`, `citation_count`

**Output Files (Tab-Separated):**

1. **output**: Unique artifacts file with citation counts
   - Columns: `artifact_id`, `indirect_citations`
   - One row per unique artifact, sorted by `indirect_citations` descending

**Note:** All input and output files must be tab-separated (TSV format).

**Usage:**
    python indirect_citations.py \\
        --artifacts-to-publications <artifacts_to_publications_csv> \\
        --publications <publications_csv> \\
        [--output <output_csv>]

**Example:**
    python indirect_citations.py \\
        --artifacts-to-publications artifacts_to_publications.csv \\
        --publications publications.csv \\
        --output enriched_output.csv

**Note:** All files are tab-separated (TSV format), even though they use .csv extension.
"""

import pandas as pd
import sys
import argparse
from collections import defaultdict


def load_publications_info(publications_csv):
    """
    Load paper information from CSV.
    
    Creates lookup dictionaries for DOI -> paper info and computes
    openaire_id uniqueness (whether each openaire_id appears only once).
    
    Args:
        publications_csv: Path to CSV with columns: doi, paper_id, openaire_id, citation_count
        
    Returns:
        tuple: (paper_info_dict, openaire_uniqueness_dict)
        - paper_info_dict: {doi: {'paper_id': ..., 'openaire_id': ..., 'citation_count': ...}}
        - openaire_uniqueness_dict: {openaire_id: True/False}
    """
    print(f"  -> Reading publications file: {publications_csv}")
    publications_df = pd.read_csv(publications_csv, sep='\t')
    print(f"  -> Loaded {len(publications_df)} publications from file")
    
    # Validate required columns
    required_cols = ['doi', 'paper_id', 'openaire_id', 'citation_count']
    missing_cols = [col for col in required_cols if col not in publications_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in publications CSV: {missing_cols}")
    
    # Create DOI -> paper info dictionary
    paper_info = {}
    for _, row in publications_df.iterrows():
        doi = str(row['doi']).strip() if pd.notna(row['doi']) else None
        if doi:
            paper_info[doi] = {
                'paper_id': row['paper_id'],
                'openaire_id': str(row['openaire_id']) if pd.notna(row['openaire_id']) else None,
                'citation_count': int(row['citation_count']) if pd.notna(row['citation_count']) else 0
            }
    
    print(f"  -> Created lookup for {len(paper_info)} unique DOIs")
    
    # Compute openaire_id uniqueness
    # An openaire_id is unique if it appears only once in the publications CSV
    openaire_counts = publications_df['openaire_id'].value_counts()
    openaire_uniqueness = {
        str(openaire_id): (count == 1) 
        for openaire_id, count in openaire_counts.items()
        if pd.notna(openaire_id)
    }
    
    unique_count = sum(1 for is_unique in openaire_uniqueness.values() if is_unique)
    print(f"  -> {unique_count} out of {len(openaire_uniqueness)} openaire_ids are unique")
    
    return paper_info, openaire_uniqueness


def calculate_indirect_citations(publications_df, paper_info, openaire_uniqueness):
    """
    Calculate indirect citations for artifacts based on their publications.
    
    For each artifact, finds all associated DOIs from publications, matches them
    with paper information, and sums citation counts. Each openaire_id is
    counted only once per artifact to avoid double-counting.
    
    Args:
        publications_df: DataFrame with columns: artifact_id, doi
        paper_info: Dict {doi: {paper_id, openaire_id, citation_count}}
        openaire_uniqueness: Dict {openaire_id: True/False}
        
    Returns:
        dict: {artifact_id: total_indirect_citations}
    """
    print("\nCalculating indirect citations...")
    
    indirect_citation_totals = defaultdict(int)
    
    processed = 0
    matched = 0
    unique_count = 0
    
    # Group by artifact_id to process all publications for an artifact together
    for artifact_id, group in publications_df.groupby('artifact_id'):
        artifact_openaire_ids = set()  # Track openaire_ids for this artifact
        
        for _, row in group.iterrows():
            if pd.isna(row['doi']) or str(row['doi']).strip() == '':
                continue
            
            doi = str(row['doi']).strip()
            processed += 1
            
            if doi in paper_info:
                paper_data = paper_info[doi]
                matched += 1
                openaire_id = paper_data['openaire_id']
                
                # Only count each openaire_id once per artifact
                if openaire_id and openaire_id not in artifact_openaire_ids:
                    artifact_openaire_ids.add(openaire_id)
                    unique_count += 1
                    indirect_citation_totals[artifact_id] += paper_data['citation_count']
        
        if processed % 1000 == 0:
            print(f"  -> Processed {processed} DOIs, matched {matched}, unique: {unique_count}")
    
    print(f"  -> Processed {processed} DOIs, matched {matched}, unique openaire_ids: {unique_count}")
    print(f"  -> Found indirect citations for {len(indirect_citation_totals)} artifacts")
    
    return indirect_citation_totals


def main():
    """Main function to orchestrate the enrichment process."""
    parser = argparse.ArgumentParser(
        description='Enrich artifacts with indirect citations from CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--artifacts-to-publications', required=True,
                        help='Path to artifacts-to-publications CSV (columns: artifact_id, doi)')
    parser.add_argument('--publications', required=True,
                        help='Path to publications CSV (columns: doi, paper_id, openaire_id, citation_count)')
    parser.add_argument('--output', default='enriched_output.csv',
                        help='Output path for enriched tab-separated file (default: enriched_output.csv)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ARTIFACTS INDIRECT CITATION ENRICHMENT SCRIPT")
    print("=" * 80)
    print(f"\nArtifacts-to-publications file: {args.artifacts_to_publications}")
    print(f"Publications file: {args.publications}")
    print(f"Output: {args.output}")
    print("=" * 80)
    
    # STEP 1: Load publications information
    print("\nSTEP 1: Loading publications information...")
    try:
        paper_info, openaire_uniqueness = load_publications_info(args.publications)
    except Exception as e:
        print(f"ERROR: Failed to load publications CSV: {e}")
        sys.exit(1)
    
    # STEP 2: Load artifacts-to-publications file
    print("\nSTEP 2: Loading artifacts-to-publications file...")
    try:
        artifacts_publications_df = pd.read_csv(args.artifacts_to_publications, sep='\t')
        print(f"  -> Loaded {len(artifacts_publications_df)} publication entries")
        
        if 'artifact_id' not in artifacts_publications_df.columns or 'doi' not in artifacts_publications_df.columns:
            raise ValueError("Artifacts-to-publications file must have 'artifact_id' and 'doi' columns")
    except Exception as e:
        print(f"ERROR: Failed to load artifacts-to-publications file: {e}")
        sys.exit(1)
    
    # STEP 3: Calculate indirect citations
    print("\nSTEP 3: Processing indirect citations...")
    indirect_citation_totals = calculate_indirect_citations(
        artifacts_publications_df, paper_info, openaire_uniqueness
    )
    
    # STEP 4: Create enriched output with unique artifacts
    print("\nSTEP 4: Creating enriched output...")
    
    # Create output DataFrame with unique artifacts
    unique_artifacts = pd.DataFrame({
        'artifact_id': list(indirect_citation_totals.keys()),
        'indirect_citations': list(indirect_citation_totals.values())
    })
    
    # Sort by indirect_citations descending
    unique_artifacts = unique_artifacts.sort_values('indirect_citations', ascending=False, na_position='last')
    
    # Save enriched CSV
    print(f"\nSTEP 5: Saving enriched output file...")
    unique_artifacts.to_csv(args.output, index=False, sep='\t')
    print(f"  -> Saved to: {args.output}")
    print(f"  -> Total unique artifacts: {len(unique_artifacts)}")


if __name__ == "__main__":
    main()
