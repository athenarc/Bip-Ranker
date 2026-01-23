# Artifact Indicators Scripts

This directory contains scripts for calculating citation-based indicators for artifacts (datasets and software).

## indirect_citations.py

Enriches artifacts with indirect citation count from related publications.

### Overview

This script calculates indirect citation metrics:

**Indirect Citations**: Citations to publications associated with artifacts via DOIs
- Sums citation counts for each artifact's publications
- Counts each `openaire_id` only once to avoid double-counting

### Quick Start

#### Installation

```bash
pip install pandas
```

#### Sample Input Data

Sample tab-separated files are provided in `sample_data/` directory:

- `artifacts-to-publications_sample.csv` - Artifact publications with DOIs
- `publications_sample.csv` - Paper information with citation counts

**Note:** All files are tab-separated (TSV format), even though they use `.csv` extension.

#### Quick Run

From the `artifact_indicators` directory:

```bash
python indirect_citations.py \
    --artifacts-to-publications sample_data/artifacts-to-publications_sample.csv \
    --publications sample_data/publications_sample.csv \
    --output indirect_citations_output.csv
```

This will create:
- `enriched_output.csv` - Enriched artifacts with citations (tab-separated)

## mentions.py

Aggregates mention counts for artifacts (software and datasets) from tab-separated files (TSV format).

### Overview

This script counts how many papers mention each artifact:

**Mentions**: Papers that mention artifacts
- Counts unique papers mentioning each artifact
- Outputs mention count per artifact

### Quick Start

#### Sample Data

Sample tab-separated file is provided in `sample_data/` directory:

- `artifacts-to-mentions_sample.csv` - Papers mentioning artifacts

#### Quick Run

From the `artifact_indicators` directory:

```bash
python mentions.py \
    --artifacts-to-mentions sample_data/artifacts-to-mentions_sample.csv \
    --output mentions_output.csv
```

This will create:
- `enriched_output.csv` - Enriched artifacts with mention counts (tab-separated)