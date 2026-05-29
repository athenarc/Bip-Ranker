# Artifact Indicators Scripts

Scripts for artefact-oriented indicators introduced by ARC:

| Indicator | Script | Output column |
|-----------|--------|---------------|
| **ICC** — Indirect Citation Count | `indirect_citations.py` | `indirect_citations` |
| **IM** — In-text Mentions | `mentions.py` | `mention_count` |
| **ACII** — Artefact Composite Impact Indicator | `acii.py` | `acii` |

All input and output files are tab-separated (TSV format), even when using a `.csv` extension.

## Installation

```bash
pip install pandas
```

## indirect_citations.py (ICC)

Sums citation counts from publications linked to each artifact via DOIs. Each `openaire_id` is counted only once per artifact.

**Inputs:**
- `artifacts-to-publications` — columns: `artifact_id`, `doi`
- `publications` — columns: `doi`, `paper_id`, `openaire_id`, `citation_count`

**Output:** `artifact_id`, `indirect_citations`

```bash
python indirect_citations.py \
    --artifacts-to-publications sample_data/artifacts-to-publications_sample.csv \
    --publications sample_data/publications_sample.csv \
    --output icc_output.csv
```

## mentions.py (IM)

Counts unique papers that mention each artifact.

**Input:**
- `artifacts-to-mentions` — columns: `artifact_id`, `paper_id`

**Output:** `artifact_id`, `mention_count`

```bash
python mentions.py \
    --artifacts-to-mentions sample_data/artifacts-to-mentions_sample.csv \
    --output im_output.csv
```

## acii.py (ACII)

Combines normalized ICC and IM with configurable weights (default 0.5 / 0.5):

```
norm_icc = (icc - min) / (max - min)
norm_im  = (im  - min) / (max - min)
acii     = weight_icc * norm_icc + weight_im * norm_im
```

**Inputs:**
- `--icc` — output from `indirect_citations.py`
- `--im` — output from `mentions.py`

**Output:** `artifact_id`, `indirect_citations`, `mention_count`, `acii`

```bash
python acii.py \
    --icc icc_output.csv \
    --im im_output.csv \
    --weight-icc 0.5 \
    --weight-im 0.5 \
    --output acii_output.csv
```

### Full pipeline (sample data)

```bash
python indirect_citations.py \
    --artifacts-to-publications sample_data/artifacts-to-publications_sample.csv \
    --publications sample_data/publications_sample.csv \
    --output icc_output.csv

python mentions.py \
    --artifacts-to-mentions sample_data/artifacts-to-mentions_sample.csv \
    --output im_output.csv

python acii.py \
    --icc icc_output.csv \
    --im im_output.csv \
    --output acii_output.csv
```
