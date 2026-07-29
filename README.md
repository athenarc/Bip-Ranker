# Bip-Ranker

PySpark scripts for impact-based ranking of scientific publications (as used in [Bip! Finder](https://bip.imsi.athenarc.gr/) and related datasets).

| Script | Method |
|--------|--------|
| `PageRank.py` | Classic PageRank on a citation graph |
| `AttRank.py` | Attention-based ranking (AttRank) |
| `CC.py` | Citation count and *n*-year citation impulse |
| `TAR.py` | Time-aware ranking (RAM / ECM) |
| `TopicClassesAndFWCI.py` | Topic-based impact classes and Field-Weighted Citation Impact (FWCI) |

Artefact-oriented indicators (ICC, IM, ACII) live under [`artifact_indicators/`](artifact_indicators/) and use pandas, not Spark. See that folder’s README for details.

---

## Requirements

- **Python** 3.8+
- **Java** 8 or 11+ (`java -version` should work)
- **PySpark** (see `requirements.txt`)

---

## Installation (venv)

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Confirm the install:

```bash
python -c "import pyspark; print(pyspark.__version__)"
```

---

## Graph input format

`PageRank.py`, `AttRank.py`, `CC.py`, and `TAR.py` expect a tab-separated citation graph. Each line:

```text
paper_id <tab> referenced_papers|num_references <tab> initial_score <tab> publication_year
```

- `referenced_papers`: comma-separated IDs of papers cited by this paper, or `0` if the paper has no references
- `num_references`: number of references (or `0`)
- `initial_score`: starting score used by iterative ranking methods (PageRank, AttRank, TAR/ECM)
- Paper IDs must not contain `,` or tab characters (replace them before ranking if needed)

A 50-paper example network is provided in [`sample-data/citation_graph.tsv`](sample-data/citation_graph.tsv). See [`sample-data/README.md`](sample-data/README.md) for the schema of the Topic/FWCI inputs.

---

## Quick demo

After installing dependencies in a venv, run all root scripts on the sample network with local Spark (no `spark-submit`, no HDFS):

```bash
python run_demo.py
```

Results are written under `sample-data/output/`. Before topic/FWCI, the demo joins PageRank, AttRank, CC, and 3-year CC into `sample-data/output/scores.tsv`. Optional filters:

```bash
python run_demo.py --only cc pagerank
python run_demo.py --skip ecm
```

---

## Run on the sample network (local Spark)

All examples below use **local** Spark (no HDFS). Prefer **absolute paths** for the graph and checkpoint directory. Ranking scripts write results into the **current working directory**, so the examples `cd` into `sample-data/output` first. Or use [`run_demo.py`](run_demo.py) to run everything at once.

```bash
# From the repo root, with the venv activated
cd /path/to/Bip-Ranker
source .venv/bin/activate

REPO="$(pwd)"
GRAPH="$REPO/sample-data/citation_graph.tsv"
CKPT="$REPO/sample-data/checkpoints"
OUT="$REPO/sample-data/output"
mkdir -p "$CKPT" "$OUT"
cd "$OUT"
```

### Citation count (`CC.py`)

```bash
python "$REPO/CC.py" "$GRAPH" 1
```

Optional *n*-year impulse (e.g. 3-year citations):

```bash
python "$REPO/CC.py" "$GRAPH" 1 3y 3
```

Outputs are named like `CC_*` or `3y_*`.

### PageRank (`PageRank.py`)

```bash
python "$REPO/PageRank.py" "$GRAPH" 0.85 1e-3 "$CKPT" 1 local
```

Arguments: `<input> <alpha> <convergence_error> <checkpoint_dir> [num_partitions] [checkpoint_mode]`.

Use `local` as the checkpoint mode for small graphs. Output directory: `PR_*_local_*`.

### AttRank (`AttRank.py`)

```bash
python "$REPO/AttRank.py" "$GRAPH" 0.2 0.5 0.3 0.6 2017 2014 1e-3 "$CKPT" 1 local
```

Arguments: `<input> <alpha> <beta> <gamma> <exponential_rho> <current_year> <start_year_for_attention> <convergence_error> <checkpoint_dir> [num_partitions] [checkpoint_mode]`.

### Time-aware ranking — RAM (`TAR.py`)

```bash
python "$REPO/TAR.py" "$GRAPH" 0.6 2017 RAM 1 "$CKPT" local
```

### Time-aware ranking — ECM (`TAR.py`)

```bash
python "$REPO/TAR.py" "$GRAPH" 0.6 2017 ECM 1 "$CKPT" local 0.5 1e-3
```

### Topic classes and FWCI (`TopicClassesAndFWCI.py`)

Uses score and concept tables (not the citation graph):

```bash
python "$REPO/TopicClassesAndFWCI.py" \
  --scores-file "$OUT/scores.tsv" \
  --concepts-file "$REPO/sample-data/concepts.tsv" \
  --output-dir "$OUT"
```

Build `scores.tsv` first by joining indicator outputs (as `run_demo.py` does), or provide an equivalent table with columns `pid`, `type`, `year`, `pagerank`, `attrank`, `cc`, `3y-cc`.

Writes under `--output-dir`:
- `topics/` — topic-based impact classes
- `FWCI/` — field-weighted citation impact
- `3-year_FWCI/` — 3-year FWCI

---

## Script reference

### Output of ranking scripts

`PageRank.py`, `AttRank.py`, `CC.py`, and `TAR.py` write tab-separated results with score, normalized score, and impact classes. Global ranking scripts assign five-point classes **C1–C5** based on score percentiles (top 0.01%, 0.1%, 1%, 10%, and the remaining 90%). Thresholds are also printed to stdout.

### Checkpointing

Iterative methods (`PageRank`, `AttRank`, `TAR`/ECM) checkpoint DataFrames each iteration to avoid Spark lineage blow-ups.

| Mode | When to use |
|------|-------------|
| `local` | Small graphs (e.g. the sample); faster; less durable |
| `dfs` | Large graphs; writes checkpoints to the configured checkpoint directory (HDFS or other configured FS) |

### Distributed / cluster runs

If the input path contains a URI scheme (`hdfs://`, `s3a://`, …), scripts switch to distributed mode. Example:

```bash
spark-submit --executor-memory 7G --executor-cores 4 --driver-memory 7G \
  PageRank.py hdfs:///user/<user>/graph_dir 0.85 1e-6 hdfs:///user/<user>/checkpoints 7680 dfs
```

Tune partitions and memory for your cluster. Prefer a directory of partitioned graph files over a single large file when possible.

---

## Citation

If you use these scripts for RAM / PageRank / CC, please cite:

> Kanellos I, Vergoulis T, Sacharidis D, Dalamagas T, Vassiliou Y. Impact-based ranking of scientific publications: a survey and experimental evaluation. IEEE Transactions on Knowledge and Data Engineering. 2019 Sep 13;33(4):1567-84.

> Vergoulis T, Chatzopoulos S, Kanellos I, Deligiannis P, Tryfonopoulos C, Dalamagas T. Bip! finder: Facilitating scientific literature search by exploiting impact-based ranking. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management 2019 Nov 3 (pp. 2937-2940).

> Vergoulis T, Kanellos I, Atzori C, Mannocci A, Chatzopoulos S, Bruzzo SL, Manola N, Manghi P. Bip! db: A dataset of impact measures for scientific publications. In Companion Proceedings of the Web Conference 2021 2021 Apr 19 (pp. 456-460).

Also cite the original works for the methods you implement or compare against.

For AttRank:

> Kanellos I, Vergoulis T, Sacharidis D, Dalamagas T, Vassiliou Y. Ranking papers by their short-term scientific impact. In 2021 IEEE 37th International Conference on Data Engineering (ICDE) 2021 Apr 19 (pp. 1997-2002). IEEE.

## Acknowledgments

This work was supported by the European Union's Horizon Europe research and innovation programme under grant agreement No. 101058573 ([SciLake](https://scilake.eu/)).
