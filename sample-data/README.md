# Sample data

Toy inputs for running the root ranking scripts locally.

## `citation_graph.tsv`

Citation network with **50 papers** (years 2010–2017) in the graph format expected by `PageRank.py`, `AttRank.py`, `CC.py`, and `TAR.py`.

Each line is tab-separated:

```text
paper_id <tab> referenced_papers|num_references <tab> initial_score <tab> publication_year
```

- `referenced_papers`: comma-separated IDs of papers cited by this paper, or `0` if none
- `num_references`: count of those references (or `0`)
- `initial_score`: starting score for iterative methods (here `1/50 = 0.02`)

Every paper has at least one inbound citation and at least one citation within three years of publication, so demo CC / 3-year CC / FWCI values are positive.

## `scores.tsv` and `concepts.tsv`

- `concepts.tsv` (no header): `paper_id`, `concept_id`, `confidence` — toy topic mappings for the 50 papers (`paper_00` … `paper_49`).
- `scores.tsv` is **not** hand-maintained for the demo. `run_demo.py` builds `output/scores.tsv` from PageRank, AttRank, CC, and 3-year CC outputs before running `TopicClassesAndFWCI.py`.

## Directories

- `checkpoints/` — suggested local checkpoint directory for iterative methods
- `output/` — suggested output directory for `TopicClassesAndFWCI.py`
