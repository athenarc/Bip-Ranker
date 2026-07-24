# Sample data

Toy inputs for running the root ranking scripts locally.

## `citation_graph.tsv`

Small citation network (8 papers, years 2015–2020) in the graph format expected by `PageRank.py`, `AttRank.py`, `CC.py`, and `TAR.py`.

Each line is tab-separated:

```text
paper_id <tab> referenced_papers|num_references <tab> initial_score <tab> publication_year
```

- `referenced_papers`: comma-separated IDs of papers cited by this paper, or `0` if none
- `num_references`: count of those references (or `0`)
- `initial_score`: starting score for iterative methods (here `1/8 = 0.125`)

| Paper   | Year | Cites              |
|---------|------|--------------------|
| paper_a | 2015 | (none)             |
| paper_b | 2016 | paper_a            |
| paper_c | 2017 | paper_a, paper_b   |
| paper_d | 2018 | paper_b, paper_c   |
| paper_e | 2019 | paper_a, paper_d   |
| paper_f | 2020 | paper_c, paper_e   |
| paper_g | 2018 | paper_a            |
| paper_h | 2019 | paper_g, paper_b   |

## `scores.tsv` and `concepts.tsv`

Inputs for `TopicClassesAndFWCI.py`:

- `scores.tsv` (with header): `openaire_id`, `pid`, `type`, `year`, `pagerank`, `attrank`, `cc`, `3y-cc`
- `concepts.tsv` (no header): `doi/pid`, `concept_id`, `confidence`

## Directories

- `checkpoints/` — suggested local checkpoint directory for iterative methods
- `output/` — suggested output directory for `TopicClassesAndFWCI.py`
