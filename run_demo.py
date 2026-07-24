#!/usr/bin/env python3
"""
Run the root ranking scripts on sample-data/ using local Spark (no HDFS).

Usage (from the repo root, with the venv activated):

    python run_demo.py
    python run_demo.py --only cc pagerank
    python run_demo.py --skip ecm

Requires: Java on PATH, and ``pip install -r requirements.txt``.

Before TopicClassesAndFWCI, the demo builds ``sample-data/output/scores.tsv`` by
joining PageRank, AttRank, CC, and 3-year CC outputs with years from the graph.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SAMPLE = REPO / "sample-data"
GRAPH = SAMPLE / "citation_graph.tsv"
CONCEPTS = SAMPLE / "concepts.tsv"
CKPT = SAMPLE / "checkpoints"
OUT = SAMPLE / "output"
SCORES_OUT = OUT / "scores.tsv"

# Demo parameters tuned for the sample graph (fast convergence).
ALPHA = "0.85"
TOL = "1e-3"
PARTITIONS = "1"
CHECKPOINT_MODE = "local"
ATTRANK_ARGS = ["0.2", "0.5", "0.3", "0.6", "2017", "2014", TOL]
TAR_GAMMA = "0.6"
TAR_YEAR = "2017"
ECM_ALPHA = "0.5"

# Steps that produce the indicator outputs joined into scores.tsv
SCORE_STEPS = ("cc", "cc3y", "pagerank", "attrank")
ALL_STEPS = SCORE_STEPS + ("ram", "ecm", "fwci")


def check_environment():
    try:
        import pyspark  # noqa: F401
    except ImportError:
        print(
            "ERROR: pyspark is not installed.\n"
            "  python3 -m venv .venv && source .venv/bin/activate\n"
            "  pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    for path in (GRAPH, CONCEPTS):
        if not path.is_file():
            print("ERROR: missing sample file: {}".format(path), file=sys.stderr)
            sys.exit(1)


def run_step(name, argv):
    print("\n" + "=" * 60)
    print("Running: {}".format(name))
    print("  " + " ".join(argv))
    print("=" * 60 + "\n")
    subprocess.check_call(argv, cwd=str(OUT))


def read_spark_tsv_dir(directory):
    """Load a Spark CSV/TSV output directory (header + part-*.csv) into a dict keyed by doi."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError("Missing Spark output directory: {}".format(directory))

    parts = sorted(directory.glob("part-*.csv"))
    if not parts:
        raise FileNotFoundError("No part-*.csv files in {}".format(directory))

    rows = {}
    header = None
    for part in parts:
        with part.open(newline="") as fh:
            reader = csv.reader(fh, delimiter="\t")
            part_header = next(reader)
            if header is None:
                header = part_header
            elif part_header != header:
                # Some parts may be header-only empty; skip mismatched empties
                if not any(True for _ in reader):
                    continue
                raise ValueError(
                    "Header mismatch in {}: {!r} vs {!r}".format(part, part_header, header)
                )
            for row in reader:
                if not row:
                    continue
                record = dict(zip(header, row))
                rows[record["doi"]] = record
    return rows


def find_unique_dir(pattern):
    matches = sorted(OUT.glob(pattern))
    matches = [p for p in matches if p.is_dir()]
    if not matches:
        raise FileNotFoundError(
            "No output directory matching {!r} under {}".format(pattern, OUT)
        )
    if len(matches) > 1:
        # Prefer the lexicographically last (usually the latest param string is fine for demo)
        print("WARNING: multiple matches for {!r}, using {}".format(pattern, matches[-1]))
    return matches[-1]


def load_graph_years(graph_path):
    years = {}
    with Path(graph_path).open(newline="") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            paper, _cite, _score, year = line.split("\t")
            years[paper] = year
    return years


def build_scores_tsv(destination=SCORES_OUT):
    """
    Join indicator outputs into the TopicClassesAndFWCI scores table:

        pid, type, year, pagerank, attrank, cc, 3y-cc
    """
    graph_name = GRAPH.name
    years = load_graph_years(GRAPH)

    pr_dir = find_unique_dir("PR_{}_local_*".format(graph_name))
    ar_dir = find_unique_dir("AttRank_{}_local_*".format(graph_name))
    cc_dir = OUT / "CC_{}".format(graph_name)
    cc3_dir = OUT / "3-year_3y_{}".format(graph_name)

    pr = read_spark_tsv_dir(pr_dir)
    ar = read_spark_tsv_dir(ar_dir)
    cc = read_spark_tsv_dir(cc_dir)
    cc3 = read_spark_tsv_dir(cc3_dir)

    papers = sorted(set(years) | set(pr) | set(ar) | set(cc) | set(cc3))
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(["pid", "type", "year", "pagerank", "attrank", "cc", "3y-cc"])
        for paper in papers:
            if paper not in pr or paper not in ar or paper not in cc or paper not in cc3:
                raise RuntimeError(
                    "Missing indicator row for {!r} "
                    "(need PageRank, AttRank, CC, and 3-year CC outputs)".format(paper)
                )
            writer.writerow(
                [
                    paper,
                    "publication",
                    years.get(paper, ""),
                    pr[paper]["pr"],
                    ar[paper]["attrank"],
                    cc[paper]["cc"],
                    cc3[paper]["3-cc"],
                ]
            )

    print("\nBuilt scores table from indicator outputs:")
    print("  PageRank: {}".format(pr_dir.name))
    print("  AttRank:  {}".format(ar_dir.name))
    print("  CC:       {}".format(cc_dir.name))
    print("  3y-CC:    {}".format(cc3_dir.name))
    print("  -> {}\n".format(destination))
    return destination


def build_jobs(selected):
    py = sys.executable
    graph = str(GRAPH)
    ckpt = str(CKPT)
    out = str(OUT)

    jobs = []
    if "cc" in selected:
        jobs.append(("CC", [py, str(REPO / "CC.py"), graph, PARTITIONS]))
    if "cc3y" in selected:
        jobs.append(("CC (3-year)", [py, str(REPO / "CC.py"), graph, PARTITIONS, "3y", "3"]))
    if "pagerank" in selected:
        jobs.append(
            (
                "PageRank",
                [
                    py,
                    str(REPO / "PageRank.py"),
                    graph,
                    ALPHA,
                    TOL,
                    ckpt,
                    PARTITIONS,
                    CHECKPOINT_MODE,
                ],
            )
        )
    if "attrank" in selected:
        jobs.append(
            (
                "AttRank",
                [py, str(REPO / "AttRank.py"), graph] + ATTRANK_ARGS + [ckpt, PARTITIONS, CHECKPOINT_MODE],
            )
        )
    if "ram" in selected:
        jobs.append(
            (
                "TAR / RAM",
                [
                    py,
                    str(REPO / "TAR.py"),
                    graph,
                    TAR_GAMMA,
                    TAR_YEAR,
                    "RAM",
                    PARTITIONS,
                    ckpt,
                    CHECKPOINT_MODE,
                ],
            )
        )
    if "ecm" in selected:
        jobs.append(
            (
                "TAR / ECM",
                [
                    py,
                    str(REPO / "TAR.py"),
                    graph,
                    TAR_GAMMA,
                    TAR_YEAR,
                    "ECM",
                    PARTITIONS,
                    ckpt,
                    CHECKPOINT_MODE,
                    ECM_ALPHA,
                    TOL,
                ],
            )
        )
    if "fwci" in selected:
        jobs.append(
            (
                "TopicClassesAndFWCI",
                [
                    py,
                    str(REPO / "TopicClassesAndFWCI.py"),
                    "--scores-file",
                    str(SCORES_OUT),
                    "--concepts-file",
                    str(CONCEPTS),
                    "--output-dir",
                    out,
                ],
            )
        )
    return jobs


def main():
    parser = argparse.ArgumentParser(
        description="Run Bip-Ranker root scripts on sample-data/ (local Spark)."
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=ALL_STEPS,
        metavar="STEP",
        help="Run only these steps (default: all). Choices: {}".format(", ".join(ALL_STEPS)),
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=ALL_STEPS,
        metavar="STEP",
        help="Skip these steps.",
    )
    args = parser.parse_args()

    selected = list(ALL_STEPS) if not args.only else list(args.only)
    if args.skip:
        selected = [s for s in selected if s not in args.skip]
    if not selected:
        print("ERROR: no steps left to run.", file=sys.stderr)
        sys.exit(1)

    check_environment()
    CKPT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    # FWCI needs scores.tsv built from indicator outputs. Prefer existing outputs;
    # if they are incomplete, run the missing indicator steps first.
    if "fwci" in selected:
        try:
            build_scores_tsv(SCORES_OUT)
            print("(Re)using indicator outputs already present under {}".format(OUT))
        except (FileNotFoundError, RuntimeError, ValueError):
            missing = [s for s in SCORE_STEPS if s not in selected]
            if missing:
                selected = missing + selected
                print(
                    "FWCI needs indicator outputs; also running: {}".format(
                        ", ".join(missing)
                    )
                )

    print("Repository: {}".format(REPO))
    print("Graph:      {}".format(GRAPH))
    print("Output:     {}".format(OUT))
    print("Steps:      {}".format(", ".join(selected)))

    jobs = build_jobs(selected)
    for name, argv in jobs:
        if name == "TopicClassesAndFWCI":
            try:
                build_scores_tsv(SCORES_OUT)
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                print(
                    "ERROR: could not build scores.tsv from indicator outputs.\n"
                    "  {}\n"
                    "  Run the cc, cc3y, pagerank, and attrank steps first "
                    "(e.g. python run_demo.py --skip ram ecm fwci).".format(exc),
                    file=sys.stderr,
                )
                sys.exit(1)
        run_step(name, argv)

    print("\nDemo finished. Results are under:\n  {}\n".format(OUT))


if __name__ == "__main__":
    main()
