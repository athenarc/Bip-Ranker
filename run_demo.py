#!/usr/bin/env python3
"""
Run the root ranking scripts on sample-data/ using local Spark (no HDFS).

Usage (from the repo root, with the venv activated):

    python run_demo.py
    python run_demo.py --only cc pagerank
    python run_demo.py --skip ecm

Requires: Java on PATH, and ``pip install -r requirements.txt``.
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SAMPLE = REPO / "sample-data"
GRAPH = SAMPLE / "citation_graph.tsv"
SCORES = SAMPLE / "scores.tsv"
CONCEPTS = SAMPLE / "concepts.tsv"
CKPT = SAMPLE / "checkpoints"
OUT = SAMPLE / "output"

# Demo parameters tuned for the small sample graph (fast convergence).
ALPHA = "0.85"
TOL = "1e-3"
PARTITIONS = "1"
CHECKPOINT_MODE = "local"
ATTRANK_ARGS = ["0.2", "0.5", "0.3", "0.6", "2020", "2018", TOL]
TAR_GAMMA = "0.6"
TAR_YEAR = "2020"
ECM_ALPHA = "0.5"

ALL_STEPS = ("cc", "cc3y", "pagerank", "attrank", "ram", "ecm", "fwci")


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

    for path in (GRAPH, SCORES, CONCEPTS):
        if not path.is_file():
            print("ERROR: missing sample file: {}".format(path), file=sys.stderr)
            sys.exit(1)


def run_step(name, argv):
    print("\n" + "=" * 60)
    print("Running: {}".format(name))
    print("  " + " ".join(argv))
    print("=" * 60 + "\n")
    subprocess.check_call(argv, cwd=str(OUT))


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
                    str(SCORES),
                    "--concepts-file",
                    str(CONCEPTS),
                    "--openaire-concepts-output",
                    str(OUT / "openaire_concepts"),
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

    print("Repository: {}".format(REPO))
    print("Graph:      {}".format(GRAPH))
    print("Output:     {}".format(OUT))
    print("Steps:      {}".format(", ".join(selected)))

    for name, argv in build_jobs(selected):
        run_step(name, argv)

    print("\nDemo finished. Results are under:\n  {}\n".format(OUT))


if __name__ == "__main__":
    main()
