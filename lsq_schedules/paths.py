"""Canonical locations of the repository's data and figure folders.

All paths are absolute and derived from the location of this file, so that
experiments and plotting scripts can be launched from any working directory.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"   # JSON outputs of the experiments
FIGURES_DIR = REPO_ROOT / "figures"   # PDF figures of the paper
