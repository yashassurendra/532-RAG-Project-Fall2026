import os
import re
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"

FILTER_ORDER = ["unfiltered", "50pct", "10pct", "1pct"]
FILTER_LABELS = ["Unfiltered (100%)", "50% Selectivity", "10% Selectivity", "1% Selectivity"]
BACKEND_COLORS = {"redis": "#d62728", "faiss": "#1f77b4", "postgres": "#2ca02c"}


def get_latest_dataset_suffix() -> str:
    """Resolve the current dataset suffix without importing a backend client."""
    env_suffix = os.getenv("DATASET_SUFFIX")
    if env_suffix:
        return env_suffix

    if DATA_DIR.exists():
        passage_files = list(DATA_DIR.glob("passages_*.npy"))
        if passage_files:
            latest_file = max(passage_files, key=lambda file: file.stat().st_mtime)
            match = re.search(r"passages_(.+)\.npy", latest_file.name)
            if match:
                return match.group(1)

    return "test1k"


DATASET_SUFFIX = get_latest_dataset_suffix()


def load_summary_data(suffix: str = DATASET_SUFFIX) -> pd.DataFrame:
    """Load midpoint summary CSVs for all available backends."""
    frames = []
    for backend in ["faiss", "redis", "postgres"]:
        csv_path = RESULTS_DIR / f"{backend}_midpoint_summary_{suffix}.csv"
        if csv_path.exists():
            frames.append(pd.read_csv(csv_path))

    if not frames:
        raise FileNotFoundError(
            f"No '*_midpoint_summary_{suffix}.csv' files found in {RESULTS_DIR}. "
            "Run a backend bench_selectivity.py script first."
        )
    return pd.concat(frames, ignore_index=True)


def load_build_data(suffix: str = DATASET_SUFFIX) -> pd.DataFrame:
    """Load index-build CSVs for all available backends."""
    frames = []
    for backend in ["faiss", "redis", "postgres"]:
        csv_path = RESULTS_DIR / f"{backend}_index_build_{suffix}.csv"
        if csv_path.exists():
            frames.append(pd.read_csv(csv_path))

    if not frames:
        raise FileNotFoundError(
            f"No '*_index_build_{suffix}.csv' files found in {RESULTS_DIR}. "
            "Run a backend load_and_index.py script first."
        )
    return pd.concat(frames, ignore_index=True)
