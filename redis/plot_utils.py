import sys
from pathlib import Path
import pandas as pd

# Ensure local redis_config is imported cleanly
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from redis_config import DATASET_SUFFIX, RESULTS_DIR

FILTER_ORDER = ["unfiltered", "50pct", "10pct", "1pct"]
FILTER_LABELS = ["Unfiltered (100%)", "50% Selectivity", "10% Selectivity", "1% Selectivity"]
BACKEND_COLORS = {"redis": "#d62728", "faiss": "#1f77b4", "postgres": "#2ca02c"}


def load_summary_data(suffix: str = DATASET_SUFFIX) -> pd.DataFrame:
    """Loads midpoint summary CSVs for all available backends."""
    dfs = []
    for backend in ["faiss", "redis", "postgres"]:
        csv_path = RESULTS_DIR / f"{backend}_midpoint_summary_{suffix}.csv"
        if csv_path.exists():
            dfs.append(pd.read_csv(csv_path))

    if not dfs:
        raise FileNotFoundError(
            f"No '*_midpoint_summary_{suffix}.csv' files found in {RESULTS_DIR}. "
            "Run bench_selectivity.py first!"
        )
    return pd.concat(dfs, ignore_index=True)


def load_build_data(suffix: str = DATASET_SUFFIX) -> pd.DataFrame:
    """Loads index build CSVs for all available backends."""
    dfs = []
    for backend in ["faiss", "redis", "postgres"]:
        csv_path = RESULTS_DIR / f"{backend}_index_build_{suffix}.csv"
        if csv_path.exists():
            dfs.append(pd.read_csv(csv_path))

    if not dfs:
        raise FileNotFoundError(
            f"No '*_index_build_{suffix}.csv' files found in {RESULTS_DIR}. "
            "Run load_and_index.py first!"
        )
    return pd.concat(dfs, ignore_index=True)