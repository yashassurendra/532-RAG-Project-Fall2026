import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"

CONTAINER_NAME = "rag_postgres"
TABLE_NAME = "passages"
INDEX_NAME = "passages_embedding_hnsw"
K = 10

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "ragdb")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")


def get_latest_dataset_suffix() -> str:
    """Find the most recently generated dataset, unless DATASET_SUFFIX overrides it."""
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


def get_postgres_connection():
    """Connect to PostgreSQL using the POSTGRES_* environment settings."""
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def load_artifacts(need_passages: bool = False):
    """Load embeddings and metadata for the resolved dataset suffix."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    queries = np.load(DATA_DIR / f"queries_{DATASET_SUFFIX}.npy")
    metadata = pd.read_parquet(DATA_DIR / f"metadata_{DATASET_SUFFIX}.parquet")
    passages = np.load(DATA_DIR / f"passages_{DATASET_SUFFIX}.npy") if need_passages else None
    return passages, queries, metadata


def load_ground_truth(filter_name: str) -> np.ndarray:
    """Load top-10 ground-truth row indices for a filter."""
    return np.load(DATA_DIR / f"gt_top10_{filter_name}_{DATASET_SUFFIX}.npy")


def compute_recall_at_k(retrieved: np.ndarray, ground_truth: np.ndarray, k: int = K) -> float:
    """Compute mean Recall@k from retrieved row indices and ground truth."""
    recalls = [
        len(set(retrieved[i, :k]).intersection(set(ground_truth[i, :k]))) / float(k)
        for i in range(len(ground_truth))
    ]
    return float(np.mean(recalls))


def vector_literal(vector: np.ndarray) -> str:
    """Format an embedding as a PostgreSQL vector literal."""
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def get_postgres_and_docker_stats(connection):
    """Return cached PostgreSQL buffer MB and Docker container RAM/CPU usage."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT count(*) * current_setting('block_size')::int
            FROM pg_buffercache
            WHERE reldatabase = (
                SELECT oid FROM pg_database WHERE datname = current_database()
            )
              AND usagecount > 0
            """
        )
        cached_bytes = cursor.fetchone()[0]
    postgres_ram_mb = cached_bytes / (1024.0 * 1024.0)

    output = subprocess.check_output(
        [
            "docker",
            "stats",
            CONTAINER_NAME,
            "--no-stream",
            "--format",
            "{{.CPUPerc}},{{.MemUsage}}",
        ],
        text=True,
    ).strip()
    cpu_text, memory_text = output.split(",", maxsplit=1)
    cpu_pct = float(cpu_text.replace("%", "").strip())
    memory_used = memory_text.split("/", maxsplit=1)[0].strip()
    memory_match = re.fullmatch(r"([\d.]+)\s*(B|kB|MB|GB|KiB|MiB|GiB)", memory_used)
    if not memory_match:
        raise ValueError(f"Unrecognized Docker memory value: {memory_used!r}")
    memory_value = float(memory_match.group(1))
    memory_units = {
        "B": 1.0 / (1024.0 * 1024.0),
        "kB": 1.0 / 1024.0,
        "MB": 1.0,
        "GB": 1024.0,
        "KiB": 1.0 / 1024.0,
        "MiB": 1.0,
        "GiB": 1024.0,
    }
    docker_ram_mb = memory_value * memory_units[memory_match.group(2)]

    return postgres_ram_mb, docker_ram_mb, cpu_pct
