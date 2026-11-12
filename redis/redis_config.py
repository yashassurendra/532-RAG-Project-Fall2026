import os
import re
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import redis

# ---------------------------------------------------------
# Dynamic Paths
# ---------------------------------------------------------
# Resolves root directory (532-RAG-Project-Fall2026) regardless
# of whether scripts are executed from repo root or inside redis/
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"

# Redis & Benchmark Configuration
CONTAINER_NAME = "rag_redis_stack"
INDEX_NAME = "idx:passages"
DOC_PREFIX = "doc:"
K = 10

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)


# ---------------------------------------------------------
# Dynamic Dataset Suffix Detection
# ---------------------------------------------------------
def get_latest_dataset_suffix() -> str:
    """
    Dynamically finds the dataset suffix (e.g., 'test1k', '100k', 'full171k').
    Checks:
      1. DATASET_SUFFIX environment variable override.
      2. Most recently modified 'passages_*.npy' file in data/processed/.
      3. Falls back to 'test1k'.
    """
    env_suffix = os.getenv("DATASET_SUFFIX")
    if env_suffix:
        return env_suffix

    if DATA_DIR.exists():
        passage_files = list(DATA_DIR.glob("passages_*.npy"))
        if passage_files:
            latest_file = max(passage_files, key=lambda f: f.stat().st_mtime)
            match = re.search(r"passages_(.+)\.npy", latest_file.name)
            if match:
                return match.group(1)

    return "test1k"


DATASET_SUFFIX = get_latest_dataset_suffix()


# ---------------------------------------------------------
# Redis Client Connection
# ---------------------------------------------------------
def get_redis_client() -> redis.Redis:
    """
    Initializes a Redis client connection and confirms
    the RediSearch ('search') module is active.
    """
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=False,
    )
    r.ping()
    modules = [m.get(b"name", b"").decode("utf-8", errors="ignore") for m in r.module_list()]
    assert "search" in modules, (
        f"Connected to Redis, but the 'search' (RediSearch) module is not loaded! "
        f"Loaded modules: {modules}. Make sure you are running redis/redis-stack-server."
    )
    return r


# ---------------------------------------------------------
# Artifact Loaders & Metrics
# ---------------------------------------------------------
def load_artifacts(need_passages: bool = False):
    """
    Loads queries and metadata (and optionally passage vectors)
    for the dynamically resolved DATASET_SUFFIX.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    q_vecs = np.load(DATA_DIR / f"queries_{DATASET_SUFFIX}.npy")
    meta = pd.read_parquet(DATA_DIR / f"metadata_{DATASET_SUFFIX}.parquet")
    p_vecs = np.load(DATA_DIR / f"passages_{DATASET_SUFFIX}.npy") if need_passages else None
    return p_vecs, q_vecs, meta


def load_ground_truth(filter_name: str) -> np.ndarray:
    """Loads the top-10 ground-truth row indices for the specified filter."""
    return np.load(DATA_DIR / f"gt_top10_{filter_name}_{DATASET_SUFFIX}.npy")


def compute_recall_at_k(retrieved: np.ndarray, ground_truth: np.ndarray, k: int = 10) -> float:
    """Computes mean Recall@k by comparing retrieved row_idx values against ground truth."""
    recalls = [
        len(set(retrieved[i, :k]).intersection(set(ground_truth[i, :k]))) / float(k)
        for i in range(len(ground_truth))
    ]
    return float(np.mean(recalls))


# ---------------------------------------------------------
# Resource Monitoring (Docker + Server-side)
# ---------------------------------------------------------
def get_docker_and_redis_stats(r: redis.Redis):
    """
    Collects internal Redis server RAM (MB) alongside
    external Docker container CPU % and RAM (MB) via `docker stats`.
    """
    redis_ram_mb = r.info("memory").get("used_memory", 0) / (1024.0 * 1024.0)
    docker_cpu_pct, docker_ram_mb = 0.0, 0.0

    try:
        out = subprocess.check_output(
            ["docker", "stats", CONTAINER_NAME, "--no-stream", "--format", "{{.CPUPerc}},{{.MemUsage}}"],
            text=True,
        ).strip()
        cpu_str, mem_str = out.split(",")
        docker_cpu_pct = float(cpu_str.replace("%", "").strip())
        mem_used = mem_str.split("/")[0].strip()
        if "GiB" in mem_used:
            docker_ram_mb = float(mem_used.replace("GiB", "").strip()) * 1024.0
        elif "MiB" in mem_used:
            docker_ram_mb = float(mem_used.replace("MiB", "").strip())
    except Exception:
        pass

    return redis_ram_mb, docker_ram_mb, docker_cpu_pct


def get_redis_stats(r: redis.Redis):
    """
    Collects memory (MB), peak memory (MB), and cumulative CPU
    execution time (seconds) directly from the Redis server engine.
    """
    mem = r.info("memory")
    cpu = r.info("cpu")
    used_mb = mem.get("used_memory", 0) / (1024.0 * 1024.0)
    peak_mb = mem.get("used_memory_peak", 0) / (1024.0 * 1024.0)
    cpu_s = float(cpu.get("used_cpu_sys", 0.0)) + float(cpu.get("used_cpu_user", 0.0))
    return used_mb, peak_mb, cpu_s