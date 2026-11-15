import os
import time
from pathlib import Path

import psutil
import numpy as np
import pandas as pd
from faiss_backend import FaissBackend


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"
DATASET_SUFFIX = "100k"

def calculate_recall(retrieved_indices, ground_truth_indices):
    """Calculates how many of the retrieved documents are in the ground truth."""
    if len(retrieved_indices) == 0:
        return 0.0
    hits = len(set(retrieved_indices) & set(ground_truth_indices))
    return hits / len(ground_truth_indices)

def main():
    # --- 1. SETUP PATHS & PROCESS MONITOR ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    process = psutil.Process(os.getpid())
    base_memory = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
    
    # CHANGED: Now pointing to the new 100k dataset files
    passages_path = DATA_DIR / f"passages_{DATASET_SUFFIX}.npy"
    metadata_path = DATA_DIR / f"metadata_{DATASET_SUFFIX}.parquet"
    queries_path = DATA_DIR / f"queries_{DATASET_SUFFIX}.npy"

    # --- 2. LOAD ENGINE & MEASURE RAM ---
    print("Starting evaluation...")
    process.cpu_percent(interval=None)
    build_start = time.perf_counter()
    backend = FaissBackend(passages_path, metadata_path)
    build_time_s = time.perf_counter() - build_start
    build_cpu_pct = process.cpu_percent(interval=None)
    
    loaded_memory = process.memory_info().rss / (1024 * 1024)
    index_ram_mb = loaded_memory - base_memory
    print(f"Index + Metadata RAM Usage: {index_ram_mb:.2f} MB")

    build_df = pd.DataFrame(
        [
            {
                "backend": "faiss",
                "dataset": DATASET_SUFFIX,
                "num_docs": len(backend.vectors),
                "index_build_s": round(build_time_s, 3),
                "faiss_ram_mb": round(index_ram_mb, 2),
                "docker_ram_mb": 0.0,
                "docker_cpu_pct": round(build_cpu_pct, 2),
            }
        ]
    )
    build_path = RESULTS_DIR / f"faiss_index_build_{DATASET_SUFFIX}.csv"
    build_df.to_csv(build_path, index=False)
    
    query_vectors = np.load(queries_path)
    
    # --- 3. DEFINE TESTS ---
    # CHANGED: Now using the 100k ground truth files
    tests = [
        {"name": "unfiltered", "filter_dict": None, "gt_file": "gt_top10_unfiltered_100k.npy"},
        {"name": "50pct", "filter_dict": {"sel_50pct": True}, "gt_file": "gt_top10_50pct_100k.npy"},
        {"name": "10pct", "filter_dict": {"sel_10pct": True}, "gt_file": "gt_top10_10pct_100k.npy"},
        {"name": "1pct", "filter_dict": {"sel_1pct": True}, "gt_file": "gt_top10_1pct_100k.npy"}
    ]

    summary_rows = []
    per_query_rows = []

    # --- 4. RUN BENCHMARK LOOP ---
    for test in tests:
        print(f"\nRunning test: {test['name']}")
        ground_truth = np.load(DATA_DIR / test["gt_file"])
        
        latencies = []
        recalls = []

        psutil.cpu_percent(interval=None)
        wall_start = time.perf_counter()

        for i in range(len(query_vectors)):
            q_vec = query_vectors[i]
            gt_indices = ground_truth[i]

            start_time = time.perf_counter()
            retrieved_indices = backend.search(q_vec, filter_dict=test["filter_dict"], k=10)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            recalls.append(calculate_recall(retrieved_indices, gt_indices))
            per_query_rows.append(
                {
                    "backend": "faiss",
                    "dataset": DATASET_SUFFIX,
                    "filter": test["name"],
                    "query_idx": i,
                    "latency_ms": round(float(latency_ms), 4),
                }
            )

        wall_time_s = time.perf_counter() - wall_start
        cpu_usage = psutil.cpu_percent(interval=None)

        # --- 5. CALCULATE STATS ---
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        mean_recall = np.mean(recalls)
        qps = len(query_vectors) / wall_time_s

        print(f"  Recall@10: {mean_recall:.2f}")
        print(f"  QPS:       {qps:.2f}")
        print(f"  p99 Lat.:  {p99:.2f} ms")
        print(f"  CPU Usage: {cpu_usage:.1f}%")

        summary_rows.append(
            {
                "backend": "faiss",
                "dataset": DATASET_SUFFIX,
                "filter": test["name"],
                "p50_ms": round(float(p50), 3),
                "p95_ms": round(float(p95), 3),
                "p99_ms": round(float(p99), 3),
                "qps": round(float(qps), 2),
                "recall_at_10": round(float(mean_recall), 4),
                "faiss_ram_mb": round(float(index_ram_mb), 2),
                "docker_ram_mb": 0.0,
                "docker_cpu_pct": round(float(cpu_usage), 2),
            }
        )

    # --- 6. EXPORT RESULTS ---
    summary_path = RESULTS_DIR / f"faiss_midpoint_summary_{DATASET_SUFFIX}.csv"
    per_query_path = RESULTS_DIR / f"faiss_midpoint_per_query_{DATASET_SUFFIX}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(per_query_rows).to_csv(per_query_path, index=False)
    print(f"\nAll done! Results saved to:\n  - {build_path}\n  - {summary_path}\n  - {per_query_path}")

if __name__ == "__main__":
    main()
