import os
import time
import psutil
import numpy as np
import pandas as pd
from faiss_backend import FaissBackend

def calculate_recall(retrieved_indices, ground_truth_indices):
    """Calculates how many of the retrieved documents are in the ground truth."""
    if len(retrieved_indices) == 0:
        return 0.0
    hits = len(set(retrieved_indices) & set(ground_truth_indices))
    return hits / len(ground_truth_indices)

def main():
    # --- 1. SETUP PATHS & PROCESS MONITOR ---
    os.makedirs("metrics", exist_ok=True)
    
    process = psutil.Process(os.getpid())
    base_memory = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
    
    data_dir = "data/processed/"
    
    # CHANGED: Now pointing to the new 100k dataset files
    passages_path = data_dir + "passages_100k.npy"
    metadata_path = data_dir + "metadata_100k.parquet"
    queries_path = data_dir + "queries_100k.npy"
    
    results_output_path = "metrics/faiss_benchmark_results.csv"

    # --- 2. LOAD ENGINE & MEASURE RAM ---
    print("Starting evaluation...")
    backend = FaissBackend(passages_path, metadata_path)
    
    loaded_memory = process.memory_info().rss / (1024 * 1024)
    index_ram_mb = loaded_memory - base_memory
    print(f"Index + Metadata RAM Usage: {index_ram_mb:.2f} MB")
    
    query_vectors = np.load(queries_path)
    
    # --- 3. DEFINE TESTS ---
    # CHANGED: Now using the 100k ground truth files
    tests = [
        {"name": "unfiltered", "filter_dict": None, "gt_file": "gt_top10_unfiltered_100k.npy"},
        {"name": "50pct", "filter_dict": {"sel_50pct": True}, "gt_file": "gt_top10_50pct_100k.npy"},
        {"name": "10pct", "filter_dict": {"sel_10pct": True}, "gt_file": "gt_top10_10pct_100k.npy"},
        {"name": "1pct", "filter_dict": {"sel_1pct": True}, "gt_file": "gt_top10_1pct_100k.npy"}
    ]

    all_results = []

    # --- 4. RUN BENCHMARK LOOP ---
    for test in tests:
        print(f"\nRunning test: {test['name']}")
        ground_truth = np.load(data_dir + test["gt_file"])
        
        latencies = []
        recalls = []

        psutil.cpu_percent(interval=None)

        for i in range(len(query_vectors)):
            q_vec = query_vectors[i]
            gt_indices = ground_truth[i]

            start_time = time.perf_counter()
            retrieved_indices = backend.search(q_vec, filter_dict=test["filter_dict"], k=10)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            recalls.append(calculate_recall(retrieved_indices, gt_indices))

        cpu_usage = psutil.cpu_percent(interval=None)

        # --- 5. CALCULATE STATS ---
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        mean_recall = np.mean(recalls)
        total_time_s = sum(latencies) / 1000
        qps = len(query_vectors) / total_time_s

        print(f"  Recall@10: {mean_recall:.2f}")
        print(f"  QPS:       {qps:.2f}")
        print(f"  p99 Lat.:  {p99:.2f} ms")
        print(f"  CPU Usage: {cpu_usage:.1f}%")

        all_results.append({
            "test_name": test["name"],
            "recall_10": mean_recall,
            "qps": qps,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "p99_latency_ms": p99,
            "index_ram_mb": index_ram_mb,
            "cpu_percent": cpu_usage
        })

    # --- 6. EXPORT RESULTS ---
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_output_path, index=False)
    print(f"\nAll done! Results saved to: {results_output_path}")

if __name__ == "__main__":
    main()
