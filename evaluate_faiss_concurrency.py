import os
import time
import psutil
import numpy as np
import pandas as pd
import concurrent.futures
from faiss_backend import FaissBackend

def main():
    # --- 1. SETUP PATHS ---
    os.makedirs("metrics", exist_ok=True)
    
    data_dir = "data/processed/"
    passages_path = data_dir + "passages_100k.npy"
    metadata_path = data_dir + "metadata_100k.parquet"
    queries_path = data_dir + "queries_100k.npy"
    
    results_output_path = "metrics/faiss_concurrency_results.csv"

    # --- 2. LOAD ENGINE ---
    print("Loading FAISS engine for concurrency stress test...")
    backend = FaissBackend(passages_path, metadata_path)
    query_vectors = np.load(queries_path)
    
    # --- 3. CONCURRENCY SWEEP CONFIGURATION ---
    # Testing how the system scales under simultaneous load
    concurrency_levels = [1, 4, 8, 16, 32]
    all_results = []

    # --- 4. RUN STRESS TEST ---
    for threads in concurrency_levels:
        print(f"\nSimulating {threads} simultaneous client(s)...")
        
        def run_query(q_vec):
            start = time.perf_counter()
            # Running unfiltered to isolate raw vector search contention
            backend.search(q_vec, filter_dict=None, k=10)
            end = time.perf_counter()
            return (end - start) * 1000

        # Reset CPU tracker
        psutil.cpu_percent(interval=None)
        
        # Start wall-clock timer for QPS calculation
        start_total = time.perf_counter()
        
        # Launch concurrent threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            latencies = list(executor.map(run_query, query_vectors))
            
        end_total = time.perf_counter()
        
        # Snapshot peak CPU usage
        cpu_usage = psutil.cpu_percent(interval=None)
        
        # --- 5. CALCULATE METRICS ---
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        total_time_s = end_total - start_total
        qps = len(query_vectors) / total_time_s
        
        print(f"  QPS:       {qps:.2f}")
        print(f"  p99 Lat.:  {p99:.2f} ms")
        print(f"  CPU Usage: {cpu_usage:.1f}%")

        all_results.append({
            "threads": threads,
            "qps": qps,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "p99_latency_ms": p99,
            "cpu_percent": cpu_usage
        })

    # --- 6. EXPORT RESULTS ---
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_output_path, index=False)
    print(f"\nConcurrency sweep complete! Results saved to: {results_output_path}")

if __name__ == "__main__":
    main()
