import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs('graphs', exist_ok=True)

# --- 1. FILTER SELECTIVITY GRAPHS ---
try:
    df_filter = pd.read_csv('metrics/faiss_benchmark_results.csv')

    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_filter['test_name'], df_filter['qps'], color='skyblue', edgecolor='black')
    plt.title('Queries Per Second (QPS) by Filter Selectivity (FAISS)', fontsize=14)
    plt.xlabel('Filter Selectivity', fontsize=12)
    plt.ylabel('Queries Per Second (QPS)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (yval*0.02), round(yval, 2), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig('graphs/qps_chart_100k.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(df_filter['test_name'], df_filter['p99_latency_ms'], marker='o', linestyle='-', color='red', markersize=8)
    plt.title('p99 Latency by Filter Selectivity (FAISS)', fontsize=14)
    plt.xlabel('Filter Selectivity', fontsize=12)
    plt.ylabel('p99 Latency (ms)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    for i, txt in enumerate(df_filter['p99_latency_ms']):
        plt.text(i, txt + (txt*0.02), round(txt, 2), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig('graphs/latency_chart_100k.png')
    plt.close()
    
    print("Filter selectivity graphs generated.")
except Exception as e:
    print(f"Skipping filter graphs: {e}")

# --- 2. CONCURRENCY GRAPHS ---
try:
    df_conc = pd.read_csv('metrics/faiss_concurrency_results.csv')

    plt.figure(figsize=(10, 6))
    plt.plot(df_conc['threads'].astype(str), df_conc['qps'], marker='s', linestyle='-', color='blue', markersize=8)
    plt.title('FAISS Throughput vs. Concurrency', fontsize=14)
    plt.xlabel('Simultaneous Clients (Threads)', fontsize=12)
    plt.ylabel('Queries Per Second (QPS)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    for i, txt in enumerate(df_conc['qps']):
        plt.text(i, txt + 1, round(txt, 2), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig('graphs/concurrency_qps.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(df_conc['threads'].astype(str), df_conc['p99_latency_ms'], marker='o', linestyle='-', color='darkorange', markersize=8)
    plt.title('FAISS Tail Latency (p99) vs. Concurrency', fontsize=14)
    plt.xlabel('Simultaneous Clients (Threads)', fontsize=12)
    plt.ylabel('p99 Latency (ms)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    for i, txt in enumerate(df_conc['p99_latency_ms']):
        plt.text(i, txt + 5, round(txt, 2), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig('graphs/concurrency_latency.png')
    plt.close()

    print("Concurrency graphs generated.")
except Exception as e:
    print(f"Skipping concurrency graphs: {e}")
