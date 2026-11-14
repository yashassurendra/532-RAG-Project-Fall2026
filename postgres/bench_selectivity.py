import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from postgres_config import (
    DATASET_SUFFIX,
    K,
    RESULTS_DIR,
    TABLE_NAME,
    compute_recall_at_k,
    get_postgres_and_docker_stats,
    get_postgres_connection,
    load_artifacts,
    load_ground_truth,
    vector_literal,
)


def run_filter_level(connection, query_vectors: np.ndarray, filter_name: str):
    where_clause = "" if filter_name == "unfiltered" else f"WHERE sel_{filter_name} = TRUE"
    sql = f"""
        SELECT row_idx
        FROM {TABLE_NAME}
        {where_clause}
        ORDER BY embedding <#> %s::vector
        LIMIT {K}
    """

    with connection.cursor() as cursor:
        cursor.execute("SET hnsw.ef_search = 50")
        cursor.execute("SET hnsw.iterative_scan = strict_order")
        for query_vector in query_vectors[:5]:
            cursor.execute(sql, (vector_literal(query_vector),))
            cursor.fetchall()

        latencies_ms = []
        retrieved_ids = np.full((len(query_vectors), K), -1, dtype=np.int64)
        start = time.perf_counter()

        for query_idx, query_vector in enumerate(query_vectors):
            query_start = time.perf_counter()
            cursor.execute(sql, (vector_literal(query_vector),))
            results = cursor.fetchall()
            latencies_ms.append((time.perf_counter() - query_start) * 1000.0)
            for rank, (row_idx,) in enumerate(results[:K]):
                retrieved_ids[query_idx, rank] = int(row_idx)

        wall_s = time.perf_counter() - start

    return np.array(latencies_ms), retrieved_ids, len(query_vectors) / wall_s


def main():
    _, query_vectors, _ = load_artifacts(need_passages=False)
    connection = get_postgres_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT count(*) FROM {TABLE_NAME}")
            num_docs = int(cursor.fetchone()[0])
        assert num_docs > 0, f"Table '{TABLE_NAME}' is empty! Run load_and_index.py first."
        print(f"Running midpoint benchmark on PostgreSQL ({num_docs:,} passages)...")

        summary_rows, per_query_rows = [], []
        for filter_name in ["unfiltered", "50pct", "10pct", "1pct"]:
            ground_truth = load_ground_truth(filter_name)
            latencies, retrieved_ids, qps = run_filter_level(
                connection, query_vectors, filter_name
            )
            recall = compute_recall_at_k(retrieved_ids, ground_truth, k=K)
            postgres_ram_mb, docker_ram_mb, docker_cpu_pct = (
                get_postgres_and_docker_stats(connection)
            )

            for query_idx, latency in enumerate(latencies):
                per_query_rows.append(
                    {
                        "backend": "postgres",
                        "dataset": DATASET_SUFFIX,
                        "filter": filter_name,
                        "query_idx": query_idx,
                        "latency_ms": round(float(latency), 4),
                    }
                )

            summary_rows.append(
                {
                    "backend": "postgres",
                    "dataset": DATASET_SUFFIX,
                    "filter": filter_name,
                    "p50_ms": round(float(np.percentile(latencies, 50)), 3),
                    "p95_ms": round(float(np.percentile(latencies, 95)), 3),
                    "p99_ms": round(float(np.percentile(latencies, 99)), 3),
                    "qps": round(qps, 2),
                    "recall_at_10": round(recall, 4),
                    "postgres_ram_mb": round(postgres_ram_mb, 2),
                    "docker_ram_mb": round(docker_ram_mb, 2),
                    "docker_cpu_pct": round(docker_cpu_pct, 2),
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        per_query_df = pd.DataFrame(per_query_rows)
        summary_csv = RESULTS_DIR / f"postgres_midpoint_summary_{DATASET_SUFFIX}.csv"
        per_query_csv = RESULTS_DIR / f"postgres_midpoint_per_query_{DATASET_SUFFIX}.csv"
        summary_df.to_csv(summary_csv, index=False)
        per_query_df.to_csv(per_query_csv, index=False)

        print("\n=== POSTGRES MIDPOINT BENCHMARK SUMMARY ===")
        print(summary_df.to_string(index=False))
        print(f"\nSaved CSVs:\n  - {summary_csv}\n  - {per_query_csv}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
