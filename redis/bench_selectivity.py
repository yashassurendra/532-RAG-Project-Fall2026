import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
from redis.commands.search.query import Query

sys.path.insert(0, str(Path(__file__).resolve().parent))
from redis_config import (
    DATASET_SUFFIX,
    RESULTS_DIR,
    INDEX_NAME,
    K,
    get_redis_client,
    load_artifacts,
    load_ground_truth,
    compute_recall_at_k,
    get_docker_and_redis_stats,
    get_redis_stats,
)


def run_filter_level(r, q_vecs: np.ndarray, filter_name: str):
    filter_expr = "*" if filter_name == "unfiltered" else f"@sel_{filter_name}:{{1}}"
    q_str = f"({filter_expr})=>[KNN {K} @embedding $vec AS score]"

    # Warm-up (5 queries)
    for qv in q_vecs[:5]:
        q = Query(q_str).sort_by("score", asc=True).return_fields("row_idx", "score").paging(0, K).dialect(2)
        r.ft(INDEX_NAME).search(q, query_params={"vec": qv.tobytes()})

    latencies_ms = []
    retrieved_ids = np.full((len(q_vecs), K), -1, dtype=np.int64)

    # Snapshot exact Redis server CPU seconds right BEFORE the timed query loop
    _, _, cpu_before_s = get_redis_stats(r)
    t_start = time.perf_counter()

    for qi, qv in enumerate(q_vecs):
        q = Query(q_str).sort_by("score", asc=True).return_fields("row_idx", "score").paging(0, K).dialect(2)
        t0 = time.perf_counter()
        res = r.ft(INDEX_NAME).search(q, query_params={"vec": qv.tobytes()})
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        for rank, doc in enumerate(res.docs[:K]):
            retrieved_ids[qi, rank] = int(doc.row_idx)

    wall_s = time.perf_counter() - t_start
    # Snapshot exact Redis server CPU seconds right AFTER the timed query loop
    _, _, cpu_after_s = get_redis_stats(r)

    qps = len(q_vecs) / wall_s
    active_cpu_pct = ((cpu_after_s - cpu_before_s) / max(wall_s, 1e-6)) * 100.0

    # Still collect Docker RAM & Redis RAM
    redis_ram_mb, docker_ram_mb, _ = get_docker_and_redis_stats(r)

    return np.array(latencies_ms), retrieved_ids, qps, redis_ram_mb, docker_ram_mb, active_cpu_pct


def main():
    _, q_vecs, _ = load_artifacts(need_passages=False)
    r = get_redis_client()

    num_docs = int(r.ft(INDEX_NAME).info().get("num_docs", 0))
    assert num_docs > 0, f"Index '{INDEX_NAME}' is empty! Run load_and_index.py first."
    print(f"Running midpoint benchmark on '{INDEX_NAME}' ({num_docs:,} passages)...")

    summary_rows, per_query_rows = [], []

    for fname in ["unfiltered", "50pct", "10pct", "1pct"]:
        gt = load_ground_truth(fname)
        lats, ret_ids, qps, redis_ram, docker_ram, active_cpu = run_filter_level(r, q_vecs, fname)
        recall = compute_recall_at_k(ret_ids, gt, k=K)

        for qi, lat in enumerate(lats):
            per_query_rows.append(
                {
                    "backend": "redis",
                    "dataset": DATASET_SUFFIX,
                    "filter": fname,
                    "query_idx": qi,
                    "latency_ms": round(float(lat), 4),
                }
            )

        summary_rows.append(
            {
                "backend": "redis",
                "dataset": DATASET_SUFFIX,
                "filter": fname,
                "p50_ms": round(float(np.percentile(lats, 50)), 3),
                "p95_ms": round(float(np.percentile(lats, 95)), 3),
                "p99_ms": round(float(np.percentile(lats, 99)), 3),
                "qps": round(qps, 2),
                "recall_at_10": round(recall, 4),
                "redis_ram_mb": round(redis_ram, 2),
                "docker_ram_mb": round(docker_ram, 2),
                "docker_cpu_pct": round(active_cpu, 2),
            }
        )

    sum_df = pd.DataFrame(summary_rows)
    pq_df = pd.DataFrame(per_query_rows)

    sum_csv = RESULTS_DIR / f"redis_midpoint_summary_{DATASET_SUFFIX}.csv"
    pq_csv = RESULTS_DIR / f"redis_midpoint_per_query_{DATASET_SUFFIX}.csv"
    sum_df.to_csv(sum_csv, index=False)
    pq_df.to_csv(pq_csv, index=False)

    print("\n=== REDIS MIDPOINT BENCHMARK SUMMARY ===")
    print(sum_df.to_string(index=False))
    print(f"\nSaved CSVs:\n  - {sum_csv}\n  - {pq_csv}")


if __name__ == "__main__":
    main()