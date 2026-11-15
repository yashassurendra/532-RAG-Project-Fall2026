import sys
from pathlib import Path
import time
import pandas as pd
from redis.commands.search.field import TagField, NumericField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType

# Ensure local redis_config is imported cleanly
sys.path.insert(0, str(Path(__file__).resolve().parent))
from redis_config import (
    DATASET_SUFFIX,
    RESULTS_DIR,
    INDEX_NAME,
    DOC_PREFIX,
    get_redis_client,
    load_artifacts,
    get_docker_and_redis_stats,
    get_redis_stats,
)

BATCH_SIZE = 2000


def create_schema(r, dim: int = 384):
    try:
        r.ft(INDEX_NAME).dropindex(delete_documents=True)
    except Exception:
        pass

    schema = (
        NumericField("row_idx", sortable=True),
        TagField("doc_id"),
        TagField("category"),
        NumericField("access_group"),
        NumericField("timestamp"),
        TagField("selectivity_bucket"),
        TagField("sel_50pct"),
        TagField("sel_10pct"),
        TagField("sel_1pct"),
        VectorField(
            "embedding",
            "HNSW",
            {
                "TYPE": "FLOAT32",
                "DIM": dim,
                "DISTANCE_METRIC": "IP",
                "M": 16,
                "EF_CONSTRUCTION": 200,
                "EF_RUNTIME": 50,
            },
        ),
    )
    definition = IndexDefinition(prefix=[DOC_PREFIX], index_type=IndexType.HASH)
    r.ft(INDEX_NAME).create_index(fields=schema, definition=definition)


def main():
    p_vecs, _, meta = load_artifacts(need_passages=True)
    r = get_redis_client()

    print(f"[1/3] Creating HNSW index '{INDEX_NAME}' (dim={p_vecs.shape[1]})...")
    create_schema(r, dim=p_vecs.shape[1])

    print(f"[2/3] Bulk loading {len(meta):,} passages ({DATASET_SUFFIX})...")
    _, _, cpu_before_s = get_redis_stats(r)
    t0 = time.perf_counter()
    pipe = r.pipeline(transaction=False)

    for i in range(len(meta)):
        row = meta.iloc[i]
        pipe.hset(
            f"{DOC_PREFIX}{int(row['row_idx'])}",
            mapping={
                "row_idx": int(row["row_idx"]),
                "doc_id": str(row["doc_id"]),
                "category": str(row["category"]),
                "access_group": int(row["access_group"]),
                "timestamp": int(row["timestamp"]),
                "selectivity_bucket": str(row["selectivity_bucket"]),
                "sel_50pct": "1" if row["sel_50pct"] else "0",
                "sel_10pct": "1" if row["sel_10pct"] else "0",
                "sel_1pct": "1" if row["sel_1pct"] else "0",
                "embedding": p_vecs[i].tobytes(),
            },
        )
        if (i + 1) % BATCH_SIZE == 0 or (i + 1) == len(meta):
            pipe.execute()

    # Wait until background HNSW indexing finishes
    while int(r.ft(INDEX_NAME).info().get("indexing", 0)) == 1:
        time.sleep(0.05)

    build_time_s = time.perf_counter() - t0
    _, _, cpu_after_s = get_redis_stats(r)
    redis_ram_mb, docker_ram_mb, _ = get_docker_and_redis_stats(r)
    docker_cpu_pct = ((cpu_after_s - cpu_before_s) / max(build_time_s, 1e-6)) * 100.0

    build_df = pd.DataFrame(
        [
            {
                "backend": "redis",
                "dataset": DATASET_SUFFIX,
                "num_docs": len(meta),
                "index_build_s": round(build_time_s, 3),
                "redis_ram_mb": round(redis_ram_mb, 2),
                "docker_ram_mb": round(docker_ram_mb, 2),
                "docker_cpu_pct": round(docker_cpu_pct, 2),
            }
        ]
    )

    out_csv = RESULTS_DIR / f"redis_index_build_{DATASET_SUFFIX}.csv"
    build_df.to_csv(out_csv, index=False)
    print("\n[3/3] Index Build Complete:")
    print(build_df.to_string(index=False))
    print(f"Saved -> {out_csv}")


if __name__ == "__main__":
    main()