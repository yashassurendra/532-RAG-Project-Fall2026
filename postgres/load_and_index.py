import sys
import time
from pathlib import Path

import pandas as pd
from psycopg2.extras import execute_values

sys.path.insert(0, str(Path(__file__).resolve().parent))
from postgres_config import (
    DATASET_SUFFIX,
    INDEX_NAME,
    RESULTS_DIR,
    TABLE_NAME,
    get_postgres_and_docker_stats,
    get_postgres_connection,
    load_artifacts,
    vector_literal,
)

BATCH_SIZE = 500


def create_schema(connection, dim: int) -> None:
    """Create the vector table and its HNSW inner-product index."""
    with connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_buffercache")
        cursor.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
        cursor.execute(
            f"""
            CREATE TABLE {TABLE_NAME} (
                row_idx BIGINT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                category TEXT NOT NULL,
                access_group INTEGER NOT NULL,
                timestamp BIGINT NOT NULL,
                selectivity_bucket TEXT NOT NULL,
                sel_50pct BOOLEAN NOT NULL,
                sel_10pct BOOLEAN NOT NULL,
                sel_1pct BOOLEAN NOT NULL,
                embedding vector({dim}) NOT NULL
            )
            """
        )
    connection.commit()


def main():
    passages, _, metadata = load_artifacts(need_passages=True)
    connection = get_postgres_connection()
    try:
        print(f"[1/3] Creating PostgreSQL HNSW table (dim={passages.shape[1]})...")
        create_schema(connection, dim=passages.shape[1])

        print(f"[2/3] Bulk loading {len(metadata):,} passages ({DATASET_SUFFIX})...")
        start = time.perf_counter()
        insert_sql = f"""
            INSERT INTO {TABLE_NAME} (
                row_idx, doc_id, category, access_group, timestamp,
                selectivity_bucket, sel_50pct, sel_10pct, sel_1pct, embedding
            ) VALUES %s
        """

        with connection.cursor() as cursor:
            for start_idx in range(0, len(metadata), BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, len(metadata))
                rows = []
                for i in range(start_idx, end_idx):
                    row = metadata.iloc[i]
                    rows.append(
                        (
                            int(row["row_idx"]),
                            str(row["doc_id"]),
                            str(row["category"]),
                            int(row["access_group"]),
                            int(row["timestamp"]),
                            str(row["selectivity_bucket"]),
                            bool(row["sel_50pct"]),
                            bool(row["sel_10pct"]),
                            bool(row["sel_1pct"]),
                            vector_literal(passages[i]),
                        )
                    )
                execute_values(cursor, insert_sql, rows, page_size=BATCH_SIZE, template=(
                    "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)"
                ))
                connection.commit()

        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE INDEX {INDEX_NAME} ON {TABLE_NAME}
                USING hnsw (embedding vector_ip_ops)
                WITH (m = 16, ef_construction = 200)
                """
            )
        connection.commit()

        build_time_s = time.perf_counter() - start
        postgres_ram_mb, docker_ram_mb, docker_cpu_pct = get_postgres_and_docker_stats(connection)
        build_df = pd.DataFrame(
            [
                {
                    "backend": "postgres",
                    "dataset": DATASET_SUFFIX,
                    "num_docs": len(metadata),
                    "index_build_s": round(build_time_s, 3),
                    "postgres_ram_mb": round(postgres_ram_mb, 2),
                    "docker_ram_mb": round(docker_ram_mb, 2),
                    "docker_cpu_pct": round(docker_cpu_pct, 2),
                }
            ]
        )

        out_csv = RESULTS_DIR / f"postgres_index_build_{DATASET_SUFFIX}.csv"
        build_df.to_csv(out_csv, index=False)
        print("\n[3/3] Index Build Complete:")
        print(build_df.to_string(index=False))
        print(f"Saved -> {out_csv}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
