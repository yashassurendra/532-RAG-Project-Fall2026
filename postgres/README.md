# PostgreSQL + pgvector

This backend mirrors the Redis workflow: it loads the generated vectors and metadata,
builds an inner-product HNSW index, and benchmarks unfiltered and 50%, 10%, and 1%
filtered top-10 retrieval.

Start PostgreSQL from the repository root:

```sh
docker compose up -d postgres
```

Install the Python dependencies if they are not already available:

```sh
python -m pip install numpy pandas pyarrow psycopg2-binary matplotlib
```

From the repository root, run:

```sh
python postgres/load_and_index.py
python postgres/bench_selectivity.py
```

After running all three backend benchmarks, generate the canonical comparison
figures from the repository root with:

```sh
python plot_midpoint.py --suffix 100k
```

This reads the standardized FAISS, Redis, and PostgreSQL summary and index-build
CSVs and writes the latency/QPS, Recall@10, and resource figures to `results/`.

`POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, and
`POSTGRES_PASSWORD` configure the connection. `DATASET_SUFFIX` selects the dataset;
otherwise, the scripts use the most recently modified `passages_*.npy` file.
Results are written to `results/` using the `postgres_*` CSV names consumed by
these plotting scripts.

The PostgreSQL memory metric is the number of used database buffer-cache pages.
Docker memory and CPU are sampled from the `rag_postgres` container.
