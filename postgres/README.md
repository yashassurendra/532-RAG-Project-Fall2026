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

Generate the latency/QPS comparison and PostgreSQL resource plots with:

```sh
python postgres/plot_latency_qps.py
python postgres/plot_resources.py
python postgres/plot_recall.py
```

The latency/QPS and recall plots compare available backend summary CSVs. The
resource plot uses the PostgreSQL summary and index-build CSVs. These scripts
read result CSVs and do not require the Redis Python client.

`POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, and
`POSTGRES_PASSWORD` configure the connection. `DATASET_SUFFIX` selects the dataset;
otherwise, the scripts use the most recently modified `passages_*.npy` file.
Results are written to `results/` using the `postgres_*` CSV names consumed by
these plotting scripts.

The PostgreSQL memory metric is the number of used database buffer-cache pages.
Docker memory and CPU are sampled from the `rag_postgres` container.
