# Shared ground-truth artifacts

This pipeline creates the single canonical artifact set used by FAISS, Redis,
and PostgreSQL. Ground truth is exact top-10 cosine similarity over normalized
`all-MiniLM-L6-v2` embeddings. Backend search output must be compared using the
shared `row_idx` values in the metadata.

## Setup

Use Python 3.11 or 3.12 and create a virtual environment:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-ground-truth.txt
```

Put the BEIR TREC-COVID files under `data/trec-covid/`:

```text
data/trec-covid/corpus.jsonl
data/trec-covid/queries.jsonl
data/trec-covid/qrels/test.tsv
```

## Quick verification run

```sh
python verify_and_test_pipeline.py --num-passages 1000
```

This writes `test1k` artifacts under `data/processed/`.

## Canonical 100k run

```sh
python verify_and_test_pipeline.py --num-passages 100000
```

The required output is:

```text
passages_100k.npy
queries_100k.npy
metadata_100k.parquet
gt_top10_unfiltered_100k.npy
gt_top10_50pct_100k.npy
gt_top10_10pct_100k.npy
gt_top10_1pct_100k.npy
manifest_100k.json
```

The manifest records configuration, shapes, file sizes, and SHA-256 checksums.
Team members should compare manifests before benchmarking. Do not commit the
generated dataset; `passages_100k.npy` alone is larger than GitHub's 100 MB
single-file limit.
