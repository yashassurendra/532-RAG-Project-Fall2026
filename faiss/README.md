# FAISS backend

The FAISS backend loads the canonical artifacts from `data/processed/` and
uses exact inner-product search over normalized embeddings.

Run these commands from the repository root:

```sh
python faiss/evaluate_faiss.py
python faiss/evaluate_faiss_concurrency.py
python faiss/generate_faiss_graphs.py
```

Midpoint benchmark CSV files are written to `results/`; concurrency CSV files
are written to `metrics/`; and backend-specific charts are written to `graphs/`.
The midpoint results are kept for review, while `metrics/` and `graphs/` are
ignored by Git.

The `faiss/` directory intentionally has no `__init__.py`; this prevents it
from shadowing the installed `faiss-cpu` package used by `faiss_backend.py`.
