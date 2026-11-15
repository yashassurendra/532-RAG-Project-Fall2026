# FAISS backend

The FAISS backend loads the canonical artifacts from `data/processed/` and
uses exact inner-product search over normalized embeddings.

Run these commands from the repository root:

```sh
python faiss/evaluate_faiss.py
python faiss/evaluate_faiss_concurrency.py
```

Midpoint benchmark CSV files are written to `results/`; concurrency CSV files
are written to `metrics/`. Generate the canonical cross-backend checkpoint
figures from the repository root after all three backends have run:

```sh
python plot_midpoint.py --suffix 100k
```

Midpoint results and figures are kept for review, while exploratory metrics are
ignored by Git.

The `faiss/` directory intentionally has no `__init__.py`; this prevents it
from shadowing the installed `faiss-cpu` package used by `faiss_backend.py`.
