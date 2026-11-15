"""Generate and verify the shared retrieval artifacts for every backend.

The ground truth is exact top-k cosine similarity over normalized embeddings.
FAISS, Redis, and PostgreSQL must all use the row_idx values written here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_NUM_PASSAGES = 100_000
DEFAULT_SEED = 42
DEFAULT_K = 10
EXPECTED_DIMENSION = 384
EXPECTED_QUERY_COUNT = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic BEIR embeddings, metadata, and exact ground truth."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing corpus.jsonl, queries.jsonl, and qrels/test.tsv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Artifact output directory (default: data/processed).",
    )
    parser.add_argument(
        "--num-passages",
        type=int,
        default=DEFAULT_NUM_PASSAGES,
        help=f"Number of passages to include (default: {DEFAULT_NUM_PASSAGES}).",
    )
    parser.add_argument(
        "--suffix",
        help="Filename suffix. Defaults to test1k for 1,000 rows or <N/1000>k.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", help="Optional pinned Hugging Face model revision.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def resolve_data_dir(explicit_path: Path | None) -> Path:
    candidates = [explicit_path] if explicit_path else [Path("data/trec-covid"), Path("data")]
    for candidate in candidates:
        if candidate and (candidate / "corpus.jsonl").is_file():
            return candidate.resolve()
    checked = ", ".join(str(path) for path in candidates if path)
    raise FileNotFoundError(f"Could not find corpus.jsonl. Checked: {checked}")


def format_suffix(num_passages: int) -> str:
    if num_passages <= 0:
        raise ValueError("num_passages must be positive")
    if num_passages == 1_000:
        return "test1k"
    if num_passages % 1_000 == 0:
        return f"{num_passages // 1_000}k"
    return str(num_passages)


def load_beir_dataset(data_dir: Path) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    try:
        from beir.datasets.data_loader import GenericDataLoader
    except ImportError as exc:
        raise RuntimeError(
            "BEIR is not installed. Run: python3 -m pip install -r requirements-ground-truth.txt"
        ) from exc

    return GenericDataLoader(data_folder=str(data_dir)).load(split="test")


def select_document_ids(
    corpus: dict[str, Any], qrels: dict[str, dict[str, int]], num_passages: int
) -> list[str]:
    """Select documents deterministically, prioritizing positively judged documents."""
    if num_passages > len(corpus):
        raise ValueError(f"Requested {num_passages:,} passages, but corpus has {len(corpus):,}")

    judged_ids = [
        doc_id
        for query_id in sorted(qrels)
        for doc_id, score in qrels[query_id].items()
        if score > 0 and doc_id in corpus
    ]
    ordered_ids = list(dict.fromkeys(judged_ids))
    seen = set(ordered_ids)

    for doc_id in corpus:
        if len(ordered_ids) >= num_passages:
            break
        if doc_id not in seen:
            ordered_ids.append(doc_id)
            seen.add(doc_id)

    return ordered_ids[:num_passages]


def query_sort_key(query_id: str) -> tuple[int, int | str]:
    return (0, int(query_id)) if query_id.isdigit() else (1, query_id)


def embed_texts(
    passage_texts: list[str],
    query_texts: list[str],
    model_name: str,
    model_revision: str | None,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Run: "
            "python3 -m pip install -r requirements-ground-truth.txt"
        ) from exc

    model_kwargs = {"revision": model_revision} if model_revision else {}
    model = SentenceTransformer(model_name, **model_kwargs)
    passage_vectors = model.encode(
        passage_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    query_vectors = model.encode(
        query_texts,
        batch_size=min(batch_size, 64),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    return passage_vectors, query_vectors


def generate_selectivity_masks(num_documents: int, seed: int) -> dict[str, np.ndarray]:
    if num_documents < 1_000:
        raise ValueError("At least 1,000 documents are required for an exact 1% filter with top-10")

    permutation = np.random.default_rng(seed).permutation(num_documents)
    masks = {"unfiltered": np.ones(num_documents, dtype=bool)}
    for percentage in (50, 10, 1):
        mask = np.zeros(num_documents, dtype=bool)
        mask[permutation[: num_documents * percentage // 100]] = True
        masks[f"{percentage}pct"] = mask
    return masks


def build_metadata(document_ids: list[str], masks: dict[str, np.ndarray], seed: int):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas/pyarrow are not installed. Run: "
            "python3 -m pip install -r requirements-ground-truth.txt"
        ) from exc

    num_documents = len(document_ids)
    rng = np.random.default_rng(seed)
    # Match the original pipeline's RNG sequence: metadata values are generated
    # after the permutation used to construct the selectivity masks.
    rng.permutation(num_documents)
    selectivity_bucket = np.where(
        masks["1pct"],
        "1pct",
        np.where(masks["10pct"], "10pct", np.where(masks["50pct"], "50pct", "100pct")),
    )
    return pd.DataFrame(
        {
            "row_idx": np.arange(num_documents, dtype=np.int64),
            "doc_id": document_ids,
            "category": rng.choice(["eng", "sales", "hr", "legal", "ops"], num_documents),
            "access_group": rng.integers(1, 11, size=num_documents, dtype=np.int32),
            "timestamp": rng.integers(
                1_704_067_200, 1_735_689_600, size=num_documents, dtype=np.int64
            ),
            "selectivity_bucket": selectivity_bucket,
            "sel_50pct": masks["50pct"],
            "sel_10pct": masks["10pct"],
            "sel_1pct": masks["1pct"],
        }
    )


def compute_exact_ground_truth(
    query_vectors: np.ndarray,
    passage_vectors: np.ndarray,
    masks: dict[str, np.ndarray],
    k: int,
) -> dict[str, np.ndarray]:
    """Return exact top-k row indices for each filter using cosine/inner product."""
    if query_vectors.ndim != 2 or passage_vectors.ndim != 2:
        raise ValueError("query_vectors and passage_vectors must be two-dimensional")
    if query_vectors.shape[1] != passage_vectors.shape[1]:
        raise ValueError("Query and passage dimensions do not match")
    if k <= 0:
        raise ValueError("k must be positive")

    similarities = query_vectors @ passage_vectors.T
    results: dict[str, np.ndarray] = {}
    for filter_name, mask in masks.items():
        if mask.shape != (len(passage_vectors),):
            raise ValueError(f"Invalid mask shape for {filter_name}: {mask.shape}")
        if int(mask.sum()) < k:
            raise ValueError(f"Filter {filter_name} has fewer than k={k} candidates")
        masked_scores = np.where(mask[None, :], similarities, -np.inf)
        results[filter_name] = np.argsort(
            -masked_scores, axis=1, kind="stable"
        )[:, :k].astype(np.int64)
    return results


def validate_artifacts(
    passage_vectors: np.ndarray,
    query_vectors: np.ndarray,
    metadata,
    masks: dict[str, np.ndarray],
    ground_truth: dict[str, np.ndarray],
    k: int,
) -> None:
    num_documents = len(passage_vectors)
    assert passage_vectors.shape == (num_documents, EXPECTED_DIMENSION)
    assert query_vectors.shape == (EXPECTED_QUERY_COUNT, EXPECTED_DIMENSION)
    assert passage_vectors.dtype == query_vectors.dtype == np.float32
    assert np.isfinite(passage_vectors).all() and np.isfinite(query_vectors).all()
    assert np.allclose(np.linalg.norm(passage_vectors, axis=1), 1.0, atol=1e-4)
    assert np.allclose(np.linalg.norm(query_vectors, axis=1), 1.0, atol=1e-4)
    assert len(metadata) == num_documents
    assert np.array_equal(metadata["row_idx"].to_numpy(), np.arange(num_documents))

    for percentage in (50, 10, 1):
        name = f"{percentage}pct"
        assert int(masks[name].sum()) == num_documents * percentage // 100
        assert np.array_equal(metadata[f"sel_{name}"].to_numpy(), masks[name])

    assert np.all(masks["1pct"] <= masks["10pct"])
    assert np.all(masks["10pct"] <= masks["50pct"])
    for name, indices in ground_truth.items():
        assert indices.shape == (len(query_vectors), k)
        assert indices.dtype == np.int64
        assert indices.min() >= 0 and indices.max() < num_documents
        assert all(len(set(row)) == k for row in indices)
        assert masks[name][indices].all()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.k <= 0 or args.batch_size <= 0:
        raise ValueError("k and batch-size must be positive")

    data_dir = resolve_data_dir(args.data_dir)
    suffix = args.suffix or format_suffix(args.num_passages)
    print(f"[1/6] Loading BEIR data from {data_dir}")
    corpus, queries, qrels = load_beir_dataset(data_dir)
    if len(queries) != EXPECTED_QUERY_COUNT:
        raise ValueError(f"Expected {EXPECTED_QUERY_COUNT} queries, found {len(queries)}")

    print(f"[2/6] Selecting {args.num_passages:,} passages deterministically")
    document_ids = select_document_ids(corpus, qrels, args.num_passages)
    query_ids = sorted(queries, key=query_sort_key)
    passage_texts = [
        f"{corpus[doc_id].get('title', '')} {corpus[doc_id].get('text', '')}".strip()
        for doc_id in document_ids
    ]
    query_texts = [queries[query_id] for query_id in query_ids]

    print(f"[3/6] Embedding passages and queries with {args.model}")
    passage_vectors, query_vectors = embed_texts(
        passage_texts, query_texts, args.model, args.model_revision, args.batch_size
    )

    print(f"[4/6] Generating metadata with seed={args.seed}")
    masks = generate_selectivity_masks(len(document_ids), args.seed)
    metadata = build_metadata(document_ids, masks, args.seed)

    print(f"[5/6] Computing exact top-{args.k} ground truth")
    ground_truth = compute_exact_ground_truth(query_vectors, passage_vectors, masks, args.k)
    validate_artifacts(
        passage_vectors, query_vectors, metadata, masks, ground_truth, args.k
    )

    print(f"[6/6] Writing verified artifacts to {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "passages": args.output_dir / f"passages_{suffix}.npy",
        "queries": args.output_dir / f"queries_{suffix}.npy",
        "metadata": args.output_dir / f"metadata_{suffix}.parquet",
    }
    np.save(artifact_paths["passages"], passage_vectors)
    np.save(artifact_paths["queries"], query_vectors)
    metadata.to_parquet(artifact_paths["metadata"], index=False)
    for filter_name, indices in ground_truth.items():
        path = args.output_dir / f"gt_top{args.k}_{filter_name}_{suffix}.npy"
        np.save(path, indices)
        artifact_paths[f"ground_truth_{filter_name}"] = path

    manifest = {
        "schema_version": 1,
        "dataset": "BEIR/trec-covid",
        "num_passages": len(document_ids),
        "num_queries": len(query_ids),
        "embedding_dimension": passage_vectors.shape[1],
        "embedding_dtype": str(passage_vectors.dtype),
        "normalized": True,
        "model": args.model,
        "model_revision": args.model_revision,
        "seed": args.seed,
        "k": args.k,
        "suffix": suffix,
        "query_ids": query_ids,
        "artifacts": {
            name: {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in sorted(artifact_paths.items())
        },
    }
    manifest_path = args.output_dir / f"manifest_{suffix}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"All checks passed. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
