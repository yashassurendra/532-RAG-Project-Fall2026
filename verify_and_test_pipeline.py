import os
import numpy as np
import pandas as pd
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
# Keep True (1,000 rows) to verify the entire pipeline in ~5 seconds.
# Change to False (100,000 rows) once all 8 checks pass!
TEST_RUN = False
TARGET_N = 1_000 if TEST_RUN else 100_000
K = 10
SEED = 42

def get_data_path():
    for candidate in [os.path.join("data", "trec-covid"), "data"]:
        if os.path.exists(os.path.join(candidate, "corpus.jsonl")):
            return candidate
    raise FileNotFoundError("Could not find corpus.jsonl in 'data/' or 'data/trec-covid/'")

def main():
    # 1. Verify Raw BEIR Files
    data_path = get_data_path()
    print(f"[1/5] Loading raw BEIR dataset from '{data_path}'...")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    assert len(corpus) == 171_332, f"Expected 171,332 passages, got {len(corpus)}"
    assert len(queries) == 50, f"Expected 50 queries, got {len(queries)}"
    assert len(qrels) == 50, f"Expected 50 qrel entries, got {len(qrels)}"
    print(f"      Raw check passed: {len(corpus):,} passages, {len(queries)} queries.")

    # 2. Deterministically Slice Corpus (Prioritizing Qrel Relevant Passages)
    print(f"[2/5] Slicing {TARGET_N:,} passages...")
    judged_docs = [
        did for qid in sorted(qrels.keys())
        for did, score in qrels[qid].items()
        if score > 0 and did in corpus
    ]
    ordered_doc_ids = list(dict.fromkeys(judged_docs))
    seen = set(ordered_doc_ids)

    for did in corpus:
        if len(ordered_doc_ids) >= TARGET_N:
            break
        if did not in seen:
            ordered_doc_ids.append(did)
            seen.add(did)

    ordered_doc_ids = ordered_doc_ids[:TARGET_N]
    n_docs = len(ordered_doc_ids)

    passage_texts = [
        (corpus[did].get("title", "") + " " + corpus[did].get("text", "")).strip()
        for did in ordered_doc_ids
    ]
    query_ids = sorted(list(queries.keys()), key=lambda x: int(x) if x.isdigit() else x)
    query_texts = [queries[qid] for qid in query_ids]

    # 3. Generate 384-dim Embeddings (all-MiniLM-L6-v2)
    print(f"[3/5] Embedding {n_docs:,} passages & {len(query_ids)} queries with all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    passage_vecs = model.encode(
        passage_texts, batch_size=256, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)

    query_vecs = model.encode(
        query_texts, batch_size=64, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)

    # 4. Generate Seeded Synthetic Metadata
    print("[4/5] Generating seeded synthetic metadata & selectivity buckets...")
    rng = np.random.default_rng(seed=SEED)
    # Shuffle exact index positions so 1%, 10%, and 50% have exact counts at any N
    perm = rng.permutation(n_docs)
    sel_1pct = np.zeros(n_docs, dtype=bool)
    sel_10pct = np.zeros(n_docs, dtype=bool)
    sel_50pct = np.zeros(n_docs, dtype=bool)

    sel_1pct[perm[: max(10, int(0.01 * n_docs))]] = True
    sel_10pct[perm[: max(10, int(0.10 * n_docs))]] = True
    sel_50pct[perm[: max(10, int(0.50 * n_docs))]] = True

    selectivity_bucket = np.where(
        sel_1pct, "1pct", np.where(sel_10pct, "10pct", np.where(sel_50pct, "50pct", "100pct"))
    )

    meta_df = pd.DataFrame({
        "row_idx": np.arange(n_docs, dtype=np.int64),
        "doc_id": ordered_doc_ids,
        "category": rng.choice(["eng", "sales", "hr", "legal", "ops"], size=n_docs),
        "access_group": rng.integers(1, 11, size=n_docs, dtype=np.int32),
        "timestamp": rng.integers(1704067200, 1735689600, size=n_docs, dtype=np.int64),
        "selectivity_bucket": selectivity_bucket,
        "sel_50pct": sel_50pct,
        "sel_10pct": sel_10pct,
        "sel_1pct": sel_1pct,
    })

    out_dir = os.path.join("data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    suffix = "test1k" if TEST_RUN else "100k"

    passages_path = os.path.join(out_dir, f"passages_{suffix}.npy")
    queries_path = os.path.join(out_dir, f"queries_{suffix}.npy")
    meta_path = os.path.join(out_dir, f"metadata_{suffix}.parquet")

    np.save(passages_path, passage_vecs)
    np.save(queries_path, query_vecs)
    meta_df.to_parquet(meta_path, index=False)

    # 5. Compute Brute-Force Ground Truth (Unfiltered, 50%, 10%, 1%)
    print("[5/5] Computing brute-force top-10 ground truth...")
    sim_matrix = query_vecs @ passage_vecs.T  # Exact cosine similarity (N_queries, N_docs)

    filter_masks = {
        "unfiltered": np.ones(n_docs, dtype=bool),
        "50pct": meta_df["sel_50pct"].values,
        "10pct": meta_df["sel_10pct"].values,
        "1pct": meta_df["sel_1pct"].values,
    }

    gt_Paths = {}
    for fname, mask in filter_masks.items():
        masked_scores = np.where(mask[None, :], sim_matrix, -np.inf)
        top_k_indices = np.argsort(-masked_scores, axis=1)[:, :K].astype(np.int64)
        gt_file = os.path.join(out_dir, f"gt_top10_{fname}_{suffix}.npy")
        np.save(gt_file, top_k_indices)
        gt_Paths[fname] = gt_file

    # --- FULL VERIFICATION SUITE (8 CHECKS) ---
    print("\n=== RUNNING 8-POINT ARTIFACT VERIFICATION ===")
    vp = np.load(passages_path)
    vq = np.load(queries_path)
    vm = pd.read_parquet(meta_path)

    # Check 1: Passage embedding dimensions & dtype
    assert vp.shape == (n_docs, 384) and vp.dtype == np.float32
    print(f"PASS [1/8] Passage vectors shape={vp.shape}, dtype={vp.dtype}")

    # Check 2: Query embedding dimensions & dtype
    assert vq.shape == (50, 384) and vq.dtype == np.float32
    print(f"PASS [2/8] Query vectors shape={vq.shape}, dtype={vq.dtype}")

    # Check 3: No NaNs or Infs
    assert not np.isnan(vp).any() and not np.isinf(vp).any()
    assert not np.isnan(vq).any() and not np.isinf(vq).any()
    print("PASS [3/8] Zero NaNs or Infs in passage and query embeddings")

    # Check 4: Unit L2 Normalization (Inner Product == Cosine Similarity)
    p_norms = np.linalg.norm(vp, axis=1)
    q_norms = np.linalg.norm(vq, axis=1)
    assert np.allclose(p_norms, 1.0, atol=1e-4) and np.allclose(q_norms, 1.0, atol=1e-4)
    print(f"PASS [4/8] L2 normalization verified (mean norm={p_norms.mean():.6f})")

    # Check 5: Metadata alignment & required schema
    req_cols = {"row_idx", "doc_id", "category", "access_group", "timestamp", "selectivity_bucket", "sel_50pct", "sel_10pct", "sel_1pct"}
    assert req_cols.issubset(set(vm.columns))
    assert len(vm) == n_docs and (vm["row_idx"].values == np.arange(n_docs)).all()
    print(f"PASS [5/8] Metadata schema & 0..{n_docs-1} row_idx alignment verified")

    # Check 6: Exact Selectivity Ratios
    s50, s10, s1 = vm["sel_50pct"].mean(), vm["sel_10pct"].mean(), vm["sel_1pct"].mean()
    assert abs(s50 - 0.50) < 1e-4 and abs(s10 - 0.10) < 1e-4 and abs(s1 - 0.01) < 1e-4
    print(f"PASS [6/8] Filter selectivities exact: 50%={s50:.2%}, 10%={s10:.2%}, 1%={s1:.2%}")

    # Check 7: Ground-truth shapes & valid index bounds
    for fname, gpath in gt_Paths.items():
        gt = np.load(gpath)
        assert gt.shape == (50, K) and gt.dtype == np.int64
        assert gt.min() >= 0 and gt.max() < n_docs
        assert len({ len(set(row)) for row in gt }) == 1  # 10 unique IDs per query
    print("PASS [7/8] Ground-truth files have shape (50, 10) with unique valid row_idx values")

    # Check 8: Ground-truth filter compliance (every returned neighbor actually satisfies the filter!)
    for fname in ["50pct", "10pct", "1pct"]:
        gt = np.load(gt_Paths[fname])
        mask = vm[f"sel_{fname}"].values
        assert mask[gt].all(), f"Ground truth for {fname} returned an unfiltered row!"
    print("PASS [8/8] Every filtered ground-truth neighbor strictly satisfies its metadata predicate!")
    print(f"\nALL 8 CHECKS PASSED! Files saved in: {out_dir}")

if __name__ == "__main__":
    main()