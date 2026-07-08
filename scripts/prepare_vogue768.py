"""
Download and preprocess Vogue-768 dataset for ANN benchmarking.
Output: data/vogue768/{base,query,groundtruth}.{fvecs,ivecs}
"""
import numpy as np
import struct
import time
import os
import sys

OUT_DIR = "data/vogue768"
os.makedirs(OUT_DIR, exist_ok=True)

def write_fvecs(path, vecs):
    vecs = np.asarray(vecs, dtype=np.float32)
    with open(path, "wb") as f:
        for v in vecs:
            f.write(struct.pack("i", len(v)))
            f.write(v.tobytes())
    print(f"  wrote {len(vecs):,} × {vecs.shape[1]} → {path}")

def write_ivecs(path, vecs):
    vecs = np.asarray(vecs, dtype=np.int32)
    with open(path, "wb") as f:
        for v in vecs:
            f.write(struct.pack("i", len(v)))
            f.write(v.tobytes())
    print(f"  wrote {len(vecs):,} × {vecs.shape[1]} → {path}")

# ── Step 1: Download ─────────────────────────────────────────────────────────
print("Step 1: Downloading tonyassi/vogue933k-embeddings …")
t0 = time.time()

try:
    from datasets import load_dataset
    ds = load_dataset("tonyassi/vogue933k-embeddings", split="train",
                      cache_dir="/tmp/hf_cache")
    print(f"  loaded {len(ds):,} rows in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# ── Step 2: Extract embeddings ───────────────────────────────────────────────
print("Step 2: Extracting embeddings …")
t1 = time.time()

import ast

def to_vec(x):
    if isinstance(x, (list, np.ndarray)):
        return np.array(x, dtype=np.float32)
    if isinstance(x, str):
        return np.array(ast.literal_eval(x), dtype=np.float32)
    raise ValueError(type(x))

sample = to_vec(ds[0]["embedding"])
d = len(sample)
print(f"  dimension d = {d}")

n_total = len(ds)
embeddings = np.zeros((n_total, d), dtype=np.float32)
for i, row in enumerate(ds):
    embeddings[i] = to_vec(row["embedding"])
    if i % 100000 == 0 and i > 0:
        print(f"  {i:,}/{n_total:,} …")

print(f"  extracted {n_total:,} vectors in {time.time()-t1:.1f}s")

# L2-normalize (paper uses normalized embeddings)
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms = np.where(norms == 0, 1.0, norms)
embeddings /= norms
print(f"  L2-normalized")

# ── Step 3: Train/query split ────────────────────────────────────────────────
print("Step 3: Creating train/query split (1000 queries) …")
rng = np.random.default_rng(42)
query_idx = rng.choice(n_total, size=1000, replace=False)
mask = np.ones(n_total, dtype=bool)
mask[query_idx] = False

base   = embeddings[mask]
queries = embeddings[query_idx]
print(f"  base={len(base):,}  queries={len(queries):,}")

# ── Step 4: Ground truth (top-100 exact NN) ──────────────────────────────────
print("Step 4: Computing ground truth with FAISS brute-force …")
t2 = time.time()
import faiss
index_flat = faiss.IndexFlatL2(d)
index_flat.add(base)
k_gt = 100
_, gt = index_flat.search(queries, k_gt)
print(f"  GT computed in {time.time()-t2:.1f}s")

# ── Step 5: Save ─────────────────────────────────────────────────────────────
print("Step 5: Saving …")
write_fvecs(f"{OUT_DIR}/vogue768_base.fvecs", base)
write_fvecs(f"{OUT_DIR}/vogue768_query.fvecs", queries)
write_ivecs(f"{OUT_DIR}/vogue768_groundtruth.ivecs", gt)

print(f"\nDone in {time.time()-t0:.1f}s total.")
print(f"Files in {OUT_DIR}/")
