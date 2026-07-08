"""
FAISS IVFPQ baseline comparison for SIFT1M.
Compares: FAISS IVFPQ-16 (same bytes as our JQ) and IVFPQ-80 (same bytes as our JHQ).
"""
import numpy as np
import faiss
import time
import sys

# ── helpers ──────────────────────────────────────────────────────────────────

def read_fvecs(path):
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.int32)
    d = data[0]
    # each row: [d, x0, x1, ..., x_{d-1}]
    data = data.reshape(-1, d + 1)[:, 1:].view(np.float32).copy()
    return data

def read_ivecs(path):
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.int32)
    d = data[0]
    data = data.reshape(-1, d + 1)[:, 1:].copy()
    return data

def recall_at_k(I_pred, I_gt, k):
    """Standard Recall@k: fraction of true top-k found in predicted top-k."""
    hits = 0
    nq = I_pred.shape[0]
    for i in range(nq):
        gt_set = set(I_gt[i, :k])
        hits += len(set(I_pred[i, :k]) & gt_set)
    return hits / (nq * k)

# ── load data ─────────────────────────────────────────────────────────────────

base_path  = "data/sift/sift_base.fvecs"
query_path = "data/sift/sift_query.fvecs"
gt_path    = "data/sift/sift_groundtruth.ivecs"

print("Loading data...")
xb = read_fvecs(base_path);  print(f"  base:  {xb.shape}")
xq = read_fvecs(query_path); print(f"  query: {xq.shape}")
gt = read_ivecs(gt_path);    print(f"  gt:    {gt.shape}")

d = xb.shape[1]   # 128

# ── run one IVFPQ configuration ───────────────────────────────────────────────

def bench_ivfpq(nlist, M, nbits, label):
    print(f"\n{'='*60}")
    print(f"{label}: IVFPQ  nlist={nlist}  M={M}  nbits={nbits}  "
          f"({M*nbits//8} bytes/vec)")
    print(f"{'='*60}")

    quant = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quant, d, nlist, M, nbits)

    # Train
    t0 = time.time()
    train_vecs = xb[:65536]          # same budget as our k-means++ (max_train_n)
    index.train(train_vecs)
    index.add(xb)
    build_time = time.time() - t0
    print(f"Index build: {build_time:.2f} s")

    print(f"\n{'nprobe':<10} {'Recall@10':<14} {'QPS':<10}")
    print(f"{'------':<10} {'---------':<14} {'---':<10}")

    k = 10
    for nprobe in [1, 2, 4, 8, 16, 32, 64, 128]:
        if nprobe > nlist:
            break
        index.nprobe = nprobe
        t0 = time.time()
        _, I = index.search(xq, k)
        elapsed = time.time() - t0
        rec = recall_at_k(I, gt, k)
        qps = len(xq) / elapsed
        print(f"{nprobe:<10} {rec:<14.4f} {qps:<10.1f}")

# ── JQ-equivalent: M=16 × 8 bits = 16 bytes  (same as our JQ+IVF) ─────────
bench_ivfpq(nlist=1024, M=16, nbits=8, label="FAISS-IVFPQ (16 bytes, same as JQ)")

# ── JHQ-equivalent: M=64 × 8 bits = 64 bytes (closest valid to JHQ's 80 bytes) ──
# Our JHQ: 16 bytes primary + 64 bytes residual (d=128, Br=4) = 80 bytes total
# FAISS PQ requires d%M==0; M=64 (64 bytes) is the closest valid value.
bench_ivfpq(nlist=1024, M=64, nbits=8, label="FAISS-IVFPQ (64 bytes, ~JHQ byte budget)")
