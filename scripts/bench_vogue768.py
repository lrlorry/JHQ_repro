"""
Full benchmark on Vogue-768: JQ+IVF, JHQ+IVF, FAISS IVFPQ variants.
Writes results/vogue768_results.csv, then calls plot_comparison.py.

Run from repo root:
    python3 scripts/bench_vogue768.py
"""
import subprocess, sys, os, time, csv
import numpy as np
import faiss

OUT_DIR  = "results"
DATA_DIR = "data/vogue768"
BUILD    = "build/examples"
os.makedirs(OUT_DIR, exist_ok=True)

BASE_F  = f"{DATA_DIR}/vogue768_base.fvecs"
QUERY_F = f"{DATA_DIR}/vogue768_query.fvecs"
GT_F    = f"{DATA_DIR}/vogue768_groundtruth.ivecs"

# ── fvecs / ivecs helpers ────────────────────────────────────────────────────
def read_fvecs(path):
    with open(path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.int32)
    d = raw[0]; return raw.reshape(-1, d+1)[:, 1:].view(np.float32).copy()

def read_ivecs(path):
    with open(path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.int32)
    d = raw[0]; return raw.reshape(-1, d+1)[:, 1:].copy()

def recall_at_k(I_pred, I_gt, k):
    hits = sum(
        len(set(I_pred[i, :k]) & set(I_gt[i, :k]))
        for i in range(len(I_pred))
    )
    return hits / (len(I_pred) * k)

print("Loading data …")
xb = read_fvecs(BASE_F);  print(f"  base  {xb.shape}")
xq = read_fvecs(QUERY_F); print(f"  query {xq.shape}")
gt = read_ivecs(GT_F);    print(f"  gt    {gt.shape}")
d  = xb.shape[1]  # 768

NPROBES    = [1, 2, 4, 8, 16, 32, 64, 128]
K          = 10
NLIST      = 1024
RESULTS    = []

# ─────────────────────────────────────────────────────────────────────────────
# 1. FAISS IVFPQ variants
# ─────────────────────────────────────────────────────────────────────────────
def bench_faiss_ivfpq(M, label):
    print(f"\n{'='*60}")
    print(f"{label}  (M={M}, nbits=8, {M} bytes/vec)")
    quant = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quant, d, NLIST, M, 8)
    t0 = time.time()
    index.train(xb[:65536])
    index.add(xb)
    build = time.time() - t0
    print(f"  build: {build:.2f}s")
    for np_ in NPROBES:
        if np_ > NLIST: break
        index.nprobe = np_
        t1 = time.time()
        _, I = index.search(xq, K)
        elapsed = time.time() - t1
        rec = recall_at_k(I, gt, K)
        qps = len(xq) / elapsed
        print(f"  nprobe={np_:<4}  recall={rec:.4f}  QPS={qps:.1f}")
        RESULTS.append(dict(method=label, nprobe=np_, recall=rec,
                            qps=qps, build_time=build))

# Valid M for d=768 with B=8: need 8%(d/M)==0 → Ds∈{1,2,4,8} → M∈{96,192,384,768}
# JQ:  M=96, 96 bytes/vec (96 subspaces × 8 bits)
# JHQ: M=96 primary (96B) + d×Br/8=768×4/8=384B residual = 480B total
bench_faiss_ivfpq(M=96,  label="FAISS-IVFPQ-96B")   # same budget as JQ
bench_faiss_ivfpq(M=192, label="FAISS-IVFPQ-192B")  # 2× JQ budget
bench_faiss_ivfpq(M=384, label="FAISS-IVFPQ-384B")  # ~JHQ residual-only budget

# ─────────────────────────────────────────────────────────────────────────────
# 2. Our JQ+IVF  (M=96, B=8, Ds=8, K1D=2 → 8 bytes×96=768 bits primary)
# ─────────────────────────────────────────────────────────────────────────────
def run_binary(exe, args, label, build_time_key):
    """Run demo binary, parse stdout, return list of (nprobe,recall,qps) + build_time."""
    cmd = [exe] + args
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"  cmd: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        print(f"ERROR:\n{proc.stderr[:2000]}")
        return None, None
    out = proc.stdout
    print(out[:500])
    rows = []
    build_time = None
    for line in out.splitlines():
        if "Index build:" in line:
            build_time = float(line.split(":")[1].strip().rstrip("s"))
        parts = line.split()
        if len(parts) == 3:
            try:
                np_, rec, qps = int(parts[0]), float(parts[1]), float(parts[2])
                rows.append((np_, rec, qps))
            except ValueError:
                pass
    return rows, build_time

# JQ+IVF: M=96, B=8, Ds=8, K1D=2 → same "byte budget" as FAISS IVFPQ M=96
rows, bt = run_binary(
    f"./{BUILD}/demo_jq_ivf",
    [BASE_F, QUERY_F, GT_F, "96", "8", str(NLIST)],
    label="JQ+IVF",
    build_time_key="JQ+IVF",
)
if rows:
    for np_, rec, qps in rows:
        RESULTS.append(dict(method="JQ+IVF", nprobe=np_, recall=rec,
                            qps=qps, build_time=bt))

# JHQ+IVF: M=96, B=8, Br=4, alpha=4.0
rows, bt = run_binary(
    f"./{BUILD}/demo_jhq_ivf",
    [BASE_F, QUERY_F, GT_F, "96", "8", "4", str(NLIST), "4.0"],
    label="JHQ+IVF",
    build_time_key="JHQ+IVF",
)
if rows:
    for np_, rec, qps in rows:
        RESULTS.append(dict(method="JHQ+IVF", nprobe=np_, recall=rec,
                            qps=qps, build_time=bt))

# ─────────────────────────────────────────────────────────────────────────────
# 3. Save CSV
# ─────────────────────────────────────────────────────────────────────────────
csv_path = f"{OUT_DIR}/vogue768_results.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method","nprobe","recall","qps","build_time"])
    w.writeheader()
    w.writerows(RESULTS)
print(f"\nResults saved to {csv_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Plot
# ─────────────────────────────────────────────────────────────────────────────
subprocess.run(["python3", "scripts/plot_comparison.py", csv_path])
