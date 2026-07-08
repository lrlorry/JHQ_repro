"""
Evaluate JHQ reproduction results.

Parses JHQ binary output files, computes recall@10 and QPS,
and plots the speed-accuracy trade-off curve (replicating paper Figure 4).

Usage:
    python scripts/evaluate.py \
        --results_dir results/openai-1024-100k \
        --groundtruth datasets/openai-1024-100k_groundtruth.ivecs \
        --k 10 \
        --plot
"""

import argparse
import glob
import json
import os
import struct

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def read_ivecs(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        data = f.read()
    d = struct.unpack_from("i", data, 0)[0]
    n = len(data) // (4 + 4 * d)
    arr = np.frombuffer(data, dtype=np.int32).reshape(n, d + 1)
    return np.ascontiguousarray(arr[:, 1:])


def read_results_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_k(pred: np.ndarray, gt: np.ndarray, k: int = 10) -> float:
    """
    pred: (n_queries, k_pred)  — retrieved neighbor IDs
    gt:   (n_queries, k_gt)    — ground-truth neighbor IDs
    """
    assert pred.shape[0] == gt.shape[0], "Query count mismatch"
    gt_k = gt[:, :k]
    hits = 0
    for i in range(len(pred)):
        hits += len(set(pred[i]) & set(gt_k[i]))
    return hits / (len(pred) * k)


# ---------------------------------------------------------------------------
# Result loading
#
# JHQ binary outputs one JSON file per (algorithm, parameter_config):
#   {
#     "algorithm": "JHQ",
#     "dataset": "openai-1024-100k",
#     "params": {"M": 128, "B": 8, "Br": 4, "nprobe": 32, "alpha": 4.0},
#     "qps": 2345.6,
#     "recall_at_10": 0.934,
#     "index_time_sec": 12.3,
#     "neighbors": [[...], ...]   // optional: raw neighbor IDs
#   }
# ---------------------------------------------------------------------------

def load_all_results(results_dir: str) -> list[dict]:
    results = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        r = read_results_json(path)
        results.append(r)
    return results


def recompute_recall(results: list[dict], gt: np.ndarray, k: int = 10) -> list[dict]:
    """If raw neighbor IDs are stored, recompute recall from scratch."""
    for r in results:
        if "neighbors" in r:
            pred = np.array(r["neighbors"], dtype=np.int32)
            r["recall_at_10"] = recall_at_k(pred, gt, k)
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_table(results: list[dict], k: int = 10) -> None:
    algos = sorted(set(r["algorithm"] for r in results))
    print(f"\n{'Algorithm':<20} {'Params':<40} {'Recall@'+str(k):<12} {'QPS':<10} {'IndexTime(s)'}")
    print("-" * 100)
    for algo in algos:
        rows = [r for r in results if r["algorithm"] == algo]
        rows.sort(key=lambda r: r.get("recall_at_10", 0))
        for r in rows:
            params = ", ".join(f"{k}={v}" for k, v in r.get("params", {}).items())
            recall = r.get("recall_at_10", float("nan"))
            qps = r.get("qps", float("nan"))
            idx_t = r.get("index_time_sec", float("nan"))
            print(f"{algo:<20} {params:<40} {recall:<12.4f} {qps:<10.1f} {idx_t:.1f}")


# ---------------------------------------------------------------------------
# Paper reference values (Table 2 / Figure 4 of the JHQ paper)
# These are read points from the paper at ~90% and ~95% recall on OpenAI3-3072.
# ---------------------------------------------------------------------------

PAPER_REFERENCE = {
    "JHQ":  {"recall": [0.90, 0.95], "qps": [2230, 889]},
    "JQ":   {"recall": [0.90, 0.95], "qps": [3963, 723]},
    "PQ":   {"recall": [0.57, 0.73], "qps": [3963, 723]},   # PQ tops out lower
    "OPQ":  {"recall": [0.73, 0.80], "qps": [3963, 723]},
}


def check_against_paper(results: list[dict]) -> None:
    print("\n=== Comparison against paper (OpenAI3-3072, Figure 4) ===")
    for algo, ref in PAPER_REFERENCE.items():
        rows = [r for r in results if r["algorithm"] == algo]
        if not rows:
            print(f"  {algo}: no results found")
            continue
        best_qps_at_95 = max(
            (r["qps"] for r in rows if r.get("recall_at_10", 0) >= 0.95),
            default=None,
        )
        paper_qps_at_95 = ref["qps"][1]
        if best_qps_at_95 is not None:
            ratio = best_qps_at_95 / paper_qps_at_95
            print(f"  {algo}: {best_qps_at_95:.0f} QPS @ ≥95% recall  "
                  f"(paper: {paper_qps_at_95} QPS,  ratio: {ratio:.2f}×)")
        else:
            print(f"  {algo}: did not reach 95% recall")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_tradeoff(results: list[dict], output_path: str = "results/speed_accuracy.png") -> None:
    if not HAS_MPL:
        print("matplotlib not installed, skipping plot.")
        return

    algo_styles = {
        "JHQ":  ("red",    "o",  "-",  "JHQ (ours)"),
        "JQ":   ("blue",   "s",  "-",  "JQ (ours)"),
        "PQ":   ("green",  "^",  "--", "PQ"),
        "OPQ":  ("orange", "D",  "--", "OPQ"),
        "IRVQ": ("purple", "v",  "--", "IRVQ"),
        "LSQ":  ("brown",  "P",  "--", "LSQ++"),
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    algos = sorted(set(r["algorithm"] for r in results))

    for algo in algos:
        rows = sorted(
            [r for r in results if r["algorithm"] == algo],
            key=lambda r: r.get("recall_at_10", 0),
        )
        xs = [r.get("recall_at_10", 0) for r in rows]
        ys = [r.get("qps", 0) for r in rows]
        style = algo_styles.get(algo, ("gray", "x", "-", algo))
        ax.semilogy(xs, ys, color=style[0], marker=style[1],
                    linestyle=style[2], label=style[3], linewidth=1.5)

    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Queries Per Second (QPS)")
    ax.set_title("Speed–Accuracy Trade-off (reproducing Figure 4)")
    ax.legend(loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0.5, 1.01)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate JHQ reproduction results")
    parser.add_argument("--results_dir", required=True, help="Directory with *.json result files")
    parser.add_argument("--groundtruth", default=None, help=".ivecs ground truth file (optional recompute)")
    parser.add_argument("--k", type=int, default=10, help="Recall@k to compute")
    parser.add_argument("--plot", action="store_true", help="Generate speed-accuracy plot")
    parser.add_argument("--plot_out", default="results/speed_accuracy.png")
    parser.add_argument("--check_paper", action="store_true", help="Compare against paper numbers")
    args = parser.parse_args()

    results = load_all_results(args.results_dir)
    print(f"Loaded {len(results)} result entries from {args.results_dir}")

    if args.groundtruth and os.path.exists(args.groundtruth):
        gt = read_ivecs(args.groundtruth)
        results = recompute_recall(results, gt, k=args.k)

    print_table(results, k=args.k)

    if args.check_paper:
        check_against_paper(results)

    if args.plot:
        plot_tradeoff(results, output_path=args.plot_out)
