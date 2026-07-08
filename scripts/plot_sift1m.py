"""
Publication-quality comparison figure for SIFT1M.
Two panels: (a) 16B budget: JQ vs FAISS PQ,  (b) ~80B budget: JHQ vs FAISS PQ.
"""
import sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.family":        "DejaVu Serif",
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     11.5,
    "legend.fontsize":    9,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "axes.linewidth":     0.9,
    "lines.linewidth":    2.0,
    "lines.markersize":   7,
    "grid.linewidth":     0.45,
    "grid.alpha":         0.45,
    "figure.dpi":         180,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.06,
})

STYLES = {
    "JQ+IVF":           dict(color="#1F77B4", marker="s",  ls="-",  lw=2.2, zorder=5,
                              label="JQ (Ours, 16 B/vec)",
                              mfc="#1F77B4", mew=1.8),
    "FAISS-IVFPQ-16B":  dict(color="#2CA02C", marker="^",  ls=":",  lw=1.6, zorder=3,
                              label="FAISS IVFPQ (16 B/vec)",
                              mfc="white", mew=1.6),
    "JHQ+IVF":          dict(color="#D62728", marker="o",  ls="-",  lw=2.2, zorder=5,
                              label="JHQ (Ours, ~80 B/vec)",
                              mfc="#D62728", mew=1.8),
    "FAISS-IVFPQ-64B":  dict(color="#9467BD", marker="v",  ls=":",  lw=1.6, zorder=3,
                              label="FAISS IVFPQ (64 B/vec)",
                              mfc="white", mew=1.6),
}

def load_results(csv_path):
    data = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            m = row["method"]
            if m not in data:
                data[m] = {"recall": [], "qps": [], "nprobe": [],
                            "build_time": float(row["build_time"])}
            data[m]["recall"].append(float(row["recall"]))
            data[m]["qps"].append(float(row["qps"]))
            data[m]["nprobe"].append(int(row["nprobe"]))
    return data

def _qps_fmt(x, _):
    if x >= 1000: return f"{x/1000:.0f}K"
    return f"{x:.0f}"

def plot_panel(ax, data, methods, title, xlim=(25, 62)):
    for mname in methods:
        if mname not in data: continue
        d = data[mname]
        st = STYLES[mname]
        recalls = [r * 100 for r in d["recall"]]
        ax.semilogy(recalls, d["qps"],
                    marker=st["marker"], color=st["color"],
                    linestyle=st["ls"], linewidth=st["lw"],
                    markerfacecolor=st["mfc"],
                    markeredgewidth=st["mew"],
                    markeredgecolor=st["color"],
                    zorder=st["zorder"], label=st["label"])

    ax.set_xlabel("Recall@10 (%)")
    ax.set_ylabel("Queries per Second (QPS)")
    ax.set_title(title, fontweight="bold", pad=6)
    ax.set_xlim(*xlim)
    ax.set_ylim(bottom=40)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_qps_fmt))
    ax.grid(True, which="major", ls="-",  alpha=0.35)
    ax.grid(True, which="minor", ls=":",  alpha=0.20)
    ax.legend(framealpha=0.92, edgecolor="0.75",
              handlelength=2.2, labelspacing=0.4)

def plot_build(ax, data):
    order = [
        ("JHQ+IVF",         "JHQ\n(Ours)"),
        ("JQ+IVF",          "JQ\n(Ours)"),
        ("FAISS-IVFPQ-64B", "FAISS PQ\n64B"),
        ("FAISS-IVFPQ-16B", "FAISS PQ\n16B"),
    ]
    order = [(k, l) for k, l in order if k in data]
    labels = [l for _, l in order]
    colors = [STYLES[k]["color"] for k, _ in order]
    times  = [data[k]["build_time"] for k, _ in order]

    x = np.arange(len(order))
    bars = ax.bar(x, times, color=colors, edgecolor="white",
                  linewidth=0.9, width=0.6, zorder=3, alpha=0.88)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Index Build Time (s)")
    ax.set_title("(c) Build Time", fontweight="bold", pad=6)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.0f}s" if v >= 1 else f"{v:.1f}s"))
    ax.grid(True, which="both", axis="y", ls=":", alpha=0.4)
    ax.set_ylim(top=max(times) * 5)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2,
                t * 1.5, f"{t:.1f}s",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/sift1m_results.csv"
    data = load_results(csv_path)

    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 0.8], wspace=0.30)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    plot_panel(ax0, data, ["JQ+IVF", "FAISS-IVFPQ-16B"],
               "(a) 16 B/vec Budget: JQ vs FAISS PQ",
               xlim=(25, 58))
    plot_panel(ax1, data, ["JHQ+IVF", "FAISS-IVFPQ-64B"],
               "(b) ~80 B/vec Budget: JHQ vs FAISS PQ",
               xlim=(30, 98))
    plot_build(ax2, data)

    # 注释：JQ recall 非单调（K1D=2 tie 问题）
    ax0.annotate("Non-monotone recall:\nK1D=2 ADC ties (d=128)",
                 xy=(55.85, 700), xytext=(40, 2000),
                 fontsize=7.5, color="#1F77B4",
                 arrowprops=dict(arrowstyle="->", color="#1F77B4", lw=1.0))

    fig.suptitle(
        "JHQ vs. FAISS IVFPQ  —  SIFT1M  (d=128, N=1M, nlist=1024, k=10)",
        fontsize=11.5, fontweight="bold", y=1.03
    )

    out_pdf = csv_path.replace(".csv", "_figure.pdf")
    out_png = csv_path.replace(".csv", "_figure.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")

if __name__ == "__main__":
    main()
