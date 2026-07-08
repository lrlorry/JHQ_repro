"""
Publication-quality comparison figure: JQ/JHQ vs FAISS IVFPQ on Vogue-768.
Style modelled after Figure 4 of the JHQ paper (PVLDB 2026).
"""
import sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Global style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":          "DejaVu Serif",
    "font.size":            11,
    "axes.labelsize":       12,
    "axes.titlesize":       11.5,
    "legend.fontsize":      8.5,
    "xtick.labelsize":      10,
    "ytick.labelsize":      10,
    "axes.linewidth":       0.9,
    "lines.linewidth":      2.0,
    "lines.markersize":     7,
    "grid.linewidth":       0.45,
    "grid.alpha":           0.45,
    "figure.dpi":           180,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.06,
})

# ── Palette ──────────────────────────────────────────────────────────────────
# Ours: solid lines, official source: dashed lines (same hue family)
METHODS = {
    # ---- Official source code results (paper repo) -------------------------
    "Official-JHQ":      dict(color="#D62728", marker="o",  ls="--", lw=2.2, zorder=6,
                              label="JHQ Official (480 B/vec)",
                              mfc="white", mew=2.0),
    "Official-JQ":       dict(color="#1F77B4", marker="s",  ls="--", lw=2.2, zorder=6,
                              label="JQ Official (96 B/vec)",
                              mfc="white", mew=2.0),
    # ---- Our reproduction --------------------------------------------------
    "JHQ+IVF":           dict(color="#D62728", marker="o",  ls="-",  lw=2.2, zorder=5,
                              label="JHQ Repro (Ours, 480 B/vec)",
                              mfc="#D62728", mew=1.8),
    "JQ+IVF":            dict(color="#1F77B4", marker="s",  ls="-",  lw=2.2, zorder=5,
                              label="JQ Repro (Ours, 96 B/vec)",
                              mfc="#1F77B4", mew=1.8),
    # ---- FAISS baselines ---------------------------------------------------
    "FAISS-IVFPQ-96B":   dict(color="#2CA02C", marker="^",  ls=":",  lw=1.6, zorder=3,
                              label="FAISS IVFPQ (96 B/vec)",
                              mfc="white", mew=1.6),
    "FAISS-IVFPQ-192B":  dict(color="#FF7F0E", marker="D",  ls=":",  lw=1.6, zorder=3,
                              label="FAISS IVFPQ (192 B/vec)",
                              mfc="white", mew=1.6),
    "FAISS-IVFPQ-384B":  dict(color="#9467BD", marker="v",  ls=":",  lw=1.6, zorder=3,
                              label="FAISS IVFPQ (384 B/vec)",
                              mfc="white", mew=1.6),
}

# ── I/O ─────────────────────────────────────────────────────────────────────
def load_results(csv_path):
    data = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            m = row["method"]
            if m not in data:
                data[m] = {"recall": [], "qps": [], "nprobe": [],
                            "build_time": float(row.get("build_time", 0))}
            data[m]["recall"].append(float(row["recall"]))
            data[m]["qps"].append(float(row["qps"]))
            data[m]["nprobe"].append(int(row["nprobe"]))
    return data

# ── Plot helpers ─────────────────────────────────────────────────────────────
def _qps_fmt(x, _):
    if x >= 1000:  return f"{x/1000:.0f}K"
    return f"{x:.0f}"

def plot_speed_accuracy(ax, data):
    for mname, style in METHODS.items():
        if mname not in data:
            continue
        d = data[mname]
        recalls = [r * 100 for r in d["recall"]]
        ax.semilogy(recalls, d["qps"],
                    marker=style["marker"], color=style["color"],
                    linestyle=style["ls"], linewidth=style["lw"],
                    markerfacecolor=style["mfc"],
                    markeredgewidth=style["mew"],
                    markeredgecolor=style["color"],
                    zorder=style["zorder"], label=style["label"])

    # nprobe annotations on Official-JHQ curve
    for ann_key in ("Official-JHQ", "JHQ+IVF"):
        if ann_key not in data:
            continue
        d = data[ann_key]
        style = METHODS[ann_key]
        for r, q, np_ in zip(d["recall"], d["qps"], d["nprobe"]):
            if np_ in (1, 8, 32):
                offset = (4, 3) if ann_key == "Official-JHQ" else (4, -10)
                ax.annotate(f"np={np_}", xy=(r*100, q),
                            xytext=offset, textcoords="offset points",
                            fontsize=7, color=style["color"], alpha=0.85)
        break  # only annotate first found

    ax.set_xlabel("Recall@10 (%)")
    ax.set_ylabel("Queries per Second (QPS)")
    ax.set_title("(a) Speed–Accuracy Trade-off", fontweight="bold", pad=6)
    ax.set_xlim(55, 101)
    ax.set_ylim(bottom=8)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_qps_fmt))
    ax.grid(True, which="major", ls="-",  alpha=0.35)
    ax.grid(True, which="minor", ls=":",  alpha=0.20)
    for xv, lbl in [(90, "90%"), (95, "95%"), (99, "99%")]:
        ax.axvline(xv, color="gray", lw=0.7, ls=":", alpha=0.6)
        ax.text(xv+0.3, ax.get_ylim()[0]*1.5, lbl, fontsize=7.5,
                color="gray", rotation=90, va="bottom")

    # Legend in two columns
    ax.legend(loc="upper left", framealpha=0.92, edgecolor="0.75",
              handlelength=2.2, labelspacing=0.35, ncol=1,
              fontsize=8.0)

def plot_build_time(ax, data):
    # Bar order: Official-JHQ, Official-JQ, JHQ (Ours), JQ (Ours), FAISS baselines
    bar_order = [
        ("Official-JHQ",    "Official\nJHQ"),
        ("Official-JQ",     "Official\nJQ"),
        ("JHQ+IVF",         "JHQ\n(Ours)"),
        ("JQ+IVF",          "JQ\n(Ours)"),
        ("FAISS-IVFPQ-96B", "FAISS PQ\n96B"),
        ("FAISS-IVFPQ-192B","FAISS PQ\n192B"),
        ("FAISS-IVFPQ-384B","FAISS PQ\n384B"),
    ]
    order  = [(k, lbl) for k, lbl in bar_order if k in data]
    labels = [lbl for _, lbl in order]
    colors = [METHODS[k]["color"] for k, _ in order]
    # hatching: official = hatched, ours = solid
    hatches = ["//" if k.startswith("Official") else "" for k, _ in order]
    times  = [data[k]["build_time"] for k, _ in order]

    x = np.arange(len(order))
    bars = ax.bar(x, times, color=colors, hatch=hatches,
                  edgecolor="white", linewidth=0.9,
                  width=0.62, zorder=3, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.0)
    ax.set_ylabel("Index Build Time (s)")
    ax.set_title("(b) Index Construction Time", fontweight="bold", pad=6)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.0f}s" if v >= 10 else f"{v:.1f}s"))
    ax.grid(True, which="both", axis="y", ls=":", alpha=0.4)
    ax.set_ylim(top=max(times) * 6)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2,
                t * 1.6, f"{t:.0f}s",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/vogue768_results.csv"
    data = load_results(csv_path)

    fig = plt.figure(figsize=(13, 4.8))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.65, 1], wspace=0.30)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    plot_speed_accuracy(ax0, data)
    plot_build_time(ax1, data)

    fig.suptitle(
        "JHQ Official Source vs. Our Reproduction vs. FAISS IVFPQ  —  Vogue-768"
        "   (d=768, N=932K, nlist=1024, k=10)",
        fontsize=11, fontweight="bold", y=1.03
    )

    out_pdf = csv_path.replace(".csv", "_figure.pdf")
    out_png = csv_path.replace(".csv", "_figure.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")

if __name__ == "__main__":
    main()
