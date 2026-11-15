#!/usr/bin/env python3
"""Plot the qgemv roofline from sweep.sh's CSV.

  plot.py roofline.csv [outdir]

Writes roofline.png (one roofline per DRAM bandwidth), bandwidth.png (throughput
against bandwidth, one panel per format) and summary.md (tables for the README).

Throughput is weights per core cycle, M*K / cycles. The x axis of the roofline is
weights per DRAM byte, fixed by each format's storage size. Two roofs bound a point:
  memory: peak B/cycle * weights per byte
  issue:  cores * ISSUE_WIDTH / (instructions per weight), the decode-cost roof; it
          is what DCMP.LD raises, so a kernel's speedup comes from lifting this roof
          until the memory roof binds.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

# storage bytes per weight, from ggml's block sizes (block bytes / block weights)
BYTES_PER_WEIGHT = {
    "q8_0": 34 / 32,
    "q4_0": 18 / 32,
    "iq4_nl": 18 / 32,
    "mxfp4": 17 / 32,
    "q4_k": 144 / 256,
    "q3_k": 110 / 256,
}
FORMATS = list(BYTES_PER_WEIGHT)
ENGINES = ["scalar", "tcu", "dcmp"]
ISSUE_WIDTH = 1  # VX_CFG_ISSUE_WIDTH for 4 warps

# one hue per engine (validated categorical slots 1-3), one marker per format
ENGINE_COLOR = {"scalar": "#2a78d6", "tcu": "#eb6834", "dcmp": "#1baf7a"}
FORMAT_MARKER = {"q8_0": "s", "q4_0": "o", "iq4_nl": "D", "mxfp4": "^", "q4_k": "v", "q3_k": "P"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dedcd6"
# q4_0, iq4_nl and q4_k share one intensity, so nudge formats and engines apart along x
# (a few percent; the true positions are in summary.md)
FORMAT_NUDGE = {"q8_0": 1.0, "q4_0": 0.95, "iq4_nl": 1.0, "mxfp4": 1.04, "q4_k": 1.05, "q3_k": 1.0}
ENGINE_NUDGE = {"scalar": 0.985, "tcu": 1.0, "dcmp": 1.015}


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["status"] != "PASSED":
                print(f"skipping failed run: {r}", file=sys.stderr)
                continue
            mk = int(r["M"]) * int(r["K"])
            rows.append({
                "cores": int(r["cores"]),
                "bw": float(r["peak_Bpc"]),
                "type": r["type"],
                "engine": r["engine"],
                "wpc": mk / int(r["cycles"]),
                "ipw": int(r["instrs"]) / mk,
                # DRAM requests are 64-byte lines (VX_CFG_MEM_BLOCK_SIZE)
                "achieved": int(r["reqs"]) * 64 / int(r["cycles"]),
            })
    return rows


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def legend_handles():
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=ENGINE_COLOR[e], marker="o", linestyle="", markersize=7, label=e)
         for e in ENGINES]
    h += [Line2D([], [], color=MUTED, marker=FORMAT_MARKER[t], linestyle="", markersize=7,
                 markerfacecolor="none", label=t) for t in FORMATS]
    return h


def plot_roofline(rows, out):
    bws = sorted({r["bw"] for r in rows}, reverse=True)
    cores = rows[0]["cores"]
    n = len(bws)
    cols = 3
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(13, 4.2 * ((n + cols - 1) // cols)),
                             sharex=True, sharey=True)
    axes = axes.flatten()
    xs = [0.8, 2.6]
    for ax, bw in zip(axes, bws):
        style(ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot(xs, [bw * x for x in xs], color=INK, linewidth=1.5)
        ax.text(xs[1] / 1.03, bw * xs[1] / 1.5, "memory roof", color=INK, fontsize=8, ha="right")
        ax.set_xlim(*xs)
        ax.set_xticks([1, 1.5, 2, 2.5])
        ax.set_xticks([], minor=True)
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        for r in (r for r in rows if r["bw"] == bw):
            x = FORMAT_NUDGE[r["type"]] * ENGINE_NUDGE[r["engine"]] / BYTES_PER_WEIGHT[r["type"]]
            c = ENGINE_COLOR[r["engine"]]
            roof = cores * ISSUE_WIDTH / r["ipw"]
            # the issue roof for this kernel, a short dash above its point
            ax.plot([x / 1.012, x * 1.012], [roof, roof], color=c, linewidth=2, alpha=0.55)
            ax.plot(x, r["wpc"], marker=FORMAT_MARKER[r["type"]], color=c, markersize=8,
                    markeredgecolor="white", markeredgewidth=1, linestyle="")
        ax.set_title(f"peak DRAM {bw:g} B/cycle", fontsize=10, color=INK, loc="left")
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[::cols]:
        ax.set_ylabel("weights / cycle", fontsize=9, color=MUTED)
    for ax in axes[-cols:]:
        ax.set_xlabel("weights per DRAM byte (operational intensity)", fontsize=9, color=MUTED)
    fig.suptitle(f"qgemv roofline, {cores} cores. Points: measured. Dashes: each kernel's issue roof, "
                 f"{cores} x {ISSUE_WIDTH} instr/cycle / instrs per weight",
                 fontsize=11, color=INK, y=0.99)
    fig.legend(handles=legend_handles(), loc="upper center", ncol=9, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_bandwidth(rows, out):
    cores = rows[0]["cores"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, sharey=True)
    for ax, t in zip(axes.flatten(), FORMATS):
        style(ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        for e in ENGINES:
            pts = sorted((r["bw"], r["wpc"]) for r in rows if r["type"] == t and r["engine"] == e)
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=ENGINE_COLOR[e], linewidth=2,
                    marker="o", markersize=5, label=e)
            ax.annotate(e, pts[-1], xytext=(4, 0), textcoords="offset points", fontsize=8,
                        color=MUTED, va="center")
        # memory roof for this format
        bws = sorted({r["bw"] for r in rows})
        ax.plot(bws, [b / BYTES_PER_WEIGHT[t] for b in bws], color=INK, linewidth=1, linestyle="--")
        ax.set_title(f"{t} ({8 * BYTES_PER_WEIGHT[t]:.2f} bits/weight)", fontsize=10, color=INK, loc="left")
    for ax in axes[:, 0]:
        ax.set_ylabel("weights / cycle", fontsize=9, color=MUTED)
    for ax in axes[-1]:
        ax.set_xlabel("peak DRAM bandwidth (B/cycle)", fontsize=9, color=MUTED)
    fig.suptitle(f"qgemv throughput vs DRAM bandwidth, {cores} cores (dashed: memory roof)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def summary(rows):
    bws = sorted({r["bw"] for r in rows}, reverse=True)
    by = {(r["bw"], r["type"], r["engine"]): r for r in rows}
    kernels = [(t, e) for e in ENGINES for t in FORMATS if any(k[1:] == (t, e) for k in by)]
    out = ["Throughput in weights per cycle (higher is better); best per bandwidth in bold.", ""]
    out.append("| format | engine | instrs/weight | " + " | ".join(f"{b:g} B/c" for b in bws) + " |")
    out.append("|---|---|---|" + "---|" * len(bws))
    best = {b: max((r for r in rows if r["bw"] == b), key=lambda r: r["wpc"]) for b in bws}
    for t, e in kernels:
        ipw = next(r["ipw"] for k, r in by.items() if k[1:] == (t, e))
        cells = []
        for b in bws:
            r = by.get((b, t, e))
            if r is None:
                cells.append("-")
            else:
                s = f"{r['wpc']:.2f}"
                cells.append(f"**{s}**" if r is best[b] else s)
        out.append(f"| {t} | {e} | {ipw:.2f} | " + " | ".join(cells) + " |")
    out += ["", "Fastest kernel at each bandwidth, and its measured DRAM traffic as a share of peak:", ""]
    out.append("| peak B/cycle | fastest | weights/cycle | DRAM use |")
    out.append("|---|---|---|---|")
    for b in bws:
        r = best[b]
        out.append(f"| {b:g} | {r['type']} {r['engine']} | {r['wpc']:.2f} | {100 * r['achieved'] / b:.0f}% |")
    return "\n".join(out) + "\n"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rows = load(sys.argv[1])
    outdir = Path(sys.argv[2] if len(sys.argv) > 2 else ".")
    outdir.mkdir(parents=True, exist_ok=True)
    plot_roofline(rows, outdir / "roofline.png")
    plot_bandwidth(rows, outdir / "bandwidth.png")
    (outdir / "summary.md").write_text(summary(rows))
    print(summary(rows))


if __name__ == "__main__":
    main()
