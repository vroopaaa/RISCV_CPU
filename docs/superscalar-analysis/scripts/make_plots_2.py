#!/usr/bin/env python3
"""Generates the figures for the superscalar analysis report from
superscalar_sweep_n10.json (produced by 32I/tests/npu/analyze_superscalar.py
--max-n 10). Writes PDF (for LaTeX \includegraphics) and PNG (for quick
preview) into figures/.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.dirname(HERE)  # docs/superscalar-analysis/
JSON_PATH = os.path.join(REPORT_DIR, "data", "superscalar_sweep_n10.json")
OUT_DIR = os.path.join(REPORT_DIR, "figures_2")
os.makedirs(OUT_DIR, exist_ok=True)

with open(JSON_PATH) as f:
    data = json.load(f)

LEVELS = ["O0", "O1", "O2", "O3"]
MAX_N = 10

# Categorical palette for line plots (O0-O3)
COLOR = {
    "O0": "#2a78d6",  # blue
    "O1": "#eb6834",  # orange
    "O2": "#1baf7a",  # aqua
    "O3": "#eda100",  # yellow
}
MARKER = {"O0": "o", "O1": "s", "O2": "^", "O3": "D"}

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#dddddb"

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_PRIMARY,
    "text.color": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "svg.fonttype": "none",
})


def savefig(fig, name):
    fig.savefig(os.path.join(OUT_DIR, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def clean_axes(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(length=0)


# ---------------------------------------------------------------------------
# Fig 1: scalar cycles vs optimisation level (bar)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 3.8))
scalar_cycles = [data[l]["runs"]["scalar"]["cycles"] for l in LEVELS]
bars = ax.bar(LEVELS, scalar_cycles, color=[COLOR[l] for l in LEVELS], width=0.55)
for b, v in zip(bars, scalar_cycles):
    ax.annotate(f"{v:,}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=9, color=TEXT_PRIMARY)
ax.set_ylabel("Scalar-pipeline cycles")
ax.set_xlabel("Compiler optimisation level")
ax.set_title("256×256 NPU matmul: cycles vs. optimisation level (issue width 1)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
clean_axes(ax)
savefig(fig, "fig1_cycles_vs_optimisation")

# ---------------------------------------------------------------------------
# Fig 2: cycles vs issue width, 4 series (log y - O0 dwarfs O1-O3)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.3))
ns = list(range(1, MAX_N + 1))
for l in LEVELS:
    cycles = [data[l]["runs"][f"n{n}"]["cycles"] for n in ns]
    ax.plot(ns, cycles, color=COLOR[l], marker=MARKER[l], markersize=5,
            linewidth=2, label=l, markerfacecolor="white", markeredgewidth=1.5)
ax.set_yscale("log")
ax.set_xlabel("Issue width $n$")
ax.set_ylabel("Cycles (log scale)")
ax.set_title("Cycles vs. superscalar issue width, by optimisation level")
ax.set_xticks(ns)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(frameon=False, title=None, ncol=4, loc="upper center",
          bbox_to_anchor=(0.5, -0.16))
clean_axes(ax)
savefig(fig, "fig2_cycles_vs_issue_width")

# ---------------------------------------------------------------------------
# Fig 3: IPC vs issue width, 4 series
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.3))
for l in LEVELS:
    ipc = [data[l]["runs"][f"n{n}"]["instructions"] / data[l]["runs"][f"n{n}"]["cycles"] for n in ns]
    ax.plot(ns, ipc, color=COLOR[l], marker=MARKER[l], markersize=5,
            linewidth=2, label=l, markerfacecolor="white", markeredgewidth=1.5)
ax.axhline(1.0, color=TEXT_SECONDARY, linewidth=1, linestyle=":", zorder=0)
ax.set_xlabel("Issue width $n$")
ax.set_ylabel("Average IPC (instructions / cycle)")
ax.set_title("IPC vs. superscalar issue width, by optimisation level")
ax.set_xticks(ns)
ax.set_ylim(0.9, 3.1)
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
clean_axes(ax)
savefig(fig, "fig3_ipc_vs_issue_width")

# ---------------------------------------------------------------------------
# Fig 4 (bonus): speedup vs issue width, 4 series
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.3))
for l in LEVELS:
    base = data[l]["runs"]["scalar"]["cycles"]
    speedup = [base / data[l]["runs"][f"n{n}"]["cycles"] for n in ns]
    ax.plot(ns, speedup, color=COLOR[l], marker=MARKER[l], markersize=5,
            linewidth=2, label=l, markerfacecolor="white", markeredgewidth=1.5)
ax.axhline(1.0, color=TEXT_SECONDARY, linewidth=1, linestyle=":", zorder=0)
ax.set_xlabel("Issue width $n$")
ax.set_ylabel("Speedup vs. scalar pipeline ($\\times$)")
ax.set_title("Speedup vs. superscalar issue width, by optimisation level")
ax.set_xticks(ns)
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
clean_axes(ax)
savefig(fig, "fig4_speedup_vs_issue_width")

# ---------------------------------------------------------------------------
# Fig 5 (bonus): best speedup per optimisation level (bar)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 3.8))
best = []
for l in LEVELS:
    base = data[l]["runs"]["scalar"]["cycles"]
    best.append(max(base / data[l]["runs"][f"n{n}"]["cycles"] for n in ns))
bars = ax.bar(LEVELS, best, color=[COLOR[l] for l in LEVELS], width=0.55)
for b, v in zip(bars, best):
    ax.annotate(f"{v:.3f}x", (b.get_x() + b.get_width() / 2, v), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=9, color=TEXT_PRIMARY)
ax.set_ylabel("Best achievable speedup ($n\\leq 10$)")
ax.set_xlabel("Compiler optimisation level")
ax.set_title("Best superscalar speedup by optimisation level")
clean_axes(ax)
savefig(fig, "fig5_best_speedup_by_level")

# ---------------------------------------------------------------------------
# Fig 6 (bonus): co-issue width distribution at converged n, stacked bar
# ---------------------------------------------------------------------------
CONVERGED_N = {"O0": 5, "O1": 4, "O2": 6, "O3": 6}
fig, ax = plt.subplots(figsize=(6.5, 4.3))

bucket_colors = {
    1: "#3b528b",  # deep slate blue/indigo
    2: "#20938c",  # teal
    3: "#35b779",  # green
    4: "#e69f00",  # amber yellow
    5: "#d55e00",  # vermilion
    6: "#781c6d",  # wine / deep purple
}

bottoms = [0.0] * len(LEVELS)
max_k = max(CONVERGED_N.values())

for k in range(1, max_k + 1):
    heights = []
    for l in LEVELS:
        n = CONVERGED_N[l]
        r = data[l]["runs"][f"n{n}"]
        cycles = r["cycles"]
        cnt = r["hist"].get(str(k), r["hist"].get(k, 0))
        heights.append(100.0 * cnt / cycles)
    if all(h == 0 for h in heights):
        continue
    ax.bar(
        LEVELS,
        heights,
        bottom=bottoms,
        color=bucket_colors[k],
        width=0.55,
        label=f"{k} issued",
        edgecolor="#ffffff",
        linewidth=0.7,
    )
    bottoms = [b + h for b, h in zip(bottoms, heights)]

# Write the cycle counts above the bars with guaranteed visibility
for l in LEVELS:
    n = CONVERGED_N[l]
    total_cycles = data[l]["runs"][f"n{n}"]["cycles"]
    ax.annotate(
        f"{total_cycles:,} cycles",
        xy=(l, 100.0),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=TEXT_PRIMARY,
        fontweight="bold",
        clip_on=False,
    )

# 0 to 118 creates space between the bar tops and the plot title
ax.set_ylim(0, 118)
ax.set_yticks([0, 20, 40, 60, 80, 100])

ax.set_ylabel("Share of cycles (%)")
ax.set_xlabel("Compiler optimisation level (each at its converged $n$)")
ax.set_title("Co-issue width distribution at converged issue width", pad=14)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
clean_axes(ax)
savefig(fig, "fig6_coissue_distribution")

print("Wrote figures to", OUT_DIR)
for f in sorted(os.listdir(OUT_DIR)):
    print(" ", f)