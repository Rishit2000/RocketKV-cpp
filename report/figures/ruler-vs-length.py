"""RULER accuracy versus prompt length with budget 1024, with and without Stage 1 eviction. Each point is the mean over 5 tasks, with 10 samples per task up to 32K and 3 at 64K and 128K; the x axis is logarithmic. Without eviction, Stage 2 alone stays close to Full KV up to 32K (mean 78.4 vs 79.5, against 71.9 for Stage 1+2), but at 128K it falls to 35.6, below both Full KV (67.1) and Stage 1+2 (49.8)."""

import json
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

sys.path.insert(0, "style")
import paper  # applies figures/DESIGN.md
from paper import finalize_figure

colors = paper.colors
BENCH = "../../results/a106/bench"

acc = {}
for name in ("ruler", "ruler_stage2", "ruler_128k", "ruler_128k_stage2"):
    for run, units in json.load(open(f"{BENCH}/{name}/summary.json")).items():
        for unit, v in units.items():
            acc.setdefault(run, {}).setdefault(int(unit.split("/")[0]), []).append(v["score_pct"])
acc = {run: {n: np.mean(v) for n, v in lengths.items()} for run, lengths in acc.items()}

# name, run, color, line style, marker: ours first, then its ablation, then Full KV
series = [
    ("Stage 1+2", "hsa_b1024", colors["ours"], "-", "o"),
    ("Stage 2 only (no eviction)", "stage2_b1024", colors["series-4"], "-", "D"),
    ("Full KV", "off", colors["series-2"], "--", "^"),
]

short = [n for n in sorted(acc["off"]) if n <= 32768]
mean_short = {run: np.mean([acc[run][n] for n in short]) for _, run, _, _, _ in series}
top = max(acc["off"])
print("takeaway:", {run: f"<=32K {mean_short[run]:#.3g}, {top // 1024}K {acc[run][top]:#.3g}" for _, run, _, _, _ in series})
assert mean_short["off"] - mean_short["stage2_b1024"] < 2 < mean_short["off"] - mean_short["hsa_b1024"], \
    "the takeaway does not hold"
assert acc["stage2_b1024"][top] < min(acc["hsa_b1024"][top], acc["off"][top]), "the takeaway does not hold"

fig, ax = plt.subplots(figsize=(3.25, 3.25))
ax.set_xscale("log", base=2)
for name, run, color, style, marker in series:
    lengths = sorted(acc[run])
    width = rcParams["lines.linewidth"] if style == "-" else paper.dashed_linewidth
    ax.plot(lengths, [acc[run][n] for n in lengths], linestyle=style, marker=marker, color=color,
            linewidth=width, label=name)

lengths = sorted(acc["off"])
ax.set_xticks([], minor=True)
ax.set_xticks(lengths, labels=[f"{n // 1024}K" for n in lengths])
margin = 2 ** (rcParams["axes.xmargin"] * math.log2(top / lengths[0]))
ax.set_xlim(lengths[0] / margin, top * margin)
ax.set_xlabel("Prompt length (tokens)")
ax.set_ylabel("RULER accuracy (%)")
ax.legend()

finalize_figure(fig, ax)
fig.savefig("outputs/ruler-vs-length.pdf")
fig.savefig("outputs/ruler-vs-length.png")
