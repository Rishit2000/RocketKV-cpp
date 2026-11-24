"""RULER accuracy of Stage 1+2 versus token budget at 16K and 32K prompts, averaged over five tasks. Budgets 512 to 2048 use 10 samples per task, 256 and 4096 use 5; the x axis is logarithmic and the dashed line marks Full KV. At 16K a budget of 4096 reaches Full KV (79.5 vs 78.4), while at 32K accuracy stays near 69 for every budget, 7 points below Full KV."""

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
runs = {**json.load(open(f"{BENCH}/ruler/summary.json")), **json.load(open(f"{BENCH}/ruler_budget/summary.json"))}


def accuracy(run, length):
    return np.mean([v["score_pct"] for unit, v in runs[run].items() if unit.startswith(f"{length}/")])


lengths = [16384, 32768]
budgets = sorted(int(r.split("_b")[1]) for r in runs if r.startswith("hsa_b"))
curves = {n: [accuracy(f"hsa_b{b}", n) for b in budgets] for n in lengths}
full = {n: accuracy("off", n) for n in lengths}
print("takeaway:", {n: ([f"{v:#.3g}" for v in curves[n]], f"full {full[n]:#.3g}") for n in lengths})
assert curves[16384][-1] >= full[16384] and max(curves[32768]) < full[32768] - 5, "the takeaway does not hold"

fig, axes = plt.subplots(1, 2, figsize=(3.25, 3.25), sharex=True, sharey=True)
for ax, n in zip(axes, lengths):
    ax.set_xscale("log", base=2)
    ax.plot(budgets, curves[n], "o-", color=colors["ours"], label="Stage 1+2")
    ax.axhline(full[n], color=colors["reference"], linestyle="--", linewidth=paper.dashed_linewidth, zorder=1,
               label="Full KV")
    ax.set_title(f"{n // 1024}K prompt")
    ax.set_xticks([], minor=True)
    ax.set_xticks(budgets, labels=[str(b) for b in budgets])
    margin = 2 ** (rcParams["axes.xmargin"] * math.log2(budgets[-1] / budgets[0]))
    ax.set_xlim(budgets[0] / margin, budgets[-1] * margin)
axes[0].set_ylabel("Accuracy (%)")
fig.supxlabel("Token budget")
axes[0].legend()

finalize_figure(fig, axes)
fig.savefig("outputs/accuracy-vs-budget.pdf")
fig.savefig("outputs/accuracy-vs-budget.png")
