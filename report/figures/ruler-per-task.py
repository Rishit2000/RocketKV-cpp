"""RULER accuracy per task with budget 1024, averaged over prompt lengths 2K to 32K (10 samples per task and length). Stage 1+2 matches Full KV within a few points on four of the five tasks; the loss is concentrated in niah_multikey_2, which drops from 96.0 to 62.0, and Stage 1 alone loses the same amount there."""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "style")
import paper  # applies figures/DESIGN.md
from paper import finalize_figure, shift_lightness

colors = paper.colors
summary = json.load(open("../../results/a106/bench/ruler/summary.json"))

tasks = ["niah_single_2", "niah_multikey_2", "vt", "fwe", "qa_1"]  # the benchmark's task order
# name, run, color: compared first, ours last (left to right)
series = [
    ("Full KV", "off", colors["series-2"]),
    ("Stage 1", "stage1_b1024", colors["series-3"]),
    ("Stage 1+2", "hsa_b1024", colors["ours"]),
]
scores = {}
for name, run, _ in series:
    scores[name] = [np.mean([v["score_pct"] for unit, v in summary[run].items() if unit.endswith("/" + t)])
                    for t in tasks]

drop = {t: scores["Full KV"][i] - scores["Stage 1+2"][i] for i, t in enumerate(tasks)}
print("takeaway:", {t: f"{scores['Full KV'][i]:#.3g} -> {scores['Stage 1+2'][i]:#.3g}" for i, t in enumerate(tasks)})
assert max(drop, key=drop.get) == "niah_multikey_2" and drop["niah_multikey_2"] > 3 * max(
    v for t, v in drop.items() if t != "niah_multikey_2"), "the takeaway does not hold"

# task names crowd at half width: full text width (Collisions step 3)
fig, ax = plt.subplots(figsize=(6.5, 6.5))
x = np.arange(len(tasks))
width = 0.8 / len(series)
for i, (name, _, color) in enumerate(series):
    bars = ax.bar(x + (i - (len(series) - 1) / 2) * width, scores[name], width, color=color, linewidth=0,
                  label=name)
    ax.bar_label(bars, labels=[f"{v:#.3g}" for v in scores[name]], color=shift_lightness(color, -1))
ax.grid(axis="x", visible=False)
ax.set_xticks(x, labels=tasks)
ax.tick_params(axis="x", length=0)
ax.set_ylabel("Accuracy (%)")
ax.legend()

finalize_figure(fig, ax)
fig.savefig("outputs/ruler-per-task.pdf")
fig.savefig("outputs/ruler-per-task.png")
