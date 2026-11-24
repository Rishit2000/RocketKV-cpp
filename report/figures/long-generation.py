"""Decode latency at each step of a 1024-token generation from a 32K and a 128K prompt (budget 1024). Lines are smoothed with a moving average over 5% of the steps; faint lines show raw values (median over 2 runs at 32K, 1 run at 128K). Full KV slows as the context grows, while both RocketKV variants stay flat; total decode time at 128K falls from 22.5 s to 7.54 s with Stage 1+2 (−66%)."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "style")
import paper  # applies figures/DESIGN.md
from paper import finalize_figure, shift_lightness

colors = paper.colors
BENCH = Path("../../results/a106/bench")

steps = {}  # (config, length) -> list of per-step series
for path in [*(BENCH / "latency_long/runs").glob("E*_r*.json"), *(BENCH / "latency_long_128k/runs").glob("E*_r*.json")]:
    r = json.loads(path.read_text())
    steps.setdefault((r["config"], r["length"]), []).append(r["metrics"]["decode_step_ms"])

series = [
    ("Stage 1+2 (Metal)", "E4_hsa_metal", colors["ours"], "-"),
    ("Stage 1 (Metal)", "E2_s1_metal", colors["series-3"], "-"),
    ("Full KV", "E0_full", colors["series-2"], "--"),
]
lengths = [32768, 131072]
total = {key: np.median([sum(s) for s in runs]) / 1000 for key, runs in steps.items()}
for n in lengths:
    print("takeaway:", n, {c: f"{total[(c, n)]:#.3g} s" for _, c, _, _ in series},
          f"E4 change {total[('E4_hsa_metal', n)] / total[('E0_full', n)] - 1:.0%}")
assert total[("E4_hsa_metal", 131072)] < 0.4 * total[("E0_full", 131072)], "the takeaway does not hold"

fig, axes = plt.subplots(1, 2, figsize=(3.25, 3.25), sharex=True)
for ax, n in zip(axes, lengths):
    for name, config, color, style in series:
        y = np.median(np.array(steps[(config, n)]), axis=0)[1:]  # the first step builds the chunk summaries
        x = np.arange(2, len(y) + 2)
        w = max(1, len(y) // 40) * 2 + 1
        smooth = np.convolve(y, np.ones(w) / w, mode="valid")
        ax.plot(x, y, color=color, alpha=0.25, linewidth=paper.dashed_linewidth)
        ax.plot(x[w // 2: len(x) - w // 2], smooth, linestyle=style, color=color,
                linewidth=paper.dashed_linewidth, label=name)
    ax.set_title(f"{n // 1024}K prompt")
axes[0].set_ylabel("Decode latency (ms/token)")
fig.supxlabel("Decode step")
axes[1].legend()

finalize_figure(fig, axes)
fig.savefig("outputs/long-generation.pdf")
fig.savefig("outputs/long-generation.png")
