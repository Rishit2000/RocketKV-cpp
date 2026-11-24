"""Per-token decode latency versus prompt length on Apple M5 Ultra (Llama-3.2-3B, budget 1024). Each point is the median inter-token latency over 6 runs (3 at 64K and 128K); the x axis is logarithmic. Stage 1+2 stays between 7.13 and 7.51 ms from 2K to 128K, while Full KV grows to 21.9 ms, so at 128K Stage 1+2 is 2.91× faster."""

import json
import math
import sys

import matplotlib.pyplot as plt
from matplotlib import rcParams

sys.path.insert(0, "style")
import paper  # applies figures/DESIGN.md
from paper import finalize_figure, shift_lightness

colors = paper.colors
BENCH = "../../results/a106/bench"

rows = json.load(open(f"{BENCH}/latency/summary.json")) + json.load(open(f"{BENCH}/latency_128k/summary.json"))
itl = {}
for r in rows:
    itl.setdefault(r["config"], {})[r["length"]] = r["itl_p50_ms"]

# name, config, color, line style, marker: ours first, then compared
series = [
    ("Stage 1+2 (Metal)", "E4_hsa_metal", colors["ours"], "-", "o"),
    ("Stage 1 (Metal)", "E2_s1_metal", colors["series-3"], "-", "s"),
    ("Full KV", "E0_full", colors["series-2"], "--", "^"),
]

top = max(itl["E0_full"])
e4 = [itl["E4_hsa_metal"][n] for n in sorted(itl["E4_hsa_metal"])]
speedup = itl["E0_full"][top] / itl["E4_hsa_metal"][top]
print("takeaway:", f"E4 {min(e4):.2f}-{max(e4):.2f} ms; Full KV at {top}: {itl['E0_full'][top]:.3g} ms;",
      f"E4 at {top}: {itl['E4_hsa_metal'][top]:.3g} ms; speedup {speedup:.3g}×")
assert max(e4) - min(e4) < 0.5 and speedup > 2.5, "the takeaway does not hold"

fig, ax = plt.subplots(figsize=(3.25, 3.25))
ax.set_xscale("log", base=2)
for name, config, color, style, marker in series:
    lengths = sorted(itl[config])
    width = rcParams["lines.linewidth"] if style == "-" else paper.dashed_linewidth
    ax.plot(lengths, [itl[config][n] for n in lengths], linestyle=style, marker=marker, color=color,
            linewidth=width, label=name)

lengths = sorted(itl["E0_full"])
ax.set_xticks([], minor=True)
ax.set_xticks(lengths, labels=[f"{n // 1024}K" for n in lengths])
margin = 2 ** (rcParams["axes.xmargin"] * math.log2(top / lengths[0]))
ax.set_xlim(lengths[0] / margin, top * margin)
ax.set_xlabel("Prompt length (tokens)")
ax.set_ylabel("Decode latency (ms/token)")
ax.legend()

finalize_figure(fig, ax)
fig.savefig("outputs/itl-vs-length.pdf")
fig.savefig("outputs/itl-vs-length.png")
