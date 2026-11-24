"""Paper figures in the paper-figure design. `import paper` finds the DESIGN.md above this folder and
applies it to matplotlib; scripts then read `paper.colors` and call the helpers below.

design.py reads DESIGN.md; layout.py sizes the figure; palette.py keeps colors apart for color-blind
readers and in black and white; checks.py reports what reads badly; helpers.py holds helpers for drawing."""
from pathlib import Path

from . import design as _design
from .helpers import add_legend_rows
from .layout import finalize_figure
from .palette import make_part_colors, make_shades, shift_lightness

__all__ = ["finalize_figure", "shift_lightness", "make_shades", "make_part_colors", "add_legend_rows"]

# figures/style/paper finds figures/DESIGN.md; the skill's assets/paper finds assets/DESIGN.md. A DESIGN.md at
# or above the folder a script runs in that is not that one changes only the tokens it names (one paper's look)
_base = next(d / "DESIGN.md" for d in Path(__file__).resolve().parents if (d / "DESIGN.md").exists())
_local = next((d / "DESIGN.md" for d in [Path.cwd(), *Path.cwd().parents] if (d / "DESIGN.md").exists()), _base)
_design.apply_design(*dict.fromkeys([_base, _local]))


def __getattr__(name):
    # design values (colors, dashed_linewidth, highlight_markersize, ...) live in design.py
    return getattr(_design, name)
