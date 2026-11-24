"""The design: reads DESIGN.md's tokens into matplotlib's style and holds the values the other modules use."""
from pathlib import Path

import matplotlib
import matplotlib.style
import yaml
from cycler import cycler
from matplotlib import font_manager, rcParams

# fonts shipped beside this module (Inter, which no environment has installed): a DESIGN.md may name any of
# them, or any font installed on the system
for _font in [*Path(__file__).parent.glob("*.ttf"), *Path(__file__).parent.glob("*.otf")]:
    font_manager.fontManager.addfont(str(_font))


# the tokens the other modules and the scripts use, all set by apply_design()
colors = {}  # the colors tokens by name: colors["ours"], colors["neutral"], ...
series_order = []  # the colors tokens series on equal terms take, in order
series_colors = []  # their colors: series_colors[0] for the first series, [1] for the second, ...
shades = part_colors = {}  # L* settings of make_shades and make_part_colors
dashed_linewidth = None  # compared series: pass it to dashed and dotted lines
highlight_markersize = None  # the marker highlight-size, for a point of interest
filled_markeredgewidth = None  # the edge of filled markers
highlight_markeredgecolor = None  # the dark edge of highlight-size markers
canvas_pad = None  # inches of free space at each edge of the canvas
panel_gap = None  # free space between panels, as a share of the panel's size
legend_gap = None  # inches between a legend outside the panels and their box
value_label_small_fontsize = tick_label_text_fontsize = None  # points
colorbar_width = colorbar_gap = None  # inches
panel_heights = {}  # inches by step: panel_heights["S"] == 1.5
tick_spacing = None  # inches between numeric ticks
data_margin = None  # inches between the data and the panel edge


def _resolve_token_refs(value, tokens):
    """Replace token references such as "{colors.series-2}" with the value they point to."""
    if isinstance(value, dict):
        return {k: _resolve_token_refs(v, tokens) for k, v in value.items()}
    if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
        target = tokens
        for key in value[1:-1].split("."):
            target = target[key]
        return _resolve_token_refs(target, tokens)
    return value


def _points(value):
    """A token dimension ("3pt", "0.025in") in points, the unit matplotlib takes for sizes and pads."""
    number = float(str(value).rstrip("ptin"))
    return number * 72 if str(value).endswith("in") else number


def _inches(value):
    """A token dimension ("0.05in", "2pt") in inches, the unit of the canvas and the colorbar."""
    return _points(value) / 72


def _merge(base, extra):
    return {k: _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
            for k, v in {**base, **extra}.items()}


def apply_design(*paths):
    """Set matplotlib's style from paper.mplstyle and the tokens in the front matter of `paths`, DESIGN.md files:
    each later file changes only the tokens it names. Returns the colors by name, e.g. colors["ours"]."""
    global colors, series_order, series_colors, shades, part_colors, dashed_linewidth, highlight_markersize, filled_markeredgewidth, highlight_markeredgecolor, canvas_pad, panel_gap, legend_gap, colorbar_width, colorbar_gap, value_label_small_fontsize, tick_label_text_fontsize, panel_heights, tick_spacing, data_margin
    # one renderer everywhere: screen backends measure text differently, so finalize_figure's layout and
    # checks would differ from what Agg saves; the scripts write files and never open a window
    matplotlib.use("agg", force=True)
    tokens = {}
    for path in paths:
        tokens = _merge(tokens, yaml.safe_load(Path(path).read_text().split("---")[1]))
    tokens = _resolve_token_refs(tokens, tokens)
    c, t, s, k = tokens["colors"], tokens["typography"], tokens["spacing"], tokens["components"]
    colors = c
    matplotlib.style.use(Path(__file__).parent / "paper.mplstyle")
    value_label_small_fontsize = _points(t["value-label-small"]["fontSize"])
    tick_label_text_fontsize = _points(t["tick-label-text"]["fontSize"])
    label, tick_label, legend, value_label, title = (
        _points(t[name]["fontSize"]) for name in ("label", "tick-label", "legend", "value-label", "title"))
    line = _points(k["line"]["width"])
    weight = {role: line * k["line"][role] for role in ("data", "compared", "outline", "context")}
    marker = k["marker"]
    series_order = k["series"]["order"]
    series_colors = [c[name] for name in series_order]
    shades, part_colors = k["shades"], k["part-colors"]
    # palette reads this module's colors, so it is imported once they exist
    from .palette import make_diverging_colormap, make_sequential_colormap, shift_lightness
    # the legend frame, lighter than the grey compared series: data > frame > grid
    frame = shift_lightness(c["neutral"], k["frame"]["lighter"])
    tick_pad, tick_width = _points(s["tick-pad"]), weight["context"]
    rcParams.update({
        "font.sans-serif": [t["label"]["fontFamily"]],
        "font.size": value_label, "axes.labelsize": label, "figure.labelsize": label, "legend.fontsize": legend,
        "xtick.labelsize": tick_label, "ytick.labelsize": tick_label,
        "axes.titlesize": title, "axes.titleweight": t["title"].get("fontWeight", 400),
        "axes.titlepad": _points(s["title-pad"]), "axes.labelpad": _points(s["label-pad"]),
        "xtick.major.pad": tick_pad, "xtick.minor.pad": tick_pad, "ytick.major.pad": tick_pad, "ytick.minor.pad": tick_pad,
        "xtick.major.width": tick_width, "xtick.minor.width": tick_width,
        "ytick.major.width": tick_width, "ytick.minor.width": tick_width,
        "xtick.major.size": _points(k["tick"]["size"]), "ytick.major.size": _points(k["tick"]["size"]),
        "xtick.minor.size": _points(k["tick"]["minor-size"]), "ytick.minor.size": _points(k["tick"]["minor-size"]),
        "text.color": c["ink"], "axes.labelcolor": c["ink"], "xtick.color": c["ink"], "ytick.color": c["ink"],
        "xtick.labelcolor": c["ink"], "ytick.labelcolor": c["ink"],
        # the panel box: all four sides, black, thin
        "axes.edgecolor": c["ink"], "axes.linewidth": weight["context"],
        "grid.color": k["grid"]["color"], "grid.alpha": k["grid"]["opacity"], "grid.linewidth": weight["context"],
        "lines.linewidth": weight["data"],
        "lines.markersize": line * marker["size"], "lines.markeredgewidth": weight["context"], "lines.markeredgecolor": marker["edge"],
        "patch.linewidth": weight["outline"],
        "boxplot.boxprops.linewidth": weight["outline"],
        "boxplot.whiskerprops.linewidth": weight["context"],
        "boxplot.capprops.linewidth": weight["context"],
        "boxplot.medianprops.linewidth": weight["context"],
        "boxplot.flierprops.markersize": line * marker["size"], "boxplot.flierprops.markeredgewidth": weight["context"],
        "boxplot.flierprops.markeredgecolor": marker["edge"], "boxplot.flierprops.markerfacecolor": k["box"]["color"],
        **{f"boxplot.{part}props.color": k["box"]["color"] for part in ("box", "whisker", "cap", "median")},
        "contour.linewidth": weight["context"],
        "errorbar.elinewidth": weight["context"], "errorbar.capthick": weight["context"],
        "boxplot.meanprops.linewidth": weight["context"], "boxplot.flierprops.linewidth": weight["outline"],
        "savefig.transparent": True,  # a figure sits on the paper's own page: no white box around it
        "legend.linewidth": weight["context"], "legend.edgecolor": frame, "legend.facecolor": k["legend"]["fill"],
        "legend.handlelength": k["legend"]["handle-length"], "legend.handletextpad": k["legend"]["handle-text-pad"],
        "legend.columnspacing": k["legend"]["column-spacing"], "legend.labelspacing": k["legend"]["label-spacing"],
        "legend.borderpad": k["legend"]["border-pad"], "legend.borderaxespad": k["legend"]["border-axes-pad"], "hatch.linewidth": weight["context"],
        "axes.prop_cycle": cycler(color=series_colors + [c["ours"], c["neutral"]]),
        "image.cmap": "paper",
        "figure.constrained_layout.w_pad": _inches(s["layout-pad"]),
        "figure.constrained_layout.h_pad": _inches(s["layout-pad"]),
    })
    # by name, as matplotlib's own: cmap="paper" (the default, sequential) or cmap="paper-diverging"; a later
    # apply_design replaces them
    cmap = k["colormap"]
    matplotlib.colormaps.register(make_sequential_colormap("paper", cmap["sequential"], cmap["lightest"], cmap["darkest"]), force=True)
    matplotlib.colormaps.register(make_diverging_colormap("paper-diverging", cmap["negative"], cmap["positive"],
                                                          cmap["lightest"], cmap["darkest"]), force=True)
    dashed_linewidth = weight["compared"]
    highlight_markersize = line * marker["highlight-size"]
    highlight_markeredgecolor = marker["highlight-edge"]
    canvas_pad = _inches(s["canvas-pad"])
    panel_gap = float(s["panel-gap"].rstrip("%")) / 100
    legend_gap = _inches(s["legend-gap"])
    colorbar_width, colorbar_gap = _inches(k["colorbar"]["width"]), _inches(k["colorbar"]["gap"])
    panel_heights = {key.removeprefix("height-").upper(): _inches(v) for key, v in k["panel"].items()}
    tick_spacing = _inches(k["tick"]["spacing"])
    data_margin = _inches(s["data-margin"])
    filled_markeredgewidth = weight["context"]
    return c
