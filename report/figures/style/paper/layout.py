"""Sizes: finalize_figure sets each figure's height, the standard margins and panel widths, and where legends
and labels sit."""
import re
import warnings

import numpy as np
from matplotlib import rcParams
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.legend import Legend
from matplotlib.lines import AxLine, Line2D
from matplotlib.transforms import Bbox
from matplotlib.text import Annotation
from matplotlib.ticker import AutoLocator, FixedLocator, Formatter, FuncFormatter, LogFormatterSciNotation, MultipleLocator, NullFormatter, ScalarFormatter

from . import design, checks


def _panel_size(a, original=False):
    """Panel width and height in inches."""
    box = a.get_position(original=original).transformed(a.figure.transFigure)
    return box.width / a.figure.dpi, box.height / a.figure.dpi


def _measure_panels(fig, panels):
    """Each panel's height / width and width. NaN ratios when the layout collapsed at this height:
    matplotlib then leaves the panels where they were, so their sizes mean nothing."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.canvas.draw()
    sizes = [_panel_size(a) for a in panels]
    if any("constrained_layout not applied" in str(w.message) for w in caught):
        return [np.nan] * len(panels), [w for w, _ in sizes]
    for w in caught:
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    return [h / w for w, h in sizes], [w for w, _ in sizes]


def _fit_height_to_aspect(fig, panels, aspect, iters, tol):
    h0 = fig.get_figheight()
    ratios, widths = _measure_panels(fig, panels)
    for _ in range(6):  # the start is too short for the decorations: grow until the layout stands
        if np.isfinite(ratios[0]):
            break
        h0 *= 2
        fig.set_figheight(h0)
        ratios, widths = _measure_panels(fig, panels)
    if not np.isfinite(ratios[0]):
        warnings.warn("the layout collapses at any height: the panels are too narrow for their labels and "
                      "legend; use one column fewer or a wider figure")
        return ratios, widths
    r0 = ratios[0]
    h1 = h0 * aspect / r0
    for _ in range(iters):
        fig.set_figheight(h1)
        ratios, widths = _measure_panels(fig, panels)
        r1 = ratios[0]
        if not np.isfinite(r1) or r1 <= 0:
            # too short for the decorations (titles, outside legends): the layout collapsed
            h0, r0, h1 = h1, r0, 2 * h1
            continue
        if abs(r1 - aspect) / aspect < tol or r1 == r0:
            break
        step = h1 + (aspect - r1) * (h1 - h0) / (r1 - r0)
        # a secant step can overshoot below zero while the layout is still settling: halve instead; on a
        # nearly flat ratio (a legend wider than its panel) it can shoot far up: at most four times taller
        h0, r0, h1 = h1, r1, min(step, 4 * h1) if np.isfinite(step) and step > 0 else h1 / 2
    return ratios, widths


def _set_canvas_pad(fig):
    """Free space on all four sides of the canvas, for text that reaches past the panels."""
    if isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine):
        # left and right as a rect; top and bottom as the layout's edge pad, which outside legends obey too
        x = design.canvas_pad / fig.get_figwidth()
        fig.get_layout_engine().set(rect=(x, 0, 1 - 2 * x, 1), h_pad=design.canvas_pad)


def _freeze_ticks(panels):
    """Fix the ticks where they are now; the locator they came from is kept for `_unfreeze_ticks`."""
    for a in panels:
        for axis in (a.xaxis, a.yaxis):
            if not hasattr(axis, "_auto_locator"):
                axis._auto_locator = axis.get_major_locator()
            lo, hi = sorted(axis.get_view_interval())
            axis.set_major_locator(FixedLocator([t for t in axis.get_majorticklocs() if lo <= t <= hi]))


def _unfreeze_ticks(panels):
    for a in panels:
        for axis in (a.xaxis, a.yaxis):
            if hasattr(axis, "_auto_locator"):
                axis.set_major_locator(axis._auto_locator)


def _limit_leaving_room(axis, lo, hi, v, reach, size, upper):
    """The new upper (or lower) limit that leaves `reach` pixels between value `v` and the panel edge,
    solved in the axis's scale (linear, log), where data maps linearly to pixels."""
    scale = axis.get_transform()
    lo, hi, v = scale.transform([lo, hi, v])
    limit = lo + (v - lo) * size / (size - reach) if upper else (v * size - reach * hi) / (size - reach)
    return scale.inverted().transform([limit])[0]


def _keep_value_labels_inside(fig, panels):
    """Grow each panel's limits so every tagged value label, bar label (`ax.bar_label`) and point name
    (an annotation in data coordinates) lies inside its panel. Tagged annotations centered over
    their mark turn inward at a side edge instead. A point name gets at most a quarter more axis range; one that needs more
    is left out and reported, to be moved. Returns whether any limits changed."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    changed = False
    for a in panels:
        if checks._is_round_panel(a):
            continue
        box = a.get_window_extent(renderer)
        for t in a.texts:
            # a value centered over its mark turns inward at a side edge; one beside its mark makes room
            turns_inward = t.get_gid() == "value-label" and isinstance(t, Annotation) and t.get_ha() == "center"
            value_label = checks._is_value_label(a, t)
            point_name = not value_label and isinstance(t, Annotation) and t.xycoords == "data"
            if not (value_label or point_name):
                continue
            label = t.get_window_extent(renderer)
            gap = 0.35 * t.get_fontsize() * fig.dpi / 72  # clear space between a label and the panel edge
            # a label's reach past its point is fixed in pixels, so solve for the limit
            # that leaves exactly that much room between the point and the box
            x, y = t.xy if isinstance(t, Annotation) else t.get_position()
            point_x, point_y = a.transData.transform((x, y))
            for axis, low_edge, high_edge, v, point, size, set_lim, get_lim in (
                    (a.xaxis, label.x0 < box.x0 + gap, label.x1 > box.x1 - gap, x, point_x, box.width, a.set_xlim, a.get_xlim),
                    (a.yaxis, label.y0 < box.y0 + gap, label.y1 > box.y1 - gap, y, point_y, box.height, a.set_ylim, a.get_ylim)):
                if not (low_edge or high_edge):
                    continue
                if turns_inward and axis is a.xaxis:
                    t.set_ha("left" if low_edge else "right")
                    continue
                lo, hi = get_lim()
                if high_edge:
                    reach = (label.x1 if axis is a.xaxis else label.y1) - point + gap
                    new = (lo, _limit_leaving_room(axis, lo, hi, v, reach, size, True))
                else:
                    reach = point - (label.x0 if axis is a.xaxis else label.y0) + gap
                    new = (_limit_leaving_room(axis, lo, hi, v, reach, size, False), hi)
                if lo >= 0 and new[0] < 0:
                    # a non-negative quantity never shows negative values to make room
                    new = (0, new[1])
                scale = axis.get_transform()
                old_span = np.ptp(scale.transform([lo, hi]))
                if point_name and np.ptp(scale.transform(list(new))) > 1.25 * old_span:
                    continue
                set_lim(*new)
                changed = True
    return changed


def _fits_at(texts, room, renderer):
    """Whether every text fits its room (width, height in pixels) at its current size."""
    return all(t.get_window_extent(renderer).width <= w and t.get_window_extent(renderer).height <= h
               for t, (w, h) in zip(texts, room))


def _fit_heatmap_values(fig, panels):
    """Every row and column of a heatmap is named and every cell's value written. Cell values and names take
    the value-label size (7 pt), or the small one (5 pt) when that does not fit; not fitting at 5 pt either,
    the checks report them."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    space = 0.7 * fig.dpi / 72  # the word space checks._texts_overlap asks between neighbours, per point of size
    for a in [a for a in panels if a.images]:
        rows, cols = a.images[0].get_array().shape[:2]
        panel = a.get_window_extent(renderer)
        cell_w, cell_h = panel.width / cols, panel.height / rows
        cells = [t for t in a.texts if t.get_ha() == "center" and t.get_va() == "center" and t.get_visible() and t.get_text()]
        for t in cells:
            t._in_cell = True
        for size in (rcParams["font.size"], design.value_label_small_fontsize):
            for t in cells:
                t.set_fontsize(size)
            if _fits_at(cells, [(0.9 * cell_w, 0.8 * cell_h)] * len(cells), renderer):
                break
        # each axis keeps the size its labels already have (numbers 5 pt, names 7 pt, `_size_text_tick_labels`)
        # and takes the small size only where its names do not fit their cells
        for axis, room in ((a.xaxis, lambda size: (cell_w - space * size, np.inf)), (a.yaxis, lambda size: (np.inf, cell_h))):
            labels = [t for t in axis.get_ticklabels() if t.get_visible() and t.get_text()]
            if labels and not _fits_at(labels, [room(labels[0].get_fontsize())] * len(labels), renderer):
                axis.set_tick_params(labelsize=design.value_label_small_fontsize)


def _shrink_crowded_texts(fig, panels):
    """Text inside a panel at the value-label size (7 pt) that overlaps other text or spills out of the fill it
    sits on takes the small size (5 pt); every text of its panel that is alike (bar labels, point names) too."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for a in panels:
        if a.images:
            continue  # heatmaps: _fit_heatmap_values
        texts = [t for t in a.texts if t.get_visible() and t.get_text() and t.get_bbox_patch() is None]
        boxes = [t.get_window_extent(renderer) for t in texts]
        fills = checks._filled_shapes(a)
        crowded = any(checks._texts_overlap(texts[i], boxes[i], texts[j], boxes[j], renderer)
                      for i in range(len(texts)) for j in range(i + 1, len(texts)))
        spills = any(f.contains_point(((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2))
                     and not f.contains_points([(b.x0, b.y0), (b.x1, b.y0), (b.x1, b.y1), (b.x0, b.y1)]).all()
                     for b in boxes for f in fills)
        if crowded or spills:
            for t in texts:
                t.set_fontsize(design.value_label_small_fontsize)


def _turn_crowded_bar_labels(fig, panels):
    """Bar values written side by side that overlap all turn upward, in every panel where any two do."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for a in panels:
        anchors = checks._bar_label_anchors(a)
        labels = [t for t in a.texts if isinstance(t, Annotation) and t.get_rotation() == 0 and len(anchors)
                  and bool(np.isclose(anchors, t.xy).all(axis=1).any()) and t.get_va() in ("bottom", "top")]
        boxes = [t.get_window_extent(renderer) for t in labels]
        if any(checks._texts_overlap(labels[i], boxes[i], labels[j], boxes[j], renderer)
               for i in range(len(labels)) for j in range(i + 1, len(labels))):
            for t in labels:
                t.set_rotation(90)
                t.xyann = (0, 2 if t.get_va() == "bottom" else -2)


INSIDE_SPOTS = ["upper left", "upper right", "lower left", "lower right",
                "upper center", "lower center", "center left", "center right"]


def _clear_spot(fig, a, legend, panels, renderer):
    """Put the legend at the first inside spot of panel `a` where it hides nothing; whether one was found.
    The legend is marked as placed here, so `_recheck_legends` can move it again at the final size."""
    legend._placed_inside = True
    for spot in INSIDE_SPOTS:
        legend.set_loc(spot)
        if not checks._legend_problems(fig, a, panels, renderer):
            return True
    return False


def _recheck_legends(fig, panels):
    """Panels change size a little when they are put on the standard box: a legend placed inside earlier
    may now touch data; look again for a clear spot in its panel, else put it above. Returns whether
    any went above (the panels then need placing again)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    above = False
    for a in panels:
        legend = a.get_legend()
        if legend is None or not getattr(legend, "_placed_inside", False) or not checks._legend_problems(fig, a, panels, renderer):
            continue
        if not _clear_spot(fig, a, legend, panels, renderer):
            handles, labels, layout = _legend_entries(legend)
            legend.remove()
            if not _place_inside(fig, [a], handles, labels, panels, renderer, layout):
                _legend_above(fig, a, handles, labels, several=len(panels) > 1, layout=layout)
                above = True
    return above


def _legend_entries(legend):
    """A legend's handles and labels in the order it was made, and its layout (columns, spacing) when its
    rows are set (`add_legend_rows`), else None."""
    if hasattr(legend, "_entries"):
        return legend._entries
    return legend.legend_handles, [t.get_text() for t in legend.get_texts()], None


def _place_inside(fig, candidates, handles, labels, panels, renderer, layout=None):
    """Put the legend inside the first of `candidates` with a spot where it hides nothing. Whether it found one."""
    for a in candidates:
        legend = a.legend(handles, labels, **(layout or {}))
        if layout:
            legend._entries = (handles, labels, layout)
        if _clear_spot(fig, a, legend, panels, renderer):
            return True
        legend.remove()
    return False


def _legend_above(fig, a, handles, labels, several, layout=None):
    """A legend above the panels, in as few rows as fit the figure's width (or in its set rows, `layout`).
    Above several panels it is the figure's legend: an axes legend that wide would count as part of its one
    panel and collapse the layout."""
    room = fig.get_figwidth() * fig.dpi - 2 * design.canvas_pad * fig.dpi
    fixed = layout is not None
    style = {k: v for k, v in (layout or {}).items() if k != "ncols"}
    for ncols in [layout["ncols"]] if fixed else range(len(labels), 0, -1):
        # matplotlib fills a legend column by column; reorder so its rows read left to right in entry order
        # (set rows come in matplotlib's order already)
        order = list(range(len(labels))) if fixed else [i for c in range(ncols) for i in range(c, len(labels), ncols)]
        h, l = [handles[i] for i in order], [labels[i] for i in order]
        if several:
            legend = fig.legend(h, l, loc="outside upper center", ncols=ncols, borderaxespad=0, **style)
        else:
            legend = a.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1), ncols=ncols,
                              borderaxespad=design.legend_gap * 72 / rcParams["legend.fontsize"], **style)
        if fixed or ncols == 1 or legend.get_window_extent(fig.canvas.get_renderer()).width <= room:
            return
        legend.remove()


def _lift_legends_wider_than_panel(fig, panels):
    """A panel legend made without `loc` that is wider than its panel can never go inside it, and while it
    stays there matplotlib's layout collapses: it goes above the panels at once."""
    renderer = fig.canvas.get_renderer()
    for a in panels:
        legend = a.get_legend()
        if legend is None or legend._loc != Legend.codes["best"]:
            continue
        if legend.get_window_extent(renderer).width > a.get_position().width * fig.bbox.width:
            handles, labels, layout = _legend_entries(legend)
            legend.remove()
            _legend_above(fig, a, handles, labels, several=len(panels) > 1, layout=layout)


def _place_legends(fig, panels):
    """A panel legend made without `loc` names series that may run through every panel, so it goes
    inside the first panel, starting with its own, that has a spot where it hides nothing; the other panels carry none. Where no panel has room, it goes above its own panel. Returns
    whether any went above."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    above = False
    for a in panels:
        legend = a.get_legend()
        if legend is None or legend._loc != Legend.codes["best"]:
            continue
        handles, labels, layout = _legend_entries(legend)
        legend.remove()
        candidates = [a, *[o for o in panels if o is not a and o.get_legend() is None]]
        if not _place_inside(fig, candidates, handles, labels, panels, renderer, layout):
            _legend_above(fig, a, handles, labels, several=len(panels) > 1, layout=layout)
            above = True
    return above


def _standard_y_widths(fig):
    """The standard widths, in inches, of the y decorations: four-character tick labels
    and their pads, and one line of the axis label with its pad."""
    renderer = fig.canvas.get_renderer()
    sample = fig.text(0, 0, "0000", fontsize=rcParams["ytick.labelsize"])  # four characters: −0.5, 1000, 2.5K, 100%
    text = sample.get_window_extent(renderer).width / fig.dpi
    sample.remove()
    ticks = text + (rcParams["ytick.major.pad"] + rcParams["ytick.major.size"]) / 72
    label = (rcParams["axes.labelsize"] * 1.35 + rcParams["axes.labelpad"]) / 72  # rotated, with room for (, β, _
    return ticks, label, text / 2


def _standard_inner_y_width(column, ticks, label):
    """The room a column's y decorations take at least, as the left margin does: four-character tick labels
    when it shows tick labels, and a label line when it has a label. So two figures of one shape match,
    whether a column's ticks read "90" or "100"."""
    shows_ticks = any(t.get_visible() and t.get_text() for a in column for t in a.get_yticklabels())
    return (ticks if shows_ticks else 0) + (label if any(a.get_ylabel() for a in column) else 0)


def _standard_side_margins(fig):
    """The left and right margins, in inches, of a standard panel: on each side a y-axis label and
    four-character tick labels. The right one holds a second y axis (`ax.twinx()`) whether or not
    the figure has one, so a panel with a twin is as wide as one without."""
    ticks, label, half = _standard_y_widths(fig)
    return design.canvas_pad + label + ticks, design.canvas_pad + label + ticks


def _center_row_legends(fig, panels):
    """A legend above a panel names the series of its whole row, so it is centred over the row."""
    fig.canvas.draw()
    for a in panels:
        legend = a.get_legend()
        box, panel = (legend.get_window_extent() if legend else None), a.get_position()
        if not legend or box.y0 < a.get_window_extent().y1 - 1:
            continue
        row = [p.get_position() for p in panels if abs(p.get_position().y1 - panel.y1) < 1e-3]
        center = (min(r.x0 for r in row) + max(r.x1 for r in row)) / 2
        legend.set_loc("lower center")
        # the anchor sits below the legend by its border pad, which matplotlib adds again above the anchor
        pad = legend.borderaxespad * legend._fontsize * fig.dpi / 72
        legend.set_bbox_to_anchor((center, (box.y0 - pad) / fig.bbox.height), transform=fig.transFigure)


def _unframe_outside_legends(fig):
    """A legend outside the panels sits on white with nothing under it: its box would only add a line. Inside a
    panel the box keeps it apart from the grid and the data."""
    fig.canvas.draw()
    for legend in fig.legends:
        legend.set_frame_on(False)
    for a in fig.axes:
        legend = a.get_legend()
        box = legend.get_window_extent() if legend else None
        if legend and not a.get_window_extent().contains((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2):
            legend.set_frame_on(False)


def _y_decoration_width(axes, side="left"):
    """How far, in inches, the y tick labels and y label of these axes reach out from their panels on
    `side`; 0 when they show none."""
    renderer = axes[0].figure.canvas.get_renderer()
    widths = [0.0]
    for a in axes:
        if not any(t.get_visible() and t.get_text() for t in a.get_yticklabels()) and not a.yaxis.label.get_text():
            continue
        box, panel = a.yaxis.get_tightbbox(renderer), a.get_window_extent(renderer)
        if box is not None:
            widths.append((panel.x0 - box.x0 if side == "left" else box.x1 - panel.x1) / a.figure.dpi)
    return max(widths)


def _x_end_clearance(left_column, right_column):
    """The gap, in inches, that keeps the last x tick label of the panels on the left half an em clear of
    the first one of the panels on the right, so "2000" and "0" never read as "20000"."""
    renderer = left_column[0].figure.canvas.get_renderer()

    def end_label(a, last):
        lo, hi = sorted(a.xaxis.get_view_interval())
        shown = [(loc, t) for loc, t in zip(a.xaxis.get_majorticklocs(), a.xaxis.get_majorticklabels())
                 if lo <= loc <= hi and t.get_visible() and t.get_text()]
        return (max if last else min)(shown, key=lambda p: p[0])[1] if shown else None

    def reach(a, last):
        label = end_label(a, last)
        if label is None:
            return 0.0
        box, panel = label.get_window_extent(renderer), a.get_window_extent(renderer)
        return max(0.0, (box.x1 - panel.x1) if last else (panel.x0 - box.x0)) / a.figure.dpi

    half_em = rcParams["xtick.labelsize"] / 72 / 2
    return max(reach(a, True) for a in left_column) + max(reach(a, False) for a in right_column) + half_em


def _colorbar_end_clearance(left_bar, right_bar):
    """The gap, in inches, that keeps two colorbars side by side clear at their facing ends: their ticks are
    laid out again at the final length, so a tick may land on either end, its label centered there; half the
    widest tick label of each, and half an em between."""
    renderer = left_bar.figure.canvas.get_renderer()
    half_widest = [max((t.get_window_extent(renderer).width for t in bar.get_xticklabels() if t.get_text()), default=0) / 2
                   for bar in (left_bar, right_bar)]
    return sum(half_widest) / left_bar.figure.dpi + rcParams["xtick.labelsize"] / 72 / 2


def _x_tick_height(fig):
    """x tick labels with their pad and tick mark, in inches."""
    sample = fig.text(0, 0, "0", fontsize=rcParams["xtick.labelsize"])
    height = sample.get_window_extent(fig.canvas.get_renderer()).height / fig.dpi
    sample.remove()
    return height + (rcParams["xtick.major.pad"] + rcParams["xtick.major.size"]) / 72


def _half_tick_label_height(fig):
    """Half a tick label: how far the top y tick label reaches above its panel."""
    return (_x_tick_height(fig) - (rcParams["xtick.major.pad"] + rcParams["xtick.major.size"]) / 72) / 2


def _title_line_height():
    return (rcParams["axes.titlesize"] * 1.35 + rcParams["axes.titlepad"]) / 72


def _line_count(text):
    return text.get_text().count("\n") + 1 if text.get_text() else 0


def _place_on_standard_box(fig, panels, aspect, panel_height=None):
    """Panels sit at the standard margins and take the standard size, so the boxes of all figures at
    one width (and one number of columns) line up and match. The right margin is fixed; left labels
    that need more room widen the left margin and narrow the panels, which keep the standard height
    (the aspect gives way). Between columns: the panel-gap share of the panel width plus the column's own y
    labels; between rows: the labels between them plus the panel-gap share of the height.
    One colorbar is kept beside the panels (right) or under them (bottom), its labels inside a standard
    margin of their own; several colorbars stand one beside each column of panels, each with the room its
    labels take before the next column, or one under each column of a single row, their end labels apart. Returns whether they moved."""
    cbars = [a for a in fig.axes if getattr(a, "_colorbar", None) is not None]
    others = [a for a in fig.axes if a not in panels and a not in cbars and a.get_visible() and a.axison]  # an empty grid cell is no axis
    twin_of = {t: p for t in others for p in panels
               if np.allclose(t.get_position().bounds, p.get_position().bounds)}  # ax.twinx(): a right y axis
    owned = (all(getattr(c._colorbar.mappable, "axes", None) in panels for c in cbars)
             and len({c._colorbar.orientation for c in cbars}) == 1)
    if (fig.subfigs or (len(cbars) > 1 and not owned) or len(twin_of) != len(others)
            or checks._is_round_panel(panels[0])):
        return False
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    w, h = fig.get_size_inches()
    boxes = {a: a.get_position() for a in panels}
    columns = sorted({round(b.x0 * w, 2) for b in boxes.values()})
    col_of = {a: columns.index(round(b.x0 * w, 2)) for a, b in boxes.items()}
    row_ys = sorted({round(b.y0 * h, 2) for b in boxes.values()}, reverse=True)
    row_of = {a: row_ys.index(round(b.y0 * h, 2)) for a, b in boxes.items()}
    rows = range(max(row_of.values()) + 1)
    column_cbar = {col_of[c._colorbar.mappable.axes]: c for c in cbars} if len(cbars) > 1 else {}
    if len(cbars) > 1 and len(column_cbar) < len(cbars):
        return False  # two colorbars for one column
    vertical_bars = bool(cbars) and cbars[0]._colorbar.orientation == "vertical"
    beside = column_cbar if vertical_bars else {}
    under = {} if vertical_bars else column_cbar
    if under and len(rows) > 1:
        return False
    left, right = _standard_side_margins(fig)
    side = left - design.canvas_pad  # an axis label and four-character tick labels
    pad = design.canvas_pad
    cbar = cbars[0] if len(cbars) == 1 else None
    vertical = cbar is not None and cbar._colorbar.orientation == "vertical"
    # a colorbar stands off its panels as far as panels stand off each other: the panel-gap share of their width
    cb_gap, cb_w = design.panel_gap * np.mean([b.width * w for b in boxes.values()]), design.colorbar_width
    # the right margin never moves, so every figure's right edge lines up; a left label that needs
    # more room takes it from the panels' width, never their height, so one width keeps one height
    # a colorbar's labels take the standard margin, or what they need when wider ("0.945"), so the canvas pad stays clear
    last_cbar = cbar if vertical else beside.get(len(columns) - 1)
    if last_cbar is not None:
        right = pad + max(side, _y_decoration_width([last_cbar], "right")) + cb_w + cb_gap
    std_left, std_right = left, right
    ticks, label, half = _standard_y_widths(fig)
    need = max(b.x0 * w - a.get_tightbbox(renderer).x0 / fig.dpi + pad for a, b in boxes.items() if col_of[a] == 0)
    sup_y = fig._supylabel if fig._supylabel and fig._supylabel.get_text() else None
    if sup_y:  # it stands in for the panels' y labels, whose room the standard margin already holds
        need += label
    left = max(left, need)
    # a right y axis mirrors the left one: its tick labels and label take room from the panels' width
    right_column = {col_of[p] for p in twin_of.values()}
    right_axis = ticks + label * max((_line_count(t.yaxis.label) for t in twin_of), default=0)
    if len(columns) - 1 in right_column and not vertical:
        # wider tick labels than the standard four characters take room from the panels, as on the left
        right = pad + max(right_axis, _y_decoration_width([t for t, p in twin_of.items() if col_of[p] == len(columns) - 1], "right"))
    # inside the grid, a right y axis takes only what it shows (none when its tick labels are hidden)
    column_panels = {c: [a for a in panels if col_of[a] == c] for c in range(len(columns))}
    # free space between panels is a share of their size: of the width the layout gave them between columns,
    # of the height between rows
    free_x = design.panel_gap * np.mean([b.width * w for b in boxes.values()])
    for _ in range(3):  # the gaps and the panel width they are a share of settle together
        col_gaps = [max(free_x + max(_y_decoration_width(column_panels[c]), _standard_inner_y_width(column_panels[c], ticks, label))
                        + (_y_decoration_width([t for t, p in twin_of.items() if col_of[p] == c - 1], "right") if c - 1 in right_column else 0)
                        + (cb_gap + cb_w + _y_decoration_width([beside[c - 1]], "right") if c - 1 in beside else 0),
                        _x_end_clearance([a for a in panels if col_of[a] == c - 1], [a for a in panels if col_of[a] == c]),
                        _colorbar_end_clearance(under[c - 1], under[c]) if c in under and c - 1 in under else 0)
                    for c in range(1, len(columns))]
        free_x = design.panel_gap * (w - std_left - std_right - sum(col_gaps)) / len(columns)
    # rows: above each, half a tick label or a title line; below, x tick labels and x label lines;
    # anything more (a legend above a panel, a second line, turned labels) widens that margin
    above, below = [], []
    for r in range(len(rows)):
        row = [a for a in panels if row_of[a] == r]
        tight = [Bbox.union([a.get_tightbbox(renderer)] + [t.get_tightbbox(renderer) for t, p in twin_of.items() if p is a])
                 for a in row]
        y0, y1 = boxes[row[0]].y0 * h, boxes[row[0]].y1 * h
        std_above = max(_half_tick_label_height(fig), max(_line_count(a.title) for a in row) * _title_line_height())
        std_below = (_x_tick_height(fig) * any(t.label1.get_visible() and t.label1.get_text()
                                             for a in row for t in a.xaxis.get_major_ticks())
                     + label * max(_line_count(a.xaxis.label) for a in row))
        above.append(max(std_above, max(b.y1 for b in tight) / fig.dpi - y1))
        below.append(max(std_below, y0 - min(b.y0 for b in tight) / fig.dpi))
    row_gaps = [below[r] + above[r + 1] for r in range(len(rows) - 1)]
    # a legend made "outside upper" belongs above the panels even while it is still drawn over them
    top_legends = [lg for lg in fig.legends if getattr(lg, "_outside_loc", None) == "upper"
                   or lg.get_window_extent(renderer).y0 / fig.dpi >= max(b.y1 for b in boxes.values()) * h - 1e-3]
    legend_h = sum(lg.get_window_extent(renderer).height / fig.dpi for lg in top_legends)
    top = pad + legend_h + (max(above[0], design.legend_gap) if top_legends else above[0])
    sup_x = fig._supxlabel if fig._supxlabel and fig._supxlabel.get_text() else None
    bottom = pad + below[-1] + label * _line_count(sup_x) if sup_x else pad + below[-1]
    if (cbar is not None and not vertical) or under:
        # a colorbar under the panels (or one under each column): as thick as one beside them, the colorbar gap
        # under the x labels; its own tick labels and label keep the room the layout gave them below it
        cb_bottom, cb_h = min(c.get_position().y0 for c in cbars) * h, design.colorbar_width
        bottom = cb_bottom + cb_h + design.colorbar_gap + below[-1]
    n = len(columns)
    # a height step is inches at any width; a shape (SQ, square heatmap cells) follows the panel's own width
    panel_h = panel_height or aspect * (w - std_left - std_right - sum(col_gaps)) / n
    row_gaps = [g + design.panel_gap * panel_h for g in row_gaps]
    col_x, col_w = {}, {}
    x, width = left, (w - right - left - sum(col_gaps)) / n
    for c in range(n):
        col_x[c], col_w[c] = x, width
        x += width + (col_gaps[c] if c < n - 1 else 0)
    row_h = [panel_h] * len(rows)
    new_h = bottom + top + sum(row_h) + sum(row_gaps)
    fig.set_layout_engine("none")
    fig.set_figheight(new_h)
    for a in panels:
        c, r = col_of[a], row_of[a]
        y0 = bottom + sum(row_h[r + 1:]) + sum(row_gaps[r:])
        a.set_position([col_x[c] / w, y0 / new_h, col_w[c] / w, row_h[r] / new_h])
    for t, p in twin_of.items():
        t.set_position(p.get_position())
    for c, cb in beside.items():
        y0 = min(a.get_position().y0 for a in column_panels[c])
        y1 = max(a.get_position().y1 for a in column_panels[c])
        cb.set_position([(col_x[c] + col_w[c] + cb_gap) / w, y0, cb_w / w, y1 - y0])
    for c, cb in under.items():
        cb.set_position([col_x[c] / w, cb_bottom / new_h, col_w[c] / w, cb_h / new_h])
    span_w = col_x[n - 1] + col_w[n - 1] - left
    span_h = sum(row_h) + sum(row_gaps)
    if sup_x:
        sup_x.set_x((left + span_w / 2) / w)
        sup_x.set_y(pad / new_h)
        sup_x.set_verticalalignment("bottom")
    y = new_h - pad
    for lg in top_legends:
        # centred over the panels, but never past the canvas pad: a wide legend shifts toward the figure's middle
        half_w = lg.get_window_extent(renderer).width / fig.dpi / 2
        x = min(max(left + span_w / 2, pad + half_w), w - pad - half_w)
        lg.set_loc("upper center")
        lg.set_bbox_to_anchor((x / w, y / new_h), transform=fig.transFigure)
        y -= lg.get_window_extent(renderer).height / fig.dpi
    if sup_y:
        sup_y.set_x(pad / w)
        sup_y.set_horizontalalignment("left")
        sup_y.set_y((bottom + span_h / 2) / new_h)
    if vertical:
        cbar.set_position([(left + span_w + cb_gap) / w, bottom / new_h, cb_w / w, span_h / new_h])
    elif cbar is not None:
        cbar.set_position([left / w, cb_bottom / new_h, span_w / w, cb_h / new_h])
    for c in cbars:
        c.set_box_aspect(None)  # placed at their exact size: no cap on their length (`_colorbars_span_panels`)
    return True


def _blank_touching_corner_label(fig, panels):
    """Where the first tick labels of both axes would touch at the shared corner, blank the y one."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for a in panels:
        firsts = []
        for axis in (a.xaxis, a.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            shown = [(loc, t) for loc, t in zip(axis.get_majorticklocs(), axis.get_majorticklabels())
                     if lo <= loc <= hi and t.get_visible() and t.get_text()]
            firsts.append(min(shown, key=lambda p: p[0]) if shown else None)
        if None in firsts:
            continue
        (_, tx), (loc, ty) = firsts
        if checks._texts_overlap(tx, tx.get_window_extent(renderer), ty, ty.get_window_extent(renderer), renderer):
            label = a.yaxis.get_major_formatter()
            a.yaxis.set_major_formatter(FuncFormatter(lambda v, pos, first=loc, label=label: "" if np.isclose(v, first) else label(v, pos)))


def _start_at_zero(panels):
    """A quantity that is never negative never shows negative values: raise a default margin below 0 to 0."""
    for a in panels:
        if checks._is_round_panel(a):
            continue
        lim = a.dataLim
        for get, set_, low, scale in ((a.get_xlim, a.set_xlim, lim.x0, a.get_xscale()),
                                      (a.get_ylim, a.set_ylim, lim.y0, a.get_yscale())):
            lo, hi = get()
            if scale == "linear" and np.isfinite(low) and low > 0 and lo < 0 < hi:
                set_(0, hi)


NICE = (1, 2, 2.5, 5)


def _nice_step(raw):
    """The largest step of 1, 2, 2.5 or 5 x 10^n up to a quarter above `raw`: when in doubt, more ticks."""
    power = 10.0 ** np.floor(np.log10(raw))
    return max(m * power * k for m in NICE for k in (0.1, 1, 10) if m * power * k <= 1.25 * raw)


def _next_step(step):
    power = 10.0 ** np.floor(np.log10(step) + 1e-9)
    return min(m * power * k for m in NICE for k in (1, 10) if m * power * k > step * (1 + 1e-9))


def _step_for(axis, length):
    """A tick about every tick spacing; on x, wider until the widest label and a word space fit between ticks."""
    lo, hi = sorted(axis.get_view_interval())
    step = _nice_step((hi - lo) * design.tick_spacing / length)
    if axis.axis_name == "x":
        size = rcParams["xtick.labelsize"]
        while True:
            ticks = np.arange(np.ceil(lo / step) * step, hi + step / 2, step)
            widest = max(len(f"{t:.10g}") for t in ticks) * 0.62 * size / 72 + 0.7 * size / 72
            if widest <= step * length / (hi - lo):
                break
            step = _next_step(step)
    return step


def _margins_by_length(fig, panels, aspect):
    """The data-margin token, in inches, between the data and each panel edge: as a share of the size the
    panel will take. Axes with limits the script set keep them."""
    for a in panels:
        w = _panel_size(a)[0]
        for owner in [a] + [t for t in fig.axes if t is not a and np.allclose(t.get_position().bounds, a.get_position().bounds)]:
            owner.set_xmargin(design.data_margin / w)
            owner.set_ymargin(design.data_margin / (aspect * w))
            owner.autoscale_view()


def _ticks_by_length(fig, panels, aspect):
    """matplotlib's default numeric ticks, one about every tick spacing along the axis, at a step of 1, 2,
    2.5 or 5 x 10^n: the panel's size, not the data, sets how many. Sizes are the ones the panels will take."""
    for a in [a for a in panels if not a.images]:  # a heatmap's ticks name its rows and columns
        w = _panel_size(a)[0]
        twins = [t for t in fig.axes if t is not a and np.allclose(t.get_position().bounds, a.get_position().bounds)]
        for owner in [a, *twins]:
            for axis, length, scale in ((owner.xaxis, w, owner.get_xscale()), (owner.yaxis, aspect * w, owner.get_yscale())):
                if type(axis.get_major_locator()) is AutoLocator and scale == "linear":
                    axis.set_major_locator(MultipleLocator(_step_for(axis, length)))


class _TickText(Formatter):
    """Tick numbers as plain numbers (0, 0.5, 1) with a true minus sign; from 1000 up with an SI suffix,
    one unit for the whole axis, when that writes the ticks shorter (0.5K, 1K, 1.5K, 2K; but 200 ... 1000
    and 1024 stay). The axis's ticks come from `set_locs`, once per axis update: asking the axis for
    them on every tick made this the slowest step of `finalize_figure`."""
    ticks = []

    @staticmethod
    def plain(v):
        return f"{v:.10g}".replace("-", "\u2212")

    def set_locs(self, locs):
        self.ticks = list(locs)

    def __call__(self, v, pos=None):
        ticks = self.ticks
        top = max((abs(t) for t in ticks), default=0)
        for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
            if top >= scale:
                short = lambda t: "0" if t == 0 else self.plain(t / scale) + suffix
                if sum(map(len, map(short, ticks))) < sum(map(len, map(self.plain, ticks))):
                    return short(v)
                break
        return self.plain(v)


def _log_tick_text(v, _):
    """A log axis's tick: plain from 0.001 to 1000, an SI suffix above (10K, 1M), 1e−4 below."""
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e4, "K")):
        if v >= scale:
            return f"{v / (1e3 if suffix == 'K' else scale):.10g}{suffix}"
    if v >= 1e-3:
        return f"{v:.10g}"
    return f"1e{np.log10(v):.0f}".replace("-", "\u2212") if np.isclose(v, 10 ** np.round(np.log10(v))) else f"{v:.1e}".replace("-", "\u2212")


def _write_plain_tick_numbers(fig):
    """Write matplotlib's default numeric ticks short and plain (`_TickText`, `_log_tick_text`); a log
    axis's minor ticks stay unlabelled."""
    for a in fig.axes:
        for axis in (a.xaxis, a.yaxis):
            if type(axis.get_major_formatter()) is ScalarFormatter:
                axis.set_major_formatter(_TickText())
            elif type(axis.get_major_formatter()) is LogFormatterSciNotation:
                axis.set_major_formatter(FuncFormatter(_log_tick_text))
                axis.set_minor_formatter(NullFormatter())


NUMBER = re.compile(r"[\u2212+-]?[\d.,]+(e[\u2212+-]?\d+)?[KMB%]?")


def _size_text_tick_labels(fig):
    """Tick labels that are names, not numbers (methods, benchmarks), at the tick-label-text size."""
    fig.canvas.draw()
    for a in fig.axes:
        for axis in (a.xaxis, a.yaxis):
            texts = [t.get_text() for t in axis.get_ticklabels() if t.get_text()]
            if any(not NUMBER.fullmatch(t) for t in texts):
                axis.set_tick_params(labelsize=design.tick_label_text_fontsize)


def _pad_outside_legends(fig):
    """The layout puts a figure legend outside the panels flush with the canvas edge; give it the
    canvas pad like everything else, keeping every panel where it is in inches."""
    fig.canvas.draw()
    h = fig.get_figheight()
    tops = [lg for lg in fig.legends if lg.get_window_extent().y1 / fig.dpi > h - design.canvas_pad]
    if not tops:
        return
    need = max(design.canvas_pad - (h - lg.get_window_extent().y1 / fig.dpi) for lg in tops)
    boxes = {a: a.get_position() for a in fig.axes}
    centers = {lg: (lg.get_window_extent().x0 + lg.get_window_extent().x1) / 2 / fig.bbox.width for lg in tops}
    fig.set_layout_engine("none")
    fig.set_figheight(h + need)
    for a, b in boxes.items():
        a.set_position([b.x0, b.y0 * h / (h + need), b.width, b.height * h / (h + need)])
    for lg in tops:
        lg.set_loc("upper center")
        lg.set_bbox_to_anchor((centers[lg], 1 - design.canvas_pad / (h + need)), transform=fig.transFigure)


def _trim_round_panel_height(fig, panels):
    """A round panel (pie, polar) is always a circle, as large as its slot allows. Start tall so the
    figure width sets its size, then cut the empty height of its slot above and below it."""
    fig.set_figheight(3 * fig.get_figwidth())
    for _ in range(10):
        fig.canvas.draw()
        rows = {}
        for a in panels:
            row = round(a.get_position(original=True).transformed(a.figure.transSubfigure).y0)
            rows[row] = min(rows.get(row, np.inf), _panel_size(a, original=True)[1] - _panel_size(a)[1])
        empty = sum(rows.values())
        if empty < 0.005:
            break
        fig.set_figheight(fig.get_figheight() - empty)


def _colorbars_span_panels(fig):
    """Each colorbar runs the full height (or width) of the panels it belongs to, at the thickness the layout
    made room for. matplotlib caps its length at 20 times its thickness and constrained layout sets that cap
    again at every draw, so the layout is fixed first."""
    cbars = [a for a in fig.axes if getattr(a, "_colorbar", None) is not None]
    if not cbars:
        return
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for a in cbars:
        drawn, slot = a.get_position(), a.get_position(original=True)
        a.set_box_aspect(None)
        if a._colorbar.orientation == "vertical":
            a.set_position([drawn.x0, slot.y0, drawn.width, slot.height])
        else:
            a.set_position([slot.x0, drawn.y0, slot.width, drawn.height])


def _shape_of_data(panels):
    """The panel shape the data fixes, else None (the height step sets it): a heatmap's cells square; a square
    panel when it draws the y = x diagonal (ROC, reliability, predicted vs actual: DESIGN.md asks for it) or
    neither axis has ticks (an embedding)."""
    a = panels[0]
    if a.images:
        rows, cols = a.images[0].get_array().shape[:2]
        return rows / cols
    lines = [line for line in a.get_lines() if isinstance(line, AxLine) and line.get_xy2() is not None]
    diagonal = any(np.isclose(np.subtract(*line.get_xy2()), np.subtract(*line.get_xy1())) for line in lines)
    no_ticks = not len(a.get_xticks()) and not len(a.get_yticks())
    return 1.0 if diagonal or no_ticks else None


def _markers_on_top(fig):
    """Every line drawn with markers above every plain line and band, so no curve runs over a measured point."""
    for a in fig.axes:
        for line in a.get_lines():
            if line.get_marker() not in (None, "", "None", "none"):
                line.set_zorder(max(line.get_zorder(), 2.5))




def _edge_highlight_markers(fig):
    """Every highlight-size marker, in the legend too, gets a thin dark edge: its lightest shades stand apart from the paper."""
    for line in fig.findobj(Line2D):
        if np.isclose(line.get_markersize(), design.highlight_markersize):
            line.set_markeredgecolor(design.highlight_markeredgecolor)
            line.set_markeredgewidth(design.filled_markeredgewidth)


def finalize_figure(fig, ax, height="S", min_width=0.9, iters=10, tol=0.005, _aspect=None):
    """Change only the figure height so every panel is `height` tall: a step of DESIGN.md Panel height, "XS" to
    "XL" in inches at any width, or "SQ" as tall as wide. Where the data fixes the shape (`_shape_of_data`: heatmap cells, a y = x
    diagonal, an embedding), that shape instead.
    Numeric ticks follow the panel's size, one about every tick spacing.

    The panel is the data area inside the axes spines; tick labels, axis labels, titles,
    outside legends and colorbars are not measured. Call it last, after everything is drawn,
    on a figure that uses constrained layout. `ax` is an Axes, an array or list of Axes, or
    the dict from `subplot_mosaic`. Hidden axes and colorbar axes are skipped. If panels have
    different widths, the first panel is fitted. Returns the final panel ratios.

    Auto ticks can change with the height (e.g. 75, 80 → 77.5, 80.0), which changes the tick
    label width and so the panel width. If that keeps the ratio from converging, the major
    ticks are fixed at their current positions and the height is solved again.

    Tagged value labels and bar labels that would stick out of their panel are brought inside:
    the limits grow to make room, and tagged annotations at the left or right edge turn inward.
    Other text is never moved.

    Bar values side by side that overlap all turn upward (rotation 90), panel by panel.

    A panel legend made without `loc` is placed here: the first spot of `INSIDE_SPOTS` where it
    hides no data or text, else above its panel in one row.

    Last, it warns about what reads badly in print: a panel narrower than `min_width` (0.9 in; a
    two-factor grid passes 0.7, small multiples 0.45); text cut off at the
    figure edge, overlapping, sticking out of its panel or spilling out of its fill; text or a
    legend covering a curve or data points; a legend over text or with more than a tenth of its
    area over filled marks; a font other than Inter. Text on a white box and contour labels may
    sit over data.

    Round panels (pie charts, polar axes such as radar charts) are always circles, so `aspect`
    does not apply: the height is cut to the content once the circles are as large as the width
    allows.


    Never save with `bbox_inches="tight"`: it changes the final size and breaks the width and
    the ratio this function set.
    """
    axes = list(ax.values()) if isinstance(ax, dict) else list(np.ravel(ax))
    panels = [a for a in axes
              if a.get_visible() and a.axison and getattr(a, "_colorbar", None) is None]
    _write_plain_tick_numbers(fig)
    _size_text_tick_labels(fig)
    _markers_on_top(fig)
    _edge_highlight_markers(fig)
    _start_at_zero(panels)
    if isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine):
        _set_canvas_pad(fig)
    checks._warn_legends_wider_than_figure(fig)
    _lift_legends_wider_than_panel(fig, panels)
    if any(checks._is_round_panel(a) for a in panels):
        _trim_round_panel_height(fig, panels)
        _pad_outside_legends(fig)
        _unframe_outside_legends(fig)
        checks.run_checks(fig, panels, min_width)
        return _measure_panels(fig, panels)[0]
    for a in panels:
        if a.get_aspect() != "auto" or a.get_box_aspect() is not None:
            raise ValueError(
                f"panel {a.get_title()!r} has a fixed aspect, so the figure height cannot set "
                "its ratio; use aspect='auto' (e.g. imshow(..., aspect='auto')).")
    if not isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine):
        warnings.warn("finalize_figure expects layout='constrained'; ratios may be off.")
    fixed = _aspect or _shape_of_data(panels) or (1.0 if height == "SQ" else None)
    panel_height = None if fixed else design.panel_heights[height]
    fig.canvas.draw()
    aspect = fixed or panel_height / _panel_size(panels[0])[0]
    _margins_by_length(fig, panels, aspect)
    _ticks_by_length(fig, panels, aspect)

    ratios, widths = _fit_height_to_aspect(fig, panels, aspect, iters, tol)
    _shrink_crowded_texts(fig, panels)
    _turn_crowded_bar_labels(fig, panels)
    if _keep_value_labels_inside(fig, panels):
        ratios, widths = _fit_height_to_aspect(fig, panels, aspect, iters, tol)
    if _place_legends(fig, panels):
        ratios, widths = _fit_height_to_aspect(fig, panels, aspect, iters, tol)
    if abs(ratios[0] - aspect) / aspect >= tol:
        _freeze_ticks(fig.axes)  # colorbars too: their auto ticks change width with the height
        ratios, widths = _fit_height_to_aspect(fig, panels, aspect, iters, tol)
    if abs(ratios[0] - aspect) / aspect >= tol:
        warnings.warn(f"finalize_figure did not converge in {iters} iterations: ratio {ratios[0]:.3f}; "
                      "check for an inside legend or text wider than the panel.")
    if max(widths) - min(widths) > tol * max(widths):
        warnings.warn("panels have different widths, so not all can match the aspect; "
                      f"panel ratios: {[round(r, 3) for r in ratios]}")
    _blank_touching_corner_label(fig, panels)
    placed = _place_on_standard_box(fig, panels, aspect, panel_height)
    if placed:
        # ticks frozen while the height was searched fit a smaller panel: lay them out again at the final size,
        # then fix them, since growing limits for labels must not add ticks the layout has no room for
        _unfreeze_ticks(fig.axes)
        fig.canvas.draw()
        _freeze_ticks(panels)
        _turn_crowded_bar_labels(fig, panels)  # the panels may now be narrower than when bar values were first checked
        if _recheck_legends(fig, panels):
            _place_on_standard_box(fig, panels, aspect, panel_height)
        _keep_value_labels_inside(fig, panels)
        ratios = _measure_panels(fig, panels)[0]
    _colorbars_span_panels(fig)
    _fit_heatmap_values(fig, panels)
    _center_row_legends(fig, panels)
    _unframe_outside_legends(fig)
    # on the standard box, shapes follow its rules (a half with wider left labels has narrower panels)
    if not placed and max(ratios) - min(ratios) > tol * max(ratios):
        warnings.warn("panels have different shapes; one figure takes one aspect (no height_ratios or width_ratios)")
    checks.run_checks(fig, panels, min_width)
    return ratios
