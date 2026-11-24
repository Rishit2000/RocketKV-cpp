"""Readability checks finalize_figure runs last: text cut off or overlapping, labels and legends covering data,
series that merge in grayscale or share every channel, panels too narrow to read; and the tests on texts and panels that the
layout shares (does a text overlap another, is it a value label, is a panel round)."""
import math
import warnings

import matplotlib.path as mpath
import numpy as np
from matplotlib import rcParams
from matplotlib.collections import FillBetweenPolyCollection, LineCollection, PathCollection, PolyCollection
from matplotlib.container import BarContainer
from matplotlib.contour import ContourSet
from matplotlib.transforms import Bbox
from matplotlib.lines import AxLine
from matplotlib.patches import Rectangle, Wedge
from matplotlib.text import Annotation
from typing import NamedTuple

from . import design, palette


def _is_value_label(a, t):
    """A tagged value label or a bar label (`ax.bar_label`), the labels layout.finalize_figure keeps inside the panel."""
    if t.get_gid() == "value-label":
        return True
    ends = _bar_label_anchors(a)
    return (isinstance(t, Annotation) and t.xycoords == "data" and len(ends) > 0
            and bool(np.isclose(ends, t.xy).all(axis=1).any()))


def _bar_label_anchors(a):
    """The points `ax.bar_label` anchors its labels to: each bar's ends and middle, in data coordinates."""
    ends = []
    for p in a.patches:
        if isinstance(p, Rectangle):
            x, y, w, h = p.get_x(), p.get_y(), p.get_width(), p.get_height()
            ends += [(x + w / 2, y + h), (x + w / 2, y), (x + w, y + h / 2), (x, y + h / 2), (x + w / 2, y + h / 2)]
    return np.array(ends)


def _is_round_panel(a):
    return a.name == "polar" or any(isinstance(p, Wedge) for p in a.patches)


def _rotated_text_outline(t, box, renderer):
    """The text's outline as a closed path in display coordinates: its upright box turned by its
    rotation about the center of `box`, its upright bounding box once rotated."""
    angle = t.get_rotation()
    t.set_rotation(0)
    upright = t.get_window_extent(renderer)
    t.set_rotation(angle)
    along = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))]) * upright.width / 2
    across = np.array([-along[1], along[0]]) / np.hypot(*along) * upright.height / 2
    center = np.array([box.x0 + box.x1, box.y0 + box.y1]) / 2
    corners = [center - along - across, center + along - across, center + along + across, center - along + across]
    return mpath.Path(corners + corners[:1], closed=True)


def _texts_overlap(a, box_a, b, box_b, renderer):
    """Whether two texts overlap. A rotated text (e.g. a 45° tick label) is compared by its turned
    outline, since its upright bounding box also covers the empty corners around it."""
    a_rotated, b_rotated = a.get_rotation() % 180 != 0, b.get_rotation() % 180 != 0
    if getattr(a, "_in_cell", False) or getattr(b, "_in_cell", False):
        return box_a.overlaps(box_b)  # a heatmap value fitted to its cell: the cell edge keeps it apart from its neighbours
    if not (a_rotated or b_rotated):
        # side by side, upright texts need a clear word space between them (0.35 of the type size on each
        # side), or neighbours read as one at print size ("2561024", "Prefix tuningAdapter")
        gap = 0.35 * max(a.get_fontsize(), b.get_fontsize()) * a.figure.dpi / 72
        return Bbox.from_extents(box_a.x0 - gap, box_a.y0, box_a.x1 + gap, box_a.y1).overlaps(box_b)
    if a_rotated and b_rotated:
        outline_a, outline_b = _rotated_text_outline(a, box_a, renderer), _rotated_text_outline(b, box_b, renderer)
        return outline_a.intersects_path(outline_b, filled=True)
    rotated, box_rotated, other_box = (a, box_a, box_b) if a_rotated else (b, box_b, box_a)
    return _rotated_text_outline(rotated, box_rotated, renderer).intersects_bbox(other_box, filled=True)


class Problem(NamedTuple):
    """One thing that reads badly: where it is, what is wrong, and how to fix it."""
    where: str
    what: str
    fix: str

    def __str__(self):
        return f"{self.where}: {self.what}. Fix: {self.fix}"


class Context(NamedTuple):
    """What every check gets: the figure, its panels in reading order, a renderer, the narrowest panel allowed."""
    fig: object
    panels: list
    renderer: object
    min_width: float


CHECKS = []


def check(fn):
    """Register a check. A check takes a Context and returns or yields Problems; finalize_figure runs
    every registered check last and warns once per Problem. A new check is a new decorated function."""
    CHECKS.append(fn)
    return fn


def run_checks(fig, panels, min_width):
    fig.canvas.draw()
    ctx = Context(fig, panels, fig.canvas.get_renderer(), min_width)
    for fn in CHECKS:
        for problem in fn(ctx):
            warnings.warn(str(problem))


def _panel_name(a, panels):
    """A panel as the script's author finds it: its place in reading order, and its title if any."""
    title = f" {a.get_title()!r}" if a.get_title() else ""
    if a not in panels:
        return f"twin axis{title}"
    if len(panels) == 1:
        return f"the panel{title}"
    return f"panel {panels.index(a) + 1} of {len(panels)}{title}"


def _shown_texts(fig, panels):
    """Every visible text of the figure, each with where it is ("panel 2 of 4, x tick label '0.5'"), and the
    ids of the x and y tick labels among them."""
    figures = [fig]
    shown = [(t, "figure text") for f in figures for t in f.texts]
    x_ticks, y_ticks = set(), set()
    for f in figures:
        for legend in f.legends:
            shown += [(t, "figure legend") for t in legend.get_texts()]
    for a in fig.axes:
        if a.get_legend():
            shown += [(t, f"{_panel_name(a, panels)}, legend") for t in a.get_legend().get_texts()]
    for a in fig.axes:
        if not (a.get_visible() and a.axison):
            continue
        name = _panel_name(a, panels)
        shown += [(a.title, f"{name}, title"), (a.xaxis.label, f"{name}, x label"), (a.yaxis.label, f"{name}, y label")]
        shown += [(t, f"{name}, text") for t in a.texts]
        for axis, side in ((a.xaxis, "x"), (a.yaxis, "y")):
            lo, hi = sorted(axis.get_view_interval())
            # matplotlib keeps labels for ticks just outside the view, which are never drawn;
            # label2 is the right (top) side's: a twin axis draws its labels there
            labels = [label for loc, tick in zip(axis.get_majorticklocs(), axis.get_major_ticks())
                      if lo <= loc <= hi and axis.get_visible() for label in (tick.label1, tick.label2)]
            shown += [(t, f"{name}, {side} tick label") for t in labels]
            (x_ticks if side == "x" else y_ticks).update(map(id, labels))
    shown = [(t, f"{where} {t.get_text()!r}") for t, where in shown if t.get_visible() and t.get_text()]
    return shown, x_ticks, y_ticks


@check
def narrow_panels(ctx):
    for a in ctx.panels:
        if a.get_window_extent(ctx.renderer).width / ctx.fig.dpi < ctx.min_width:
            yield Problem(_panel_name(a, ctx.panels), f"narrower than {ctx.min_width} in",
                          "use one column fewer (multi-panel.md Grid step 3)")


def _long_axis_labels(ctx):
    """y labels taller than their panel; a figure's y label (`fig.supylabel`) is not one of them."""
    texts, _, _ = _shown_texts(ctx.fig, ctx.panels)
    span = lambda axes: max(a.get_window_extent(ctx.renderer).y1 for a in axes) - min(a.get_window_extent(ctx.renderer).y0 for a in axes)
    owner_of = {id(a.yaxis.label): [a] for a in ctx.fig.axes}
    return [(t, where) for t, where in texts
            if id(t) in owner_of and t.get_window_extent(ctx.renderer).height > span(owner_of[id(t)])]


@check
def long_axis_labels(ctx):
    for t, where in _long_axis_labels(ctx):
        yield Problem(where, "is longer than its panel is tall",
                      "shorten it; put the detail in the caption (DESIGN.md Collisions step 2)")


@check
def cut_off_text(ctx):
    texts, _, _ = _shown_texts(ctx.fig, ctx.panels)
    too_long = {id(t) for t, _ in _long_axis_labels(ctx)}
    for t, where in texts:
        box = t.get_window_extent(ctx.renderer)
        # text set flush to the edge (no canvas pad) lands within a pixel of it
        if id(t) not in too_long and (box.x0 < -1 or box.y0 < -1 or box.x1 > ctx.fig.bbox.x1 + 1 or box.y1 > ctx.fig.bbox.y1 + 1):
            yield Problem(where, "is cut off at the figure edge", "shorten it or move it inside (DESIGN.md Collisions)")


@check
def wrong_font(ctx):
    texts, _, _ = _shown_texts(ctx.fig, ctx.panels)
    for t, where in texts:
        if t.get_fontname() != rcParams["font.sans-serif"][0]:
            yield Problem(where, f"is in {t.get_fontname()}", "import paper before plotting")


@check
def overlapping_text(ctx):
    texts, x_ticks, y_ticks = _shown_texts(ctx.fig, ctx.panels)
    boxes = [t.get_window_extent(ctx.renderer) for t, _ in texts]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            (a, where_a), (b, where_b) = texts[i], texts[j]
            if not _texts_overlap(a, boxes[i], b, boxes[j], ctx.renderer):
                continue
            pair = {id(a), id(b)}
            if pair & x_ticks and pair & y_ticks:
                fix = "drop the y axis's first tick label (DESIGN.md Corner labels)"
            elif pair <= x_ticks or pair <= y_ticks:
                fix = "numbers: a wider tick step; names: turn them (DESIGN.md Typography, Collisions)"
            else:
                fix = "move one to its next placement, then make room (DESIGN.md Collisions)"
            yield Problem(f"{where_a} and {where_b}", "overlap", fix)


def _filled_shapes(a):
    """Bars and boxes, and bands, violins and stacked areas (PolyCollections), as paths in display coordinates."""
    fills = [p.get_path().transformed(p.get_transform()) for p in a.patches if p.get_visible() and p.get_fill()]
    return fills + [path.transformed(c.get_transform()) for c in a.collections
                    if type(c) in (PolyCollection, FillBetweenPolyCollection) and c.get_visible() for path in c.get_paths()]


def _data_marks(fig, a, panels):
    """The panel's curves (paths) and data points (arrays), in display coordinates, its twin axis's included."""
    # reference lines (axhline, axline) span the panel on purpose and are not data;
    # a twin axis (ax.twinx()) draws its curves over the same panel
    owners = [a] + [t for t in fig.axes if t is not a and t not in panels
                    and np.allclose(t.get_position().bounds, a.get_position().bounds)]
    data_lines = [line for o in owners for line in o.get_lines() if line.get_transform() == o.transData
                  and len(line.get_xdata()) and not isinstance(line, AxLine)]
    curves = [line.get_path().transformed(line.get_transform())
              for line in data_lines if line.get_linestyle() not in ("None", "none", "")]
    points = [line.get_transform().transform(line.get_xydata())
              for line in data_lines if line.get_marker() not in (None, "None", "none", "")]
    points += [c.get_offset_transform().transform(c.get_offsets()) for c in a.collections
               if isinstance(c, PathCollection) and len(c.get_offsets())]
    curves += [path.transformed(c.get_transform()) for c in a.collections
               if isinstance(c, LineCollection) and not isinstance(c, PolyCollection) for path in c.get_paths()]
    return curves, points


@check
def text_over_data(ctx):
    for a in ctx.panels:
        yield from _text_over_data(ctx.fig, a, ctx.panels, ctx.renderer)


def _text_over_data(fig, a, panels, renderer):
    """Text in a panel that sticks out of it, spills out of the fill it sits on, or covers a curve or data points."""
    curves, points = _data_marks(fig, a, panels)
    fills = _filled_shapes(a)
    name = _panel_name(a, panels)
    contour_labels = [t for c in a.collections if isinstance(c, ContourSet) for t in c.labelTexts]
    # text on a white box is meant to sit over data, and contour labels sit in gaps cut for them
    plain = [t for t in a.texts if t.get_visible() and t.get_text() and t.get_bbox_patch() is None
             and t not in contour_labels and t.get_gid() != "contour-label"]
    panel = a.get_window_extent(renderer)
    for t in plain:
        box = t.get_window_extent(renderer)
        if not _is_value_label(a, t) and not _is_round_panel(a) and isinstance(t, Annotation) and t.xycoords == "data" and not (
                panel.x0 - 1 <= box.x0 and box.x1 <= panel.x1 + 1 and panel.y0 - 1 <= box.y0 and box.y1 <= panel.y1 + 1):
            yield Problem(f"{name}, text {t.get_text()!r}", "sticks out of panel",
                          "move it to the other side of its point (the chart file's placements)")
        corners = [(box.x0, box.y0), (box.x1, box.y0), (box.x1, box.y1), (box.x0, box.y1)]
        for fill in fills:
            if fill.contains_point(((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)) and not fill.contains_points(corners).all():
                yield Problem(f"{name}, text {t.get_text()!r}", "spills out of its fill",
                              "drop every label of its kind, numbers to the caption (DESIGN.md Collisions step 4)")
                break
    # a label anchored on a data point may touch the curve through that point (its own series)
    covers = [(t.get_text(), t.get_window_extent(renderer),
               a.transData.transform(t.xy) if isinstance(t, Annotation) and t.xycoords == "data" else None) for t in plain]
    fix = "move it to its next placement (the chart file's placements, DESIGN.md Collisions step 1)"
    for text, box, anchor in covers:
        others = [c for c in curves if anchor is None or np.hypot(*(c.vertices - anchor).T).min() > 2]
        if any(curve.intersects_bbox(box, filled=False) for curve in others):
            yield Problem(f"{name}, text {text!r}", "covers a curve", fix)
        elif any(box.contains(x, y) for xy in points for x, y in xy):
            yield Problem(f"{name}, text {text!r}", "covers data points", fix)


@check
def legends_over_data(ctx):
    for a in ctx.panels:
        if a.get_legend():
            yield from _legend_problems(ctx.fig, a, ctx.panels, ctx.renderer)


def _legend_problems(fig, a, panels, renderer):
    """What the panel's legend hides: a curve or data points, text, filled marks or a bar's end; or
    an inside legend sticking out of its panel. Empty when the legend sits clear."""
    box, panel = a.get_legend().get_window_extent(renderer), a.get_window_extent(renderer)
    where = f"{_panel_name(a, panels)}, legend"
    move = "drop its loc so finalize_figure places it, or put it above the panel (DESIGN.md Legend)"
    curves, points = _data_marks(fig, a, panels)
    problems = []
    if any(curve.intersects_bbox(box, filled=False) for curve in curves):
        problems.append(Problem(where, "the legend covers a curve", move))
    elif any(box.contains(x, y) for xy in points for x, y in xy):
        problems.append(Problem(where, "the legend covers data points", move))
    # an inside legend (one that overlaps its panel) must fit in it; a wider one goes above
    if box.overlaps(panel) and (box.x0 < panel.x0 - 1 or box.x1 > panel.x1 + 1 or box.y1 > panel.y1 + 1):
        problems.append(Problem(where, "the legend sticks out of panel",
                                "put it above the panel, above its row in a grid (DESIGN.md Legend)"))
    if any(t.get_visible() and t.get_text() and box.overlaps(t.get_window_extent(renderer)) for t in a.texts):
        problems.append(Problem(where, "the legend covers text", move))
    # the filled area under the legend, not the shapes' bounding boxes: a histogram is one wide polygon
    covered = sum(_polygon_area(poly) for shape in _filled_shapes(a) for poly in shape.clip_to_bbox(box).to_polygons())
    if covered > box.width * box.height / 10:
        problems.append(Problem(where, "the legend covers filled marks (bars, boxes, bands, areas)", move))
    # a bar's value is its end: covering even a sliver of it hides the number the bar shows
    if any(_bar_value_edge(bar, c.orientation, renderer).overlaps(box)
           for c in a.containers if isinstance(c, BarContainer) for bar in c.patches):
        problems.append(Problem(where, "the legend covers the end of a bar", move))
    return problems


def _bar_value_edge(bar, orientation, renderer):
    """The edge of a bar at its value, as a thin box: the top (bottom when negative) of a vertical
    bar, the right (left) end of a horizontal one."""
    box, value = bar.get_window_extent(renderer), (bar.get_height() if orientation == "vertical" else bar.get_width())
    if orientation == "vertical":
        y = box.y1 if value >= 0 else box.y0
        return Bbox([[box.x0, y - 0.5], [box.x1, y + 0.5]])
    x = box.x1 if value >= 0 else box.x0
    return Bbox([[x - 0.5, box.y0], [x + 0.5, box.y1]])


def _polygon_area(xy):
    x, y = xy[:, 0], xy[:, 1]
    return abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2


def _warn_legends_wider_than_figure(fig):
    fig.draw_without_rendering()
    room = fig.bbox.width - 2 * design.canvas_pad * fig.dpi
    for legend in fig.legends + [a.get_legend() for a in fig.axes if a.get_legend()]:
        if legend.get_window_extent().width > room:
            n, ncol = len(legend.get_texts()), legend._ncols
            fewer = [math.ceil(n / k) for k in range(2, n + 1) if math.ceil(n / k) < ncol]
            fix = f"pass ncol={fewer[0]}" if fewer else "shorten its entries"
            warnings.warn(str(Problem(f"legend of {n} entries in {ncol} columns", "is wider than the figure",
                                      f"{fix} (DESIGN.md Legend)")))


@check
def series_lookalikes(ctx):
    for a in ctx.panels:
        yield from _series_lookalikes(a, ctx.panels)


def _series_marks(a):
    """Each labelled series' look: its line styles, markers and colors. A series may be drawn as many artists
    under one label (a line per checkpoint, a dot per run); their colors may vary (lightness for compute)."""
    series = {}

    def add(label, kind, colors, style=None, marker=None):
        if not label or label.startswith("_"):
            return
        s = series.setdefault(label, {"shape": set(), "colors": []})
        s["shape"].add((kind, style, marker))
        s["colors"] += [palette._over_white(c) for c in colors]

    for line in a.get_lines():
        style, marker = str(line.get_linestyle()), str(line.get_marker())
        add(line.get_label(), "line", [line.get_color()], None if style in ("None", "none", "") else style,
            None if marker in ("None", "none", "") else marker)
    for c in a.containers:
        if isinstance(c, BarContainer) and c.patches:
            add(str(c.get_label()), "bar", [p.get_facecolor() for p in c.patches])
    # histograms (step outlines or filled shapes), translucent fills as seen over white. Parts of one whole
    # (wedges, stacked bands) are not series: their place and the seams between them tell them apart
    for p in a.patches:
        if not isinstance(p, (Rectangle, Wedge)):
            add(p.get_label(), "fill", [p.get_facecolor() if p.get_fill() else p.get_edgecolor()])
    for c in a.collections:
        if isinstance(c, PathCollection) and len(c.get_facecolor()):
            shape = tuple(np.round(c.get_paths()[0].vertices, 2).ravel()) if c.get_paths() else None
            add(c.get_label(), "scatter", c.get_facecolor(), marker=shape)
    return series


def _series_lookalikes(a, panels):
    """Series told apart by color alone (same line styles and markers, or bars) must differ in lightness, or they
    merge when printed in black and white. A series whose own lightness varies (it shows a value, as compute)
    leaves lightness no room to tell series apart: those series need a line style or marker of their own."""
    series = list(_series_marks(a).items())
    for i in range(len(series)):
        for j in range(i + 1, len(series)):
            (n1, s1), (n2, s2) = series[i], series[j]
            if s1["shape"] != s2["shape"]:
                continue
            where = f"{_panel_name(a, panels)}, series {n1!r} and {n2!r}"
            spent = [n for n, s in ((n1, s1), (n2, s2)) if np.ptp([palette._lightness(c) for c in s["colors"]]) >= 10]
            if spent:
                yield Problem(where, f"share line style and marker, and {spent[0]!r} already uses lightness for a value: "
                                     "color alone cannot tell them apart",
                              "give each variable its own channel: a line style or marker per series (DESIGN.md Colors)")
                continue
            l1, l2 = palette._lightness(s1["colors"][0]), palette._lightness(s2["colors"][0])
            if abs(l1 - l2) < 10:
                yield Problem(where, f"look alike in grayscale (L* {l1:.0f} vs {l2:.0f})",
                              "use make_shades(colors[\"neutral\"], n), or tell them apart by line style or marker (DESIGN.md Colors)")
