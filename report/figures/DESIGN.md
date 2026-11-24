---
version: alpha
name: Paper Figure
description: Print-first figures for machine learning papers, read at the size they are printed, where the reader gets one message at a glance.
colors:
  # Seaborn "deep" palette, in its own order; its blue is ours; methods take one hue each (warm and cool mixed)
  series-1: "#4C72B0"  # blue
  series-2: "#F4A261"  # orange, lighter than seaborn's #DD8452 so it stays apart from green (L* 74 vs 62) in grayscale
  series-3: "#55A868"  # green
  series-4: "#C44E52"  # red
  series-5: "#8172B3"  # purple
  series-6: "#937860"  # brown
  series-7: "#DA8BC3"  # pink
  series-8: "#8C8C8C"  # grey
  series-9: "#CCB974"  # olive
  ours: "{colors.series-1}"  # blue
  neutral: "#A9A9A9"  # grey
  reference: "{colors.series-2}"  # orange: Full KV, also when drawn as a reference line
  ink: "#000000"
  paper: "#FFFFFF"  # the page: marker edges, legend fill
typography:
  tick-label:
    fontFamily: Inter
    fontSize: 5pt
  tick-label-text:
    fontFamily: Inter
    fontSize: 7pt
  label:
    fontFamily: Inter
    fontSize: 7pt
  legend:
    fontFamily: Inter
    fontSize: 7pt
  value-label:
    fontFamily: Inter
    fontSize: 7pt
    fontWeight: 400
  title:
    fontFamily: Inter
    fontSize: 7pt
    fontWeight: 400
  value-label-small:
    fontFamily: Inter
    fontSize: 5pt
spacing:
  layout-pad: 0in
  canvas-pad: 0in
  panel-gap: 5%
  legend-gap: 0.25in
  title-pad: 4pt
  label-pad: 1pt
  tick-pad: 1pt
  data-margin: 0.125in
components:
  tick:
    size: 2pt
    minor-size: 1pt
    spacing: 0.25in
  panel:
    height-xs: 1in
    height-s: 1.5in
    height-m: 2in
    height-l: 2.5in
    height-xl: 3in
  grid:
    color: "{colors.ink}"
    opacity: 0.1
  line:
    width: 1pt  # the one line weight; every other line is a multiple of it, by role (Shapes)
    data: 1
    compared: 1
    outline: 1
    context: 0.5
  marker:
    size: 3  # × line.width
    highlight-size: 6  # × line.width: points of interest, and points read on a color scale
    edge: "{colors.paper}"
    highlight-edge: "{colors.ink}"  # dark, so the lightest shades stand apart from the paper
  box:
    color: "{colors.ink}"  # box plot outline, whiskers, caps, median and outliers
  frame:
    lighter: 0  # legend frame: `neutral` this many stops lighter
  legend:
    fill: "{colors.paper}"
    handle-length: 1.5  # this and below in legend font sizes
    handle-text-pad: 0.5
    column-spacing: 1
    label-spacing: 0.25
    border-pad: 0.3
    border-axes-pad: 0.25
  series:
    order: [series-1, series-2, series-4, series-5, series-6, series-7, series-8, series-9]  # equal terms: Tableau order, ours skipped
  shades:
    stop: 12  # L* one lightness stop moves (shift_lightness)
    step: 18  # L* between neighbouring make_shades shades: apart in grayscale
    lightest: 90  # L* range make_shades stays in; a color's darkest shade is the color itself
    darkest: 15
  part-colors:
    order: [series-1, series-5, series-4, series-6, series-2, series-7, series-9, series-3, series-8]  # make_part_colors: the first four ≥10 L* apart, then neighbours
  colormap:
    # as Tableau's default continuous palettes: "Blue", and "Orange-Blue Diverging"
    sequential: "{colors.series-1}"  # low values light, high dark: cmap="paper" (the default)
    negative: "{colors.series-2}"  # signed values below zero: cmap="paper-diverging"
    positive: "{colors.series-1}"  # above zero
    lightest: 95  # L* at the low end, and at zero of the diverging map
    darkest: 20  # L* at the high end, and at both ends of the diverging map
  colorbar:
    width: 0.1in
    gap: 0.05in
---

# Paper Figure

Design system for paper figures. Front matter holds the tokens: every color, colormap, font, text size, line weight, marker size and spacing. `import paper` loads them into matplotlib's style, so a token edit, or another DESIGN.md in this one's place, changes every figure; scripts take colors from `paper.colors` and sizes from `rcParams` or `paper`, never write them. Chart type file conflicts with a rule here: chart file wins. In a project, the copy in the figures folder is the design; edit it there for a paper-specific look.

## Overview

Figure is read in print, at its set width, for a few seconds. Goal: one message at a glance, exact numbers the paper claims readable. Case not covered: apply these principles.

1. Print first: sizes are printed sizes, from tokens. Never scale text down; draw at printed width.
2. One message, one emphasis: name the emphasis first (what the eye must land on: a series, a gap, a point, a change along x), then set color and panel height from it. Color only for what the takeaway is about (ours, or series it names). Everything compared or context (baselines, ablations, fronts, reference lines, grids): `neutral` grey or `ink`. Several colors only when takeaway compares series on equal terms.
3. Alike looks alike: one treatment for every item of a kind, or none. Kind = items with one role: bars of one series, levels of one contour plot, shares of one band, models the takeaway cites (and, separately, those it does not). Series keeps color, marker, name in every panel and figure. Kind labelled throughout or not at all; each label may sit where it fits (inside or outside its mark). Style that tells two series apart: used the same way throughout.
4. Nothing hides data: text, legends, marks never cover data or each other, never leave panel or fill. Context (grid, reference lines) behind data, never decides placement.
5. Exact and verifiable: number the takeaway cites written on plot in Value labels table format; in caption if it cannot fit. Every number computed from data in the script. Drawing shows all given data: nothing hidden by limits, dropped, or smoothed away when a reported result.
6. Tested, not judged: collision = what `finalize_figure` reports; resolve by Collisions (Layout).

## Colors

`colors` tokens: Tableau 10, Tableau's default palette, in its own order; its red is `ours`, its grey `neutral`. Its colors sit close in lightness (L* 50 to 82): series told apart by color alone in black and white need the lightness rule below, else a line style or marker. `paper.colors` holds them by name:

```python
colors = paper.colors
colors["series-1"], colors["ours"]
```

First row that fits:

| Case | Color |
|---|---|
| "Ours" among baselines or ablations: authors' method, else series the takeaway is about (RL-trained model vs. its base) | `ours`; every compared series `neutral` |
| Ours vs. its own ablation (with vs. without a component; takeaway is the component) | full method `ours`, ablated variants `neutral`, as compared series |
| Variants of ours together vs. other methods | all `ours`, apart by marker `"o"`, `"s"`, `"^"` on solid lines (dashes belong to grey series). Curves without markers (ROC/PR staircases, smoothed logs): by line style, full method solid, others `"--"`, `":"`. Full method first in legend |
| Parts of one whole (pie wedges, stacked area bands, stacked bar segments) | each part own color, `paper.make_part_colors(n)`: the palette's colors unchanged, in stacking order by the `part-colors` token (blue, green, teal, yellow first: any four stay ≥10 L* apart in grayscale; Tableau 10 has no fifth, so parts are told apart by their place, the seams in `paper` between them and their names on the parts; the grayscale check skips them). Not compared series: all nine colors free |
| Some series the takeaway names, among others it does not | named ones in equal-terms order below (`series-1`, `series-4`, …); rest `make_shades(colors["neutral"], n)` |
| Several series on equal terms, no "ours" (models in a survey, correct vs. incorrect answers, error types) | `series-1`, `series-2`, `series-4`, `series-5`, … in Tableau order (`series.order` token), as `paper.series_colors[0]`, `[1]`, … in scripts. Their lightness is close: on lines, also a marker of their own; on bars, check the grayscale warning. Skip `ours` color: means ours in every figure of the paper |
| More than nine unordered classes (embedding) | `tab20` (≤20), plus own marker per class: `"o"`, `"s"`, `"^"`, `"D"`, `"v"`, `"<"`, `">"`, `"P"`, `"X"`, `"*"` |
| Groups with sizes or sweeps inside (model sizes per family, network depths, batch sizes) | one color per group (ours: `ours`; a baseline family: `neutral`, dashed); within a group `paper.make_shades(group_color, n)`: one hue, light to dark, `shades.step` L* apart and the darkest the color itself, the largest setting darkest |
| Single series, or one quantity over categories (groups in one box plot) | `series-1`; `ours` when the series is ours, here or in another figure of the paper |
| Reference lines | `reference` (yellow), dashed (white over filled plots, Shapes) |
| Text and tick labels | `ink` |
| Panel box and tick marks | `ink` (black) at the context weight (×0.5): thin, so the data stays darker in weight (set by style) |
| Continuous values | `colormap` tokens, built from the palette: `"paper"` (the default, as Tableau's "Blue": Tableau blue from light to dark, low values light, lightness evenly spread); signed values `"paper-diverging"` (as Tableau's "Orange-Blue Diverging": orange below zero, near-white at zero, blue above) (`heatmap.md`) |

- Grey series: assigned in Order rule's order (Layout). Lines: line style `"--"`, `":"`, `"-."` (solid is ours' alone), dashed/dotted at `linewidth=paper.dashed_linewidth`. Fills (bars, boxes, violins, areas): shade, `make_shades(colors["neutral"], n)` for n grey series, light to dark, skipping `ours` lightness. One grey series: `neutral`.
- Many grey series: >4 lines: add markers, one shape per series (`line-chart.md`). >6: draw series the takeaway names; group rest as one grey series, "Other baselines".
- Shared color (ours and its variants, greys): differ by line style or shade as above.
- One channel per variable: each variable the marks differ by (method, split, compute, seed) gets one channel of its own: hue, lightness, line style, marker, or panel. Lightness that shows a value (compute, a sweep in `make_shades` shades) is spent: series then differ by line style or marker, never by color alone, since in grayscale color is only lightness. Ours keeps its hue; compared series keep their line styles (`"--"`, `":"`, `"-."`) as in every other figure of the paper. No channel left: one panel per value of a variable. `finalize_figure` warns when two labelled series share line styles and markers while one's lightness varies.
- On white: grey lines and markers `neutral` at full opacity; light greys for fills only. Series text: `shift_lightness(series_color, -1)` (grey series: `shift_lightness(colors["neutral"], -1)`). Text on fill: Typography.
- Grayscale / color blindness: series told apart by color alone also differ in lightness by ≥10 L*: ours (`series-3`, L* 56) vs. lighter `neutral` (L* 73); greys skip its lightness. Lines also differ by style or marker, bars by shade. `finalize_figure` warns when two such series look alike in grayscale (it reads each mark's `label`).
- Emphasis: user names a series to emphasize: color it as ours, all others as compared.
- Other palettes: paper-wide (lab, brand): change `colors` tokens in design copy. One-off for one figure ("match the original"): dict in that figure's script:

```python
PALETTE = {"ours": "#5F16A7", "base": "#63C9D2"}
```

## Typography

One family, `Inter` (bundled), all text and math, regular weight only (nothing bold). Text `ink` unless it belongs to one series.

| Token | Size | Used for |
|---|---|---|
| `tick-label` | 5 pt | numeric tick labels (`0.5`, `1K`, `1e−4`, `50%`) |
| `tick-label-text` | 7 pt | text tick labels (method, benchmark, category names); set by `finalize_figure` |
| `value-label` | 7 pt | all text inside a chart: values, bar labels, heatmap cells, contour levels, point names, pie labels; default for `ax.text`, `ax.annotate`, `ax.bar_label`, `ax.clabel`, so pass no `fontsize` |
| `value-label-small` | 5 pt | text inside a chart that does not fit at 7 pt: crowded or spilling out of its fill; set by `finalize_figure` |
| `label` | 7 pt | axis labels, subcaptions |
| `legend` | 7 pt | legend entries |
| `title` | 7 pt | panel titles, row and column headers |

- Values given as text in the data (`"1e-5"`, `"3e-4"`): written as given, keeping the scientific notation; only the hyphen becomes a true minus (`1e−5`).
- Numeric tick labels: horizontal, plain, shortest form (`0, 0.5, 1`, `1024`); never rotated, never `2³` or `1e+06`. `finalize_figure` does this for matplotlib's default numeric ticks (`0.0, 0.5, 1.0` → `0, 0.5, 1`). Formatter, keeps true minus:

  ```python
  ax.yaxis.set_major_formatter(lambda v, _: f"{v:.10g}".replace("-", "−"))
  ```

- Axis labels (and colorbar labels): quantity, then its unit in parentheses: "Success rate (%)", "Length (tokens)",
  "Compute (GFLOPs)", "Training tokens (B)". Never drop the unit, not for room either: shorten the quantity instead.
  No parentheses only for a quantity without a unit (a 0–1 accuracy or rate, loss, a count of the thing named).
  A row header standing in for the y label (`multi-panel.md`) carries the unit too.
  - Linear axis, values <0.001: scale in axis label ("Learning rate (×10⁻⁴)"), ticks 0, 1, 2, 3.
  - ≥1000: SI suffixes, one unit per axis, when that writes the ticks shorter (`0, 0.5K, 1K, 1.5K, 2K`, `10K`, `1M`; `200 … 1000` and `1024` stay); `finalize_figure` writes matplotlib's default ticks this way. Or the unit scaled in the axis label ("Training step (K)").
  - Log axes: powers of ten plain from 0.001 to 1000; above, SI suffix (10K, 1M); below, `1e−4` (true minus). `finalize_figure` writes them so.
  - Shares: percent, `0%, 50%, 100%`, as values are written (Value labels).
- Tick choice: x values the chart file names (stage switches), else evenly spaced nameable numbers. `finalize_figure` puts a numeric tick about every `tick.spacing` (0.25 in) along the axis, at a step of 1, 2, 2.5 or 5 × 10ⁿ, so a taller panel shows more; on x, the step widens until the widest label fits. Every numeric tick is labelled; never skip labels. Log axis: powers of the base. Panels sharing an axis share its ticks.

  ```python
  ax.set_xticks(ticks, labels=[str(t) if i % m == 0 else "" for i, t in enumerate(ticks)])
  ```

- Corner labels: first tick labels of both axes overlap at corner: `finalize_figure` drops y axis's first label itself.
- Text tick labels (non-numeric) reported overlapping:
  - Unordered categories, one value each (methods, models, groups in one bar, error bar or box chart): swap axes, names horizontal down y (horizontal bars, forest plot, horizontal boxes), top to bottom per Order.
  - Names down y axis (heatmap rows, horizontal bars): a taller step (Panel height), then Collisions step 3.
  - Everything keeping the x axis (named bins, categories of grouped series, benchmarks shared across panels): 45° (`ha="left"` on top axis), then 90°. Still overlapping at 90°: Collisions steps 3 and 4 (colorbar beside panel first moves below it, giving its width to the panel).

    ```python
    ax.set_xticks(x, labels=names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xticks(x, labels=names, rotation=90, ha="center")
    ```

- Math: mathtext in Inter (`r"$\mathrm{PPL}_{\mathrm{Base}}$"`). Symbols Inter lacks (∝, ≈): mathtext too (`$\propto$`).
- Text beside a mark, on its line (a value right of an error bar, a name left of a point): `va="center_baseline"`, so the digits sit on the mark's center line; `"center"` centres the text box, descender room included, and lifts them.
- Text on fill (bar segment, wedge, heatmap cell, area band): black or white by the fill's luminance:

  ```python
  text_color = colors["paper"] if np.dot(to_rgb(fill), [0.299, 0.587, 0.114]) < 0.5 else colors["ink"]  # from matplotlib.colors import to_rgb
  ```

- Text over mixed content (note over bars, gridlines, reference line; no single fill sets contrast): white box. Contour labels: none, they sit in the gap cut for them.

  ```python
  bbox=dict(facecolor=colors["paper"], edgecolor="none", pad=1)
  ```

## Layout

### Width

| Venue | Single column | 1.5 column | Double column | Column gap |
|---|---|---|---|---|
| ICML | 3.25 in | 5.0625 in | 6.75 in | 0.25 in |
| CVPR | 3.25 in | 5.15625 in | 6.875 in | 0.375 in |
| ACL (ARR) | 3 in | 4.6875 in | 6.25 in | 0.25 in |

| Single-column venue | Half text width | 1.5 column | Full text width |
|---|---|---|---|
| ICLR, NeurIPS | 2.75 in | 4.125 in | 5.5 in |
| This report (US Letter, 1 in margins) | 3.25 in | 4.875 in | 6.5 in |

Column span not given: ask the user before drawing. Recommend the first row that applies:

| Case | Width |
|---|---|
| User or task gives a figure width (one column of the paper, inches; venue text width alone is not one) | that width |
| User asks for 1.5 column (`figure*` in a two-column venue), or panels at single column still fail `min_width` after grid's step 3 (`multi-panel.md`) | 1.5 column |
| Single-column venue, several panels | half text width if they fit in one row there (every panel ≥ `min_width`), else full text width; then rows as `multi-panel.md` adds them |
| Single-column venue, one panel | half text width; full only when Collisions step 3 calls for it |
| Two-column venue | single column if the panels fit in one row there, else double column; then rows as `multi-panel.md` adds them |

Venue: the one the user names; default ICLR.

Venue not in the tables (a journal, a workshop, another conference): read its widths from its LaTeX template or author guide (`\textwidth`, `\columnwidth`, `\columnsep`; ask the user when neither is at hand), add its row to the table above in the project's DESIGN.md, then use it like the others. Two-column venue: 1.5 column = (single + double) / 2 + column gap / 4. Single-column venue: half text width = text width / 2, 1.5 column = 0.75 × text width.

### Panels

One panel: no title (caption names the figure). One figure carries one takeaway: several takeaways or chart types are separate figures. Several panels or second y axis (`ax.twinx()`): read `multi-panel.md` in skill's `references/` for grid, shared axes, axis labels, panel names.

### Panel height

A panel's height is fixed in inches, the `panel` tokens, at any width: one step gives the same height, and so
the same number of y ticks (one per tick spacing), in a single-column figure and a full-width one. `"SQ"` makes
the panel as tall as it is wide.
`finalize_figure(fig, ax, height=...)`, default `"S"`:

| Step | Height | For |
|---|---|---|
| `"XS"` | 1 in | a short chart with few values; a wide panel across the full text width |
| `"S"` | 1.5 in | default: one panel, a row of panels |
| `"M"` | 2 in | the emphasis is a difference in y |
| `"L"`, `"XL"` | 2.5, 3 in | many categories down y (horizontal bars, long legends inside): the smallest step where they fit |
| `"SQ"` | panel width | square: grids and small multiples compared by shape |

Pass any step but S with a one-line comment giving the emphasis it serves:

```python
# emphasis: the gap between Ours and the best baseline; M gives it height
finalize_figure(fig, ax, height="M")
```

Where the data fixes the shape, `finalize_figure` sets it and ignores `height`; nothing to pass:

| Data | Shape |
|---|---|
| heatmap | cells square |
| a y = x diagonal, `ax.axline((0, 0), (1, 1))`: ROC curve, reliability diagram, predicted vs actual | square |
| embedding (t-SNE, UMAP), no ticks on either axis | square |

Pie and radar: circles; `finalize_figure` sizes them to width. Height follows the emphasis, never room for labels.

### Collisions

`finalize_figure` reports text overlapping, covering data, cut off at figure edge, out of its panel, or spilling out of its fill: first step that clears, render after each:

1. Move it: that item to next placement its chart file lists (other side of point, below value, next legend spot). Placements may differ per item.
2. Make room inside panel: share axes (`multi-panel.md`); Typography's text-tick rule (swap axes, then 45°, then 90°); shorten text (axis label longer than panel: short name of quantity, unit kept, detail in caption).
3. Give it width, if crowding is labels along an axis (tick labels, bar values, side-by-side category names, heatmap rows or columns) and no figure width given: full width, or own aspect for fixed-aspect panel (heatmap rows grow with it). After widening, retry earlier placements (horizontal before turned). Crowding inside data area (value labels on curves, scatter point names, heatmap cell values, contour labels): step 4 instead.
4. Write less, alike: label every m-th contour level, keep every line; numeric ticks: a wider step instead, never skipped labels; or drop a kind whole (all point names of non-cited models, all bar values, all cell values, all shares of a band), cited numbers to caption.

Never: shrink text, change a token, or change height for room. One exception: text inside a chart and a heatmap's row and column names, which `finalize_figure` sets to 5 pt when 7 pt does not fit; not fitting at 5 pt either: Collisions step 4.

### Height

Set by `finalize_figure`. Create figure at target width, height = width (too short a start can collapse a layout with outside legend). Draw everything (labels, legend, colorbar), call `finalize_figure` last, save with plain `fig.savefig`:

```python
fig, axes = plt.subplots(2, 1, figsize=(3.25, 2), sharex=True)
...  # plot, labels, legend, colorbar
finalize_figure(fig, axes)
fig.savefig("outputs/{id}.pdf")
```

- Panels: `ax` = one Axes, array, list, or `subplot_mosaic` dict; hidden cells and colorbars skipped.
- Standard box: `finalize_figure` sets standard margins, gaps, size; figures of one width and column count line up. Never set margins, `subplots_adjust`, `height_ratios`, `width_ratios`, sub-figures. Right of panels: only colorbar or right y axis. Names that would go there (band names, line-end names): legend above panel.
- Labels inside: `finalize_figure` keeps bar labels and tagged value labels inside panel by growing limits; other text never moved.
- Errors: `ValueError` (fixed aspect): images with `aspect="auto"`. "Different widths" warning: give all panels same width.

  ```python
  ax.imshow(image, aspect="auto")
  ```

### Axes

- Limits: matplotlib defaults plus `data-margin` token (0.125 in at print size, whatever the panel's size) each end, edge marks clear the box. Except axis whose marks together fill a bounded whole: fixed to it (stacked shares and areas 0–1, ROC and reliability unit square, radar scale, field on fixed range such as loss landscape −1 to 1). Metrics (accuracy, histogram per-bin fractions): defaults. Never-negative quantity: starts at ≥0; `finalize_figure` raises negative lower limit to 0 when all data >0, never grows one below 0.
- Order: categories and series with no natural order (methods, models, ablation variants): ascending by value the takeaway compares, left to right (top to bottom when horizontal); ours, usually best, ends right. Natural order stays: sizes, time, bins, swept values, any order task or data gives explicitly (benchmark list, list of opponents). Legends of side-by-side series: same order.
- Category axes: tick marks hidden on that axis, every panel (`axis="y"` when horizontal):

  ```python
  ax.tick_params(axis="x", length=0)
  ```

- Log axes: size, count or rate with positive values spanning >100× (max / min > 100, over every panel showing that quantity: one scale per quantity in a figure): log. Positions along a process (training steps, epochs, time): linear. Set log scale before drawing on the axis; name it in caption. Log₁₀ axis spanning ≥1 power of ten: unlabelled minor ticks at 2–9 × each power; with gridlines, gridlines follow them.

  ```python
  ax.set_xscale("log")
  ax.xaxis.set_minor_formatter(NullFormatter())
  ax.grid(True, which="both")
  ```

- Colorbars (heatmaps, contour plots, colored lines): always (except a heatmap with every cell's value written, `heatmap.md`), labelled with quantity, on right. Below panel when `finalize_figure` reports panel narrower than `min_width` with it on right. A colorbar spans its panels: as tall as they are, or below, as wide as all of them.
  - Several panels on one scale (`vmin`, `vmax` shared): one colorbar for all. Scales differ: one beside (or under) each panel, each labelled; `finalize_figure` makes room for it before the next panel.

  ```python
  fig.colorbar(mappable, ax=ax, label="Validation loss")
  fig.colorbar(mappable, ax=ax, label="Validation loss", location="bottom")
  fig.colorbar(mappable, ax=axes, label="Validation loss")  # one scale across panels
  ```

## Shapes

Marks every chart type is built from. Every line is a multiple of the one `line.width` token (1 pt), by role (the `line` tokens below it);
change the tokens and every weight follows. Matplotlib applies them; code reads from `rcParams` or `paper`, never writes the numbers.

| Role | Weight | Lines |
|---|---|---|
| Data | ×1 | data lines |
| Compared | ×1 | dashed and dotted lines (grey series, reference lines, fits, fronts): `linewidth=paper.dashed_linewidth` |
| Outline | ×1 | the edge of a shape that holds data: box plot boxes, violin bodies (black) |
| Context | ×0.5 | lines inside or beside such shapes: box plot medians, means, whiskers and caps; a violin's inner range line; error bars; legend frame (`frame` color); panel box and tick marks (black), grid (`ink` at 10%), seams between fills (white), edges of touching one-color bars, contour lines, colorbar outline; edges of every marker: white around series markers and box plot outliers (style default), black around highlight-size markers (set by `finalize_figure`) and embedding points (`markeredgewidth=paper.filled_markeredgewidth`) |

| Mark | Look | In code |
|---|---|---|
| Panel box | full box, all four sides, `ink` (black), ×0.5; tick marks the same, outside, 2 pt long (minor 1 pt); tick labels `ink` | set by style |
| Grid | `ink` at 10% opacity, behind data, both directions unless chart file says otherwise | `rcParams["grid.linewidth"]`, `rcParams["grid.alpha"]` |
| Markers | filled in series color, thin edge in `paper` (`marker.edge`), `marker.size` (3) × `line.width`; above every line and band (set by `finalize_figure`) | `rcParams["lines.markersize"]` |
| Points of interest (a minimum, the trained solution) | filled star, `marker.highlight-size` (6) × `line.width`, thin `marker.highlight-edge` edge (set by `finalize_figure` for every marker of this size, legend too) | `"*", markersize=paper.highlight_markersize` |
| Seams between fills (stacked segments, wedges, area bands) | `paper` | `edgecolor=colors["paper"], linewidth=rcParams["grid.linewidth"]` |
| Outline of touching one-color bars, each its own value (reliability bins) | darker shade of fill, never white | `edgecolor=shift_lightness(color, -1), linewidth=rcParams["grid.linewidth"]` |
| Bars and histogram bins | no edge | `linewidth=0` |
| Colorbar | 0.1 in wide, 0.05 in from panels, labels in standard margin beside it | placed by `finalize_figure` |

- Opacity: fills at full series color, except bands (`alpha=0.2`), fills that show a quantity by their area (stacked areas, area under a curve: `alpha=0.875`), scatter clouds (`alpha=0.5`), light polygon fills (radar, `alpha=0.1`), box plots and violins: `shift_lightness(color, 2)` so black median and points on them stay readable; grey series too (their `make_shades(colors["neutral"], n)` shade, shifted +2). Ours keeps full color there (lightened colors all print pale grey; ours must stay apart).
- Reference lines (previous SOTA, baseline score, threshold, diagonal): look like no data series: `reference` (yellow), dashed, ×1 (`paper.dashed_linewidth`), behind the data; the hue keeps them apart from dashed grey series. Its name and value written on the plot (a direct label, "Prev. SOTA 44.4%"): `shift_lightness(colors["reference"], -1)`, as series text. Across filled plot (stacked area, heatmap, contour gradient; grey sinks into fills): white, dashed. Second and later in one panel: dotted. Line with own value (baseline score, threshold): legend entry if panel has a legend, else caption's reading guide names it with its value (Legend: a single entry). Line whose meaning the axes give (y = x, chance, zero): no entry; caption's reading guide names it.

  ```python
  ax.axhline(y, color=colors["reference"], linestyle="--", linewidth=paper.dashed_linewidth, zorder=1)
  ax.axhline(y2, color=colors["reference"], linestyle=":", linewidth=paper.dashed_linewidth, zorder=1)
  ```

## Components

### Chart types

Each chart type has its own file next to this one. Read the file of every chart type used before drawing.

| Chart type | Use it for | File |
|---|---|---|
| Line chart | values along continuous x: learning curves, scaling curves, pass@k, ROC and precision-recall curves | `line-chart.md` |
| Bar chart | reported results across methods, ablations or benchmarks; parts of a whole per category | `bar-chart.md` |
| Scatter plot | two quantities per item: accuracy vs. cost; embeddings | `scatter-plot.md` |
| Box plot | spread of many values per group, compared across groups | `box-plot.md` |
| Heatmap | values over two categorical axes: confusion matrix, attention map, hyperparameter grid | `heatmap.md` |
| Histogram | how one quantity's values are spread | `histogram.md` |
| Error bar plot | means with uncertainty, when compared values differ by <10% | `error-bar-plot.md` |
| Area chart | shares of a whole along continuous x | `area-chart.md` |
| Contour plot | function of two continuous variables | `contour-plot.md` |
| Radar chart | a few models' profiles across 5 to 10 benchmarks | `radar-chart.md` |
| Pie chart | composition of one whole, two to six parts | `pie-chart.md` |
| Violin plot | as box plot, when shape of spread matters | `violin-plot.md` |

- Tables: many exact numbers, no trend or comparison to see (every method on every dataset): suggest a table instead.

### Legend

First row that fits:

| Case | Legend |
|---|---|
| Single entry (one series, or one marked point such as a minimum) | none (lone entry above panel reads as title); caption's reading guide names it ("the star marks the trained minimizer") |
| Filled plot (heatmap, contour gradient, stacked area, stacked or win-rate bars, embedding) | outside, above panels (above top row in a grid); none when every segment named inside |
| One panel | `ax.legend()` without `loc`: `finalize_figure` puts it at the first inside spot where it hides nothing (corners, then edge centers), else above the panel in one row |
| Several panels, same series | one legend, shown once: `ax.legend()` without `loc` on the first panel; `finalize_figure` puts it inside the first panel that has a spot where it hides nothing; the others carry none; no panel has room: above, centred over the row |
| Panels with different series | one per panel, placed as for one panel |
| One series per panel | none; title names it |
| Outside, above one panel | `outside panel legend` below |

```python
# outside panel legend
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), borderaxespad=paper.legend_gap * 72 / rcParams["legend.fontsize"], ncol=n_entries)
```

- Frame: a legend inside a panel in its box (`frame` color edge, opaque fill); outside the panels, no box (`finalize_figure` removes it: nothing under it to keep apart). Never `frameon=False` by hand.
- Inside: worth a small overlap (opaque frame, no height cost). Hides nothing: no curve, data point, text or bar end; ≤1/10 of its area over filled marks. `finalize_figure` checks this when placing it, and warns when a legend with an explicit `loc` breaks it.
- Outside legend columns: start in one row. On cut-off text or matplotlib layout collapse: `ncol = ceil(n / 2)`, then `ceil(n / 3)`, …, skip repeated counts, down to 1.
- Shared words: entries that share a word and differ by a setting ("Depth 4", "Depth 8", …; a sweep in `make_shades` shades): `paper.add_legend_rows`, one row per shared word, written once at the row's start, each entry only its setting ("Depth  4 8 16 32 64"; two sweeps: two rows). A row too wide for any panel: on the figure, above the panels.

  ```python
  paper.add_legend_rows(ax, [("Depth", handles, [4, 8, 16, 32, 64])])
  paper.add_legend_rows(fig, [("Depth", depth_handles, depths), ("Width", width_handles, widths)],
                    loc="outside upper center", borderaxespad=0)
  ```
- Entry order: curves: ours first, then others in color, line-style or shade order. Side-by-side series (grouped bars, boxes): left to right. Stacked bars and areas: segments as met (top to bottom for vertical stacks, left to right for horizontal). Drawing order differs: pass handles in entry order. Always pass entries in reading order: when `finalize_figure` lays a legend out in several columns, it keeps the rows reading left to right.

  ```python
  ax.legend([h_ours, h_a, h_b], ["Ours", "A", "B"])
  ```

- `loc`: none for a panel legend (`finalize_figure` places it); explicit for figure and outside legends. Never `"best"`.

### Value labels

- Reported results the takeaway compares: written on plot.
- Diagnostics: none.
- Distributions: only summary statistics the takeaway cites (means), where chart file places them, else caption.
- Placement per chart file (`ax.bar_label` on bars, cell text in heatmaps). Curves: none; compared values in caption (`line-chart.md`).

Format: first row that applies, labels and caption alike (formats in block below):

| Value | Written as |
|---|---|
| Copied from source (task says numbers as printed in a paper) | exactly as printed: `"0.82"` stays `"0.82"` |
| Count (number of items; a statistic of counts, such as a median, is not one) | integer, `count`: `93`, `1024`; ≥10,000 `count_10k`: `12,345`; ≥1,000,000 three significant figures with axis's SI suffix: `1.47B` |
| Share of a whole (pie, stacked bar, win rate, area; not a metric such as accuracy) | percent: ≥10% whole, `share`; below, two significant figures, `small_share` (small shares keep their size): `38%`, `1.8%`, `1.0%`, `0.69%` |
| Given in data, shown unchanged (not computed by script) | as given; every value of the kind with as many significant figures as the most precise one, at most three (data drops trailing zeros): `0.076` stays `0.076`; `0.64` among `0.612` and `0.774` → `0.640`; `0.82`, `0.0069` stay; `0.54063` → `0.541` |
| ≥1000 | whole number, `large`: `1234`; ≥10,000 `large_10k` |
| Difference of two shown values | decimals of the shown values, so reader's subtraction agrees: 0.879 − 0.865 = `0.014` |
| <0.001 (not 0) | three significant figures as power of ten, true minus, `tiny` with superscript exponent: `3.00×10⁻⁵` |
| Ratio | three significant figures: `1.91×` |
| Anything else | three significant figures, trailing zeros kept, `other`: `85.0`, `0.912`, `439` |

```python
count = f"{round(v)}"
count_10k = f"{round(v):,}"
share = f"{v:.0%}"
small_share = f"{100 * v:#.2g}%"
large = f"{v:.0f}"
large_10k = f"{v:,.0f}"
tiny = f"{m:#.3g}×10{exp}"
other = f"{v:#.3g}".rstrip(".")
```

- Look: 7 pt (`value-label`), 5 pt where it does not fit; beside a mark `shift_lightness(series_color, -1)`, for every grey series one color, `shift_lightness(colors["neutral"], -1)`; on a fill black or white (Typography).
- Tag: value label at a data point via `ax.annotate` or `ax.text`: pass `gid="value-label"`, so `finalize_figure` keeps it inside panel; an annotation at a side edge turns inward. `ax.bar_label` needs none.

  ```python
  ax.plot(x, y, "o", color=color)  # a compared value: marker, value above
  ax.annotate(text, (x, y), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
              color=shift_lightness(color, -1), gid="value-label")
  ```
 No tag: labels outside panel on purpose (histogram means above it), point names (even with a value joined).
