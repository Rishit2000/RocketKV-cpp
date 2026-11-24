"""Helpers for scripts: a legend in named rows."""
from matplotlib.lines import Line2D


def add_legend_rows(parent, rows, **kwargs):
    """A legend for series named by a shared word and a setting ("Depth 4", "Depth 8", ...): one row per
    shared word, written once at the row's start, each entry only its setting ("4", "8", ...). `rows` is a
    list of (name, handles, settings). On an Axes without `loc`, `finalize_figure` places it like any panel
    legend and keeps its rows; on the figure with `loc="outside upper center"`, it stands above the panels."""
    width = 1 + max(len(handles) for _, handles, _ in rows)
    grid = []
    for name, handles, settings in rows:
        row = [(Line2D([], [], linestyle="none"), name), *zip(handles, map(str, settings))]
        grid.append(row + [(Line2D([], [], linestyle="none"), "")] * (width - len(row)))
    # matplotlib fills a legend column by column
    handles, labels = zip(*[grid[r][c] for c in range(width) for r in range(len(grid))])
    # entries are short settings: columns close together, markers without a line need only a short handle
    markers_only = all(isinstance(h, Line2D) and h.get_linestyle() == "None" for _, row, _ in rows for h in row)
    layout = dict(ncols=width, columnspacing=0.8, handletextpad=0.4, handlelength=1.0 if markers_only else 1.5)
    legend = parent.legend(handles, labels, **layout, **kwargs)
    legend._entries = (list(handles), list(labels), layout)
    return legend

