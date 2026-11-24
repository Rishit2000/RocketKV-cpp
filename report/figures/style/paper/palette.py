"""Color-blind and grayscale safety: greys and part colors spread in lightness, and color helpers."""
import colorsys

import numpy as np
from matplotlib import colors

from . import design




def _lightness(color):
    """CIE L* (0 black to 100 white): how light a color prints in grayscale."""
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in colors.to_rgb(color)]
    y = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    return 116 * y ** (1 / 3) - 16 if y > 0.008856 else 903.3 * y


def _over_white(rgba):
    """A translucent color as it looks over the white page."""
    r, g, b, alpha = colors.to_rgba(rgba)
    return tuple(alpha * c + (1 - alpha) for c in (r, g, b))


def _hue_and_saturation(color):
    h, _, sat = colorsys.rgb_to_hls(*colors.to_rgb(color))
    return h, (sat if sat > 0.1 else 0.0)  # a grey has no hue to keep (colorsys reports hue 0, red)


def _at_hls_lightness(h, sat, target):
    """The color of hue `h` and saturation `sat` whose grayscale lightness is `target` L*."""
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _lightness(colorsys.hls_to_rgb(h, mid, sat)) < target else (lo, mid)
    return colorsys.hls_to_rgb(h, (lo + hi) / 2, sat)


def shift_lightness(color, stops):
    """`color` lighter (stops > 0) or darker (stops < 0), like an exposure dial: each stop moves its
    grayscale lightness by the `shades.stop` token's L*, keeping hue and saturation. -1: series text and edges on white;
    +2: box and violin fills, light enough for the black median on them."""
    h, sat = _hue_and_saturation(color)
    return _at_hls_lightness(h, sat, min(100, max(0, _lightness(color) + design.shades["stop"] * stops)))


def make_shades(color, n):
    """n shades of `color`, light to dark: a sweep inside one group (depths, sizes), or greys for n compared series.
    Neighbours sit `step` L* apart, so they stay apart in grayscale. A color's darkest shade is the color itself;
    when the shades would run past `lightest`, they slide darker. Greys start at `lightest` and skip the band
    around ours' lightness, so ours stays apart in grayscale. When `lightest` to `darkest` cannot hold n shades
    `step` apart, they spread evenly over it. One shade: `color` itself."""
    if n == 1:
        return [tuple(colors.to_rgb(color))]
    h, sat = _hue_and_saturation(color)
    lightest, darkest, step, stop = (design.shades[k] for k in ("lightest", "darkest", "step", "stop"))
    segments = [(lightest, darkest)]
    start = max(0, lightest - _lightness(color) - step * (n - 1))  # L* below `lightest` of the lightest shade
    if not sat:
        ours = _lightness(design.colors["ours"])
        segments = [(a, b) for a, b in [(lightest, min(lightest, ours + stop)), (max(darkest, ours - stop), darkest)] if a > b]
        start = 0
    gap = min(step, sum(a - b for a, b in segments) / (n - 1))
    out = []
    for i in range(n):
        t = start + gap * i
        for a, b in segments:
            if t <= a - b + 1e-9:
                out.append(_at_hls_lightness(h, sat, a - t))
                break
            t -= a - b
    return out


def make_sequential_colormap(name, color, lightest, darkest):
    """One palette color as a colormap for values that only grow: low values light, high dark, its grayscale
    lightness evenly spread from `lightest` to `darkest` L*, so the order reads in black and white too."""
    h, sat = _hue_and_saturation(color)
    return colors.LinearSegmentedColormap.from_list(name, [_at_hls_lightness(h, sat, l) for l in np.linspace(lightest, darkest, 256)])


def make_diverging_colormap(name, negative, positive, lightest, darkest):
    """Two palette colors as a colormap for signed values: dark negative through near-white zero to dark
    positive, each side the same lightness at the same distance from zero."""
    sides = [_hue_and_saturation(c) for c in (negative, positive)]
    ramp = np.linspace(darkest, lightest, 128)
    return colors.LinearSegmentedColormap.from_list(
        name, [_at_hls_lightness(*sides[0], l) for l in ramp] + [_at_hls_lightness(*sides[1], l) for l in ramp[::-1]])


def make_part_colors(n):
    """n colors for the parts of one whole (bands, wedges, segments), in stacking order: the palette's own colors,
    unchanged, in the `part-colors` token's order, whose neighbours stay apart in black and white."""
    return [design.colors[name] for name in design.part_colors["order"][:n]]
