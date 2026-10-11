"""Shared Matplotlib style and helpers used to draw the figures of the paper."""
from enum import Enum
import os
import shutil

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D

# Width and height (inches) of a single panel in the paper.
FIGSIZE = (3.7, 2.7)


def set_style(usetex: bool | None = None):
    """Apply the LaTeX-like style of the paper.

    ``usetex`` defaults to True when a ``latex`` executable is found and the
    environment variable ``LSQ_NO_TEX`` is not set.
    """
    if usetex is None:
        usetex = shutil.which("latex") is not None and not os.environ.get("LSQ_NO_TEX")
    plt.rcParams.update({
        "text.usetex": usetex,
        "font.family": "serif",
        "font.serif": ["Computer Modern"] if usetex else ["DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 16,
        "font.size": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.figsize": FIGSIZE,
    })
    return usetex


class ScheduleCmap(Enum):
    """One Matplotlib colormap per schedule (consistent across all figures)."""
    CONSTANT = "Blues"
    LINEAR = "Oranges"
    WSD = "Greens"
    MULTI = "Purples"

    def __call__(self, intensity: float):
        assert 0.0 <= intensity <= 1.0, "Intensity must be between 0.0 and 1.0"
        return plt.get_cmap(self.value)(intensity)


def shade(index: int, n: int, low: float = 0.5, high: float = 1.0) -> float:
    """Intensity of the ``index``-th curve out of ``n`` (light -> dark)."""
    return low + (high - low) * index / max(1, n - 1)


class VerticalLineHandler(HandlerBase):
    """Legend handler drawing a tuple of lines stacked vertically in one entry."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        artists = []
        step = height / max(len(orig_handle), 1)
        for i, handle in enumerate(orig_handle):
            y = height - (i + 0.5) * step - ydescent
            artists.append(Line2D([0, width], [y, y], color=handle.get_color(),
                                  linestyle=handle.get_linestyle(), linewidth=handle.get_linewidth()))
        return artists


def grouped_legend(ax, groups, labels, loc="upper right", handleheight=2.5, fontsize=None):
    """Legend where each entry groups several curves (e.g. one per value of alpha).

    ``fontsize`` defaults to ``rcParams["legend.fontsize"]``; use a smaller value
    for long labels so that the legend stays inside the axes.
    """
    ax.legend([tuple(g) for g in groups], labels,
              handler_map={tuple: VerticalLineHandler()},
              handleheight=handleheight, loc=loc, fontsize=fontsize,
              handlelength=1.5, borderaxespad=0.3)


def finish(ax, xlabel, ylabel, xscale="linear", yscale="linear"):
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.6)
