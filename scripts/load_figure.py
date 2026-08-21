#!/usr/bin/env python3
"""
Reopen a figure pickled by ``visualization.save_fig`` (``<name>.fig.pkl``).

Every report/evaluation PNG is written together with a pickle of the live
matplotlib Figure, so a plot can be restyled after the fact without re-running
inference. Load it, tweak the axes, save it again.

NOTE: matplotlib Figure pickles are only guaranteed to reload under the same
matplotlib version that wrote them (3.11.0 for the CLEAN evaluation runs).

This is also how you zoom in on an interesting region of any committed figure: the
data is still live inside the pickled axes, so re-cropping costs no inference and
no re-run. `--xlim/--ylim` work headlessly (no display needed); `--show` opens the
usual matplotlib pan/zoom toolbar when a display is available.

Examples
--------
    # re-render unchanged (sanity check that the pickle is intact)
    python scripts/load_figure.py .../eval_2024/error_maps.fig.pkl --png /tmp/check.png

    # what panels does it have?
    python scripts/load_figure.py .../tmax_distribution_overlay.fig.pkl

    # zoom the hot tail of panel 2 (the ratio panel) and blow it up full-size
    python scripts/load_figure.py .../tmax_distribution_overlay.fig.pkl \
        --only 2 --xlim 25 38 --ylim 0.5 3 --png /tmp/hot_tail.png

    # interactive pan/zoom (needs a display)
    python scripts/load_figure.py .../error_maps.fig.pkl --show

    # interactive editing
    python -i scripts/load_figure.py .../eval_2024/error_maps.fig.pkl
    >>> fig.axes[0].set_title("New title")
    >>> fig.savefig("edited.png", dpi=150, bbox_inches="tight")
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


def load_figure(path: str | Path):
    """Unpickle and return the matplotlib Figure stored at ``path``."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pickle_path", help="Path to a *.fig.pkl written next to a PNG.")
    ap.add_argument("--png", default=None, help="Re-render the figure to this PNG path.")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--show", action="store_true",
                    help="Open interactively (pan/zoom via the matplotlib toolbar). "
                         "Needs a display; falls back to a message if none is available.")
    ap.add_argument("--axes", type=int, default=None, metavar="I",
                    help="Which subplot --xlim/--ylim apply to (default: all). "
                         "Index as printed by this script, left-to-right, top-to-bottom.")
    ap.add_argument("--xlim", type=float, nargs=2, metavar=("LO", "HI"),
                    help="Zoom the x-axis to this range and re-render.")
    ap.add_argument("--ylim", type=float, nargs=2, metavar=("LO", "HI"),
                    help="Zoom the y-axis to this range and re-render.")
    ap.add_argument("--only", type=int, default=None, metavar="I",
                    help="Keep only subplot I, so one panel fills the whole figure.")
    args = ap.parse_args(argv)

    fig = load_figure(args.pickle_path)
    print(f"loaded {args.pickle_path}: {len(fig.axes)} axes, size {fig.get_size_inches()}")
    for i, ax in enumerate(fig.axes):
        print(f"  [{i}] {ax.get_title() or ax.get_xlabel() or '(untitled)'} "
              f"x={ax.get_xlim()} y={ax.get_ylim()}")

    # Zooming a pickled figure is the point of keeping the pickles: no re-run, no
    # inference, and the data is still live inside the axes.
    targets = fig.axes if args.axes is None else [fig.axes[args.axes]]
    for ax in targets:
        if args.xlim:
            ax.set_xlim(*args.xlim)
        if args.ylim:
            ax.set_ylim(*args.ylim)

    if args.only is not None:
        keep = fig.axes[args.only]
        for ax in list(fig.axes):
            if ax is not keep:
                ax.remove()
        # Leave room under the suptitle, and re-attach a legend: the original one usually
        # lived on a panel we just removed.
        keep.set_position([0.08, 0.12, 0.89, 0.78])
        handles, labels = keep.get_legend_handles_labels()
        if labels and keep.get_legend() is None:
            keep.legend(handles, labels, fontsize=8, frameon=False, loc="best")

    if args.show:
        try:
            import matplotlib.pyplot as plt
            matplotlib.use("TkAgg", force=True)
            plt.show()
        except Exception as exc:  # no display, no GUI toolkit, …
            print(f"[warn] cannot open a window ({exc}). Use --xlim/--ylim/--only with "
                  f"--png instead — that path needs no display.")
    if args.png:
        Path(args.png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.png, dpi=args.dpi, bbox_inches="tight")
        print(f"wrote {args.png}")
    return fig


if __name__ == "__main__":
    fig = main()
    sys.exit(0) if not sys.flags.interactive else None
