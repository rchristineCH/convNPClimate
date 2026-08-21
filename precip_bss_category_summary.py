#!/usr/bin/env python3
"""Pooled Brier skill score by intensity category, all five precipitation runs.

Reads the per-point, per-category BSS already cached by eval_precip_figures.py's
precip_category_spatial figure (``bss_cat``, shape (points, 5)) and pools it into
one median-BSS-per-category series per run, for both evaluation regimes. The
companion threshold view is precip_bss_threshold_summary.py; this one uses the
five ordered categories the RPSS itself is built on, so the two read together.

No new inference: purely a read of existing cached arrays.

Produces ``CLEAN_trained_models/precip_processing_comparison/bss_category_summary.png``.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_precip_models import MODELS, COLORS

OUT = Path(__file__).resolve().parent / "CLEAN_trained_models/precip_processing_comparison"
OUT.mkdir(parents=True, exist_ok=True)

REGIMES = [("CV holdout 2020-2023", "eval_figures"), ("2024 holdout", "eval_figures_2024")]
# Same left-to-right objective grouping as the headline figures.
ORDER = ["NLL", "WIND", "SFC-TP", "CRPS", "FT-CRPS"]


def main():
    results, labels = {}, None
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True)
    for ax, (title, subdir) in zip(axes, REGIMES):
        for name in ORDER:
            npz = MODELS[name] / subdir / "precip_category_spatial.npz"
            if not npz.exists():
                continue
            d = np.load(npz, allow_pickle=True)
            labels = [str(x) for x in d["cat_labels"]]
            med = [float(np.nanmedian(d["bss_cat"][:, k])) for k in range(len(labels))]
            results.setdefault(title, {})[name] = dict(zip(labels, med))
            ax.plot(range(len(labels)), med, "o-", color=COLORS[name], lw=1.7,
                    ms=5, label=name)
        ax.axhline(0, color="k", lw=0.9, ls=":")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9.5)
        ax.set_xlabel("daily accumulation category (mm)")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("median Brier skill score vs ERA5-Land")
    axes[0].legend(fontsize=9, loc="lower left", framealpha=0.9)
    fig.suptitle("Brier skill by intensity category, per precipitation run "
                 "(positive = better than the bilinear ERA5-Land reference)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "bss_category_summary.png", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT / 'bss_category_summary.png'}")
    (OUT / "bss_category_summary.json").write_text(json.dumps(results, indent=2))
    for regime, runs in results.items():
        print(f"\n{regime}")
        for r, v in runs.items():
            print("  %-8s %s" % (r, {k: round(x, 3) for k, x in v.items()}))


if __name__ == "__main__":
    main()
