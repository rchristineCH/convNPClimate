#!/usr/bin/env python
"""Combine the feature-importance method CSVs into one evaluation.

Invoked by ``scripts/fi_evaluate.sh``. Combines whichever of ``pfi.csv``,
``shap.csv`` and ``lime.csv`` are present (at least two are needed for a
cross-method report) and says explicitly which ones were missing — SHAP is the
~30 h cluster method, so PFI+LIME reports are the common case and should not be
blocked on it. It does NOT re-run any expensive scoring: it rebuilds
the lightweight backbone once (for the channel groupings + a single baseline
pass) and reuses ``feature_importance.write_summary`` to regenerate the
cross-method ``summary.md`` and the level x hour heatmaps, then assembles a
full-run ``REPORT.md``.

Standalone tmax model: CSVs live directly in ``<fi_dir>`` (one report).
Joint two-stage model (``joint_meta.json`` present): both heads are combined —
CSVs are read from ``<fi_dir>/tmax`` and ``<fi_dir>/precip`` and a
``summary.md`` + ``REPORT.md`` are written into each head's subdir.

Usage:  python scripts/fi_combine.py <model_dir> <year> <fi_dir>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Run-from-repo-root import: this file lives in scripts/, the modules are at the root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

import feature_importance as fi  # noqa: E402
import params as params_mod  # noqa: E402

METHODS = ("pfi", "shap", "lime")


def _available(head_dir: Path) -> list[str]:
    """Methods whose CSV is present in ``head_dir``, in canonical order."""
    return [m for m in METHODS if (head_dir / f"{m}.csv").exists()]


def _top_group(df: pd.DataFrame, method: str, grouping: str, metric: str,
               n: int = 3) -> list[tuple[str, float]]:
    """Top-n group labels for a method/grouping by |importance|, label stripped of prefix."""
    d = df[(df.method == method) & (df.grouping == grouping) & (df.metric == metric)].copy()
    d = d.reindex(d["importance"].abs().sort_values(ascending=False).index).head(n)
    return [(str(r["feature"]).split("=")[-1], float(r["importance"])) for _, r in d.iterrows()]


def _fmt_top(items: list[tuple[str, float]]) -> str:
    return ", ".join(f"{label} ({val:+.3f})" for label, val in items)


def build_report(model_dir: Path, span: str, head_dir: Path, bb, head: fi._Head,
                 frames: dict[str, pd.DataFrame]) -> Path:
    days = bb.context.shape[0]
    folds = len(bb.models)
    targets = (bb.truth_precip.shape[1] if head.name == "precip" and bb.truth_precip is not None
               else bb.truth_n.shape[1])
    base = bb.baseline
    is_precip = head.name == "precip"

    # Describe the actual channel composition (variables / levels / hours) from the
    # model's channel names instead of hardcoding z/t/q, so wind runs read correctly.
    _meta = [fi._parse_channel(n) for n in bb.channel_names]
    _phys = [m for m in _meta if m["var"] != "scaffold"]
    _var_order = ("z", "t", "q", "r", "u", "v")
    _present = {m["var"] for m in _phys}
    _vars = [v for v in _var_order if v in _present] + sorted(_present - set(_var_order))
    _levels = sorted({m["level"] for m in _phys if str(m["level"]).isdigit()}, key=int, reverse=True)
    _hours = sorted({m["hour"] for m in _phys})
    _n_scaffold = len(_meta) - len(_phys)
    # A surface run has one 'data' channel with no pressure-level / hour structure;
    # describing it with an empty "{ hPa}" axis (and calling it atmospheric) is wrong.
    is_atmos = bool(_levels and [h for h in _hours if h != "-"])
    if is_atmos:
        channel_desc = (
            f"{len(bb.channel_names)} channels: {'/'.join(_vars)} x {len(_levels)} levels "
            f"{{{','.join(_levels)} hPa}} x {len(_hours)} hours {{{','.join(_hours)} UTC}} "
            f"+ {_n_scaffold} scaffold")
        grid_desc = "atmospheric, native coarse grid"
    else:
        channel_desc = (f"{len(bb.channel_names)} channels: {'/'.join(_vars)} "
                        f"+ {_n_scaffold} scaffold")
        grid_desc = "surface ERA5 input (no pressure levels)"

    consensus = (
        "moisture (q) and low-level dynamics (geopotential z / temperature) drive where and how "
        "much precipitation the model downscales — physically expected for daily accumulation."
        if is_precip else
        "near-surface (low-level) air temperature in the afternoon/evening is what the model relies "
        "on to downscale daily-max 2 m temperature, with geopotential a secondary cue and humidity "
        "minor — physically expected.")
    if not is_atmos:
        # The canned consensus describes a pressure-level model; a surface run has no
        # levels or hours to talk about, so state what it actually shows.
        consensus = ("this is a surface run — the single ERA5 input channel carries the signal, "
                     "with the scaffold (lat/lon/elevation/season) as the secondary cue; there "
                     "are no pressure levels or hours to attribute across.")
    elif len(_vars) > 3 or "u" in _present or "v" in _present:
        # The canned consensus above assumes a z/t/q model; with extra fields
        # (e.g. wind) defer to the data-driven rankings below rather than overclaim.
        consensus = ("see the per-variable rankings below for how each atmospheric field "
                     f"({'/'.join(_vars)}) contributes.")

    methods = [m for m in METHODS if m in frames]
    absent = [m for m in METHODS if m not in frames]
    blurb = {"pfi": "PFI", "shap": "SHAP (KernelSHAP + exact grouped)", "lime": "LIME"}

    # Only a joint run has two heads; a standalone run must not claim to be one.
    kind = (f"joint two-stage tmax+precip — **{head.name} head**" if bb.is_joint
            else f"standalone {head.name}")

    L: list[str] = []
    L += [f"# Feature importance — full-run analysis report — {head.name} ({span})", ""]
    L += [
        f"**Model:** `{model_dir.parent.name}/{model_dir.name}` "
        f"({grid_desc}; {kind}; {channel_desc}).",
        f"**Methods:** {', '.join(blurb[m] for m in methods)} — self-implemented, no external deps."
        + (f" **{'/'.join(m.upper() for m in absent)} not available for this run** "
           f"(no {', '.join(f'`{m}.csv`' for m in absent)}), so the tables below are"
           f" {len(methods)}-method." if absent else ""),
        (f"**Attribution metrics:** MAE (mm) + Bernoulli-Gamma NLL vs MeteoSwiss RhiresD truth."
         if is_precip else
         "**Attribution metrics:** MAE / NLL / CRPS (degC) vs MeteoSwiss TmaxD truth."),
        "",
    ]

    L += ["## Run configuration", "",
          "| | value |", "|---|---|",
          f"| Data span | {span} |",
          f"| Days scored | **{days}** |",
          f"| Folds | **{folds}** of 5 (ensemble) |",
          f"| Target points | {targets:,} |", ""]

    base_txt = " · ".join(f"{m.upper()} **{base[m]:.3f}**" for m in head.metrics)
    L += ["## Baseline (all channels present)", "",
          f"{base_txt}  [{head.unit}]. Importance = how much removing/scrambling a feature degrades "
          "these metrics (higher = more important).", ""]

    # Data-driven consensus over whatever methods ran, on each one's own metric
    # (LIME attributes to the prediction, PFI/SHAP to the error metric).
    L += ["## Bottom line", ""]
    for grouping, lead in (("variable", "By variable"), ("level", "By pressure level"),
                           ("hour", "By hour (UTC)")):
        tops = {m: _top_group(frames[m], m, grouping, head.metric_for(m)) for m in methods}
        if not any(tops.values()):
            continue    # surface runs have no level/hour grouping — omit, don't print blanks
        per = "; ".join(f"{m.upper()}: {_fmt_top(tops[m])}" for m in methods)
        L.append(f"- **{lead}** — {per}")
    L += ["", f"Consensus across methods: {consensus}", ""]

    # Embed the authoritative machine-generated cross-method tables.
    summary_path = head_dir / "summary.md"
    if summary_path.exists():
        body = summary_path.read_text().splitlines()
        if body and body[0].startswith("# "):
            body = body[1:]
        L += ["## Cross-method tables", "",
              "*(machine-generated; see `summary.md`)*", ""]
        L += body
        L += [""]

    # Figure gallery.
    L += ["## Figure gallery", ""]
    for method in methods:
        metric = head.metric_for(method)
        pngs = [f"{method}_channel.png", f"{method}_variable.png", f"{method}_level.png",
                f"{method}_hour.png", f"{method}_heatmap_{metric}.png"]
        L += [f"### {method.upper()}"]
        for png in pngs:
            if (head_dir / png).exists():
                L.append(f"![{method} {png}]({png})")
        L.append("")

    L += ["*Generated by `scripts/fi_evaluate.sh` from "
          + ", ".join(f"`{m}.csv`" for m in methods)
          + " and the regenerated `summary.md` in this directory.*"]

    out = head_dir / "REPORT.md"
    out.write_text("\n".join(L))
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 2
    model_dir = Path(argv[1])
    year = int(argv[2])
    fi_dir = Path(argv[3])

    params_mod.configure_renku_cuda()
    device = params_mod.select_device()
    # Backbone is needed only for grouping metadata + a single baseline pass. These
    # env knobs let the combine path be smoke-tested cheaply; unset → full resolution.
    stride = int(os.environ.get("FI_EVAL_STRIDE", "1"))
    n_days = os.environ.get("FI_EVAL_NDAYS")
    folds = os.environ.get("FI_EVAL_FOLDS")
    n_days = int(n_days) if n_days else None
    folds = int(folds) if folds else None

    is_joint = (model_dir / "joint_meta.json").exists()
    if is_joint:
        p_meta = params_mod.Params.load_json(model_dir / "params.json")
        span = f"{p_meta.DATA_YEAR_START}-{p_meta.DATA_YEAR_END or p_meta.DATA_YEAR_START}"
        bb = fi._build_backbone_joint(model_dir, device, day_stride=stride, n_days=n_days,
                                      folds=folds, seed=0)
    else:
        span = str(year)
        bb = fi._build_backbone(model_dir, year, device, day_stride=stride, n_days=n_days,
                                folds=folds, seed=0)

    json.dump(bb.baseline, open(fi_dir / "baseline.json", "w"), indent=2)

    for head in fi._report_heads(bb):
        head_dir = fi_dir if head.subdir == "." else (fi_dir / head.subdir)
        avail = _available(head_dir)
        if len(avail) < 2:
            raise SystemExit(
                f"{head_dir}: need at least two method CSVs for a cross-method report, "
                f"found {avail or 'none'}")
        if len(avail) < len(METHODS):
            print(f"NOTE {head_dir}: combining {avail} — "
                  f"{[m for m in METHODS if m not in avail]} not present")
        frames = {m: pd.read_csv(head_dir / f"{m}.csv") for m in avail}
        fi.write_summary(frames, bb, head_dir, head, span)
        report = build_report(model_dir, span, head_dir, bb, head, frames)
        print(f"wrote {head_dir / 'summary.md'}")
        print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
