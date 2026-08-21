#!/usr/bin/env python3
"""
Render the full ConvCNP model architecture as a diagram.

The diagram is built by introspecting a live model from ``model_factory`` -- the
pipeline stages, their classes, trainable-parameter counts, the output
distribution, its parameters, and the loss function are all read from the built
objects and the LIKELIHOODS registry. Changing the architecture, distribution,
or config (channels, blocks, encoder, seasonal features) is reflected
automatically the next time the script runs; nothing is hard-coded.

Examples
--------
    # Default tmax (Gaussian) model
    python visualize_architecture.py

    # Precip Bernoulli-Gamma, more input channels, no seasonal MLP inputs
    python visualize_architecture.py --variable precip --in-channels 40 --no-seasonal

    # Match an exact trained run (reads its params.json)
    python visualize_architecture.py --model-dir trained_models/<run>/precip
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

import model_factory
from params import Params


# Output-parameter names per distribution (falls back to p0..pn for unknowns).
PARAM_NAMES = {
    "gaussian": ["mu", "sigma"],
    "bernoulli_gamma": ["rho", "alpha", "beta"],
    "bernoulli_gamma_crps": ["rho", "alpha", "beta"],
}

# Human labels for the loss node (falls back to the function name).
LOSS_LABELS = {
    "gll": "Gaussian NLL",
    "gamma_ll": "Bernoulli-Gamma NLL",
    "gamma_bernoulli_crps": "Bernoulli-Gamma CRPS",
}


def _loss_name(loss_fn):
    """Loss function name, robust to functools.partial wrappers (e.g. the CRPS
    loss whose n_samples is bound in build_model — a partial has no __name__)."""
    return getattr(loss_fn, "__name__", None) or getattr(
        getattr(loss_fn, "func", None), "__name__", "loss")

# Canonical forward-order pipeline. Each stage names the model attribute(s) that
# hold it (first match wins) plus a short description of what it does. Any child
# module not matched here is appended afterwards so new sub-modules still show up.
PIPELINE = [
    ("encoder", ["encoder"], "SetConv encoder (depthwise RBF)"),
    ("CNN decoder", ["decoder"], "ResNet CNN"),
    ("MLP (grid to params)", ["mlp"], "1x1 conv MLP"),
    ("Final layer (grid to target)", ["final_layer", "out_layer"],
     "RBF interpolation to target points"),
    ("Elevation / seasonal MLP", ["elev_mlp"], "DEM + TPI + seasonal bias"),
]

COL = {
    "stage": "#cfe3f7",
    "stage_edge": "#3b6ea5",
    "io": "#e8e8e8",
    "io_edge": "#888888",
    "out": "#d5efd5",
    "out_edge": "#3f8f3f",
    "loss": "#f7d6cf",
    "loss_edge": "#b0503f",
    "side": "#fdf3d0",
    "side_edge": "#c9a24b",
    "cond": "#8e5fa8",
}


def n_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build_params(args):
    """Build the Params to visualise, from a run's params.json or CLI overrides."""
    if args.model_dir:
        p = Params.load_json(Path(args.model_dir) / "params.json")
    else:
        p = Params(
            VARIABLE=args.variable,
            DISTRIBUTION=args.distribution,
            SEASONAL_FEATURES_IN_MLP=not args.no_seasonal,
            ENCODER=args.encoder,
        )
    # IN_CHANNELS must be positive to build the model; take it from the run if
    # present, otherwise from the CLI (it drives the encoder's parameter count).
    if getattr(p, "IN_CHANNELS", 0) <= 0:
        p = p.with_in_channels(args.in_channels)
    return p


def detect_finetune(model_dir, params):
    """Fine-tune provenance for a warm-started run (see finetune_precip.py).

    A fine-tuned run's manifest carries ``finetuned_from`` -> the source run
    dir. Returns None for normal runs so their diagram is unchanged; otherwise
    a dict with the source loss/LR and the fine-tune schedule from params.
    """
    if not model_dir:
        return None
    man_path = Path(model_dir) / "manifest.json"
    if not man_path.exists():
        return None
    src = json.loads(man_path.read_text()).get("finetuned_from")
    if not src:
        return None
    info = {"src": src, "ft_lr": params.LR, "ft_epochs": params.N_EPOCHS,
            "ft_patience": params.PATIENCE,
            "src_distribution": None, "src_loss": None, "src_lr": None}
    src_params = Path(src) / "params.json"
    if src_params.exists():
        sp = Params.load_json(src_params)
        src_dist = model_factory.resolve_distribution(sp)
        info.update(
            src_distribution=src_dist,
            src_loss=_loss_name(model_factory.LIKELIHOODS[src_dist].loss_fn),
            src_lr=sp.LR)
    return info


def collect_stages(model):
    """Return ordered [(title, subtitle, n_params, extra)] for the model."""
    used = set()
    stages = []
    for title, attrs, desc in PIPELINE:
        for a in attrs:
            mod = getattr(model, a, None)
            if mod is not None and hasattr(mod, "parameters"):
                used.add(a)
                extra = ""
                # Highlight the learnable RBF length-scales inside a final layer.
                n_ls = sum(1 for name, _ in mod.named_parameters() if "init_ls" in name)
                if n_ls:
                    extra = f"+ {n_ls} learnable RBF length-scale(s)"
                stages.append((title, f"{type(mod).__name__} - {desc}",
                               n_trainable(mod), extra))
                break
    # Any additional top-level child not covered above (keeps the diagram honest
    # if someone adds a new sub-module to a model).
    for name, mod in model.named_children():
        if name not in used and n_trainable(mod) > 0:
            stages.append((name, type(mod).__name__, n_trainable(mod), ""))
    return stages


def draw_box(ax, cx, cy, w, h, title, lines, face, edge, title_size=10):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5, facecolor=face, edgecolor=edge, zorder=2))
    ax.text(cx, cy + h / 2 - 0.16, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", zorder=3)
    ax.text(cx, cy + h / 2 - 0.44, "\n".join(lines), ha="center", va="top",
            fontsize=8, color="#333333", zorder=3)


def arrow(ax, x0, y0, x1, y1, label="", style="-", color="#444444", rad=0.0,
          lx=0.15, ly=0.0, fs=7.5):
    cs = f"arc3,rad={rad}" if rad else "arc3,rad=0"
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
        linewidth=1.3, linestyle=style, color=color, zorder=1,
        shrinkA=0, shrinkB=0, connectionstyle=cs))
    if label:
        ax.text((x0 + x1) / 2 + lx, (y0 + y1) / 2 + ly, label, ha="left",
                va="center", fontsize=fs, color="#555555", style="italic")


def color_legend(ax, x, y, extra=None):
    """Standard colour legend, anchored at axes-fraction (x, y)."""
    handles = [
        Patch(facecolor=COL["io"], edgecolor=COL["io_edge"], label="Input / data tensor"),
        Patch(facecolor=COL["stage"], edgecolor=COL["stage_edge"],
              label="Learnable stage (trainable params)"),
        Patch(facecolor=COL["side"], edgecolor=COL["side_edge"],
              label="Static input (precomputed geometry / topography)"),
        Patch(facecolor=COL["out"], edgecolor=COL["out_edge"],
              label="Output distribution parameters"),
        Patch(facecolor=COL["loss"], edgecolor=COL["loss_edge"],
              label="Loss (training objective)"),
    ]
    if extra:
        handles.append(extra)
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(x, y),
              fontsize=8.5, frameon=True, framealpha=0.95, borderpad=0.8,
              labelspacing=0.6, title="Legend", title_fontsize=9)


def render(model, loss_fn, params, distribution, spec, out_path, dpi,
           finetune=None):
    n_ch = params.N_CHANNELS
    n_p = spec.n_params
    c_in = params.IN_CHANNELS
    param_names = PARAM_NAMES.get(distribution, [f"p{i}" for i in range(n_p)])
    seasonal = getattr(params, "SEASONAL_FEATURES_IN_MLP", True)

    stages = collect_stages(model)
    total = n_trainable(model)

    # Layout: main spine of boxes stacked top-to-bottom, plus an output-params
    # box and a loss box; a few dashed side inputs feed the later stages.
    nodes = ["input"] + [s[0] for s in stages] + ["output", "loss"]
    n = len(nodes)
    box_w, box_h, gap = 5.2, 1.15, 0.72
    step = box_h + gap
    top = n * step
    cx = 4.0

    fig_h = max(7.0, 0.95 * n + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, top + 1.2)
    ax.axis("off")

    # Header / summary panel.
    title = f"ConvCNP architecture  -  {params.VARIABLE}  /  {distribution}"
    if finetune:
        title += "   (warm-started fine-tune)"
    ax.text(0.2, top + 0.9, title, fontsize=14, fontweight="bold", va="center")
    loss_fn_name = _loss_name(loss_fn)
    loss_label = LOSS_LABELS.get(loss_fn_name, loss_fn_name)
    n_ls_total = sum(1 for name, _ in model.named_parameters() if "init_ls" in name)
    ax.text(0.2, top + 0.35,
            f"trainable params: {total:,}   |   encoder: {params.ENCODER}   |   "
            f"N_CHANNELS: {n_ch}   |   N_BLOCKS: {params.N_BLOCKS}   |   "
            f"RBF length-scales: {n_ls_total}   |   loss: {loss_fn_name}",
            fontsize=9, color="#444444", va="center")

    def y_of(i):  # box i from the top
        return top - i * step - box_h / 2

    # Input node.
    draw_box(ax, cx, y_of(0), box_w, box_h, "ERA5 context",
             [f"(B, {c_in}, lat, lon) + context mask"], COL["io"], COL["io_edge"])

    # Pipeline stages.
    shape_after = {
        "encoder": f"(B, {n_ch}, lat, lon)",
        "CNN decoder": f"(B, {n_ch}, lat, lon)",
        "MLP (grid to params)": f"(B, {n_p}, lat, lon)",
        "Final layer (grid to target)": f"(B, N_pts, {n_p})",
        "Elevation / seasonal MLP": f"(B, N_pts, {n_p})",
    }
    prev_shape = f"(B, {c_in}, lat, lon)"
    for i, (title, subtitle, npar, extra) in enumerate(stages, start=1):
        lines = [subtitle, f"{npar:,} params"]
        if extra:
            lines.append(extra)
        draw_box(ax, cx, y_of(i), box_w, box_h, title, lines,
                 COL["stage"], COL["stage_edge"])
        arrow(ax, cx, y_of(i - 1) - box_h / 2, cx, y_of(i) + box_h / 2, prev_shape)
        prev_shape = shape_after.get(title, prev_shape)

    # Output-params node.
    i_out = len(stages) + 1
    draw_box(ax, cx, y_of(i_out), box_w, box_h, "Distribution parameters",
             [f"{distribution}:  [{', '.join(param_names)}]",
              f"(B, N_pts, {n_p})  at MeteoSwiss target points"],
             COL["out"], COL["out_edge"])
    arrow(ax, cx, y_of(i_out - 1) - box_h / 2, cx, y_of(i_out) + box_h / 2, prev_shape)

    # Loss node.
    i_loss = i_out + 1
    loss_subtitle = ("mean CRPS over target points" if "crps" in loss_fn_name
                     else "mean negative log-likelihood over target points")
    loss_lines = [loss_label, loss_subtitle]
    loss_h = box_h
    if finetune:
        lr_note = (f"lr {finetune['ft_lr']:g} (pre-train {finetune['src_lr']:g})"
                   if finetune["src_lr"] else f"lr {finetune['ft_lr']:g}")
        loss_lines.append(
            f"fine-tune: {lr_note}, <= {finetune['ft_epochs']} epochs")
        loss_lines.append(
            f"early stop: patience {finetune['ft_patience']} on held-out loss")
        loss_h = box_h + 0.55
    draw_box(ax, cx, y_of(i_loss), box_w, loss_h, f"Loss:  {loss_fn_name}",
             loss_lines, COL["loss"], COL["loss_edge"])
    arrow(ax, cx, y_of(i_out) - box_h / 2, cx, y_of(i_loss) + loss_h / 2,
          "+ observed y")

    # Dashed side inputs feeding the geometry-aware stages.
    side_x = 9.7
    side_specs = [("Final layer (grid to target)",
                   "dists", "(N_pts, lat, lon)\nsquared-distance matrix")]
    side_specs.append(("Elevation / seasonal MLP", "elev", "(N_pts, 3)\nDEM, elev diff, TPI"))
    if seasonal:
        side_specs.append(("Elevation / seasonal MLP", "seasonal",
                           "(B, 2)\ncos/sin day-of-year"))
    stage_index = {s[0]: k for k, s in enumerate(stages, start=1)}
    # Group inputs by the stage they feed so that several inputs to the same
    # stage stack vertically instead of overlapping.
    groups, order = {}, []
    for spec in side_specs:
        if spec[0] not in groups:
            groups[spec[0]] = []
            order.append(spec[0])
        groups[spec[0]].append(spec)
    for target_title in order:
        if target_title not in stage_index:
            continue
        yi = y_of(stage_index[target_title])
        specs = groups[target_title]
        for j, (_, name, desc) in enumerate(specs):
            dy = ((len(specs) - 1) / 2 - j) * 1.24  # spread around the stage y
            by = yi + dy
            draw_box(ax, side_x, by, 3.4, 0.95, name, [desc],
                     COL["side"], COL["side_edge"], title_size=9)
            arrow(ax, side_x - 1.7, by, cx + box_w / 2, yi + dy * 0.35,
                  style="--", color="#c9a24b")

    # Warm-start provenance for fine-tuned runs: the weights do not start at
    # random init but from the source run's per-fold checkpoints.
    ft_handle = None
    if finetune:
        anchor = stage_index.get("CNN decoder", 1)
        yi = y_of(anchor)
        src_loss_label = LOSS_LABELS.get(finetune["src_loss"],
                                         finetune["src_loss"]) or "pre-trained"
        draw_box(ax, side_x, yi, 3.8, 1.25, "Warm start",
                 [f"weights from {src_loss_label} run",
                  "per-fold checkpoints, same fold split",
                  "all stages stay trainable"],
                 "#ece2f2", COL["cond"], title_size=9)
        arrow(ax, side_x - 1.9, yi, cx + box_w / 2, yi,
              style="--", color=COL["cond"])
        ft_handle = Patch(facecolor="#ece2f2", edgecolor=COL["cond"],
                          label="Warm-start initialisation (fine-tune)")

    color_legend(ax, 1.0, top / (top + 1.2), extra=ft_handle)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path, total, stages


def render_joint(model, params, out_path, dpi):
    """Branching diagram for JointConditionalConvCNP (shared backbone -> two heads)."""
    nt = n_trainable
    seasonal = getattr(params, "SEASONAL_FEATURES_IN_MLP", True)
    c_in, n_ch = params.IN_CHANNELS, params.N_CHANNELS
    total = nt(model)

    box_w, box_h, step = 4.0, 1.05, 1.62
    x_t, x_c, x_p = 4.2, 8.0, 11.8            # tmax col, centre, precip col
    n_rows = 9
    top = n_rows * step + 0.6

    fig, ax = plt.subplots(figsize=(15, 13))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, top + 1.4)
    ax.axis("off")

    def y(r):
        return top - r * step - box_h / 2

    ax.text(0.2, top + 1.05, f"Joint ConvCNP architecture  -  {type(model).__name__}",
            fontsize=15, fontweight="bold", va="center")
    ax.text(0.2, top + 0.5,
            f"trainable params: {total:,}   |   shared backbone -> tmax + precip heads   |   "
            f"N_CHANNELS: {n_ch}   |   N_BLOCKS: {params.N_BLOCKS}   |   "
            f"cross-conditioning: tmax -> precip",
            fontsize=9.5, color="#444444", va="center")

    def box(cx, r, title, lines, face, edge, w=box_w, ts=10):
        draw_box(ax, cx, y(r), w, box_h, title, lines, face, edge, title_size=ts)

    # --- shared backbone (centre column, rows 0-2) ---
    box(x_c, 0, "ERA5 context", [f"(B, {c_in}, lat, lon) + mask"], COL["io"], COL["io_edge"])
    box(x_c, 1, "encoder", [f"{type(model.encoder).__name__}", f"{nt(model.encoder):,} params"],
        COL["stage"], COL["stage_edge"])
    box(x_c, 2, "CNN decoder (shared)", [f"{type(model.decoder).__name__}",
        f"{nt(model.decoder):,} params"], COL["stage"], COL["stage_edge"])
    arrow(ax, x_c, y(0) - box_h / 2, x_c, y(1) + box_h / 2, f"(B, {c_in}, lat, lon)")
    arrow(ax, x_c, y(1) - box_h / 2, x_c, y(2) + box_h / 2, f"(B, {n_ch}, lat, lon)")

    # --- head columns (rows 3-7) ---
    heads = [
        (x_t, "tmax head", "gaussian", ["mu", "sigma"], "Gaussian NLL (gll)",
         [("MLP (grid to params)", model.mlp_t), ("Final layer (grid to target)", model.final_t),
          ("Elevation / seasonal MLP", model.elev_mlp_t)]),
        (x_p, "precip head", "bernoulli_gamma", ["rho", "alpha", "beta"],
         "Bernoulli-Gamma NLL (gamma_ll)",
         [("MLP (grid to params)", model.mlp_p), ("Final layer (grid to target)", model.final_p),
          ("Elevation / seasonal MLP", model.elev_mlp_p)]),
    ]
    for xh, hname, dist, pnames, loss_lbl, mods in heads:
        # decoder -> head MLP
        arrow(ax, x_c + (0.9 if xh > x_c else -0.9), y(2) - box_h / 2,
              xh, y(3) + box_h / 2, "shared grid feats")
        for k, (title, mod) in enumerate(mods):
            r = 3 + k
            extra = [f"{nt(mod):,} params"]
            n_ls = sum(1 for nm, _ in mod.named_parameters() if "init_ls" in nm)
            if n_ls:
                extra.append(f"+ {n_ls} learnable length-scale(s)")
            box(xh, r, title, [type(mod).__name__] + extra, COL["stage"], COL["stage_edge"])
            if k > 0:
                arrow(ax, xh, y(r - 1) - box_h / 2, xh, y(r) + box_h / 2)
        # output params + loss
        box(xh, 6, f"{hname} params", [f"[{', '.join(pnames)}]  (B, N_pts, {len(pnames)})"],
            COL["out"], COL["out_edge"])
        arrow(ax, xh, y(5) - box_h / 2, xh, y(6) + box_h / 2)
        box(xh, 7, f"Loss: {loss_lbl.split('(')[1].rstrip(')')}", [loss_lbl],
            COL["loss"], COL["loss_edge"])
        arrow(ax, xh, y(6) - box_h / 2, xh, y(7) + box_h / 2, "+ obs y")

    # --- combined loss (row 8, centre) ---
    box(x_c, 8, "combined_loss", ["Kendall uncertainty weighting",
        "learnable log_var_t, log_var_p (2 scalars)"], COL["loss"], COL["loss_edge"], w=5.0)
    arrow(ax, x_t, y(7) - box_h / 2, x_c - 1.2, y(8) + box_h / 2, "NLL_t")
    arrow(ax, x_p, y(7) - box_h / 2, x_c + 1.2, y(8) + box_h / 2, "NLL_p")

    # --- cross-conditioning arrows (tmax -> precip) ---
    arrow(ax, x_t + box_w / 2, y(3), x_p - box_w / 2, y(3), style="--", color=COL["cond"],
          label="h_t: on-grid tmax params", lx=-1.9, ly=0.28, fs=7.5)
    arrow(ax, x_t + box_w / 2, y(6), x_p - box_w / 2, y(5) - 0.2, style="--", color=COL["cond"],
          rad=-0.25, label="predicted tmax mean", lx=-1.6, ly=-0.35, fs=7.5)

    # --- shared static inputs (centre, feeding both heads) ---
    draw_box(ax, x_c, y(4), 2.6, 0.9, "dists", ["(N_pts, lat, lon)"], COL["side"],
             COL["side_edge"], title_size=9)
    arrow(ax, x_c - 1.3, y(4), x_t + box_w / 2, y(4), style="--", color=COL["side_edge"])
    arrow(ax, x_c + 1.3, y(4), x_p - box_w / 2, y(4), style="--", color=COL["side_edge"])
    elev_lbl = "elev (+ seasonal)" if seasonal else "elev"
    draw_box(ax, x_c, y(5), 2.6, 0.9, elev_lbl, ["(N_pts, 3) DEM/TPI"], COL["side"],
             COL["side_edge"], title_size=9)
    arrow(ax, x_c - 1.3, y(5), x_t + box_w / 2, y(5), style="--", color=COL["side_edge"])
    arrow(ax, x_c + 1.3, y(5), x_p - box_w / 2, y(5), style="--", color=COL["side_edge"])

    cond_handle = Patch(facecolor="none", edgecolor=COL["cond"],
                        label="tmax -> precip conditioning")
    color_legend(ax, 1.0, top / (top + 1.4), extra=cond_handle)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path, total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=None,
                    help="Trained-model dir; reads its params.json for an exact match.")
    ap.add_argument("--variable", default="tmax", choices=["tmax", "precip"])
    ap.add_argument("--distribution", default=None,
                    choices=list(model_factory.LIKELIHOODS),
                    help="Override the per-variable default distribution.")
    ap.add_argument("--in-channels", type=int, default=5,
                    help="IN_CHANNELS to assume when not reading a run (default 5).")
    ap.add_argument("--encoder", default="flat", help="Encoder type (default flat).")
    ap.add_argument("--no-seasonal", action="store_true",
                    help="Drop seasonal features from the elevation MLP.")
    ap.add_argument("--output", default=None, help="Output image path.")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    # Joint models (JointConditionalConvCNP) have their own builder and a
    # branching diagram; detect them by the joint_meta.json a joint run writes.
    if args.model_dir and (Path(args.model_dir) / "joint_meta.json").exists():
        import train_joint
        p = Params.load_json(Path(args.model_dir) / "params.json")
        meta = json.load(open(Path(args.model_dir) / "joint_meta.json"))
        in_ch = meta.get("in_channels") or p.IN_CHANNELS
        if getattr(p, "IN_CHANNELS", 0) <= 0:
            p = p.with_in_channels(in_ch)
        model = train_joint.build_joint_model(p, in_ch)
        model.eval()
        # Store the graph in the model's own result folder by default.
        out = (Path(args.output) if args.output
               else Path(args.model_dir) / "model_architecture.png")
        out, total = render_joint(model, p, out, args.dpi)
        print(f"model     : {type(model).__name__} (joint)")
        print(f"heads     : tmax (gaussian/gll) + precip (bernoulli_gamma/gamma_ll)")
        print(f"trainable : {total:,} params")
        print(f"written   : {out}")
        return

    params = build_params(args)
    distribution = model_factory.resolve_distribution(params)
    spec = model_factory.LIKELIHOODS[distribution]
    finetune = detect_finetune(args.model_dir, params)
    model, loss_fn, _ = model_factory.build_model(params)
    model.eval()

    # Default into the model's result folder when evaluating a trained run;
    # otherwise fall back to docs/ for ad-hoc from-config renders.
    if args.output:
        out = Path(args.output)
    elif args.model_dir:
        out = Path(args.model_dir) / "model_architecture.png"
    else:
        out = Path("docs") / f"model_architecture_{params.VARIABLE}_{distribution}.png"
    out, total, stages = render(model, loss_fn, params, distribution, spec, out,
                                args.dpi, finetune=finetune)

    print(f"distribution : {distribution}")
    print(f"loss         : {_loss_name(loss_fn)}")
    if finetune:
        print(f"fine-tuned   : warm start from {finetune['src']} "
              f"({finetune['src_loss']}, lr {finetune['src_lr']})")
    print(f"stages       : {' -> '.join(s[0] for s in stages)}")
    print(f"trainable    : {total:,} params")
    print(f"written      : {out}")


if __name__ == "__main__":
    main()
