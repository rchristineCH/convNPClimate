#!/usr/bin/env python3
"""Render the pipeline diagrams to static SVG + PNG via Graphviz `dot`.

No mermaid / node / browser needed — the outputs open in VS Code's built-in
image preview, any image viewer, or a browser. Run:

    python scripts/gen_pipeline_diagrams.py

Outputs land in docs/diagrams/. Companion prose: docs/data_pipelines.md.
"""
import subprocess, sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "diagrams"

# role palette (matches the pipeline atlas legend)
ROLE = {
    "data":  ('#e6eef8', '#2560a6', '#173a63'),
    "model": ('#e4f1ea', '#2c8b58', '#1c4f33'),
    "train": ('#f6ecd9', '#b9791c', '#6b470f'),
    "down":  ('#ece5f6', '#7154a8', '#3f2c63'),
    "io":    ('#eef1f4', '#8a97a3', '#3a4650'),
}

def node(nid, label, role="io", shape="box"):
    fill, stroke, ink = ROLE[role]
    lbl = label.replace('"', r'\"')
    return (f'  {nid} [label="{lbl}", shape={shape}, style="rounded,filled", '
            f'fillcolor="{fill}", color="{stroke}", fontcolor="{ink}", '
            f'penwidth=1.5];')

def graph(name, title, rankdir, body):
    header = (
        f'digraph "{name}" {{\n'
        f'  bgcolor="#ffffff";\n'
        f'  rankdir={rankdir};\n'
        f'  labelloc="t"; fontname="Helvetica"; fontsize=15; fontcolor="#131c24";\n'
        f'  label="{title}";\n'
        f'  node [fontname="Helvetica", fontsize=11, margin="0.16,0.09"];\n'
        f'  edge [color="#7d8b98", penwidth=1.3, arrowsize=0.8];\n'
    )
    return header + body + "\n}\n"

# ---------------------------------------------------------------- diagrams
DIAGRAMS = {}

DIAGRAMS["00_master"] = graph("master", "convNPClimate — all pipelines", "LR", "\n".join([
    node("IN", "ERA5 context\\n(surface / atmospheric)", "data"),
    node("MS", "MeteoSwiss + DEM/TPI", "data"),
    node("CTX", "context grid", "data"),
    node("TG", "targets + dists", "data"),
    node("ENC", "encoder → CNN → MLP", "model"),
    node("FL", "RBF grid→point", "model"),
    node("EL", "elevation MLP", "model"),
    node("HEAD", "output head", "model", shape="diamond"),
    node("GA", "Gaussian μ, σ", "model"),
    node("BG", "Bernoulli-Gamma ρ, α, β", "model"),
    node("TR", "k-fold train\\nNLL / CRPS", "train"),
    node("DS", "evaluate · eval_precip\\npredict · feature_importance", "down"),
    "  IN->CTX; MS->TG; CTX->ENC->FL->EL->HEAD; TG->FL;",
    '  HEAD->GA [label="tmax", fontsize=9, fontcolor="#57697a"];',
    '  HEAD->BG [label="precip", fontsize=9, fontcolor="#57697a"];',
    "  GA->TR; BG->TR; TR->DS;",
]))

DIAGRAMS["s0_backbone"] = graph("s0", "§0  Shared training backbone", "TB", "\n".join([
    node("CTX", "input context\\n(1461, C, H, W)", "data"),
    node("RAW", "MeteoSwiss + DEM/TPI", "data"),
    node("TGT", "prepare_meteoswiss_targets\\ndatasets.py:728", "data"),
    node("X", "x locations (88800,2)", "data"),
    node("Y", "y truth (1461,88800)", "data"),
    node("E3", "e topo (88800,3)", "data"),
    node("DIST", "calculate_dists_meteoswiss\\ndatasets.py:932  → (88800,H,W)", "data"),
    node("FOLD", "get_fold_data (per epoch)\\ntraining/utils.py:26", "train"),
    node("FWD", "model(x, mask, dists, elev, seasonal)", "model"),
    node("LOSS", "obj = -ll(y, v)  → backward → Adam", "train"),
    "  CTX->TGT; RAW->TGT; TGT->X; TGT->Y; TGT->E3;",
    "  CTX->DIST; X->DIST; CTX->FOLD; Y->FOLD;",
    "  FOLD->FWD; DIST->FWD; E3->FWD; FWD->LOSS;",
]))

DIAGRAMS["s1_surface"] = graph("s1", "§1  Surface tmax  (--use-surface)", "TB", "\n".join([
    node("A", "ERA5-Land t2m_max /*.nc\\n(65 yearly files)", "data"),
    node("G", "ERA5-Land geopotential", "data"),
    node("L1", "load_era5_data (datasets.py:184)\\nsnap coords → clean 29x61", "data"),
    node("C1", "context (1461, C, 29, 61)\\nC=6  (5 with --no-geopotential)", "data"),
    node("BK", "§0 backbone", "train"),
    "  A->L1; G->L1; L1->C1; C1->BK;",
]))

DIAGRAMS["s2_atmospheric"] = graph("s2", "§2  Atmospheric tmax  (--use-atmospheric)", "TB", "\n".join([
    node("PL", "load_era5_pressure_levels\\nnative_grid=True → 90 ch + coarse lat/lon", "data"),
    node("SCAF", "coarse scaffold\\nlat, lon, cos_time, sin_time, elevation", "data"),
    node("SFC", "optional surface anchors (10 ch)\\n--use-sfc-atmos", "data"),
    node("CAT", "torch.cat → context", "data"),
    node("OUT", "(1461, 95, coarse)\\ndists_metadata = coarse", "data"),
    node("BK", "§0 backbone", "train"),
    "  PL->SCAF; PL->CAT; SCAF->CAT; SFC->CAT [style=dashed]; CAT->OUT->BK;",
]))

DIAGRAMS["s3_precip"] = graph("s3", "§3  Precip  (Bernoulli-Gamma; NLL / CRPS)", "TB", "\n".join([
    node("H", "ERA5 context (B, C, lat, lon)", "data"),
    node("ENC", "encoder → CNN → MLP(128→3)", "model"),
    node("GFL", "GammaFinalLayer\\n3 ParamLayers (RBF grid→point)", "model"),
    node("RAB", "ρ=sigmoid;  α,β=force_positive+clamp", "model"),
    node("EMLP", "elev_mlp  cat[3 params,3 elev,2 seasonal]=8 → 3", "model"),
    node("ACT", "out0=clamp(sigmoid);  out1:=force_positive", "model"),
    node("L1", "gamma_bernoulli  (NLL, default)", "train"),
    node("L2", "gamma_bernoulli_crps  (opt-in)", "train"),
    node("T", "targets: raw mm, y≥0", "data"),
    "  H->ENC->GFL->RAB->EMLP->ACT; ACT->L1; ACT->L2; T->L1; T->L2;",
]))

DIAGRAMS["s4_evaluate"] = graph("s4", "§4  evaluate.py  (tmax holdout)", "TB", "\n".join([
    node("MAN", "load_manifest + Params.load_json\\nmanifest_to_dists_metadata", "down"),
    node("CTX", "build_*_context (T,C,lat,lon)", "data"),
    node("TGT", "targets + truth", "data"),
    node("DIST", "dists", "data"),
    node("FOLDS", "per fold: load ckpt + predict_all_days", "down"),
    node("ENS", "ensemble: mean μ, σ=√(within+between)", "down"),
    node("DEN", "denormalize → degC", "down"),
    node("REF", "ERA5 bilinear reference", "data"),
    node("MET", "metrics + PIT calibration", "down"),
    node("OUT", "report_metrics.json + report.md + maps", "down"),
    "  MAN->CTX; MAN->TGT->DIST; CTX->FOLDS; DIST->FOLDS->ENS->DEN->MET;",
    "  REF->MET; MET->OUT;",
]))

DIAGRAMS["s5_eval_precip"] = graph("s5", "§5  eval_precip.py  (physical metrics)", "TB", "\n".join([
    node("LP", "Params + resolve_distribution\\nguard in _SAMPLERS", "down"),
    node("BUN", "train.DataBundle", "data"),
    node("TGT", "targets RhiresD, raw mm", "data"),
    node("DIST", "dists", "data"),
    node("FL", "per fold: holdout + predict_params_fold → (T,P,3)", "down"),
    node("MET", "compute_precip_metrics (+ sampling)", "down"),
    node("PIT", "randomized PIT", "down"),
    node("BASE", "bilinear ERA5 tp skill", "down"),
    node("OUT", "eval_precip_metrics.json", "down"),
    "  LP->BUN->TGT->DIST->FL; FL->MET; FL->PIT; FL->BASE; MET->OUT;",
]))

DIAGRAMS["s6_predict"] = graph("s6", "§6  predict.py  (inference)", "TB", "\n".join([
    node("MAN", "load_manifest + Params", "down"),
    node("CTX", "build_*_context (T,C,lat,lon)", "data"),
    node("TGT", "MeteoSwiss / custom targets", "data"),
    node("DIST", "dists", "data"),
    node("LOOP", "per fold: predict_all_days", "down"),
    node("ENS", "ensemble mean + within/between σ", "down"),
    node("DEN", "denormalize → degC", "down"),
    node("SAVE", "np.savez_compressed(.npz)", "down"),
    "  MAN->CTX; MAN->TGT->DIST; CTX->LOOP; DIST->LOOP->ENS->DEN->SAVE;",
]))

DIAGRAMS["s7_feature_importance"] = graph("s7", "§7  feature_importance.py  (PFI / SHAP / LIME)", "TB", "\n".join([
    node("BB", "_build_backbone\\ncontext, dists, truth, folds, mean_field", "down"),
    node("BASE", "baseline = score()", "down"),
    node("SCORE", "score(context): fold-ensemble forward\\npredict_all_days → metrics degC", "down"),
    node("PFI", "run_pfi: permute channel/group over days", "down"),
    node("SHAP", "run_shap: KernelSHAP + exact Shapley", "down"),
    node("LIME", "run_lime: weighted Ridge surrogate", "down"),
    node("OUT", "feature_importance/{method}.csv + PNGs + summary.md", "down"),
    "  BB->BASE->SCORE; PFI->SCORE; SHAP->SCORE; LIME->SCORE; SCORE->OUT;",
    "  PFI->OUT; SHAP->OUT; LIME->OUT;",
]))

# ---- one comprehensive tmax diagram: surface (top) + atmospheric (bottom) inputs
#      feeding a shared ConvCNP, with each architecture stage boxed as its own zone.
DIAGRAMS["10_tmax_full_architecture"] = r'''digraph tmax_full {
  bgcolor="#ffffff"; rankdir=LR; compound=true; newrank=true;
  labelloc="t"; fontname="Helvetica"; fontsize=16; fontcolor="#131c24";
  labeljust="l";
  label=<<FONT POINT-SIZE="11" COLOR="#57697a"><B>Shapes</B><BR ALIGN="LEFT"/>B = batch (days, 8)<BR ALIGN="LEFT"/>C = channels (6 surface / 95 atmospheric)<BR ALIGN="LEFT"/>lat × lon = ERA5 context grid (29×61 surface)<BR ALIGN="LEFT"/>P = 88,800 target points<BR ALIGN="LEFT"/></FONT>>;
  nodesep=0.3; ranksep=0.6;
  node [fontname="Helvetica", fontsize=10, shape=box, style="rounded,filled", margin="0.15,0.08", penwidth=1.4];
  edge [color="#7d8b98", penwidth=1.3, arrowsize=0.8];

  subgraph cluster_surf {
    label="Surface input  (--use-surface)"; labeljust="l"; style="rounded,filled"; fillcolor="#f1f6fc"; color="#2560a6"; fontcolor="#173a63"; fontsize=11;
    s_in [label="ERA5-Land t2m_max\n+ geopotential", fillcolor="#e6eef8", color="#2560a6", fontcolor="#173a63"];
    s_ld [label="load_era5_data\nsnap → clean 29×61", fillcolor="#e6eef8", color="#2560a6", fontcolor="#173a63"];
    s_cx [label="surface context\n(B, 6, 29, 61)", fillcolor="#e6eef8", color="#2560a6", fontcolor="#173a63"];
    s_in -> s_ld -> s_cx;
  }
  subgraph cluster_atm {
    label="Atmospheric input  (--use-atmospheric)"; labeljust="l"; style="rounded,filled"; fillcolor="#f1f6fc"; color="#2560a6"; fontcolor="#173a63"; fontsize=11;
    a_in [label="ERA5 pressure levels z,t,q\n(6 levels × 5 hours) + anchors", fillcolor="#e6eef8", color="#2560a6", fontcolor="#173a63"];
    a_bd [label="build native / regrid\nper-channel z-score", fillcolor="#e6eef8", color="#2560a6", fontcolor="#173a63"];
    a_cx [label="atmospheric context\n(B, 95, grid)", fillcolor="#e6eef8", color="#2560a6", fontcolor="#173a63"];
    a_in -> a_bd -> a_cx;
  }

  ctx [label="context\n(B, C, lat, lon)", fillcolor="#dde6ef", color="#8a97a3", fontcolor="#2b3640"];
  s_cx -> ctx;
  a_cx -> ctx;

  subgraph cluster_enc {
    label="Encoder — depthwise SetConv"; labeljust="l"; style="rounded,filled"; fillcolor="#edf6f0"; color="#2c8b58"; fontcolor="#1c4f33"; fontsize=11;
    enc [label="conv(x·mask) / conv(mask)\n+ density confidence → 128", fillcolor="#e4f1ea", color="#2c8b58", fontcolor="#1c4f33"];
  }
  subgraph cluster_cnn {
    label="CNN decoder — ResNet × 6"; labeljust="l"; style="rounded,filled"; fillcolor="#edf6f0"; color="#2c8b58"; fontcolor="#1c4f33"; fontsize=11;
    cnn [label="6 × ResConvBlock\nwidth 128", fillcolor="#e4f1ea", color="#2c8b58", fontcolor="#1c4f33"];
  }
  subgraph cluster_gmlp {
    label="Grid MLP"; labeljust="l"; style="rounded,filled"; fillcolor="#edf6f0"; color="#2c8b58"; fontcolor="#1c4f33"; fontsize=11;
    gmlp [label="MLP 128 → 2\n(μ, σ per grid cell)", fillcolor="#e4f1ea", color="#2c8b58", fontcolor="#1c4f33"];
  }
  subgraph cluster_rbf {
    label="RBF final layer — grid → point"; labeljust="l"; style="rounded,filled"; fillcolor="#edf6f0"; color="#2c8b58"; fontcolor="#1c4f33"; fontsize=11;
    rbf [label="ParamLayer × 2\nexp(−0.5·dists/ls²) → 88,800 pts", fillcolor="#e4f1ea", color="#2c8b58", fontcolor="#1c4f33"];
  }
  subgraph cluster_emlp {
    label="Elevation + seasonal MLP"; labeljust="l"; style="rounded,filled"; fillcolor="#edf6f0"; color="#2c8b58"; fontcolor="#1c4f33"; fontsize=11;
    emlp [label="MLP 7 → 2\n+ topo(3) + seasonal(2)", fillcolor="#e4f1ea", color="#2c8b58", fontcolor="#1c4f33"];
  }

  ctx -> enc -> cnn -> gmlp -> rbf -> emlp -> out;

  out  [label="Gaussian μ, σ\n@ 88,800 target points", fillcolor="#e4f1ea", color="#2c8b58", fontcolor="#1c4f33"];
  loss [label="gll  (Gaussian NLL)", fillcolor="#f6ecd9", color="#b9791c", fontcolor="#6b470f"];
  out -> loss;

  tgt   [label="MeteoSwiss targets\nx, e=[elev, Δelev, TPI]", fillcolor="#e6eef8", color="#2560a6", fontcolor="#173a63"];
  dists [label="dists (P, lat, lon)", fillcolor="#eef1f4", color="#8a97a3", fontcolor="#2b3640"];
  tgt -> dists -> rbf;
  tgt -> emlp [label="e", fontsize=9, fontcolor="#57697a"];
}
'''


DIAGRAMS["11_tmax_full_architecture_vertical"] = r'''digraph tmax_full_v {
  bgcolor="#ffffff"; rankdir=TB; compound=true; newrank=true;
  labelloc="t"; fontname="Helvetica"; fontsize=16; fontcolor="#131c24";
  labeljust="l";
  label=<<FONT POINT-SIZE="11" COLOR="#57697a"><B>Shapes</B><BR ALIGN="LEFT"/>B = batch (days, 8)<BR ALIGN="LEFT"/>C = channels (6 surface / 95 atmospheric)<BR ALIGN="LEFT"/>lat × lon = ERA5 context grid (29×61 surface)<BR ALIGN="LEFT"/>P = 88,800 target points<BR ALIGN="LEFT"/></FONT>>;
  nodesep=0.35; ranksep=0.5;
  node [fontname="Helvetica", fontsize=10, shape=box, style="rounded,filled", margin="0.15,0.08", penwidth=1.4];
  edge [color="#7d8b98", penwidth=1.3, arrowsize=0.8];

  subgraph cluster_surf {
    label="Surface input  (--use-surface)"; labeljust="l"; style="rounded,filled"; fillcolor="#f1f6fc"; color="#2560a6"; fontcolor="#173a63"; fontsize=11;
    s_in [label="ERA5-Land t2m_max\n+ geopotential", fillcolor="#e6eef8", color="#2560a6", fontcolor="#173a63"];
    s_ld [label="load_era5_data\nsnap → clean 29×61", fillcolor="#e6eef8", color="#2560a6", fontcolor="#173a63"];
    s_cx [label="surface context\n(B, 6, 29, 61)", fillcolor="#e6eef8", color="#2560a6", fontcolor="#173a63"];
    s_in -> s_ld -> s_cx;
  }
  subgraph cluster_atm {
    label="Atmospheric input  (--use-atmospheric)"; labeljust="l"; style="rounded,filled"; fillcolor="#f1f6fc"; color="#2560a6"; fontcolor="#173a63"; fontsize=11;
    a_in [label="ERA5 pressure levels z,t,q\n(6 levels × 5 hours) + anchors", fillcolor="#e6eef8", color="#2560a6", fontcolor="#173a63"];
    a_bd [label="build native / regrid\nper-channel z-score", fillcolor="#e6eef8", color="#2560a6", fontcolor="#173a63"];
    a_cx [label="atmospheric context\n(B, 95, grid)", fillcolor="#e6eef8", color="#2560a6", fontcolor="#173a63"];
    a_in -> a_bd -> a_cx;
  }

  ctx [label="context  (B, C, lat, lon)", fillcolor="#dde6ef", color="#8a97a3", fontcolor="#2b3640"];
  s_cx -> ctx;
  a_cx -> ctx;

  subgraph cluster_enc {
    label="Encoder — depthwise SetConv"; labeljust="l"; style="rounded,filled"; fillcolor="#edf6f0"; color="#2c8b58"; fontcolor="#1c4f33"; fontsize=11;
    enc [label="conv(x·mask) / conv(mask)  + density confidence → 128", fillcolor="#e4f1ea", color="#2c8b58", fontcolor="#1c4f33"];
  }
  subgraph cluster_cnn {
    label="CNN decoder — ResNet × 6"; labeljust="l"; style="rounded,filled"; fillcolor="#edf6f0"; color="#2c8b58"; fontcolor="#1c4f33"; fontsize=11;
    cnn [label="6 × ResConvBlock  ·  width 128", fillcolor="#e4f1ea", color="#2c8b58", fontcolor="#1c4f33"];
  }
  subgraph cluster_gmlp {
    label="Grid MLP"; labeljust="l"; style="rounded,filled"; fillcolor="#edf6f0"; color="#2c8b58"; fontcolor="#1c4f33"; fontsize=11;
    gmlp [label="MLP 128 → 2   (μ, σ per grid cell)", fillcolor="#e4f1ea", color="#2c8b58", fontcolor="#1c4f33"];
  }
  subgraph cluster_rbf {
    label="RBF final layer — grid → point"; labeljust="l"; style="rounded,filled"; fillcolor="#edf6f0"; color="#2c8b58"; fontcolor="#1c4f33"; fontsize=11;
    rbf [label="ParamLayer × 2   exp(−0.5·dists/ls²) → 88,800 pts", fillcolor="#e4f1ea", color="#2c8b58", fontcolor="#1c4f33"];
  }
  subgraph cluster_emlp {
    label="Elevation + seasonal MLP"; labeljust="l"; style="rounded,filled"; fillcolor="#edf6f0"; color="#2c8b58"; fontcolor="#1c4f33"; fontsize=11;
    emlp [label="MLP 7 → 2   + topo(3) + seasonal(2)", fillcolor="#e4f1ea", color="#2c8b58", fontcolor="#1c4f33"];
  }

  ctx -> enc -> cnn -> gmlp -> rbf -> emlp -> out;
  out  [label="Gaussian μ, σ  @ 88,800 target points", fillcolor="#e4f1ea", color="#2c8b58", fontcolor="#1c4f33"];
  loss [label="gll  (Gaussian NLL)", fillcolor="#f6ecd9", color="#b9791c", fontcolor="#6b470f"];
  out -> loss;

  tgt   [label="MeteoSwiss targets\nx, e=[elev, Δelev, TPI]", fillcolor="#e6eef8", color="#2560a6", fontcolor="#173a63"];
  dists [label="dists (P, lat, lon)", fillcolor="#eef1f4", color="#8a97a3", fontcolor="#2b3640"];
  tgt -> dists -> rbf;
  tgt -> emlp [label="e", fontsize=9, fontcolor="#57697a"];
  { rank=same; ctx; tgt; }
}
'''


# ---- temperature (tmax) model head — counterpart to the §3 precip head, styled like #11
_SHAPE_LEGEND = ('<<FONT POINT-SIZE="11" COLOR="#57697a"><B>Shapes</B>'
  '<BR ALIGN="LEFT"/>B = batch (days, 8)'
  '<BR ALIGN="LEFT"/>C = channels (6 surface / 95 atmospheric)'
  '<BR ALIGN="LEFT"/>lat × lon = ERA5 context grid (29×61 surface)'
  '<BR ALIGN="LEFT"/>P = 88,800 target points<BR ALIGN="LEFT"/></FONT>>')

DIAGRAMS["13_tmax_head"] = (
  'digraph tmax_head {\n'
  '  bgcolor="#ffffff"; rankdir=TB;\n'
  '  labelloc="t"; labeljust="l"; fontname="Helvetica";\n'
  f'  label={_SHAPE_LEGEND};\n'
  '  node [fontname="Helvetica", fontsize=11, margin="0.16,0.09"];\n'
  '  edge [color="#7d8b98", penwidth=1.3, arrowsize=0.8];\n'
  + "\n".join([
      node("H",    "ERA5 context (B, C, lat, lon)", "data"),
      node("ENC",  "encoder → CNN → MLP(128→2)", "model"),
      node("GFL",  "GaussianFinalLayer\\n2 ParamLayers (RBF grid→point)", "model"),
      node("MS",   "μ = raw;   σ = force_positive", "model"),
      node("EMLP", "elev_mlp  cat[2 params, 3 elev, 2 seasonal]=7 → 2", "model"),
      node("ACT",  "out0 = μ (free);   out1 := force_positive (σ)", "model"),
      node("L1",   "gll   (Gaussian NLL)", "train"),
      node("T",    "targets: z-scored Kelvin", "data"),
      "  H -> ENC -> GFL -> MS -> EMLP -> ACT; ACT -> L1; T -> L1;",
  ])
  + "\n}\n"
)


DIAGRAMS["14_both_heads"] = (
  'digraph both_heads {\n'
  '  bgcolor="#ffffff"; rankdir=TB; compound=true;\n'
  '  labelloc="t"; labeljust="l"; fontname="Helvetica";\n'
  f'  label={_SHAPE_LEGEND};\n'
  '  node [fontname="Helvetica", fontsize=10.5, margin="0.15,0.08"];\n'
  '  edge [color="#7d8b98", penwidth=1.3, arrowsize=0.8];\n'
  # shared trunk
  + node("ctx", "ERA5 context (B, C, lat, lon)", "data") + "\n"
  + node("enc", "encoder — depthwise SetConv → 128", "model") + "\n"
  + node("cnn", "CNN decoder — ResNet × 6  (128)", "model") + "\n"
  + "  ctx -> enc -> cnn;\n"
  # temperature head
  + '  subgraph cluster_t {\n'
  + '    label="Temperature head — Gaussian"; labeljust="l"; style="rounded,filled"; '
    'fillcolor="#fbeee9"; color="#c0392b"; fontcolor="#7a241a"; fontsize=12;\n'
  + node("t_mlp", "grid MLP   128 → 2", "model") + "\n"
  + node("t_fl", "GaussianFinalLayer\\n2 ParamLayers (RBF grid→point)", "model") + "\n"
  + node("t_ms", "μ = raw;   σ = force_positive", "model") + "\n"
  + node("t_el", "elev_mlp   cat[2, 3, 2] = 7 → 2", "model") + "\n"
  + node("t_act", "out0 = μ (free);   out1 := force_positive (σ)", "model") + "\n"
  + node("t_tgt", "targets: z-scored Kelvin", "data") + "\n"
  + node("t_loss", "gll   (Gaussian NLL)", "train") + "\n"
  + "    t_mlp -> t_fl -> t_ms -> t_el -> t_act -> t_loss; t_tgt -> t_loss;\n"
  + "  }\n"
  # precipitation head
  + '  subgraph cluster_p {\n'
  + '    label="Precipitation head — Bernoulli-Gamma"; labeljust="l"; style="rounded,filled"; '
    'fillcolor="#e9f0fb"; color="#2560a6"; fontcolor="#173a63"; fontsize=12;\n'
  + node("p_mlp", "grid MLP   128 → 3", "model") + "\n"
  + node("p_fl", "GammaFinalLayer\\n3 ParamLayers (RBF grid→point)", "model") + "\n"
  + node("p_ra", "ρ = sigmoid;   α, β = force_positive + clamp", "model") + "\n"
  + node("p_el", "elev_mlp   cat[3, 3, 2] = 8 → 3", "model") + "\n"
  + node("p_act", "out0 = clamp(sigmoid, ρ);   out1 := force_positive", "model") + "\n"
  + node("p_tgt", "targets: raw mm, y ≥ 0", "data") + "\n"
  + node("p_l1", "gamma_bernoulli   (NLL, default)", "train") + "\n"
  + node("p_l2", "gamma_bernoulli_crps   (opt-in)", "train") + "\n"
  + "    p_mlp -> p_fl -> p_ra -> p_el -> p_act; p_act -> p_l1; p_act -> p_l2; "
    "p_tgt -> p_l1; p_tgt -> p_l2;\n"
  + "  }\n"
  + "  cnn -> t_mlp; cnn -> p_mlp;\n"
  + "}\n"
)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ok = 0
    for name, dot in DIAGRAMS.items():
        for fmt in ("svg", "png"):
            dst = OUT / f"{name}.{fmt}"
            p = subprocess.run(["dot", f"-T{fmt}", "-o", str(dst)],
                               input=dot.encode(), capture_output=True)
            if p.returncode != 0:
                print(f"FAIL {dst}: {p.stderr.decode()[:200]}", file=sys.stderr)
                return 1
        ok += 1
        print(f"  rendered {name}.svg + .png")
    print(f"Done: {ok} diagrams → {OUT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
