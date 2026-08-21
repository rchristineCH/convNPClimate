#!/usr/bin/env python
"""Sample inference on the NLL precip model: what (rho, alpha, beta) does it actually emit?

One fold only, so this is a sample rather than the full CV pass. Writes a small JSON of
percentiles so the illustrative parameters in figure 17 can be checked against the real
ranges. Deliberately calls predict_all_folds directly rather than eval_precip's CLI, which
would overwrite the model's metrics JSON and figures.

params_full spans the whole time axis even when one fold is requested, so day_mask is what
restricts this to the days fold 0 actually predicted.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/marc/convNPClimate")
sys.path.insert(0, str(ROOT))

import eval_precip as ep  # noqa: E402

MODEL = (ROOT / "CLEAN_trained_models"
         / "clean_solo_precip__precip_atm_natg_flat_y2020-2023_e30f5_b8" / "precip")
OUT = ROOT / "docs" / "diagrams" / "precip_param_percentiles.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}", flush=True)

B = ep.predict_all_folds(MODEL, device, folds=[0])
day = np.asarray(B.day_mask)
params = np.asarray(B.params_full)[day]        # (fold days, P, 3)
truth = np.asarray(B.truth_full)[day]
print(f"predicted days={int(day.sum())} of {day.size}; params {params.shape}", flush=True)

valid = np.isfinite(truth.reshape(-1))
flat = params.reshape(-1, params.shape[-1])[valid]
rho, alpha, beta = flat[:, 0], flat[:, 1], flat[:, 2]
wet_mean = alpha / beta

qs = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
summary = {"n": int(flat.shape[0]), "n_days": int(day.sum()), "percentiles": qs}
for name, arr in (("rho", rho), ("alpha", alpha), ("beta", beta),
                  ("wet_mean_alpha_over_beta", wet_mean)):
    v = np.percentile(arr, qs)
    summary[name] = {"min": float(arr.min()), "max": float(arr.max()),
                     "mean": float(arr.mean()), "pct": [float(x) for x in v]}
    print(f"\n{name}: min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}",
          flush=True)
    print("  " + "  ".join(f"p{q}={x:.3f}" for q, x in zip(qs, v)), flush=True)

summary["frac_alpha_gt_1"] = float((alpha > 1.0).mean())
summary["frac_rho_gt_half"] = float((rho > 0.5).mean())
print(f"\nfraction alpha > 1 (interior mode): {summary['frac_alpha_gt_1']:.4f}", flush=True)
print(f"fraction rho > 0.5:                {summary['frac_rho_gt_half']:.4f}", flush=True)
OUT.write_text(json.dumps(summary, indent=1))
print(f"\nwrote {OUT}", flush=True)
