"""Fine-tune a trained NLL precip model on the Bernoulli-Gamma CRPS loss.

Rationale (see precip_processing_comparison/): trained from scratch, the NLL
model wins the standard-event regime (bulk calibration, wet-day frequency, PIT)
while the CRPS model wins heavy-precip events (BSS at 40/80 mm, intensity-
category RPSS) -- and their P98 tail biases bracket the observations from
opposite sides. A short, low-LR CRPS fine-tune warm-started from the NLL
checkpoints interpolates between the two optima: it stays near the NLL
solution's calibration while the CRPS gradient reshapes the tail.

Usage:
    python finetune_precip.py \
        --model-dir CLEAN_trained_models/<nll_run>/precip [--lr 5e-5] [--n-epochs 10]

The source run directory is never written to; the fine-tuned run lands in a
sibling directory (default ``<trial>__ft-crps/<variable>``) with its own
``params.json`` / ``metadata.json`` / ``manifest.json`` sidecars, so
``eval_precip.py`` and ``eval_precip_figures.py`` work on it unchanged.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
import torch

import datasets as ds
import model_factory
import params as params_mod
import train  # reuse DataBundle + VARIABLE_SPECS + STATS_HEADER so fine-tuning matches training
import visualization as vis
from convCNP.training.training_elev import train_elev

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("finetune_precip")

FINETUNE_DISTRIBUTION = "bernoulli_gamma_crps"
DEFAULT_LR_SHRINK = 10.0


def finetune(model_dir: Path, out_dir: Optional[Path], lr: Optional[float],
             n_epochs: int, patience: int, folds: Optional[list[int]],
             device: torch.device) -> Path:
    p = params_mod.Params.load_json(model_dir / "params.json")
    src_distribution = model_factory.resolve_distribution(p)
    if src_distribution != "bernoulli_gamma":
        raise ValueError(
            "finetune_precip warm-starts NLL-trained Bernoulli-Gamma models; "
            f"source run has distribution {src_distribution!r}."
        )

    spec = train.VARIABLE_SPECS[p.VARIABLE]

    # Fine-tune params: same data/fold/seed config as the source run (so the
    # fold partition is bit-identical and no held-out day leaks into training),
    # only the loss, LR and schedule change.
    ft_lr = lr if lr is not None else p.LR / DEFAULT_LR_SHRINK
    trial_name = f"{p.TRIAL_NAME}__ft-crps"
    p_ft = dataclasses.replace(
        p,
        DISTRIBUTION=FINETUNE_DISTRIBUTION,
        LR=ft_lr,
        N_EPOCHS=n_epochs,
        PATIENCE=patience,
        TRIAL_NAME=trial_name,
        DEVICE=str(device),
    )

    if out_dir is None:
        out_dir = model_dir.parent.parent / trial_name / p.VARIABLE
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("fine-tuning %s -> %s (lr=%g, n_epochs=%d, patience=%d)",
                model_dir, out_dir, ft_lr, n_epochs, patience)

    # Sidecars: params for the fine-tune run, metadata verbatim, manifest with
    # the distribution key updated -- this is what makes eval_precip.py and
    # eval_precip_figures.py work on the fine-tuned dir with no changes.
    p_ft.save_json(out_dir / "params.json")
    shutil.copy2(model_dir / "metadata.json", out_dir / "metadata.json")
    manifest = json.loads((model_dir / "manifest.json").read_text())
    manifest["distribution"] = FINETUNE_DISTRIBUTION
    manifest["finetuned_from"] = str(model_dir)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Rebuild the data pipeline exactly as training/eval do.
    logger.info("reconstructing data pipeline (variable=%s, encoder=%s)...",
                p.VARIABLE, p.ENCODER)
    data = train.DataBundle(SimpleNamespace(base_params=p_ft, device=device))
    meteoswiss_glob = getattr(p_ft, spec.meteoswiss_glob_attr)
    target_x, target_y, target_topo = ds.prepare_meteoswiss_targets(
        meteoswiss_glob,
        normalization_stats=data.era5_metadata,
        data_var=spec.data_var,
        grid_elevation=data.grid_elevation if p_ft.USE_ELEVATION else None,
        hi_res_elevation=data.hi_res_elevation if p_ft.USE_ELEVATION else None,
        hi_res_tpi=data.hi_res_tpi if p_ft.USE_MTPI else None,
        convert_to_kelvin=spec.convert_to_kelvin,
        normalize_targets=spec.normalize_targets,
        year_start=p_ft.DATA_YEAR_START,
        year_end=p_ft.DATA_YEAR_END,
        device=device,
    )
    # Match the trainer: the context axis can be a trimmed subset of the year range
    # (a source whose series stops short, e.g. ERA5-Land precip), so select the
    # target on the context's days by date rather than assuming both are equal.
    target_y = ds.align_target_to_days(
        target_y, data.time_coords, f"{p_ft.VARIABLE} fine-tune context")
    dists = ds.calculate_dists_meteoswiss(data.dists_metadata, target_x, device=device)
    target_tensor = torch.from_numpy(target_y.values.astype(np.float32)).to(device)
    if target_tensor.shape[0] != data.context.shape[0]:
        raise ValueError(
            f"target days ({target_tensor.shape[0]}) != context days "
            f"({data.context.shape[0]}) for variable {p_ft.VARIABLE!r}.")

    channel_groups = data.channel_groups if p_ft.ENCODER != "flat" else None
    # loss_fn comes from the factory so CRPS_N_SAMPLES binding matches training.
    _, loss_fn, get_value_fn = model_factory.build_model(p_ft, channel_groups=channel_groups)
    crps_fn = model_factory.build_crps_diagnostic(p_ft)
    grad_clip = spec.grad_clip  # 1.0 for precip; stabilises the Gamma terms

    fold_ids = folds if folds is not None else list(range(p_ft.N_FOLDS))
    fold_times = []
    for fold in fold_ids:
        ckpt = model_dir / f"model_fold_{fold}"
        if not ckpt.exists():
            logger.warning("checkpoint %s missing; skipping fold %d", ckpt, fold)
            continue
        logger.info("[fold %d/%d] warm-starting from %s", fold + 1, p_ft.N_FOLDS, ckpt)

        stats_file = out_dir / f"stats_fold{fold}.csv"
        if not stats_file.exists():
            stats_file.write_text(train.Trainer.STATS_HEADER)

        params_mod.set_seed(p_ft.SEED + fold)
        # load_model_checkpoint handles the DataParallel prefix + IN_CHANNELS
        # drift guard; it returns the model in eval mode, hence .train().
        model, src_epoch = model_factory.load_model_checkpoint(
            ckpt, p_ft, device, channel_groups=channel_groups
        )
        model.train()
        logger.info("[fold %d] source checkpoint epoch %d", fold, src_epoch)

        # Fresh optimizer: the saved Adam moments are NLL-scale, wrong for CRPS.
        optimizer = torch.optim.Adam(model.parameters(), lr=p_ft.LR)

        t0 = time.time()
        train_elev(
            model=model,
            opt=optimizer,
            ll=loss_fn,
            elev=target_topo,
            dists=dists,
            y_context=data.context,
            y_target=target_tensor,
            output_dir=str(out_dir),
            y_target_t=None,
            get_value=get_value_fn,
            fold=fold,
            n_folds=p_ft.N_FOLDS,
            n_epochs=p_ft.N_EPOCHS,
            batch_size=p_ft.BATCH_SIZE,
            patience=p_ft.PATIENCE,
            stats_file=str(stats_file),
            seasonal=data.seasonal_features,
            device=device,
            grad_clip=grad_clip,
            crps_fn=crps_fn,
            # Warm start: the held-out CRPS starts wherever the NLL model left
            # it, so the epoch-0 fine-tune state must always checkpoint.
            init_best_obj=float("inf"),
        )
        fold_times.append(time.time() - t0)

    # Consolidate per-fold CSVs and plot, mirroring train.Trainer.
    fold_csvs = [out_dir / f"stats_fold{f}.csv" for f in range(p_ft.N_FOLDS)]
    frames = [pd.read_csv(c) for c in fold_csvs if c.exists()]
    stats = pd.concat(frames) if frames else pd.DataFrame()
    stats.to_csv(out_dir / "stats.csv", index=False)
    if not stats.empty:
        try:
            vis.plot_training_curves(stats, save_path=str(out_dir / "trainingstats"))
        except Exception as exc:  # plotting must never abort a fine-tune
            logger.warning("plot_training_curves failed: %s", exc)

    logger.info("done: %d fold(s) fine-tuned in %s (total %.0fs)",
                len(fold_times), out_dir, sum(fold_times))
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", required=True,
                    help="Trained NLL run dir with params.json/metadata.json/model_fold_*.")
    ap.add_argument("--out-dir", default=None,
                    help="Output dir (default: sibling <trial>__ft-crps/<variable>).")
    ap.add_argument("--lr", type=float, default=None,
                    help="Fine-tune LR (default: source LR / 10).")
    ap.add_argument("--n-epochs", type=int, default=10,
                    help="Max fine-tune epochs per fold (default 10).")
    ap.add_argument("--patience", type=int, default=3,
                    help="Early-stopping patience on held-out CRPS (default 3; "
                         "short leash keeps the run near the NLL optimum).")
    ap.add_argument("--folds", type=int, nargs="+", default=None,
                    help="Fold subset (default: all folds).")
    ap.add_argument("--device", default=None, help="Override device (e.g. cuda, cpu).")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    params_mod.configure_renku_cuda()
    device = torch.device(args.device) if args.device else params_mod.select_device()
    finetune(
        model_dir=Path(args.model_dir),
        out_dir=Path(args.out_dir) if args.out_dir else None,
        lr=args.lr,
        n_epochs=args.n_epochs,
        patience=args.patience,
        folds=args.folds,
        device=device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
