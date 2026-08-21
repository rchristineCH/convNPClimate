"""
Shared model construction logic for convCNP climate models.

This module provides a single source of truth for building the model architecture,
used by both training and prediction notebooks.
"""

import dataclasses
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from params import Params
from convCNP.models.elev_models import (
    TmaxBiasConvCNPElev,
    GammaBiasConvCNPElev,
)
from convCNP.models.cnn import CNN, ResConvBlock
from convCNP.models.grouped_encoder import build_grouped_encoder
from functools import partial

from convCNP.training.loss_functions import gll, gamma_ll, gamma_bernoulli_crps
from convCNP.training.utils import get_value_tmax


LossFn = Callable
GetValueFn = Callable


# Deprecated. The evaluation is now fully probabilistic and no longer classifies
# days wet/dry with a fixed rho threshold; retained only for backward-compat with
# any external caller. See eval_precip.py for the probabilistic definitions.
DRY_PROBABILITY_THRESHOLD = 0.5


def _get_value_precip(p):
    """Point prediction for precipitation: the FULL Bernoulli-Gamma mean.

    E[Y] = rho * E[Gamma] = rho * (alpha / beta) with the rate parameterization
    Gamma(concentration=alpha, rate=beta). No fixed rho>=0.5 wet/dry threshold is
    applied (previously the conditional mean alpha/beta was hard-zeroed where
    rho<=0.5); the full expectation is the natural threshold-free point estimate.
    """
    rho, alpha, beta = p[:, :, 0], p[:, :, 1], p[:, :, 2]
    return rho * (alpha / beta)


@dataclasses.dataclass(frozen=True)
class LikelihoodSpec:
    """Describes one output distribution and how to train/evaluate it.

    The RBF final layer and parameter count live inside ``model_class`` (each
    model wires its own GaussianFinalLayer/GammaFinalLayer); this registry maps a
    distribution key to its model, loss, and point-prediction extractor so adding
    a new distribution is a single entry plus its model class -- no change to the
    training loop. ``n_params`` is recorded for documentation/logging.
    """
    model_class: type
    n_params: int
    loss_fn: LossFn
    get_value_fn: GetValueFn


# Distribution registry. Add a new distribution as a single entry here plus its
# model class (in convCNP/models/elev_models.py) -- no change to the training loop.
LIKELIHOODS: dict[str, LikelihoodSpec] = {
    "gaussian": LikelihoodSpec(TmaxBiasConvCNPElev, 2, gll, get_value_tmax),
    "bernoulli_gamma": LikelihoodSpec(GammaBiasConvCNPElev, 3, gamma_ll, _get_value_precip),
    # Same Bernoulli-Gamma model/parameterization, trained on a sample-based CRPS
    # loss instead of the NLL. The number of MC samples comes from
    # params.CRPS_N_SAMPLES (wired in build_model).
    "bernoulli_gamma_crps": LikelihoodSpec(
        GammaBiasConvCNPElev, 3, gamma_bernoulli_crps, _get_value_precip),
}

# Default mapping from a target variable to its output distribution.
VARIABLE_TO_DISTRIBUTION: dict[str, str] = {
    "tmax": "gaussian",
    "precip": "bernoulli_gamma",
}

# CRPS diagnostic per distribution. This is *always* the same energy-form CRPS
# used by the CRPS loss, exposed as an eval-only metric so precip runs trained
# on either the NLL or the CRPS loss report a common CRPS curve -- the point is
# to trace comparison curves across runs regardless of their training objective.
# Like the loss functions, each entry returns ``-mean(CRPS)`` (score-to-maximize
# convention); callers negate it to get CRPS. Distributions without an entry
# (currently gaussian/tmax) simply report no CRPS.
CRPS_DIAGNOSTICS: dict[str, LossFn] = {
    "bernoulli_gamma": gamma_bernoulli_crps,
    "bernoulli_gamma_crps": gamma_bernoulli_crps,
}


def build_crps_diagnostic(params: Params) -> LossFn | None:
    """Return an eval-only CRPS diagnostic ``fn(target, v) -> -mean(CRPS)``.

    Returns ``None`` for distributions that have no registered CRPS diagnostic
    (e.g. gaussian/tmax), in which case the training loop simply omits CRPS from
    the status line and the stats CSV. For the Bernoulli-Gamma distributions the
    Monte-Carlo sample count is bound from ``params.CRPS_N_SAMPLES`` so the
    diagnostic matches the CRPS loss exactly.
    """
    distribution = resolve_distribution(params)
    fn = CRPS_DIAGNOSTICS.get(distribution)
    if fn is None:
        return None
    if distribution in ("bernoulli_gamma", "bernoulli_gamma_crps"):
        return partial(fn, n_samples=params.CRPS_N_SAMPLES)
    return fn


def resolve_distribution(params: Params) -> str | None:
    """Resolve the output distribution key for a Params.

    An explicit ``params.DISTRIBUTION`` overrides the per-variable default in
    ``VARIABLE_TO_DISTRIBUTION``. Returns ``None`` if neither resolves (callers
    raise a clear error). Use this everywhere the distribution is needed
    (model building, manifest, run summary) so there is one source of truth.
    """
    return getattr(params, "DISTRIBUTION", None) or VARIABLE_TO_DISTRIBUTION.get(
        params.VARIABLE
    )


def build_model(
    params: Params,
    channel_groups=None,
) -> tuple[nn.Module, LossFn, GetValueFn]:
    """
    Build convCNP model based on variable type.

    Args:
        params: Training/model configuration parameters.
        channel_groups: Optional mapping {label: [channel indices]} (or list of
            (label, [indices])) describing how to group input channels for a
            targeted encoder. Required when ``params.ENCODER != 'flat'``; ignored
            for the default flat encoder.

    Returns:
        model: The constructed nn.Module (not yet moved to device).
        loss_fn: The loss function for training.
        get_value_fn: Function to extract point predictions from model output.
    """
    if params.IN_CHANNELS <= 0:
        raise ValueError(
            f"IN_CHANNELS={params.IN_CHANNELS} is invalid. "
            "It must be set from the data before building the model (see cell 5)."
        )

    # Resolve the output distribution: an explicit params.DISTRIBUTION overrides
    # the per-variable default in VARIABLE_TO_DISTRIBUTION.
    distribution = resolve_distribution(params)
    if distribution is None or distribution not in LIKELIHOODS:
        raise ValueError(
            f"No registered likelihood for VARIABLE={params.VARIABLE!r} "
            f"(DISTRIBUTION={params.DISTRIBUTION!r}, resolved={distribution!r}). "
            f"Known: {list(LIKELIHOODS)}."
        )
    spec = LIKELIHOODS[distribution]

    # Build CNN decoder (same architecture for all variables/distributions)
    decoder = CNN(
        n_channels=params.N_CHANNELS,
        ConvBlock=ResConvBlock,
        n_blocks=params.N_BLOCKS,
        Conv=nn.Conv2d,
        Normalization=nn.Identity,
        kernel_size=params.KERNEL_SIZE,
    )

    # Optionally build a targeted (grouped) encoder; otherwise the model builds
    # its own flat set-convolution encoder.
    encoder = None
    if params.ENCODER != "flat":
        if channel_groups is None:
            raise ValueError(
                f"ENCODER={params.ENCODER!r} requires channel_groups (a "
                "{label: [indices]} grouping of the input channels) to be passed "
                "to build_model."
            )
        encoder = build_grouped_encoder(
            params.ENCODER,
            channel_groups,
            out_channels=params.N_CHANNELS,
            kernel_size=params.KERNEL_SIZE,
        )

    model = spec.model_class(
        decoder=decoder,
        in_channels=params.IN_CHANNELS,
        ls=params.LENGTH_SCALE,
        use_seasonal_in_mlp=params.SEASONAL_FEATURES_IN_MLP,
        encoder=encoder,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Built {type(model).__name__} ({distribution}, {spec.n_params} params) "
        f"with {n_params:,} trainable parameters; "
        f"encoder={params.ENCODER}"
    )

    # The sample-based CRPS loss takes its MC sample count from params; bind it so
    # the training loop can keep calling loss_fn(target, v) unchanged.
    loss_fn = spec.loss_fn
    if distribution == "bernoulli_gamma_crps":
        loss_fn = partial(spec.loss_fn, n_samples=params.CRPS_N_SAMPLES)

    return model, loss_fn, spec.get_value_fn


def load_model_checkpoint(
    checkpoint_path: Path,
    p: Params,
    device: torch.device,
    channel_groups=None,
) -> tuple[nn.Module, int]:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to the saved checkpoint file.
        p: Training parameters used to build the model architecture.
        device: Torch device to load the model onto.
        channel_groups: Channel-group spec, required when the model was trained
            with a targeted encoder (p.ENCODER != 'flat').

    Returns:
        Tuple of (model, epoch) where epoch is the training epoch of the checkpoint.
    """
    # Build model architecture using shared factory
    model, _, _ = build_model(p, channel_groups=channel_groups)
    model.to(device)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # When a model is trained with nn.DataParallel, PyTorch saves each key in
    # the state dict with a 'module.' prefix (e.g. 'module.decoder.weight').
    # Since we load onto a plain (non-DataParallel) model, we need to strip
    # that prefix so the keys match the model's own parameter names.
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = OrderedDict([
            (k.removeprefix('module.'), v) for k, v in state_dict.items()
        ])

    # Guard against params/checkpoint drift: the checkpoint's encoder conv has
    # one output channel per input channel, so its shape is the authoritative
    # record of how many channels the model was actually trained with. If this
    # disagrees with params.IN_CHANNELS the architectures won't match and
    # load_state_dict would fail with a cryptic size-mismatch error -- raise a
    # clear one instead.
    conv_weight = state_dict.get('encoder.conv.weight')
    if conv_weight is not None:
        ckpt_in_channels = conv_weight.shape[0]
        if ckpt_in_channels != p.IN_CHANNELS:
            raise ValueError(
                f"IN_CHANNELS mismatch for checkpoint '{checkpoint_path}': "
                f"params.json says IN_CHANNELS={p.IN_CHANNELS}, but the "
                f"checkpoint was trained with {ckpt_in_channels} input channels "
                f"(encoder.conv.weight shape {tuple(conv_weight.shape)}). "
                f"Fix IN_CHANNELS in the model's params.json to "
                f"{ckpt_in_channels}, or point at the correct checkpoint."
            )

    model.load_state_dict(state_dict)
    model.eval()

    return model, checkpoint.get('epoch', -1)
