"""
Joint temperature + precipitation downscaling convCNP.

A single model that predicts BOTH variables at once via a conditional
factorisation P(tmax) * P(precip | tmax):

  * a shared set-convolution encoder + ResNet CNN backbone produces one grid
    representation from the ERA5 context;
  * a Gaussian head predicts tmax (mean, sigma);
  * a Bernoulli-Gamma head predicts precip, conditioned on tmax at two levels —
    the tmax on-grid features feed the precip grid MLP, and the predicted tmax
    value at each target point feeds the precip elevation MLP.

This is a clean, fixed reimplementation of the intent sketched in
``multivar_models_tmax_init.py`` (which hardcoded ``.cuda()``/grid shapes and a
frozen tmax model). Here tmax is trained jointly with precip.

The two negative log-likelihoods are combined with Kendall et al. (2018)
homoscedastic-uncertainty weighting via the learnable ``log_var_t``/``log_var_p``
parameters (see :meth:`combined_loss`), so the differing NLL scales (tmax ~1.5,
precip ~2.3) are balanced automatically.
"""

import torch
import torch.nn as nn

from .encoder import Encoder
from .mlp import MLP
from .final_layers import GaussianFinalLayer, GammaFinalLayer
from .utils import force_positive


def append_target_features(base, elev, seasonal, use_seasonal):
    """Concat per-target elevation (and seasonal) features onto ``base``.

    ``base`` is (batch, n_points, C); ``elev`` is (n_points, 3); ``seasonal`` is
    (batch, 2) or None. Mirrors the per-variable elev-MLP input assembly.
    """
    batch_size, n_points = base.shape[0], base.shape[1]
    elev_expanded = elev.repeat(batch_size, 1, 1)
    if use_seasonal and seasonal is not None:
        seasonal_expanded = seasonal.unsqueeze(1).expand(-1, n_points, -1)
        return torch.cat([base, elev_expanded, seasonal_expanded], dim=2)
    return torch.cat([base, elev_expanded], dim=2)


class JointConditionalConvCNP(nn.Module):
    """Joint tmax (Gaussian) + precip (Bernoulli-Gamma) convCNP, precip | tmax.

    Parameters
    ----------
    decoder : nn.Module
        Shared CNN backbone (built like the per-variable models, with
        ``decoder.out_channels`` working width, e.g. 128).
    in_channels : int
        Number of ERA5 context channels.
    ls : float
        Initial RBF length scale for the precip final layer (the Gaussian final
        layer keeps the codebase default of 0.1, matching ``TmaxBiasConvCNPElev``).
    use_seasonal_in_mlp : bool
        Whether the elevation MLPs receive the cos/sin day-of-year features.
    encoder : nn.Module or None
        Optional prebuilt (grouped) encoder; otherwise a flat encoder is built.
    detach_tmax_cond : bool
        If True, stop precip-head gradients from flowing into the tmax prediction
        used as conditioning (treats tmax as a fixed cause). Default False = fully
        joint optimisation.
    loss_weighting : {'kendall', 'clamped', 'fixed'}
        How :meth:`combined_loss` balances the two NLLs.
        - ``'kendall'`` (default): learned homoscedastic-uncertainty weighting
          (Kendall 2018). NB this can run away when one task is a Gaussian NLL that
          goes negative (``w -> inf``); use the other modes to prevent that.
        - ``'clamped'``: Kendall, but the log-variances are clamped to
          ``[-logvar_clamp, logvar_clamp]`` so the weights stay bounded.
        - ``'fixed'``: no learned weights; ``total = nll_t + precip_weight * nll_p``.
    precip_weight : float
        Multiplier on the precip loss term. In ``'fixed'`` mode it is the precip
        weight directly; in ``'kendall'``/``'clamped'`` it scales the whole precip
        term (default 1.0 = unchanged from the original Kendall formula).
    logvar_clamp : float or None
        Clamp bound for the log-variances in ``'clamped'`` mode (e.g. 1.5).
    """

    def __init__(self, decoder, in_channels=1, ls=0.1, use_seasonal_in_mlp=True,
                 encoder=None, detach_tmax_cond=False, loss_weighting='kendall',
                 precip_weight=1.0, logvar_clamp=None):
        super().__init__()
        self.in_channels = in_channels
        self.use_seasonal_in_mlp = use_seasonal_in_mlp
        self.detach_tmax_cond = detach_tmax_cond
        self.loss_weighting = loss_weighting
        self.precip_weight = float(precip_weight)
        self.logvar_clamp = logvar_clamp
        self.activation = torch.relu
        self.sigmoid = nn.Sigmoid()

        # Shared backbone.
        self.encoder = encoder if encoder is not None else Encoder(in_channels)
        self.decoder = decoder
        width = decoder.out_channels

        # --- tmax head (Gaussian, 2 params) ---
        self.mlp_t = MLP(width, 2, hidden_channels=64, hidden_layers=4)
        self.final_t = GaussianFinalLayer(0.1, 2)
        # elev MLP: 2 (mu,sigma) + 3 (elev) + 2 (seasonal)
        elev_t_in = 7 if use_seasonal_in_mlp else 5
        self.elev_mlp_t = MLP(elev_t_in, 2, hidden_channels=64, hidden_layers=4)

        # --- precip head (Bernoulli-Gamma, 3 params), conditioned on tmax ---
        # Grid-level conditioning: concat the tmax on-grid params (2) to the
        # shared representation before the precip grid MLP.
        self.mlp_p = MLP(width + 2, 3, hidden_channels=64, hidden_layers=4)
        self.final_p = GammaFinalLayer(init_ls=ls, n_params=3)
        # elev MLP: 3 (rho,alpha,beta) + 1 (predicted tmax mean) + 3 (elev) + 2 (seasonal)
        elev_p_in = (8 if use_seasonal_in_mlp else 6) + 1
        self.elev_mlp_p = MLP(elev_p_in, 3, hidden_channels=64, hidden_layers=4)

        # --- Kendall homoscedastic-uncertainty log-variances (checkpointed) ---
        self.log_var_t = nn.Parameter(torch.zeros(()))
        self.log_var_p = nn.Parameter(torch.zeros(()))

    # ------------------------------------------------------------------
    def _append_targets_features(self, base, elev, seasonal):
        return append_target_features(base, elev, seasonal, self.use_seasonal_in_mlp)

    def forward(self, x, mask, dists, elev, seasonal=None):
        """Return (out_tmax, out_precip).

        out_tmax  : (batch, n_points, 2) = [mu, sigma]
        out_precip: (batch, n_points, 3) = [rho, alpha, beta]
        """
        # Shared backbone.
        g = self.activation(self.encoder(x, mask))
        g = self.activation(self.decoder(g))           # (batch, lat, lon, width)

        # --- tmax head ---
        h_t = self.mlp_t(g)                             # (batch, lat, lon, 2)
        mu, sigma = self.final_t(h_t, dists)            # each (batch, n_points)
        out_t = torch.cat([mu.view(*mu.shape, 1), sigma.view(*sigma.shape, 1)], dim=2)
        out_t = self._append_targets_features(out_t, elev, seasonal)
        out_t = self.elev_mlp_t(out_t)                  # (batch, n_points, 2)
        out_t[..., 1] = force_positive(out_t[..., 1])

        # --- precip head, conditioned on tmax ---
        h_p = self.mlp_p(torch.cat([g, h_t], dim=-1))   # grid-level conditioning
        rho, alpha, beta = self.final_p(h_p, dists)     # each (batch, n_points, 1)
        out_p = torch.cat([rho, alpha, beta], dim=2)    # (batch, n_points, 3)
        # target-level conditioning: predicted tmax mean at each point.
        tmax_mean = out_t[..., 0:1]
        if self.detach_tmax_cond:
            tmax_mean = tmax_mean.detach()
        out_p = torch.cat([out_p, tmax_mean], dim=2)
        out_p = self._append_targets_features(out_p, elev, seasonal)
        out_p = self.elev_mlp_p(out_p)                  # (batch, n_points, 3)
        out_p[..., 0] = torch.clamp(self.sigmoid(out_p[..., 0]), 1e-5, 1 - 1e-5)
        out_p[..., 1:] = force_positive(out_p[..., 1:])

        return out_t, out_p

    # ------------------------------------------------------------------
    def combined_loss(self, nll_t, nll_p):
        """Balance the two NLLs per ``self.loss_weighting``. Returns (total, w_t, w_p).

        - ``'fixed'``:  total = NLL_t + precip_weight * NLL_p  (no learned weights;
          w_t, w_p returned as the fixed 1.0 / precip_weight for the stats log).
        - ``'clamped'``/``'kendall'``: Kendall et al. (2018) uncertainty weighting
          L = exp(-s_t)*NLL_t + s_t + precip_weight*(exp(-s_p)*NLL_p + s_p), with
          learnable s = log variance per task. ``'clamped'`` first clamps s to
          [-logvar_clamp, logvar_clamp] so exp(-s) cannot run away.

        With the defaults (loss_weighting='kendall', precip_weight=1.0) this is
        byte-for-byte the original Kendall formula.
        """
        if self.loss_weighting == 'fixed':
            total = nll_t + self.precip_weight * nll_p
            w_t = torch.ones((), device=nll_t.device, dtype=nll_t.dtype)
            w_p = w_t * self.precip_weight
            return total, w_t, w_p

        lvt, lvp = self.log_var_t, self.log_var_p
        if self.loss_weighting == 'clamped' and self.logvar_clamp is not None:
            c = abs(float(self.logvar_clamp))
            lvt = lvt.clamp(-c, c)
            lvp = lvp.clamp(-c, c)
        w_t = torch.exp(-lvt)
        w_p = torch.exp(-lvp)
        total = w_t * nll_t + lvt + self.precip_weight * (w_p * nll_p + lvp)
        return total, w_t, w_p


class ConditionalPrecipModel(nn.Module):
    """Precip (Bernoulli-Gamma) head with its OWN backbone, conditioned on a
    frozen pretrained tmax model — the two-stage P(precip | tmax) design.

    Unlike :class:`JointConditionalConvCNP` (one shared backbone), this gives
    precip a dedicated encoder + CNN so its representation is never starved by
    tmax's (larger, cleaner) gradients. The frozen ``tmax_model`` supplies the
    predicted tmax at each target point, appended to the precip elevation MLP.

    Parameters mirror the per-variable models; ``tmax_model`` is any module with
    the ``(x, mask, dists, elev, seasonal)`` signature returning (batch, P, 2).
    """

    def __init__(self, decoder, tmax_model, in_channels=1, ls=0.1,
                 use_seasonal_in_mlp=True, encoder=None):
        super().__init__()
        self.tmax_model = tmax_model
        for q in self.tmax_model.parameters():   # freeze stage-1 tmax
            q.requires_grad = False
        self.use_seasonal_in_mlp = use_seasonal_in_mlp
        self.activation = torch.relu
        self.sigmoid = nn.Sigmoid()

        self.encoder = encoder if encoder is not None else Encoder(in_channels)
        self.decoder = decoder
        self.mlp = MLP(decoder.out_channels, 3, hidden_channels=64, hidden_layers=4)
        self.out_layer = GammaFinalLayer(init_ls=ls, n_params=3)
        elev_p_in = (8 if use_seasonal_in_mlp else 6) + 1   # +1 predicted tmax mean
        self.elev_mlp = MLP(elev_p_in, 3, hidden_channels=64, hidden_layers=4)

    def forward(self, x, mask, dists, elev, seasonal=None):
        """Return (out_tmax, out_precip) — out_tmax is the frozen prediction."""
        self.tmax_model.eval()
        with torch.no_grad():
            out_t = self.tmax_model(x, mask, dists, elev, seasonal=seasonal)
        tmax_mean = out_t[..., 0:1]

        h = self.activation(self.encoder(x, mask))
        h = self.activation(self.decoder(h))
        h = self.mlp(h)
        rho, alpha, beta = self.out_layer(h, dists)
        out_p = torch.cat([rho, alpha, beta], dim=2)
        out_p = torch.cat([out_p, tmax_mean], dim=2)
        out_p = append_target_features(out_p, elev, seasonal, self.use_seasonal_in_mlp)
        out_p = self.elev_mlp(out_p)
        out_p[..., 0] = torch.clamp(self.sigmoid(out_p[..., 0]), 1e-5, 1 - 1e-5)
        out_p[..., 1:] = force_positive(out_p[..., 1:])
        return out_t, out_p
