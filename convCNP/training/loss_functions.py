"""
Functions to calculate NLL for various distributions
"""
import torch
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal

from .utils import make_r_mask

def gll(target_vals, v):
    """
    Calculate mean Gaussian log likelihood over the batch
    """
    # Reshape
    target_vals = target_vals.reshape(-1)
    v = v.reshape(-1, 2)
    
    # Deal with cases where data is missing for a station
    v = v[~torch.isnan(target_vals), :]
    target_vals = target_vals[~torch.isnan(target_vals)]
    
    dist = Normal(loc=v[:,0], scale=v[:,1])
    logp = dist.log_prob(target_vals).view(-1)
    return torch.mean(logp)

def gamma_ll(target_vals, v):
    """
    Evaluate gamma-bernoulli mixture likelihood
    Parameters:
    ----------
    v: torch.Tensor(batch,86,channels)
        parameters from model [rho, alpha, beta]
    target_vals: torch.Tensor(batch,86)
        target vals to eval at
    """

    # Reshape
    target_vals = target_vals.reshape(-1)
    v = v.reshape(-1, 3)
    
    # Deal with cases where data is missing for a station
    v = v[~torch.isnan(target_vals), :]
    target_vals = target_vals[~torch.isnan(target_vals)]

    # Make r mask
    r, target_vals = make_r_mask(target_vals)

    gamma = Gamma(concentration = v[:,1], rate = v[:,2])
    logp = gamma.log_prob(target_vals)

    total = r*(torch.log(v[:,0])+logp)+(1-r)*torch.log(1-v[:,0])

    return torch.mean(total)


def gamma_bernoulli_crps(target_vals, v, n_samples=25):
    """
    Differentiable CRPS for the Bernoulli-Gamma mixture, returned as a *score to
    maximize* (i.e. ``-mean(CRPS)``), so it is a drop-in replacement for
    ``gamma_ll`` under the training convention ``obj = -ll(target, v)`` — the
    optimizer then minimizes the mean CRPS.

    The predictive distribution is a point mass ``(1-rho)`` at 0 plus a Gamma
    (shape ``alpha``, rate ``beta``) with weight ``rho``:
        F(x) = (1-rho)*1{x>=0} + rho*Gamma_cdf(x; alpha, beta).
    CRPS is estimated with the energy form
        CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|,   X, X' iid ~ F,
    decomposed over the wet/dry mixture so the (non-differentiable) Bernoulli
    part is handled analytically and only the Gamma part is Monte-Carlo
    estimated with *reparameterized* (pathwise-differentiable) draws:
        E|X - y|  = (1-rho)*|y| + rho * E_G|X_G - y|
        E|X - X'| = 2*rho*(1-rho)*E[X_G] + rho^2 * E|X_G - X_G'|
    (both-dry term is 0; E[X_G] = alpha/beta). Targets are raw mm (y >= 0), so
    exact-zero dry days are used as-is — no wet/dry masking is needed (a dry day
    is correctly penalized for any predicted wet mass).

    Parameters
    ----------
    v : torch.Tensor(..., 3)
        model params [rho, alpha, beta].
    target_vals : torch.Tensor
        observed accumulations (raw mm).
    n_samples : int
        Monte-Carlo Gamma draws for the two expectations (default 25).
    """
    # Reshape
    target_vals = target_vals.reshape(-1)
    v = v.reshape(-1, 3)

    # Deal with cases where data is missing for a station
    v = v[~torch.isnan(target_vals), :]
    target_vals = target_vals[~torch.isnan(target_vals)]

    rho, alpha, beta = v[:, 0], v[:, 1], v[:, 2]
    y = target_vals  # raw mm, >= 0

    gamma = Gamma(concentration=alpha, rate=beta)
    # Reparameterized (pathwise-differentiable) samples, shape (n_samples, N).
    xs = gamma.rsample((n_samples,))
    xs2 = gamma.rsample((n_samples,))  # independent set for the E|X_G - X_G'| term

    e_abs_xy = (xs - y.unsqueeze(0)).abs().mean(dim=0)   # E_G|X_G - y|
    e_abs_xx = (xs - xs2).abs().mean(dim=0)              # E|X_G - X_G'|
    mean_g = alpha / beta                                # E[X_G]

    term_xy = (1.0 - rho) * y.abs() + rho * e_abs_xy
    term_xx = 2.0 * rho * (1.0 - rho) * mean_g + rho * rho * e_abs_xx
    crps = term_xy - 0.5 * term_xx

    return -torch.mean(crps)

