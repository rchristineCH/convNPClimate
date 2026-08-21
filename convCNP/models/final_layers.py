import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class ParamLayer(nn.Module):
    """
    Calculate predicted of a parameter from gridded output
    Parameters:
    -----------
    init_ls: float
        initial length scale for the RBF kernel
    chunk_size: int
        number of target points to process per chunk. The full
        (n_points, lat, lon) RBF kernel is never materialised at once; instead
        each chunk's kernel is built, contracted, and (in training) recomputed
        in the backward pass via gradient checkpointing. This bounds peak memory
        to ~one chunk's kernel rather than n_params * (n_points * grid), which is
        what previously overflowed the 8 GB GPU on dense target grids.
    """

    def __init__(self, init_ls, chunk_size=16384):
        super().__init__()
        self.init_ls = torch.nn.Parameter(torch.tensor([init_ls]))
        self.init_ls.requires_grad = True
        self.chunk_size = chunk_size

    def _chunk(self, wt_flat, dists_chunk):
        # RBF kernel for this chunk of target points, contracted over the
        # spatial dims (lat, lon) with a single GEMM. einsum('bij,pij->bp')
        # would broadcast to a (batch, n_points, lat, lon) intermediate before
        # summing; flattening into a matmul gives the identical result without
        # that intermediate.
        kernel = torch.exp(-0.5 * dists_chunk / self.init_ls ** 2)
        p = dists_chunk.shape[0]
        return wt_flat @ kernel.reshape(p, -1).t()

    def forward(self, wt, dists):
        b = wt.shape[0]
        wt_flat = wt.reshape(b, -1)
        n_points = dists.shape[0]
        outs = []
        for start in range(0, n_points, self.chunk_size):
            dists_chunk = dists[start:start + self.chunk_size]
            if self.training and torch.is_grad_enabled():
                # Recompute the kernel in backward instead of storing it.
                outs.append(checkpoint(self._chunk, wt_flat, dists_chunk, use_reentrant=False))
            else:
                outs.append(self._chunk(wt_flat, dists_chunk))
        return torch.cat(outs, dim=1)

class FinalLayer(nn.Module):
    """
    Final layer for converting gridded parameter 
    predictions to off the grid points
    Parameters:
    ----------
    init_ls: Int
        Initial length scale for the RBF kernel
    n_params:
        Total number of parameters in the target 
        distribution
    """

    def __init__(self, 
                 init_ls, 
                 n_params):
        
        super(FinalLayer, self).__init__()
        self.param_layers = nn.ModuleList(
            [ParamLayer(init_ls)
             for _ in range(n_params)]
        )
        self.sigmoid = nn.Sigmoid()

    def _log_exp(self, x):
        """
        Fix overflow
        """
        lt = torch.where(torch.exp(x)<1000)
        if lt[0].shape[0] > 0:
            x[lt] = torch.log(1+torch.exp(x[lt]))
        return x

    def _force_positive(self, x):
        """
        Make values greater than zero
        """
        return 0.01+ (1-0.1)*self._log_exp(x)

    def forward(self, x, dists):
        pass

class GaussianFinalLayer(FinalLayer):
    """
    On-grid -> off-grid layer for Gaussian distribution
    """

    def __init__(self,
                 init_ls,
                 n_params):

        FinalLayer.__init__(self, 
                 init_ls, 
                 n_params)

    def forward(self, h, dists):
        mu_h = h[...,0]
        sigma_h = h[..., 1]

        mu = self.param_layers[0](mu_h, dists)
        sigma = self._force_positive(self.param_layers[1](sigma_h, dists))

        return mu, sigma

class GammaFinalLayer(FinalLayer):
    """
    On-grid -> off-grid layer for Bernoulli-Gamma 
    mixture distribution
    """

    def __init__(self,  
                 init_ls,
                 n_params):

        FinalLayer.__init__(self, 
                 init_ls, 
                 n_params)

    def forward(self, h, dists):

        rho = self.param_layers[0](h[..., 0], dists)
        alpha = self.param_layers[1](h[..., 1], dists)
        beta = self.param_layers[2](h[..., 2], dists)

        rho = self.sigmoid(rho).view(*rho.shape, 1)
        alpha =self._force_positive(alpha).view(*rho.shape)
        beta = self._force_positive(beta).view(*rho.shape)

        # clamp values
        rho = torch.clamp(rho, min = 1e-5, max=1-1e-5)
        alpha = torch.clamp(alpha, min = 1e-5, max=1e5)
        beta = torch.clamp(beta, min = 1e-5, max=1e5)

        return rho, alpha, beta
