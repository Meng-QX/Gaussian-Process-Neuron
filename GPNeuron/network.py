
from collections.abc import Callable
import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Parameter
from torch.nn.functional import softplus

from . import const as C, func as F
from .layer import SGP, ICGP, FCGP, Affine, ShiftedSum


class Network(Module):
    
    def __init__(self, out_dim: int) -> None:
        super().__init__()

        # obs_noise ~ [D^L]
        self._obs_noise = Parameter(torch.full((out_dim,), F.inv_softplus(C.OBS_NOISE_INIT)))
        self.layers = ModuleList()

    @property
    def obs_noise(self) -> Tensor:
        return softplus(self._obs_noise).add(C.OBS_NOISE_MIN)

    @property
    def induc_nll(self) -> Tensor:
        return sum(layer.induc_nll.sum() for layer in self.layers)
    
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        ...

    def ell(self, y: Tensor, f_mean: Tensor, f_var: Tensor) -> Tensor:
        # expected log-likelihood E_{p(f|x)}[ln(p(y|f))]
        obs_noise = self.obs_noise
        return F.normal_log_prob(y, f_mean, obs_noise).sub(f_var.div(obs_noise), alpha=0.5)
    
    def mll(self, y: Tensor, f_mean: Tensor, f_var: Tensor) -> Tensor:
        # marginal log-likelihood ln(p(y|x))
        return F.normal_log_prob(y, f_mean, f_var.add(self.obs_noise))
    
    def loglikelihood(self, x: Tensor, y: Tensor, mll: bool = True) -> Tensor:
        # x ~ [B, ...], y ~ [B, D^L]
        # -> ll ~ []
        ll = self.mll if mll else self.ell
        return ll(y, *self.forward(x)).sum(-1).mean()

    @torch.no_grad()
    def evaluate(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # x ~ [B, ...], y ~ [B, D^L]
        # -> f_mean, nll, crps ~ [B, D^L]
        f_mean, f_var = self.forward(x)
        f_var = f_var.add(self.obs_noise)
        nll = F.normal_log_prob(y, f_mean, f_var).neg()
        crps = F.normal_crps(y, f_mean, f_var)
        return f_mean, nll, crps


class DenseNet(Network):

    def __init__(self, dims: tuple[int, ...]) -> None:
        super().__init__(dims[-1])

        # [D^0, ..., D^L]
        self.dims = tuple(dims)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # x ~ [B, D^0]
        # -> f_mean, f_var ~ [B, D^L]
        x_mean = x.mT
        x_var = None
        for layer in self.layers:
            x_mean, x_var = layer.forward(x_mean, x_var)
        return x_mean.mT, x_var.mT

    def extra_repr(self) -> str:
        return f'dims={self.dims}'


class DGP(DenseNet):

    def __init__(self, dims: tuple[int, ...], num_induc: int) -> None:
        super().__init__(dims)

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.layers.append(SGP(in_dim, out_dim, num_induc))

    @torch.no_grad()
    def init_induc_loc(
        self, X: Tensor, batch_size: int = 256, max_iter: int = 100, tol: float = 1e-3,
    ) -> None:
        sample_size = X.size(0)
        X_mean = X.mT
        X_var = torch.zeros(X_mean.size())
        num_layer = len(self.layers)
        for i, layer in enumerate(self.layers):
            layer.init_induc_loc(X_mean, max_iter, tol)
            if i == num_layer - 1:
                break
            F_mean = torch.empty(layer.size[0], sample_size)
            F_var = torch.empty(F_mean.size())
            for start_idx in range(0, sample_size, batch_size):
                end_idx = min(start_idx + batch_size, sample_size)
                idx = range(start_idx, end_idx)
                F_mean[:, idx], F_var[:, idx] = layer.forward(X_mean[:, idx], X_var[:, idx])
            X_mean = F_mean
            X_var = F_var


class GPLAN(DenseNet):

    def __init__(
        self, dims: tuple[int, ...], num_induc: int,
        shared: bool = False, mean_func: Callable[[Tensor], Tensor] = torch.tanh,
    ) -> None:
        super().__init__(dims)

        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            self.layers.append(Affine(in_dim, out_dim))
            size = () if shared else out_dim
            self.layers.append(ICGP(size, num_induc, mean_func=mean_func))
        self.layers.append(Affine(dims[-2], dims[-1]))


class GPKAN(DenseNet):

    def __init__(self, dims: tuple[int, ...], num_induc: int) -> None:
        super().__init__(dims)

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.layers.append(FCGP(in_dim, out_dim, num_induc))
            self.layers.append(ShiftedSum(out_dim))


