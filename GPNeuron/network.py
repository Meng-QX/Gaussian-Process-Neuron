
from abc import ABC, abstractmethod
from collections.abc import Callable
import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Parameter
from torch.nn.functional import pad

from .layer import GP, SGP, ICGP, FCGP, Affine, ShiftedSum


class Network(Module, ABC):
    
    @abstractmethod
    def __init__(
        self, dims: tuple[int, ...] | None = None, nums_induc: int | tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()

        # [D^0, ..., D^L]
        self.dims = dims
        # [M^1, ..., M^L]
        if isinstance(nums_induc, int):
            nums_induc = (nums_induc,) * (len(dims) - 1)
        self.nums_induc = nums_induc
        self.layers = ModuleList()

    @property
    def induc_nll(self) -> Tensor | None:
        # -ln(p(u|z)) ~ []
        nlls = [
            nll.sum()
            for layer in self.layers
            if isinstance(layer, GP) and (nll := layer.induc_nll) is not None
        ]
        return sum(nlls) if nlls else None
    
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # x ~ [B, ...]
        # -> f_mean, f_var ~ [B, D^L]
        x_mean = x.movedim(0, -1)
        x_var = None
        for layer in self.layers:
            x_mean, x_var = layer(x_mean, x_var)
        return x_mean.movedim(-1, 0), x_var.movedim(-1, 0)

    def extra_repr(self) -> str:
        d = '' if self.dims is None else f'dims={self.dims}'
        m = '' if self.nums_induc is None else f'nums_induc={self.nums_induc}'
        return '\n'.join(filter(None, (d, m)))


class DGP(Network):

    def __init__(self, dims: tuple[int, ...], nums_induc: int | tuple[int, ...]) -> None:
        super().__init__(dims, nums_induc)

        for in_dim, out_dim, num_induc in zip(dims[:-1], dims[1:], self.nums_induc, strict=True):
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
            layer.train(False)
            F_mean = torch.empty(layer.size[0], sample_size)
            F_var = torch.empty(F_mean.size())
            for start_idx in range(0, sample_size, batch_size):
                end_idx = min(start_idx + batch_size, sample_size)
                idx = range(start_idx, end_idx)
                F_mean[:, idx], F_var[:, idx] = layer(X_mean[:, idx], X_var[:, idx])
            X_mean = F_mean
            X_var = F_var


class GPLAN(Network):

    def __init__(
        self, dims: tuple[int, ...], nums_induc: int | tuple[int, ...],
        shared: bool = False, mean_func: Callable[[Tensor], Tensor] = torch.tanh,
    ) -> None:
        super().__init__(dims, nums_induc)

        for in_dim, out_dim, num_induc in zip(dims[:-1], dims[1:], self.nums_induc, strict=True):
            self.layers.append(Affine(in_dim, out_dim))
            size = () if shared else out_dim
            self.layers.append(ICGP(size, num_induc, mean_func))


class GPKAN(Network):

    def __init__(self, dims: tuple[int, ...], nums_induc: int | tuple[int, ...]) -> None:
        super().__init__(dims, nums_induc)

        for in_dim, out_dim, num_induc in zip(dims[:-1], dims[1:], self.nums_induc, strict=True):
            self.layers.append(FCGP(in_dim, out_dim, num_induc))
            self.layers.append(ShiftedSum(out_dim))


class MDGP(Network):

    def __init__(
        self, dims: tuple[int, ...], nums_induc: int | tuple[int, ...], num_comp: int,
    ) -> None:
        super().__init__(dims, nums_induc)

        # K
        self.num_comp = num_comp
        # mix_weight ~ [K, 1, D^L]
        self._mix_weight = Parameter(torch.zeros(num_comp-1, 1, dims[-1]))
        for in_dim, out_dim, num_induc in zip(
            dims[:-1], dims[1:-1]+(dims[-1]*num_comp,), self.nums_induc, strict=True,
        ):
            self.layers.append(SGP(in_dim, out_dim, num_induc))

    @property
    def mix_weight(self) -> Tensor:
        return pad(self._mix_weight, pad=(0, 0, 0, 0, 0, 1), value=0).softmax(0)
    
    def init_induc_loc(
        self, X: Tensor, batch_size: int = 256, max_iter: int = 100, tol: float = 1e-3,
    ) -> None:
        DGP.init_induc_loc(self, X, batch_size, max_iter, tol)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # x ~ [B, D^0]
        # -> f_mean, f_var ~ [K, B, D^L]
        x_mean, x_var = super().forward(x)
        x_mean = x_mean.unflatten(1, (self.num_comp, -1)).movedim(1, 0)
        x_var = x_var.unflatten(1, (self.num_comp, -1)).movedim(1, 0)
        return x_mean, x_var

    def extra_repr(self) -> str:
        return super().extra_repr() + f', num_comp={self.num_comp}'


