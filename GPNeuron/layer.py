
from collections.abc import Callable
import torch
from torch import Tensor
from torch.nn import Module, Parameter

from . import func as F
from .base import Layer
from .mean import ZeroMean, AffineMean, LinearMean, ActivMean
from .kernel import Kernel1D, KernelND


class GP(Layer):

    def __init__(self, size: list[int], num_induc: int) -> None:
        Layer.__init__(self, size[-1], size[0])

        self.num_induc = num_induc  # M
        self.mean = ZeroMean()
        self.kernel = Kernel1D(size)
        # z, u ~ [(D), Q, M]
        self.induc_loc = Parameter(torch.rand(*size, num_induc).mul(2).sub(1))
        self.induc_mean = Parameter(self.mean.forward(self.induc_loc))

    def forward(self, x_mean: Tensor, x_var: Tensor | None = None) -> tuple[Tensor, Tensor]:
        # x ~ [Q, B]
        # -> f ~ [(D), Q, B]
        f_mean, f_var, self.induc_reg = self.kernel.forward(
            x_mean, x_var, self.induc_loc, self.induc_mean.sub(self.mean.forward(self.induc_loc)),
        )
        return f_mean.add(self.mean.forward(x_mean)), f_var

    def extra_repr(self) -> str:
        return Layer.extra_repr(self) + f', num_induc={self.num_induc}'


class SGP(GP):

    def __init__(self, in_dim: int, out_dim: int, num_induc: int) -> None:
        GP.__init__(self, [out_dim, in_dim], num_induc)

        self.mean = AffineMean(in_dim, out_dim)
        self.kernel = KernelND(in_dim, out_dim)
        # u ~ [D, M]
        self.induc_mean.data = self.mean.forward(self.induc_loc)

    @ torch.no_grad()
    def init_induc_loc(self, X: Tensor) -> None:
        self.induc_loc.data = F.batched_kmeans(X, self.num_induc, self.out_dim).mT
        self.induc_mean.data = self.mean.forward(self.induc_loc)


class FCGP(GP):

    def __init__(self, in_dim: int, out_dim: int, num_induc: int) -> None:
        GP.__init__(self, [out_dim, in_dim], num_induc)

        self.mean = LinearMean(in_dim, out_dim)
        self.induc_mean.data = self.mean.forward(self.induc_loc)

    def forward(self, x_mean: Tensor, x_var: Tensor | None = None) -> tuple[Tensor, Tensor]:
        # x ~ [Q, B]
        # -> f ~ [D, B]
        f_mean, f_var = GP.forward(self, x_mean, x_var)
        return f_mean.sum(1), f_var.sum(1)


class ICGP(GP):

    def __init__(
        self, dim: int | None, num_induc: int, mean_func: Callable[[Tensor], Tensor] = torch.tanh,
    ) -> None:
        GP.__init__(self, [dim], num_induc)

        self.mean = ActivMean(func=mean_func)
        self.induc_mean.data = self.mean.forward(self.induc_loc)

    def extra_repr(self) -> str:
        return f'dim={self.in_dim}, num_induc={self.num_induc}'


class SICGP(ICGP):

    def __init__(
        self, num_induc: int, mean_func: Callable[[Tensor], Tensor] = torch.tanh,
    ) -> None:
        ICGP.__init__(self, 1, num_induc, mean_func)

    def forward(self, x_mean: Tensor, x_var: Tensor | None = None) -> tuple[Tensor, Tensor]:
        # x ~ [Q, B]
        # -> f ~ [Q, B]
        s = x_mean.size()
        x_var = None if x_var is None else x_var.view(1,-1)
        x_mean, x_var = GP.forward(self, x_mean.view(1,-1), x_var)
        return x_mean.view(s), x_var.view(s)

    def extra_repr(self) -> str:
        return f'num_induc={self.num_induc}'


class Affine(Layer):
    
    def __init__(self, in_dim: int, out_dim: int) -> None:
        Layer.__init__(self, in_dim, out_dim)

        # weight ~ [D, Q]
        self.weight = Parameter(torch.randn(out_dim, in_dim).mul(in_dim**-0.5))
        # bias ~ [D, 1]
        self.bias = Parameter(torch.zeros(out_dim, 1))

    def forward(self, x_mean: Tensor, x_var: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        # x ~ [Q, B]
        # -> f ~ [D, B]
        f_mean = self.weight.matmul(x_mean).add(self.bias)
        f_var = None if x_var is None else self.weight.square().matmul(x_var)
        return f_mean, f_var


