
from collections.abc import Callable
import torch
from torch import Tensor
from torch.nn import Module, Parameter

from . import const as C, func as F
from .mean import ZeroMean, AffineMean, LinearMean, ActivMean
from .kernel import Kernel1D, KernelND


class Layer(Module):

    def __init__(self):
        super().__init__()

        # u_nll ~ []
        self.induc_nll = torch.zeros(())

    def forward(self, x_mean: Tensor, x_var: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        ...


class GP(Layer):

    def __init__(self, dim: int | None, size: int | tuple[int, ...], num_induc: int) -> None:
        super().__init__()

        # (Q)
        self.dim = dim
        # [...]
        self.size = size = (size,) if isinstance(size, int) else tuple(size)
        # M
        self.num_induc = num_induc
        # m(.)
        self.mean = ZeroMean(size)
        # k(.,.)
        self.kernel = Kernel1D(size) if dim is None else KernelND(dim, size)
        # z ~ [..., (Q), M]
        z_size = size + (num_induc,) if dim is None else size + (dim, num_induc)
        self.induc_loc = Parameter(torch.rand(z_size).mul(2).sub(1))
        # u ~ [..., M]
        self.init_induc_value()
    
    @torch.no_grad()
    def init_induc_value(self) -> None:
        self.induc_value = Parameter(self.mean.forward(self.induc_loc))
        
    def forward(self, x_mean: Tensor, x_var: Tensor | None = None) -> tuple[Tensor, Tensor]:
        # x_mean, x_var ~ [..., (Q), B]
        # -> f_mean, f_var ~ [..., B]
        f_mean, f_var, self.induc_nll = self.kernel.forward(
            x_mean, x_var, self.induc_loc, self.induc_value.sub(self.mean.forward(self.induc_loc)),
        )
        return f_mean.add(self.mean.forward(x_mean)), f_var

    def extra_repr(self) -> str:
        d = '' if self.dim is None else f'dim={self.dim}'
        s = f'size={self.size}' if self.size else ''
        m = f'num_induc={self.num_induc}'
        return ', '.join(filter(None, (d, s, m)))


class SGP(GP):

    def __init__(self, in_dim: int, out_dim: int, num_induc: int) -> None:
        super().__init__(in_dim, out_dim, num_induc)

        self.mean = AffineMean(in_dim, out_dim)
        self.init_kernel_param()
        self.init_induc_value()

    @torch.no_grad()
    def init_kernel_param(self) -> None:
        in_dim = self.dim
        out_dim = self.size[0]
        if in_dim > 64:
            ls = C.LENGTHSCALE_INIT * 8 / in_dim ** 0.5
            self.kernel.lengthscale = Parameter(torch.full((out_dim, in_dim, 1), ls))
        _os = F.inv_softplus(C.OUTPUTSCALE_INIT / out_dim)
        self.kernel._outputscale = Parameter(torch.full((out_dim, 1), _os))

    @ torch.no_grad()
    def init_induc_loc(self, X: Tensor, max_iter: int = 100, tol: float = 1e-3) -> None:
        induc_loc, _ = F.kmeans(X, self.num_induc, self.size[0], max_iter, tol)
        self.induc_loc = Parameter(induc_loc)
        self.init_induc_value()

    def extra_repr(self) -> str:
        return f'in_dim={self.dim}, out_dim={self.size[0]}, num_induc={self.num_induc}'


class ICGP(GP):

    def __init__(
        self, size: int | tuple[int, ...], num_induc: int,
        mean_func: Callable[[Tensor], Tensor] = torch.tanh,
    ) -> None:
        super().__init__(None, size, num_induc)

        self.mean = ActivMean(func=mean_func)
        self.init_induc_value()


class FCGP(GP):

    def __init__(self, in_dim: int, out_dim: int, num_induc: int) -> None:
        super().__init__(None, (out_dim, in_dim), num_induc)

        self.mean = LinearMean((out_dim, in_dim))
        self.mean.weight = Parameter(torch.randn(out_dim, in_dim, 1).mul(in_dim**-0.5))
        self.init_induc_value()

    def extra_repr(self) -> str:
        return f'in_dim={self.size[1]}, out_dim={self.size[0]}, num_induc={self.num_induc}'


class Affine(Layer):
    
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()

        # Q
        self.in_dim = in_dim
        # D
        self.out_dim = out_dim
        # weight ~ [D, Q]
        self.weight = Parameter(torch.randn(out_dim, in_dim).mul(in_dim**-0.5))
        # bias ~ [D, 1]
        self.bias = Parameter(torch.zeros(out_dim, 1))

    def forward(self, x_mean: Tensor, x_var: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        # x_mean, x_var ~ [Q, B]
        # -> f_mean, f_var ~ [D, B]
        x_mean = self.weight.matmul(x_mean).add(self.bias)
        if x_var is not None:
            x_var = self.weight.square().matmul(x_var)
        return x_mean, x_var

    def extra_repr(self) -> str:
        return f'in_dim={self.in_dim}, out_dim={self.out_dim}'


class ShiftedSum(Layer):

    def __init__(self, out_dim: int) -> None:
        super().__init__()

        # D
        self.out_dim = out_dim
        # bias ~ [D, 1]
        self.bias = Parameter(torch.zeros(out_dim, 1))

    def forward(self, x_mean: Tensor, x_var: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        # x_mean, x_var ~ [D, Q, B]
        # -> f_mean, f_var ~ [D, B]
        x_mean = x_mean.sum(1).add(self.bias)
        if x_var is not None:
            x_var = x_var.sum(1)
        return x_mean, x_var

    def extra_repr(self) -> str:
        return f'out_dim={self.out_dim}'


