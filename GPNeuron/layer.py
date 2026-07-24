
from abc import ABC, abstractmethod
from collections.abc import Callable
import math
import torch
from torch import Tensor
from torch.linalg import cholesky_ex, solve_triangular
from torch.nn import Module, Parameter
from torch.nn.functional import silu, softplus

from . import constant as C, function as F
from .mean import ZeroMean, ActivMean, LinearMean, AffineMean
from .kernel import Kernel1D, KernelND


class Layer(Module, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x_mean: Tensor, x_var: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        ...


class GP(Layer):

    def __init__(self, dim: int | None, size: int | tuple[int, ...], num_induc: int) -> None:
        super().__init__()

        # Q
        self.dim = dim
        # [...]
        self.size = size = (size,) if isinstance(size, int) else size
        # M
        self.num_induc = num_induc
        # m(.)
        self.mean = ZeroMean(size)
        # k(.,.)
        self.kernel = Kernel1D(size) if dim is None else KernelND(dim, size)
        # induc_noise ~ []
        self._induc_noise = Parameter(torch.tensor(F.inv_softplus(C.INDUC_NOISE_INIT)))
        # z ~ [..., (Q), M]
        z_size = size + (num_induc,) if dim is None else size + (dim, num_induc)
        self.induc_loc = Parameter(torch.rand(z_size).mul(2).sub(1))
        # u ~ [..., M]
        u_size = size + (num_induc,)
        self.induc_value = Parameter(torch.empty(u_size))
        self.init_induc_value()

    @property
    def induc_noise(self) -> Tensor:
        return softplus(self._induc_noise).add(C.INDUC_NOISE_MIN)

    @torch.no_grad()
    def init_induc_value(self) -> None:
        # u ~ [..., M]
        induc_mean = self.mean(self.induc_loc)
        self.induc_value.copy_(induc_mean)

    @torch.no_grad()
    def induc_const(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # -> induc_loc_s ~ [..., (Q), M], mean_coef ~ [..., 1, M],
        #    induc_prec ~ [..., M, M], induc_nll ~ [...]
        ls = self.kernel.length_scale
        os = self.kernel.output_scale
        induc_loc_s = self.induc_loc.mul(ls)
        induc_mean = self.mean(self.induc_loc)
        induc_cov = self.kernel.cov_func(induc_loc_s, induc_loc_s)
        induc_cov = F.add_diag(induc_cov, self.induc_noise)
        induc_chol, _ = cholesky_ex(induc_cov)
        induc_prec = torch.cholesky_inverse(induc_chol)
        induc_shift = self.induc_value.sub(induc_mean).unsqueeze(-2)
        mean_coef = induc_shift.matmul(induc_prec)
        logdet_term = F.diag(induc_chol).log().sum(-1)
        dist_term = mean_coef.matmul(induc_shift.mT).squeeze(-1).div(os)
        const_term = os.log().add(C.LN_2PI).mul(self.num_induc)
        induc_nll = logdet_term.add(dist_term.add(const_term).squeeze(-1), alpha=0.5)
        return induc_loc_s, mean_coef, induc_prec, induc_nll

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        if mode:
            self.induc_loc_s = self.mean_coef = self.induc_prec = self.induc_nll = None
        else:
            self.induc_loc_s, self.mean_coef, self.induc_prec, self.induc_nll = self.induc_const()

    def forward(self, x_mean: Tensor, x_var: Tensor | None = None) -> tuple[Tensor, Tensor]:
        # x_mean, x_var ~ [..., (Q), B]
        # -> f_mean, f_var ~ [..., B]
        ls = self.kernel.length_scale
        os = self.kernel.output_scale
        data_mean = self.mean(x_mean)
        data_var = x_mean.new_ones(())
        if self.training:
            induc_mean = self.mean(self.induc_loc)
            induc_loc_s = self.induc_loc.mul(ls)
            induc_cov = self.kernel.cov_func(induc_loc_s, induc_loc_s)
            induc_cov = F.add_diag(induc_cov, self.induc_noise)
            induc_chol, _ = cholesky_ex(induc_cov)
            cross_cov = self.kernel.cov_func(induc_loc_s, x_mean.mul(ls))
            cross_cov_w = solve_triangular(induc_chol, cross_cov, upper=False)
            induc_shift = self.induc_value.sub(induc_mean).unsqueeze(-1)
            induc_shift_w = solve_triangular(induc_chol, induc_shift, upper=False)
            mean_shift = induc_shift_w.mT.matmul(cross_cov_w).squeeze(-2)
            var_reduc = cross_cov_w.square().sum(-2)
            logdet_term = F.diag(induc_chol).log().sum(-1)
            dist_term = induc_shift_w.square().sum(-2).div(os)
            const_term = os.log().add(C.LN_2PI).mul(self.num_induc)
            self.induc_nll = logdet_term.add(dist_term.add(const_term).squeeze(-1), alpha=0.5)
        else:
            cross_cov = self.kernel.cov_func(self.induc_loc_s, x_mean.mul(ls))
            mean_shift = self.mean_coef.matmul(cross_cov).squeeze(-2)
            var_reduc = self.induc_prec.matmul(cross_cov).mul(cross_cov).sum(-2)
        if x_var is not None:
            update_prop = self.kernel.update_prop_func(x_var, ls)
            mean_shift = update_prop.mul(mean_shift)
            var_reduc = update_prop.square().mul(var_reduc)
        x_mean = data_mean.add(mean_shift)
        x_var = data_var.sub(var_reduc).mul(os)
        return x_mean, x_var

    def extra_repr(self) -> str:
        d = '' if self.dim is None else f'dim={self.dim}'
        s = f'size={self.size}' if self.size else ''
        m = f'num_induc={self.num_induc}'
        return ', '.join(filter(None, (d, s, m)))


class SGP(GP):

    def __init__(self, in_dim: int, out_dim: int, num_induc: int) -> None:
        super().__init__(in_dim, out_dim, num_induc)

        self.mean = AffineMean(in_dim, out_dim)
        with torch.no_grad():
            if self.dim > 64:
                ls = C.LENGTH_SCALE_INIT * 8 / self.dim ** 0.5
                self.kernel.length_scale.fill_(ls)
            _os = F.inv_softplus(C.OUTPUT_SCALE_INIT / self.size[0])
            self.kernel._output_scale.fill_(_os)
        self.init_induc_value()

    @torch.no_grad()
    def init_induc_loc(self, X: Tensor, max_iter: int = 100, tol: float = 1e-3) -> None:
        induc_loc, _ = F.kmeans(X, self.num_induc, self.size[0], max_iter, tol)
        self.induc_loc.copy_(induc_loc)
        self.init_induc_value()


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
        with torch.no_grad():
            self.mean.weight.copy_(torch.randn(out_dim, in_dim, 1).mul(in_dim**-0.5))
        self.init_induc_value()


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


