
import torch
from torch import Tensor
from torch.linalg import cholesky_ex, solve_triangular
from torch.nn import Module, Parameter
from torch.nn.functional import softplus

from . import const as C, func as F


class Kernel(Module):

    def __init__(self, dim: int | None, size: int | tuple[int, ...]) -> None:
        super().__init__()

        # (Q)
        self.dim = dim
        # [...]
        self.size = size = (size,) if isinstance(size, int) else tuple(size)
        # lengthscale ~ [..., (Q), 1]
        ls_size = size + (1,) if dim is None else size + (dim, 1)
        self.lengthscale = Parameter(torch.full(ls_size, C.LENGTHSCALE_INIT))
        # outputscale ~ [..., 1]
        os_size = size + (1,)
        self._outputscale = Parameter(torch.full(os_size, F.inv_softplus(C.OUTPUTSCALE_INIT)))
        # induc_noise ~ []
        self._induc_noise = Parameter(torch.tensor(F.inv_softplus(C.INDUC_NOISE_INIT)))

    @property
    def outputscale(self) -> Tensor:
        return softplus(self._outputscale).add(C.OUTPUTSCALE_MIN)
    
    @property
    def induc_noise(self) -> Tensor:
        return softplus(self._induc_noise).add(C.INDUC_NOISE_MIN)
    
    @staticmethod
    def dist_sq(a: Tensor, b: Tensor) -> Tensor:
        ...

    def induc_cholesky_factor(self, z: Tensor) -> Tensor:
        # z ~ [..., (Q), M]
        # -> L ~ [..., M, M]
        z = z.mul(self.lengthscale)
        Kuu = self.dist_sq(z, z).neg().exp()
        L, _ = cholesky_ex(F.add_diag(Kuu, self.induc_noise))
        return L
    
    def cov(self, z: Tensor, x_mean: Tensor) -> tuple[Tensor, Tensor]:
        # z ~ [..., (Q), M], x_mean ~ [..., (Q), B]
        # -> Kuu ~ [..., M, M], Kuf ~ [..., M, B]
        z = z.mul(self.lengthscale)
        Kuu = self.dist_sq(z, z).neg().exp()
        x_mean = x_mean.mul(self.lengthscale)
        Kuf = self.dist_sq(z, x_mean).neg().exp()
        return Kuu, Kuf

    def alpha(self, x_var: Tensor) -> Tensor:
        ...

    def sparse_approx(
        self, L: Tensor, Kuf: Tensor, alpha: Tensor | None, u: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # L ~ [..., M, M], Kuf ~ [..., M, B], alpha ~ [..., B], u ~ [..., M]
        # -> f_mean, f_var ~ [..., B], u_nll ~ [...]
        os = self.outputscale
        Kuf = solve_triangular(L, Kuf, upper=False)
        u = solve_triangular(L, u.unsqueeze(-1), upper=False)
        f_mean = u.mT.matmul(Kuf).squeeze(-2)
        Qf = Kuf.square().sum(-2)
        if alpha is not None:
            f_mean = f_mean.mul(alpha)
            Qf = Qf.mul(alpha.square())
        f_var = torch.ones(()).sub(Qf).mul(os)
        Md2 = L.size(-1) * 0.5
        quad_term = u.square().sum(-2).div(os).mul(0.5)
        logdet_term = F.diag(L).log().sum(-1, keepdim=True).add(os.log(), alpha=Md2)
        pi_term = C.LN_2PI * Md2
        u_nll = quad_term.add(logdet_term).squeeze(-1).add(pi_term)
        return f_mean, f_var, u_nll
    
    def forward(
        self, x_mean: Tensor, x_var: Tensor | None, z: Tensor, u: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # x_mean, x_var ~ [..., (Q), B], z ~ [..., (Q), M], u ~ [..., M]
        # -> f_mean, f_var ~ [..., B], u_nll ~ [...]
        Kuu, Kuf = self.cov(z, x_mean)
        L, _ = cholesky_ex(F.add_diag(Kuu, self.induc_noise))
        alpha = None if x_var is None else self.alpha(x_var)
        return self.sparse_approx(L, Kuf, alpha, u)

    def extra_repr(self) -> str:
        d = '' if self.dim is None else f'dim={self.dim}'
        s = f'size={self.size}' if self.size else ''
        return ', '.join(filter(None, (d, s)))


class Kernel1D(Kernel):

    def __init__(self, size: int | tuple[int, ...]) -> None:
        super().__init__(None, size)

    @staticmethod
    def dist_sq(a: Tensor, b: Tensor) -> Tensor:
        return F.dist_sq_1d(a, b)

    def alpha(self, x_var: Tensor) -> Tensor:
        # x_var ~ [..., B]
        # -> alpha ~ [..., B]
        return self.lengthscale.square().mul(x_var).neg().exp()


class KernelND(Kernel):

    def __init__(self, dim: int, size: int | tuple[int, ...]) -> None:
        super().__init__(dim, size)

    @staticmethod
    def dist_sq(a: Tensor, b: Tensor) -> Tensor:
        return F.dist_sq_nd(a, b)

    def alpha(self, x_var: Tensor) -> Tensor:
        # x_var ~ [..., Q, B]
        # -> alpha ~ [..., B]
        return self.lengthscale.square().mT.matmul(x_var).squeeze(-2).neg().exp()


