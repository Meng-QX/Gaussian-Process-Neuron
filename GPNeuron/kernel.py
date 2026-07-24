
from abc import ABC, abstractmethod
import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import softplus

from . import constant as C, function as F


class Kernel(Module, ABC):

    def __init__(self, dim: int | None, size: int | tuple[int, ...]) -> None:
        super().__init__()

        # Q
        self.dim = dim
        # [...]
        self.size = size = (size,) if isinstance(size, int) else size
        # length_scale ~ [..., (Q), 1]
        ls_size = size + (1,) if dim is None else size + (dim, 1)
        self._length_scale = Parameter(torch.full(ls_size, F.inv_softplus(C.LENGTH_SCALE_INIT)))
        # output_scale ~ [..., 1]
        os_size = size + (1,)
        self._output_scale = Parameter(torch.full(os_size, F.inv_softplus(C.OUTPUT_SCALE_INIT)))

    @property
    def length_scale(self) -> Tensor:
        return softplus(self._length_scale).add(C.LENGTH_SCALE_MIN)

    @property
    def output_scale(self) -> Tensor:
        return softplus(self._output_scale).add(C.OUTPUT_SCALE_MIN)
    
    @staticmethod
    @abstractmethod
    def cov_func(a: Tensor, b: Tensor) -> Tensor:
        ...
    
    @staticmethod
    @abstractmethod
    def update_prop_func(x_var: Tensor, ls: Tensor) -> Tensor:
        ...

    def forward(
        self, induc_loc: Tensor, x_mean: Tensor, x_var: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        # induc_loc ~ [..., (Q), M], x_mean, x_var ~ [..., (Q), B]
        # -> induc_cov ~ [..., M, M], cross_cov ~ [..., M, B], update_prop ~ [..., B]
        ls = self.length_scale
        induc_loc_s = induc_loc.mul(ls)
        induc_cov = self.cov_func(induc_loc_s, induc_loc_s)
        cross_cov = self.cov_func(induc_loc_s, x_mean.mul(ls))
        update_prop = None if x_var is None else self.update_prop_func(x_var, ls)
        return induc_cov, cross_cov, update_prop

    def extra_repr(self) -> str:
        d = '' if self.dim is None else f'dim={self.dim}'
        s = f'size={self.size}' if self.size else ''
        return ', '.join(filter(None, (d, s)))


class Kernel1D(Kernel):

    def __init__(self, size: int | tuple[int, ...]) -> None:
        super().__init__(None, size)

    @staticmethod
    def cov_func(a: Tensor, b: Tensor) -> Tensor:
        # a ~ [..., M], b ~ [..., B]
        # -> cov ~ [..., M, B]
        return F.dist_sq_1d(a, b).neg().exp()
    
    @staticmethod
    def update_prop_func(x_var: Tensor, ls: Tensor) -> Tensor:
        # x_var ~ [..., B], ls ~ [..., 1]
        # -> update_prop ~ [..., B]
        return ls.square().mul(x_var).neg().exp()


class KernelND(Kernel):

    def __init__(self, dim: int, size: int | tuple[int, ...]) -> None:
        super().__init__(dim, size)

    @staticmethod
    def cov_func(a: Tensor, b: Tensor) -> Tensor:
        # a ~ [..., Q, M], b ~ [..., Q, B]
        # -> cov ~ [..., M, B]
        return F.dist_sq_nd(a, b).neg().exp()
    
    @staticmethod
    def update_prop_func(x_var: Tensor, ls: Tensor) -> Tensor:
        # x_var ~ [..., Q, B], ls ~ [..., Q, 1]
        # -> update_prop ~ [..., B]
        return ls.square().mT.matmul(x_var).squeeze(-2).neg().exp()


