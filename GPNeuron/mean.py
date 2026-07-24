
from abc import ABC, abstractmethod
from collections.abc import Callable
import torch
from torch import Tensor
from torch.nn import Module, Parameter


class Mean(Module, ABC):

    def __init__(self, dim: int | None, size: int | tuple[int, ...]) -> None:
        super().__init__()

        # Q
        self.dim = dim
        # [...]
        self.size = (size,) if isinstance(size, int) else size
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        ...

    def extra_repr(self) -> str:
        d = '' if self.dim is None else f'dim={self.dim}'
        s = f'size={self.size}' if self.size else ''
        return ', '.join(filter(None, (d, s)))


class ZeroMean(Mean):

    def __init__(self, size: int | tuple[int, ...]) -> None:
        super().__init__(None, size)

    def forward(self, x: Tensor) -> Tensor:
        # x ~ [..., (Q), B]
        # -> m(x) ~ [..., B]
        return torch.zeros(*self.size, x.size(-1), dtype=x.dtype, device=x.device)


class ActivMean(Mean):

    def __init__(self, func: Callable[[Tensor], Tensor] = torch.tanh) -> None:
        super().__init__(None, ())

        self.func = func

    def forward(self, x: Tensor) -> Tensor:
        # x ~ [..., B]
        # -> m(x) ~ [..., B]
        return self.func(x)
    
    def extra_repr(self) -> str:
        return 'func=' + self.func.__name__


class LinearMean(Mean):

    def __init__(self, size: int | tuple[int, ...]) -> None:
        super().__init__(None, size)

        # weight ~ [..., 1]
        self.weight = Parameter(torch.randn(*self.size, 1))

    def forward(self, x: Tensor) -> Tensor:
        # x ~ [..., B]
        # -> m(x) ~ [..., B]
        return self.weight.mul(x)


class AffineMean(Mean):

    def __init__(self, dim: int, size: int | tuple[int, ...]) -> None:
        super().__init__(dim, size)

        # weight ~ [..., 1, Q]
        self.weight = Parameter(torch.randn(*self.size, 1, dim).mul(dim**-0.5))
        # bias ~ [..., 1]
        self.bias = Parameter(torch.zeros(*self.size, 1))

    def forward(self, x: Tensor) -> Tensor:
        # x ~ [..., Q, B]
        # -> m(x) ~ [..., B]
        return self.weight.matmul(x).squeeze(-2).add(self.bias)


