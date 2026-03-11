
from collections.abc import Callable
import torch
from torch import Tensor
from torch.nn import Module, Parameter


class ZeroMean(Module):

	def __init__(self) -> None:
		Module.__init__(self)

	def forward(self, x: Tensor) -> Tensor:
		# x ~ [Q, B]
		# -> m(x) ~ [Q, B]
		return torch.zeros(x.size())


class AffineMean(Module):

	def __init__(self, in_dim: int, out_dim: int) -> None:
		Module.__init__(self)

		# weight ~ [D, 1, Q]
		self.weight = Parameter(torch.randn(out_dim, 1, in_dim).mul(in_dim**-0.5))
		# bias ~ [D, 1, 1]
		self.bias = Parameter(torch.zeros(out_dim, 1, 1))

	def forward(self, x: Tensor) -> Tensor:
		# x ~ [(D), Q, B]
		# m(x) -> [D, B]
		return self.weight.matmul(x).add(self.bias).squeeze(1)


class LinearMean(Module):

	def __init__(self, in_dim: int, out_dim: int) -> None:
		Module.__init__(self)

		# weight ~ [D, Q, 1]
		self.weight = Parameter(torch.randn(out_dim, in_dim, 1).mul(in_dim**-0.5))

	def forward(self, x: Tensor) -> Tensor:
		# x ~ [(D), Q, B]
		# -> m(x) ~ [D, Q, B]
		return self.weight.mul(x)


class ActivMean(Module):

	def __init__(self, func: Callable[[Tensor], Tensor] = torch.tanh) -> None:
		Module.__init__(self)

		self.func = func

	def forward(self, x: Tensor) -> Tensor:
		# x ~ [Q, B]
		# -> m(x) ~ [Q, B]
		return self.func(x)
	
	def extra_repr(self) -> str:
		return 'func=' + self.func.__name__


