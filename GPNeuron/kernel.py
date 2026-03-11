
import torch
from torch import Tensor
from torch.linalg import cholesky_ex
from torch.nn import Module, Parameter
from torch.nn.functional import softplus

from . import const as C, func as F


class Kernel(Module):

	def __init__(self, size: list[int]) -> None:
		Module.__init__(self)

		# outputscale ~ [(D), Q, 1]
		self._outputscale = Parameter(torch.ones(*size, 1).mul(C.OUTPUTSCALE_INIT).expm1().log())
		# lengthscale ~ [(D), Q, 1]
		self._lengthscale = Parameter(torch.ones(*size, 1).mul(C.LENGTHSCALE_INIT).expm1().log())
		# noise_var ~ []
		self._noise_var = Parameter(torch.ones([]).mul(C.INDUC_NOISE_INIT).expm1().log())
	
	@property
	def outputscale(self) -> Tensor:
		return softplus(self._outputscale).add(C.OUTPUTSCALE_MIN)

	@property
	def lengthscale(self) -> Tensor:
		return softplus(self._lengthscale).add(C.LENGTHSCALE_MIN)
	
	@property
	def noise_var(self) -> Tensor:
		return softplus(self._noise_var).add(C.INDUC_NOISE_MIN)
	
	@staticmethod
	def sq_dist(a: Tensor, b: Tensor) -> Tensor:
		...

	@staticmethod
	def alpha(ls: Tensor, x_var: Tensor) -> Tensor:
		...
	
	def cov(
		self, z: Tensor, x_mean: Tensor, x_var: Tensor | None = None,
	) -> tuple[Tensor, Tensor, Tensor | None]:
		# z ~ [(D), Q, M], x ~ [Q, B]
		# -> Kuu ~ [(D), Q, M, M], Kuf ~ [(D), Q, M, B], alpha ~ [(D), Q, B]
		ls = self.lengthscale
		z = z.mul(ls)
		x_mean = x_mean.mul(ls)
		Kuu = self.sq_dist(z, z).neg().exp()
		Kuf = self.sq_dist(z, x_mean).neg().exp()
		alpha = None if x_var is None else self.alpha(ls, x_var)
		return Kuu, Kuf, alpha

	def sparse_approx(
		self, L: Tensor, Kuf: Tensor, alpha: Tensor | None, u: Tensor,
	) -> tuple[Tensor, Tensor, Tensor]:
		# L ~ [(D), Q, M, M], Kuf ~ [(D), Q, M, B], alpha ~ [(D), Q, B], u ~ [(D), Q, M]
		# -> f ~ [(D), Q, B], u_nll ~ [(D), Q]
		os = self.outputscale
		Kuf, u = F.whiten(L, Kuf, u[..., None])
		f_mean = u.mT.matmul(Kuf).squeeze(-2)
		Qf = Kuf.square().sum(-2)
		if alpha is not None:
			f_mean = f_mean.mul(alpha)
			Qf = Qf.mul(alpha.square())
		f_var = torch.ones([]).sub(Qf).mul(os)
		Md2 = L.size(-1) * 0.5
		quad_term = u.mT.matmul(u).squeeze(-1).div(os).mul(0.5)
		logdet_term = F.diag(L).log().sum(-1, keepdim=True).add(os.log(), alpha=Md2)
		pi_term = C.LN_2PI * Md2
		u_nll = quad_term.add(logdet_term).squeeze(-1).add(pi_term)
		return f_mean, f_var, u_nll
	
	def forward(
		self, x_mean: Tensor, x_var: Tensor | None, z: Tensor, u: Tensor,
	) -> tuple[Tensor, Tensor, Tensor]:
		# x ~ [Q, B], z, u ~ [(D), Q, M]
		# -> f ~ [(D), Q, B], u_reg ~ [(D), Q]
		Kuu, Kuf, alpha = self.cov(z, x_mean, x_var)
		L, _ = cholesky_ex(F.add_diag(Kuu, self.noise_var))
		return self.sparse_approx(L, Kuf, alpha, u)


class Kernel1D(Kernel):

	@staticmethod
	def sq_dist(a: Tensor, b: Tensor) -> Tensor:
		return F.sq_dist_1d(a, b)

	@staticmethod
	def alpha(ls: Tensor, x_var: Tensor) -> Tensor:
		return ls.square().mul(x_var).neg().exp()


class KernelND(Kernel):

	def __init__(self, in_dim: int, out_dim: int) -> None:
		Kernel.__init__(self, [out_dim, in_dim])

		# outputscale ~ [D, 1]
		os = C.OUTPUTSCALE_INIT / out_dim
		self._outputscale.data = torch.ones(out_dim, 1).mul(os).expm1().log()
		ls = C.LENGTHSCALE_INIT * min(1, 10 / in_dim ** 0.5)
		self._lengthscale.data = torch.ones(out_dim, in_dim, 1).mul(ls).expm1().log()

	@staticmethod
	def sq_dist(a: Tensor, b: Tensor) -> Tensor:
		return F.sq_dist_nd(a, b)

	@staticmethod
	def alpha(ls: Tensor, x_var: Tensor) -> Tensor:
		return ls.square().mT.matmul(x_var).squeeze(-2).neg().exp()


