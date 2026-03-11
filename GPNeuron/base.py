
import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Parameter
from torch.nn.functional import softplus

from . import const as C, func as F


class Layer(Module):

	def __init__(self, in_dim: int, out_dim: int) -> None:
		Module.__init__(self)

		self.in_dim = in_dim  # Q
		self.out_dim = out_dim  # D
		self._induc_reg = torch.zeros([])

	@property
	def induc_reg(self) -> Tensor:
		return self._induc_reg
	
	@induc_reg.setter
	def induc_reg(self, reg: Tensor) -> None:
		self._induc_reg = reg
	
	def forward(self, x_mean: Tensor, x_var: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
		...

	def extra_repr(self) -> str:
		return f'in_dim={self.in_dim}, out_dim={self.out_dim}'


class Network(Module):

	def __init__(self, dims: list[int]) -> None:
		Module.__init__(self)

		# [D^0, ..., D^L]
		self.dims = dims
		# obs_noise ~ [D^L]
		self._obs_noise = Parameter(torch.ones(dims[-1]).mul(C.OBS_NOISE_INIT).expm1().log())
		self.layers = ModuleList()

	@property
	def obs_noise(self) -> Tensor:
		return softplus(self._obs_noise).add(C.OBS_NOISE_MIN)

	@property
	def induc_reg(self) -> Tensor:
		return sum(layer.induc_reg.sum() for layer in self.layers)

	def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
		# x ~ [B, D^0]
		# -> f ~ [B, D^L]
		x_mean = x.mT
		x_var = None
		for layer in self.layers:
			x_mean, x_var = layer.forward(x_mean, x_var)
		return x_mean.mT, x_var.mT

	def ell(self, y: Tensor, f_mean: Tensor, f_var: Tensor) -> Tensor:
		# expected log-likelihood E_{p(f|x)}[ln(p(y|f))]
		obs_noise = self.obs_noise
		return F.normal_log_prob(y, f_mean, obs_noise).sub(f_var.div(obs_noise), alpha=0.5)
	
	def mll(self, y: Tensor, f_mean: Tensor, f_var: Tensor) -> Tensor:
		# marginal log-likelihood ln(p(y|x))
		return F.normal_log_prob(y, f_mean, f_var.add(self.obs_noise))
	
	def loglikelihood(self, x: Tensor, y: Tensor, mll: bool = True) -> Tensor:
		# x ~ [B, D^0], y ~ [B, D^L]
		# -> ll ~ []
		ll = self.mll if mll else self.ell
		return ll(y, *self.forward(x)).sum(-1).mean()

	@torch.no_grad()
	def evaluate(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
		# x ~ [B, D^0], y ~ [B, D^L]
		# -> f_mean, nll, crps ~ [B, D^L]
		f_mean, f_var = self.forward(x)
		f_var = f_var.add(self.obs_noise)
		nll = F.normal_log_prob(y, f_mean, f_var).neg()
		crps = F.normal_crps(y, f_mean, f_var)
		return f_mean, nll, crps

	def extra_repr(self) -> str:
		return 'dims=[{}]'.format(','.join(str(d) for d in self.dims))


