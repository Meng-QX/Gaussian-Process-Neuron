
from collections.abc import Callable
import torch
from torch import Tensor
from torch.nn import Parameter

from .import func as F
from .base import Network
from .layer import SGP, FCGP, ICGP, SICGP, Affine


class DGP(Network):

	def __init__(self, dims: list[int], num_induc: int) -> None:
		Network.__init__(self, dims)

		for i in range(len(dims)-1):
			self.layers.append(SGP(dims[i], dims[i+1], num_induc))

	@torch.no_grad()
	def init_induc_loc(self, X_mean: Tensor, batch_size: int = 256) -> None:
		sample_size = X_mean.size(0)
		X_var = None
		for layer in self.layers:
			layer.init_induc_loc(X_mean)
			F_mean = torch.empty(sample_size, layer.out_dim)
			F_var = torch.empty(F_mean.size())
			for start_idx in range(0, sample_size, batch_size):
				end_idx = min(start_idx + batch_size, sample_size)
				idx = range(start_idx, end_idx)
				x_mean = X_mean[idx].mT
				x_var = None if X_var is None else X_var[idx].mT
				f_mean, f_var = layer.forward(x_mean, x_var)
				F_mean[idx] = f_mean.mT
				F_var[idx] = f_var.mT
			X_mean = F_mean
			X_var = F_var


class GPKAN(Network):

	def __init__(self, dims: list[int], num_induc: int) -> None:
		Network.__init__(self, dims)

		for i in range(len(dims)-1):
			self.layers.append(FCGP(dims[i], dims[i+1], num_induc))


class GPLAN(Network):

	def __init__(
		self, dims: list[int], num_induc: int,
		shared: bool = True, mean_func: Callable[[Tensor], Tensor] = torch.tanh,
	) -> None:
		Network.__init__(self, dims)

		for i in range(len(dims)-1):
			if i:
				self.layers.append(
					SICGP(num_induc, mean_func=mean_func)
					if shared
					else ICGP(dims[i], num_induc, mean_func=mean_func)
				)
			self.layers.append(Affine(dims[i], dims[i+1]))


class DGMP(Network):

	def __init__(self, dims: list[int], num_induc: int, num_comp: int = 1) -> None:
		Network.__init__(self, dims)

		self.num_comp = num_comp  # K
		# mix_weight ~ [K, 1, D^L]
		self._mix_weight = Parameter(torch.zeros(num_comp, 1, dims[-1]))
		for i in range(len(dims)-1):
			k = num_comp if i == len(dims) - 2 else 1
			self.layers.append(SGP(dims[i], dims[i+1]*k, num_induc))
	
	@property
	def mix_weight(self) -> Tensor:
		return self._mix_weight.softmax(0)

	def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
		# x ~ [B, D^0]
		# -> f ~ [K, B, D^L]
		x_mean = x.mT
		x_var = None
		for layer in self.layers:
			x_mean, x_var = layer.forward(x_mean, x_var)
		x_mean = x_mean.view(self.num_comp, self.dims[-1], -1).mT
		x_var = x_var.view(self.num_comp, self.dims[-1], -1).mT
		return x_mean, x_var

	def ell(self, y: Tensor, f_mean: Tensor, f_var: Tensor, weight: Tensor) -> Tensor:
		obs_noise = self.obs_noise
		ll = F.normal_log_prob(y, f_mean, obs_noise).sub(f_var.div(obs_noise), alpha=0.5)
		ll = ll.mul(weight).sum(0)
		return ll
	
	def mll(self, y: Tensor, f_mean: Tensor, f_var: Tensor, weight: Tensor) -> Tensor:
		return F.normal_mixture_log_prob(y, f_mean, f_var.add(self.obs_noise), weight)
	
	def loglikelihood(self, x: Tensor, y: Tensor, mll: bool = True) -> Tensor:
		# x ~ [B, D^0], y ~ [B, D^L]
		# -> ll ~ []
		ll = self.mll if mll else self.ell
		return ll(y, *self.forward(x), self.mix_weight).sum(-1).mean()

	@torch.no_grad()
	def evaluate(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
		# x ~ [B, D^0], y ~ [B, D^L]
		# -> f_mean, nll, crps ~ [B, D^L]
		f_mean, f_var = self.forward(x)
		f_var = f_var.add(self.obs_noise)
		weight = self.mix_weight
		nll = F.normal_mixture_log_prob(y, f_mean, f_var, weight).neg()
		crps = F.normal_mixture_crps(y, f_mean, f_var, weight)
		f_mean = f_mean.mul(weight).sum(0)
		return f_mean, nll, crps

	@torch.no_grad()
	def pred_sampl(self, x: Tensor) -> Tensor:
		# x ~ [B, D^0]
		# -> y ~ [B, D^L]
		f_mean, f_var = self.forward(x)
		f_std = f_var.add(self.obs_noise).sqrt()
		return F.normal_mixture_sampl(f_mean, f_std, self.mix_weight)
	
	def init_induc_loc(self, X_mean: Tensor, batch_size: int = 256) -> None:
		DGP.init_induc_loc(self, X_mean, batch_size)

	def extra_repr(self) -> str:
		return Network.extra_repr(self) + f', num_comp={self.num_comp}'


