
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import softplus

from . import constant as C, function as F


class Likelihood(Module, ABC):

    def __init__(self, dim: int) -> None:
        super().__init__()

        # D^L
        self.dim = dim
        # obs_noise ~ [D^L]
        self._obs_noise = Parameter(torch.full((dim,), F.inv_softplus(C.OBS_NOISE_INIT)))

    @property
    @abstractmethod
    def obs_noise(self) -> Tensor:
        ...

    @abstractmethod
    def forward(
        self, y: Tensor, f_mean: Tensor, f_var: Tensor, mix_weight: Tensor | None = None,
    ) -> Tensor:
        ...

    @abstractmethod
    def metric(
        self, y: Tensor, f_mean: Tensor, f_var: Tensor, mix_weight: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        ...
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class GaussianLikelihood(Likelihood):

    @property
    def obs_noise(self) -> Tensor:
        return softplus(self._obs_noise).add(C.OBS_NOISE_MIN)

    def forward(
        self, y: Tensor, f_mean: Tensor, f_var: Tensor, mix_weight: Tensor | None = None,
    ) -> Tensor:
        # y ~ [B, D^L], f_mean, f_var ~ [(S), B, D^L], mix_weight ~ [S, 1, D^L]
        # -> ln(p(y|x)) ~ [B, D^L]
        f_var = f_var.add(self.obs_noise)
        if mix_weight is None:
            return F.normal_log_prob(y, f_mean, f_var)
        return F.normal_mixture_log_prob(y, f_mean, f_var, mix_weight)
    
    def metric(
        self, y: Tensor, f_mean: Tensor, f_var: Tensor, mix_weight: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # y ~ [B, D^L], f_mean, f_var ~ [(S), B, D^L], mix_weight ~ [S, 1, D^L]
        # -> mean, nll, crps ~ [B, D^L]
        f_var = f_var.add(self.obs_noise)
        if mix_weight is None:
            nll = F.normal_log_prob(y, f_mean, f_var).neg()
            crps = F.normal_crps(y, f_mean, f_var)
        else:
            nll = F.normal_mixture_log_prob(y, f_mean, f_var, mix_weight).neg()
            crps = F.normal_mixture_crps(y, f_mean, f_var, mix_weight)
            f_mean = f_mean.mul(mix_weight).sum(0)
        return f_mean, nll, crps


class BernoulliLikelihood(Likelihood):

    def __init__(self, dim: int, num_node: int = 16) -> None:
        super().__init__(dim)

        node, weight = np.polynomial.hermite_e.hermegauss(num_node)
        weight = weight / np.sqrt(np.pi * 2)
        kwargs = {'dtype': self._obs_noise.dtype, 'device': self._obs_noise.device}
        self.register_buffer('node', torch.tensor(node, **kwargs), persistent=False)
        self.register_buffer('weight', torch.tensor(weight, **kwargs), persistent=False)
        with torch.no_grad():
            self._obs_noise.fill_(torch.logit(torch.tensor(0.1)))

    @property
    def obs_noise(self) -> Tensor:
        return self._obs_noise.sigmoid().mul(0.9998).add(1e-4)
    
    def prob(
        self, y: Tensor, f_mean: Tensor, f_var: Tensor, mix_weight: Tensor | None = None,
    ) -> Tensor:
        # y ~ [B, D^L], f_mean, f_var ~ [(S), B, D^L], mix_weight ~ [S, 1, D^L]
        # -> p(y=1|x) ~ [B, D^L]
        obs_noise = self.obs_noise.unsqueeze(-1)
        y = y.unsqueeze(-1)
        f_mean = f_mean.unsqueeze(-1)
        f_var = f_var.unsqueeze(-1)
        f_sample = f_var.sqrt().mul(self.node).add(f_mean)
        prob = f_sample.mul(y.mul(2).sub(1)).sigmoid()
        prob = obs_noise.sub(prob.mul(obs_noise.mul(2).sub(1)))
        prob = prob.mul(self.weight).sum(-1)
        if mix_weight is not None:
            prob = prob.mul(mix_weight).sum(0)
        return prob

    def forward(
        self, y: Tensor, f_mean: Tensor, f_var: Tensor, mix_weight: Tensor | None = None,
    ) -> Tensor:
        # y ~ [B, D^L], f_mean, f_var ~ [(S), B, D^L], mix_weight ~ [S, 1, D^L]
        # -> ln(p(y|x)) ~ [B, D^L]
        return self.prob(y, f_mean, f_var, mix_weight).log()

    def metric(
        self, y: Tensor, f_mean: Tensor, f_var: Tensor, mix_weight: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # y ~ [B, D^L], f_mean, f_var ~ [(S), B, D^L], mix_weight ~ [S, 1, D^L]
        # -> prob, nll, rps ~ [B, D^L]
        prob = self.prob(y, f_mean, f_var, mix_weight)
        nll = prob.log().neg()
        rps = prob.sub(1).square().mul(2)
        return prob, nll, rps


