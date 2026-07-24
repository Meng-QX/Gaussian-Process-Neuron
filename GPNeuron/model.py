
import torch
from torch import Tensor
from torch.nn import Module

from .network import Network
from .likelihood import Likelihood


class Model(Module):

    def __init__(self, network: Network, likelihood: Likelihood) -> None:
        super().__init__()

        self.network = network
        self.likelihood = likelihood

    def penalized_loglikelihood(self, x: Tensor, y: Tensor, sample_size: int) -> Tensor:
        # x ~ [B, ...], y ~ [B, D^L]
        # -> ln(p(y|x))+ln(p(u|z)) ~ []
        f_mean, f_var = self.network(x)
        mix_weight = self.network.mix_weight if f_mean.ndim > y.ndim else None
        ll = self.likelihood(y, f_mean, f_var, mix_weight)
        return ll.sum(-1).mean().sub(self.network.induc_nll, alpha=sample_size**-1)
    
    @torch.no_grad()
    def evaluate(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # x ~ [B, ...], y ~ [B, D^L]
        # -> f_mean | prob, nll, crps | rps ~ [B, D^L]
        f_mean, f_var = self.network(x)
        mix_weight = self.network.mix_weight if f_mean.ndim > y.ndim else None
        return self.likelihood.metric(y, f_mean, f_var, mix_weight)


