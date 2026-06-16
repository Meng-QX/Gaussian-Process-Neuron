
import math
import torch
from torch import Tensor
from torch.special import erf

from . import const as C


def inv_softplus(x: float) -> float:
    return math.log(math.expm1(x))


def normal_expect_abs(mean: Tensor, var: Tensor) -> Tensor:
    std = var.sqrt()
    quot = mean.div(std).mul(C.RECIP_SQRT_2)
    return quot.square().neg().exp().mul(std).mul(C.Z_MAD).add(erf(quot).mul(mean))


def normal_log_prob(x: Tensor, mean: Tensor, var: Tensor) -> Tensor:
    return x.sub(mean).square().div(var).add(var.log()).add(C.LN_2PI).mul(-0.5)


def normal_crps(x: Tensor, mean: Tensor, var: Tensor) -> Tensor:
    term1 = normal_expect_abs(mean.sub(x), var)
    term2 = normal_expect_abs(torch.zeros(()), var.mul(2))
    return term1.sub(term2, alpha=0.5)


def log_sum_exp(x: Tensor, weight: Tensor | None = None) -> Tensor:
    with torch.no_grad():
        c = x.max(0).values
    x = x.sub(c).exp()
    x = x.mean(0) if weight is None else x.mul(weight).sum(0)
    x = x.log().add(c)
    return x


def normal_mixture_log_prob(
    x: Tensor, mean: Tensor, var: Tensor, weight: Tensor | None = None,
) -> Tensor:
    return log_sum_exp(normal_log_prob(x, mean, var), weight)


def normal_mixture_crps(
    x: Tensor, mean: Tensor, var: Tensor, weight: Tensor | None = None,
) -> Tensor:
    term1 = normal_expect_abs(mean.sub(x), var)
    mean_diff = mean.unsqueeze(1).sub(mean.unsqueeze(0))
    var_sum = var.unsqueeze(1).add(var.unsqueeze(0))
    term2 = normal_expect_abs(mean_diff, var_sum)
    if weight is None:
        term1 = term1.mean(0)
        term2 = term2.mean((0, 1))
    else:
        term1 = term1.mul(weight).sum(0)
        weight_outer = weight.unsqueeze(1).mul(weight.unsqueeze(0))
        term2 = term2.mul(weight_outer).sum((0, 1))
    return term1.sub(term2, alpha=0.5)


def normal_mixture_sampl(mean: Tensor, std: Tensor, weight: Tensor | None = None) -> Tensor:
    if weight is None:
        weight = torch.ones(mean.size(0), 1, mean.size(-1)).div(mean.size(0))
    r = torch.rand(1, *mean.size()[1:])
    cum_weight = weight.cumsum(0)
    mask = (r < cum_weight).to(r.dtype)
    mask[1:] = mask[1:].sub(mask[:-1])
    mean = mean.mul(mask).sum(0)
    std = std.mul(mask).sum(0)
    sampl = torch.randn(std.size()).mul(std).add(mean)
    return sampl


def dist_sq_1d(a: Tensor, b: Tensor) -> Tensor:
    return a.unsqueeze(-1).sub(b.unsqueeze(-2)).square()


def dist_sq_nd(a: Tensor, b: Tensor) -> Tensor:
    a_sq = a.square().sum(-2).unsqueeze(-1)
    b_sq = b.square().sum(-2).unsqueeze(-2)
    prod = a.mT.matmul(b)
    return a_sq.add(b_sq).sub(prod, alpha=2)


def diag(matrix: Tensor) -> Tensor:
    return matrix.diagonal(dim1=-1, dim2=-2)


def add_diag(matrix: Tensor, value: Tensor | float) -> Tensor:
    return torch.eye(matrix.size(-1)).mul(value).add(matrix)


def kmeans(
    X: Tensor, num_centroid: int, num_run: int, max_iter: int = 100, tol: float = 1e-3,
) -> Tensor:
    D, N = X.size()
    centroid = torch.empty(num_run, D, num_centroid)
    idx = torch.randint(0, N, (num_run,))
    centroid[..., 0] = X[:, idx].mT
    dist_sq_min = torch.full((num_run, N), torch.inf)
    for i in range(1, num_centroid):
        dist_sq = dist_sq_nd(centroid[..., i-1].mT, X).clamp_min(0)
        dist_sq_min = torch.min(dist_sq, dist_sq_min)
        idx = torch.multinomial(dist_sq_min, 1).squeeze(1)
        centroid[..., i] = X[:, idx].mT
    ones = torch.ones(num_run, N)
    for i in range(max_iter):
        _, labels = dist_sq_nd(centroid, X).min(1)
        labels_expand = labels.unsqueeze(1).expand(-1, D, -1)
        X_expand = X.expand(num_run, -1, -1)
        cluster_sum = torch.zeros(centroid.size()).scatter_add(-1, labels_expand, X_expand)
        count = torch.zeros(num_run, num_centroid).scatter_add(1, labels, ones)
        centroid_new = cluster_sum.div(count.clamp(min=1).unsqueeze(1))
        empty_mask = count == 0
        if empty_mask.any():
            empty_coord = empty_mask.nonzero()
            run_idx = empty_coord[:, 0]
            centroid_idx = empty_coord[:, 1]
            rand_idx = torch.randint(0, N, (empty_coord.size(0),))
            centroid_new[run_idx, :, centroid_idx] = X[:, rand_idx].mT
        converged = centroid_new.sub(centroid).abs().max() < tol
        centroid = centroid_new
        if converged:
            info = {"converged": True, "iterations": i + 1}
            break
    else:
        info = {"converged": False, "iterations": max_iter}
    return centroid, info


