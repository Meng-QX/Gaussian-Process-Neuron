
import torch
from torch import Tensor
from torch.linalg import cholesky_ex, solve_triangular
from torch.special import erf

from . import const as C


def normal_expect_abs(mean: Tensor, var: Tensor) -> Tensor:
	std = var.sqrt()
	quot = mean.div(std).mul(C.RECIP_SQRT_2)
	return quot.square().neg().exp().mul(std).mul(C.Z_MAD).add(erf(quot).mul(mean))


def normal_log_prob(x: Tensor, mean: Tensor, var: Tensor) -> Tensor:
	return x.sub(mean).square().div(var).add(var.log()).add(C.LN_2PI).mul(-0.5)


def normal_crps(x: Tensor, mean: Tensor, var: Tensor) -> Tensor:
	term1 = normal_expect_abs(mean.sub(x), var)
	term2 = normal_expect_abs(torch.zeros([]), var.mul(2))
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
		term2 = term2.mean([0,1])
	else:
		term1 = term1.mul(weight).sum(0)
		weight_outer = weight.unsqueeze(1).mul(weight.unsqueeze(0))
		term2 = term2.mul(weight_outer).sum([0,1])
	return term1.sub(term2, alpha=0.5)


def normal_mixture_sampl(mean: Tensor, std: Tensor, weight: Tensor | None = None) -> Tensor:
	if weight is None:
		weight = torch.ones(mean.size(0), 1, mean.size(-1)).div(mean.size(0))
	r = torch.rand(1, *mean.size()[1:])
	cum_weight = weight.cumsum(0)
	mask = (r < cum_weight).to(r.dtype)
	mask[1:] = mask[1:].sub(mask[:-1])
	mean = mean.mul(mask).sum(0)
	std  = std.mul(mask).sum(0)
	sampl = torch.randn(std.size()).mul(std).add(mean)
	return sampl


def sq_dist_1d(a: Tensor, b: Tensor) -> Tensor:
	return a.unsqueeze(-1).sub(b.unsqueeze(-2)).square()


def sq_dist_nd(a: Tensor, b: Tensor) -> Tensor:
	a_sq = a.square().sum(-2).unsqueeze(-1)
	b_sq = b.square().sum(-2).unsqueeze(-2)
	prod = a.mT.matmul(b)
	return a_sq.add(b_sq).sub(prod, alpha=2)


def diag(matrix: Tensor) -> Tensor:
	return matrix.diagonal(dim1=-1, dim2=-2)


def add_diag(matrix: Tensor, value: Tensor | float) -> Tensor:
	return torch.eye(matrix.size(-1)).mul(value).add(matrix)


def whiten(L: Tensor, *matrices: tuple[Tensor]) -> tuple[Tensor]:
	s = [matrix.size(-1) for matrix in matrices]
	return solve_triangular(L, torch.cat([*matrices], dim=-1), upper=False).split(s, dim=-1)


def psd_cholesky(A: Tensor, jitters: tuple[float] = C.DEFAULT_JITTERS) -> Tensor:
	for jitter in jitters:
		L, info = cholesky_ex(add_diag(A, jitter))
		if (info == 0).all() and not L.isnan().any():
			return L
	raise RuntimeError(f"Cholesky decompostion failed. Tried jitter values: {jitters}")


def batched_kmeans(
	X: Tensor, num_clusters: int, num_runs: int,
	batch_size: int = 256, max_iter: int = 100, tol: float = 1e-4,
) -> Tensor:
    N, Q = X.shape
    centroids = torch.zeros(num_runs, num_clusters, Q)
    first_idx = torch.randint(0, N, (num_runs,))
    centroids[:, 0, :] = X[first_idx]
    min_sq_dists = torch.full((num_runs, N), float('inf'))
    for i in range(1, num_clusters):
        current_centroid = centroids[:, i-1:i, :]
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            x_chunk = X[start_idx:end_idx]
            dist_chunk = torch.sum((x_chunk.unsqueeze(0) - current_centroid)**2, dim=2)
            min_sq_dists[:, start_idx:end_idx] = torch.min(
                min_sq_dists[:, start_idx:end_idx], dist_chunk
            )
        probs = min_sq_dists
        next_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
        centroids[:, i, :] = X[next_idx]
    ones_buffer = torch.ones(num_runs, batch_size)
    for i in range(max_iter):
        old_centroids = centroids.clone()
        new_centers_sum = torch.zeros_like(centroids)
        cluster_counts = torch.zeros(num_runs, num_clusters)
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            x_chunk = X[start_idx:end_idx] 
            current_bs = x_chunk.shape[0]
            x_expanded = x_chunk.unsqueeze(0).expand(num_runs, current_bs, Q)
            dists = torch.cdist(x_expanded, centroids)
            _, labels = torch.min(dists, dim=2)
            current_ones = ones_buffer[:, :current_bs]
            cluster_counts.scatter_add_(1, labels, current_ones)
            labels_expanded = labels.unsqueeze(-1).expand(num_runs, current_bs, Q)
            new_centers_sum.scatter_add_(1, labels_expanded, x_expanded)
        active_mask = (cluster_counts > 0)
        counts_safe = cluster_counts.unsqueeze(-1).clamp_min(1.0)
        proposed_centroids = new_centers_sum / counts_safe
        _, largest_cluster_idx = torch.max(cluster_counts, dim=1)
        idx_expanded = largest_cluster_idx.view(num_runs, 1, 1).expand(num_runs, 1, Q)
        source_centroids = torch.gather(proposed_centroids, 1, idx_expanded)
        if i == 0:
            noise_scale = 0.01 * torch.std(X, dim=0).mean()
        respawn_candidates = source_centroids + (torch.randn_like(centroids) * noise_scale)
        centroids = torch.where(active_mask.unsqueeze(-1), proposed_centroids, respawn_candidates)
        shift = torch.norm(centroids - old_centroids, dim=2).max()
        if shift < tol:
            break
    return centroids


