"""Alternative densities for HyperMSDE (the ``density_fn`` of ``HypersphericalMeanShiftDensityEnhancement.forward``).

The default density, ``Density`` (adaptive angular k-NN), is in ``hypermsde.py``. ``Original`` here is the density of
the HyperMSDE paper (arXiv:2602.03293), counted in the space of the rows of the fuzzy neighbour graph. It has the
signature ``density(x, density, stability, iteration, n_iters, graph, memory) -> (density, memory)`` and depends on
the initial points only, so it is ``static``: ``forward`` evaluates it once, at the first iteration.
"""
import torch

from .hypermsde import DEFAULT_CHUNK_SIZE, Component, _graph_layout, _similarity_graph


# ---------------------------------------------------------------------------------------------------------
# The paper's density: counting neighbours in the space of the rows of the fuzzy graph
# ---------------------------------------------------------------------------------------------------------
def _make_pairwise_dist_kernel():
    def _kernel(row_norm_sq, idx, cross):
        sq_dist = row_norm_sq[:, None] + row_norm_sq[idx][None, :] - 2.0 * cross
        return torch.sqrt(torch.clamp(sq_dist, min=0.0))
    return torch.compile(_kernel, fullgraph=True)


_PAIRWISE_DIST_KERNEL = _make_pairwise_dist_kernel()


def _graph_chunk_dist(S, layout, start, end):
    """Graph-space distances from the rows ``start:end`` to all rows. The cross terms are one sparse-times-dense
    matmul, ``S @ chunk.T``, which is where the density computation spends its time."""
    rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq = layout
    n = S.shape[0]
    lo, hi = row_ptr[start], row_ptr[end]
    chunk_dense = torch.zeros((end - start, n), dtype=row_norm_sq.dtype, device=S.device)
    chunk_dense[rows_sorted[lo:hi] - start, cols_sorted[lo:hi]] = vals_sorted[lo:hi]
    cross = torch.sparse.mm(S, chunk_dense.T)
    idx = torch.arange(start, end, device=S.device)
    return _PAIRWISE_DIST_KERNEL(row_norm_sq, idx, cross), idx


def _radius_counts(S, layout, radius, chunk_size):
    """For every row, the number of other rows within ``radius`` in graph space."""
    n = S.shape[0]
    counts = torch.zeros(n, dtype=torch.float32, device=S.device)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        dist, _ = _graph_chunk_dist(S, layout, start, end)
        counts[start:end] = (dist < radius).sum(dim=0).float() - 1.0
    return counts


def _graph_distance_range(S, layout, chunk_size):
    """Global (max, min) distance between two different rows in graph space."""
    n = S.shape[0]
    running_max = torch.tensor(float("-inf"), device=S.device)
    running_min = torch.tensor(float("inf"), device=S.device)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        dist, idx = _graph_chunk_dist(S, layout, start, end)
        self_mask = torch.zeros_like(dist, dtype=torch.bool)
        self_mask[idx, torch.arange(idx.shape[0], device=S.device)] = True
        running_max = torch.maximum(running_max, dist.masked_fill(self_mask, float("-inf")).max())
        running_min = torch.minimum(running_min, dist.masked_fill(self_mask, float("inf")).min())
    return running_max.item(), running_min.item()


def _search_radius(S, layout, low, high, threshold, required, chunk_size, tol=1e-4, max_iter=50):
    """Smallest radius at which at least ``required`` rows have more than ``threshold`` others inside it."""
    lo, hi, result, found = low, high, high, False
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        counts = _radius_counts(S, layout, torch.tensor(mid, device=S.device), chunk_size)
        if int((counts > threshold).sum().item()) >= required:
            result, found, hi = mid, True, mid
        else:
            lo = mid
        if abs(hi - lo) < tol:
            break
    return result, found


@torch.no_grad()
def graph_density(S, nbd_sample_count_threshold=70, max_iters_weight_count=4, satisfiability_proportion=0.3,
                  chunk_size=DEFAULT_CHUNK_SIZE):
    """Empirical density of every row of the fuzzy graph ``S`` (an (n, n) coalesced sparse tensor), as an (n,)
    tensor (larger = denser).

    The rows of ``S`` are compared as sparse vectors (sparse matmul), ``chunk_size`` rows at a time. The radius
    is the smallest one at which at least ``satisfiability_proportion`` of the points have more than
    ``nbd_sample_count_threshold`` others inside it (found by bisection, relaxed once if that fails). The density
    of a point is its number of others within the radius, averaged over ``max_iters_weight_count`` radii
    shrinking linearly to zero.
    """
    n = S.shape[0]
    layout = _graph_layout(S)

    max_dist, min_dist = _graph_distance_range(S, layout, chunk_size)
    threshold = max(1, n - 1) if nbd_sample_count_threshold >= n else nbd_sample_count_threshold
    required = int(satisfiability_proportion * n)
    radius, found = _search_radius(S, layout, min_dist, max_dist, threshold, required, chunk_size)
    if not found:
        radius, found = _search_radius(S, layout, min_dist, max_dist, max(1, threshold // 2), required // 2, chunk_size)
    if not found:
        radius = max_dist
    radius = torch.tensor(radius, dtype=torch.float32, device=S.device)

    delta = (radius - 1e-6) / max_iters_weight_count
    total = torch.zeros(n, dtype=torch.float32, device=S.device)
    for _ in range(max_iters_weight_count):
        total = total + _radius_counts(S, layout, radius, chunk_size)
        radius = radius - delta
    return total / max_iters_weight_count


@torch.no_grad()
def density_weights(X, nbd_sample_count_threshold=70, n_neighbors=15, n_epochs=200, max_iters_weight_count=4,
                    satisfiability_proportion=0.3, chunk_size=DEFAULT_CHUNK_SIZE, metric="euclidean",
                    return_graph=False):
    """Empirical density of every row of ``X``, returned as an (n,) tensor (larger = denser).

    A fuzzy ``n_neighbors``-nearest-neighbour graph of ``X`` (the UMAP fuzzy simplicial set: per-point scale,
    fuzzy-union symmetrised weights in (0, 1], weak edges dropped) is built, by Euclidean or cosine ``metric``,
    and its density is computed by ``graph_density``. ``chunk_size`` is the number of rows handled at a time,
    both by the exact neighbour search that builds the graph and by the graph-space distances; it bounds the
    memory of the whole computation. With ``return_graph`` the result is ``(density, graph)`` where ``graph`` is
    that (n, n) coalesced sparse tensor.
    """
    S = _similarity_graph(X, n_neighbors, n_epochs, chunk_size, metric)
    density = graph_density(S, nbd_sample_count_threshold, max_iters_weight_count, satisfiability_proportion, chunk_size)
    return (density, S) if return_graph else density


class Original(Component):
    """The density of the HyperMSDE paper (arXiv:2602.03293), as a density component (see ``graph_density``).

    The rows of the fuzzy ``graph`` are compared as sparse vectors. The radius is the smallest one at which at
    least ``satisfiability_proportion`` of the points have more than ``nbd_sample_count_threshold`` others inside
    it; a point's density is its number of others within the radius, averaged over ``max_iters_weight_count``
    radii shrinking linearly to zero. ``chunk_size`` is the number of rows handled at a time.
    """

    static = True

    def __init__(self, nbd_sample_count_threshold=70, max_iters_weight_count=4, satisfiability_proportion=0.3,
                 chunk_size=DEFAULT_CHUNK_SIZE, static=None):
        super().__init__(static)
        self.nbd_sample_count_threshold = nbd_sample_count_threshold
        self.max_iters_weight_count = max_iters_weight_count
        self.satisfiability_proportion = satisfiability_proportion
        self.chunk_size = chunk_size

    def __call__(self, x, density, stability, iteration, n_iters, graph, memory):
        counts = graph_density(graph, self.nbd_sample_count_threshold, self.max_iters_weight_count,
                               self.satisfiability_proportion, self.chunk_size)
        return counts.clamp_min(0.0).to(x.dtype), memory
