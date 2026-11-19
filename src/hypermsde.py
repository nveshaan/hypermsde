"""Mean-shift density enhancement (HyperMSDE), on the unit sphere or in the raw feature space, (the density functions are in ``density.py``).

Every point of an embedding set is repeatedly moved towards a weighted mean of its neighbours until the
movement converges. The pipeline, in the order ``HypersphericalMeanShiftDensityEnhancement.forward`` runs it:

1. **Normalise** (``sphere``; ``forward(sphere={...})`` can give the graph, density, score and weight
   their own frozen space): 0 leaves the embeddings as they are and the loop runs in the raw space. 1 scales
   them to unit length and re-normalises them right after every shift, so every state the loop passes through
   (and the output) lies on the sphere. 2 does the same after first subtracting the mean of the points and
   normalising again (``unit(x - mean)``). 3 does the same after first centring the points on their Fréchet
   mean: each is mapped to the tangent space there (log map), which removes the shared mean direction, and put
   back on the sphere.
2. **Density**: a density per point, by default the adaptive angular k-NN estimate ``Density``,
   re-estimated on the current points at every iteration (pass ``Density(static=True)``, or any static ``density_fn``, to
   compute it once, at the first iteration).
   ``src.density.Original`` is the density of the paper: points are compared as rows of the fuzzy
   k-nearest-neighbour graph (built here, from the initial points), and a point's density is the average number
   of other points within a radius, over a few shrinking radii.
3. **Select neighbours**: every other point is scored by ``Score`` (distance x density x stability)
   and the ``k``
   best-scoring ones are kept. A dense, stable point can outrank a closer, sparse one.
4. **Step**: each point moves a fraction ``learning_rate`` towards the weighted mean of its selected neighbours
   (weights from ``Weight``), computed with one sparse matmul ``W @ X``, plus the bias of the support
   points if any are given (``bias_fn``); with ``sphere`` it is then put back on the sphere.
5. **Repeat** (optionally re-selecting neighbours every ``recompute_neighbors`` iterations) until the mean
   per-point movement drops below ``shift_threshold`` or ``max_iters_shift`` is reached.

Stability scores (values in [0, 1], higher = more stable) start at 1.0 for every point. Pass a ``stability_fn`` to
``forward`` and they are updated after every iteration from the previous and moved positions; the updated scores feed
the next neighbour weights and, whenever neighbours are reselected, the selection score. The density, the selection score, the
neighbour weights and the stability update can each be replaced by a ``Component`` (a class that carries its own
parameters and a ``static`` flag) passed to ``forward``; all of them receive the index of the current iteration and the
iteration cap, so they can change over the run, and the UMAP fuzzy graph that was built for the density. ``forward`` also
creates one empty dict before the first iteration; all the components receive it and return it (updated) with their result, so they
can remember things across calls and share them.

Support points (``support``, e.g. known normal embeddings for few-shot inference) act as high-mass points: every
iteration ``bias_fn`` turns the feature-wise differences between them and the points into a bias added to every
point's step, and ``step_fn``, if given, moves them; otherwise they stay where they are. They interact with the
points only through the bias.

Only torch is required.
"""

import logging

import torch

logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

METRICS = ("euclidean", "cosine")
SPHERE_MODES = (0, 1, 2, 3)
SPHERE_STAGES = ("graph", "density", "score", "weight")   # the stages ``forward(sphere=...)`` can give a space of their own

# Rows handled at a time by every chunked step (neighbour searches, density, selection blocks, gathers).
DEFAULT_CHUNK_SIZE = 1024


# ---------------------------------------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------------------------------------
def _raise_dynamo_recompile_limit(limit=128):
    """Compiled kernels built from one factory share a code object, and dynamo counts recompiles per code
    object (default limit 8). Using several values of ``k`` or several dtypes would otherwise hit the limit,
    which is a hard error under ``fullgraph=True``."""
    try:
        import torch._dynamo.config as dynamo_config
        for name in ("recompile_limit", "cache_size_limit"):
            if hasattr(dynamo_config, name):
                setattr(dynamo_config, name, max(getattr(dynamo_config, name), limit))
    except Exception:       # a torch build without dynamo: nothing to raise
        pass


_raise_dynamo_recompile_limit()

_SPARSE_MM_SUPPORT_CACHE = {}


def sparse_mm_supported(device):
    """Whether sparse-times-dense matrix multiplication works on ``device`` (probed once per device)."""
    device = str(device)
    if device not in _SPARSE_MM_SUPPORT_CACHE:
        try:
            idx = torch.zeros((2, 1), dtype=torch.long, device=device)
            t = torch.sparse_coo_tensor(idx, torch.ones(1, device=device), (1, 1), device=device).coalesce()
            torch.sparse.mm(t, torch.ones(1, 1, device=device))
            _SPARSE_MM_SUPPORT_CACHE[device] = True
        except Exception:
            _SPARSE_MM_SUPPORT_CACHE[device] = False
    return _SPARSE_MM_SUPPORT_CACHE[device]


def unit(X, eps=1e-12):
    """Rows scaled to unit length (a zero row stays zero)."""
    return X / X.norm(dim=1, keepdim=True).clamp_min(eps)


def pairwise_distance(A, B, metric="euclidean"):
    """(len(A), len(B)) distances between the rows of ``A`` and ``B``: Euclidean, or cosine (``1 - cos``, which
    for the unit-norm rows the pipeline works on is half the squared Euclidean distance)."""
    if metric == "cosine":
        return (1.0 - A @ B.T / (A.norm(dim=1, keepdim=True) * B.norm(dim=1)).clamp_min(1e-12)).clamp_min(0.0)
    return torch.cdist(A, B)


def _check_strength(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value >= 0 or value == float("inf"):
        raise ValueError(f"{name} must be a finite number >= 0; got {value!r}")
    return float(value)


def _call_with_memory(fn, args, memory, name):
    """``fn(*args, memory)`` for any of the replaceable callables; returns ``(output, memory)``.

    The callable gets the shared memory dict after its regular arguments and must return
    ``(output, memory)``, the dict being the updated memory. ``output`` may be ``None``: the caller then skips
    that call (see the ``None`` paragraph of ``Component``)."""
    result = fn(*args, memory)
    if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[1], dict):
        raise ValueError(f"{name} must return (output, memory) with the memory as a dict")
    return result


class Component:
    """Base class of everything ``forward`` calls: the density, score, weight, stability, metric, bias and step.

    ``static`` tells ``forward`` how often the component has to run (it can be overridden per instance with
    ``Component(static=...)``):

    - density: ``True`` evaluates it once, at the first iteration; ``False`` at the start of every iteration.
    - weight: ``True`` prepares the neighbour weights only when neighbours are selected; ``False`` at every iteration.
    - stability, step: ``True`` means it never changes anything, so ``forward`` does not call it.
    - score: ``True`` selects neighbours only once, at the first iteration, whatever ``recompute_neighbors`` says.
    - metric, bias: informational (always called when due).

    Any component may return ``(None, memory)`` to skip that call (e.g. depending on ``iteration`` and ``n_iters``):
    density keeps the previous density, score keeps the current neighbours (no reselection), weight keeps the previous
    weights, stability keeps the scores, metric adds nothing, bias is zero and step moves nothing. Density, score and
    weight have nothing to keep on the first call, and a weight has nothing consistent to keep right after neighbours
    were selected: ``None`` there raises a ``ValueError``.

    Subclasses implement ``__call__`` with the signature documented on their own class.
    """

    static = False

    def __init__(self, static=None):
        if static is not None:
            if not isinstance(static, bool):
                raise ValueError(f"static must be a bool; got {static!r}")
            self.static = static

    def __call__(self, *args):
        raise NotImplementedError(f"{type(self).__name__} must implement __call__")


def _resolve(fn, default, name):
    """``fn`` if given, else ``default``; either must be a ``Component``."""
    fn = default if fn is None else fn
    if not isinstance(fn, Component):
        raise TypeError(f"{name} must be an instance of Component (a class with a `static` flag); "
                        f"got {type(fn).__name__}")
    return fn


# ---------------------------------------------------------------------------------------------------------
# 1. Normalise (``sphere``): centring on the mean of the points
# ---------------------------------------------------------------------------------------------------------
@torch.no_grad()
def frechet_mean(X, max_iters=100, tol=1e-7):
    """Fréchet (Karcher) mean of unit-norm rows on the sphere: start from the normalised Euclidean mean and
    repeat ``mu <- Exp_mu(mean of Log_mu(x_i))`` until the update is shorter than ``tol``."""
    mu = unit(X.mean(0, keepdim=True))
    for _ in range(max_iters):
        g = _log_map(mu, X).mean(0, keepdim=True)
        size = g.norm()
        if size < tol:
            break
        mu = unit(mu * torch.cos(size) + g / size * torch.sin(size))
    return mu


def _log_map(mu, X):
    """Log map at the unit vector ``mu`` (1, d) of unit-norm rows ``X``: the tangent vector, of length equal to
    the angle between ``mu`` and the row, pointing away from ``mu`` towards it (zero for a row equal to ``mu``)."""
    cos = X @ mu.T
    v = X - cos * mu
    vnorm = v.norm(dim=1, keepdim=True)
    return v * (torch.atan2(vnorm, cos) / vnorm.clamp_min(1e-12))


@torch.no_grad()
def spherical_center(X):
    """Centre unit-norm rows on their Fréchet mean: ``Log_mu(x) / ||Log_mu(x)||``. Every returned row is
    orthogonal to the mean. Returns ``(centered, mu (d,))``."""
    mu = frechet_mean(X)
    return unit(_log_map(mu, X)), mu[0]


def _normalise(X, mode):
    """``X`` mapped by ``sphere`` mode 0 (as is), 1 (unit rows), 2 (mean-subtracted, unit) or 3 (Fréchet-centred);
    returns ``(points, center)`` with ``center`` the removed mean (``None`` for 0 and 1)."""
    if mode == 0:
        return X, None
    x = unit(X)
    if mode == 1:
        return x, None
    if mode == 2:
        center = x.mean(0)                                                   # Euclidean mean of the unit-norm points
        return unit(x - center), center
    return spherical_center(x)                                               # Fréchet-mean centred, still on the sphere


def _map_like(points, mode, center):
    """``points`` (e.g. the support) mapped into the space ``_normalise`` put the data in, with its ``center``."""
    if mode == 0:
        return points
    points = unit(points)
    if mode == 2:
        return unit(points - center)
    if mode == 3:
        return unit(_log_map(center.unsqueeze(0), points))
    return points


# ---------------------------------------------------------------------------------------------------------
# 2. Density: exact k-NN, the fuzzy neighbour graph, and the default density
# ---------------------------------------------------------------------------------------------------------
_KNN_KERNELS = {}


def _knn_kernel(k, metric):
    if (k, metric) not in _KNN_KERNELS:
        def _kernel(X_chunk, X_full):
            return pairwise_distance(X_chunk, X_full, metric).topk(k, largest=False, dim=1)   # (chunk, n): all-pairs
        _KNN_KERNELS[(k, metric)] = torch.compile(_kernel, fullgraph=True)
    return _KNN_KERNELS[(k, metric)]


@torch.no_grad()
def knn(X, k, chunk_size=DEFAULT_CHUNK_SIZE, metric="euclidean"):
    """Exact k-nearest neighbours of every row among all rows (itself included at distance 0), by
    Euclidean or cosine distance (``metric``).

    Returns ``(indices (n, k) int64, distances (n, k))``. A point's distance to itself is set to exactly 0:
    the matmul-based ``cdist`` leaves platform-dependent rounding noise there that would otherwise be
    mistaken for the nearest-neighbour distance.
    """
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= X.shape[0]:
        raise ValueError(f"k={k!r} is invalid for {X.shape[0]} points; expected an integer 1 <= k <= {X.shape[0]}")
    X = X.detach()
    n = X.shape[0]
    kernel = _knn_kernel(k, metric)
    all_idx = torch.empty((n, k), dtype=torch.long, device=X.device)
    all_dist = torch.empty((n, k), dtype=X.dtype, device=X.device)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        dists, idx = kernel(X[start:end], X)
        rows = torch.arange(start, end, device=X.device).unsqueeze(1)
        all_idx[start:end] = idx
        all_dist[start:end] = dists.masked_fill(idx == rows, 0.0)
    return all_idx, all_dist


def _make_symmetrize_kernel():
    def _kernel(knn_dists, knn_indices, rhos, sigma, n, k):
        dists_shifted = torch.clamp(knn_dists - rhos[:, None], min=0.0)
        weights = torch.exp(-dists_shifted / sigma[:, None])

        rows = torch.arange(n, device=knn_dists.device).repeat_interleave(k)
        cols = knn_indices.reshape(-1)
        vals = weights.reshape(-1)

        fwd_keys = rows * n + cols
        rev_keys = cols * n + rows

        sort_idx = torch.argsort(fwd_keys)
        sorted_keys = fwd_keys[sort_idx]
        sorted_vals = vals[sort_idx]

        pos = torch.searchsorted(sorted_keys, rev_keys)
        pos = torch.clamp(pos, max=sorted_keys.shape[0] - 1)
        matched = sorted_keys[pos] == rev_keys
        w_rev = torch.where(matched, sorted_vals[pos], torch.zeros_like(vals))

        w_sym = vals + w_rev - vals * w_rev
        return rows, cols, w_sym

    return torch.compile(_kernel, fullgraph=True)


_SYMMETRIZE_KERNEL = _make_symmetrize_kernel()


def _fuzzy_graph(knn_indices, knn_dists, n, k, n_epochs):
    """Sparse fuzzy graph (COO rows, cols, values) from k-NN results: per-point scale ``sigma`` by bisection,
    fuzzy-union symmetrisation, and removal of edges weaker than ``max / n_epochs``."""
    device = knn_dists.device
    target = float(torch.log2(torch.tensor(k, dtype=torch.float32)))

    mask = knn_dists > 0
    rhos = torch.where(mask, knn_dists, torch.tensor(float("inf"), device=device))
    rhos = torch.clamp(rhos.min(dim=1).values, min=1e-8)

    lo = torch.full((n,), 1e-20, device=device)
    hi = torch.full((n,), 1e3, device=device)
    sigma = torch.ones(n, device=device)
    shifted_tail = torch.clamp(knn_dists - rhos[:, None], min=0.0)[:, 1:]

    for _ in range(64):
        vals_sum = torch.exp(-shifted_tail / sigma[:, None]).sum(dim=1)
        converged = (vals_sum - target).abs() < 1e-5
        too_high = (vals_sum > target) & ~converged
        too_low = (vals_sum < target) & ~converged

        hi = torch.where(too_high, sigma, hi)
        lo = torch.where(too_low, sigma, lo)
        sigma = torch.where(too_high, (lo + sigma) / 2.0, sigma)
        sigma = torch.where(too_low, torch.where(hi >= 1e3, sigma * 2.0, (sigma + hi) / 2.0), sigma)

        if bool(converged.all()):
            break

    sigma = torch.clamp(sigma, min=1e-10)
    rows, cols, w_sym = _SYMMETRIZE_KERNEL(knn_dists, knn_indices, rhos, sigma, n, k)
    active = torch.nonzero(w_sym >= w_sym.max() / max(n_epochs, 1), as_tuple=True)[0]
    return rows[active], cols[active], w_sym[active]


def _similarity_graph(X, n_neighbors, n_epochs, chunk_size=DEFAULT_CHUNK_SIZE, metric="euclidean"):
    """The fuzzy neighbour graph of ``X`` as a coalesced sparse (n, n) matrix. The exact neighbour search that
    feeds it works ``chunk_size`` rows at a time; everything after it is ``O(n * n_neighbors)``."""
    n = X.shape[0]
    knn_idx, knn_dist = knn(X, n_neighbors, chunk_size=chunk_size, metric=metric)
    rows, cols, vals = _fuzzy_graph(knn_idx, knn_dist, n, n_neighbors, n_epochs)
    return torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, size=(n, n), device=X.device, check_invariants=False,
    ).coalesce()


def _graph_layout(S):
    """Row offsets and squared row norms of the sparse graph, for streamed passes."""
    idx = S.indices()
    rows_sorted, cols_sorted, vals_sorted = idx[0], idx[1], S.values()
    n = S.shape[0]
    row_ptr = torch.searchsorted(rows_sorted, torch.arange(0, n + 1, device=S.device)).tolist()
    row_norm_sq = torch.zeros(n, dtype=torch.float32, device=S.device)
    row_norm_sq.scatter_add_(0, rows_sorted, vals_sorted ** 2)
    return rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq


class Density(Component):
    """Adaptive angular k-NN density on the unit hypersphere (the default density).

    1. Normalise every point, and take the chord distance ``sqrt(2 - 2 x_i . x_j)`` (the Euclidean distance of
       unit vectors) to the ``k`` nearest other points; ``r_ij`` is the distance to the j-th neighbour.
    2. Estimate the intrinsic dimension with the k-NN maximum-likelihood estimator
       ``D = [mean_i mean_{j<k} log(r_ik / r_ij)]^-1`` (one value for all the points).
    3. The raw density is ``q_i = -D log(r_ik)``: a small neighbour radius means a dense point.
    4. ``z_i = (q_i - mean(q)) / std(q)`` and the density is ``exp(clip(z_i, -clip, clip))``, positive and
       within ``[exp(-clip), exp(clip)]``.

    ``k`` is lowered to ``n - 1`` for a small set; with fewer than 3 points the density is 1.0 everywhere.
    ``chunk_size`` is the number of rows handled at a time by the exact neighbour search. It is not static: it is
    re-estimated on the current points at every iteration (pass ``static=True`` to compute it once, on the initial ones).
    """

    def __init__(self, k=20, clip=3.0, chunk_size=DEFAULT_CHUNK_SIZE, static=None):
        super().__init__(static)
        if isinstance(k, bool) or not isinstance(k, int) or k < 2:
            raise ValueError(f"k must be an integer >= 2; got {k!r}")
        if not clip > 0:
            raise ValueError(f"clip must be > 0; got {clip!r}")
        self.k = k
        self.clip = clip
        self.chunk_size = chunk_size

    @torch.no_grad()
    def __call__(self, x, density, stability, iteration, n_iters, graph, memory):
        n = x.shape[0]
        k = min(self.k, n - 1)
        if k < 2:
            return torch.ones(n, dtype=x.dtype, device=x.device), memory
        eps = 1e-12
        _, dist = knn(unit(x), k + 1, chunk_size=self.chunk_size)            # the point itself comes first, at 0
        dist = dist[:, 1:].clamp_min(eps)                                    # (n, k): the k nearest other points
        r = dist[:, -1]                                                      # k-th neighbour radius
        intrinsic_dim = 1.0 / torch.log(r.unsqueeze(1) / dist[:, :-1]).mean().clamp_min(eps)
        raw = -intrinsic_dim * torch.log(r)
        z = (raw - raw.mean()) / raw.std().clamp_min(eps)
        return torch.exp(z.clamp(-self.clip, self.clip)).to(x.dtype), memory


# ---------------------------------------------------------------------------------------------------------
# 3. Select neighbours: the selection score and the k best candidates of every point
# ---------------------------------------------------------------------------------------------------------
def _graph_rows(layout, start, end, n, dtype, device):
    """Dense (end - start, n) block of the graph rows ``start:end`` (row-aligned with a distance block)."""
    rows_sorted, cols_sorted, vals_sorted, row_ptr, _ = layout
    lo, hi = row_ptr[start], row_ptr[end]
    block = torch.zeros((end - start, n), dtype=dtype, device=device)
    block[rows_sorted[lo:hi] - start, cols_sorted[lo:hi]] = vals_sorted[lo:hi].to(dtype)
    return block


@torch.no_grad()
def graph_values_at(graph, indices):
    """(n, k) fuzzy-graph weights of the pairs (i, indices[i, j]); 0 where the graph has no edge for a pair."""
    n = graph.shape[0]
    idx, vals = graph.indices(), graph.values()
    keys = idx[0] * n + idx[1]                                    # sorted: a coalesced sparse tensor is ordered by (row, col)
    query = (torch.arange(n, device=indices.device).unsqueeze(1) * n + indices).reshape(-1)
    position = torch.searchsorted(keys, query).clamp_(max=keys.shape[0] - 1)
    found = keys[position] == query
    return torch.where(found, vals[position], torch.zeros_like(vals[position])).reshape(indices.shape)


@torch.no_grad()
def median_neighbor_distance(X, k, chunk_size=DEFAULT_CHUNK_SIZE, metric="euclidean"):
    """Median distance from a point to its ``k`` nearest other points: a data-driven kernel width."""
    n = X.shape[0]
    if n < 2:
        return 1.0
    _, dist = knn(X, min(k, n - 1) + 1, chunk_size=chunk_size, metric=metric)           # column 0 is the point itself
    return max(dist[:, 1:].median().item(), 1e-12)


class Score(Component):
    """How attractive every candidate is to a point: ``density^a * stability^b * exp(-c * d^2 / (2 h^2))``.

    The default selection score. ``a``, ``b`` and ``c`` are the density, stability and distance strengths
    (>= 0; 0 switches that factor off, larger values weight it more). ``d`` is the distance between the point and
    the candidate (Euclidean or cosine, see the ``dist`` of ``forward``) and
    ``h`` the kernel width.

    A replacement must have the same call signature and return a score of the same shape as ``dist``, together
    with the memory: ``score(dist, density, stability, h, iteration, n_iters, graph, memory) -> (score, memory)``.
    The memory is one dict that ``forward`` creates empty before the first iteration and hands to all four
    callables; return it (updated or not) to keep what you stored. The score function is called once per block of
    rows, so the memory passes through every block in turn. ``dist`` is a (chunk, n) block of
    distances from a chunk of points to all points, ``density`` and ``stability`` the (n,) per-candidate vectors,
    ``h`` the kernel width in use, ``iteration`` the 0-based index of the current iteration, ``n_iters`` the
    iteration cap (``max_iters_shift``; the run may stop earlier by convergence) and ``graph`` the (chunk, n)
    dense rows of the UMAP fuzzy graph computed for the density, aligned with ``dist`` (``graph[r, j]`` is the
    fuzzy weight of the edge from the point of row ``r`` to point ``j``, 0 where there is none). It is called
    whenever neighbours are selected. Larger means more attractive; scores <= 0 mean "no attraction" and are only used to fill slots
    once the positive scores run out (the nearest remaining candidates are taken). Scores must be finite.
    """

    def __init__(self, distance_strength=1.0, density_strength=1.0, stability_strength=1.0, static=None):
        super().__init__(static)
        self.distance_strength = _check_strength("distance_strength", distance_strength)
        self.density_strength = _check_strength("density_strength", density_strength)
        self.stability_strength = _check_strength("stability_strength", stability_strength)

    def __call__(self, dist, density, stability, h, iteration, n_iters, graph, memory):
        kernel = torch.exp(-self.distance_strength * dist.square() / (2.0 * h * h))
        return density.pow(self.density_strength) * stability.pow(self.stability_strength) * kernel, memory


@torch.no_grad()
def select_neighbors(X, density, stability, k, h, score_fn, graph,
                     labels=None, chunk_size=DEFAULT_CHUNK_SIZE, iteration=0, n_iters=1, memory=None,
                     metric="euclidean"):
    """The ``k`` best-scoring other points for every point of ``X``.

    Distances are Euclidean on ``X`` (or cosine, with ``metric="cosine"``). All other points are candidates (no distance pre-filter); a point is never
    its own neighbour. Candidates with a positive score rank by score,
    the remaining slots (if any) go to the nearest candidates. With ``labels`` a point may only pick points
    labelled 0, whatever its own label.

    ``iteration``, ``n_iters``, the rows of the fuzzy ``graph`` (see ``_similarity_graph``) and the shared
    ``memory`` dict (an empty one if omitted) are handed on to ``score_fn``, a chunk of rows at a time, the memory
    passing through the chunks in turn.

    Returns ``(indices (n, k), dist (n, k), valid (n, k) bool or None, orphan (n,) bool or None, h, memory)``
    where ``valid`` marks picks that respect the label mask, ``orphan`` marks points with no valid pick at all,
    ``dist`` the distances to the picks, ``h`` the kernel width used (``"auto"`` resolves to the median distance
    to the ``k`` nearest neighbours) and ``memory`` is the updated dict. If ``score_fn`` returns ``None`` for any
    block, the selection is abandoned and the result is ``(None, None, None, None, h, memory)``.
    """
    n = X.shape[0]
    memory = {} if memory is None else memory
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= n - 1:
        raise ValueError(f"k={k!r} is invalid for {n} points; expected an integer 1 <= k <= {n - 1}")
    if h == "auto":
        h = median_neighbor_distance(X, k, chunk_size, metric)
    layout = _graph_layout(graph)
    indices = torch.empty((n, k), dtype=torch.long, device=X.device)
    dist = torch.empty((n, k), dtype=X.dtype, device=X.device)
    valid = torch.empty((n, k), dtype=torch.bool, device=X.device)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        d = pairwise_distance(X[start:end], X, metric)                      # (c, n) distances
        score, memory = _call_with_memory(
            score_fn, (d, density, stability, h, iteration, n_iters, _graph_rows(layout, start, end, n, X.dtype, X.device)),
            memory, "the selection score function")
        if score is None:
            return None, None, None, None, h, memory
        if score.shape != d.shape or not bool(torch.isfinite(score).all()):
            raise ValueError("the selection score must be finite and have the shape of the distance block "
                             f"{tuple(d.shape)}; got {tuple(score.shape)}")
        key = torch.where(score > 0, score, -d)                              # positives first, then nearest

        excluded = torch.zeros_like(d, dtype=torch.bool)
        rows = torch.arange(end - start, device=X.device)
        excluded[rows, rows + start] = True                                  # never its own neighbour
        if labels is not None:
            excluded |= (labels != 0).unsqueeze(0)                         # only label-0 points are candidates
        top, idx = key.masked_fill(excluded, float("-inf")).topk(k, dim=1)

        indices[start:end] = idx
        dist[start:end] = d.gather(1, idx)
        valid[start:end] = torch.isfinite(top)

    if labels is None:
        return indices, dist, None, None, h, memory
    return indices, dist, valid, ~valid.any(dim=1), h, memory


@torch.no_grad()
def neighbor_distances(X, indices, chunk_size=DEFAULT_CHUNK_SIZE, metric="euclidean"):
    """Current distances (n, k) from every point of ``X`` to its selected neighbours ``indices``: Euclidean, or
    cosine with ``metric="cosine"``."""
    out = torch.empty(indices.shape, dtype=X.dtype, device=X.device)
    for start in range(0, X.shape[0], chunk_size):
        end = min(X.shape[0], start + chunk_size)
        if metric == "cosine":
            a, b = X[start:end].unsqueeze(1), X[indices[start:end]]
            out[start:end] = (1.0 - (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)).clamp_min(0.0)
        else:
            out[start:end] = (X[start:end].unsqueeze(1) - X[indices[start:end]]).norm(dim=-1)
    return out


# ---------------------------------------------------------------------------------------------------------
# 4. Weights: how much each selected neighbour pulls
# ---------------------------------------------------------------------------------------------------------
class Weight(Component):
    """How much each selected neighbour pulls its point in the weighted mean: ``density^a * stability^b``.

    The default neighbour weight (distance is ignored, so every selected neighbour counts according to how
    dense and stable it is). ``a`` and ``b`` are the density and stability strengths (>= 0).

    A replacement must have the same call signature and also return the memory,
    ``weight(dist, density, stability, h, iteration, n_iters, graph, memory) -> (weights, memory)`` (the memory is
    the dict shared by the four callables, see ``Score``), with the tensors ``dist``, ``density``,
    ``stability`` and ``graph`` of shape (n, k) for the k selected neighbours of every point (``dist`` the current
    distances to them, ``stability`` their current scores, ``graph`` the weight of the UMAP
    fuzzy graph computed for the density between each point and each of its selected neighbours, 0 where the graph
    has no such edge), ``h`` the kernel width in use, ``iteration`` the 0-based index of the iteration whose step
    will use the weights and ``n_iters`` the iteration cap (``max_iters_shift``). It returns non-negative finite (n, k) weights, normalised per row
    afterwards, so only their ratios matter.

    The weights are always prepared when neighbours are selected. Unless the weight is ``static``, they are also
    prepared at every other iteration, from the current distances, densities and stability scores. The default is
    not static, because its density and stability can change; pass ``static=True`` when neither does.
    """

    def __init__(self, density_strength=1.0, stability_strength=1.0, static=None):
        super().__init__(static)
        self.density_strength = _check_strength("density_strength", density_strength)
        self.stability_strength = _check_strength("stability_strength", stability_strength)

    def __call__(self, dist, density, stability, h, iteration, n_iters, graph, memory):
        return density.pow(self.density_strength) * stability.pow(self.stability_strength), memory


@torch.no_grad()
def prepare_weights(indices, dist, density, stability, h, weight_fn, valid, n, iteration, n_iters, graph, memory):
    """Normalised (n, k) neighbour weights, where the device supports it the sparse (n, n) weight matrix, and the
    updated shared ``memory`` dict, as ``(indices, w, W, memory)`` (``w`` and ``W`` are ``None`` if ``weight_fn``
    returned ``None``).

    ``weight_fn`` is evaluated here with the (n, k) tensors of the selected neighbours; ``graph`` is the (n, k)
    fuzzy-graph weights of the selected pairs (see ``graph_values_at``). The label-validity mask
    is applied and every row is normalised to sum to 1 (a row with no weight at all stays zero, and ``forward``
    leaves its point in place).
    """
    w, memory = _call_with_memory(
        weight_fn, (dist, density[indices], stability[indices], h, iteration, n_iters, graph), memory,
        "the neighbour weight function")
    if w is None:
        return indices, None, None, memory
    if w.shape != indices.shape or not bool(torch.isfinite(w).all()) or bool((w < 0).any()):
        raise ValueError("the neighbour weights must be finite, non-negative and have shape "
                         f"{tuple(indices.shape)}; got {tuple(w.shape)}")
    if valid is not None:
        w = w * valid
    w = w / w.sum(dim=1, keepdim=True).clamp_min(torch.finfo(w.dtype).tiny)

    W = None
    if sparse_mm_supported(w.device):
        k = indices.shape[1]
        rows = torch.arange(n, device=w.device).repeat_interleave(k)
        W = torch.sparse_coo_tensor(
            torch.stack([rows, indices.reshape(-1)]), w.reshape(-1), size=(n, n), device=w.device,
            check_invariants=False,
        ).coalesce()
    return indices, w, W, memory


# ---------------------------------------------------------------------------------------------------------
# 5. Step: the weighted mean, and the pull and step of the support points
# ---------------------------------------------------------------------------------------------------------
@torch.no_grad()
def weighted_mean(X, indices, w, W, chunk_size=DEFAULT_CHUNK_SIZE):
    """Row-wise weighted mean of the selected neighbours: one sparse matmul ``W @ X`` (or, on a device without
    sparse matmul, a gather over row chunks)."""
    if W is not None:
        return torch.sparse.mm(W, X)
    out = torch.empty_like(X)
    for start in range(0, X.shape[0], chunk_size):
        end = min(X.shape[0], start + chunk_size)
        out[start:end] = (w[start:end].unsqueeze(-1) * X[indices[start:end]]).sum(dim=1)
    return out


class Bias(Component):
    """The default pull of the support points on the points (the bias used when ``support`` is given).

    A step of ``learning_rate`` towards the mean of the support points: ``learning_rate * dist.mean(0)``. The bias
    is added outside ``learning_rate``, so without it every point would jump onto the support mean.

    A replacement must have the same call signature and also return the memory,
    ``bias(dist, density, stability, h, learning_rate, iteration, n_iters, memory) -> (bias, memory)``, where
    ``dist`` is the (s, n, d) difference ``support[j] - x[i]`` and ``bias`` the (n, d) vector added to every
    point's step.
    """

    def __call__(self, dist, density, stability, h, learning_rate, iteration, n_iters, memory):
        return learning_rate * dist.mean(0), memory


class Step(Component):
    """The default step of the support points: none, they stay where they are.

    It is static, so ``forward`` does not call it. A replacement has the same signature as the bias,
    ``step(dist, density, stability, h, learning_rate, iteration, n_iters, memory) -> (step, memory)``, and
    returns the (s, d) vectors added to the support points.
    """

    static = True

    def __call__(self, dist, density, stability, h, learning_rate, iteration, n_iters, memory):
        return torch.zeros_like(dist[:, 0]), memory


def support_vectors(fn, diff, density, stability, h, learning_rate, iteration, n_iters, memory, rows, name):
    """``fn(diff, density, stability, h, learning_rate, iteration, n_iters, memory)`` for a ``bias_fn`` or
    ``step_fn``, checked to be a finite (rows, d) tensor; returns ``(vectors, memory)``.

    ``diff`` is the (s, n, d) tensor of feature-wise differences ``support[j] - x[i]`` between the s support
    points and the n points, ``density`` and ``stability`` are (n,) per point, ``h`` the kernel width in use."""
    out, memory = _call_with_memory(fn, (diff, density, stability, h, learning_rate, iteration, n_iters), memory, name)
    shape = (rows, diff.shape[2])
    if out is None:                                                          # skipped: no bias / no step
        return torch.zeros(shape, dtype=diff.dtype, device=diff.device), memory
    if not isinstance(out, torch.Tensor) or tuple(out.shape) != shape or not bool(torch.isfinite(out).all()):
        raise ValueError(f"{name} must return a finite tensor of shape {shape}; "
                         f"got {tuple(getattr(out, 'shape', ()))}")
    return out.to(device=diff.device, dtype=diff.dtype), memory


# ---------------------------------------------------------------------------------------------------------
# 6. After every iteration: the stability update and the metric
# ---------------------------------------------------------------------------------------------------------
class Stability(Component):
    """The default stability update: none, the scores stay at their initial 1.0.

    It is static, so ``forward`` does not call it. A replacement has the signature
    ``stability(moved, x, density, stability, h, iteration, n_iters, graph, memory) -> (stability, memory)``.
    """

    static = True

    def __call__(self, moved, x, density, stability, h, iteration, n_iters, graph, memory):
        return stability, memory


@torch.no_grad()
def update_stability(stability_fn, moved, x, density, stability, h, iteration, n_iters, graph, memory):
    """New (n,) stability scores and the updated shared ``memory`` dict, as ``(scores, memory)``.

    ``stability_fn(moved, x, density, stability, h, iteration, n_iters, graph, memory)`` receives the new and
    previous (n, d) positions, the (n,) density, the (n,) scores before this update, the kernel width ``h``, the
    0-based iteration index, the iteration cap and the full (n, n) sparse UMAP fuzzy graph. It must return finite
    (n,) scores within [0, 1].
    """
    new, memory = _call_with_memory(
        stability_fn, (moved, x, density, stability, h, iteration, n_iters, graph), memory, "the stability function")
    if new is None:                                                          # skipped: the scores stay
        return stability, memory
    if not isinstance(new, torch.Tensor) or new.shape != stability.shape:
        raise ValueError(f"the stability scores must be a tensor of shape {tuple(stability.shape)}; "
                         f"got {tuple(getattr(new, 'shape', ()))}")
    if not (bool(torch.isfinite(new).all()) and bool((new >= 0).all()) and bool((new <= 1).all())):
        raise ValueError("stability scores must be finite and within [0, 1]")
    return new.to(device=stability.device, dtype=stability.dtype), memory


class Metric(Component):
    """The default metric: how far each point moved in the iteration, ``|moved - x|``.

    Summed over the iterations it is the distance each point travelled (the sum of its step lengths; chords on
    the sphere). A replacement has the same signature and also returns the memory,
    ``metric(moved, x, density, stability, h, iteration, n_iters, graph, memory) -> (metric, memory)``, with an
    (n,) metric; ``forward`` sums the values over the iterations.
    """

    @torch.no_grad()
    def __call__(self, moved, x, density, stability, h, iteration, n_iters, graph, memory):
        return (moved - x).norm(dim=1), memory


# ---------------------------------------------------------------------------------------------------------
# 7. Repeat: the model
# ---------------------------------------------------------------------------------------------------------
class HypersphericalMeanShiftDensityEnhancement(torch.nn.Module):
    """Mean-shift density enhancement of unit-normalised embeddings (see the module docstring).

    Parameters
    ----------
    k : number of neighbours each point is pulled towards.
    n_neighbors, n_epochs : the fuzzy neighbour graph (UMAP fuzzy simplicial set) of the initial points.
    learning_rate : fraction of the way a point moves towards its weighted mean each iteration.
    max_iters_shift, shift_threshold : the loop stops when the mean per-point movement is below
        ``shift_threshold`` (convergence) or after ``max_iters_shift`` iterations (a safety cap).
    keep_trajectory : also return the unit-norm state after every iteration.
    chunk_size : rows handled at a time by every chunked step, which bounds the memory of the run: the exact
        neighbour searches (the fuzzy graph, the default density and ``h="auto"``), the graph-space distances of ``Original``, the
        (chunk, n) blocks of the neighbour selection, and the (chunk, k, d) gathers of the live neighbour
        distances and of the weighted-mean fallback. Smaller uses less memory; results change only through
        floating-point rounding.
    device : where the computation runs.
    recompute_neighbors : 0 or None selects neighbours once, from the initial points; a positive N reselects
        every N iterations (iterations 0, N, 2N, ...), re-evaluating the neighbour weights.
    sphere : 0 runs in the raw feature space (no normalisation). 1 normalises the points onto the unit sphere and
        back onto it after every shift. 2 is 1, after first subtracting the Euclidean mean of the unit-norm points
        and normalising again. 3 is 1, after first centring the points on their Fréchet mean (see
        ``spherical_center``). With 2 or 3, everything after the centring, including the output and the
        trajectory's first state, is in the centred space. The mean is available as ``center_`` afterwards
        (``None`` for 0 and 1).
    distance_strength, density_strength, stability_strength : the strengths of the default selection score
        and neighbour weight (see ``Score`` and ``Weight``); >= 0, 0 switches a factor off.
    h : kernel width of the distance term, a positive number, or "auto" (median distance to the ``k`` nearest neighbours,
        re-estimated every time neighbours are selected).
    """

    def __init__(
        self,
        k=100,
        n_neighbors=15,
        n_epochs=200,
        learning_rate=0.1,
        max_iters_shift=1000,
        shift_threshold=1e-4,
        keep_trajectory=False,
        chunk_size=DEFAULT_CHUNK_SIZE,
        device=DEFAULT_DEVICE,
        recompute_neighbors=0,
        sphere=1,
        distance_strength=1.0,
        density_strength=1.0,
        stability_strength=1.0,
        h="auto",
    ):
        super().__init__()
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be a positive integer; got {k!r}")
        if not learning_rate > 0:
            raise ValueError("learning_rate must be > 0")
        if not isinstance(max_iters_shift, int) or max_iters_shift < 1:
            raise ValueError("max_iters_shift must be a positive integer")
        if not shift_threshold >= 0:
            raise ValueError("shift_threshold must be >= 0")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError(f"chunk_size must be a positive integer; got {chunk_size!r}")
        if recompute_neighbors is not None and (
            not isinstance(recompute_neighbors, int) or isinstance(recompute_neighbors, bool) or recompute_neighbors < 0
        ):
            raise ValueError("recompute_neighbors must be None, 0 (select once) or a positive integer N "
                             "(reselect every N iterations)")
        if sphere not in SPHERE_MODES or isinstance(sphere, bool):
            raise ValueError(f"sphere must be one of {SPHERE_MODES}; got {sphere!r}")
        if isinstance(h, str):
            if h != "auto":
                raise ValueError(f"h must be a positive number or 'auto'; got {h!r}")
        elif isinstance(h, bool) or not h > 0:
            raise ValueError(f"h must be a positive number or 'auto'; got {h!r}")

        self.k = k
        self.n_neighbors = n_neighbors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.max_iters_shift = max_iters_shift
        self.shift_threshold = shift_threshold
        self.keep_trajectory = keep_trajectory
        self.chunk_size = chunk_size
        self.device_name = device
        self.recompute_neighbors = recompute_neighbors or 0
        self.sphere = sphere
        self.h = h
        self.density = Density(chunk_size=chunk_size)
        self.metric = Metric()
        self.stability = Stability()
        self.step = Step()
        self.score = Score(distance_strength, density_strength, stability_strength)
        self.bias = Bias()
        self.weight = Weight(density_strength, stability_strength)

    def _labels(self, labels, n):
        """``labels`` given to ``forward`` as a long tensor on the device (or ``None``)."""
        if labels is None:
            return None
        labels = torch.as_tensor(labels, device=self.device_name).detach().long()
        if labels.dim() != 1 or labels.shape[0] != n:
            raise ValueError(f"labels must be 1D with length n={n}; got shape {tuple(labels.shape)}")
        return labels

    def _support(self, support, d, dtype):
        """``support`` given to ``forward`` (a list of (d,) tensors or an (s, d) tensor) as an (s, d) tensor on
        the device (or ``None``)."""
        if support is None:
            return None
        if isinstance(support, (list, tuple)):
            if len(support) == 0:
                raise ValueError("support must contain at least one point")
            support = torch.stack([torch.as_tensor(p) for p in support])
        support = torch.as_tensor(support).detach().to(device=self.device_name, dtype=dtype)
        if support.dim() != 2 or support.shape[0] == 0 or support.shape[1] != d:
            raise ValueError(f"support must be points of shape ({d},); got shape {tuple(support.shape)}")
        if not bool(torch.isfinite(support).all()):
            raise ValueError("support must be finite")
        return support

    def _check_input(self, X, dist):
        """``X`` as a floating-point (n, d) tensor on the device; checks ``dist`` and the neighbour counts."""
        X = torch.as_tensor(X)
        if X.dim() != 2 or not X.is_floating_point():
            raise ValueError(f"X must be a 2D floating-point tensor; got shape {tuple(X.shape)}, dtype {X.dtype}")
        X = X.detach().to(self.device_name)
        n = X.shape[0]
        if dist not in METRICS:
            raise ValueError(f"dist must be one of {METRICS}; got {dist!r}")
        if not 1 <= self.k <= n - 1 or not 1 <= self.n_neighbors <= n:
            raise ValueError(f"k={self.k} and n_neighbors={self.n_neighbors} need k <= n-1 and n_neighbors <= n "
                             f"for n={n} points")
        return X

    def _check_sphere(self, sphere, stages):
        """The per-stage ``sphere`` dict of ``forward`` as a plain dict (``{}`` for ``None``), validated.

        ``stages`` maps the stage names that may be set to their components. With the init ``sphere`` 2 or 3 every
        stage in the dict has to be static (``"graph"`` is built once anyway): it is evaluated once, from the centre of
        the initial points. With 0 or 1 nothing is checked."""
        if sphere is None:
            return {}
        if not isinstance(sphere, dict):
            raise TypeError(f"sphere must be a dict {{stage: mode}} or None; got {type(sphere).__name__}")
        for stage, mode in sphere.items():
            if stage not in SPHERE_STAGES:
                raise ValueError(f"sphere stages must be among {SPHERE_STAGES}; got {stage!r} "
                                 "(the other stages use the sphere of the model)")
            if isinstance(mode, bool) or mode not in SPHERE_MODES:
                raise ValueError(f"sphere[{stage!r}] must be one of {SPHERE_MODES}; got {mode!r}")
            if self.sphere in (2, 3) and stage != "graph" and not stages[stage].static:
                raise ValueError(f"with sphere={self.sphere}, sphere[{stage!r}] needs a static {stage} component "
                                 "(pass static=True): it is evaluated once, before the iterations")
        return dict(sphere)

    def _support_pull(self, support, x, density, stability, h, iteration, n_iters, memory, bias_fn, step_fn):
        """The bias of the support on the points and the moved support, as ``(bias, support, memory)``."""
        n = x.shape[0]
        diff = support.unsqueeze(1) - x.unsqueeze(0)                          # (s, n, d), from each point to each support point
        bias, memory = support_vectors(
            bias_fn, diff, density, stability, h, self.learning_rate, iteration, n_iters, memory, n, "the bias function")
        if not step_fn.static:
            step, memory = support_vectors(
                step_fn, diff, density, stability, h, self.learning_rate, iteration, n_iters, memory,
                support.shape[0], "the support step function")
            support = support + step
            if self.sphere:
                support = unit(support)
        return bias, support, memory

    @torch.no_grad()
    def forward(self, X, labels=None, support=None, dist="euclidean", density_fn=None, score_fn=None, weight_fn=None,
                stability_fn=None, metric_fn=None, bias_fn=None, step_fn=None, sphere=None):
        """Run HyperMSDE and return ``(shifted, metric, trajectory)``.

        Every ``*_fn`` is an instance of ``Component`` (a class with a ``static`` flag, see ``Component``).

        Parameters
        ----------
        X : (n, d) floating-point tensor
            Input embeddings; rows are normalised onto the unit sphere unless ``sphere=0``.
        labels : optional (n,) integer array
            Restricts neighbour selection to points labelled 0, whatever a point's own label. Points with no valid neighbour stay in place
            (unless ``support`` is given: they still receive the bias).
        support : optional list of (d,) tensors (or an (s, d) tensor)
            High-mass support points. They interact with ``X`` only through ``bias_fn``: they are never neighbours
            and take no part in the density, the fuzzy graph or ``h``. They are mapped into the same space as ``X``
            (normalised, and centred with the same mean as ``X`` with ``sphere`` 2 or 3), move only through ``step_fn``
            and are available as ``support_`` afterwards. Acts on the points through ``bias_fn`` (default ``Bias``).
        dist : {``"euclidean"``, ``"cosine"``}
            Distance used for the fuzzy graph, neighbour selection, ``h="auto"``, and ``score_fn``/``weight_fn``.
            With ``sphere`` 1, 2 or 3 the points always stay on the unit sphere.
        density_fn : optional Component
            Replaces the density (see ``Density``, and ``Original`` in ``src/density.py``). It is called at the start
            of every iteration, before neighbour selection, unless it is static: then it is called once, at the first
            iteration. Without one, ``Density`` (adaptive angular k-NN, not static) is used. The final density is
            ``density_``.
        score_fn : optional Component
            Replaces the selection score (see ``Score``). It is called at every neighbour selection,
            including the initial one.
        weight_fn : optional Component
            Replaces the weights used for the weighted mean (see ``Weight``). The weights are prepared whenever
            neighbours are selected and, unless the weight is static, at every other iteration too (from the current
            neighbour distances, densities and stability scores).
        stability_fn : optional Component
            Updates the (n,) stability scores after each iteration (skipped when static). It is called as
            ``stability_fn(moved, x, density, stability, h, iteration, n_iters, graph, memory)``. ``moved`` and
            ``x`` are the new and previous (n, d) unit-norm positions; ``stability`` contains the scores before
            this update. Return ``(scores, memory)`` where scores are finite and in [0, 1]. Scores start at 1.0;
            updated scores affect the next iteration's weights and later neighbour selections. The final scores
            are available as ``stability_``.
        metric_fn : optional Component
            Computes a metric after each iteration. It receives the same arguments as ``stability_fn`` and
            returns ``(metric, memory)``; the values of all the iterations are summed and returned as the second
            item. Without one, ``Metric`` is used: the distance each point moved.
        bias_fn : optional Component
            The pull of the support on the points; without one, ``Bias`` (a step of ``learning_rate`` towards the
            mean of the support points) is used. It is called every iteration as
            ``bias_fn(dist, density, stability, h, learning_rate, iteration, n_iters, memory)``, where ``dist`` is
            the (s, n, d) tensor of feature-wise differences ``support[j] - x[i]`` at the current positions,
            ``density`` and ``stability`` are (n,) and ``h`` the kernel width in use. It returns a finite (n, d)
            bias, added to every point's step outside the learning rate:
            ``x + learning_rate * (mean - x) + bias``.
        step_fn : optional Component
            Moves the support points, with the same arguments as ``bias_fn`` (evaluated at the same positions,
            before the step). It returns a finite (s, d) step added to the support. Without one (``Step``, static)
            the support stays where it is. Requires ``support``.
        sphere : optional dict ``{stage: mode}``
            The space (0 raw, 1 unit, 2 mean-subtracted, 3 Fréchet-centred, see the ``sphere`` of the model) in
            which a stage sees the points. ``stage`` is ``"graph"``, ``"density"``, ``"score"`` or ``"weight"``;
            the other stages use the sphere of the model. A stage that is not in the dict sees the current points.
            With the sphere of the model 0 or 1, a stage in the dict sees the current points mapped with its mode
            every time it runs (live; mode 0 on a model of sphere 1 is the points as they are, since the
            normalisation cannot be undone), and nothing is checked. With 2 or 3 it sees the raw ``X`` mapped with
            its mode, made once and fixed for the whole run, and every stage in the dict needs a static component
            (``"graph"`` always is), else a ``ValueError``.

        Returns
        -------
        shifted : (n, d) tensor
            Unit-norm points after the last iteration.
        metric :
            The sum of the per-iteration values of ``metric_fn``; by default the (n,) sum of each point's
            per-iteration step lengths (chords on the sphere with ``sphere`` 1, 2 or 3).
        trajectory : list of (n, d) tensors
            Initial state followed by each iteration's state when ``keep_trajectory=True``; otherwise empty.

        Every callable above receives the same mutable ``memory`` dict as its final argument and returns the
        updated dict with its result as ``(result, memory)``. A new run starts with an empty dict; the final dict
        is available as ``memory_``. ``result`` may be ``None`` to skip that call (see ``Component``): the previous
        density, neighbours, weights or stability scores stay, the metric adds nothing, and the bias and the step are zero.
        """
        X = self._check_input(X, dist)
        n = X.shape[0]
        if support is None and (bias_fn is not None or step_fn is not None):
            raise ValueError("bias_fn and step_fn need support points")
        labels = self._labels(labels, n)
        support = self._support(support, X.shape[1], X.dtype)
        density_fn = _resolve(density_fn, self.density, "density_fn")
        score_fn = _resolve(score_fn, self.score, "score_fn")
        weight_fn = _resolve(weight_fn, self.weight, "weight_fn")
        stability_fn = _resolve(stability_fn, self.stability, "stability_fn")
        metric_fn = _resolve(metric_fn, self.metric, "metric_fn")
        bias_fn = _resolve(bias_fn, self.bias, "bias_fn")
        step_fn = _resolve(step_fn, self.step, "step_fn")
        spaces = self._check_sphere(sphere, {"density": density_fn, "score": score_fn, "weight": weight_fn})
        n_iters = self.max_iters_shift

        # 1. Normalise: the loop state, and the spaces of the stages that have one of their own.
        x, self.center_ = _normalise(X, self.sphere)
        frozen = {self.sphere: x}                                            # sphere 2, 3: the raw points in each mode, made once

        def points(stage):
            """The points ``stage`` works on: the current ones, unless the stage has a mode of its own. Then, with the
            sphere of the model 0 or 1, the current ones mapped with that mode (live); with 2 or 3, the raw ``X`` mapped
            with it (once, then fixed)."""
            mode = spaces.get(stage)
            if mode is None or mode == self.sphere and self.sphere < 2:
                return x
            if self.sphere < 2:
                return _normalise(x, mode)[0]
            if mode not in frozen:
                frozen[mode] = _normalise(X, mode)[0]
            return frozen[mode]

        if support is not None:
            support = _map_like(support, self.sphere, self.center_)          # the same space as the points
        graph = _similarity_graph(points("graph"), self.n_neighbors, self.n_epochs, self.chunk_size, dist)

        stability = torch.ones(n, dtype=X.dtype, device=X.device)           # every point starts fully stable
        density = torch.ones(n, dtype=X.dtype, device=X.device)             # replaced at the start of the iterations
        memory = {}                                                          # shared by all the callables: empty at the start
        metric = None
        trajectory = [x.clone()] if self.keep_trajectory else []
        indices, nbr_dist, valid, orphan, h, graph_pairs = None, None, None, None, None, None
        w, W, unweighted = None, None, None                                  # the neighbour weights of the current step
        for iteration in range(n_iters):
            # 2. Density
            if iteration == 0 or not density_fn.static:
                new_density, memory = _call_with_memory(
                    density_fn, (points("density"), density, stability, iteration, n_iters, graph), memory,
                    "the density function")
                if new_density is None:                                      # skipped: the previous density stays
                    if iteration == 0:
                        raise ValueError("the density function returned None at the first iteration: nothing to keep")
                else:
                    if (new_density.shape != (n,) or not bool(torch.isfinite(new_density).all())
                            or bool((new_density < 0).any())):
                        raise ValueError(f"the density must be a finite non-negative tensor of shape ({n},); "
                                         f"got shape {tuple(new_density.shape)}")
                    density = new_density.to(x.dtype)

            # 3. Select neighbours
            selected = iteration == 0 or bool(
                self.recompute_neighbors and not score_fn.static and iteration % self.recompute_neighbors == 0)
            if selected:
                picked = select_neighbors(
                    points("score"), density, stability, self.k, self.h, score_fn, graph,
                    labels=labels, chunk_size=self.chunk_size,
                    iteration=iteration, n_iters=n_iters, memory=memory, metric=dist,
                )
                memory = picked[5]
                if picked[0] is None:                                        # skipped: the current neighbours stay
                    if indices is None:
                        raise ValueError("the selection score function returned None at the first selection: "
                                         "nothing to keep")
                    selected = False
                else:
                    indices, nbr_dist, valid, orphan, h = picked[:5]
                    graph_pairs = graph_values_at(graph, indices).to(x.dtype)  # fuzzy-graph weight of every selected pair

            # 4. Weights
            if (selected and ("score" in spaces or "weight" in spaces)) or (not selected and not weight_fn.static):
                nbr_dist = neighbor_distances(points("weight"), indices, self.chunk_size, dist)
            if selected or not weight_fn.static:
                _, new_w, new_W, memory = prepare_weights(
                    indices, nbr_dist, density, stability, h, weight_fn, valid, n, iteration, n_iters, graph_pairs, memory)
                if new_w is None:                                            # skipped: the previous weights stay
                    if selected:
                        raise ValueError("the neighbour weight function returned None right after neighbours were "
                                         "selected: there are no weights for them to keep")
                else:
                    w, W = new_w, new_W
                    unweighted = w.sum(dim=1) == 0                           # no weight at all: the point stays put

            # 5. Step
            mean = torch.where(unweighted.unsqueeze(1), x, weighted_mean(x, indices, w, W, chunk_size=self.chunk_size))
            moved = x + self.learning_rate * (mean - x)
            if support is not None:
                bias, support, memory = self._support_pull(
                    support, x, density, stability, h, iteration, n_iters, memory, bias_fn, step_fn)
                moved = moved + bias                                          # outside the learning rate
            if self.sphere:
                moved = unit(moved)                                           # back onto the sphere
            if orphan is not None and support is None:
                moved = torch.where(orphan.unsqueeze(1), x, moved)            # with support, orphans still get the bias
            previous, x = x, moved
            if self.keep_trajectory:
                trajectory.append(x.clone())
            mean_change = (x - previous).norm(dim=1).mean().item()
            logger.debug("iteration %d: mean movement %.6f", iteration + 1, mean_change)

            # 6. After the iteration: stability, metric
            stability_before = stability
            if not stability_fn.static:
                stability, memory = update_stability(
                    stability_fn, x, previous, density, stability_before, h, iteration, n_iters, graph, memory)
            metric_step, memory = _call_with_memory(
                metric_fn, (x, previous, density, stability_before, h, iteration, n_iters, graph), memory,
                "the metric function")
            if metric_step is not None:                                      # None: nothing to add
                metric = metric_step if metric is None else metric + metric_step
            if mean_change < self.shift_threshold:
                logger.info("converged after %d iterations", iteration + 1)
                break

        if metric is None:                                                   # the metric was skipped in every iteration
            metric = torch.zeros(n, dtype=x.dtype, device=x.device)
        self.density_ = density                                              # the final density
        self.stability_ = stability                                          # the final scores (1.0 with the default Stability)
        self.memory_ = memory                                                # the shared memory after the run
        self.support_ = support                                              # the final support (None without one)
        return x, metric, trajectory


hypermsde = HypersphericalMeanShiftDensityEnhancement
