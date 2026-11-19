# HyperMSDE

Mean-shift density enhancement (HyperMSDE) of embeddings, in [`src/hypermsde.py`](src/hypermsde.py), with the default of every replaceable component (a class with a `static` flag) in the same file. The alternative `Original` density, metric, score and weight of the paper are in [`src/density.py`](src/density.py), [`src/metric.py`](src/metric.py), [`src/score.py`](src/score.py) and [`src/weight.py`](src/weight.py). Every point is repeatedly moved towards a weighted mean of its neighbours until the movement converges, and the embeddings stay on the unit sphere throughout. Only `torch` is required.

```python
import torch
from src.hypermsde import hypermsde

X = torch.randn(3000, 512)                      # any (n, d) float embeddings; they are normalised internally

model = hypermsde(k=100, learning_rate=0.1, device="mps")   # "cpu" / "cuda" / "mps"
shifted, movement, trajectory = model(X)
```

`shifted` is the `(n, d)` result (unit-norm unless `sphere=0`), `movement` the `(n,)` distance each point travelled (summed step lengths: chords on the sphere), and `trajectory` a list of states (only if `keep_trajectory=True`). With `sphere` 2 or 3 all of them are in the centred space. Everything runs under `torch.no_grad()`: nothing is differentiable.

## Pipeline

`forward` runs these stages in order.

1. **Normalise** (`sphere`). With `sphere=0` nothing is normalised and the whole loop runs in the raw feature space. With 1 the input is scaled to unit length, with 2 it is also mean-subtracted and normalised again, and with 3 it is also centred on its Fréchet mean. In all three, points are re-normalised immediately after every shift, so every state the loop passes through, every trajectory entry and the output are unit-norm. (A zero vector has no direction: it is pulled along by its neighbours and is unit-norm after the first shift.)
2. **Density** of every point. By default it is the adaptive angular k-NN estimate `Density` (see [The densities](#the-densities)), which is **not static**: it is re-estimated on the current points at the start of every iteration. A static density (`Density(static=True)`, or `Original`) is computed once, at the first iteration. The UMAP fuzzy graph of the initial points is built in any case, kept, and handed to the density, score, weight and stability components. A `density_fn` replaces the default (see [Replacing the density, ...](#replacing-the-density-the-selection-score-the-neighbour-weights-and-the-stability-update)).
3. **Select neighbours.** In the geometry chosen by `dist` (Euclidean or cosine) of the current points, every other point is scored by `Score` and the `k` best are kept: with the default score `density^a * stability^b * exp(-c * d^2 / (2 h^2))`, so a dense, stable point can outrank a closer, sparse one. All points are candidates (no distance pre-filter) and a point is never its own neighbour. Candidates with a positive score rank by score; if fewer than `k` have one, the remaining slots go to the nearest candidates.
4. **Weights.** Each selected neighbour gets a weight from `Weight` (default `density^a * stability^b`); weights of picks that violate the label mask are zeroed and rows are normalised to sum to 1. A point whose weights are all zero stays where it is. The weights are always computed when neighbours are selected. If the weight is **not static** (the default `Weight` is not) they are also computed at the start of every other iteration, with the current distances to the selected neighbours and the current density and stability; a static weight (`Weight(static=True)`) is computed only when neighbours are selected, so it only makes sense with a static density and stability.
5. **Step.** `mean = W @ X` with `W` a sparse `(n, n)` matrix holding the `n*k` normalised weights (one sparse matmul), then `x' = x + learning_rate * (mean - x)`, plus `bias` from `bias_fn` when `support` is given, normalised back onto the sphere unless `sphere=0`.
6. **Repeat** until the mean per-point movement falls below `shift_threshold` (convergence) or `max_iters_shift` iterations have run. With `recompute_neighbors = N > 0`, stage 3 (the neighbour selection) is redone every `N` iterations on the current points (iterations 0, N, 2N, ...); a static density is never recomputed. A non-static density, a non-static weight, a non-static `stability_fn` and the `metric_fn` run in every iteration regardless of `recompute_neighbors`. Within an iteration the order is: density update (`density_fn`) → select neighbours (if due) → weights → support bias and step (`bias_fn`, `step_fn`) → step → stability update (`stability_fn`) → metric update (`metric_fn`).

## Constructor

The arguments up to `device` are those of the original HyperMSDE (arXiv:2602.03293); the ones from `recompute_neighbors` on were added in this repo.

| Argument | Default | Meaning |
| --- | --- | --- |
| `k` | 100 | Neighbours each point is pulled towards; needs `k <= n - 1`. |
| `n_neighbors` | 15 | Fuzzy graph: neighbours of every point (`<= n`). |
| `n_epochs` | 200 | Fuzzy graph: edges weaker than `max / n_epochs` are dropped from the graph. |
| `learning_rate` | 0.1 | Fraction of the way a point moves towards its weighted mean per iteration. |
| `max_iters_shift` | 1000 | Safety cap on iterations; the loop normally stops earlier by convergence (see [Speed](#speed) for typical counts). |
| `shift_threshold` | 1e-4 | Convergence: stop when the mean per-point movement is below this. |
| `keep_trajectory` | `False` | Also return the state after every iteration. |
| `chunk_size` | 1024 | Rows handled at a time by **every** chunked step, which bounds the memory of the run: the exact neighbour searches (the fuzzy graph, the default density and `h="auto"`), the graph-space distances of `Original`, the `(chunk, n)` blocks of the neighbour selection, and the `(chunk, k, d)` gathers of the live neighbour distances and of the weighted-mean fallback. Lower it if memory runs out. In principle it can change results through floating-point rounding in the distances (a different block shape can round differently); in tests it did not: sizes from 32 to 4096 gave identical outputs on synthetic data, and sizes from 256 to 4096 on ARMD OCT embeddings (MPS), with the same run time to within about 5%. |
| `device` | auto | `"mps"`, `"cuda"` or `"cpu"`; inputs are moved there. |
| `recompute_neighbors` | 0 | 0 / `None`: select neighbours once; `N > 0`: reselect every `N` iterations (never, if the `score_fn` is static). |
| `sphere` | 1 | 0: no normalisation, the loop runs in the raw feature space. 1: normalise onto the unit sphere, and back onto it after every shift. 2: as 1, after first subtracting the Euclidean mean of the unit-norm points and normalising again, `normalise(x - mean(x))`; it is not distance-preserving, and the mean is `model.center_`. 3: as 1, after first centring the points on their Fréchet (Karcher) mean: each is mapped to the tangent space at the mean (log map) and put back on the sphere, i.e. `normalise(x - (x·mu) mu)`, which removes the shared mean direction (every point becomes orthogonal to the mean). It does not spread the points evenly. With 2 and 3, everything after the centring (density, selection, output, the trajectory's first state, `movement`) is in the centred space and the output is **not** mapped back; the mean is `model.center_` (`None` for 0 and 1). |
| `distance_strength` | 1.0 | Strength `c` of the distance term of the default selection score. |
| `density_strength` | 1.0 | Strength `a` of the density term, in the default selection score and neighbour weight. |
| `stability_strength` | 1.0 | Strength `b` of the stability term, in the default selection score and neighbour weight. |
| `h` | `"auto"` | Kernel width of the distance term, a positive number in the units of `dist`, or `"auto"` = median distance to the `k` nearest neighbours, re-estimated at every selection. |

A strength is a number `>= 0`; 0 switches that factor off and larger values weight it more (they are exponents on density and stability, and a multiplier on the squared distance in the kernel).

## `forward(X, labels=None, support=None, dist="euclidean", density_fn=None, score_fn=None, weight_fn=None, stability_fn=None, metric_fn=None, bias_fn=None, step_fn=None, sphere=None)`

| Argument | Meaning |
| --- | --- |
| `X` | `(n, d)` floating-point embeddings. |
| `labels` | `(n,)` integer labels restricting neighbour selection: every point may only pick neighbours labelled 0, whatever its own label. A point with no valid neighbour is left where it is (with `support`, it still receives the bias). |
| `support` | Optional list of `(d,)` tensors (or an `(s, d)` tensor): high-mass support points that act on `X` only through `bias_fn` (default `Bias`). See [Support points](#support-points). |
| `dist` | `"euclidean"` (default) or `"cosine"` (`1 - cos`; on the unit sphere this is half the squared Euclidean distance, so it orders pairs the same way but on a different scale, which matters for `h` and the distance term). It sets the distance of the whole run: the k-NN that builds the fuzzy graph, the neighbour selection, `h="auto"`, and the `dist` that `score_fn` and `weight_fn` receive. The points always stay on the unit sphere. Any other value raises a `ValueError`. |
| `density_fn` | Replacement for the density. Called at the start of **every** iteration, before the neighbours are selected, unless it is static: then it is called once, at the first iteration. Without one, `Density` (not static) is used. The final density is available as `model.density_`. |
| `score_fn` | Replacement for the selection score. Called whenever neighbours are selected, including the initial selection, so with `recompute_neighbors=0` it defines the single neighbour graph the run uses. |
| `weight_fn` | Replacement for the neighbour weights of the weighted mean. Evaluated whenever neighbours are selected and, unless it is static, at the start of every other iteration too, with the current distances and stability scores. `Weight` (not static) is used when it is not given. |
| `stability_fn` | Optional component that updates the stability scores (`(n,)`, in `[0, 1]`, higher = more stable) after every iteration. It receives the new and previous positions. Without one, `Stability` is used, which does nothing: every point stays at 1.0. See [Stability](#stability). |
| `metric_fn` | Optional component evaluated after every iteration. Its per-iteration values are summed and returned. Without one, `Metric` is used: the `(n,)` summed movement (`Original` of [`src/metric.py`](src/metric.py) does the same). |
| `bias_fn` | The pull of the support on the points. Without one, `Bias` is used: a step of `learning_rate` towards the mean of the support points. Called every iteration; its `(n, d)` result is added to every point's step, outside the learning rate. |
| `step_fn` | Optional step of the support points, called every iteration with the same arguments as `bias_fn`. Without one, `Step` is used, which does nothing: the support stays where it is. Needs `support`. |
| `sphere` | Optional dict `{stage: mode}` giving a stage its own space. `stage` is `"graph"` (the fuzzy graph), `"density"`, `"score"` (the distances of the neighbour selection and `h="auto"`) or `"weight"` (the live neighbour distances); `mode` is 0 (raw), 1 (unit), 2 (mean-subtracted, unit) or 3 (Fréchet-centred), as for the model's `sphere`. The raw `X` is mapped **once**, before the iterations, and that copy stays frozen for the whole run, so such a stage ignores the moving points. A stage that is not in the dict sees the current points, in the space of the model's `sphere`. The other stages (weighted mean, bias, step, stability, metric) always use the space of the model, and any other key raises a `ValueError`. With the model's `sphere` 0 or 1 nothing else is checked; with 2 or 3, a stage set to 2 or 3 needs a **static** component (`"graph"` always qualifies), else a `ValueError`. |

### Replacing the density, selection score, neighbour weights, stability update, returned metric and support functions

These are instances of `Component` (in `src/hypermsde.py`), classes that carry their own parameters; a bare function or lambda raises a `TypeError`. A component implements `__call__` with the signature below and has a **`static`** flag (class attribute, `False` unless the class says otherwise; `Component(static=...)`, and every default's constructor, overrides it per instance):

| component | `static=True` means |
| --- | --- |
| density | computed once, at the first iteration (default `Density`: `False`; `Original`: `True`) |
| weight | prepared only when neighbours are selected, not at every iteration (default `Weight` and `Original`: `False`) |
| stability, step | does nothing, so `forward` never calls it (defaults `Stability` and `Step`: `True`) |
| score | neighbours are selected once, at the first iteration, whatever `recompute_neighbors` says (default `Score` and `Original`: `False`) |
| metric, bias | informational only: always called when due |

Besides the arguments below, all of them receive and return the **shared memory dict** (next section), and all of them receive the **0-based index of the current iteration** and the **iteration cap** `n_iters` (`max_iters_shift`; a run may stop earlier by convergence), so they can change over the run, and the **UMAP fuzzy graph** that was computed for the initial points (below). The defaults, all in `src/hypermsde.py`, are the callable classes `Density(k, clip, chunk_size)` for the density, `Score(distance_strength, density_strength, stability_strength)` and `Weight(density_strength, stability_strength)`; `Metric()`, `Bias()`, and `Stability()` and `Step()` which do nothing (the stability stays 1.0 for every point and the support stays put); the last two are static, so `forward` skips them. `Original` in `src/score.py` and `src/weight.py` are the paper's score and weight.

```python
density_fn(x, density, stability, iteration, n_iters, graph, memory)
        -> (density, memory)
    # x: (n, d) unit-norm points as they are now; density: (n,) density before this update (all 1.0 at the first
    # iteration); stability: (n,) current scores; graph: the full (n, n) sparse fuzzy graph of the initial points
    # density: (n,), finite and >= 0; larger = denser

score_fn(dist, density, stability, h, iteration, n_iters, graph, memory)
        -> (score, memory)
    # dist: (chunk, n) distances from a chunk of points to all points; density, stability: (n,) per candidate
    # graph: (chunk, n) rows of the UMAP fuzzy graph, aligned with dist
    # h: kernel width in use; score: (chunk, n), finite; larger = more attractive;
    # <= 0 means "no attraction" (only used to fill remaining slots)

weight_fn(dist, density, stability, h, iteration, n_iters, graph, memory)
        -> (weights, memory)
    # dist, density, stability, graph: (n, k), for the k selected neighbours of every point
    # (dist = their CURRENT distances, stability = their current scores,
    #  graph = fuzzy-graph weight between each point and each selected neighbour, 0 if there is no such edge)
    # weights: (n, k), finite and >= 0; normalised per row afterwards

stability_fn(moved, x, density, stability, h, iteration, n_iters, graph, memory)
        -> (stability, memory)
    # moved and x: (n, d) positions after and before this iteration; density, stability: (n,)
    # graph: the full (n, n) sparse fuzzy graph; stability: (n,) scores in [0, 1]

metric_fn(moved, x, density, stability, h, iteration, n_iters, graph, memory)
        -> (metric, memory)
    # moved and x: (n, d) positions after and before this iteration; other arguments match stability_fn
    # metric values are summed over iterations and become the second item returned by forward

bias_fn(dist, density, stability, h, learning_rate, iteration, n_iters, memory)
        -> (bias, memory)
    # dist: (s, n, d) feature-wise differences support[j] - x[i] at the current positions (s support points);
    # density, stability: (n,) of the points; h: kernel width in use; learning_rate: the model's
    # bias: (n, d), finite; added to every point's step: x + learning_rate * (mean - x) + bias

step_fn(dist, density, stability, h, learning_rate, iteration, n_iters, memory)
        -> (step, memory)
    # same arguments as bias_fn, at the same (pre-step) positions
    # step: (s, d), finite; added to the support points
```

**Memory.** At the start of `forward`, before the first iteration, one empty dict is created. **Every** component receives it after its regular arguments and returns it, updated, after its result as `(result, memory)`; the returned dict is what the next call (of any of the components) receives. It is a plain dict, so the components can keep running statistics, counters or caches in it, and read what another component wrote (use distinct keys; nothing enforces ownership). A new `forward` call starts with an empty dict again; the final one is available as `model.memory_`. `score_fn` is called once per block of `chunk_size` rows, so the memory passes through the blocks in turn within a selection. A component that returns anything other than a 2-tuple whose second entry is a dict raises a `ValueError`; the default `Score` and `Weight` pass the memory through untouched.

| component | called | `iteration` is |
| --- | --- | --- |
| `density_fn` | at the start of every iteration, before any selection (once, at the first, if static) | the iteration about to run |
| `score_fn` | whenever neighbours are selected: at the start, and every `recompute_neighbors` iterations after | the iteration whose step will use the selected neighbours |
| `weight_fn` | whenever neighbours are selected, and at every other iteration unless static, after any selection | the iteration whose step will use the weights |
| `stability_fn` | after every iteration (never, if static) | the iteration that just finished |
| `metric_fn` | after every iteration | the iteration that just finished |
| `bias_fn` | every iteration, after the weights, before the step | the iteration whose step will use the bias |
| `step_fn` | every iteration, right after `bias_fn` | the iteration whose step it is |

**The fuzzy graph.** The graph is a UMAP-style fuzzy `n_neighbors`-nearest-neighbour graph of the initial points: every point has a per-point scale, an edge to each of its `n_neighbors` nearest points, the two directions of a pair are combined by fuzzy union, and edges weaker than `max / n_epochs` are dropped. Weights are in `(0, 1]`. There is one entry per directed nearest-neighbour edge (about `n_neighbors` per point), so the matrix is **not** symmetric: on 3,240 ARMD OCT embeddings (CLIP) about 58% of the edges also exist in the reverse direction. It is computed **once**, before the first iteration, and is the same for the whole run. Each function gets the view that fits its other arguments: the `(chunk, n)` dense rows aligned with `dist` for `score_fn` (0 where there is no edge), the `(n, k)` values for the selected `(point, neighbour)` pairs for `weight_fn` (0 where a selected neighbour is not a graph neighbour; this differs from the graph neighbours because selection also looks at density), and the whole `(n, n)` coalesced sparse tensor for `density_fn`, `stability_fn` and `metric_fn`.

Within an iteration the order is: density update → select neighbours (if due) → weights → support bias and step → step → stability update → metric update. A component with an older signature (without `iteration`, `n_iters`, `graph` or the memory) raises a `TypeError`. `dist` is the distance chosen by the `dist` argument of `forward`; for `weight_fn` it is recomputed from the current positions at every iteration it is prepared. Shape or value violations raise `ValueError`. Examples:

```python
from src.hypermsde import hypermsde, Component, Density, Score, Weight
import torch

model = hypermsde(k=30, recompute_neighbors=1)

# weaker emphasis on distance, stronger on stability, chosen per call
shifted, _, _ = model(X, score_fn=Score(distance_strength=0.5, density_strength=1, stability_strength=2))

# a custom score with its own parameter: ignore distance beyond a cutoff
class CutoffScore(Component):
    def __init__(self, cutoff=3.0):
        super().__init__()
        self.cutoff = cutoff
    def __call__(self, dist, density, stability, h, iteration, n_iters, graph, memory):
        return density * stability * (dist < self.cutoff * h), memory

# distance-aware neighbour weights: attraction decays with the points' current distance (not static: the
# distances change at every iteration)
class GravityWeight(Component):
    def __call__(self, dist, density, stability, h, iteration, n_iters, graph, memory):
        return density * stability * torch.exp(-dist.square() / (2 * h * h)), memory
shifted, _, _ = model(X, weight_fn=GravityWeight())

# a schedule: let the distance matter more and more as the run progresses
class AnnealedScore(Component):
    def __init__(self, start=0.2, end=3.0):
        super().__init__()
        self.start, self.end = start, end
    def __call__(self, dist, density, stability, h, iteration, n_iters, graph, memory):
        strength = self.start + (self.end - self.start) * iteration / max(n_iters - 1, 1)
        return Score(distance_strength=strength)(dist, density, stability, h, iteration, n_iters, graph, memory)
shifted, _, _ = model(X, score_fn=AnnealedScore())

# using the fuzzy graph: neighbours that are also graph neighbours count extra. Static: it reads nothing that
# changes (density and stability stay at their initial values here), so it is prepared only at selection
class ConnectedWeight(Component):
    static = True
    def __call__(self, dist, density, stability, h, iteration, n_iters, graph, memory):
        return density * stability * (0.1 + graph), memory
shifted, _, _ = model(X, density_fn=Density(static=True), weight_fn=ConnectedWeight())

# a density that follows the points: the number of others within a radius of their current positions
class LiveDensity(Component):
    def __init__(self, radius=1.0):
        super().__init__()
        self.radius = radius
    def __call__(self, x, density, stability, iteration, n_iters, graph, memory):
        return (torch.cdist(x, x) < self.radius).sum(dim=1).float() - 1, memory
shifted, _, _ = model(X, density_fn=LiveDensity())

# a stability that compares a point's displacement with the average displacement of its graph neighbours
class MovesLikeNeighbours(Component):
    def __call__(self, moved, x, density, stability, h, iteration, n_iters, graph, memory):
        displacement = moved - x
        weight_sum = torch.sparse.mm(graph, torch.ones(x.shape[0], 1, device=x.device)).clamp_min(1e-9)
        local = torch.sparse.mm(graph, displacement) / weight_sum
        scores = torch.exp(-(displacement - local).norm(dim=1) / displacement.norm(dim=1).median().clamp_min(1e-9))
        return scores, memory
shifted, _, _ = model(X, stability_fn=MovesLikeNeighbours())

# the density and the neighbour selection in their own frozen spaces, whatever the loop does
shifted, _, _ = model(X, sphere={"density": 3, "score": 1})
```

The `sphere` dict only changes the space of the stages it names; with the model's own `sphere` 2 or 3, `sphere={"density": 3}` needs `density_fn=Density(static=True)`.

With `recompute_neighbors=0` the neighbour graph is fixed after the first selection, so a `score_fn` is called once; the weights, the density and the stability keep following the moving points as long as `weight_fn` and `density_fn` are not static.

### Stability

Every point starts with stability 1.0. A `stability_fn` (a component that carries its own parameters) updates the scores **after every iteration** from what the points just did; its signature is given above. `moved` and `x` are the new and previous unit positions, and `stability` contains the scores before this update (all 1.0 at the start), so running updates are possible.

- **Timing.** The first neighbour selection and the first step use scores of 1.0, since there is no step yet. After that the updated scores feed the neighbour weights of the next iteration (prepared at every iteration if the weight is not static, as the default is) and, whenever neighbours are reselected (`recompute_neighbors > 0`), the selection score. With `recompute_neighbors=0` the neighbour graph itself is fixed by the initial scores, but the weights on it still follow the stability.
- **Result.** The final scores are available afterwards as `model.stability_` (all 1.0 without a `stability_fn`).
- **A fixed set of scores** can be supplied with a small `Component` whose `__call__` returns `(my_scores, memory)`; they take effect from the second iteration on.
- **Cost.** Per iteration a `stability_fn` adds one call. A non-static weight (the default) rebuilds the weights every iteration (live distances `O(n k d)`, and the sparse weight matrix `O(n k log(n k))`), and a non-static density (the default) one k-NN search; `Density(static=True)` and `Weight(static=True)` give that cost up and the weights then follow stability only at the next selection.

```python
import torch

from src.hypermsde import Component

class MoversUnstable(Component):
    # a point that moves much more than the typical point in an iteration is unstable
    def __call__(self, moved, x, density, stability, h, iteration, n_iters, graph, memory):
        size = (moved - x).norm(dim=1)
        return torch.exp(-size / size.median().clamp_min(1e-9)), memory

class RunningStability(Component):
    # an exponential moving average of the same score, with its own parameter
    def __init__(self, momentum=0.5):
        super().__init__()
        self.momentum = momentum
    def __call__(self, moved, x, density, stability, h, iteration, n_iters, graph, memory):
        size = (moved - x).norm(dim=1)
        target = torch.exp(-size / size.median().clamp_min(1e-9))
        memory["median_step"] = memory.get("median_step", []) + [float(size.median())]   # the memory keeps a history
        return (1 - self.momentum) * stability + self.momentum * target, memory

model = hypermsde(k=30, recompute_neighbors=1)
shifted, movement, _ = model(X, stability_fn=RunningStability(momentum=0.3))
scores = model.stability_            # (n,) in [0, 1]
history = model.memory_["median_step"]   # one entry per iteration
```

### Support points

`support` adds high-mass points, e.g. a few known normal embeddings for few-shot inference. They interact with `X` **only** through `bias_fn`: they are never neighbour candidates and take no part in the density, the fuzzy graph or `h="auto"`.

- **Same space as `X`.** They are normalised onto the unit sphere with `sphere` 1, 2 or 3, and with `sphere` 2 or 3 centred with the same mean as `X` (`model.center_`), the same map as `X`.
- **The bias.** Every iteration `bias_fn` receives `dist[j, i] = support[j] - x[i]` (pointing from each point towards each support point) and returns an `(n, d)` bias. Each point then moves by `learning_rate * (mean - x) + bias` (the bias is not scaled by `learning_rate`; it is passed in so `bias_fn` can use it), then goes back onto the sphere. Points with no valid neighbour still receive the bias. The bias counts towards `movement` and convergence.
- **Moving the support.** With `step_fn`, the support moves by its `(s, d)` result every iteration (computed at the same positions as the bias) and goes back onto the sphere with `sphere` 1, 2 or 3. Without one it stays where it is. Its movement does not count towards convergence. The final support is `model.support_` (`None` without one); the trajectory holds `X` only.
- **Errors.** `bias_fn` or `step_fn` without `support`, an empty list, points that are not `(d,)`, and results of the wrong shape or with non-finite values raise a `ValueError`.
- **Memory.** `dist` is a dense `s x n x d` tensor built every iteration: 16 support points, 10,000 points and `d = 1152` are about 0.7 GB in float32.

```python
class Pull(Component):
    def __call__(self, dist, density, stability, h, learning_rate, iteration, n_iters, memory):
        return learning_rate * dist.mean(dim=0), memory      # towards the support's centroid

model = hypermsde()
shifted, movement, _ = model(X, support=[X[i] for i in normal_idx], bias_fn=Pull())
```

## Speed

The two heavy matrix products are sparse matmuls, kept on purpose:

- the density's radius counting (`S @ chunk.T`, about 20 passes), which dominates the density cost;
- the weighted mean `W @ X` of every iteration.

If a device cannot do sparse matmul (probed once per device), the weighted mean falls back to a chunked gather. Measured for one weighted-mean step with `k = 50`:

| device | `n` | `d` | sparse `W @ X` | gather | gather / sparse |
| --- | --- | --- | --- | --- | --- |
| CPU | 3,240 | 512 | 34 ms | 235 ms | 7.0x |
| CPU | 3,240 | 1152 | 88 ms | 528 ms | 6.0x |
| CPU | 16,000 | 512 | 362 ms | 1164 ms | 3.2x |
| Apple MPS | 3,240 | 512 | 2.7 ms | 45 ms | 17x |
| Apple MPS | 3,240 | 1152 | 3.0 ms | 101 ms | 34x |
| Apple MPS | 16,000 | 512 | 11 ms | 222 ms | 20x |

Cost model (`n` points of dimension `d`; `P` about 20 counting passes, `q` = `n_neighbors`):

- density: `O(n^2 d)` for the fuzzy graph's k-NN search, once, in any case; the default `Density` adds one more k-NN search, `O(n^2 d)`, while `Original` adds `O(P n^2 q)` for the counting passes;
- selection: `O(n^2 d)` time and `O(chunk_size * n)` working memory per selection (plus one extra k-NN pass when `h="auto"`); every chunked step uses the same `chunk_size`, so peak memory is about `chunk_size * n` values per block (several blocks at once during selection) or `chunk_size * k * d` for the neighbour gathers;
- each iteration: one sparse matmul, `O(n k d)`.

With the earlier defaults (`k = 50`, `nbd_sample_count_threshold = 30`, `learning_rate = 0.3`), 402 of 405 benchmark runs (27 datasets x 5 encoders x 3 seeds, 200-4,000 points, `d` 512-1152) converged within the cap: a median of 50 iterations, 90th percentile 164, and some cells needing several hundred (the largest converged run took 988). The three that did not converge were the same dataset/encoder pair and had a final mean movement of about 1.5e-4, just above the threshold. About 19% of runs needed more than 100 iterations, so a cap of 100 stopped them early, and for some cells that changed the resulting scores noticeably. The current `learning_rate = 0.1` takes smaller steps, so expect more iterations than these counts. On an Apple MPS device the average run took about 0.7 s. `torch.compile` is used for the k-NN, graph and distance kernels, so the first call for a new shape includes compilation time.
