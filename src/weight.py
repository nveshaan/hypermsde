"""Neighbour weights for HyperMSDE (the ``weight_fn`` of ``HypersphericalMeanShiftDensityEnhancement.forward``).

Every weight has the signature of ``src.hypermsde.Weight``:
``weight(dist, density, stability, h, iteration, n_iters, graph, memory) -> (weights, memory)``.
"""


def original(dist, density, stability, h, iteration, n_iters, graph, memory):
    """The weights of the HyperMSDE paper (arXiv:2602.03293): each selected neighbour counts by its density alone.

    Distance and stability are ignored. With the default strengths of 1 and a stability of 1.0 this equals
    ``Weight``, but it stays density-only when a ``stability_fn`` is given.
    """
    return density, memory
