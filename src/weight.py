"""Neighbour weights for HyperMSDE (the ``weight_fn`` of ``HypersphericalMeanShiftDensityEnhancement.forward``).

Every weight is a ``Component`` with the signature of ``src.hypermsde.Weight``:
``weight(dist, density, stability, h, iteration, n_iters, graph, memory) -> (weights, memory)``.
"""
from .hypermsde import Component


class Original(Component):
    """The weights of the HyperMSDE paper (arXiv:2602.03293): each selected neighbour counts by its density alone.

    Distance and stability are ignored. With the default strengths of 1 and a stability of 1.0 this equals
    ``Weight``, but it stays density-only when a stability update is given. It is not static (the density it reads
    can change every iteration); pass ``static=True`` together with a static density.
    """

    def __call__(self, dist, density, stability, h, iteration, n_iters, graph, memory):
        return density, memory
