"""Metrics for HyperMSDE (the ``metric_fn`` of ``HypersphericalMeanShiftDensityEnhancement.forward``).

Every metric is a ``Component`` with the signature
``metric(moved, x, density, stability, h, iteration, n_iters, graph, memory) -> (metric, memory)``
and returns an (n,) tensor; ``forward`` sums the values over the iterations and returns the sum as its second item.
"""
import torch

from .hypermsde import Component


class Original(Component):
    """The metric of the HyperMSDE paper (arXiv:2602.03293): how far each point moved in the iteration.

    Summed over the iterations it is the distance each point travelled (the sum of its step lengths; chords on
    the sphere).
    """

    @torch.no_grad()
    def __call__(self, moved, x, density, stability, h, iteration, n_iters, graph, memory):
        return (moved - x).norm(dim=1), memory
