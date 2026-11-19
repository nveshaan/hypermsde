"""Neighbour selection scores for HyperMSDE (the ``score_fn`` of ``HypersphericalMeanShiftDensityEnhancement.forward``).

Every score is a ``Component`` with the signature of ``src.hypermsde.Score``:
``score(dist, density, stability, h, iteration, n_iters, graph, memory) -> (score, memory)``.
"""
import torch

from .hypermsde import Component


class Original(Component):
    """The selection of the HyperMSDE paper (arXiv:2602.03293): the ``k`` nearest neighbours, by distance alone.

    No candidate gets a positive score, so ``select_neighbors`` fills every slot with the nearest candidates.
    """

    def __call__(self, dist, density, stability, h, iteration, n_iters, graph, memory):
        return torch.zeros_like(dist), memory
