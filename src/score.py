"""Neighbour selection scores for HMSDE (the ``score_fn`` of ``MeanShiftDensityEnhancement.forward``).

Every score has the signature of ``src.hmsde.NeighborScore``:
``score(dist, density, stability, h, iteration, n_iters, graph, memory) -> (score, memory)``.
"""
import torch


def original(dist, density, stability, h, iteration, n_iters, graph, memory):
    """The selection of the HMSDE paper (arXiv:2602.03293): the ``k`` nearest neighbours, by distance alone.

    No candidate gets a positive score, so ``select_neighbors`` fills every slot with the nearest candidates.
    """
    return torch.zeros_like(dist), memory
