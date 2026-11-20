"""Shifts for HyperMSDE (the ``shift_fn`` of ``HypersphericalMeanShiftDensityEnhancement.forward``).

Every shift is a ``Component`` with the signature of ``src.hypermsde.Shift``:
``shift(x, x0, mean, indices, w, density, stability, h, learning_rate, iteration, n_iters, graph, memory)
-> (moved, memory)``.
"""
from .hypermsde import Shift


class Original(Shift):
    """The shift of the HyperMSDE paper (arXiv:2602.03293): a step of ``learning_rate`` towards the weighted mean
    of the selected neighbours, ``x + learning_rate * (mean(x) - x)``. The same as the default ``Shift``."""
