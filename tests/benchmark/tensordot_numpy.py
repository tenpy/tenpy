"""To be used in the `-m` argument of benchmark.py."""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np
import tensordot_npc


def setup_benchmark(**kwargs):
    a, b, axes = tensordot_npc.setup_benchmark(**kwargs)

    axes_a, axes_b = axes
    axes_a = a.get_leg_indices(axes_a)
    axes_b = b.get_leg_indices(axes_b)
    return a.to_ndarray(), b.to_ndarray(), (axes_a, axes_b)


def benchmark(data):
    a, b, axes = data
    x = np.tensordot(a, b, axes)
