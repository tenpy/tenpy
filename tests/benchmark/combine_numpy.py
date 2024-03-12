"""To be used in the `-m` argument of benchmark.py."""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np
import tensordot_npc


def setup_benchmark(**kwargs):
    a, b, axes = tensordot_npc.setup_benchmark(**kwargs)
    axes_a, axes_b = axes
    non_axes_a = [i for i in range(a.rank) if i not in axes_a]
    non_axes_b = [i for i in range(b.rank) if i not in axes_b]
    return a.to_ndarray(), b.to_ndarray(), ((non_axes_a, axes_a), (axes_b, non_axes_b))


def combine_legs(a, axes):
    axes = list(axes)
    pipe = [[a.shape[i] for i in comb] for comb in axes]
    transp = []
    newshape = []
    for ax in axes:
        transp.extend(ax)
        newshape.append(np.prod([a.shape[i] for i in ax]))
    a = np.transpose(a, transp)
    a = np.reshape(a, newshape)
    return np.ascontiguousarray(a).copy(), pipe


def benchmark(data):
    a, b, axes = data
    axes_a, axes_b = axes
    combine_legs(a, axes_a)
    combine_legs(b, axes_b)
