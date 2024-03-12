"""To be used in the `-m` argument of benchmark.py."""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np
import tensordot_npc

from combine_numpy import combine_legs


def setup_benchmark(**kwargs):
    a, b, axes = tensordot_npc.setup_benchmark(**kwargs)
    axes_a, axes_b = axes
    non_axes_a = [i for i in range(a.rank) if i not in axes_a]
    non_axes_b = [i for i in range(b.rank) if i not in axes_b]
    a, pipes_a = combine_legs(a.to_ndarray(), (non_axes_a, axes_a))
    b, pipes_b = combine_legs(b.to_ndarray(), (non_axes_b, axes_b))
    return a, b, pipes_a, pipes_b


def split_legs(a, pipes):
    new_shape = [np.prod(p) for p in pipes]
    a = a.reshape(new_shape)
    return a.copy()


def benchmark(data):
    a, b, pipes_a, pipes_b = data
    split_legs(a, pipes_a)
    split_legs(b, pipes_b)
