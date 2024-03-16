"""To be used in the `-m` argument of benchmark.py."""
# Copyright (C) TeNPy Developers, GNU GPLv3

import tensordot_npc


def setup_benchmark(*args, **kwargs):
    a, b, axes = tensordot_npc.setup_benchmark(*args, **kwargs)
    axes_a, axes_b = axes
    non_axes_a = [i for i in range(a.rank) if i not in axes_a]
    non_axes_b = [i for i in range(b.rank) if i not in axes_b]
    return a.combine_legs((non_axes_a, axes_a)), b.combine_legs((non_axes_b, axes_b))


def benchmark(data):
    a, b = data
    a.split_legs([0, 1])
    b.split_legs([0, 1])
    return
