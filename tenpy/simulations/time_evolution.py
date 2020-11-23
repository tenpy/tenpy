"""Simulations for (real) time evolution."""
# Copyright 2020 TeNPy Developers, GNU GPLv3

from . import simulation
from .simulation import *

__all__ = simulation.__all__ + ['TimeEvolution']


class TimeEvolution(Simulation):
    default_algorithm = 'TEBDEngine'

    # TODO when exactly to measure
