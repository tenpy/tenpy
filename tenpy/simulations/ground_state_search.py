"""Simulations for ground state searches."""
# Copyright 2020 TeNPy Developers, GNU GPLv3

from . import simulation
from .simulation import *

__all__ = simulation.__all__ + ['GroundStateSearch']


class GroundStateSearch(MPSSimulation):
    default_algorithm = 'TwoSiteDMRGEngine'
