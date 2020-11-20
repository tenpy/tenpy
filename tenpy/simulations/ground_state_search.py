"""Simulations for ground state searches."""
# Copyright 2020 TeNPy Developers, GNU GPLv3

from . import simulation
from .simulation import *

__all__ = simulation.__all__ + ['GroundStateSearch']


class GroundStateSearch(MPSSimulation):
    default_algorithm = 'TwoSiteDMRGEngine'

    def run_algorithm(self):
        """Run the algorithm. Calls ``self.engine.run()``."""
        E, psi = self.engine.run()
        self.results['energy'] = E
