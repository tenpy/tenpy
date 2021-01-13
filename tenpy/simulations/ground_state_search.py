"""Simulations for ground state searches."""
# Copyright 2020-2021 TeNPy Developers, GNU GPLv3

from . import simulation
from .simulation import *

__all__ = simulation.__all__ + ['GroundStateSearch']


class GroundStateSearch(Simulation):
    """Simutions for variational ground state searches.

    Parameters
    ----------
    options : dict-like
        The simulation parameters. Ideally, these options should be enough to fully specify all
        parameters of a simulation to ensure reproducibility.

    Options
    -------
    .. cfg:config :: GroundStateSearch
        :include: Simulation

    """
    default_algorithm = 'TwoSiteDMRGEngine'
    default_measurements = Simulation.default_measurements + []

    def init_algorithm(self):
        """Initialize the algortihm.

        Options
        -------
        .. cfg:configoptions :: GroundStateSearch

            save_stats : bool
                Whether to include the
        """
        super().init_algorithm()
        if self.options.get("save_stats", True):
            for name in ['sweep_stats', 'update_stats']:
                stats = getattr(self.engine, name, None)
                if stats is not None:
                    self.results[name] = stats

    def run_algorithm(self):
        """Run the algorithm. Calls ``self.engine.run()``."""
        E, psi = self.engine.run()
        self.results['energy'] = E
