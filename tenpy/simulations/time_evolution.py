"""Simulations for (real) time evolution."""
# Copyright 2020-2021 TeNPy Developers, GNU GPLv3

import numpy as np

from . import simulation
from .simulation import *

__all__ = simulation.__all__ + ['RealTimeEvolution']


class RealTimeEvolution(Simulation):
    """Perform a real-time evolution on a tensor network state.

    Parameters
    ----------
    options : dict-like
        The simulation parameters. Ideally, these options should be enough to fully specify all
        parameters of a simulation to ensure reproducibility.

    Options
    -------
    .. cfg:config :: TimeEvolution
        :include: Simulation

        final_time : float
            Mandatory. Perform time evolution until ``engine.evolved_time`` reaches this value.
            Note that we can go (slightly) beyond this time if it is not a multiple of
            the individual time steps.
    """
    default_algorithm = 'TEBDEngine'
    default_measurements = Simulation.default_measurements + [
        ('tenpy.simulations.measurement', 'evolved_time'),
    ]

    def __init__(self, options):
        super().__init__(options)
        self.final_time = self.options['final_time'] - 1.e-10  # subtract eps: roundoff errors

    def run_algorithm(self):
        """Run the algorithm. Calls ``self.engine.run()`` and :meth:`make_measurements`."""
        # TODO: more fine-grained/custom break criteria?
        while True:
            if np.real(self.engine.evolved_time) >= self.final_time:
                break
            self.engine.run()
            # for time-dependent H (TimeDependentExpMPOEvolution) the engine can re-init the model;
            # use it for the measurements....
            self.model = self.engine.model

            self.logger.info("reached time %.2f, max chi=%d", self.engine.evolved_time.real,
                             max(self.psi.chi))
            # TODO: call engine.checkpoint.emit() ?
            self.make_measurements()

    def resume_run_algorithm(self):
        self.run_algorithm()

    def final_measurements(self):
        """Do nothing.

        We already performed a set of measurements after the evolution in :meth:`run_algorithm`.
        """
        pass
