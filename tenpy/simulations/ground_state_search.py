"""Simulations for ground state searches."""
# Copyright 2020 TeNPy Developers, GNU GPLv3

from . import simulation
from .simulation import *

__all__ = simulation.__all__ + ['GroundStateSearch']


class GroundStateSearch(Simulation):
    # TODO document all the config options below!
    # TODO document
    default_algorithm = 'TwoSiteDMRGEngine'
    default_measurements = Simulation.default_measurements + []

    def init_algorithm(self):
        super().init_algorithm()
        if self.options.get("keep_stats", True):
            self.results['sweep_stats'] = self.engine.sweep_stats
            self.results['update_stats'] = self.engine.update_stats

    def run_algorithm(self):
        """Run the algorithm. Calls ``self.engine.run()``."""
        E, psi = self.engine.run()
        self.results['energy'] = E

    def prepare_results_for_save(self):
        results = super().prepare_results_for_save()
        if self.options.get("keep_environment_data", self.options['keep_psi']):
            results['init_env_data'] = self.engine.env.get_initialization_data()
        # hack: remove initial environments from options to avoid blowing up the output size,
        # in particular if `keep_psi` is false, this can reduce the file size dramatically.
        init_env_data = self.options['algorithm_params'].get('init_env_data', {})
        for k in ['init_LP', 'init_RP']:
            if k in init_env_data:
                if isinstance(init_env_data[k], npc.Array):
                    init_env_data[k] = repr(init_env_data[k])
        return results
