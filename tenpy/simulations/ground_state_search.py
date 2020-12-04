"""Simulations for ground state searches."""
# Copyright 2020 TeNPy Developers, GNU GPLv3

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

    def prepare_results_for_save(self):
        """Bring the `results` into a state suitable for saving.

        For example, this can be used to convert lists to numpy arrays, to add more meta-data,
        or to clean up unnecessarily large entries.

        Options
        -------
        .. cfg:configoptions :: GroundStateSearch

            save_environment_data : bool
                Whether to the environment data should be included into the output :attr:`results`.

        Returns
        -------
        results : dict
            A copy of :attr:`results` containing everything to be saved.
        """
        results = super().prepare_results_for_save()
        if self.options.get("save_environment_data", self.options['save_psi']):
            results['init_env_data'] = self.engine.env.get_initialization_data()
        # hack: remove initial environments from options to avoid blowing up the output size,
        # in particular if `keep_psi` is false, this can reduce the file size dramatically.
        init_env_data = self.options['algorithm_params'].silent_get('init_env_data', {})
        for k in ['init_LP', 'init_RP']:
            if k in init_env_data:
                if isinstance(init_env_data[k], npc.Array):
                    init_env_data[k] = repr(init_env_data[k])
        return results
