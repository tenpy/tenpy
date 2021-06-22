"""Simulations for ground state searches."""
# Copyright 2020-2021 TeNPy Developers, GNU GPLv3

from . import simulation
from ..tools import hdf5_io
from .simulation import *

__all__ = simulation.__all__ + ['GroundStateSearch', 'OrthogonalExcitations']


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

    def init_algorithm(self, **kwargs):
        """Initialize the algorithm.

        Options
        -------
        .. cfg:configoptions :: GroundStateSearch

            save_stats : bool
                Whether to include the `sweep_stats` and `update_stats` of the engine into the
                output.
        """
        super().init_algorithm(**kwargs)
        if self.options.get("save_stats", True):
            for name in ['sweep_stats', 'update_stats']:
                stats = getattr(self.engine, name, None)
                if stats is not None:
                    self.results[name] = stats

    def run_algorithm(self):
        E, psi = self.engine.run()
        self.results['energy'] = E

    def resume_run_algorithm(self):
        """Run the algorithm. Calls ``self.engine.run()``."""
        E, psi = self.engine.resume_run()
        self.results['energy'] = E


class OrthogonalExcitations(GroundStateSearch):
    """Find an excitation by another GroundStateSearch orthogalizing against previous states.

    .. note ::
        If you want to find the first excitation in *another* symmetry sector than the ground
        state, you can just run the :class:`GroundStateSearch` search again with an initial state
        in the desired symmetry sector. Charge conservation then forces DMRG (or whatever algorithm
        you use) to stay in that symmetry sector.


    Parameters
    ----------
    orthogonal_to : None list
        States to orthogonalize against.

    Options
    -------
    .. cfg:config :: OrthogonalExcitations
        :include: GroundStateSearch

        N_excitations : int
            Number of excitations to find. Don't make this too big!


    Attributes
    ----------
    orthogonal_to : list
        States to orthogonalize against.
    exctiations : list
        Tensor network states representing the excitations.
        The ground state in `orthogonal_to` is not included in the `excitations`.
        While being optimized, a state is saved as :attr:`psi` and not yet included in
        `excitations`.
    results : dict
        In addition to :attr:`~tenpy.simulations.simulation.Simulation.results`, it contains

            ground_state_energy : float
                Reference energy for the ground state.
            excitations : list
                Tensor network states representing the excitations.
                Only defined if :cfg:option:`Simulation.save_psi` is True.


    """
    def __init__(self, options, *, orthogonal_to=None, **kwargs):
        super().__init__(options, **kwargs)
        resume_data = kwargs.get('resume_data', {})
        if orthogonal_to is None:
            if 'orthogonal_to' in resume_data:
                orthogonal_to = kwargs['resume_data']['orthogonal_to']
                self.options.touch('groundstate_filename')
            else:
                orthogonal_to = self.init_orthogonal_from_groundstate()
        self.orthogonal_to = orthogonal_to
        self.excitations = resume_data.get('excitations', [])
        self.results['excitation_energies'] = []
        if self.options.get('save_psi', True):
            self.results['excitations'] = self.excitations
        assert len(orthogonal_to) > 0

    def init_orthogonal_from_groundstate(self):
        """Initialize :attr:`orthogonal_to` from the ground state.

        Options
        -------
        .. cfg:configoptions :: OrthogonalExcitations

            groundstate_filename :
                File from which the ground state should be loaded.

        """
        gs_fn = self.options['ground_state_filename']
        data = hdf5_io.load(gs_fn)
        data_options = data['simulation_parameters']
        psi0 = data['psi']
        if psi0.bc != 'finite':
            raise NotImplementedError("TODO")
        orthogonal_to = [psi0]
        for key in ['model_class', 'model_params']:
            if key not in self.options and key in data_options:
                self.options[key] = data_options[key]
            # TODO: take segment of inifinte MPO
        self.results['ground_state_energy'] = data['energy']

        if psi0.bc != 'finite':
            raise NotImplementedError("TODO")
            # TODO: take segment of inifinte ground state
            # setup init_env_data
        return orthogonal_to

    def init_state(self):
        """Initialize :attr:`orthogonal_to` from the ground state.

        Options
        -------
        .. cfg:configoptions :: OrthogonalExcitations

            initial_state_from_previous : bool
                If True, initialize `psi` by perturbing the last state in :attr:`orthogonal_to`.
                If False, initialize from the usual initialization parameters as defined in
                :meth:`~tenpy.simulations.simulation.Simulation.init_state`.
        """
        if self.options.get('initial_state_from_previous', True):
            psi = self.orthogonal_to[-1]
            if isinstance(psi, dict):
                psi = psi['ket']
            self.psi = psi.copy()
            # perturb such that overlap with next state is not exactly zero
            self.psi.perturb(close_1=True)
            # TODO envs?
        else:
            # TODO bad idea if from checkpoint?
            del self.psi  # force re-initialization of new state!
            super().init_state()

    def init_algorithm(self, **kwargs):
        kwargs.setdefault('orthogonal_to', self.orthogonal_to)
        super().init_algorithm(**kwargs)

    def run_algorithm(self):
        N_excitations = self.options.get("N_excitations", 1)
        ground_state_energy = self.results['ground_state_energy']
        while len(self.excitations) < N_excitations:

            E, psi = self.engine.run()

            self.results['excitation_energies'].append(E - ground_state_energy)
            self.logger.info("excitation energy: %.14f \n%s", E - ground_state_energy, "+"*80)
            self.excitations.append(psi)
            self.orthogonal_to.append(psi)
            # save in list of excitations
            if len(self.excitations) >= N_excitations:
                break

            self.make_measurements()
            self.init_state()  # initialize a new state to be optimized
            self.init_algorithm()
        # done

    def resume_run_algorithm(self):
        raise NotImplementedError("TODO")


    def prepare_results_for_save(self):
        results = super().prepare_results_for_save()
        if 'resume_data' in results:
            results['resume_run']['excitations'] = self.excitations
            # TODO: further data?!
        return results
