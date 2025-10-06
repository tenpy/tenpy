"""Simulations for ground state searches."""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np
from pathlib import Path

from . import simulation
from ..tools import hdf5_io, string
from .simulation import *  # noqa F403
from ..linalg import np_conserved as npc
from ..networks.mpo import MPOEnvironment, MPOTransferMatrix
from ..networks.mps import MPS, InitialStateBuilder
from ..networks.uniform_mps import UniformMPS
from ..algorithms.mps_common import ZeroSiteH
from ..linalg import krylov_based
from ..linalg.sparse import SumNpcLinearOperator
from ..tools.misc import find_subclass
from ..tools.params import asConfig

__all__ = simulation.__all__ + [
    'GroundStateSearch',
    'PlaneWaveExcitations',
    'OrthogonalExcitations',
    'TopologicalExcitations',
    'ExcitationInitialState',
]


class GroundStateSearch(Simulation):
    """Simulation for variational ground state searches.

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
        if self.options.get("save_stats", True, bool):
            for name in ['sweep_stats', 'update_stats']:
                stats = getattr(self.engine, name, None)
                if stats is not None:
                    self.results[name] = stats

    def run_algorithm(self):
        E, psi = self.engine.run()
        self.results['energy'] = E

    def resume_run_algorithm(self):
        """Run the algorithm.

        Calls ``self.engine.run()``.
        """
        E, psi = self.engine.resume_run()
        self.results['energy'] = E


class PlaneWaveExcitations(GroundStateSearch):
    """Simulation for plane-wave excitations.

    Parameters
    ----------
    options : dict-like
        The simulation parameters. Ideally, these options should be enough to fully specify all
        parameters of a simulation to ensure reproducibility.

    Options
    -------
    .. cfg:config :: PlaneWaveExcitations
        :include: Simulation
    """
    default_algorithm = 'PlaneWaveExcitationEngine'

    def __init__(self, options, *, gs_data=None, **kwargs):
        super().__init__(options, **kwargs)
        resume_data = kwargs.get('resume_data', {})
        self.excitations = resume_data.get('excitations', [])
        self.results['excitation_energies'] = []
        if self.options.get('save_psi', True):
            self.results['excitations'] = self.excitations
        self.init_env_data = {}
        self._gs_data = gs_data
        self.initial_state_builder = None
        assert 'group_sites' not in self.options.keys(), 'No grouping allowed for Plane Wave through simulations since we cannot ungroup.'

    def run(self):
        self.load_groundstate()
        return super().run()

    def resume_run(self):
        self.load_groundstate()
        return super().resume_run()

    def load_groundstate(self):
        """Load ground state and convert to uMPS.

        Load the ground state and initialize the model from it.

        Options
        -------
        .. cfg:configoptions :: OrthogonalExcitations

            ground_state_filename :
                File from which the ground state should be loaded.
            orthogonal_norm_tol : float
                Tolerance how large :meth:`~tenpy.networks.mps.MPS.norm_err` may be for states
                to be added to :attr:`orthogonal_to`.

        Returns
        -------
        gs_data : dict
            The data loaded from :cfg:option:`OrthogonalExcitations.ground_state_filename`.
        """
        gs_fn, gs_data = self._load_gs_data()
        gs_data_options = gs_data['simulation_parameters']
        # initialize original model with model_class and model_params from ground state data
        self.logger.info("initialize original ground state model")
        for key in gs_data_options.keys():
            if not isinstance(key, str) or not key.startswith('model'):
                continue
            if key not in self.options:
                self.options[key] = gs_data_options[key]
        self.init_model()

        # initialize original state
        self.psi = gs_data['psi']  # no copy!
        assert isinstance(self.psi, MPS) or isinstance(self.psi, UniformMPS)
        tol = self.options.get('orthogonal_norm_tol', 1.e-12, 'real')
        if np.linalg.norm(self.psi.norm_test()) > tol:
            if isinstance(self.psi, MPS):
                self.logger.info("call psi.canonical_form() on ground state")
                self.psi.canonical_form()
            else:
                raise ValueError('uMPS does not pass norm test. Run VUMPS to get ground state or \n' +
                                 'convert to MPS and canonicalize.')
        if isinstance(self.psi, MPS):
            self.psi = UniformMPS.from_MPS(self.psi)

        resume_data = gs_data.get('resume_data', {})
        if resume_data.get('converged_environments', False):
            self.logger.info("use converged environments from ground state file")
            env_data = resume_data['init_env_data']
            write_back = False
        else:
            self.logger.info("converge environments with MPOTransferMatrix")
            guess_init_env_data = resume_data.get('init_env_data', None)
            H = self.model.H_MPO
            env_data = MPOTransferMatrix.find_init_LP_RP(H, self.psi, 0, None,
                                                         guess_init_env_data)
            write_back = self.options.get('write_back_converged_ground_state_environments', False, bool)
        self.init_env_data = env_data

        if write_back:
            self.write_back_environments(gs_data, gs_fn)
        return gs_data

    def _load_gs_data(self):
        """Load ground state data from `ground_state_filename` or use simulation kwargs."""
        if self._gs_data is not None:
            gs_fn = None
            self.logger.info("use ground state data of simulation class arguments")
            gs_data = self._gs_data
            self._gs_data = None  # reset to None to potentially allow to free the memory
            # even though this can only work if the call structure is
            #      sim = OrthogonalExcitations(..., gs_data=gs_data)
            #      del gs_data
            #      with sim:
            #          sim.run()
        else:
            gs_fn = self.options['ground_state_filename']
            self.logger.info("loading ground state data from %s", gs_fn)
            gs_data = hdf5_io.load(gs_fn)
        return gs_fn, gs_data

    def write_back_environments(self, gs_data, gs_fn):
        """Write converged environments back into the file with the ground state.

        Parameters
        ----------
        gs_data : dict
            Data loaded from the ground state file.
        gs_fn : str | None
            Filename where to save `gs_data`. Do nothing if `gs_fn` is None.
        """
        assert self.init_env_data, "should have been defined by extract_segment()"
        orig_fn = self.output_filename
        orig_backup_fn = self._backup_filename
        try:
            self.output_filename = Path(gs_fn)
            self._backup_filename = self.get_backup_filename(self.output_filename)

            resume_data = gs_data.setdefault('resume_data', {})
            init_env_data = resume_data.setdefault('init_env_data', {})
            init_env_data.update(self.init_env_data)
            if resume_data.get('converged_environments', False):
                raise ValueError(f"{gs_fn!s} already has converged environments!")
            resume_data['converged_environments'] = True
            resume_data['psi'] = gs_data['psi'] # could have been modified with canonical_form;
            # in any case that's the reference ground state we use now!

            self.logger.info("write converged environments back to ground state file")
            self.save_results(gs_data)  # safely overwrite old file
        finally:
            self.output_filename = orig_fn
            self._backup_filename = orig_backup_fn

    def run_algorithm(self):
        N_excitations = self.options.get("N_excitations", 1, int)
        switch_charge_sector = self.options.get("switch_charge_sector", None)
        momentum = self.options["momentum"]
        self.results['qtotal_diff'] = switch_charge_sector
        self.results['momentum'] = momentum
        if momentum is not None:
            momentum *= 2*np.pi/self.psi.L # Momentum is in units of 2pi/L, as this is
            # allowed momenta for plane wave ansatz.

        self.orthogonal_Xs = []
        # loop over excitations
        while len(self.excitations) < N_excitations:

            E, psi, N = self.engine.run(momentum, switch_charge_sector, self.orthogonal_Xs)
            self.results['excitation_energies'].append(E)
            self.logger.info("Excitation Energy: %.14f. Lanczos Iterations: %d", E, N)

            self.orthogonal_Xs.append(psi._X)
            self.excitations.append(psi)  # save in list of excitations
            if len(self.excitations) >= N_excitations:
                break

            self.make_measurements()
            self.logger.info("got %d excitations so far, proceed to next excitation.\n%s",
                             len(self.excitations), "+" * 80)
            self.init_state()  # initialize a new state to be optimized
            self.init_algorithm()  # initialize new environments for the state!
        # done

    def resume_run_algorithm(self):
        """Not Implemented"""
        raise NotImplementedError("TODO")

    def prepare_results_for_save(self):
        results = super().prepare_results_for_save()
        if 'resume_data' in results:
            results['resume_data']['excitations'] = self.excitations
        return results


class OrthogonalExcitations(GroundStateSearch):
    """Find excitations by another GroundStateSearch orthogonalizing against previous states.

    If the ground state is an infinite MPS, it is converted to `segment` boundary conditions
    at the beginning of this simulation.

    For finite systems, the first algorithm (say DMRG) run when switching the charge sector
    can be replaced by a normal DMRG run with a different initial state (in the desired sector).
    For infinite systems, the conversion to segment boundary conditions leads to a *different*
    state! Using the 'segment' boundary conditions, this class can e.g. study a single spin flip
    excitation in the background of the ground state, localized by the segment environments.

    Note that the segment environments are *soft* boundaries: the spin flip can be outside the
    segment where we vary the MPS tensors, as far as it contained in the Schmidt states of the
    original ground state.

    Parameters
    ----------
    orthogonal_to : None list
        States to orthogonalize against.

    Options
    -------
    .. cfg:config :: OrthogonalExcitations
        :include: GroundStateSearch

        N_excitations : int
            Number of excitations to find.
            Don't make this too big, it's gonna perform that many algorithm runs!

    Attributes
    ----------
    orthogonal_to : list
        States to orthogonalize against.
    excitations : list
        Tensor network states representing the excitations.
        The ground state in `orthogonal_to` is not included in the `excitations`.
        While being optimized, a state is saved as :attr:`psi` and not yet included in
        `excitations`.
    init_env_data : dict
        Initialization data for the :class:`~tenpy.networks.mpo.MPOEnvironment`.
        Passed to the algorithm class.
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
        if orthogonal_to is None and 'orthogonal_to' in resume_data:
            orthogonal_to = kwargs['resume_data']['orthogonal_to']
            self.options.touch('groundstate_filename')
        self.orthogonal_to = orthogonal_to
        self.excitations = resume_data.get('excitations', [])
        self.results['excitation_energies'] = []
        if self.options.get('save_psi', True, bool):
            self.results['excitations'] = self.excitations
        self.init_env_data = {}

    def run(self):
        if self.orthogonal_to is None:
            self.init_orthogonal_from_groundstate()
        return super().run()

    def resume_run(self):
        if self.orthogonal_to is None:
            self.init_orthogonal_from_groundstate()
        return super().resume_run()

    def init_orthogonal_from_groundstate(self):
        """Initialize :attr:`orthogonal_to` from the ground state.

        Load the ground state.
        If the ground state is infinite, call :meth:`extract_segment_from_infinite`.

        An empty :attr:`orthogonal_to` indicates that we will :meth:`switch_charge_sector`
        in the first :meth:`init_algorithm` call.

        Options
        -------
        .. cfg:configoptions :: OrthogonalExcitations

            ground_state_filename :
                File from which the ground state should be loaded.
            orthogonal_norm_tol : float
                Tolerance how large :meth:`~tenpy.networks.mps.MPS.norm_err` may be for states
                to be added to :attr:`orthogonal_to`.
            segment_enlarge, segment_first, segment_last : int | None
                Only for initially infinite ground states.
                Arguments for :meth:`~tenpy.models.lattice.Lattice.extract_segment`.
            apply_local_op: dict | None
                If not `None`, apply :meth:`~tenpy.networks.mps.MPS.apply_local_op` with given
                keyword arguments to change the charge sector compared to the ground state.
                Alternatively, use `switch_charge_sector`.
            switch_charge_sector : list of int | None
                If given, change the charge sector of the excitations compared to the ground state.
                Alternative to `apply_local_op` where we run a small zero-site diagonalization on
                the (left-most/center for infinite/finite) bond
                in the desired charge sector to update the state.
            write_back_converged_ground_state_environments : bool
                Only used for infinite ground states, indicating that we should write converged
                environments of the ground state back to `ground_state_filename`.
                This is an optimization if you intend to run another `OrthogonalExcitations`
                simulation in the future with the same `ground_state_filename`.
                (However, it is not faster when the simulations run at the same time; instead it
                might even lead to errors!)

        Returns
        -------
        data : dict
            The data loaded from :cfg:option:`OrthogonalExcitations.ground_state_filename`.
        """
        # TODO: allow to pass ground state data as kwargs to sim instead!
        gs_fn = self.options['ground_state_filename']
        data = hdf5_io.load(gs_fn)
        data_options = data['simulation_parameters']
        # get model from ground_state data
        for key in data_options.keys():
            if not isinstance(key, str) or not key.startswith('model'):
                continue
            if key not in self.options:
                self.options[key] = data_options[key]
        self.init_model()

        self.ground_state = psi0 = data['psi']
        resume_data = data.get('resume_data', {})
        if np.linalg.norm(psi0.norm_test()) > self.options.get('orthogonal_norm_tol', 1.e-12, 'real'):
            self.logger.info("call psi.canonical_form() on ground state")
            psi0.canonical_form()
        if psi0.bc == 'infinite':
            write_back = self.extract_segment_from_infinite(psi0, self.model, resume_data)
            if write_back:
                self.write_converged_environments(data, gs_fn)
        else:
            self.init_env_data = resume_data.get('init_env_data', {})
            self.ground_state_infinite = None
            self.results['ground_state_energy'] = data['energy']

        apply_local_op = self.options.get("apply_local_op", None)
        switch_charge_sector = self.options.get("switch_charge_sector", None)
        if apply_local_op is None and switch_charge_sector is None:
            self.orthogonal_to = [self.ground_state]
            self.results['ground_state_energy'] = data['energy']
        else:
            # we will switch charge sector
            self.orthogonal_to = []  # so we don't need to orthogonalize against original g.s.
            # optimization: delay calculation of the reference ground_state_energy
            # until self.switch_charge_sector() is called by self.init_algorithm()
        return data

    def extract_segment_from_infinite(self, psi0_inf, model_inf, resume_data):
        """Extract a finite segment from the infinite model/state.

        Parameters
        ----------
        psi0_inf : :class:`~tenpy.networks.mps.MPS`
            Original ground state with infinite boundary conditions.
        model_inf : :class:`~tenpy.models.model.MPOModel`
            Original infinite model.
        resume_data : dict
            Possibly contains `init_env_data` with environments.

        Returns
        -------
        write_back : bool
            Whether we should call :meth:`write_converged_environments`.
        """
        enlarge = self.options.get('segment_enlarge', None, int)
        first = self.options.get('segment_first', 0, int)
        last = self.options.get('segment_last', None, int)
        self.model = model_inf.extract_segment(first, last, enlarge)
        first, last = self.model.lat.segment_first_last
        write_back = self.options.get('write_back_converged_ground_state_environments', False, bool)
        if resume_data.get('converged_environments', False):
            self.logger.info("use converged environments from ground state file")
            env_data = resume_data['init_env_data']
            psi0_inf = resume_data.get('psi', psi0_inf)
            write_back = False
        else:
            self.logger.info("converge environments with MPOTransferMatrix")
            guess_init_env_data = resume_data.get('init_env_data', None)
            H = model_inf.H_MPO
            env_data = MPOTransferMatrix.find_init_LP_RP(H, psi0_inf, first, last,
                                                         guess_init_env_data)
        self.init_env_data = env_data
        self.ground_state_infinite = psi0_inf
        self.ground_state = psi0_inf.extract_segment(first, last)
        return write_back

    def write_converged_environments(self, gs_data, gs_fn):
        """Write converged environments back into the file with the ground state.

        Parameters
        ----------
        gs_data : dict
            Data loaded from the ground state file.
        gs_fn : str
            Filename where to save `gs_data`.
        """
        if not self.init_env_data:
            raise ValueError("Didn't converge new environments!")
        orig_fn = self.output_filename
        orig_backup_fn = self._backup_filename
        try:
            self.output_filename = Path(gs_fn)
            self._backup_filename = self.get_backup_filename(self.output_filename)

            resume_data = gs_data.setdefault('resume_data', {})
            init_env_data = resume_data.setdefault('init_env_data', {})
            init_env_data.update(self.init_env_data)
            if resume_data.get('converged_environments', False):
                raise ValueError(f"{gs_fn!s} already has converged environments!")
            resume_data['converged_environments'] = True
            resume_data['psi'] = gs_data['psi']

            self.logger.info("write converged environments back to ground state file")
            self.save_results(gs_data)  # safely overwrite old file
        finally:
            self.output_filename = orig_fn
            self._backup_filename = orig_backup_fn

    def init_state(self):
        """Initialize the state.

        Options
        -------
        .. cfg:configoptions :: OrthogonalExcitations

            initial_state_params : dict
                The initial state parameters, :cfg:config:`ExcitationInitialState` defined below.
        """
        if len(self.orthogonal_to) == 0 and not self.loaded_from_checkpoint:
            self.psi = self.ground_state  # will switch charge sector in init_algorithm()
            if self.options.get('save_psi', True, bool):
                self.results['psi'] = self.psi
            return
        builder_class = self.options.get('initial_state_builder_class', 'ExcitationInitialState')
        params = self.options.subconfig('initial_state_params')
        Builder = find_subclass(InitialStateBuilder, builder_class)
        if issubclass(Builder, ExcitationInitialState):
            # incompatible with InitialStateBuilder: pass `sim` to __init__
            initial_state_builder = Builder(self, params)
        else:
            initial_state_builder = Builder(self.model.lat, params, self.model.dtype)
        self.psi = initial_state_builder.run()

        if self.options.get('save_psi', True, bool):
            self.results['psi'] = self.psi

    def init_algorithm(self, **kwargs):
        kwargs.setdefault('orthogonal_to', self.orthogonal_to)
        resume_data = kwargs.setdefault('resume_data', {})
        resume_data['init_env_data'] = self.init_env_data
        super().init_algorithm(**kwargs)

        if len(self.orthogonal_to) == 0:
            self.switch_charge_sector()

    def switch_charge_sector(self):
        """Change the charge sector of :attr:`psi` in place."""
        if self.psi.chinfo.qnumber == 0:
            raise ValueError("can't switch charge sector with trivial charges!")
        self.logger.info("switch charge sector of the ground state "
                         "[contracts environments from right]")
        apply_local_op = self.options.get("apply_local_op", None)
        switch_charge_sector = self.options.get("switch_charge_sector", None)
        qtotal_before = self.psi.get_total_charge()
        env = self.engine.env
        if apply_local_op is not None:
            if switch_charge_sector is not None:
                raise ValueError("give only one of `switch_charge_sector` and `apply_local_op`")
            self.results['ground_state_energy'] = env.full_contraction(0)
            for i in range(0, apply_local_op['i'] - 1):
                env.del_RP(i)
            for i in range(apply_local_op['i'] + 1, env.L):
                env.del_LP(i)
            apply_local_op['unitary'] = True  # no need to call psi.canonical_form
            self.psi.apply_local_op(**apply_local_op)
        else:
            i = self.psi.L // 2 if self.psi.finite else 0
            assert switch_charge_sector is not None
            # get the correct environments on site 0
            LP = env.get_LP(i)
            if i == 0:
                RP = env._contract_RP(0, env.get_RP(0, store=True))  # saves the environments!
            else:
                RP = env.get_RP(i - 1)
            self.results['ground_state_energy'] = env.full_contraction(i)
            for j in range(i + 1, self.engine.n_optimize):
                env.del_LP(j)  # but we might have gotten more than we need
            H0 = ZeroSiteH.from_LP_RP(LP, RP)
            if self.model.H_MPO.explicit_plus_hc:
                H0 = SumNpcLinearOperator(H0, H0.adjoint())
            vL, vR = LP.get_leg('vR').conj(), RP.get_leg('vL').conj()
            th0 = npc.Array.from_func(np.ones, [vL, vR],
                                      dtype=self.psi.dtype,
                                      qtotal=switch_charge_sector,
                                      labels=['vL', 'vR'])
            if th0.norm() == 0:
                raise ValueError(f"Can't switch to desired charge sector at bond lef of site {i:d}"
                                 f" with vL leg {vL:d} and vR {vR!r}")
            lanczos_params = self.engine.lanczos_params
            _, th0, _ = krylov_based.LanczosGroundState(H0, th0, lanczos_params).run()
            th0 = npc.tensordot(th0, self.psi.get_B(i, 'B'), axes=['vR', 'vL'])
            self.psi.set_B(i, th0, form='Th')
            if self.psi.finite:
                self.psi.canonical_form()
                env.clear()
        qtotal_after = self.psi.get_total_charge()
        qtotal_diff = self.psi.chinfo.make_valid(qtotal_after - qtotal_before)
        self.logger.info("changed charge by %r compared to previous state", list(qtotal_diff))
        assert not np.all(qtotal_diff == 0)

    def run_algorithm(self):
        N_excitations = self.options.get("N_excitations", 1, int)
        ground_state_energy = self.results['ground_state_energy']
        self.logger.info("reference ground state energy: %.14f", ground_state_energy)
        if ground_state_energy > - 1.e-7:
            # the orthogonal projection does not lead to a different ground state!
            lanczos_params = self.engine.lanczos_params
            if self.engine.diag_method != 'lanczos' or \
                    ground_state_energy + 0.5 * lanczos_params.get('E_shift', 0., 'real') > 0:
                # the factor of 0.5 is somewhat arbitrary, to ensure that
                # also excitations have energy < 0
                raise ValueError("You need to set use diag_method='lanczos' and small enough "
                                 f"lanczos_params['E_shift'] < {-2.* ground_state_energy:.2f}")

        while len(self.excitations) < N_excitations:

            E, psi = self.engine.run()

            self.results['excitation_energies'].append(E - ground_state_energy)
            self.logger.info("excitation energy: %.14f", E - ground_state_energy)
            tol = self.options.get('orthogonal_norm_tol', 1.e-12, 'real')
            if np.linalg.norm(psi.norm_test()) > tol:
                self.logger.info("call psi.canonical_form() on excitation")
                psi.canonical_form()
            self.excitations.append(psi)
            self.orthogonal_to.append(psi)
            # save in list of excitations
            if len(self.excitations) >= N_excitations:
                break

            self.make_measurements()
            self.logger.info("got %d excitations so far, proceed to next excitation.\n%s",
                             len(self.excitations), "+" * 80)
            self.init_state()  # initialize a new state to be optimized
            self.init_algorithm()
        # done

    def resume_run_algorithm(self):
        raise NotImplementedError("TODO")

    def prepare_results_for_save(self):
        results = super().prepare_results_for_save()
        if 'resume_data' in results:
            results['resume_data']['excitations'] = self.excitations
        return results


class TopologicalExcitations(OrthogonalExcitations):
    def __init__(self, options, *, gs_data_alpha=None, gs_data_beta=None, **kwargs):
        super().__init__(options, **kwargs)
        resume_data = kwargs.get('resume_data', {})
        #  if orthogonal_to is None and 'orthogonal_to' in resume_data:
        #      orthogonal_to = kwargs['resume_data']['orthogonal_to']
        #      self.options.touch('ground_state_filename_left', 'ground_state_filename_right')
        self.orthogonal_to = None # TODO: allow orthogonal_to
        self.excitations = resume_data.get('excitations', [])
        self.results['excitation_energies'] = []
        self.results['excitation_energies_MPO'] = []
        if self.options.get('save_psi', True, bool):
            self.results['excitations'] = self.excitations
        self.init_env_data = {}
        self._gs_data_alpha = gs_data_alpha
        self._gs_data_beta = gs_data_beta
        self.initial_state_builder = None

    def init_from_groundstate(self):
        """Initialize :attr:`orthogonal_to` from the ground state.

        Load the ground state and initialize the model from it.
        Calls :meth:`extract_segment`.

        An empty :attr:`orthogonal_to` indicates that we will :meth:`switch_charge_sector`
        in the first :meth:`init_algorithm` call.

        Options
        -------
        .. cfg:configoptions :: OrthogonalExcitations

            ground_state_filename :
                File from which the ground state should be loaded.
            orthogonal_norm_tol : float
                Tolerance how large :meth:`~tenpy.networks.mps.MPS.norm_err` may be for states
                to be added to :attr:`orthogonal_to`.
            apply_local_op: list | None
                If not `None`, use :meth:`~tenpy.networks.mps.MPS.apply_local_op` to change
                the charge sector compared to the ground state.
                Should have the form  ``[site1, operator1, site2, operator2, ...]``.
                with the operators given as strings (to be read out from the site class).
                Alternatively, use `switch_charge_sector`.
                `site#` are MPS indices in the *original* ground state, not the segment!
            switch_charge_sector : list of int | None
                If given, change the charge sector of the excitations compared to the ground state.
                Alternative to `apply_local_op` where we run a small zero-site diagonalization on
                the left-most bond in the desired charge sector to update the state.
            switch_charge_sector_site: int
                To the left of which site we switch charge sector.
                MPS index in the *original* ground state, not the segment!

        Returns
        -------
        gs_data : dict
            The data loaded from :cfg:option:`OrthogonalExcitations.ground_state_filename`.
        """
        gs_fn_alpha, gs_data_alpha, gs_fn_beta, gs_data_beta = self._load_gs_data()
        gs_data_options_alpha = gs_data_alpha['simulation_parameters']
        # initialize original model with model_class and model_params from ground state data
        self.logger.info("initialize original ground state model")
        for key in gs_data_options_alpha.keys(): # Assume same model params for left and right
            if not isinstance(key, str) or not key.startswith('model'):
                continue
            if key not in self.options:
                self.options[key] = gs_data_options_alpha[key]
        self.init_model() # FOR NOW, WE ASSUME LEFT AND RIGHT MODELS ARE THE SAME
        self.model_orig = self.model

        # initialize original state
        self.ground_state_orig_alpha = psi0_alpha = gs_data_alpha['psi']  # no copy!
        self.ground_state_orig_beta = psi0_beta = gs_data_beta['psi']  # no copy!
        assert self.ground_state_orig_alpha.L == self.ground_state_orig_beta.L
        tol = self.options.get('orthogonal_norm_tol', 1.e-12, 'real')
        if np.linalg.norm(psi0_alpha.norm_test()) > tol:
            self.logger.info("call psi.canonical_form() on left ground state")
            psi0_alpha.canonical_form()
        if np.linalg.norm(psi0_beta.norm_test()) > tol:
            self.logger.info("call psi.canonical_form() on right ground state")
            psi0_beta.canonical_form()

        # extract segments if necessary; get `init_env_data`.
        resume_data_alpha = gs_data_alpha.get('resume_data', {}) # TODO this is probably wrong
        resume_data_beta = gs_data_beta.get('resume_data', {}) # TODO this is probably wrong
        #psi0_seg, write_back_left, write_back_right = self.extract_segment(psi0_alpha, psi0_beta, self.model, resume_data_alpha, resume_data_beta)
        self.initial_state_seg, self.qtotal_diff, write_back_left, write_back_right = self.extract_segment(psi0_alpha, psi0_beta, self.model, resume_data_alpha, resume_data_beta)
        ########################################
        if write_back_left:
            init_env_data = self.init_env_data
            self.init_env_data = self.init_env_data_alpha
            self.write_back_environments(gs_data_alpha, gs_fn_alpha)
            self.init_env_data = init_env_data
        if write_back_right:
            init_env_data = self.init_env_data
            self.init_env_data = self.init_env_data_beta
            self.write_back_environments(gs_data_beta, gs_fn_beta)
            self.init_env_data = init_env_data
        self.results['segment_first_last'] = self.model.lat.segment_first_last

        # here, psi0_seg is the *unperturbed* ground state in the segment!
        self.get_reference_energy(psi0_alpha, psi0_beta)

        # switch_charge_sector defines `self.initial_state_seg`
        #self.initial_state_seg, self.qtotal_diff = self.switch_charge_sector(psi0_seg)
        self.results['qtotal_diff'] = self.qtotal_diff

        self.orthogonal_to = []  # Segment is inherently different than either left or right ground state.
        # Or at least the two sides will be different for non-trivial calculation.
        return None # return isn't used

    def _load_gs_data(self):
        """Load ground state data from `ground_state_filename` or use simulation kwargs."""
        gs_data_return = []
        for which, gs_D in zip(['left', 'right'], [self._gs_data_alpha, self._gs_data_beta]):
            if gs_D is not None:
                gs_F = None
                self.logger.info("use ground state data of simulation class arguments")
                gs_data = gs_D
                gs_D = None  # reset to None to potentially allow to free the memory
                # even though this can only work if the call structure is
                #      sim = OrthogonalExcitations(..., gs_data=gs_data)
                #      del gs_data
                #      with sim:
                #          sim.run()
            else:
                gs_F = self.options['ground_state_filename_' + which]
                self.logger.info("loading " + which + " ground state data from %s", gs_F)
                gs_D = hdf5_io.load(gs_F)
            gs_data_return.extend((gs_F, gs_D))
        assert len(gs_data_return) == 4
        return gs_data_return

    def extract_segment(self, psi0_alpha_Orig, psi0_beta_Orig, model_orig, resume_data_alpha, resume_data_beta):
        """Extract a finite segment from the original model and states.

        In case the original state is already finite, we might still extract a sub-segment
        (if `segment_first` and/or `segment_last` are given) or just use the full system.

        Defines :attr:`ground_state_seg` to be the ground state of the segment.
        Further :attr:`model` and :attr:`init_env_data` are extracted.

        Options
        -------
        .. cfg:configoptions :: OrthogonalExcitations

            segment_enlarge, segment_first, segment_last : int | None
                Arguments for :meth:`~tenpy.models.lattice.Lattice.extract_segment`.
                `segment_enlarge` is only used for initially infinite ground states.
            write_back_converged_ground_state_environments : bool
                Only used for infinite ground states, indicating that we should write converged
                environments of the ground state back to `ground_state_filename`.
                This is an optimization if you intend to run another `OrthogonalExcitations`
                simulation in the future with the same `ground_state_filename`.
                (However, it is not faster when the simulations run at the same time; instead it
                might even lead to errors!)

        Parameters
        ----------
        psi0_orig : :class:`~tenpy.networks.mps.MPS`
            Original ground state.
        model_orig : :class:`~tenpy.models.model.MPOModel`
            Original model.
        resume_data : dict
            Possibly contains `init_env_data` with environments.

        Returns
        -------
        psi0_seg :
            Unperturbed ground state in the segment, against which to orthogonalize
            if we don't switch charge sector.
        write_back : bool
            Whether :meth:`write_back_environments` should be called.
        """
        if psi0_alpha_Orig.bc == 'infinite':
            return self._extract_segment_from_infinite(psi0_alpha_Orig, psi0_beta_Orig, model_orig, resume_data_alpha, resume_data_beta)
        else:
            return self._extract_segment_from_finite(psi0_alpha_Orig, psi0_beta_Orig, model_orig)

    def _extract_segment_from_finite(self, psi0_fin_alpha, psi0_fin_beta, model_fin):
        """ Extract segment from finite MPS. """
        first = self.options.get('segment_first', 0, int)
        last = self.options.get('segment_last', None, int)
        # boundary should be defined in terms of the ORIGINAL MPS and NOT first.
        boundary = self.options.get(
            'segment_boundary',
            (last-first)//2 + first if last is not None else (psi0_fin_alpha.L-first)//2 + first,
            int
        )
        assert first < boundary
        if last is not None:
            assert boundary < last

        self.model = model_fin.extract_segment(first, last) # subset of original model
        first, last = self.model.lat.segment_first_last

        env = MPOEnvironment(psi0_fin_alpha, self.model_orig.H_MPO, psi0_fin_alpha)
        self.env_data_alpha = env.get_initialization_data(first, last) # Found by contracting gauge fixed (rho, trace, dmrg)
        # eigenvectors from unit cell boundary to the desired position).
        self.env_data_alpha_seg = env.get_initialization_data(first, boundary-1) # 2nd index included in segment

        env = MPOEnvironment(psi0_fin_beta, self.model_orig.H_MPO, psi0_fin_beta)
        self.env_data_beta = env.get_initialization_data(first, last)
        self.env_data_beta_seg = env.get_initialization_data(boundary, last)

        ground_state_seg_alpha = psi0_fin_alpha.extract_segment(first, boundary-1) # 2nd index included in segment
        ground_state_seg_beta = psi0_fin_beta.extract_segment(boundary, last)
        ground_state_seg, qtotal_diff = self._glue_segments(ground_state_seg_alpha, ground_state_seg_beta,
                                               psi0_fin_alpha, psi0_fin_beta,
                                               self.model, (first, last, boundary))

        self.init_env_data = {'init_LP': self.env_data_alpha['init_LP'],
                                  'init_RP': self.env_data_beta['init_RP'],
                                  'age_LP': 0,
                                  'age_RP': 0}
        # This should work even if the segment is the entire finite MPS.
        """
        if first != 0 or last is not None:
            self.init_env_data = {'init_LP': self.env_data_L['init_LP'],
                                  'init_RP': self.env_data_R['init_RP'],
                                  'age_LP': 0,
                                  'age_RP': 0}

            #self.init_env_data = self._contract_segment_boundaries(self.init_env_data, *ground_state_seg.segment_boundaries)
        else:
            assert ground_state_seg_L.L + ground_state_seg_R.L == psi0_fin_L.L
            self.init_env_data = {}
        """
        return ground_state_seg, qtotal_diff, False, False

    def _extract_segment_from_infinite(self, psi0_inf_alpha, psi0_inf_beta, model_inf, resume_data_alpha, resume_data_beta):
        enlarge = self.options.get('segment_enlarge', None, int)
        first = self.options.get('segment_first', 0, int)
        if enlarge is not None:
            assert first == 0
        last = self.options.get('segment_last', None, int)

        assert (enlarge is None) ^ (last is None), "'enlarge' xor 'last' must be not None."
        boundary = self.options.get(
            'segment_boundary',
            (last - first) // 2 + first if enlarge is None else (enlarge//2)*psi0_inf_alpha.L + first,
            int
        ) # boundary should be measured from the SAME site as 'first' and not w.r.t. 'first'.
        assert first < boundary
        if last is not None:
            assert boundary < last
        else:
            assert boundary < enlarge * psi0_inf_alpha.L
        write_back = self.options.get('write_back_converged_ground_state_environments', False, bool)

        self.model = model_inf.extract_segment(first, last, enlarge)
        first, last = self.model.lat.segment_first_last
        H = model_inf.H_MPO

        gauge = self.options.get('gauge', 'rho', str)
        if resume_data_alpha.get('converged_environments', False):
            self.logger.info("use converged environments from left ground state file")
            self.init_env_data_alpha = resume_data_alpha['init_env_data'] # Environments for infinite ground states
            psi0_inf_alpha = resume_data_alpha.get('psi', psi0_inf_alpha)
            write_back_left = False
        else:
            self.logger.info("converge left ground state environments with MPOTransferMatrix")
            guess_init_env_data = resume_data_alpha.get('init_env_data', None)
            self.init_env_data_alpha = MPOTransferMatrix.find_init_LP_RP(H, psi0_inf_alpha, guess_init_env_data=guess_init_env_data, _subtraction_gauge=gauge)
            # On bond 0 of the unit cell

            write_back_left = write_back

        if resume_data_beta.get('converged_environments', False):
            self.logger.info("use converged environments from right ground state file")
            self.init_env_data_beta = resume_data_beta['init_env_data']
            psi0_inf_beta = resume_data_beta.get('psi', psi0_inf_beta)
            write_back_right = False
        else:
            self.logger.info("converge right ground state environments with MPOTransferMatrix")
            guess_init_env_data = resume_data_beta.get('init_env_data', None)
            self.init_env_data_beta = MPOTransferMatrix.find_init_LP_RP(H, psi0_inf_beta, guess_init_env_data=guess_init_env_data, _subtraction_gauge=gauge)
            # On bond 0 of the unit cell

            write_back_right = write_back
        self.logger.info("converge segment environments with MPOTransferMatrix")

        env = MPOEnvironment(psi0_inf_alpha, H, psi0_inf_alpha, **self.init_env_data_alpha)
        self.env_data_alpha = env.get_initialization_data(first, last)
        self.env_data_alpha_seg = env.get_initialization_data(first, boundary-1)

        env = MPOEnvironment(psi0_inf_beta, H, psi0_inf_beta, **self.init_env_data_beta)
        self.env_data_beta = env.get_initialization_data(first, last)
        self.env_data_beta_seg = env.get_initialization_data(boundary, last)

        self.init_env_data = {'init_LP': self.env_data_alpha['init_LP'],
                              'init_RP': self.env_data_beta['init_RP'],
                              'age_LP': 0,
                              'age_RP': 0}

        ground_state_seg_alpha = psi0_inf_alpha.extract_segment(first, boundary-1)
        ground_state_seg_beta = psi0_inf_beta.extract_segment(boundary, last)
        ground_state_seg, qtotal_diff = self._glue_segments(ground_state_seg_alpha, ground_state_seg_beta,
                                               psi0_inf_alpha, psi0_inf_beta,
                                               self.model, (first, last, boundary))

        return ground_state_seg, qtotal_diff, write_back_left, write_back_right


    def _glue_segments(self, seg_alpha, seg_beta, inf_alpha, inf_beta, model, indices):
        join_method = self.join_method = self.options.get('join_method', "average charge", str)
        switch_charge_sector = self.options.get("switch_charge_sector", None)
        if inf_alpha.finite or inf_beta.finite:
            assert join_method == "most probable charge"
        first, last, boundary = indices
        self.logger.info("First: %d, Last: %d, Boundary: %d", first, last, boundary)
        left_half_model = self.model_orig.extract_segment(first, boundary-1, None)
        right_half_model = self.model_orig.extract_segment(boundary, last, None)

        env_alpha_BC = MPOEnvironment(seg_alpha, left_half_model.H_MPO, seg_alpha, **self.env_data_alpha_seg)
        env_beta_BC = MPOEnvironment(seg_beta, right_half_model.H_MPO, seg_beta, **self.env_data_beta_seg)
        LP = env_alpha_BC._contract_LP(seg_alpha.L-1, env_alpha_BC.get_LP(seg_alpha.L-1, store=False))
        RP = env_beta_BC._contract_RP(0, env_beta_BC.get_RP(0, store=False))  # saves the environments!
        H0 = ZeroSiteH.from_LP_RP(LP, RP)
        if self.model.H_MPO.explicit_plus_hc:
            H0 = SumNpcLinearOperator(H0, H0.adjoint())
        vL, vR = LP.get_leg('vR').conj(), RP.get_leg('vL').conj()

        if seg_alpha.chinfo.qnumber == 0:    # Handles the case of no charge-conservation
            Q_offset = None
        else:
            Qs_alpha, ps_alpha = seg_alpha.probability_per_charge(seg_alpha.L)
            Qs_beta, ps_beta = seg_beta.probability_per_charge(0)

            side_by_side = string.vert_join(["left seg\n" + str(Qs_alpha), "prob\n" + str(np.array([ps_alpha]).T), "right seg\n" + str(Qs_beta),"prob\n" +str(np.array([ps_beta]).T)], delim=' | ')
            self.logger.info(side_by_side)
            # NOTE: chinfo.make_valid() turns charges into INT by discarding fractional part. Round first.
            if join_method == "average charge":
                Q_bar_alpha = inf_alpha.average_charge(0)
                for i in range(1, inf_alpha.L):
                    Q_bar_alpha += inf_alpha.average_charge(i)
                #Q_bar_alpha = vL.chinfo.make_valid(np.around(Q_bar_alpha / inf_alpha.L))
                Q_bar_alpha = Q_bar_alpha / inf_alpha.L
                self.logger.info("Charge of left BC, averaged over site and unit cell: %r", Q_bar_alpha)

                Q_bar_beta = inf_beta.average_charge(0)
                for i in range(1, inf_beta.L):
                    Q_bar_beta += inf_beta.average_charge(i)
                #Q_bar_beta = vR.chinfo.make_valid(np.around(Q_bar_beta / inf_beta.L)) # -1*
                Q_bar_beta = Q_bar_beta / inf_beta.L # -1*
                self.logger.info("Charge of right BC, averaged over site and unit cell: %r", Q_bar_beta)
                #Q_offset = (vL.chinfo.make_valid(Q_bar_alpha - Q_bar_beta))
                Q_offset = Q_bar_alpha - Q_bar_beta
            elif join_method == "most probable charge":
                Qmostprobable_alpha = Qs_alpha[np.argmax(ps_alpha)]
                Qmostprobable_beta = Qs_beta[np.argmax(ps_beta)] #-1*
                self.logger.info("Most probable left: %r", Qmostprobable_alpha)
                self.logger.info("Most probable right: %r", Qmostprobable_beta)
                #Q_offset = (vL.chinfo.make_valid(Qmostprobable_alpha - Qmostprobable_beta))
                Q_offset = Qmostprobable_alpha - Qmostprobable_beta
            else:
                raise ValueError("Invalid `join_method` %s " % join_method)

            self.logger.info("Q Offset (offset L - offset R): %r", Q_offset)
            #switch_charge_sector = vL.chinfo.make_valid(switch_charge_sector)
            self.logger.info("Desired excitation charge: %r", switch_charge_sector)
            self.gluing_charge = Q_offset
            if switch_charge_sector is not None:
                self.gluing_charge = switch_charge_sector + self.gluing_charge
            self.gluing_charge = vL.chinfo.make_valid(np.around(self.gluing_charge))
            self.logger.info("Gluing charge (round(Q_ex + Q_off)): %r", self.gluing_charge)
            qtotal_diff = self.gluing_charge - Q_offset
            self.logger.info("Targeted excitation charge (Q_gl - Q_off): %r", qtotal_diff)

        # We need a tensor that is non-zero only when Q = (Q^i_L - bar(Q_L)) + (Q^i_R - bar(Q_R))
        # Q is the the charge we insert. Here we only do charge gluing to get a valid segment.
        # Changing charge sector is done below by basically identical code when the segment is already formed.
        th0 = npc.Array.from_func(np.ones, [vL, vR],
                                  dtype=seg_alpha.dtype,
                                  qtotal=list(self.gluing_charge),
                                  labels=['vL', 'vR'])
        lanczos_params = self.options.get("lanczos_params", {}) # See if lanczos_params is in yaml, if not use empty dictionary
        _, th0, _ = krylov_based.LanczosGroundState(H0, th0, lanczos_params).run()

        norm = npc.norm(th0)
        self.logger.info("Norm of theta guess: %.8f", norm)
        if np.isclose(norm, 0):
            raise ValueError(f"Norm of inserted theta with charge {list(self.gluing_charge)} on site index {boundary:d} is zero.")

        U, s, Vh = npc.svd(th0, inner_labels=['vR', 'vL'])
        seg_alpha.set_B(seg_alpha.L-1, npc.tensordot(seg_alpha.get_B(seg_alpha.L-1, 'A'), U, axes=['vR', 'vL']), form='A') # Put AU into last site of left segment
        seg_alpha.set_SR(seg_alpha.L-1, s)
        seg_beta.set_B(0, npc.tensordot(Vh, seg_beta.get_B(0, 'B'), axes=['vR', 'vL']), form='B') # Put Vh B into first site of right segment
        seg_beta.set_SL(0, s)

        combined_seg = self._concatenate_segments(seg_alpha, seg_beta, inf_alpha)

        return combined_seg, qtotal_diff

    def _concatenate_segments(self, seg_alpha, seg_beta, inf_alpha):
        l_sites = [seg_alpha.sites[i] for i in range(seg_alpha.L)]
        lA = [seg_alpha.get_B(i, 'A') for i in range(seg_alpha.L)]
        lS = [seg_alpha.get_SL(i) for i in range(seg_alpha.L)]
        #lS.append(seg_L.get_SR(seg_L.L-1))

        # Building segment MPS on right half
        r_sites = [seg_beta.sites[i] for i in range(seg_beta.L)]
        rB = [seg_beta.get_B(i) for i in range(seg_beta.L)]
        rS = [seg_beta.get_SL(i) for i in range(seg_beta.L)]
        rS.append(seg_beta.get_SR(seg_beta.L-1))

        assert npc.norm(seg_alpha.get_SR(seg_alpha.L-1) - rS[0]) < 1.e-12

        cp = MPS(l_sites + r_sites, lA + rB, lS + rS, 'segment',
                 ['A'] * seg_alpha.L + ['B'] * seg_beta.L, inf_alpha.norm)
        cp.grouped = inf_alpha.grouped
        cp.canonical_form_finite(cutoff=1e-15) #to strip out vanishing singular values at the interface
        return cp

    def correction(self, psi0_alpha, psi0_beta, env_alpha, env_beta, last):
        # 'last' is last site in segment
        correction = 0
        if psi0_alpha.finite:
            sites = range(last, psi0_alpha.L-1)
        else:
            sites = reversed(range(0, psi0_alpha.L))

        for i in sites:
            RP_alpha = env_alpha.get_RP(i)
            RP_beta = env_beta.get_RP(i)
            S_alpha = psi0_alpha.get_SR(i)
            S_beta = psi0_beta.get_SR(i)
            wR = self.model_orig.H_MPO.get_W(i).get_leg('wR')
            IdL = self.model_orig.H_MPO.get_IdL(i+1)

            vR = psi0_alpha.get_B(i, 'B').get_leg('vR')
            if isinstance(S_alpha, npc.Array):
                rho_alpha = npc.tensordot(S_alpha, S_alpha.conj(), axes=['vL', 'vL*'])
            else:
                S2 = S_alpha**2
                rho_alpha = npc.diag(S2, vR, labels=['vR', 'vR*'])
            rho_alpha = rho_alpha.add_leg(wR, IdL, axis=1, label='wR')

            vR = psi0_beta.get_B(i, 'B').get_leg('vR')
            if isinstance(S_beta, npc.Array):
                rho_beta = npc.tensordot(S_beta, S_beta.conj(), axes=['vL', 'vL*'])
            else:
                S2 = S_beta**2
                rho_beta = npc.diag(S2, vR, labels=['vR', 'vR*'])
            rho_beta = rho_beta.add_leg(wR, IdL, axis=1, label='wR')

            correction += npc.tensordot(rho_beta, RP_beta, axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*'])) - \
                          npc.tensordot(rho_alpha, RP_alpha, axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']))
        return correction

    def arbitrary_shift_left(self, i, psi, LP):
        # TODO JH while merging vumps branch, I saw that this function is not used, instead
        # get_reference_energy() uses arbitrary_shift_right() twice (since git commit 35f6771b81)
        # I think it's correct and we can
        # remove this function, but I should first double-check this again
        dtype = np.promote_types(psi.dtype, self.model_orig.H_MPO.dtype)
        wL = self.model.H_MPO.get_W(i % self.model.H_MPO.L).get_leg('wL')
        IdR = self.model_orig.H_MPO.get_IdR((i-1) % self.model.H_MPO.L)
        vL = psi.get_B(i, 'A').get_leg('vL')
        chi0 = vL.ind_len
        proj_trace = npc.diag(1., vL.conj(), dtype=dtype, labels=['vL*', 'vL'])
        proj_trace = proj_trace.add_leg(wL, IdR, axis=1, label='wL') / chi0  # vL* wL vL
        eta_L = npc.tensordot(LP, proj_trace, axes=(['vR*', 'wR', 'vR'], ['vL*', 'wL', 'vL'])).real
        return eta_L

    def arbitrary_shift_right(self, i, psi, RP):
        dtype = np.promote_types(psi.dtype, self.model_orig.H_MPO.dtype)
        wR = self.model.H_MPO.get_W(i % self.model.H_MPO.L).get_leg('wR')
        IdL = self.model_orig.H_MPO.get_IdL((i+1) % self.model.H_MPO.L)
        vR = psi.get_B(i, 'B').get_leg('vR')
        chi0 = vR.ind_len
        proj_trace = npc.diag(1., vR, dtype=dtype, labels=['vR', 'vR*'])
        proj_trace = proj_trace.add_leg(wR, IdL, axis=1, label='wR') / chi0 # vR wR vR*
        eta_R = npc.tensordot(proj_trace, RP, axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*'])).real
        return eta_R

    def get_reference_energy(self, psi0_alpha, psi0_beta):
        """Obtain ground state reference energy.

        Excitation energies are full contractions of the MPOEnvironment with the environments
        defined in :attr:`init_env_data`.
        Hence, the reference energy is also the contraction of the `MPOEnvironment` on the segment.

        Parameters
        ----------
        psi0_alpha : :class:`~tenpy.networks.msp.MPS`
            Infinite ground state MPS on the left, matching :attr:`init_env_data_alpha`.
        """
        self.logger.info("Calculate reference energy by contracting environments")
        first, last = self.results['segment_first_last']
        seg_alpha = psi0_alpha.extract_segment(first, last)
        gauge = self.options.get('gauge', 'rho', str)

        # This is expensive but more accurate than E0 + epsilon*L
        env_alpha = MPOEnvironment(seg_alpha, self.model.H_MPO, seg_alpha, **self.env_data_alpha)
        E_alpha = env_alpha.full_contraction(0).real

        if psi0_alpha.finite:
            env_alpha = MPOEnvironment(psi0_alpha, self.model_orig.H_MPO, psi0_alpha)
            env_beta = MPOEnvironment(psi0_beta, self.model_orig.H_MPO, psi0_beta)
        else:
            env_alpha = MPOEnvironment(psi0_alpha, self.model_orig.H_MPO, psi0_alpha, **self.init_env_data_alpha)
            env_beta = MPOEnvironment(psi0_beta, self.model_orig.H_MPO, psi0_beta, **self.init_env_data_beta)

        if psi0_alpha.finite:
            correction = self.correction(psi0_alpha, psi0_beta, env_alpha, env_beta, last)

            self.results['ground_state_energy'] = E_alpha + correction
        else:
            H = self.model_orig.H_MPO
            if (last + 1 - first) % psi0_alpha.L == 0: # last is included in segment.
                _, epsilon_alpha, E0_alpha = MPOTransferMatrix.find_init_LP_RP(H, psi0_alpha, first, last,
                                guess_init_env_data=self.init_env_data_alpha, calc_E=True, _subtraction_gauge=gauge)
                epsilon_alpha = np.mean(epsilon_alpha).real
            else:
                epsilon_alpha, E0_alpha, epsilon_beta, E0_beta = 0, 0, 0, 0

            E_alpha2 = E0_alpha + seg_alpha.L * epsilon_alpha

            self.logger.info("E_alpha, E_alpha2: %.14f, %.14f", E_alpha, E_alpha2)
            self.logger.info("epsilon_alpha, E0_alpha: %.14f, %.14f", epsilon_alpha, E0_alpha)

            eta_R_alpha = self.arbitrary_shift_right(psi0_alpha.L-1, psi0_alpha, self.init_env_data_alpha['init_RP']).real
            eta_R_beta = self.arbitrary_shift_right(psi0_alpha.L-1, psi0_beta, self.init_env_data_beta['init_RP']).real
            self.logger.info("eta_R_alpha, eta_R_beta: %.14f, %.14f", eta_R_alpha, eta_R_beta)

            correction = self.correction(psi0_alpha, psi0_beta, env_alpha, env_beta, last).real / psi0_alpha.L - eta_R_beta + eta_R_alpha
            self.logger.info("Correction term for mismatched GSs: %.14f", correction)
            self.results['ground_state_energy'] = E_alpha - eta_R_alpha + eta_R_beta + correction

        self.logger.info("Correction term for mismatched GSs: %.14f", correction)
        self.logger.info("Reference Ground State Energy: %.14f", self.results['ground_state_energy'])

        return self.results['ground_state_energy']


class ExcitationInitialState(InitialStateBuilder):
    """InitialStateBuilder for :class:`OrthogonalExcitations`.

    Parameters
    ----------
    sim : :class:`OrthogonalExcitations`
        Simulation class for which an initial state needs to be defined.
    options : dict
        Parameter dictionary as described below.

    Options
    -------
    .. cfg:config :: ExcitationInitialState
        :include: InitialStateBuilder

        randomize_params : dict-like
            Parameters for the random unitary evolution used to perturb the state a little bit
            in :meth:`~tenpy.networks.mps.MPS.perturb`.
        randomize_close_1 : bool
            Whether to randomize/perturb with unitaries close to the identity.
        use_highest_excitation : bool
            If True, start from  the last state in :attr:`orthogonal_to` and perturb it.
            If False, use the ground state (=the first entry of :attr:`orthogonal_to` and
            perturb that one a little bit.

    Attributes
    ----------
    sim : :class:`OrthogonalExcitations`
        Simulation class for which to initial a state to be used as excitation initial state.
    """
    def __init__(self, sim, options):
        self.sim = sim
        self.options = asConfig(options, self.__class__.__name__)
        self.options.setdefault('method', 'from_orthogonal')
        super().__init__(sim.model.lat, options, sim.model.dtype)

    def from_orthogonal(self):
        if self.options.get('use_highest_excitation', True, bool):
            psi = self.sim.orthogonal_to[-1]
        else:
            psi = self.sim.ground_state
        if isinstance(psi, dict):
            psi = psi['ket']
        psi = psi.copy()  # make a copy!
        return self._perturb(psi)

    def _perturb(self, psi):
        randomize_params = self.options.subconfig('randomize_params')
        close_1 = self.options.get('randomize_close_1', True, bool)
        psi.perturb(randomize_params, close_1=close_1, canonicalize=False)
        return psi
