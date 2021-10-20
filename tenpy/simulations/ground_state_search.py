"""Simulations for ground state searches."""
# Copyright 2020-2021 TeNPy Developers, GNU GPLv3

import numpy as np
from pathlib import Path

from . import simulation
from ..tools import hdf5_io
from .simulation import *
from ..linalg import np_conserved as npc
from ..networks.mpo import MPOEnvironment, MPOTransferMatrix
from ..networks.mps import InitialStateBuilder
from ..algorithms.mps_common import ZeroSiteH
from ..linalg import lanczos
from ..linalg.sparse import SumNpcLinearOperator
from ..tools.misc import find_subclass
from ..tools.params import asConfig

__all__ = simulation.__all__ + [
    'GroundStateSearch',
    'OrthogonalExcitations',
    'ExcitationInitialState',
]


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
        """Run the algorithm.

        Calls ``self.engine.run()``.
        """
        E, psi = self.engine.resume_run()
        self.results['energy'] = E


class OrthogonalExcitations(GroundStateSearch):
    """Find excitations by another GroundStateSearch orthogalizing against previous states.

    If the ground state is an infinite MPS, it is converted to `segment` boundary conditions
    at the beginning of this simulation.

    For finite systems, the first algorithm (say DMRG) run when switching the charge sector
    can be replaced by a normal DMRG run with a different intial state (in the desired sector).
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
    exctiations : list
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
        if self.options.get('save_psi', True):
            self.results['excitations'] = self.excitations
        self.init_env_data = {}

    def run(self):
        if self.orthogonal_to is None:
            self.init_orthogonal_from_groundstate()
        super().run()

    def resume_run(self):
        if self.orthogonal_to is None:
            self.init_orthogonal_from_groundstate()
        super().resume_run()

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
                If given, change the charge sector of the exciations compared to the ground state.
                Alternative to `apply_local_op` where we run a small zero-site diagonalization on
                the left-most bond in the desired charge sector to update the state.
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
        if np.linalg.norm(psi0.norm_test()) > self.options.get('orthogonal_norm_tol', 1.e-12):
            self.logger.info("call psi.canonicalf_form() on ground state")
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
            self.results['ground_state_energy'] = E0
        else:
            # we will switch charge sector
            self.orthogonal_to = []  # so we don't need to orthognalize against original g.s.
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
        enlarge = self.options.get('segment_enlarge', None)
        first = self.options.get('segment_first', 0)
        last = self.options.get('segment_last', None)
        self.model = model_inf.extract_segment(first, last, enlarge)
        first, last = self.model.lat.segment_first_last
        write_back = self.options.get('write_back_converged_ground_state_environments', False)
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
            if self.options.get('save_psi', True):
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

        if self.options.get('save_psi', True):
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
            raise ValuerError("can't switch charge sector with trivial charges!")
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
            assert switch_charge_sector is not None
            # get the correct environments on site 0
            LP = env.get_LP(0)
            RP = env._contract_RP(0, env.get_RP(0, store=True))  # saves the environments!
            self.results['ground_state_energy'] = env.full_contraction(0)
            for i in range(1, self.engine.n_optimize):
                env.del_LP(i)  # but we might have gotten more than we need
            H0 = ZeroSiteH.from_LP_RP(LP, RP)
            if self.model.H_MPO.explicit_plus_hc:
                H0 = SumNpcLinearOperator(H0, H0.adjoint())
            vL, vR = LP.get_leg('vR').conj(), RP.get_leg('vL').conj()
            th0 = npc.Array.from_func(np.ones, [vL, vR],
                                      dtype=self.psi.dtype,
                                      qtotal=switch_charge_sector,
                                      labels=['vL', 'vR'])
            lanczos_params = self.engine.lanczos_params
            _, th0, _ = lanczos.LanczosGroundState(H0, th0, lanczos_params).run()
            th0 = npc.tensordot(th0, self.psi.get_B(0, 'B'), axes=['vR', 'vL'])
            self.psi.set_B(0, th0, form='Th')
        qtotal_after = self.psi.get_total_charge()
        qtotal_diff = self.psi.chinfo.make_valid(qtotal_after - qtotal_before)
        self.logger.info("changed charge by %r compared to previous state", list(qtotal_diff))
        assert not np.all(qtotal_diff == 0)

    def run_algorithm(self):
        N_excitations = self.options.get("N_excitations", 1)
        ground_state_energy = self.results['ground_state_energy']
        self.logger.info("reference ground state energy: %.14f", ground_state_energy)
        if ground_state_energy > - 1.e-7:
            # the orthogonal projection does not lead to a different ground state!
            lanczos_params = self.engine.lancozs_params
            if self.engine.diag_method != 'lanczos' or \
                    ground_state_energy + 0.5 * lanczos_params.get('E_shift', 0.) > 0:
                # the factor of 0.5 is somewhat arbitrary, to ensure that
                # also excitations have energy < 0
                raise ValueError("You need to set use diag_method='lanczos' and small enough "
                                 f"lanczos_params['E_shift'] < {-2.* ground_state_energy:.2f}")

        while len(self.excitations) < N_excitations:

            E, psi = self.engine.run()

            self.results['excitation_energies'].append(E - ground_state_energy)
            self.logger.info("excitation energy: %.14f", E - ground_state_energy)
            if np.linalg.norm(psi.norm_test()) > self.options.get('orthogonal_norm_tol', 1.e-12):
                self.logger.info("call psi.canonical_form() on excitation")
                psi.canonical_form()
            self.excitations.append(psi)
            self.orthogonal_to.append(psi)
            # save in list of excitations
            if len(self.excitations) >= N_excitations:
                break

            self.make_measurements()
            self.logger.info("got %d excitations so far, proceeed to next excitation.\n%s",
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
        ranomzize_close_1 : bool
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
        if self.options.get('use_highest_excitation', True):
            psi = self.sim.orthogonal_to[-1]
        else:
            psi = self.sim.ground_state
        if isinstance(psi, dict):
            psi = psi['ket']
        psi = psi.copy()  # make a copy!
        return self._perturb(psi)

    def _perturb(self, psi):
        randomize_params = self.options.subconfig('randomize_params')
        close_1 = self.options.get('randomize_close_1', True)
        psi.perturb(randomize_params, close_1=close_1, canonicalize=False)
        return psi
