"""Simulations for ground state searches."""
# Copyright 2020-2021 TeNPy Developers, GNU GPLv3

import numpy as np
from pathlib import Path
import warnings

from . import simulation
from ..tools import hdf5_io, string
from .simulation import *
from ..linalg import np_conserved as npc
from ..models.model import Model
from ..networks.mpo import MPOEnvironment, MPOTransferMatrix
from ..networks.mps import MPS, InitialStateBuilder
from ..networks.umps import uMPS
from ..algorithms.mps_common import ZeroSiteH
from ..algorithms.dmrg import TwoSiteDMRGEngine
from ..linalg import lanczos
from ..linalg.sparse import SumNpcLinearOperator
from ..tools.misc import find_subclass, to_iterable, get_recursive
from ..tools.params import asConfig

import copy

__all__ = simulation.__all__ + [
    'GroundStateSearch',
    'PlaneWaveExcitations',
    'OrthogonalExcitations',
    'expectation_value_outside_segment_left',
    'expectation_value_outside_segment_right',
    'TopologicalExcitations',
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


class PlaneWaveExcitations(GroundStateSearch):
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

        # intialize original state
        self.psi = gs_data['psi']  # no copy!
        assert isinstance(self.psi, MPS) or isinstance(self.psi, uMPS)
        if np.linalg.norm(self.psi.norm_test()) > self.options.get('orthogonal_norm_tol', 1.e-12):
            if isinstance(self.psi, MPS):
                self.logger.info("call psi.canonical_form() on ground state")
                psi0.canonical_form()
            else:
                raise ValueError('uMPS does not pass norm test. Run VUMPS to get ground state or \n' +
                                 'convert to MPS and canonicalize.')
        if isinstance(self.psi, MPS):
            self.psi = uMPS.from_MPS(self.psi)

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
            write_back = self.options.get('write_back_converged_ground_state_environments', False)
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
        N_excitations = self.options.get("N_excitations", 1)
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
            self.logger.info("got %d excitations so far, proceeed to next excitation.\n%s",
                             len(self.excitations), "+" * 80)
            self.init_state()  # initialize a new state to be optimized
            self.init_algorithm()  # initialize new environemnts for the state!
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
    """Find excitations by another GroundStateSearch orthoganalizing against previous states.

    If the ground state is an infinite MPS, it is converted to `segment` boundary conditions
    at the beginning of this simulation.

    For finite systems, the first algorithm (say DMRG) run when switching the charge sector
    can be replaced by a normal DMRG run with a different intial state (in the desired sector).
    For infinite systems, the conversion to segment boundary conditions leads to a *different*
    state! Using the 'segment' boundary conditions, this class can e.g. study a single spin flip
    excitation in the background of the ground state, localized by the segment environments.

    Note that the segment environments are *soft* boundaries: the spin flip can be outside the
    segment where we vary the MPS tensors, as far as it is contained in the Schmidt states of the
    original ground state.

    Parameters
    ----------
    orthogonal_to : None | list
        States to orthogonalize against.
    gs_data : None | dict
        Data of the ground state which should be used instead of loading the date from
        :cfg:option:`OrthogonalExcitations.ground_state_filename`

    Options
    -------
    .. cfg:config :: OrthogonalExcitations
        :include: GroundStateSearch

        N_excitations : int
            Number of excitations to find.
            Don't make this too big, it's going to perform that many algorithm runs!

    Attributes
    ----------
    orthogonal_to : list
        States to orthogonalize against.
    excitations : list
        Tensor network states representing the excitations.
        The ground state in `orthogonal_to` is not included in the `excitations`.
        While being optimized, a state is saved as :attr:`psi` and not yet included in
        `excitations`.
    model_orig : :class:`~tenpy.models.model.Model`
        The original model before extracting the segment.
    ground_state_orig : :class:`~tenpy.networks.mps.MPS`
        The original ground state before extracting the segment.
    initial_state_seg :
        The initial state to be used in the segment: the ground state, but possibly perturbed
        and switched charge sector. Should be copied before modification.
    _gs_data : None | dict
        Only used to pass `gs_data` to :meth:`init_from_groundstate`;
        reset to `None` by the latter to allow to free memory.
    init_env_data : dict
        Initialization data for the :class:`~tenpy.networks.mpo.MPOEnvironment`.
        Passed to the algorithm class.
    initial_state_builder : None | :class:`~tenpy.networks.mps.MPS.InitialStateBuilder`
        Initialized after first call of :meth:`init_psi`,
        usually an :class:`ExcitationInitialState` instance.
    results : dict
        In addition to :attr:`~tenpy.simulations.simulation.Simulation.results`, it contains

            ground_state_energy : float
                Reference energy on the segment for the ground state up to an overall constant.
                Note that this is not just ``energy_density * L`` even for an originally
                translation invariant, infinite system, since terms crossing the boundaries
                need to be accounted for on both sides of the segment,
                and there can be an overall additive constant in the environments.
            excitations : list
                Tensor network states representing the excitations.
                Only defined if :cfg:option:`Simulation.save_psi` is True.
            excitation_energies : list of float
                Energies of the excited states *above* the reference `ground_state_energy`.
                These are well-defined, physical energies!
            segment_first_last : (int, int)
                First and last site of the segment extracted from the ground state.
                For finite MPS, this is usually just ``0, L-1``.

    """
    default_initial_state_builder = 'ExcitationInitialState'

    def __init__(self, options, *, orthogonal_to=None, gs_data=None, **kwargs):
        super().__init__(options, **kwargs)
        self.init_env_data = {}
        self._gs_data = gs_data
        self.initial_state_builder = None
        # Simulation.__init__() includes resume_data from kwargs into self.results['resume_data']
        # self.results might contain entries for excitations etc, if resuming from checkpoint
        resume_data = self.results.get('resume_data', {})
        if orthogonal_to is None and 'orthogonal_to' in resume_data and \
                not resume_data.get('sequential_simulations', False):
            orthogonal_to = resume_data['orthogonal_to']
        self.orthogonal_to = orthogonal_to
        if 'excitations' not in self.results:
            self.excitations = []
        else:
            self.excitations = self.results['excitations']
        self.results.setdefault('excitation_energies', [])
        self.results.setdefault('excitation_energies_MPO', [])
        if resume_data.get('sequential_simulations', False):
            self.logger.info("sequential run: start with empty orthogonal_to")
            self._previous_ortho = resume_data['orthogonal_to']
            del resume_data['orthogonal_to']  # sequential: get modified versions of those again
            self._previous_first_last = resume_data['segment_first_last']
            self._previous_offset = resume_data['ortho_offset']
            # need to reset excitations to not have any yet!
            self.excitations = []
            self.excitation_energies = []
            self.excitation_energies_MPO = []
        if self.options.get('save_psi', True):
            self.results['excitations'] = self.excitations

    def run(self):
        if not hasattr(self, 'ground_state_orig_L'):
            self.init_from_groundstate()
        return super().run()

    def resume_run(self):
        if not hasattr(self, 'ground_state_orig_L'):
            self.init_from_groundstate()
        return super().resume_run()

    def init_from_groundstate(self):
        """Initialize from the ground state data.

        Load the ground state and initialize the model from it.
        Calls :meth:`extract_segment`, :meth:`get_reference_energy`,
        and :meth:`switch_charge_sector`, to finally initialize :attr:`orthogonal_to`.

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
                If given, change the charge sector of the exciations compared to the ground state.
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
        gs_fn, gs_data = self._load_gs_data()
        gs_data_options = gs_data['simulation_parameters']
        for key in gs_data_options.keys():
            if not isinstance(key, str) or not key.startswith('model'):
                continue
            if key not in self.options:
                self.options[key] = gs_data_options[key]
        self.logger.info("initialize original ground state model")
        # initialize original model with model_class and model_params from ground state data
        self.init_model()
        self.model_orig = self.model

        # intialize original state
        self.ground_state_orig = psi0 = gs_data['psi']  # no copy!
        self.ground_state_orig_L = self.ground_state_orig_R = psi0
        if np.linalg.norm(psi0.norm_test()) > self.options.get('orthogonal_norm_tol', 1.e-12):
            self.logger.info("call psi.canonical_form() on ground state")
            psi0.canonical_form()

        # extract segments if necessary; get `init_env_data`.
        resume_data = gs_data.get('resume_data', {})
        psi0_seg, write_back = self.extract_segment(psi0, self.model, resume_data)
        if write_back:
            self.write_back_environments(gs_data, gs_fn)
        self.results['segment_first_last'] = self.model.lat.segment_first_last

        # here, psi0_seg is the *unperturbed* ground state in the segment!
        self.get_reference_energy(psi0_seg)

        # switch_charge_sector and perturb if necessary to define self.initial_state_seg
        self.initial_state_seg, self.qtotal_diff = self.switch_charge_sector(psi0_seg)
        self.results['qtotal_diff'] = self.qtotal_diff

        if self.orthogonal_to is None:
            if any(self.qtotal_diff):
                self.orthogonal_to = []  # different charge sector
                # so orthogonal to gs due to charge conservation
            else:
                self.orthogonal_to = [psi0_seg]
        else:
            # already initialized orthogonal_to from resume_data
            # check charge consistency
            assert tuple(self.results['qtotal_diff']) == tuple(self.qtotal_diff)
        self.logger.info("finished init_form_groundstate()")
        return gs_data

    def _load_gs_data(self):
        """Load ground state data from `ground_state_filename` or use simulation kwargs."""
        if self._gs_data is not None:
            self.options.touch('ground_state_filename')
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

    def extract_segment(self, psi0_orig, model_orig, resume_data):
        """Extract a finite segment from the original model and state.

        In case the original state is already finite, we might still extract a sub-segment
        (if `segment_first` and/or `segment_last` are given) or just use the full system.

        Defines :attr:`ground_state_seg` to be the ground state of the segment.
        Further :attr:`model` and :attr:`init_env_data` are extracted.

        Options
        -------
        .. cfg:configoptions :: OrthogonalExcitations

            segment_enlarge, segment_first, segment_last : int | None
                Arguments for :meth:`~tenpy.models.lattice.Lattice.extract_segment`.
                `segment_englarge` is only used for initially infinite ground states.
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
        if psi0_orig.bc == 'infinite':
            return self._extract_segment_from_infinite(psi0_orig, model_orig, resume_data)
        else:
            return self._extract_segment_from_finite(psi0_orig, model_orig)

    def _extract_segment_from_infinite(self, psi0_inf, model_inf, resume_data):
        enlarge = self.options.get('segment_enlarge', None)
        first = self.options.get('segment_first', 0)
        last = self.options.get('segment_last', None)
        write_back = self.options.get('write_back_converged_ground_state_environments', False)

        self.model = model_inf.extract_segment(first, last, enlarge)
        first, last = self.model.lat.segment_first_last

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
                                                         guess_init_env_data=guess_init_env_data)
        self.init_env_data = env_data
        ground_state_seg = psi0_inf.extract_segment(first, last)
        return ground_state_seg, write_back

    def _extract_segment_from_finite(self, psi0_fin, model_fin):
        first = self.options.get('segment_first', 0)
        last = self.options.get('segment_last', None)
        if first != 0 or last is not None:
            self.model = model_fin.extract_segment(first, last)
            first, last = self.model.lat.segment_first_last
            ground_state_seg = psi0_fin.extract_segment(first, last)
            env = MPOEnvironment(psi0_fin, self.model_orig.H_MPO, psi0_fin)
            self.init_env_data = env.get_initialization_data(first, last)
        else:
            last = psi0_fin.L - 1
            self.model = model_fin
            # always define `segment_first_last` to simplify measurments etc
            self.model.lat.segment_first_last = (first, last)
            ground_state_seg = psi0_fin
            self.init_env_data = {}
        return ground_state_seg, False

    def get_reference_energy(self, psi0_seg):
        """Obtain ground state reference energy.

        Excitation energies are full contractions of the MPOEnvironment with the environments
        defined in :attr:`init_env_data`.
        Hence, the reference energy is also the contraction of the `MPOEnvionment` on the segment.

        Parameters
        ----------
        psi0_seg : :class:`~tenpy.networks.msp.MPS`
            Ground state MPS on the segment, matching :attr:`init_env_data`.
        """
        # can't just use gs_data['energy'], since this is just energy density for infinite MPS
        self.logger.info("Calculate reference energy by contracting environments")
        env = MPOEnvironment(psi0_seg, self.model.H_MPO, psi0_seg, **self.init_env_data)
        E = env.full_contraction(0).real
        self.results['ground_state_energy'] = E
        return E

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

    def init_state(self):
        """Initialize the state.

        Options
        -------
        .. cfg:configoptions :: OrthogonalExcitations

            initial_state_params : dict
                The initial state parameters, :cfg:config:`ExcitationInitialState` defined below.
        """
        # re-initialize even if we already have a psi!
        if self.initial_state_builder is None:
            builder_class = self.options.get('initial_state_builder_class',
                                             self.default_initial_state_builder)
            params = self.options.subconfig('initial_state_params')
            Builder = find_subclass(InitialStateBuilder, builder_class)
            if issubclass(Builder, ExcitationInitialState):
                # incompatible with InitialStateBuilder: need to pass `sim` to __init__
                self.initial_state_builder = Builder(self, params)
            else:
                self.initial_state_builder = Builder(self.model.lat, params, self.model.dtype)

        if not hasattr(self, '_previous_ortho'):
            self.psi = self.initial_state_builder.run()
        else:
            # sequential run from previous simulation
            psi = self._previous_ortho[self._previous_offset + len(self.excitations)]
            self.psi = self._psi_from_previous_ortho(psi,
                                                     self._previous_first_last,
                                                     "previous sequential simulation",
                                                     False)
        if self.options.get('save_psi', True):
            self.results['psi'] = self.psi

    def perturb_initial_state(self, psi):
        """(Slightly) perturb an initial state in place.

        This is used to make sure it's not completely orthogonal to previous orthogonal states
        anymore; otherwise the effective Hamiltonian would be exactly 0 and we get numerical noise
        as first guess afterwards.

        Parameters
        ----------
        psi : :class:`~tenpy.networks.mps.MPS`
            The state to be perturbed *in place*.

        Options
        -------
        .. cfg:configoptions :: OrthogonalExcitations

            initial_state_randomize_params : dict-like
                Parameters for the random unitary evolution used to perturb the state a little bit
                in :meth:`~tenpy.networks.mps.MPS.perturb`.
                In addition, to those parameters, we read out the arguments `close_1` and
                `canonicalize` for :meth:`~tenpy.networks.mps.MPS.perturb`.

        """
        randomize_params = self.options.subconfig('initial_state_randomize_params')
        close_1 = randomize_params.get('close_1', True)
        canonicalize = randomize_params.get('canonicalize', True)
        psi.perturb(randomize_params, close_1=close_1, canonicalize=canonicalize)

    def init_algorithm(self, **kwargs):
        kwargs.setdefault('orthogonal_to', self.orthogonal_to)
        if 'resume_data' not in kwargs and 'resume_data' in self.results:
            resume_data = self.results['resume_data']
        else:
            resume_data = kwargs.setdefault('resume_data', {})
        resume_data['init_env_data'] = self.init_env_data
        super().init_algorithm(**kwargs)

    def switch_charge_sector(self, psi0_seg):
        """Change the charge sector of `psi0_seg` and obtain `initial_state_seg`.

        Parameters
        ----------
        psi0_seg : :class:`~tenpy.networks.msp.MPS`
            (Unperturbed) ground state MPS on the segment.

        Returns
        -------
        initial_state_seg : :class:`~tenpy.networks.mps.MPS`
            Suitable initial state for first exciation run.  Might also be used for later
            initial states, see :meth:`ExcitationInitialState.from_orthogonal`.
        """
        apply_local_op = self.options.get("apply_local_op", None)
        switch_charge_sector = self.options.get("switch_charge_sector", None)
        if apply_local_op is None and switch_charge_sector is None:
            # don't switch charge sector
            # need to perturb psi0_seg to make sure we have some overlap with psi1, psi2
            # and that H_eff is not exactly zero on the initial state.
            psi = psi0_seg.copy()  # copy since we perturb it!
            initial_state_params = self.options.subconfig('initial_state_params')
            randomize_params = initial_state_params.subconfig('randomize_params')
            close_1 = initial_state_params.get('randomize_close_1', True)
            self.perturb_initial_state(psi)
            return psi, psi.chinfo.make_valid()  # no change in charges

        psi = psi0_seg.copy()  # might be necessary to orthogonalize against psi0_seg in the end
        qtotal_before = psi.get_total_charge()
        self.logger.info("Charges of the original segment: %r", list(qtotal_before))

        if apply_local_op is not None and switch_charge_sector is not None:
            # TODO: we can lift this restricition if we define whether `switch_charge_sector`
            # is *in addition* to apply_local_op or overall change in qtotal
            raise ValueError("Give only one of `switch_charge_sector` and `apply_local_op`")

        if apply_local_op:  # also nothing to do if apply_local_op=[]
            self._apply_local_op(psi, apply_local_op)

        if switch_charge_sector is not None:
            self._switch_charge_sector_with_glue(psi, switch_charge_sector)

        self.logger.info("call psi.canonical_form()")
        psi.canonical_form_finite()  # no need to update envs: keep no envs!
        # psi.segment_boundaries has trafo compared to `init_env_data`.

        qtotal_after = psi.get_total_charge()
        qtotal_diff = psi.chinfo.make_valid(qtotal_after - qtotal_before)
        self.logger.info("changed charge by %r compared to previous state", list(qtotal_diff))
        return psi, qtotal_diff

    def _apply_local_op(self, psi, apply_local_op):
        #apply_local_op should have the form [site1, op1, site2, op2, ...]
        assert len(apply_local_op) % 2 == 0
        self.logger.info("apply local operators (to switch charge sector)")
        first, last = self.results['segment_first_last']
        term = list(zip(apply_local_op[-1::-2], apply_local_op[-2::-2]))  # [(op, site), ...]
        for op, i in term:
            j = int(i)  # error for apply_local_op=["Sz", i, ...] instead of [i, "Sz", ...]
            j = j - first  # convert from original MPS index to segment MPS index
            if not 0 <= j < psi.L:
                raise ValueError(f"specified site {j:d} in segment = {i:d} in original MPS"
                                 f"is not in segment [{first:d}, {last:d}]!")
        psi.apply_local_term(term, i_offset=-first, canonicalize=False)

    def _switch_charge_sector_with_glue(self, psi, qtotal_change):
        if psi.chinfo.qnumber == 0:
            raise ValueError("can't switch charge sector with trivial charges!")
        first, last = self.results['segment_first_last']
        # switch_charge_sector_site defaults to the center of the segment
        # indexed by *original* MPS index
        site = self.options.get("switch_charge_sector_site", psi.L // 2 + first) - first
        if not 0 <= site < psi.L:
            raise ValueError(f"specified site index {site + first:d} in original MPS ="
                             f"{site:d} in segment MPS is *not* in segment")
        env = MPOEnvironment(psi, self.model.H_MPO, psi, **self.init_env_data)
        LP = env.get_LP(site, store=False)
        RP = env._contract_RP(site, env.get_RP(site, store=False))
        # no need to save the environments: will anyways call `psi.canonical_form_finite()`.
        H0 = ZeroSiteH.from_LP_RP(LP, RP)
        if self.model.H_MPO.explicit_plus_hc:
            H0 = SumNpcLinearOperator(H0, H0.adjoint())
        vL, vR = LP.get_leg('vR').conj(), RP.get_leg('vL').conj()
        th0 = npc.Array.from_func(np.ones, [vL, vR],
                                  dtype=psi.dtype,
                                  qtotal=qtotal_change,
                                  labels=['vL', 'vR'])
        lanczos_params = self.options.subconfig('algorithm_params').subconfig('lanczos_params')
        _, th0, _ = lanczos.LanczosGroundState(H0, th0, lanczos_params).run()

        # Check norm after Lanczos so that it is one.
        # TODO: check this already before lanczos?
        norm = npc.norm(th0)
        self.logger.info("Norm of theta guess: %.8f", norm)
        if np.isclose(norm, 0):
            raise ValueError(f"Norm of inserted theta with charge {list(qtotal_change)} on site index {site:d} is zero.")

        U, s, Vh = npc.svd(th0, inner_labels=['vR', 'vL'])
        psi.set_B(site-1, npc.tensordot(psi.get_B(site-1, 'A'), U, axes=['vR', 'vL']), form='A')
        psi.set_B(site, npc.tensordot(Vh, psi.get_B(site, 'B'), axes=['vR', 'vL']), form='B')
        psi.set_SL(site, s)

    # TODO grouping sites doesn't work?
    #  def group_sites_for_algorithm(self):
    #      super().group_sites_for_algorithm()
    #      group_sites = self.grouped
    #      if group_sites > 1:
    #          for ortho in self.orthogonal_to:
    #              if ortho.grouped < group_sites:
    #                  ortho.group_sites(group_sites)
    #      # done

    #  def group_split(self):
    #      # TODO: trunc_params should be attribute of MPS
    #      if self.grouped > 1:
    #          trunc_params = self.options['algorithm_params']['trunc_params']
    #          for ortho in self.orthogonal_to:
    #              if ortho.grouped > 1:
    #                  orhto.group_split(trunc_params)
    #      super().group_split()

    def run_algorithm(self):
        N_excitations = self.options.get("N_excitations", 1)
        ground_state_energy = self.results['ground_state_energy']
        self.logger.info("reference ground state energy: %.14f", ground_state_energy)
        if ground_state_energy > - 1.e-7:
            # TODO can we fix all of this by using H -> H + E_shift |ortho><ortho| ?
            # the orthogonal projection does not lead to a different ground state!
            lanczos_params = self.engine.lanczos_params
            # [TODO] I was having issues where self.engine.diag_method isn't lanczos or E_shift isn't defined.
            if self.engine.diag_method == 'default': # SAJANT, 09/09/21
                self.engine.diag_method = 'lanczos'
            self.logger.info('Lanczos Params in run_algorithm: %r', lanczos_params)
            # When E_shift isn't specified, we get a None, which throws an error below.
            E_shift = lanczos_params.get('E_shift', -100) #lanczos_params['E_shift'] if lanczos_params['E_shift'] is not None else 0
            if E_shift is None:
                E_shift = -100
            self.logger.info("Shifted ground state energy: %.14f", ground_state_energy + 0.5 * E_shift)

            if self.engine.diag_method != 'lanczos' or \
                    ground_state_energy + 0.5 * E_shift > 0:
                # the factor of 0.5 is somewhat arbitrary, to ensure that
                # also excitations have energy < 0
                print("lanczos_params['E_shift']:", lanczos_params['E_shift'])
                raise ValueError("You need to set use diag_method='lanczos' and negative enough "
                                 f"lanczos_params['E_shift'] < {-2.* ground_state_energy:.2f}")

        # loop over excitations
        while len(self.excitations) < N_excitations:

            E, psi = self.engine.run()
            E_MPO = self.engine.env.full_contraction(0) # TODO: measure this while envs are still around?
            self.results['excitation_energies_MPO'].append(E_MPO - ground_state_energy)  # TODO: should be almost the same?!
            self.results['excitation_energies'].append(E - ground_state_energy)
            self.logger.info("excitation energy: %.14f", E - ground_state_energy)
            if np.linalg.norm(psi.norm_test()) > self.options.get('orthogonal_norm_tol', 1.e-12):
                self.logger.info("call psi.canonical_form() on excitation")
                psi.canonical_form_finite(envs_to_update=[self.engine.env])
            self.orthogonal_to.append(psi)  # in `orthogonal_to` we need the grouped, segment psi
            # but in `excitations` we want the "original", ungrouped, measurement psi
            psi_meas, _ = self.get_measurement_psi_model(psi, self.model)
            self.excitations.append(psi_meas)  # save in list of excitations
            if len(self.excitations) >= N_excitations:
                break

            self.make_measurements()
            self.logger.info("got %d excitations so far, proceeed to next excitation.\n%s",
                             len(self.excitations), "+" * 80)
            self.init_state()  # initialize a new state to be optimized
            self.init_algorithm()
        # done

    def resume_run_algorithm(self):
        """Not Implemented"""
        raise NotImplementedError("TODO")

    def get_measurement_psi_model(self, psi, model):
        """Get psi for measurements.

        Sometimes, the `psi` we want to use for measurements is different from the one the
        algorithm actually acts on.
        Here, we split sites, if they were grouped in :meth:`group_sites_for_algorithm`.

        Parameters
        ----------
        psi :
            Tensor network; initially just ``self.psi``.
            The method should make a copy before modification.
        model :
            Model matching `psi` (in terms of indexing, MPS order, grouped sites, ...)
            Initially just ``self.model``.

        Returns
        -------
        psi :
            The psi suitable as argument for generic measurement functions.
        model :
            Model matching `psi` (in terms of indexing, MPS order, grouped sites, ...)

        Options
        -------
        .. cfg:configoptions :: OrthogonalExcitations

            measure_add_unitcells : int | (int, int) | None
                It can be a single value (default=0), or two separate values for left/right.
                For ``bc_MPS='finite'`` in the :attr`model_orig`, only 0 is allowed.
                `None` disables the feature, but this may cause measurment functions/results to
                have unexpected (or possibly wrong, if not accounted for) indexing.
        """
        psi, model = super().get_measurement_psi_model(psi, model)  # ungroup if necessary

        measure_add_unitcells = self.options.get('measure_add_unitcells', 0)
        if measure_add_unitcells is not None:
            first, last = self.results['segment_first_last']
            psi, meas_first, meas_last = psi.extract_enlarged_segment(self.ground_state_orig_L,
                                                                      self.ground_state_orig_R,
                                                                      first,
                                                                      last,
                                                                      measure_add_unitcells)
            key = 'measure_segment_first_last'
            if key not in self.results:
                self.results[key] = (meas_first, meas_last)
            else:
                assert self.results[key] == (meas_first, meas_last)
            model  = self.model_orig.extract_segment(meas_first, meas_last)
        return psi, model

    def get_resume_data(self, sequential_simulations=False):
        resume_data = super().get_resume_data(sequential_simulations)
        if self.options['save_psi'] or sequential_simulations:
            resume_data['ortho_offset'] = len(self.excitations) - len(self.orthogonal_to)
            resume_data['orthogonal_to'] = self.orthogonal_to
        if sequential_simulations:
            resume_data['segment_first_last'] = self.results['segment_first_last']
        return resume_data

    def _psi_from_previous_ortho(self, psi, previous_first_last, source="?", perturb=False):
        """Adjust a state from a previous `resume_data['orthogonal_to']` to be used for self.

        """
        self.logger.info('initializing psi from %s', source)
        # enlarge segment size if necessary
        new_first_last = self.results['segment_first_last']
        if previous_first_last != new_first_last:
            prev_first, prev_last = previous_first_last
            psi, _, _ = psi.extract_enlarged_segment(self.sim.ground_state_orig_L,
                                                     self.sim.ground_state_orig_R,
                                                     prev_first,
                                                     prev_last,
                                                     new_first_last=new_first_last)
        # double-check that charges are what we want them to be
        psi_qtotal = psi.get_total_charge()
        expect_qtotal = self.initial_state_seg.get_total_charge()
        if np.any(psi_qtotal - expect_qtotal):
            raise ValueError(f"psi from {source!s} has different charge "
                             f"{psi_qtotal!r} than expected {expect_qtotal!r}")
        psi_legs = psi.outer_virtual_legs()
        expect_legs = self.initial_state_seg.outer_virtual_legs()
        for leg_previous, leg_expected in zip(psi_legs, expect_legs):
            try:
                leg_previous.test_equal(leg_expected)
            except ValueError as e:
                raise ValueError("psi from {source!s} has incompatible legs "
                                 "with current simulation") from e
        # finally perturb, if desired
        if perturb:
            self.perturb_initial_state(psi)
        return psi


def expectation_value_outside_segment_right(psi_segment, psi_R, ops, lat_segment, sites=None, axes=None):
    """Calculate expectation values outside of the segment to the right.

    Parameters
    ----------
    psi_S :
        Segment MPS.
    psi_R :
        Infinite MPS on the right.
    lat_segment : :class:`~tenpy.models.lattice.Lattice`
        The lattice of the segment MPS. In particular, it should have `segment_first_last`
    ops, sites, axes:
        As for :meth:`~tenpy.networks.mps.MPS.expectation_value`.
        `sites` should only have values > 0, with 0 being the first site on the right of the
        segment. If `ops` is non-uniform, it is indexed as for `psi_R`.
    """
    # TODO move these functions to a different location in code?
    # TODO rigorous tests
    psi_S = psi_segment
    assert psi_S.bc == 'segment'
    if hasattr(lat_segment,"segment_first_last"):
        first, last = lat_segment.segment_first_last
    else:
        first, last = 0,lat_segment.N_sites - 1
    assert psi_S.L == last - first + 1
    shift = last + 1 # = first + psi_S.L = index in `sites` relative to MPS index of psi_R
    if sites is None:
        # one MPS unit cell plus partially filled if non-trivial `last`
        sites = np.arange(psi_R.L + (psi_R.L - last + 1 % psi_R.L if last % psi_R.L else 0))
    sites = [i + shift for i in sites]
    ops, sites, n, (op_ax_p, op_ax_pstar) = psi_R._expectation_value_args(ops, sites, axes)
    ax_p = ['p' + str(k) for k in range(n)]
    ax_pstar = ['p' + str(k) + '*' for k in range(n)]
    UL, VR = psi_S.segment_boundaries
    S = psi_S.get_SR(psi_S.L - 1)
    if VR is None:
        rho = npc.diag(S**2,
                        psi_S.get_B(psi_S.L - 1, None).get_leg('vR'),
                        labels=['vR', 'vR*'])
    else:
        rho = VR.scale_axis(S, 'vL')
        rho = npc.tensordot(rho.conj(), rho, axes=['vL*', 'vL'])
    E = []
    k = shift  # starting on that site
    for i in sorted(sites):
        assert k <= i
        while k < i:
            B = psi_R.get_B(k, form='B')
            rho = npc.tensordot(rho, B, ['vR', 'vL'])
            rho = npc.tensordot(B.conj(), rho, [['vL*', 'p*'] , ['vR*', 'p']])
            k += 1
        op = psi_R.get_op(ops, i)
        op = op.replace_labels(op_ax_p + op_ax_pstar, ax_p + ax_pstar)
        Bs = psi_R.get_B(i, form='B', label_p='0')
        for k in range(1, n):
            Bs = npc.tensordot(Bs, psi_R.get_B(i+k, 'B', label_p=str(k)), ['vR', 'vL'])
        C = npc.tensordot(op, Bs, axes=[ax_pstar, ax_p])
        C = npc.tensordot(rho, C, axes=['vR', 'vL'])
        E.append(npc.inner(Bs.conj(), C, axes=[['vL*'] + ax_pstar + ['vR*'],
                                                ['vR*'] + ax_p + ['vR']]))
    return np.real_if_close(E)


def expectation_value_outside_segment_left(psi_segment, psi_L, ops, lat_segment, sites=None, axes=None):
    """Calculate expectation values outside of the segment to the right.

    Parameters
    ----------
    psi_S :
        Segment MPS.
    psi_L :
        Infinite MPS on the left.
    ops, sites, axes:
        As for :meth:`~tenpy.networks.mps.MPS.expectation_value`.
        `sites` should only have values < 0, with -1 being the first site on the left of the
        segment. If `ops` is non-uniform, it is indexed as for `psi_L`.
    """
    psi_S = psi_segment
    assert psi_S.bc == 'segment'
    if hasattr(lat_segment,"segment_first_last"):
        first, last = lat_segment.segment_first_last
    else:
        first, last = 0, lat_segment.N_sites - 1
    assert psi_S.L == last - first + 1
    shift = first  # = index in `sites` relative to MPS index of psi_R
    if sites is None:
        # one MPS unit cell plus partially filled if non-trivial `first`
        sites = np.arange(-psi_L.L - (first if first % psi_L.L else 0), 0)
    sites = [i + shift for i in sites]
    ops, sites, n, (op_ax_p, op_ax_pstar) = psi_L._expectation_value_args(ops, sites, axes)
    ax_p = ['p' + str(k) for k in range(n)]
    ax_pstar = ['p' + str(k) + '*' for k in range(n)]
    UL, VR = psi_S.segment_boundaries
    S = psi_S.get_SL(0)
    if UL is None:
        rho = npc.diag(S**2,
                       psi_S.get_B(0, None).get_leg('vL'),
                       labels=['vL', 'vL*'])
    else:
        rho = UL.scale_axis(S, 'vR')
        rho = npc.tensordot(rho, rho.conj(), axes=['vR', 'vR*'])
    E = []
    k = shift -1
    for i in sorted(sites, reverse=True):
        assert i <= k
        while k > i:
            A = psi_L.get_B(k, form='A')
            rho = npc.tensordot(A, rho, ['vR', 'vL'])
            rho = npc.tensordot(rho, A.conj(), [['p', 'vL*'] , ['p*', 'vR*']])
            k -= 1
        op = psi_L.get_op(ops, i)
        op = op.replace_labels(op_ax_p + op_ax_pstar, ax_p + ax_pstar)
        As = psi_L.get_B(i - (n-1), form='A', label_p='0')
        for k in range(1, n):
            As = npc.tensordot(As, psi_L.get_B(i - (n - 1) + k, 'A', label_p=str(k)), ['vR', 'vL'])
        C = npc.tensordot(op, As, axes=[ax_pstar, ax_p])
        C = npc.tensordot(C, rho, axes=['vR', 'vL'])
        E.append(npc.inner(As.conj(), C, axes=[['vL*'] + ax_pstar + ['vR*'],
                                                ['vL'] + ax_p + ['vL*']]))
    return np.real_if_close(E[::-1])


class TopologicalExcitations(OrthogonalExcitations):
    def __init__(self, options, *, gs_data_L=None, gs_data_R=None, **kwargs):
        super().__init__(options, **kwargs)
        resume_data = kwargs.get('resume_data', {})
        #  if orthogonal_to is None and 'orthogonal_to' in resume_data:
        #      orthogonal_to = kwargs['resume_data']['orthogonal_to']
        #      self.options.touch('ground_state_filename_left', 'ground_state_filename_right')
        self.orthogonal_to = None # TODO: allow orthogonal_to
        self.excitations = resume_data.get('excitations', [])
        self.results['excitation_energies'] = []
        self.results['excitation_energies_MPO'] = []
        if self.options.get('save_psi', True):
            self.results['excitations'] = self.excitations
        self.init_env_data = {}
        self._gs_data_L = gs_data_L
        self._gs_data_R = gs_data_R
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
                If given, change the charge sector of the exciations compared to the ground state.
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
        gs_fn_L, gs_data_L, gs_fn_R, gs_data_R = self._load_gs_data()
        gs_data_options_L = gs_data_L['simulation_parameters']
        # initialize original model with model_class and model_params from ground state data
        self.logger.info("initialize original ground state model")
        for key in gs_data_options_L.keys(): # Assume same model params for left and right
            if not isinstance(key, str) or not key.startswith('model'):
                continue
            if key not in self.options:
                self.options[key] = gs_data_options_L[key]
        self.init_model() # FOR NOW, WE ASSUME LEFT AND RIGHT MODELS ARE THE SAME
        self.model_orig = self.model

        # intialize original state
        self.ground_state_orig_L = psi0_L = gs_data_L['psi']  # no copy!
        self.ground_state_orig_R = psi0_R = gs_data_R['psi']  # no copy!
        assert self.ground_state_orig_L.L == self.ground_state_orig_R.L
        if np.linalg.norm(psi0_L.norm_test()) > self.options.get('orthogonal_norm_tol', 1.e-12):
            self.logger.info("call psi.canonical_form() on left ground state")
            psi0_L.canonical_form()
        if np.linalg.norm(psi0_R.norm_test()) > self.options.get('orthogonal_norm_tol', 1.e-12):
            self.logger.info("call psi.canonical_form() on right ground state")
            psi0_R.canonical_form()

        # extract segments if necessary; get `init_env_data`.
        resume_data_L = gs_data_L.get('resume_data', {}) # TODO this is probably wrong
        resume_data_R = gs_data_R.get('resume_data', {}) # TODO this is probably wrong
        psi0_seg, write_back_left, write_back_right = self.extract_segment(psi0_L, psi0_R, self.model, resume_data_L, resume_data_R)
        ########################################
        if write_back_left:
            init_env_data = self.init_env_data
            self.init_env_data = self.init_env_data_L
            self.write_back_environments(gs_data_L, gs_fn_L)
            self.init_env_data = init_env_data
        if write_back_right:
            init_env_data = self.init_env_data
            self.init_env_data = self.init_env_data_R
            self.write_back_environments(gs_data_R, gs_fn_R)
            self.init_env_data = init_env_data
        self.results['segment_first_last'] = self.model.lat.segment_first_last

        # here, psi0_seg is the *unperturbed* ground state in the segment!
        self.get_reference_energy(psi0_L, psi0_R)

        # switch_charge_sector defines `self.initial_state_seg`
        self.initial_state_seg, self.qtotal_diff = self.switch_charge_sector(psi0_seg)
        self.results['qtotal_diff'] = self.qtotal_diff

        self.orthogonal_to = []  # Segment is inherently different than either left or right ground state.
        # Or at least the two sides will be different for non-trivial calculation.
        return None # return isn't used

    def _load_gs_data(self):
        """Load ground state data from `ground_state_filename` or use simulation kwargs."""
        gs_data_return = []
        for which, gs_D in zip(['left', 'right'], [self._gs_data_L, self._gs_data_R]):
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

    def extract_segment(self, psi0_L_Orig, psi0_R_Orig, model_orig, resume_data_L, resume_data_R):
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
                `segment_englarge` is only used for initially infinite ground states.
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
        if psi0_L_Orig.bc == 'infinite':
            return self._extract_segment_from_infinite(psi0_L_Orig, psi0_R_Orig, model_orig, resume_data_L, resume_data_R)
        else:
            return self._extract_segment_from_finite(psi0_L_Orig, psi0_R_Orig, model_orig)

    def _extract_segment_from_finite(self, psi0_fin_L, psi0_fin_R, model_fin):
        first = self.options.get('segment_first', 0)
        last = self.options.get('segment_last', None)
        boundary = self.options.get('segment_boundary', (last-first)//2 +first if last is not None else (psi0_fin_L.L-first)//2 + first)
        assert first < boundary
        if last is not None:
            assert boundary < last

        self.model = model_fin.extract_segment(first, last)
        first, last = self.model.lat.segment_first_last
        ground_state_seg_L = psi0_fin_L.extract_segment(first, boundary-1) # 2nd index included in segment
        ground_state_seg_R = psi0_fin_R.extract_segment(boundary, last)

        env = MPOEnvironment(psi0_fin_L, self.model_orig.H_MPO, psi0_fin_L)
        self.env_data_L = env.get_initialization_data(first, last)
        self.env_data_L_seg = env.get_initialization_data(first, boundary-1)

        env = MPOEnvironment(psi0_fin_R, self.model_orig.H_MPO, psi0_fin_R)
        self.env_data_R = env.get_initialization_data(first, last)
        self.env_data_R_seg = env.get_initialization_data(boundary, last)

        ground_state_seg = self._glue_segments(ground_state_seg_L, ground_state_seg_R,
                                               psi0_fin_L, psi0_fin_R,
                                               self.model, (first, last, boundary))

        if first != 0 or last is not None:
            self.init_env_data = {'init_LP': self.env_data_L['init_LP'],
                                  'init_RP': self.env_data_R['init_RP'],
                                  'age_LP': 0,
                                  'age_RP': 0}

            #self.init_env_data = self._contract_segment_boundaries(self.init_env_data, *ground_state_seg.segment_boundaries)
        else:
            assert ground_state_seg_L.L + ground_state_seg_R.L == psi0_fin_L.L
            self.init_env_data = {}

        return ground_state_seg, False, False

    def _contract_segment_boundaries(self, env_data, U, Vh):
        self.logger.info("Put segment boundaries into domain wall envs.")
        if U is not None:
            init_LP = npc.tensordot(U.conj(), env_data['init_LP'], axes=(['vL*'], ['vR*']))
            init_LP = npc.tensordot(init_LP, U, axes=(['vR'], ['vL']))
            env_data['init_LP'] = init_LP

        if Vh is not None:
            init_RP = npc.tensordot(Vh, env_data['init_RP'], axes=(['vR'], ['vL']))
            init_RP = npc.tensordot(init_RP, Vh.conj(), axes=(['vL*'], ['vR*']))
            env_data['init_RP'] = init_RP

        return env_data

    def _extract_segment_from_infinite(self, psi0_inf_L, psi0_inf_R, model_inf, resume_data_L, resume_data_R):
        enlarge = self.options.get('segment_enlarge', None)
        first = self.options.get('segment_first', 0)
        if enlarge is not None:
            assert first == 0
        last = self.options.get('segment_last', None)

        assert (enlarge is None) ^ (last is None), "'enlarge' xor 'last' must be not None."
        boundary = self.options.get('segment_boundary', (last - first) // 2 + first if enlarge is None else (enlarge//2)*psi0_inf_L.L + first)
        assert first < boundary
        if last is not None:
            assert boundary < last
        write_back = self.options.get('write_back_converged_ground_state_environments', False)

        self.model = model_inf.extract_segment(first, last, enlarge)
        first, last = self.model.lat.segment_first_last
        H = model_inf.H_MPO

        gauge = self.options.get('gauge', 'rho')
        if resume_data_L.get('converged_environments', False):
            self.logger.info("use converged environments from left ground state file")
            self.init_env_data_L = resume_data_L['init_env_data'] # Environments for infinite ground states
            psi0_inf_L = resume_data_L.get('psi', psi0_inf_L)
            write_back_left = False
        else:
            self.logger.info("converge left ground state environments with MPOTransferMatrix")
            guess_init_env_data = resume_data_L.get('init_env_data', None)
            self.init_env_data_L = MPOTransferMatrix.find_init_LP_RP(H, psi0_inf_L, guess_init_env_data=guess_init_env_data, _subtraction_gauge=gauge)
            # On bond 0 of the unit cell

            write_back_left = write_back

        if resume_data_R.get('converged_environments', False):
            self.logger.info("use converged environments from right ground state file")
            self.init_env_data_R = resume_data_R['init_env_data']
            psi0_inf_R = resume_data_R.get('psi', psi0_inf_R)
            write_back_right = False
        else:
            self.logger.info("converge right ground state environments with MPOTransferMatrix")
            guess_init_env_data = resume_data_R.get('init_env_data', None)
            self.init_env_data_R = MPOTransferMatrix.find_init_LP_RP(H, psi0_inf_R, guess_init_env_data=guess_init_env_data, _subtraction_gauge=gauge)
            # On bond 0 of the unit cell

            write_back_right = write_back
        self.logger.info("converge segment environments with MPOTransferMatrix")

        env = MPOEnvironment(psi0_inf_L, H, psi0_inf_L, **self.init_env_data_L)
        self.env_data_L = env.get_initialization_data(first, last)
        self.env_data_L_seg = env.get_initialization_data(first, boundary-1)

        env = MPOEnvironment(psi0_inf_R, H, psi0_inf_R, **self.init_env_data_R)
        self.env_data_R = env.get_initialization_data(first, last)
        self.env_data_R_seg = env.get_initialization_data(boundary, last)

        self.init_env_data = {'init_LP': self.env_data_L['init_LP'],
                    'init_RP': self.env_data_R['init_RP'],
                    'age_LP': 0,
                    'age_RP': 0}

        ground_state_seg_L = psi0_inf_L.extract_segment(first, boundary-1)
        ground_state_seg_R = psi0_inf_R.extract_segment(boundary, last)
        ground_state_seg = self._glue_segments(ground_state_seg_L, ground_state_seg_R,
                                               psi0_inf_L, psi0_inf_R,
                                               self.model, (first, last, boundary))

        return ground_state_seg, write_back_left, write_back_right


    def _glue_segments(self, seg_L, seg_R, inf_L, inf_R, model, indices):
        join_method = self.join_method = self.options.get('join_method', "average charge")
        if inf_L.finite or inf_R.finite:
            assert join_method == "most probable charge"
        first, last, boundary = indices
        print(first, last, boundary)
        left_half_model = self.model_orig.extract_segment(first, boundary-1, None)
        right_half_model = self.model_orig.extract_segment(boundary, last, None)

        env_left_BC = MPOEnvironment(seg_L, left_half_model.H_MPO, seg_L, **self.env_data_L_seg)
        env_right_BC = MPOEnvironment(seg_R, right_half_model.H_MPO, seg_R, **self.env_data_R_seg)
        LP = env_left_BC._contract_LP(seg_L.L-1, env_left_BC.get_LP(seg_L.L-1, store=False))
        RP = env_right_BC._contract_RP(0, env_right_BC.get_RP(0, store=False))  # saves the environments!
        H0 = ZeroSiteH.from_LP_RP(LP, RP)
        if self.model.H_MPO.explicit_plus_hc:
            H0 = SumNpcLinearOperator(H0, H0.adjoint())
        vL, vR = LP.get_leg('vR').conj(), RP.get_leg('vL').conj()

        if seg_L.chinfo.qnumber == 0:    # Handles the case of no charge-conservation
            desired_Q = None
        else:
            if join_method == "average charge":
                Q_bar_L = inf_L.average_charge(0)
                for i in range(1, inf_L.L):
                    Q_bar_L += inf_L.average_charge(i)
                Q_bar_L = vL.chinfo.make_valid(np.around(Q_bar_L / inf_L.L))
                self.logger.info("Charge of left BC, averaged over site and unit cell: %r", Q_bar_L)

                Q_bar_R = inf_R.average_charge(0)
                for i in range(1, inf_R.L):
                    Q_bar_R += inf_R.average_charge(i)
                Q_bar_R = vR.chinfo.make_valid(-1 * np.around(Q_bar_R / inf_R.L))
                self.logger.info("Charge of right BC, averaged over site and unit cell: %r", -1*Q_bar_R)
                desired_Q = list(vL.chinfo.make_valid(Q_bar_L + Q_bar_R))
            elif join_method == "most probable charge":
                posL = seg_L.L
                posR = 0
                QsL, psL = seg_L.probability_per_charge(posL)
                QsR, psR = seg_R.probability_per_charge(posR)

                side_by_side = string.vert_join(["left seg\n" + str(QsL), "prob\n" + str(np.array([psL]).T), "right seg\n" + str(QsR),"prob\n" +str(np.array([psR]).T)], delim=' | ')
                self.logger.info(side_by_side)

                Qmostprobable_L = QsL[np.argmax(psL)]
                Qmostprobable_R = -1 * QsR[np.argmax(psR)]
                self.logger.info("Most probable left:" + str(Qmostprobable_L))
                self.logger.info("Most probable right:" + str(Qmostprobable_R))
                desired_Q = list(vL.chinfo.make_valid(Qmostprobable_L + Qmostprobable_R))
            else:
                raise ValueError("Invalid `join_method` %s " % join_method)
        self.gluing_charge = desired_Q
        self.logger.info("Desired gluing charge: %r", desired_Q)

        # We need a tensor that is non-zero only when Q = (Q^i_L - bar(Q_L)) + (Q^i_R - bar(Q_R))
        # Q is the the charge we insert. Here we only do charge gluing to get a valid segment.
        # Changing charge sector is done below by basically identical code when the segment is already formed.
        th0 = npc.Array.from_func(np.ones, [vL, vR],
                                  dtype=seg_L.dtype,
                                  qtotal=desired_Q,
                                  labels=['vL', 'vR'])
        lanczos_params = self.options.get("lanczos_params", {}) # See if lanczos_params is in yaml, if not use empty dictionary
        _, th0, _ = lanczos.LanczosGroundState(H0, th0, lanczos_params).run()

        norm = npc.norm(th0)
        self.logger.info("Norm of theta guess: %.8f", norm)
        if np.isclose(norm, 0):
            raise ValueError(f"Norm of inserted theta with charge {list(qtotal_change)} on site index {site:d} is zero.")

        U, s, Vh = npc.svd(th0, inner_labels=['vR', 'vL'])
        seg_L.set_B(seg_L.L-1, npc.tensordot(seg_L.get_B(seg_L.L-1, 'A'), U, axes=['vR', 'vL']), form='A') # Put AU into last site of left segment
        seg_L.set_SR(seg_L.L-1, s)
        seg_R.set_B(0, npc.tensordot(Vh, seg_R.get_B(0, 'B'), axes=['vR', 'vL']), form='B') # Put Vh B into first site of right segment
        seg_R.set_SL(0, s)

        combined_seg = self._concatenate_segments(seg_L, seg_R, inf_L)

        return combined_seg

    def _concatenate_segments(self, seg_L, seg_R, inf_L):
        l_sites = [seg_L.sites[i] for i in range(seg_L.L)]
        lA = [seg_L.get_B(i, 'A') for i in range(seg_L.L)]
        lS = [seg_L.get_SL(i) for i in range(seg_L.L)]
        #lS.append(seg_L.get_SR(seg_L.L-1))

        # Building segment MPS on right half
        r_sites = [seg_R.sites[i] for i in range(seg_R.L)]
        rB = [seg_R.get_B(i) for i in range(seg_R.L)]
        rS = [seg_R.get_SL(i) for i in range(seg_R.L)]
        rS.append(seg_R.get_SR(seg_R.L-1))

        assert npc.norm(seg_L.get_SR(seg_L.L-1) - rS[0]) < 1.e-12

        cp = MPS(l_sites + r_sites, lA + rB, lS + rS, 'segment',
                 ['A'] * seg_L.L + ['B'] * seg_R.L, inf_L.norm)
        cp.grouped = inf_L.grouped
        cp.canonical_form_finite(cutoff=1e-15) #to strip out vanishing singular values at the interface
        return cp

    def correction(self, psi0_alpha, psi0_beta, env_alpha, env_beta,
                   last, eta_R_alpha, eta_R_beta):
        # 'last' is last site in segment
        correction = 0
        if psi0_alpha.finite:
            sites = range(last, psi0_alpha.L-1)
        else:
            sites = range(0, psi0_alpha.L)
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
                          npc.tensordot(rho_alpha, RP_alpha, axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*'])) - \
                          eta_R_beta + eta_R_alpha
        return correction

    def arbitrary_shifts(self, psi0_alpha, psi0_beta):
        # Code taken from MPO transfer matrix
        dtype = np.promote_types(psi0_alpha.dtype,
             np.promote_types(psi0_beta, self.model_orig.H_MPO.dtype))
        wL = self.model.H_MPO.get_W(0).get_leg('wL')
        wR = wL.conj()
        IdL = self.model_orig.H_MPO.get_IdL(0)
        IdR = self.model_orig.H_MPO.get_IdR(-1)

        vR = psi0_alpha.get_B(psi0_alpha.L-1, 'B').get_leg('vR')
        vL = psi0_alpha.get_B(0, 'A').get_leg('vL')

        chi0 = vR.ind_len
        eye_R = npc.diag(1., vR.conj(), dtype=dtype, labels=['vL', 'vL*'])
        E_shift = eye_R.add_leg(wL, IdL, axis=1, label='wL')  # vL wL vL*
        proj_trace = E_shift.conj().iset_leg_labels(['vR', 'wR', 'vR*']) / chi0
        #MPO_TM = MPOTransferMatrix(H, psi0_L, transpose=False, guess = self.init_env_data_L['init_RP'])
        eta_R_alpha = npc.tensordot(proj_trace, self.init_env_data_L['init_RP'], axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*'])).real

        chi0 = vL.ind_len
        eye_L = npc.diag(1., vL, dtype=dtype, labels=['vR*', 'vR'])
        E_shift = eye_L.add_leg(wR, IdR, axis=1, label='wR')  # vR* wR vR
        proj_trace = E_shift.conj().iset_leg_labels(['vL*', 'wL', 'vL']) / chi0
        #MPO_TM = MPOTransferMatrix(H, psi0_L, transpose=True, guess = self.init_env_data_L['init_LP'])
        eta_L_alpha = npc.tensordot(self.init_env_data_L['init_LP'], proj_trace, axes=(['vR*', 'wR', 'vR'], ['vL*', 'wL', 'vL'])).real

        vR = psi0_beta.get_B(psi0_beta.L-1, 'B').get_leg('vR')
        vL = psi0_beta.get_B(0, 'A').get_leg('vL')

        chi0 = vR.ind_len
        eye_R = npc.diag(1., vR.conj(), dtype=dtype, labels=['vL', 'vL*'])
        E_shift = eye_R.add_leg(wL, IdL, axis=1, label='wL')  # vL wL vL*
        proj_trace = E_shift.conj().iset_leg_labels(['vR', 'wR', 'vR*']) / chi0
        #MPO_TM = MPOTransferMatrix(H, psi0_R, transpose=False, guess = self.init_env_data_R['init_RP'])
        eta_R_beta = npc.tensordot(proj_trace, self.init_env_data_R['init_RP'], axes=(['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*'])).real

        chi0 = vL.ind_len
        eye_L = npc.diag(1., vL, dtype=dtype, labels=['vR*', 'vR'])
        E_shift = eye_L.add_leg(wR, IdR, axis=1, label='wR')  # vR* wR vR
        proj_trace = E_shift.conj().iset_leg_labels(['vL*', 'wL', 'vL']) / chi0
        #MPO_TM = MPOTransferMatrix(H, psi0_R, transpose=True, guess = self.init_env_data_R['init_LP'])
        eta_L_beta = npc.tensordot(self.init_env_data_R['init_LP'], proj_trace, axes=(['vR*', 'wR', 'vR'], ['vL*', 'wL', 'vL'])).real

        return eta_L_alpha, eta_L_beta, eta_R_alpha, eta_R_beta

    def get_reference_energy(self, psi0_alpha, psi0_beta):
        """Obtain ground state reference energy.

        Excitation energies are full contractions of the MPOEnvironment with the environments
        defined in :attr:`init_env_data`.
        Hence, the reference energy is also the contraction of the `MPOEnvionment` on the segment.

        Parameters
        ----------
        psi0_seg : :class:`~tenpy.networks.msp.MPS`
            Ground state MPS on the segment, matching :attr:`init_env_data`.
        """
        # can't just use gs_data['energy'], since this is just energy density for infinite MPS
        self.logger.info("Calculate reference energy by contracting environments")
        first, last = self.results['segment_first_last']
        print(first, last)
        seg_alpha = psi0_alpha.extract_segment(first, last)
        seg_beta = psi0_beta.extract_segment(first, last)
        gauge = self.options.get('gauge', 'rho')

        # This is expensive but more accurate than E0 + epsilon*L
        env_alpha = MPOEnvironment(seg_alpha, self.model.H_MPO, seg_alpha, **self.env_data_L)
        E_alpha = env_alpha.full_contraction(0).real
        env_beta = MPOEnvironment(seg_beta, self.model.H_MPO, seg_beta, **self.env_data_R)
        E_beta = env_beta.full_contraction(0).real

        coeff_alpha = self.options.get('coeff_alpha', 1.)
        coeff_beta = self.options.get('coeff_beta', 1 - coeff_alpha)
        assert np.abs(coeff_alpha + coeff_beta - 1.0) < 1.e-12

        if psi0_alpha.finite:
            correction = self.correction(psi0_alpha, psi0_beta, env_alpha, env_beta, last, 0, 0)

            self.results['ground_state_energy'] = coeff_alpha * E_alpha + coeff_beta * E_beta + correction
        else:
            H = self.model_orig.H_MPO
            if (last + 1 - first) % psi0_alpha.L == 0: # last is included in segment.
                _, epsilon_alpha, E0_alpha = MPOTransferMatrix.find_init_LP_RP(H, psi0_alpha, first, last,
                                guess_init_env_data=self.init_env_data_L, calc_E=True, _subtraction_gauge=gauge)
                epsilon_alpha = np.mean(epsilon_alpha).real
                _, epsilon_beta, E0_beta = MPOTransferMatrix.find_init_LP_RP(H, psi0_beta, first, last,
                                guess_init_env_data=self.init_env_data_R, calc_E=True, _subtraction_gauge=gauge)
                epsilon_beta = np.mean(epsilon_beta).real
            else:
                epsilon_alpha, E0_alpha, epsilon_beta, E0_beta = 0, 0, 0, 0

            #E_alpha2 = E0_alpha + (seg_alpha.L + first % psi0_L.L + psi0_L.L - (1 + (last) % psi0_L.L))*epsilon_alpha
            #E_beta2 = E0_beta + (seg_beta.L + first % psi0_R.L + psi0_R.L - (1 + (last) % psi0_R.L))*epsilon_beta
            E_alpha2 = E0_alpha + seg_alpha.L * epsilon_alpha
            E_beta2 = E0_beta + seg_beta.L * epsilon_beta

            self.logger.info("E_alpha, E_beta, E_alpha2, E_beta2: %.14f, %.14f, %.14f, %.14f", E_alpha, E_beta, E_alpha2, E_beta2)
            self.logger.info("epsilon_alpha, epsilon_beta, E0_alpha, E0_beta: %.14f, %.14f, %.14f, %.14f", epsilon_alpha, epsilon_beta, E0_alpha, E0_beta)

            eta_L_alpha, eta_L_beta, eta_R_alpha, eta_R_beta = self.arbitrary_shifts(psi0_alpha, psi0_beta)

            self.logger.info("eta_L_alpha, eta_R_alpha, eta_L_beta, eta_R_beta: %.14f, %.14f, %.14f, %.14f", eta_L_alpha, eta_R_alpha, eta_L_beta, eta_R_beta)

            correction = self.correction(psi0_alpha, psi0_beta, env_alpha, env_beta, last, eta_R_alpha, eta_R_beta) / psi0_alpha.L

            self.results['ground_state_energy'] = coeff_alpha * E_alpha + coeff_beta * E_beta \
                + (1 - coeff_alpha) * eta_L_alpha - coeff_alpha * eta_R_alpha - coeff_beta * eta_L_beta + (1 - coeff_beta) * eta_R_beta + correction

            if np.abs(E0_alpha - E0_beta) > 1.e-4:
                warnings.warn('E0_alpha and E0_beta are more than 1.e-4 idfferent; single DW energy may not be well defined.\nOnly two DWs are well defined. PROCEED AT YOUR OWN RISK.')

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

    Attributes
    ----------
    sim : :class:`OrthogonalExcitations`
        Simulation class for which to initial a state to be used as excitation initial state.
    """
    def __init__(self, sim, options):
        self.sim = sim
        self.options = asConfig(options, self.__class__.__name__)
        self.options.setdefault('method', 'from_orthogonal')
        model_dtype = getattr(sim.model, 'dtype', np.float64)
        super().__init__(sim.model.lat, options, model_dtype)

    def from_orthogonal(self):
        """Default initialization for :class:`OrthogonalExcitations` from previous states.

        We want to orthogonalize against the ground state to find the next excited state.
        While formally orthogonal, perturbing the ground state a little bit in a random direction
        should lead to a significant overlap with the first excited state, much better than another
        random guess.
        Hence, the strategy of this method is to use either
        the last excited state we have (for `use_highest_excitation` = True),
        or the initial ground state segment (`use_highest_excitation` = False),
        and just perturb it a little bit with :meth:`OrthogonalExcitations.perturb_initial_state`.

        Options
        -------
        .. cfg:configoptions :: ExcitationInitialState
            :include: InitialStateBuilder

            use_highest_excitation : bool
                If True, start from  the last state in :attr:`orthogonal_to` and perturb it.
                If False, use :attr:`OrthogonalExcitations.initial_state_seg`,
                i.e.,  a perturbation of the ground state in the right charge sector.
        """

        if self.options.get('use_highest_excitation', True) and len(self.sim.orthogonal_to) > 0:
            psi = self.sim.orthogonal_to[-1]
            perturb = True
        else:
            psi = self.sim.initial_state_seg
            perturb = False  # was already perturbed
        if isinstance(psi, dict):
            psi = psi['ket']
        psi = psi.copy()
        if perturb:
            self.sim.perturb_initial_state(psi)
        return psi

    def from_file(self):
        """Load the initial state from an exisiting file of a previous excitation simulation.

        Options
        -------
        .. cfg:configoptions :: InitialStateBuilder

            filename : str
                The filename from which to load the state.
            data_key_orthogonal : str
                Key within the file to be used for loading the orthogonal_to states.
                Can be recursive (separated by '/'), see :func:`tenpy.tools.misc.get_recursive`.
            perturb : bool
                Defaults to False. If True, still perturb the state with
                :meth:`OrthogonalExcitations.perturb_initial_state` based on the
                :cfg:option:`OrthogonalExcitations.initial_state_randomize_params`.
        """
        previous_ortho = getattr(self, '_previous_ortho', None)
        if previous_ortho is None:
            # load data from previous file
            filename = self.options['filename']
            key_ortho = self.options.get('data_key_orthogonal_to', "resume_data/orthogonal_to")
            key_offset = self.options.get('data_key_ortho_offset', "resume_data/ortho_offset")
            self.logger.info("loading previous states from %r, keys %r and %r",
                             filename, key_ortho, key_offset)
            data = hdf5_io.load(filename)
            self._previous_ortho = previous_ortho = data['resume_data']['orthogonal_to']
            self._previous_offset = data['resume_data']['offset']
            self._previous_first_last = data['segment_first_last']
        # else: we already loaded the corresponding data
        psi = previous_ortho[self._previous_offset + len(self.sim.excitations)]
        perturb = self.options.get('perturb', False)
        psi = self.sim._psi_from_previous_ortho(psi.copy(),
                                                self._previous_first_last,
                                                self.options['filename'],
                                                perturb)
        return psi
