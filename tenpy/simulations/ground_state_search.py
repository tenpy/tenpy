"""Simulations for ground state searches."""
# Copyright 2020-2021 TeNPy Developers, GNU GPLv3

import numpy as np
from pathlib import Path

from . import simulation
from ..tools import hdf5_io
from .simulation import *
from ..linalg import np_conserved as npc
from ..models.model import Model
from ..networks.mpo import MPOEnvironment, MPOTransferMatrix
from ..networks.mps import MPS, InitialStateBuilder
from ..algorithms.mps_common import ZeroSiteH
from ..algorithms.dmrg import TwoSiteDMRGEngine
from ..linalg import lanczos
from ..linalg.sparse import SumNpcLinearOperator
from ..tools.misc import find_subclass, to_iterable
from ..tools.params import asConfig

import copy

__all__ = simulation.__all__ + [
    'GroundStateSearch',
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
        Only used to pass `gs_data` to :meth:`init_orthogonal_from_groundstate`;
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
    def __init__(self, options, *, orthogonal_to=None, gs_data=None, **kwargs):
        super().__init__(options, **kwargs)
        resume_data = kwargs.get('resume_data', {})
        if orthogonal_to is None and 'orthogonal_to' in resume_data:
            orthogonal_to = kwargs['resume_data']['orthogonal_to']
            self.options.touch('groundstate_filename')
        self.orthogonal_to = orthogonal_to
        self.excitations = resume_data.get('excitations', [])
        self.results['excitation_energies'] = []
        self.results['excitation_energies_MPO'] = []
        if self.options.get('save_psi', True):
            self.results['excitations'] = self.excitations
        self.init_env_data = {}
        self._gs_data = gs_data
        self.initial_state_builder = None

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
        self.model_orig = self.model

        # intialize original state
        self.ground_state_orig = psi0 = gs_data['psi']  # no copy!
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

        # switch_charge_sector defines `self.initial_state_seg`
        self.initial_state_seg, self.qtotal_diff = self.switch_charge_sector(psi0_seg)

        if any(self.qtotal_diff):
            self.orthogonal_to = []  # different charge sector
            # so orthogonal to gs due to charge conservation
        else:
            self.orthogonal_to = [psi0_seg]
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
                                                         guess_init_env_data)
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
                                             'ExcitationInitialState')
            params = self.options.subconfig('initial_state_params')
            Builder = find_subclass(InitialStateBuilder, builder_class)
            if issubclass(Builder, ExcitationInitialState):
                # incompatible with InitialStateBuilder: need to pass `sim` to __init__
                self.initial_state_builder = Builder(self, params)
            else:
                self.initial_state_builder = Builder(self.model.lat, params, self.model.dtype)

        self.psi = self.initial_state_builder.run()

        if self.options.get('save_psi', True):
            self.results['psi'] = self.psi

    def perturb_initial_state(self, psi, canonicalize=True):
        """(Slightly) perturb an initial state.

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
                In addition, to those parameters, we read out the arguments `close_1` for
                :meth:`~tenpy.networks.mps.MPS.perturb`.

        """
        randomize_params = self.options.subconfig('initial_state_randomize_params')
        close_1 = randomize_params.get('close_1', True)
        psi.perturb(randomize_params, close_1=close_1, canonicalize=True)

    def init_algorithm(self, **kwargs):
        kwargs.setdefault('orthogonal_to', self.orthogonal_to)
        resume_data = kwargs.setdefault('resume_data', {})
        resume_data['init_env_data'] = self.init_env_data
        super().init_algorithm(**kwargs)

    # group_sites_for_algorithm should be fine!

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
                                 "is not in segment [{first:d}, {last:d}]!")
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
        norm = npc.norm(th0)
        self.logger.info("Norm of theta guess: %.8f", npc.norm(th0))
        if np.isclose(norm, 0):
            raise ValueError(f"Norm of inserted theta with charge {list(qtotal_change)} on site index {site:d} is zero.")
        
        U, s, Vh = npc.svd(th0, inner_labels=['vR', 'vL'])
        psi.set_B(site-1, npc.tensordot(psi.get_B(site-1, 'A'), U, axes=['vR', 'vL']), form='A')
        psi.set_B(site, npc.tensordot(Vh, psi.get_B(site, 'B'), axes=['vR', 'vL']), form='B')
        psi.set_SL(site, s)

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
            E_shift = lanczos_params['E_shift'] if lanczos_params['E_shift'] is not None else 0

            print("E_shift", E_shift)
            self.logger.info("Shifted ground state energy: %.14f", ground_state_energy + 0.5 * E_shift)

            if self.engine.diag_method != 'lanczos' or \
                    ground_state_energy + 0.5 * E_shift > 0:
                # the factor of 0.5 is somewhat arbitrary, to ensure that
                # also excitations have energy < 0
                print("lanczos_params['E_shift']:", lanczos_params['E_shift'])
                raise ValueError("You need to set use diag_method='lanczos' and small enough "
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

    def prepare_results_for_save(self):
        results = super().prepare_results_for_save()
        if 'resume_data' in results:
            results['resume_data']['excitations'] = self.excitations
        return results


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
            psi, meas_first, meas_last = psi.extract_enlarged_segment(self.ground_state_orig,
                                                                      self.ground_state_orig,
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


def expectation_value_outside_segment_right(psi_segment, psi_R, ops, lat_segment, sites=None, axes=None):
    """Calculate expectation values outside of the segment to the right.

    Parameters
    ----------
    psi_S :
        Segment MPS.
    psi_R :
        Inifnite MPS on the right.
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
    psi_R :
        Inifnite MPS on the right.
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
    # TODO this is probably broken right now.
    # TODO adjust this to match OrthgonalExciations methods/arguments/return values again
    def init_orthogonal_from_groundstate(self):
        """Initialize :attr:`orthogonal_to` from the ground state.

        Load the ground state.
        If the ground state is infinite, call :meth:`extract_segment_from_infinite`.

        An empty :attr:`orthogonal_to` indicates that we will :meth:`switch_charge_sector`
        in the first :meth:`init_algorithm` call.

        Options
        -------
        .. cfg:configoptions :: OrthogonalExcitations

            left_BC_filename :
                File from which the ground state for left boundary should be loaded.
            right_BC_filename :
                File from which the ground state for right boundary should be loaded.
            orthogonal_norm_tol : float
                Tolerance how large :meth:`~tenpy.networks.mps.MPS.norm_err` may be for states
                to be added to :attr:`orthogonal_to`.
            segment_enlarge, segment_first, segment_last : int | None
                Only for initially infinite ground states.
                Arguments for :meth:`~tenpy.models.lattice.Lattice.extract_segment`.
            join_method: "average charge" | "most probable charge"
                Governs how the segements are joined.
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
        left_fn = self.options['left_BC_filename']
        right_fn = self.options['right_BC_filename']
        left_data = hdf5_io.load(left_fn)
        right_data = hdf5_io.load(right_fn)
        left_data_options = left_data['simulation_parameters']
        right_data_options = right_data['simulation_parameters']

        # get model from ground_state data
        for keyL, keyR in zip(left_data_options.keys(), right_data_options.keys()):
            assert keyL == keyR, 'Left and right models must have the same keys.'
            if not isinstance(keyL, str) or not keyL.startswith('model'):
                continue
            if keyL not in self.options:
                # I think this forces the left and right model to be the same? Maybe we want a case where we put a DW between two different types of states?
                self.options[keyL] = {}
                self.options[keyL]['left'] = left_data_options[keyL]
                self.options[keyR]['right'] = right_data_options[keyR]
        self.init_model()
        # FOR NOW (09/17/2021), WE ASSUME LEFT AND RIGHT MODELS ARE THE SAME

        self.ground_state_infinite_right = psi0_R = right_data['psi'] # Use right BC psi since these should be in B form already (Actually both are probs in B form)
        self.ground_state_infinite_left = psi0_L = left_data['psi']
        resume_data = right_data.get('resume_data', {})
        if np.linalg.norm(psi0_R.norm_test()) > self.options.get('orthogonal_norm_tol', 1.e-12):
            self.logger.info("call psi.canonicalf_form() on right ground state")
            psi0_R.canonical_form()
        if np.linalg.norm(psi0_L.norm_test()) > self.options.get('orthogonal_norm_tol', 1.e-12):
            self.logger.info("call psi.canonicalf_form() on left ground state")
            psi0_L.canonical_form()
        assert psi0_R.bc == psi0_L.bc == 'infinite', 'Topological excitations require segment DMRG, so infinite boundary conditions.'

        write_back = self.extract_segment_from_infinite(resume_data)
        if write_back:
            self.write_converged_environments(left_data, right_data, left_fn, right_fn)

        apply_local_op = self.options.get("apply_local_op", None)
        switch_charge_sector = self.options.get("switch_charge_sector", None)

        self.orthogonal_to = []
        return right_data

    def init_model(self):
        """Initialize a :attr:`model` from the model parameters.

        Skips initialization if :attr:`model` is already set.

        Options
        -------
        .. cfg:configoptions :: Simulation
            model_class : str | class
                Mandatory. Class or name of a subclass of :class:`~tenpy.models.model.Model`.
            model_params : dict
                Dictionary with parameters for the model; see the documentation of the
                corresponding `model_class`.
        """
        # TODO: does this make sense???? should have the same model on left and right!
        for dir in ['left', 'right']:
            model_class_name = self.options["model_class"][dir]  # no default value!
            if hasattr(self, 'model' + '_' + dir + '_inf'):
                self.options.subconfig('model_params').touch(dir)
                return  # skip actually regenerating the model
            ModelClass = find_subclass(Model, model_class_name)
            params = self.options.subconfig('model_params').subconfig(dir)
            if dir == 'left':
                self.model_left_inf = ModelClass(params)
            else:
                self.model_right_inf = ModelClass(params)

    def extract_segment_from_infinite(self, resume_data):
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

        psi0_L_inf, psi0_R_inf, model_L_inf, model_R_inf = self.ground_state_infinite_left, self.ground_state_infinite_right, \
                                                            self.model_left_inf, self.model_right_inf
        enlarge = self.options.get('segment_enlarge', None)
        first = self.options.get('segment_first', 0)
        last = self.options.get('segment_last', None)
        self.model_right = model_R_inf.extract_segment(first, last, enlarge) # I am not sure either of these are acutally used.
        self.model_left = model_L_inf.extract_segment(first, last, enlarge)  # I am not sure either of these are acutally used.
        self.model = self.model_right # TODO: using right BCs model for the segment; Different model all-together?
        first, last = self.model.lat.segment_first_last
        write_back = self.options.get('write_back_converged_ground_state_environments', False)
        if False: #resume_data.get('converged_environments', False):
            # [TODO] currently not writing converged environments to ground state files
            self.logger.info("use converged environments from ground state file")
            env_data = resume_data['init_env_data']
            psi0_inf = resume_data.get('psi', psi0_inf)
            write_back = False
        else:
            self.logger.info("converge environments with MPOTransferMatrix")
            guess_init_env_data = resume_data.get('init_env_data', None)
            H_R = model_R_inf.H_MPO
            env_data_R, self.eps_R, self.E0_R = MPOTransferMatrix.find_init_LP_RP(H_R, psi0_R_inf, first, last,
                                                         guess_init_env_data, calc_E=True)
            self.init_env_data_R = env_data_R

            H_L = model_L_inf.H_MPO
            env_data_L, self.eps_L, self.E0_L = MPOTransferMatrix.find_init_LP_RP(H_L, psi0_L_inf, first, last,
                                                         guess_init_env_data, calc_E=True)
            self.init_env_data_L = env_data_L

            env_data_mixed = {
                'init_LP': env_data_L['init_LP'],
                'init_RP': env_data_R['init_RP'],
                'age_LP': 0,
                'age_RP': 0
                }
        self.init_env_data = env_data_mixed
        self.ground_state_right = psi0_R_inf.extract_segment(first, last) # I am not sure either of these are acutally used.
        self.ground_state_left = psi0_L_inf.extract_segment(first, last) # I am not sure either of these are acutally used.
        self.ground_state, self._boundary = self.extract_segment_mixed_BC(first, last)

        return write_back

    def extract_segment_mixed_BC(self, first, last):
        join_method = self.join_method = self.options.get('join_method', "average charge")

        lL = self.ground_state_infinite_left.L
        rL = self.ground_state_infinite_right.L
        assert rL == lL, "Ground state boundary conditions must have the same unit cell length."
        gsl = self.ground_state_infinite_left
        gsr = self.ground_state_infinite_right

        # Get boundary indices for left and right half of segment
        num_segments = (last+1 - first) // lL
        lsegments = num_segments // 2
        rsegments = num_segments - lsegments
        lfirst = first
        llast = lsegments * lL - 1
        rfirst = llast + 1
        rlast = last
        assert (rlast + 1 - rfirst) // rL == rsegments

        self.logger.info("lfirst, llast, rfirst, rlast: %d, %d, %d, %d", lfirst, llast, rfirst, rlast)
        self.logger.info("first, last: %d %d", first, last)
        self.logger.info("seg_L, seg_R: %d %d", lsegments, rsegments)

        # Building segment MPS on left half
        l_sites = [gsl.sites[i % lL] for i in range(lfirst, llast + 1)]
        lA = [gsl.get_B(i, 'A') for i in range(lfirst, llast + 1)]
        #lB = [gsl.get_B(i, 'B') for i in range(lfirst, llast + 1)]
        lS = [gsl.get_SL(i) for i in range(lfirst, llast + 1)]
        lS.append(gsl.get_SR(llast))
        left_segment = MPS(l_sites, lA, lS, 'segment', 'A', gsl.norm)

        # Building segment MPS on right half
        r_sites = [gsr.sites[i % rL] for i in range(rfirst, rlast + 1)]
        rB = [gsr.get_B(i) for i in range(rfirst, rlast + 1)]
        rS = [gsr.get_SL(i) for i in range(rfirst, rlast + 1)]
        rS.append(gsr.get_SR(rlast))
        right_segment = MPS(r_sites, rB, rS, 'segment', 'B', gsr.norm)

        # [TODO] Double check on how first and last should be used when we are offsetting the unit cell
        left_half_model = self.model_left_inf.extract_segment(first, None, lsegments)
        right_half_model = self.model_right_inf.extract_segment(first, None, rsegments) # should this be rfirst? Does it make a difference?

        env_left_BC = MPOEnvironment(left_segment, left_half_model.H_MPO, left_segment, **self.init_env_data_L)
        env_right_BC = MPOEnvironment(right_segment, right_half_model.H_MPO, right_segment, **self.init_env_data_R)
        LP = env_left_BC._contract_LP(llast, env_left_BC.get_LP(llast, store=False))
        RP = env_right_BC._contract_RP(0, env_right_BC.get_RP(0, store=False))  # saves the environments!
        H0 = ZeroSiteH.from_LP_RP(LP, RP)
        if self.model.H_MPO.explicit_plus_hc:
            H0 = SumNpcLinearOperator(H0, H0.adjoint())
        vL, vR = LP.get_leg('vR').conj(), RP.get_leg('vL').conj()

        if left_segment.chinfo.qnumber == 0:    # Handles the case of no charge-conservation
            desired_Q = None
        else:
            if join_method == "average charge":
                Q_bar_L = self.ground_state_infinite_left.average_charge(0)
                for i in range(1, self.ground_state_infinite_left.L):
                    Q_bar_L += self.ground_state_infinite_left.average_charge(i)
                Q_bar_L = vL.chinfo.make_valid(np.around(Q_bar_L / self.ground_state_infinite_left.L))
                self.logger.info("Charge of left BC, averaged over site and unit cell: %r", Q_bar_L)

                Q_bar_R = self.ground_state_infinite_right.average_charge(0)
                for i in range(1, self.ground_state_infinite_right.L):
                    Q_bar_R += self.ground_state_infinite_right.average_charge(i)
                Q_bar_R = vR.chinfo.make_valid(-1 * np.around(Q_bar_R / self.ground_state_infinite_right.L))
                self.logger.info("Charge of right BC, averaged over site and unit cell: %r", -1*Q_bar_R)
                desired_Q = list(vL.chinfo.make_valid(Q_bar_L + Q_bar_R))
            elif join_method == "most probable charge":
                posL = left_segment.L
                posR = 0
                QsL, psL = left_segment.probability_per_charge(posL)
                QsR, psR = right_segment.probability_per_charge(posR)

                self.logger.info(side_by_side)
                side_by_side = vert_join(["left seg\n" + str(QsL), "prob\n" + str(np.array([psL]).T), "right seg\n" + str(QsR),"prob\n" +str(np.array([psR]).T)], delim=' | ')
                self.logger.info(side_by_side)

                Qmostprobable_L = QsL[np.argmax(psL)]
                Qmostprobable_R = -1 * QsR[np.argmax(psR)]
                self.logger.info("Most probable left:" + str(Qmostprobable_L))
                self.logger.info("Most probable right:" + str(Qmostprobable_R))
                desired_Q = list(vL.chinfo.make_valid(Qmostprobable_L + Qmostprobable_R))
            else:
                raise ValueError("Invalid `join_method` %s " % join_method)

        self.logger.info("Desired gluing charge: %r", desired_Q)

        # We need a tensor that is non-zero only when Q = (Q^i_L - bar(Q_L)) + (Q^i_R - bar(Q_R))
        # Q is the the charge we insert. Here we only do charge gluing to get a valid segment.
        # Changing charge sector is done below by basically identical code when the segment is already formed.

        th0 = npc.Array.from_func(np.ones, [vL, vR],
                                  dtype=left_segment.dtype,
                                  qtotal=desired_Q,
                                  labels=['vL', 'vR'])
        lanczos_params = self.options.get("lanczos_params", {}) # See if lanczos_params is in yaml, if not use empty dictionary
        _, th0, _ = lanczos.LanczosGroundState(H0, th0, lanczos_params).run()
        U, s, Vh = npc.svd(th0, inner_labels=['vR', 'vL'])
        left_segment.set_B(llast, npc.tensordot(left_segment.get_B(llast, 'A'), U, axes=['vR', 'vL']), form='A') # Put AU into last site of left segment
        lA[llast] = left_segment.get_B(llast, 'A')
        right_segment.set_B(0, npc.tensordot(Vh, right_segment.get_B(0, 'B'), axes=['vR', 'vL']), form='B') # Put Vh B into first site of right segment
        right_segment.set_SL(0, s)

        rB[0] = right_segment.get_B(0)
        rS[0] = right_segment.get_SL(0)
        lS = lS[0:-1] # Remove last singular values from list of singular values in A part of segment.
        ##################### BIG OL HACK #####################

        # note: __init__ makes deep copies of B, S
        cp = MPS(l_sites + r_sites, lA + rB, lS + rS, 'segment',
                 ['A'] * (llast + 1 - lfirst) + ['B'] * (rlast + 1 - rfirst), gsl.norm)
        cp.grouped = gsl.grouped
        cp.canonical_form_finite(cutoff=1e-15) #to strip out vanishing singular values at the interface
        # TODO: no longer define `self._boundary, so shouldn't return `rfirst`.
        return cp, rfirst


    def write_converged_environments(self, left_data, right_data, left_fn, right_fn):
        """Write converged environments back into the file with the ground state.

        Parameters
        ----------
        gs_data : dict
            Data loaded from the ground state file.
        gs_fn : str
            Filename where to save `gs_data`.
        """
        raise NotImplementedError("TODO")
        # make sure to avoid issues if the left file(name) is the same as the right!

    def ground_state_segment_energy(self):
        """Calculate the energy of the segment formed from tensors of one ground state or the other.
        DO NOT USE MIXED SEGMENT.
        An analogue of this function could be moved to orthogonal_excitations, as we need to do the same thing there."""
        # TODO: renamed this to the same `get_reference_energy` as in OrthogonalExciations
        self.logger.info("Calculate energy of 'vacuum' segment.")

        # [TODO] optimize this by using E_L = E_L^0 + epsilon*L where E_L^0 = LP_L * s^2 * RP_L
        # The two answers are not always the same, so we go with full contraction as the true value.
        env_left_BC = MPOEnvironment(self.ground_state_left, self.model_left.H_MPO, self.ground_state_left, **self.init_env_data_L)
        E_L = env_left_BC.full_contraction(0)
        E_L_2 = self.E0_L + self.eps_L * self.ground_state_left.L

        env_right_BC = MPOEnvironment(self.ground_state_right, self.model_right.H_MPO, self.ground_state_right, **self.init_env_data_R)
        E_R = env_right_BC.full_contraction(0)
        E_R_2 = self.E0_R + self.eps_R * self.ground_state_right.L

        self.logger.info("EL, ER, EL2, ER2: %.14f, %.14f, %.14f, %.14f", E_L, E_R, E_L_2, E_R_2)
        self.logger.info("epsilon_L, epsilon_R, E0_L, E0_R: %.14f, %.14f, %.14f, %.14f", self.eps_L, self.eps_R, self.E0_L, self.E0_R)

        self.results['ground_state_energy'] = (E_L + E_R)/2
        return


    def switch_charge_sector(self):
        """Change the charge sector of :attr:`psi` in place."""

        self.ground_state_segment_energy() # Here we calculate the ground state energy in all cases.
        apply_local_op = self.options.get("apply_local_op", None)
        switch_charge_sector = self.options.get("switch_charge_sector", None)
        if apply_local_op is None and switch_charge_sector is None:
            return
        # This should be the exact same as the switch_charge_sector() function in the parent class? Can we just do super().switch_charge_sector()
        if self.psi.chinfo.qnumber == 0:
            raise ValueError("can't switch charge sector with trivial charges!")
        self.logger.info("switch charge sector of the ground state "
                         "[contracts environments from rinit_algorithmight]")
        site = self.options.get("switch_charge_sector_site", self._boundary)
        self.logger.info("Changing charge to the left of site: %d", site)
        qtotal_before = self.psi.get_total_charge()
        self.logger.info("Charges of the original segment: %r", list(qtotal_before))

        env = self.engine.env

        if apply_local_op is not None:
            if switch_charge_sector is not None:
                raise ValueError("give only one of `switch_charge_sector` and `apply_local_op`")
            local_ops = [(int(apply_local_op[i]),str(apply_local_op[i+1])) for i in range(0,len(apply_local_op),2)]
            self.logger.info("Applying local ops: %s" % str(local_ops))
            site0 = local_ops[0][0] if len(local_ops) > 0 else 1
            # self.results['ground_state_energy'] = env.full_contraction(site0) #pretty sure this is wrong, since we compute it earlier by a better way in `glue_charge_sectors`
            # for i in range(0, site0 - 1): # TODO shouldn't we delete RP(i-1)
            #     env.del_RP(i)
            # for i in range(site0 + 1, env.L):
            #     env.del_LP(i)
            env.clear()
            #apply_local_op['unitary'] = True  # no need to call psi.canonical_form
            for (site,op_string) in local_ops:
                self.logger.info("Now applying: (%i, %s)"% (site, op_string))
                self.psi.apply_local_op(site,op_string,unitary=True) #don't canonicalize in here, we call it below.
        else:
            assert switch_charge_sector is not None
            # get the correct environments on site 0
            # SAJANT, 09/15/2021 - Change 0 -> site so that we can insert in the middle of the segment
            LP = env.get_LP(site)
            RP = env._contract_RP(site, env.get_RP(site, store=True))  # saves the environments!
            #self.results['ground_state_energy'] = env.full_contraction(site)
            for i in range(site + 1, site + self.engine.n_optimize):      # SAJANT, 09/15/2021 - what do I delete when site!=0? I just shift the range by site.
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
            U, s, Vh = npc.svd(th0, inner_labels=['vR', 'vL'])
            self.psi.set_B(site-1, npc.tensordot(self.psi.get_B(site-1, 'A'), U, axes=['vR', 'vL']), form='A')
            self.psi.set_B(site, npc.tensordot(Vh, self.psi.get_B(site, 'B'), axes=['vR', 'vL']), form='B')
            self.psi.set_SL(site, s)
            #th0 = npc.tensordot(th0, self.psi.get_B(site, 'B'), axes=['vR', 'vL'])
            #self.psi.set_B(site, th0, form='Th')
        self.psi.canonical_form_finite(cutoff=1e-15,envs_to_update=[env]) #to strip out vanishing singular values at the interface
        qtotal_after = self.psi.get_total_charge()
        qtotal_diff = self.psi.chinfo.make_valid(qtotal_after - qtotal_before)
        self.logger.info("changed charge by %r compared to previous state", list(qtotal_diff))
        # assert not np.all(qtotal_diff == 0)


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

        use_highest_excitation : bool
            If True, start from  the last state in :attr:`orthogonal_to` and perturb it.
            If False, use :attr:`OrthogonalExcitations.initial_state_seg` (= a perturbation of the
            ground state in the right charge sector).

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
