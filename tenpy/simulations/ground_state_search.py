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
from ..linalg import lanczos
from ..linalg.sparse import SumNpcLinearOperator
from ..tools.misc import find_subclass
from ..tools.params import asConfig

__all__ = simulation.__all__ + [
    'GroundStateSearch',
    'OrthogonalExcitations',
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
                apply_local_op should have the form `[(site1,operator_string1),(site2,operator_string2),...]`
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
            self.results['ground_state_energy'] = data['energy']       # BLG_DMRG states do not have an 'energy' key, but we won't be doing finite BLG_DMRG

        apply_local_op = self.options.get("apply_local_op", None)
        switch_charge_sector = self.options.get("switch_charge_sector", None)
        if apply_local_op is None and switch_charge_sector is None:
            self.orthogonal_to = [self.ground_state]
            # This is not correct; it should be energy of the entire segment, not just the ground state energy per site
            # self.results['ground_state_energy'] = data['E_dmrg'] #E0    # SAJANT, 09/08/21 - I don't think I want to do this since the energy is being calculated somehow anyway.
            #if psi0.bc == 'infinite':
            #    temp_env = MPOEnvironment(psi0, self.model.H_MPO, psi0, **self.init_env_data)
            #    self.results['ground_state_energy'] = temp_env.full_contraction(0)
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
        self.boundary = enlarge // 2 * psi0_inf.L   # Bond at the middle of the segment, aligned with a cylinder ring

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
        # print("kwargs:")
        # print(kwargs)
        if len(self.orthogonal_to) == 0:
            self.switch_charge_sector()
        
        # Sajant, 09/08/2021 - get ground state energy via full contraction if it doesn't exist
        # This only occurs when we are orthogonalizing against the ground state (no switch_charge_sector 
        if 'ground_state_energy' not in self.results.keys():
            self.results['ground_state_energy'] = self.engine.env.full_contraction(0)
            # print(vars(self.engine.env))
            # print(self.results['ground_state_energy'])
            print("Getting GS energy since it was not in 'results' dictionary before.")
            
    def switch_charge_sector(self):
        """Change the charge sector of :attr:`psi` in place."""
        if self.psi.chinfo.qnumber == 0:
            raise ValueError("can't switch charge sector with trivial charges!")
        self.logger.info("switch charge sector of the ground state "
                         "[contracts environments from right]")
        apply_local_op = self.options.get("apply_local_op", None)
        switch_charge_sector = self.options.get("switch_charge_sector", None)
        site = self.options.get("switch_charge_sector_site", self.boundary)
        self.logger.info("Changing charge to the left of site: %d", site)
        qtotal_before = self.psi.get_total_charge()
        self.logger.info("Charges of the original segment: %r", list(qtotal_before))

        env = self.engine.env
        #apply_local_op should have the form [ (site1,operator_string1),(site2,operator_string2),...]
        if apply_local_op is not None:
            if switch_charge_sector is not None:
                raise ValueError("give only one of `switch_charge_sector` and `apply_local_op`")
            self.results['ground_state_energy'] = env.full_contraction(apply_local_op[0][0])
            for i in range(0, apply_local_op['i'] - 1): # TODO shouldn't we delete RP(i-1)
                env.del_RP(i)
            for i in range(apply_local_op['i'] + 1, env.L):
                env.del_LP(i)
            #apply_local_op['unitary'] = True  # no need to call psi.canonical_form
            for (site,op_string) in apply_local_op:
                self.psi.apply_local_op(site,op_string,unitary=False)
        else:
            assert switch_charge_sector is not None
            # get the correct environments on site 0
            # SAJANT, 09/15/2021 - Change 0 -> site so that we can insert in the middle of the segment
            LP = env.get_LP(site)
            RP = env._contract_RP(site, env.get_RP(site, store=True))  # saves the environments!
            self.results['ground_state_energy'] = env.full_contraction(site)
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
            th0 = npc.tensordot(th0, self.psi.get_B(site, 'B'), axes=['vR', 'vL'])
            self.psi.set_B(site, th0, form='Th')
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
            lanczos_params = self.engine.lanczos_params
            if self.engine.diag_method == 'default': # SAJANT, 09/09/21
                self.engine.diag_method = 'lanczos'
            self.logger.info('Lanczos Params in run_algorithm: %r', lanczos_params)
            # E_shift = lanczos_params.get('E_shift', 0.) #lanczos_params['E_shift'] if lanczos_params['E_shift'] is not None else 0
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

class TopologicalExcitations(OrthogonalExcitations):
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
                self.options[keyL] = {}
                self.options[keyL]['left'] = left_data_options[keyL]
                self.options[keyR]['right'] = right_data_options[keyR]
        self.init_model() 
        # FOR NOW (09/17/2021), WE ASSUME LEFT AND RIGHT MODELS ARE THE SAME

        self.ground_state_infinite_right = psi0_R = right_data['psi'] # Use right BC psi since these should be in B form already.
        self.ground_state_infinite_left = psi0_L = left_data['psi']
        resume_data = right_data.get('resume_data', {})
        if np.linalg.norm(psi0_R.norm_test()) > self.options.get('orthogonal_norm_tol', 1.e-12):
            self.logger.info("call psi.canonicalf_form() on ground state")
            psi0_R.canonical_form()
        assert psi0_R.bc == psi0_L.bc == 'infinite', 'Topological excitations require segment DMRG, so infinite boundary conditions.'

        write_back = self.extract_segment_from_infinite(resume_data)
        if write_back:
            self.write_converged_environments(left_data, right_data, left_fn, right_fn)

        apply_local_op = self.options.get("apply_local_op", None)
        switch_charge_sector = self.options.get("switch_charge_sector", None)
        
        #assert apply_local_op is switch_charge_sector is None, "For the moment we only search for domain wall that interpolates between the two BCs."
        
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
        #self.model = self.model_right
        
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
        #self.logger.info("first, last: %d %d", first, last)
        self.model_right = model_R_inf.extract_segment(first, last, enlarge) 
        self.model_left = model_L_inf.extract_segment(first, last, enlarge)
        self.model = self.model_right # TODO: using right BCs model for the segment; Different model all-together?
        first, last = self.model.lat.segment_first_last
        write_back = self.options.get('write_back_converged_ground_state_environments', False)
        if False: #resume_data.get('converged_environments', False):
            self.logger.info("use converged environments from ground state file")
            env_data = resume_data['init_env_data']
            psi0_inf = resume_data.get('psi', psi0_inf)
            write_back = False
        else:
            self.logger.info("converge environments with MPOTransferMatrix")
            guess_init_env_data = resume_data.get('init_env_data', None)
            H_R = model_R_inf.H_MPO
            self.eps_R, self.E0_R, env_data_R = MPOTransferMatrix.find_init_LP_RP(H_R, psi0_R_inf, first, last,
                                                         guess_init_env_data, calc_E=True)
            self.init_env_data_R = env_data_R
            
            H_L = model_L_inf.H_MPO
            self.eps_L, self.E0_L, env_data_L = MPOTransferMatrix.find_init_LP_RP(H_L, psi0_L_inf, first, last,
                                                         guess_init_env_data, calc_E=True)
            self.init_env_data_L = env_data_L
            
            env_data_mixed = {
                'init_LP': env_data_L['init_LP'],
                'init_RP': env_data_R['init_RP'],
                'age_LP': 0,
                'age_RP': 0
                }
        self.init_env_data = env_data_mixed
        #self.ground_state_infinite = self.ground_state_infinite_right = psi0_R_inf
        #self.ground_state_infinite = psi0_R_inf
        self.ground_state_right = psi0_R_inf.extract_segment(first, last)
        #self.ground_state_infinite_left = psi0_L_inf
        self.ground_state_left = psi0_L_inf.extract_segment(first, last)
        self.ground_state, self.boundary = self.extract_segment_mixed_BC_v2(first, last)

        return write_back
    
    def extract_segment_mixed_BC(self, first, last):
        lL = self.ground_state_infinite_left.L
        rL = self.ground_state_infinite_right.L
        assert rL == lL, "Ground state boundary conditions must have the same unit cell length."
        gsl = self.ground_state_infinite_left
        gsr = self.ground_state_infinite_right
        self.logger.info
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
        
        l_sites = [gsl.sites[i % lL] for i in range(lfirst, llast + 1)]
        lB = [gsl.get_B(i) for i in range(lfirst, llast + 1)]
        lS = [gsl.get_SL(i) for i in range(lfirst, llast + 1)]
        #lS.append(gsl.get_SR(llast))
        
        r_sites = [gsr.sites[i % lL] for i in range(rfirst, rlast + 1)]
        rB = [gsr.get_B(i) for i in range(rfirst, rlast + 1)]
        rS = [gsr.get_SL(i) for i in range(rfirst, rlast + 1)]
        rS.append(gsr.get_SR(rlast))
        
        # note: __init__ makes deep copies of B, S
        cp = MPS(l_sites + r_sites, lB + rB, lS + rS, 'segment', 'B', gsl.norm)
        cp.grouped = gsl.grouped
        return cp, rfirst
    
    def extract_segment_mixed_BC_v2(self, first, last):
        lL = self.ground_state_infinite_left.L
        rL = self.ground_state_infinite_right.L
        assert rL == lL, "Ground state boundary conditions must have the same unit cell length."
        gsl = self.ground_state_infinite_left
        gsr = self.ground_state_infinite_right
        self.logger.info
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
        
        l_sites = [gsl.sites[i % lL] for i in range(lfirst, llast + 1)]
        lB = [gsl.get_B(i) for i in range(lfirst, llast + 1)]
        lS = [gsl.get_SL(i) for i in range(lfirst, llast + 1)]
        lS.append(gsl.get_SR(llast))
        left_segment = MPS(l_sites, lB, lS, 'segment', 'B', gsl.norm)
        
        r_sites = [gsr.sites[i % lL] for i in range(rfirst, rlast + 1)]
        rB = [gsr.get_B(i) for i in range(rfirst, rlast + 1)]
        rS = [gsr.get_SL(i) for i in range(rfirst, rlast + 1)]
        rS.append(gsr.get_SR(rlast))
        right_segment = MPS(r_sites, rB, rS, 'segment', 'B', gsr.norm)
        
        ##################### BIG OL HACK #####################
        # [TODO] Double check on how first and last should be used when we are offsetting the unit cell
        left_half_model = self.model_left_inf.extract_segment(first, None, lsegments)
        right_half_model = self.model_right_inf.extract_segment(first, None, rsegments)

        env_left_BC = MPOEnvironment(left_segment, left_half_model.H_MPO, left_segment, **self.init_env_data_L)
        env_right_BC = MPOEnvironment(right_segment, right_half_model.H_MPO, right_segment, **self.init_env_data_R)
        LP = env_left_BC._contract_LP(llast, env_left_BC.get_LP(llast, store=False))
        RP = env_right_BC._contract_RP(0, env_right_BC.get_RP(0, store=False))  # saves the environments!
        #for i in range(self.boundary + 1, self.boundary + self.engine.n_optimize):      # SAJANT, 09/15/2021 - what do I delete when site!=0? I just shift the range by site.
        #    env.del_LP(i)  # but we might have gotten more than we need
        H0 = ZeroSiteH.from_LP_RP(LP, RP)
        if self.model.H_MPO.explicit_plus_hc:
            H0 = SumNpcLinearOperator(H0, H0.adjoint())
        vL, vR = LP.get_leg('vR').conj(), RP.get_leg('vL').conj()
        
        if left_segment.chinfo.qnumber == 0:    # Handles the case of no charge-conservation
            desired_Q = None
        else:
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
        self.logger.info("Desired gluing charge: %r", desired_Q)

        # We need a tensor that is non-zero only when Q = (Q^i_L - bar(Q_L)) + (Q^i_R - bar(Q_R))
        # Q is the the charge we insert. For now I intend for this to be a trivial set of charges since we can change the charges below.
        
        th0 = npc.Array.from_func(np.ones, [vL, vR],
                                  dtype=left_segment.dtype,
                                  qtotal=desired_Q,
                                  labels=['vL', 'vR'])
        #lanczos_params = self.engine.lanczos_params # May not exist yet
        lanczos_params = self.options.get("lanczos_params", {}) # See if lanczos_params is in yaml, if not use empty dictionary
        _, th0, _ = lanczos.LanczosGroundState(H0, th0, lanczos_params).run()
        th0 = npc.tensordot(th0, right_segment.get_B(0, 'B'), axes=['vR', 'vL'])
        right_segment.set_B(0, th0, form='Th')
        rB[0] = right_segment.get_B(0)
        rS[0] = right_segment.get_SL(0)
        lS = lS[0:-1] # Remove last singular values from list of singular values in A part of segment.
        ##################### BIG OL HACK #####################

        # note: __init__ makes deep copies of B, S
        cp = MPS(l_sites + r_sites, lB + rB, lS + rS, 'segment', 'B', gsl.norm)
        cp.grouped = gsl.grouped
        cp.canonical_form_finite(cutoff=1e-15) #to strip out vanishing singular values at the interface
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
        
    def glue_charge_sector(self):
        """Fix charge ambiguity of gluing together tensors from left and right ground states.."""
        self.logger.info("Glue charge sector of the ground state "
                         "[contracts environments from right]")
        #site = self.options.get("switch_charge_sector_site", 0)
        
        # Calculate energy of "vacuum" segment
        # [TODO] optimize this by using E_L = E_L^0 + epsilon*L where E_L^0 = LP_L * s^2 * RP_L
        env_left_BC = MPOEnvironment(self.ground_state_left, self.model_left.H_MPO, self.ground_state_left, **self.init_env_data_L)
        E_L = env_left_BC.full_contraction(0)
        E_L_2 = self.E0_L + self.eps_L * self.ground_state_left.L
        # print("Kwargs:")
        # print(self.init_env_data_L)
        env_right_BC = MPOEnvironment(self.ground_state_right, self.model_right.H_MPO, self.ground_state_right, **self.init_env_data_R)
        E_R = env_right_BC.full_contraction(0)
        E_R_2 = self.E0_R + self.eps_R * self.ground_state_right.L
        
        self.logger.info("EL, ER, EL2, ER2: %.14f, %.14f, %.14f, %.14f", E_L, E_R, E_L_2, E_R_2)
        self.logger.info("epsilon_L, epsilon_R, E0_L, E0_R: %.14f, %.14f, %.14f, %.14f", self.eps_L, self.eps_R, self.E0_L, self.E0_R)
        
        assert np.isclose(E_L, E_R)
        assert np.isclose(E_L_2, E_R_2)
        assert np.isclose(E_L, E_R_2)

        # print("E_L",E_L)
        # print("E_R",E_R)
        self.results['ground_state_energy'] = (E_L_2 + E_R_2)/2
        return 
    
    
        env = self.engine.env
        # Remove ambiguity in charge from the environment
        LP = env.get_LP(self.boundary)
        RP = env._contract_RP(self.boundary, env.get_RP(self.boundary, store=True))  # saves the environments!
        for i in range(self.boundary + 1, self.boundary + self.engine.n_optimize):      # SAJANT, 09/15/2021 - what do I delete when site!=0? I just shift the range by site.
            env.del_LP(i)  # but we might have gotten more than we need
        H0 = ZeroSiteH.from_LP_RP(LP, RP)
        if self.model.H_MPO.explicit_plus_hc:
            H0 = SumNpcLinearOperator(H0, H0.adjoint())
        vL, vR = LP.get_leg('vR').conj(), RP.get_leg('vL').conj()
        
        if self.psi.chinfo.qnumber == 0:    # Handles the case of no charge-conservation
            desired_Q = None
        else:
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
        self.logger.info("Desired gluing charge: %r", desired_Q)

        
        # We need a tensor that is non-zero only when Q = (Q^i_L - bar(Q_L)) + (Q^i_R - bar(Q_R))
        # Q is the the charge we insert. For now I intend for this to be a trivial set of charges since we can change the charges below.
        
        th0 = npc.Array.from_func(np.ones, [vL, vR],
                                  dtype=self.psi.dtype,
                                  qtotal=desired_Q,
                                  labels=['vL', 'vR'])
        lanczos_params = self.engine.lanczos_params
        _, th0, _ = lanczos.LanczosGroundState(H0, th0, lanczos_params).run()
        th0 = npc.tensordot(th0, self.psi.get_B(self.boundary, 'B'), axes=['vR', 'vL'])
        self.psi.set_B(self.boundary, th0, form='Th')
    
    def switch_charge_sector(self):
        """Change the charge sector of :attr:`psi` in place."""
        
        self.glue_charge_sector()
        apply_local_op = self.options.get("apply_local_op", None)
        switch_charge_sector = self.options.get("switch_charge_sector", None)
        if apply_local_op is None and switch_charge_sector is None:
            return
        
        if self.psi.chinfo.qnumber == 0:
            raise ValueError("can't switch charge sector with trivial charges!")
        self.logger.info("switch charge sector of the ground state "
                         "[contracts environments from right]")
        site = self.options.get("switch_charge_sector_site", self.boundary)
        self.logger.info("Changing charge to the left of site: %d", site)
        qtotal_before = self.psi.get_total_charge()
        self.logger.info("Charges of the original segment: %r", list(qtotal_before))

        env = self.engine.env
        
        if apply_local_op is not None:
            if switch_charge_sector is not None:
                raise ValueError("give only one of `switch_charge_sector` and `apply_local_op`")
            #self.results['ground_state_energy'] = env.full_contraction(0)
            for i in range(0, apply_local_op['i'] - 1): # TODO shouldn't we delete RP(i-1)
                env.del_RP(i)
            for i in range(apply_local_op['i'] + 1, env.L):
                env.del_LP(i)
            apply_local_op['unitary'] = True  # no need to call psi.canonical_form
            self.psi.apply_local_op(**apply_local_op)
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
            th0 = npc.tensordot(th0, self.psi.get_B(site, 'B'), axes=['vR', 'vL'])
            self.psi.set_B(site, th0, form='Th')
        self.psi.canonical_form_finite(cutoff=1e-15) #to strip out vanishing singular values at the interface
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
        #File "/global/home/users/sajant/BLG_DMRG/TeNPy/tenpy/simulations/ground_state_search.py", line 464, in __init__                                    
        #super().__init__(sim.model.lat, options, sim.model.dtype)                                                                                        
        #AttributeError: 'MoireModel' object has no attribute 'dtype'
        try:            # SAJANT, 09/08/21
            model_dtype = sim.model.dtype
        except:
            model_dtype = np.complex128
        super().__init__(sim.model.lat, options, model_dtype) #sim.model.dtype)

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
