"""Simulations for (real) time evolution."""

# Copyright 2020-2023 TeNPy Developers, GNU GPLv3

import numpy as np

from . import simulation
from .simulation import *
from .post_processing import SpectralFunctionProcessor
from ..networks.mps import MPSEnvironment, MPS, MPSEnvironmentJW

__all__ = simulation.__all__ + ['RealTimeEvolution', 'SpectralSimulation', 'SpectralSimulationExperimental']


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
        ('tenpy.simulations.measurement', 'm_evolved_time'),
    ]

    def __init__(self, options, **kwargs):
        super().__init__(options, **kwargs)
        if 'final_time' not in self.options.keys():
            raise KeyError("A 'final_time' must be supplied for a time evolution.")
        self.final_time = self.options['final_time'] - 1.e-10  # subtract eps: roundoff errors

    def run_algorithm(self):
        """Run the algorithm.

        Calls ``self.engine.run()`` and :meth:`make_measurements`.
        """
        # TODO: more fine-grained/custom break criteria?
        while True:
            if np.real(self.engine.evolved_time) >= self.final_time:
                break
            self.logger.info("evolve to time %.2f, max chi=%d", self.engine.evolved_time.real,
                             max(self.psi.chi))
            self.engine.run()
            # for time-dependent H (TimeDependentExpMPOEvolution) the engine can re-init the model;
            # use it for the measurements....
            self.model = self.engine.model

            self.make_measurements()
            self.engine.checkpoint.emit(self.engine)  # TODO: is this a good idea?

    def perform_measurements(self):
        if getattr(self.engine, 'time_dependent_H', False):
            # might need to re-initialize model with current time
            # in particular for a sequential/resume run, the first `self.init_model()` might not
            # yet have had the initial start time of the algorithm engine!
            self.engine.reinit_model()
            self.model = self.engine.model
        return super().perform_measurements()

    def resume_run_algorithm(self):
        self.run_algorithm()

    def final_measurements(self):
        """Do nothing.

        We already performed a set of measurements after the evolution in :meth:`run_algorithm`.
        """
        pass


class SpectralSimulation(RealTimeEvolution):
    """A subclass of :class:`RealTimeEvolution` to specifically calculate the time
     dependent correlation function. In general this subclass calculates an overlap
     of the form :math:`C(r, t) <psi_0| B_r(t) A_r0 |psi_0>` where A_r0 can be
     passed as a simple on-site operator (on site r0) or as a product operator acting on
     several sites. The operator B is currently restricted to a single-site operator.
     However, calculating passing B as a list [B_1, B_2, B_3] to calculate several overlaps
     is possible.

    Parameters
    ----------
    options : dict-like
        For command line use, a .yml file should hold the information.
        These parameters are converted to a (dict-like) :class:`~tenpy.tools.params.Config`.
        The parameters should hold information about the model, (time-evolution) algorithm and the operators
        for the correlation function. It's necessary to provide a final_time, this is inherited from the
        :class:`RealTimeEvolution`.

        Parameters example for this class
        params = {'ground_state_filename': 'ground_state.h5',
                  'final_time': 1,
                  'operator_t0': {'op': 'Sigmay', 'i': 20 , 'idx_form': 'mps'},
                  'operator_t': ['Sigmax', 'Sigmay', 'Sigmaz'], # TODO: handle custom operators (not specified by name)
                  'addJW': False}

        params['operator_t0']['op']: a list of operators to apply at the given indices 'i' (they all get applied before
        the time evolution), when a more complicated operator is needed. For simple (one-site) operators simply pass
        a string.
        e.g. operator_t0 = {'op': ['Sigmax', 'Sigmay'], 'i': [[10, 3, 0], [10, 3, 1]], 'idx_form': 'lat'}

        for 'idx_form': 'lat', the list of indices must contain d+1 elements (due to the unit cell index)
        for example 2d system indices are [x, y, u]
    """
    default_measurements = RealTimeEvolution.default_measurements + [
        ('simulation_method', 'm_spectral_function'),
    ]
    # class attribute linking SpectralSimulation to its post-processor
    post_processor = SpectralFunctionProcessor

    def __init__(self, options, *, gs_data=None, **kwargs):
        super().__init__(options, **kwargs)
        self.gs_data = self._load_data_from_gs(gs_data)
        # should be a dict with model params and psi_ground_state but allows passing an MPS
        # will be read out in init_state
        self.gs_energy = self.options.get('gs_energy', None)
        if 'operator_t' and 'operator_t0' and 'final_time' not in self.options.keys():
            raise KeyError("`operator_t`, `operator_t0` and a `final_time` must be supplied")
        self.operator_t = self.options['operator_t']
        # generate info for operator before time evolution as subconfig
        self.operator_t0_config = self.options.subconfig('operator_t0')
        self.operator_t0 = None  # read out config later, since defaults depend on model parameters
        self.addJW = self.options.get('addJW', False)
        # for resuming simulation from checkpoint # this is provided in super().__init__
        # TODO: How to ensure resuming from checkpoint works, when evolve_bra is True ?
        resume_data = self.results.get("resume_data", None)
        if resume_data:
            if 'psi_ground_state' in self.results['simulation_parameters'].keys():
                self.psi_ground_state = self.results['simulation_parameters']['psi_ground_state']

    @classmethod
    def from_gs_search(cls, filename, sim_params, **kwargs):
        """Initialize an instance of a :class:`SpectralSimulation` from
        a finished run of :class:`GroundStateSearch`. This simply fetches
        the relevant parameters ('model_params', 'psi')

        Parameters
        ----------
        filename : str or dict
            The filename of the ground state search output to be loaded.
            Alternatively the results as dictionary.
        sim_params : dict
            The necessary simulation parameters, it is necessary to specify final_time (inherited from
            :class:`RealTimeEvolution`). The parameters of the spectral simulation should also be given similar
            to the example params in the :class:`SpectralSimulation`.
        **kwargs :
            Further keyword arguments given to :meth:`__init__` of the class :class:`SpectralSimulation`.
        """
        return cls(options=sim_params, gs_data=filename, **kwargs)

    def init_state(self):
        # make sure state is not reinitialized if psi and psi_ground_state are given
        if not hasattr(self, 'psi_ground_state'):
            gs_data = self.gs_data
            if gs_data is not None:
                if isinstance(gs_data, MPS):
                    self.psi_ground_state = gs_data
                else:
                    self.psi_ground_state = gs_data['psi']
                delattr(self, 'gs_data')  # possibly free memory
            else:
                self.logger.warning("No ground state data is supplied, calling the initial state builder on\
                                     SpectralSimulation class. You probably want to supply a ground state")

                super().init_state()  # this sets self.psi from init_state_builder (should be avoided)
                self.psi_ground_state = self.psi.copy()
                delattr(self, 'psi')  # free memory

        if not hasattr(self, 'psi'):
            # copy is essential, since time evolution is probably only performed on psi
            self.psi = self.psi_ground_state.copy()
            self.apply_operator_t0_to_psi()

        # check for saving
        if self.options.get('save_psi', False):
            self.results['psi'] = self.psi
            self.results['psi_ground_state'] = self.psi_ground_state

    def init_algorithm(self, **kwargs):
        super().init_algorithm(**kwargs)  # links to RealTimeEvolution class, not to Simulation
        # get the energy of the ground state
        if self.gs_energy is None:
            self.gs_energy = self.model.H_MPO.expectation_value(self.psi_ground_state)

    def _load_data_from_gs(self, gs_data_kwarg):
        # we don't need the ground_state_filename
        key = 'ground_state_filename'
        message = 'ground state data'
        if isinstance(gs_data_kwarg, MPS):
            gs_data = gs_data_kwarg
        else:
            _, gs_data = self._load_data_from_kwarg_or_options(gs_data_kwarg, key, message=message, is_mandatory=True)
            # update model parameters here!
            self.check_and_update_params_from_gs_data(gs_data)
        return gs_data

    def check_and_update_params_from_gs_data(self, gs_data):
        sim_class = gs_data['version_info']['simulation_class']
        if sim_class != 'GroundStateSearch':
            raise ValueError("Must be loaded from a GS search")
        if 'psi' not in gs_data.keys():
            raise ValueError("MPS for ground state not found")
        elif not isinstance(gs_data['psi'], MPS):
            raise TypeError("Ground state must be an MPS")

        data_options = gs_data['simulation_parameters']
        for key in data_options.keys():
            if not isinstance(key, str) or not key.startswith('model'):
                continue
            if key not in self.options:
                self.options[key] = data_options[key]
            elif self.options[key] != data_options[key]:
                raise ValueError("Different model parameters in GroundStateSearch and GroundStateSearch")

        if 'energy' in gs_data.keys():
            self.gs_energy = self.options['gs_energy'] = gs_data['energy']

    def _get_operator_t0(self):
        """Converts the specified operators and indices into a list of tuples [(op1, i_1), (op2, i_2)]"""
        idx = self.operator_t0_config.get('i', self.psi.L // 2)
        ops = self.operator_t0_config.get('op', 'Sigmay')
        ops = [ops] if not isinstance(ops, list) else ops  # pass ops as list
        form = self.operator_t0_config.get('idx_form', 'mps')
        if form not in ['mps', 'lat']:
            raise ValueError("the idx_form must be either mps or lat")
        if form == 'mps':
            idx = list(idx if isinstance(idx, list) else [idx])
        else:
            if not isinstance(idx, list):
                raise TypeError("for idx_form lat, i must be given as list [x, y, u] or list of lists")
            mps_idx = self.model.lat.lat2mps_idx(idx)
            idx = list(mps_idx) if isinstance(mps_idx, np.ndarray) else [mps_idx]  # convert to mps index

        if len(ops) > len(idx):
            if len(idx) != 1:
                raise ValueError("Ill-defined tiling: ops is longer than idx, and idx is not one")
            idx = idx*len(ops)
        elif len(ops) < len(idx):
            if len(ops) != 1:
                raise ValueError("Ill-defined tiling: idx is longer than ops, and ops is not one")
            ops = ops * len(idx)
        # generate list of tuples of form [(op1, i_1), (op2, i_2), ...]
        op_list = list(zip(ops, idx))
        return op_list

    def apply_operator_t0_to_psi(self):
        # TODO: think about segment boundary conditions
        # TODO: make JW string consistent, watch for changes in apply_local_op to have autoJW
        self.operator_t0 = self._get_operator_t0()
        operator_t0 = self.operator_t0
        if len(operator_t0) == 1:
            op, i = operator_t0[0]
            if self.model.lat.site(i).op_needs_JW(op):
                for j in range(i):
                    self.psi.apply_local_op(j, 'JW')
            self.psi.apply_local_op(i, op)  # TODO: check if renormalize=True makes sense here
        else:
            ops, i_min, _ = self.psi._term_to_ops_list(operator_t0, True)  # applies JW string automatically
            for i, op in enumerate(ops):
                self.psi.apply_local_op(i_min + i, op)

    def prepare_results_for_save(self):
        """Wrapper around :meth:`prepare_results_for_save` of :class:`Simulation`.
        Makes it possible to include post-processing run during the run of the
        actual simulation.
        """
        if self.post_processor is not None:
            self.logger.info(f"calling post-processing with {self.post_processor}")
            processing_params = self.options.get('post_processing_params', None)
            # try, except clause to not lose simulation results if post_processing fails
            try:
                post_processor_cls = self.post_processor.from_simulation(self, processing_params=processing_params)
                # TODO: make sure this is written into self.results
                post_processor_cls.run()
            except Exception as e:
                self.logger.info("Could not post-process the results because of the following exception:")
                self.logger.warning(e)
                self.logger.info("continuing saving results without post-processing")
        return super().prepare_results_for_save()

    def get_mps_environment(self):
        return MPSEnvironment(self.psi_ground_state, self.psi)

    def m_spectral_function(self, results, psi, model, simulation, **kwargs):
        """Calculate the overlap :math:`<psi_0| e^{iHt} op2^j e^{-iHt} op1_idx |psi_0>` between
        op1 at MPS position idx and op2 at the MPS position j"""
        self.logger.info("calling m_spectral_function")
        operator_t = self.operator_t
        env = self.get_mps_environment()  # custom method for subclass Experimental
        # TODO: get better naming convention, store this in dict ?
        if isinstance(operator_t, list):
            for i, op in enumerate(operator_t):
                if isinstance(op, str):
                    results[f'spectral_function_t_{op}'] = self._m_spectral_function_op(env, op)
                else:
                    results[f'spectral_function_t_{i}'] = self._m_spectral_function_op(env, op)
        else:
            if isinstance(operator_t, str):
                results[f'spectral_function_t_{operator_t}'] = self._m_spectral_function_op(env, operator_t)
            else:
                results[f'spectral_function_t'] = self._m_spectral_function_op(env, operator_t)

    def _m_spectral_function_op(self, env: MPSEnvironment, op) -> np.ndarray:
        """Calculate the overlap of <psi| op_j |phi>, where |phi> = e^{-iHt} op1_idx |psi_0>
        (the time evolved state after op1 was applied at MPS position idx) and
        <psi| is either <psi_0| e^{iHt} (if evolve_bra is True) or e^{i E_0 t} <psi| (if evolve_bra is False).

        Returns
        ----------
        spectral_function_t : 1D array
                              representing <psi_0| e^{iHt} op2^i_j e^{-iHt} op1_idx |psi_0>
                              where op2^i is the i-th operator given in the list [op2^1, op2^2, ..., op2^N]
                              and spectral_function_t[j] corresponds to this overlap at MPS site j at time t
        """
        # TODO: case dependent if op needs JW string
        if self.addJW is False:
            spectral_function_t = env.expectation_value(op)
        else:
            spectral_function_t = list()
            for i in range(self.psi.L):
                term_list, i0, _ = env._term_to_ops_list([('Id', 0), (op, i)], True)
                # this generates a list from left to right
                # ["JW", "JW", ... "JW", "op (at idx)"], the problem is, that _term_to_ops_list does not generate
                # a JW string for one operator, therefore insert Id at idx 0.
                assert i0 == 0  # make sure to really start on the left site
                spectral_function_t.append(env.expectation_value_multi_sites(term_list, i0))
                # TODO: change when :meth:`expectation_value` of :class:`MPSEnvironment` automatically handles JW-string
            spectral_function_t = np.array(spectral_function_t)

        # multiply evolution of bra (eigenstate) into spectral function
        phase = np.exp(1j * self.gs_energy * self.engine.evolved_time)
        spectral_function_t = spectral_function_t * phase

        return spectral_function_t


class SpectralSimulationExperimental(SpectralSimulation):
    """Improved version of :class:`SpectralSimulation`, which gives an advantage
    for calculating the correlation function of Fermions. This is done by calling
    the :class:`MPSEnvironmentJW` instead of the usual :class:`MPSEnvironment`.
    This class automatically adds a (hanging) JW string to each LP (only) when moving the
    environment to the right; if this wouldn't be done, much of the advantage of an MPS
    environment is lost (since only the overlap with the full operator string is calculated).

    Options:
    evolve_bra : bool
        default False. If True, instantiates a second engine and performs time_evolution on the (eigenstate) bra.
    """
    def __int__(self, options, *, gs_data=None, **kwargs):
        super().__init__(options, gs_data=gs_data, **kwargs)
        self.engine_ground_state = None
        self.evolve_bra = self.options.get('evolve_bra', False)

    def init_algorithm(self, **kwargs):
        super().init_algorithm(**kwargs)  # links to RealTimeEvolution class, not to Simulation
        # make sure a second engine is used when evolving the bra
        if self.evolve_bra is True:
            # fetch engine that evolves ket
            AlgorithmClass = self.engine.__class__
            # instantiate the second engine for the ground state
            algorithm_params = self.options.subconfig('algorithm_params')
            self.engine_ground_state = AlgorithmClass(self.psi_ground_state, self.model, algorithm_params, **kwargs)
        # TODO: think about checkpoints
        # TODO: resume data is handled by engine, how to pass this on to second engine?

    def run_algorithm(self):
        if self.evolve_bra is True:
            while True:
                if np.real(self.engine.evolved_time) >= self.final_time:
                    break
                self.logger.info("evolve to time %.2f, max chi=%d", self.engine.evolved_time.real,
                                 max(self.psi.chi))

                self.engine_ground_state.run()
                self.engine.run()
                # sanity check, bra and ket should evolve to same time
                assert self.engine.evolved_time == self.engine.evolved_time, 'Bra evolved to different time than ket'
                # for time-dependent H (TimeDependentExpMPOEvolution) the engine can re-init the model;
                # use it for the measurements....
                # TODO: is this a good idea?
                self.model = self.engine.model
                self.make_measurements()
                self.engine.checkpoint.emit(self.engine)
        else:
            super().run_algorithm()

    def get_mps_environment(self):
        if self.addJW is False:
            return MPSEnvironment(self.psi_ground_state, self.psi)
        else:
            return MPSEnvironmentJW(self.psi_ground_state, self.psi)

    def _m_spectral_function_op(self, env, op) -> np.ndarray:
        """Calculate the overlap of <psi| op_j |phi>, where |phi> = e^{-iHt} op1_idx |psi_0>
        (the time evolved state after op1 was applied at MPS position idx) and
        <psi| is either <psi_0| e^{iHt} (if evolve_bra is True) or e^{i E_0 t} <psi| (if evolve_bra is False).

        Returns
        ----------
        spectral_function_t : 1D array
                              representing <psi_0| e^{iHt} op2^i_j e^{-iHt} op1_idx |psi_0>
                              where op2^i is the i-th operator given in the list [op2^1, op2^2, ..., op2^N]
                              and spectral_function_t[j] corresponds to this overlap at MPS site j at time t
        """
        spectral_function_t = env.expectation_value(op)
        if self.evolve_bra is False:
            phase = np.exp(1j * self.gs_energy * self.engine.evolved_time)
            spectral_function_t = spectral_function_t * phase

        return spectral_function_t
