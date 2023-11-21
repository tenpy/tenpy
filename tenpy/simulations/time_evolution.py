"""Simulations for (real) time evolution and for time dependent correlation functions."""

# Copyright 2020-2023 TeNPy Developers, GNU GPLv3

import numpy as np
import traceback

from . import simulation
from .simulation import *
from .post_processing import SpectralFunctionProcessor
from ..networks.mps import MPSEnvironment, MPS, MPSEnvironmentJW
from ..tools.misc import to_iterable

__all__ = simulation.__all__ + ['RealTimeEvolution', 'SpectralSimulation', 'TimeDependentCorrelation',
                                'TimeDependentCorrelationExperimental', 'SpectralSimulationExperimental']


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


class TimeDependentCorrelation(RealTimeEvolution):
    r"""A subclass of :class:`RealTimeEvolution` to specifically calculate the time dependent correlation function.

    In general this subclass calculates an overlap
    of the form :math:`C(r, t) = <\psi_0| B_r(t) A_{r_0} |\psi_0>` where :math:`A_{r_0}` can be
    passed as a simple on-site operator (on site `r0`) or as a product operator acting on
    several sites. The operator B is currently restricted to a single-site operator.
    However, passing `B` as a list ``[B_1, B_2, B_3]`` to calculate several overlaps
    is possible.

    Parameters
    ----------
    options : dict-like
        The simulation parameters. Ideally, these options should be enough to fully specify all
        parameters of a simulation to ensure reproducibility.
        These parameters are converted to a (dict-like) :class:`~tenpy.tools.params.Config`.
        For command line use, a .yml file should hold the information.

    Options
    -------
    .. cfg:config :: TimeDependentCorrelation
        :include: TimeEvolution

        addJW : bool
            boolean flag whether to add the Jordan Wigner String or not
        ground_state_filename : str
            a filename of a given ground state search (ideally a hdf5 file coming from finished
            run of a :class:`GroundStateSearch`)

    """
    default_measurements = (RealTimeEvolution.default_measurements +
                            [('simulation_method', 'm_correlation_function')])
    # TODO: this breaks when use default measurements is set to False in options.
    # possibly just override _connect_measurements function of Simulation class

    def __init__(self, options, *, gs_data=None, **kwargs):
        super().__init__(options, **kwargs)
        self.gs_data = self._load_data_from_gs(gs_data)
        # should be a dict with model params and psi_ground_state but allows passing an MPS
        # will be read out in init_state
        self.gs_energy = self.options.get('gs_energy', None)
        self.operator_t = self.options['operator_t']
        # generate info for operator before time evolution as subconfig
        self.operator_t0_config = self.options.subconfig('operator_t0')
        self.operator_t0 = None  # read out config later, since defaults depend on model parameters
        self.addJW = self.options.get('addJW', False)
        resume_data = self.results.get("resume_data", None)
        if resume_data:
            if 'psi_ground_state' in self.results['simulation_parameters']:
                self.psi_ground_state = self.results['simulation_parameters']['psi_ground_state']

    @classmethod
    def from_gs_search(cls, filename, sim_params, **kwargs):
        r"""Create class based on file containing the ground state.

        Initialize an instance of a :class:`SpectralSimulation` from
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
            self.logger.warning("The Simulation is not loaded from a GroundStateSearch...")
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
        r"""Converts the specified operators and indices into a list of tuples [(op1, i_1), (op2, i_2)]

        Options
        -------
        .. cfg:configoptions :: TimeDependentCorrelation

            operator_t0 : dict
                Mandatory, this should specify the operator initially applied to the MPS (i.e. before a time evolution).
                For more than one single-site operator, a list of operator names should be passed, otherwise just the
                string ``name``.
                Furthermore, the corresponding position(s) to apply the operator(s) should also be passed as a list.
                Either a lattice index ``lat_idx`` or a ``mps_idx`` should be passed.

                .. note ::
                    The ``lat_idx`` must have (dim+1) i.e. [x, y, u],
                    where u = 0 for a single-site unit cell

        """
        ops = to_iterable(self.operator_t0_config['name'])
        mps_idx = self.operator_t0_config.get('mps_idx', None)
        lat_idx = self.operator_t0_config.get('lat_idx', None)
        if mps_idx and lat_idx is not None:
            raise KeyError("Either a mps_idx or a lat_idx should be passed")
        elif mps_idx is not None:
            idx = to_iterable(mps_idx)
        elif lat_idx is not None:
            idx = to_iterable(self.model.lat.lat2mps_idx(lat_idx))
        else:
            idx = to_iterable(self.model.lat.N_sites // 2)
        # tiling
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

    def get_mps_environment(self):
        return MPSEnvironment(self.psi_ground_state, self.psi)

    def m_correlation_function(self, results, psi, model, simulation, **kwargs):
        r"""Measurement function for time dependent correlations.

        Wrapper around :meth:`_m_correlation_function_op` to loop over several operators.

        Options
        -------
        .. cfg:configoptions :: TimeDependentCorrelation

            operator_t : str | list
                The (on-site) operator as string to apply at each measurement step to calculate the overlap with.
                If a list is passed i.e.: ['op1', 'op2'], it will be iterated through the operators

        """
        self.logger.info("calling m_correlation_function")
        operator_t = to_iterable(self.operator_t)
        env = self.get_mps_environment()  # custom method for subclass Experimental
        # TODO: get better naming convention, store this in dict ?
        for i, op in enumerate(operator_t):
            if isinstance(op, str):
                results[f'correlation_function_t_{op}'] = self._m_correlation_function_op(env, op)
            else:
                results[f'correlation_function_t_{i}'] = self._m_correlation_function_op(env, op)

    def _m_correlation_function_op(self, env: MPSEnvironment, op) -> np.ndarray:
        r"""Measurement function for time dependent correlations.

        This calculates the overlap of <psi| op_j |phi>, where |phi> = e^{-iHt} op1_idx |psi_0>
        (the time evolved state after op1 was applied at MPS position idx) and
        <psi| is either <psi_0| e^{iHt} (if evolve_bra is True) or e^{i E_0 t} <psi| (if evolve_bra is False).

        Returns
        ----------
        correlation_function_t : 1D array
            representing <psi_0| e^{iHt} op2^i_j e^{-iHt} op1_idx |psi_0>
            where op2^i is the i-th operator given in the list [op2^1, op2^2, ..., op2^N]
            and spectral_function_t[j] corresponds to this overlap at MPS site j at time t
        """
        # TODO: case dependent if op needs JW string
        if self.addJW is False:
            correlation_function_t = env.expectation_value(op)
        else:
            correlation_function_t = list()
            for i in range(self.psi.L):
                term_list, i0, _ = env._term_to_ops_list([('Id', 0), (op, i)], True)
                # this generates a list from left to right
                # ["JW", "JW", ... "JW", "op (at idx)"], the problem is, that _term_to_ops_list does not generate
                # a JW string for one operator, therefore insert Id at idx 0.
                assert i0 == 0  # make sure to really start on the left site
                correlation_function_t.append(env.expectation_value_multi_sites(term_list, i0))
                # TODO: change when :meth:`expectation_value` of :class:`MPSEnvironment` automatically handles JW-string
            correlation_function_t = np.array(correlation_function_t)

        # multiply evolution of bra (eigenstate) into spectral function
        phase = np.exp(1j * self.gs_energy * self.engine.evolved_time)
        correlation_function_t = correlation_function_t * phase

        return correlation_function_t


class TimeDependentCorrelationExperimental(TimeDependentCorrelation):
    r"""Improved/Experimental version of :class:`TimeDependentCorrelation`.

    This class gives an advantage when calculating the correlation function of Fermions. This is done by
    calling the :class:`MPSEnvironmentJW` instead of the usual :class:`MPSEnvironment`.
    This class automatically adds a (hanging) JW string to each LP (only) when moving the
    environment to the right; if this wouldn't be done, much of the advantage of an MPS
    environment is lost (since only the overlap with the full operator string is calculated).

    Options
    -------
    .. cfg:config :: TimeDependentCorrelationExperimental
        :include: TimeDependentCorrelation

        evolve_bra : bool=False
            If True, instantiates a second engine and performs time_evolution on the (eigenstate) bra.
    """
    def __init__(self, options, *, gs_data=None, **kwargs):
        super().__init__(options, gs_data=gs_data, **kwargs)
        self.engine_ground_state = None
        self.evolve_bra = self.options.get('evolve_bra', False)
        # for resuming simulation from checkpoint # this is provided in super().__init__
        # TODO: How to ensure resuming from checkpoint works, when evolve_bra is True ?

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

    def _m_correlation_function_op(self, env, op) -> np.ndarray:
        """Measurement function for time dependent correlations.

        See also :meth:`TimeDependentCorrelation._m_correlation_function_op`

        Returns
        ----------
        correlation_function_t : 1D array
        """
        spectral_function_t = env.expectation_value(op)
        if self.evolve_bra is False:
            phase = np.exp(1j * self.gs_energy * self.engine.evolved_time)
            spectral_function_t = spectral_function_t * phase

        return spectral_function_t


class SpectralSimulation(TimeDependentCorrelation):
    """Simulation class to calculate Spectral Functions.

    The interface to the class is the same as to :class:`TimeDependentCorrelation`.

    Options
    -------
    .. cfg:config :: SpectralSimulation
        :include: TimeDependentCorrelation

    Attributes
    ----------
    post_processor : :class:`SpectralFunctionProcessor`
        :noindex:
        A class attribute defining a Post Processor to be used.
    """
    # class attribute linking SpectralSimulation to its post-processor
    post_processor = SpectralFunctionProcessor

    def __init__(self, options, *, gs_data=None, **kwargs):
        super().__init__(options, gs_data=gs_data, **kwargs)

    def post_processing(self):
        """Read-out and apply post-processing

        .. cfg:configoptions :: SpectralSimulation

        linear_prediction : dict
            parameters for linear prediction

            .. note ::
                There are several parameters to specify:
                m: number of time steps to predict
                p: number of time steps to use for predictions

        windowing : dict
            parameters for a windowing function
        """
        self.logger.info(f"calling post-processing with {self.post_processor}")
        processing_params = self.options.get('post_processing_params', None)
        # try, except clause to not lose simulation results if post_processing fails
        try:
            post_processor_cls = self.post_processor.from_simulation(self, processing_params=processing_params)
            # TODO: make sure this is written into self.results
            post_processor_cls.run()
        except Exception:
            self.logger.info("Could not post-process the results because of the following exception:")
            self.logger.warning(traceback.format_exc())
            self.logger.info("continuing saving results without post-processing")

    def prepare_results_for_save(self):
        """Post process results and prepare them for saving.

        Wrapper around :meth:`Simulation.prepare_results_for_save`.
        Makes it possible to include post-processing run during the run of the
        actual simulation.
        """
        if hasattr(self, 'post_processor'):
            self.post_processing()
        return super().prepare_results_for_save()


class SpectralSimulationExperimental(TimeDependentCorrelationExperimental, SpectralSimulation):
    pass
