"""Simulations for (real) time evolution and for time dependent correlation functions."""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np

from . import simulation
from .simulation import *
from ..networks.mps import MPSEnvironment, MPS, MPSEnvironmentJW
from ..tools.misc import to_iterable
from ..tools import hdf5_io


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
        For command line use, a ``.yml`` file should hold the information.
    **kwargs : dict
        ground_state_data: dict
        ground_state_filename: str

    Options
    -------
    .. cfg:config :: TimeDependentCorrelation
        :include: TimeEvolution

        addJW : bool
            boolean flag whether to add the Jordan Wigner String or not
        ground_state_filename : str
            a filename of a given ground state search (ideally a hdf5 file coming from a finished
            run of a :class:`~tenpy.simulations.ground_state_search.GroundStateSearch`)

    """

    def __init__(self, options, *, ground_state_data=None, ground_state_filename=None, **kwargs):
        super().__init__(options, **kwargs)

        resume_data = kwargs.get("resume_data", None)
        if resume_data is not None:
            if 'psi_ground_state' in resume_data:
                self.psi_ground_state = resume_data['psi_ground_state']
            else:
                self.logger.warning("psi_ground_state not in resume data")
            if 'gs_energy' in resume_data:
                self.gs_energy = resume_data['gs_energy']
            else:
                self.logger.warning("ground-state energy not in resume data")

        if not self.loaded_from_checkpoint:
            if ground_state_filename is None:
                ground_state_filename = self.options.get('ground_state_filename', None)
            if ground_state_data is None and ground_state_filename is not None:
                self.logger.info(f"loading data from 'ground_state_filename'='{ground_state_filename}'")
                ground_state_data = hdf5_io.load(ground_state_filename)
            elif ground_state_data is not None and ground_state_filename is not None:
                self.logger.warning("Supplied a 'ground_state_filename' and ground_state_data as kwarg. "
                                    "Ignoring 'ground_state_filename'.")

            if ground_state_data is not None:
                self.logger.info("Initializing from ground state data")
                self._init_from_gs_data(ground_state_data)

        # will be read out in init_state
        self.gs_energy = self.options.get('gs_energy', None)
        self.operator_t = self.options['operator_t']
        # generate info for operator before time evolution as subconfig
        self.operator_t0_config = self.options.subconfig('operator_t0')
        self.operator_t0_name = self._get_operator_t0_name()
        self.operator_t0 = None  # read out config later, since defaults depend on model parameters
        self.addJW = self.options.get('addJW', False)

    def resume_run(self):
        if not hasattr(self, 'psi_ground_state'):
            # didn't get psi_ground_state in resume_data, but might still have it in the results
            if 'psi_ground_state' not in self.results:
                raise ValueError("psi_ground_state not saved in checkpoint results: can't resume!")
            if 'gs_energy' not in self.results:
                self.logger.warning("Ground state energy not saved in checkpoint results")
        super().resume_run()

    def get_resume_data(self):
        resume_data = super().get_resume_data()
        resume_data['psi_ground_state'] = self.psi_ground_state
        resume_data['gs_energy'] = self.gs_energy
        return resume_data

    def _connect_measurements(self):
        """Connect :func:`m_correlation_function` to measurements."""
        self._connect_measurements_fct('simulation_method', 'm_correlation_function', priority=1)
        super()._connect_measurements()

    def init_state(self):
        # make sure state is not reinitialized if psi and psi_ground_state are given
        if not hasattr(self, 'psi_ground_state'):
            self.logger.warning(f"No ground state data is supplied, calling the initial state builder on "
                                f"{self.__class__.__name__} class - you probably want to supply a ground state!")
            super().init_state()  # this sets self.psi from init_state_builder (should be avoided)
            self.psi_ground_state = self.psi.copy()
            delattr(self, 'psi')  # free memory

        if not hasattr(self, 'psi'):
            # copy is essential, since time evolution is probably only performed on psi
            self.psi = self.psi_ground_state.copy()
            self.apply_operator_t0_to_psi()

        # check for saving
        if self.options.get('save_psi', True):
            self.results['psi'] = self.psi
            self.results['psi_ground_state'] = self.psi_ground_state

    def init_algorithm(self, **kwargs):
        super().init_algorithm(**kwargs)  # links to RealTimeEvolution class, not to Simulation
        # make sure to get the energy of the ground state, this is needed for the correlation_function
        if self.gs_energy is None:
            self.gs_energy = self.model.H_MPO.expectation_value(self.psi_ground_state)
        if self.engine.psi.bc != 'finite':
            raise NotImplementedError('Only finite MPS boundary conditions are currently implemented for '
                                      f'{self.__class__.__name__}')

    def _init_from_gs_data(self, gs_data):
        if isinstance(gs_data, MPS):
            # self.psi_ground_state = gs_data ?
            raise NotImplementedError("Only hdf5 and dictionaries are supported as ground state input")
        sim_class = gs_data['version_info']['simulation_class']
        if sim_class != 'GroundStateSearch':
            self.logger.warning("The Simulation is not loaded from a GroundStateSearch...")

        data_options = gs_data['simulation_parameters']
        for key in data_options:
            if not isinstance(key, str) or not key.startswith('model'):
                continue
            if key not in self.options:
                self.options[key] = data_options[key]
            elif self.options[key] != data_options[key]:
                self.logger.warning("Different model parameters in Simulation and data from file. Ignoring parameters "
                                    "in data from file")
        if 'energy' in gs_data:
            self.options['gs_energy'] = gs_data['energy']

        if 'psi' not in gs_data:
            raise ValueError("MPS for ground state not found")
        psi_ground_state = gs_data['psi']
        if not isinstance(psi_ground_state, MPS):
            raise TypeError("Ground state must be an MPS class")

        if not hasattr(self, 'psi_ground_state'):
            self.psi_ground_state = psi_ground_state

    def _get_operator_t0_name(self):
        operator_t0_name = self.operator_t0_config.get('key_name', None)
        if operator_t0_name is None:
            opname = self.operator_t0_config['opname']  # opname is mandatory
            if len(to_iterable(opname)) == 1:
                operator_t0_name = opname
            else:
                raise KeyError("A key_name must be passed for multiple operators")
        return operator_t0_name

    def _get_operator_t0_list(self):
        r"""Converts the specified operators and indices into a list of tuples [(op1, i_1), (op2, i_2)]

        Options
        -------
        .. cfg:configoptions :: TimeDependentCorrelation

            operator_t0 : dict
                Mandatory, this should specify the operator initially applied to the MPS (i.e. before a time evolution).
                For more than one single-site operator, a list of operator names should be passed, otherwise just the
                string ``opname``. For several operators it is necessary to pass a ``key_name``, this
                determines the name of the corresponding measurement output see :meth:`_get_operator_t0_name`.
                The corresponding position(s) to apply the operator(s) should also be passed as
                a list (or string for a single operator).
                Either a lattice index ``lat_idx`` or a ``mps_idx`` should be passed.

                .. note ::

                    The ``lat_idx`` must have (dim+1) i.e. [x, y, u],
                    where u = 0 for a single-site unit cell

        """
        ops = to_iterable(self.operator_t0_config['opname'])  # opname is mandatory
        mps_idx = self.operator_t0_config.get('mps_idx', None)
        lat_idx = self.operator_t0_config.get('lat_idx', None)
        if mps_idx is not None and lat_idx is not None:
            raise KeyError("Either a mps_idx or a lat_idx should be passed")
        elif mps_idx is not None:
            idx = to_iterable(mps_idx)
        elif lat_idx is not None:
            idx = to_iterable(self.model.lat.lat2mps_idx(lat_idx))
        else:
            # default to the middle of the MPS sites
            idx = to_iterable(self.model.lat.N_sites // 2)
        # tiling
        if len(ops) > len(idx):
            if len(idx) != 1:
                raise ValueError("Ill-defined tiling: num. of operators must be equal to num. of indices or one")
            idx = idx * len(ops)
        elif len(ops) < len(idx):
            if len(ops) != 1:
                raise ValueError("Ill-defined tiling: num. of operators must be equal to num. of indices or one")
            ops = ops * len(idx)
        # generate list of tuples of form [(op1, i_1), (op2, i_2), ...]
        op_list = list(zip(ops, idx))
        return op_list

    def apply_operator_t0_to_psi(self):
        self.operator_t0 = self._get_operator_t0_list()
        ops = self.operator_t0
        if len(ops) == 1:
            op, i = ops[0]
            if self.model.lat.site(i).op_needs_JW(op):
                for j in range(i):
                    self.psi.apply_local_op(j, 'JW')
            self.psi.apply_local_op(i, op)
        else:
            ops, i_min, _ = self.psi._term_to_ops_list(ops, True)  # applies JW string automatically
            for i, op in enumerate(ops):
                self.psi.apply_local_op(i_min + i, op)

    def m_correlation_function(self, results, psi, model, simulation, **kwargs):  # simulation=self
        r"""Measurement function for time dependent correlations.

        Wrapper around :meth:`_m_correlation_function_op` to loop over several operators.

        Options
        -------
        .. cfg:configoptions :: TimeDependentCorrelation

            operator_t : str | list
                The (on-site) operator(s) as string(s) to apply at each measurement step.
                If a list is passed i.e.: ``['op1', 'op2']``, it will be iterated through the operators

        """
        self.logger.info("calling m_correlation_function")
        operator_t = to_iterable(self.operator_t)
        psi_gs = self.psi_ground_state
        env = MPSEnvironment(psi_gs, psi)

        for i, op in enumerate(operator_t):
            # op is a str
            results_key = f"correlation_function_t_{op}_{self.operator_t0_name}"
            results[results_key] = self._m_correlation_function_op(env, op)

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
    def __init__(self, options, *, ground_state_data=None, ground_state_filename=None, **kwargs):
        super().__init__(options, ground_state_data=ground_state_data,
                         ground_state_filename=ground_state_filename, **kwargs)
        self.engine_ground_state = None
        self.evolve_bra = self.options.get('evolve_bra', False)
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
        # TODO: think about checkpoints; resume data is handled by engine, how to pass this on to second engine?

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
                self.model = self.engine.model
                self.make_measurements()
                self.engine.checkpoint.emit(self.engine)
        else:
            super().run_algorithm()

    def m_correlation_function(self, results, psi, model, simulation, **kwargs):
        """Equivalent to :meth:`TimeDependentCorrelation._m_correlation_function`."""
        self.logger.info("calling m_correlation_function")
        operator_t = to_iterable(self.operator_t)
        psi_gs = self.psi_ground_state
        env = MPSEnvironmentJW(psi_gs, psi) if self.addJW else MPSEnvironment(psi_gs, psi)

        for i, op in enumerate(operator_t):
            # op is a str
            results_key = f"correlation_function_t_{op}_{self.operator_t0_name}"
            results[results_key] = self._m_correlation_function_op(env, op)

    def _m_correlation_function_op(self, env, op) -> np.ndarray:
        """Measurement function for time dependent correlations.

        Simplified version of :meth:`TimeDependentCorrelation._m_correlation_function_op`

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

        spectral_function_params: dict
            Additional parameters for post-processing of the spectral function (i.e. applying
            linear prediction or gaussian windowing. The keys correspond to the kwargs of
            :func:`~tenpy.tools.spectral_function_tools.spectral_function`.

    """
    default_post_processing = []

    def __init__(self, options, *, ground_state_data=None, ground_state_filename=None, **kwargs):
        super().__init__(options, ground_state_data=ground_state_data,
                         ground_state_filename=ground_state_filename, **kwargs)

    def run_post_processing(self):
        extra_kwargs = self.options.get('spectral_function_params', {})
        for key in self.results['measurements'].keys():
            if 'correlation_function_t' in key:
                results_key = 'spectral_function'+key.removeprefix('correlation_function_t')
                kwargs_dict = {'results_key': results_key, 'correlation_key': key}
                kwargs_dict.update(extra_kwargs)  # add parameters for linear prediction etc.
                pp_entry = ('tenpy.simulations.post_processing', 'pp_spectral_function', kwargs_dict)
                # create a new list here! (otherwise this is added to all instances within that session)
                self.default_post_processing = self.default_post_processing + [pp_entry]
        return super().run_post_processing()


class SpectralSimulationExperimental(TimeDependentCorrelationExperimental, SpectralSimulation):
    pass
