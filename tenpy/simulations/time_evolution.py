"""Simulations for (real) time evolution."""
# Copyright 2020-2023 TeNPy Developers, GNU GPLv3

import numpy as np

from . import simulation
from .simulation import *
from ..networks.mps import MPSEnvironment, MPS
__all__ = simulation.__all__ + ['RealTimeEvolution', 'SpectralSimulation']


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


class SpectralSimulation(RealTimeEvolution):
    """A subclass of :class:`RealTimeEvolution` to specifically calculate the time
     dependent correlation function.

    Parameters
    ----------
    options : dict-like
        For command line use, a .yml file should hold the information.
        These parameters are converted to a (dict-like) :class:`~tenpy.tools.params.Config`,
        by the :class:`Simulation` parent class.
        Must have: final_time: float and sensible to parse corr_fct: dict, example:
            params = {'final_time': 1,
                      'operator_t0': {'op': ['Sigmay', 'Sigmaz'], 'i': [5, 0] , 'idx_form': 'mps'},
                      'operator_t': ['op2_1_name', 'op2_2_name'],
                      'evolve_bra': False,
                      'addJW': True}

        It's necessary to provide a final_time, this is inherited from the RealTimeEvolution.
        params['operator_t0']['op']: a list of operators to apply at the given 'i' (they all get applied before
        the time evolution), when a more complicated operator is needed. For simple (one-site) operators simply parse
        a string, e.g.: params['operator_t0']['op'] = 'Sigmay'
        params['operator_t0']['i']: a list of indices either given in mps or lat form.

    Options
    -------
    """
    default_measurements = RealTimeEvolution.default_measurements + [
        ('simulation_method', 'm_spectral_function'),
    ]

    def __init__(self, options, *, gs_data=None, **kwargs):  # * forces gs_data to be passed explicitly
        super().__init__(options, **kwargs)
        self.psi_ground_state = self.options['gs_data']  # this is surely defined, since gs_data is kwarg
        self.engine_ground_state = None
        self.gs_energy = None
        self.operator_t0_config = self.options.subconfig('operator_t0')
        # generate info for spectral function as subconfig
        self.evolve_bra = self.options.get('evolve_bra', False)
        self.addJW = self.options.get('addJW', False)
        self.psi = None

    def init_state(self):
        # check first if psi is supplied
        # TODO: get ground state from hdf5 file

        psi = self.options.get('psi', None)
        if self.psi_ground_state is not None:
            self.psi = self.psi_ground_state
        elif isinstance(psi, MPS):
            self.psi = psi

        assert isinstance(self.psi, MPS), "psi must be an instance :class:`MPS`"
        self.logger.info("Initialized Psi from given psi")
        # super().init_state() won't reinitialize the state if psi is given
        super().init_state()  # links to bare Simulation class

        self.psi_ground_state = self.psi.copy()
        # apply the operator before performing the time evolution
        self.apply_op_list_to_psi()

        if self.options.get('save_psi', False):
            self.results['psi'] = self.psi
            self.results['psi_ground_state'] = self.psi_ground_state

    def apply_op_list_to_psi(self):
        # TODO: think about segment boundary conditions
        # TODO: make JW string consistent, watch for changes in apply_local_op to have autoJW
        op_list = self._get_op_list_from_operator_t0()
        if len(op_list) == 1:
            op, i = op_list[0]
            if self.model.lat.site(i).op_needs_JW(op):
                self.psi.apply_local_op(i, 'JW')
            self.psi.apply_local_op(i, op)  # TODO: check if renormalize=True makes sense here
        else:
            ops, i_min, _ = self.psi._term_to_ops_list(op_list, True)
            for i, op in enumerate(ops):
                self.psi.apply_local_op(i_min + i, op)

    def _get_op_list_from_operator_t0(self):
        idx = self.operator_t0_config.get('i', self.psi.L // 2)
        ops = self.operator_t0_config.get('op', 'Sigmay')
        ops = [ops] if type(ops) is not list else ops  # pass ops as list
        form = self.operator_t0_config.get('idx_form', 'mps')
        assert form == 'mps' or form == 'lat', "the idx_form must be either mps or lat"
        if form == 'mps':
            idx = list(idx if type(idx) is list else [idx])
        else:
            assert type(idx) == list, "for idx_form lat, i must be given as list [x, y, u] or list of lists"
            if len(ops) == 1 and len(idx) != 1:
                idx = [idx]
            for i, lat_idx in enumerate(idx):
                idx[i] = self.model.lat.lat2mps_idx(lat_idx)
        op_list = list(zip(ops, idx))  # form [(op1, i_1), (op2, i_2)]...
        return op_list

    def init_algorithm(self, **kwargs):
        super().init_algorithm(**kwargs)  # links to RealTimeEvolution class
        algorithm_params = self.options.subconfig('algorithm_params')
        # make sure second engine is used when evolving the bra
        if self.evolve_bra is True:
            # fetch engine that evolves ket
            AlgorithmClass = self.engine.__class__
            # instantiate the second engine for the ground state
            self.engine_ground_state = AlgorithmClass(self.psi_ground_state, self.model, algorithm_params, **kwargs)
        else:
            # get the energy of the ground state
            self.gs_energy = self.model.H_MPO.expectation_value(self.psi_ground_state)
        # TODO: think about checkpoints
        # TODO: resume data is handled by engine, how to pass this on to second engine?

    # TODO: specify run and possibly fall back to :meth:`run_algorithm` of :class: `RealTimeEvolution`
    def run_algorithm(self):
        while True:
            if np.real(self.engine.evolved_time) >= self.final_time:
                break
            self.logger.info("evolve to time %.2f, max chi=%d", self.engine.evolved_time.real,
                             max(self.psi.chi))

            if self.evolve_bra is True:
                # if threading is not used:
                self.engine_ground_state.run()
                self.engine.run()
                # sanity check, bra and ket should evolve to same time
                assert self.engine.evolved_time == self.engine.evolved_time, self.logger.warn(
                    'Bra evolved to different time than ket')
            else:
                self.engine.run()
            # for time-dependent H (TimeDependentExpMPOEvolution) the engine can re-init the model;
            # use it for the measurements....
            self.model = self.engine.model
            # TODO: is this a good idea?
            self.make_measurements()
            # TODO: think about checkpoints
            self.engine.checkpoint.emit(self.engine)  # TODO: is this a good idea?

    def m_spectral_function(self, results, psi, model, simulation, **kwargs):
        """Calculate the overlap <psi_0| e^{iHt} op2^j e^{-iHt} op1_idx |psi_0> between
        op1 at MPS position idx and op2 at the MPS position j"""
        self.logger.info("calling m_spectral_function")
        env = MPSEnvironment(self.psi_ground_state, self.psi)
        # TODO: how to parse this as input, should not be done every time the function is called.
        self.operator_t = self.options.get('operator_t', 'Sigmay')  # operator_t might be a list of operators
        # TODO: remove next two lines, depending on how results are stored
        # if 'spectral_function_t' not in results.keys():
        #    results['spectral_function_t'] = dict()
        # TODO: get better naming convention
        if isinstance(self.operator_t, list):
            for i, op in enumerate(self.operator_t):
                if isinstance(op, str):
                    results[f'spectral_function_t_{op}'] = self._m_spectral_function_op(env, op)
                else:
                    results[f'spectral_function_t_{i}'] = self._m_spectral_function_op(env, op)
        else:
            if isinstance(self.operator_t, str):
                results[f'spectral_function_t_{self.operator_t}'] = self._m_spectral_function_op(env, self.operator_t)
            else:
                results[f'spectral_function_t'] = self._m_spectral_function_op(env, self.operator_t)

    def _m_spectral_function_op(self, env: MPSEnvironment, op):
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
            spectral_function_t = []
            for i in range(self.psi.L):
                term_list, i0, _ = env._term_to_ops_list([('Id', 0), (op, i)], True)
                # this generates a list from left to right
                # ["JW", "JW", ... "JW", "op (at idx)"], the problem is, that _term_to_ops_list does not generate
                # a JW string for one operator, therefore insert Id at idx 0.
                assert i0 == 0  # make sure to really start on the left site
                spectral_function_t.append(env.expectation_value_multi_sites(term_list, i0))
                # TODO: change when :meth:`expectation_value` of :class:`MPSEnvironment` automatically handles JW-string
            spectral_function_t = np.array(spectral_function_t)

        if self.evolve_bra is False:
            phase = np.exp(1j * self.gs_energy * self.engine.evolved_time)
            spectral_function_t = spectral_function_t * phase

        return spectral_function_t

    def fourier_transform_time(self):
        raise NotImplementedError("TODO: currently outside :class:`SpectralSimulation`")

    def fourier_transform_space(self):
        """Fourier Transform in space already incorporating lattice geometry"""
        raise NotImplementedError("TODO: currently outside :class:`SpectralSimulation`")

    def linear_prediction(self):
        raise NotImplementedError("TODO: currently outside :class:`SpectralSimulation`")
