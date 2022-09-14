"""Simulations for (real) time evolution."""
# Copyright 2020-2023 TeNPy Developers, GNU GPLv3

import numpy as np

from . import simulation
from .simulation import *

__all__ = simulation.__all__ + ['RealTimeEvolution']


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


class SpectralFunction(RealTimeEvolution):
    """Calculate spectral functions through a time evolution.

    Options
    -------
    .. cfg:config :: ZeroTemperatureSpectralFunction

        ground_state_filename :
            File from which the ground state (and model parameters) should be loaded.


    .. todo ::
        Share code with OrthogonalExcitations as far as possible?
    """

    def __init__(self, options, *, gs_data=None, **kwargs):
        super().__init__(options, **kwargs)
        self._gs_data = gs_data

    def run(self):
        if not hasattr(self, 'ground_state_orig'):
            self.init_from_groundstate()
        return super().run()

    def resume_run(self):
        if not hasattr(self, 'ground_state_orig'):
            self.init_from_groundstate()
        return super().resume_run()

    def init_from_groundstate(self):
        gs_fn, gs_data = self._load_gs_data()
        self.ground_state_orig = gs_data['psi']  # no copy!
        self.psi = self.ground_state_orig.copy()

        # copy ground state model parameters
        gs_data_options = gs_data['simulation_parameters']
        for key in gs_data_options.keys():
            if not isinstance(key, str) or not key.startswith('model'):
                continue
            if key not in self.options:
                self.options[key] = gs_data_options[key]

        # apply local operator
        # TODO: generalize to allow more complicated ops
        i0 = self.options['operator_t0']['i']
        op0 = self.options['operator_t0']['op']
        self._apply_local_op(psi, [i0, op0])

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


    def init_measurements(self):
        # add measurements for overlaps
        op = self.options['operator_t']  # TODO: could be daggger of operator_t0?
        meas = list(self.options.get('connect_measurements', []))
        # TODO add function to measure <psi_0 | op | psi>
        raise NotImplementedError("TODO")
        self.options['_connect_measurements'] = meas
        super().init_measurements()


    def post_process(self):
        # optionally use linear prediction
        # TODO: fourier transform to calculate dynamic structure factor

    def linear_prediction(self):
        raise NotImplementedError("TODO")
