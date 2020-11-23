"""A collection of tests to check the functionality of modules in `tenpy.simulations`"""
# Copyright 2020 TeNPy Developers, GNU GPLv3

import copy
import numpy as np

import tenpy
from tenpy.algorithms.algorithm import Algorithm
from tenpy.simulations.simulation import Simulation


class DummyAlgorithm(Algorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dummy_value = None

    def run(self):
        N_steps = self.options.get('N_steps', 5)
        self.dummy_value = N_steps**2
        for i in range(N_steps):
            self.checkpoint.emit(self)


def dummy_measurement(results, psi, simulation):
    results['dummy_value'] = simulation.engine.dummy_value


simulation_params = {
    'model_class':
    'XXZChain',
    'model_params': {
        'bc_MPS': 'infinite',  # defaults to finite
        'L': 4,
    },
    'algorithm_class':
    'DummyAlgorithm',
    'algorithm_params': {
        'N_steps': 3
    },  # only one step
    # 'initial_state_builder': 'KagomeInitialStateBuilder',
    'initial_state_params': {
        'method': 'lat_product_state',  # mandatory -> would complain if not passed on
        'product_state': [['up'], ['down']]
    },
    'connect_measurements': [('tenpy.simulations.measurement', 'onsite_expectation_value', {
        'opname': 'Sz'
    }), (__name__, 'dummy_measurement')],
}


def test_simulation():
    sim_params = copy.deepcopy(simulation_params)
    sim = Simulation(sim_params)
    results = sim.run()  # calls the checkpoint twice
    assert sim.model.lat.bc_MPS == 'infinite'  # check whether model parameters were used
    assert 'psi' in results  # should be by default
    meas = results['measurements']
    # expect two measurements: once in `init_measurements` and in `final_measurement`.
    assert np.all(meas['measurement_index'] == np.arange(2))
    assert meas['dummy_value'] == [None, sim_params['algorithm_params']['N_steps']**2]
