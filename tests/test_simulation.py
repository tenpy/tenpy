"""A collection of tests to check the functionality of modules in `tenpy.simulations`"""
# Copyright 2020-2021 TeNPy Developers, GNU GPLv3

import copy
import numpy as np

import tenpy
from tenpy.algorithms.algorithm import Algorithm
from tenpy.simulations.simulation import Simulation
from tenpy.simulations.ground_state_search import GroundStateSearch
from tenpy.simulations.time_evolution import RealTimeEvolution
from tenpy.tools.misc import find_subclass

import pytest


class DummyAlgorithm(Algorithm):
    def __init__(self, psi, model, options, **kwargs):
        super().__init__(psi, model, options, **kwargs)
        self.dummy_value = None
        self.evolved_time = self.options.get('start_time', 0.)
        init_env_data = self.options.get("init_env_data", {})
        self.env = DummyEnv(**init_env_data)

    def run(self):
        N_steps = self.options.get('N_steps', 5)
        dt = self.options.get('dt', 5)
        self.dummy_value = N_steps**2
        for i in range(N_steps):
            self.evolved_time += dt
            self.checkpoint.emit(self)
        return None, self.psi


class SimulationStop(Exception):
    pass


def raise_SimulationStop(algorithm):
    if algorithm.evolved_time == 1.:
        raise SimulationStop("from raise_SimulationStop")


class DummyEnv:
    def __init__(self, **kwargs):
        if kwargs:
            assert kwargs == self.get_initialization_data()

    def get_initialization_data(self):
        return {"Env data": "Could be big"}


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
        'N_steps': 4,
        'dt': 0.5,
    },
    'initial_state_params': {
        'method': 'lat_product_state',  # mandatory -> would complain if not passed on
        'product_state': [['up'], ['down']]
    },
    'save_every_x_seconds':
    0.,  # save at each checkpoint
    'connect_measurements': [('tenpy.simulations.measurement', 'onsite_expectation_value', {
        'opname': 'Sz'
    }), (__name__, 'dummy_measurement')],
}


def test_Simulation(tmpdir):
    sim_params = copy.deepcopy(simulation_params)
    sim_params['directory'] = tmpdir
    sim_params['output_filename'] = 'data.pkl'
    sim = Simulation(sim_params)
    results = sim.run()  # should do exactly two measurements: one before and one after eng.run()
    assert sim.model.lat.bc_MPS == 'infinite'  # check whether model parameters were used
    assert 'psi' in results  # should be by default
    meas = results['measurements']
    # expect two measurements: once in `init_measurements` and in `final_measurement`.
    assert np.all(meas['measurement_index'] == np.arange(2))
    assert meas['dummy_value'] == [None, sim_params['algorithm_params']['N_steps']**2]
    assert (tmpdir / sim_params['output_filename']).exists()


def test_Simulation_resume(tmpdir):
    sim_params = copy.deepcopy(simulation_params)
    sim_params['directory'] = tmpdir
    sim_params['output_filename'] = 'data.pkl'
    # this should raise an error *after* saving the checkpoint
    sim_params['connect_algorithm_checkpoint'] = [(__name__, 'raise_SimulationStop', {}, -1)]
    sim = Simulation(sim_params)
    try:
        results = sim.run()
        assert False, "expected to raise a SimulationStop in sim.run()"
    except SimulationStop:
        checkpoint_results = sim.prepare_results_for_save()
    checkpoint_results['simulation_parameters']['connect_algorithm_checkpoint'] = []
    sim2 = Simulation.from_saved_checkpoint(checkpoint_results=checkpoint_results)
    res = sim2.resume_run()


groundstate_params = copy.deepcopy(simulation_params)
groundstate_params['save_environment_data'] = False


def test_GroundStateSearch():
    sim_params = copy.deepcopy(groundstate_params)
    sim = GroundStateSearch(sim_params)
    results = sim.run()  # should do exactly two measurements: one before and one after eng.run()
    assert sim.model.lat.bc_MPS == 'infinite'  # check whether model parameters were used
    assert 'psi' in results  # should be by default
    meas = results['measurements']
    # expect two measurements: once in `init_measurements` and in `final_measurement`.
    assert np.all(meas['measurement_index'] == np.arange(2))
    assert meas['dummy_value'] == [None, sim_params['algorithm_params']['N_steps']**2]
    del sim


timeevol_params = copy.deepcopy(simulation_params)
timeevol_params['final_time'] = 4.


def test_RealTimeEvolution():
    sim_params = copy.deepcopy(timeevol_params)
    sim = RealTimeEvolution(sim_params)
    results = sim.run()
    assert sim.model.lat.bc_MPS == 'infinite'  # check whether model parameters were used
    assert 'psi' in results  # should be by default
    meas = results['measurements']
    # expect two measurements: once in `init_measurements` and in `final_measurement`.
    alg_params = sim_params['algorithm_params']
    expected_times = np.arange(0., sim_params['final_time'] + 1.e-10,
                               alg_params['N_steps'] * alg_params['dt'])
    N = len(expected_times)
    assert np.allclose(meas['evolved_time'], expected_times)
    assert np.all(meas['measurement_index'] == np.arange(N))
    assert meas['dummy_value'] == [None] + [sim_params['algorithm_params']['N_steps']**2] * (N - 1)
