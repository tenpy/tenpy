"""A collection of tests to check the functionality of modules in `tenpy.simulations`"""
# Copyright 2020-2021 TeNPy Developers, GNU GPLv3

import copy
import numpy as np

import tenpy
from tenpy.algorithms.algorithm import Algorithm
from tenpy.simulations.simulation import Simulation, resume_from_checkpoint
from tenpy.simulations.ground_state_search import GroundStateSearch
from tenpy.simulations.time_evolution import RealTimeEvolution
from tenpy.tools.misc import find_subclass

import pytest

tenpy.tools.misc.skip_logging_setup = True  # skip logging setup


class DummyAlgorithm(Algorithm):
    def __init__(self, psi, model, options, *, resume_data=None):
        super().__init__(psi, model, options, resume_data=resume_data)
        self.dummy_value = None
        self.evolved_time = self.options.get('start_time', 0.)
        init_env_data = {} if resume_data is None else resume_data['init_env_data']
        self.env = DummyEnv(**init_env_data)

    def run(self):
        N_steps = self.options.get('N_steps', 5)
        dt = self.options.get('dt', 5)
        self.dummy_value = N_steps**2
        for i in range(N_steps):
            self.evolved_time += dt
            self.checkpoint.emit(self)
        return None, self.psi

    def get_resume_data(self):
        data = super().get_resume_data()
        data['init_env_data'] = self.env.get_initialization_data()
        return data


class SimulationStop(Exception):
    pass


class DummySimulation(Simulation):
    # for `test_Simulation_resume`
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


def test_Simulation(tmp_path):
    sim_params = copy.deepcopy(simulation_params)
    sim_params['directory'] = tmp_path
    sim_params['output_filename'] = 'data.pkl'
    sim = Simulation(sim_params)
    results = sim.run()  # should do exactly two measurements: one before and one after eng.run()
    assert sim.model.lat.bc_MPS == 'infinite'  # check whether model parameters were used
    assert 'psi' in results  # should be by default
    meas = results['measurements']
    # expect two measurements: once in `init_measurements` and in `final_measurement`.
    assert np.all(meas['measurement_index'] == np.arange(2))
    assert meas['dummy_value'] == [None, sim_params['algorithm_params']['N_steps']**2]
    assert (tmp_path / sim_params['output_filename']).exists()


def test_Simulation_resume(tmp_path):
    sim_params = copy.deepcopy(simulation_params)
    sim_params['directory'] = tmp_path
    sim_params['output_filename'] = 'data.pkl'
    # this should raise an error *after* saving the checkpoint
    sim_params['connect_algorithm_checkpoint'] = [(__name__, 'raise_SimulationStop', {}, -1)]
    sim = DummySimulation(sim_params)
    try:
        results = sim.run()
        assert False, "expected to raise a SimulationStop in sim.run()"
    except SimulationStop:
        checkpoint_results = sim.prepare_results_for_save()
    assert not checkpoint_results['finished_run']
    # try resuming with `resume_from_checkpoint`
    update_sim_params = {'connect_algorithm_checkpoint': []}
    res = resume_from_checkpoint(filename=tmp_path / 'data.pkl',
                                 update_sim_params=update_sim_params)

    # alternatively, resume from the checkpoint results we have
    checkpoint_results['simulation_parameters']['connect_algorithm_checkpoint'] = []
    # if we explicitly know the simulation class, it's easy
    sim2 = DummySimulation.from_saved_checkpoint(checkpoint_results=checkpoint_results)
    res2 = sim2.resume_run()
    for r in [res, res2]:
        assert r['finished_run']
        assert np.all(r['measurements']['measurement_index'] == np.arange(2))


groundstate_params = copy.deepcopy(simulation_params)


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
