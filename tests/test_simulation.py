"""A collection of tests to check the functionality of modules in `tenpy.simulations`"""
# Copyright 2020 TeNPy Developers, GNU GPLv3

import copy
import tenpy
from tenpy.algorithms.algorithm import Algorithm
from tenpy.simulations.simulation import Simulation


class DummyAlgorithm(Algorithm):
    def run(self):
        for i in range(2):
            self.checkpoint.emit(self)


simulation_params = {
    'model_class': 'XXZChain',
    'model_params': {
        'bc_MPS': 'infinite',
        'L': 2,
    },
    'algorithm_class': 'DummyAlgorithm',
    'algorithm_params': {},
    # 'initial_state_builder': 'KagomeInitialStateBuilder',
    'initial_state_params': {
        'method': 'lat_product_state',
        'product_state': [['up'], ['down']]
    },
    # 'register_measurements': [
    #     ('tenpy.simulations.measurement', 'energy_MPO'),
    #     ('tenpy.simulations.measurement', 'entropy'),
    # }
}


def test_simulation():
    sim_params = copy.deepcopy(simulation_params)
    sim = Simulation(sim_params)
    sim.run()
