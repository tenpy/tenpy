"""Short tests for the :class:`DataLoader` and for post_processing functions in a Simulation"""
# Copyright 2023 TeNPy Developers, GNU GPLv3


import tenpy
import copy
import numpy as np
import pytest

from tenpy.algorithms import Algorithm
from tenpy.simulations.simulation import Simulation
from tenpy.simulations.post_processing import DataLoader
from tenpy.models.xxz_chain import XXZChain

tenpy.tools.misc.skip_logging_setup = True  # skip logging setup


class DummyAlgorithmPP(Algorithm):
    def __init__(self, psi, model, options, *, resume_data=None, cache=None):
        super().__init__(psi, model, options, resume_data=resume_data)
        self.dummy_value = -1
        self.evolved_time = self.options.get('start_time', 0.)

    def run(self):
        N_steps = self.options.get('N_steps', 5)
        dt = self.options.get('dt', 5)
        self.dummy_value = N_steps**2
        for i in range(N_steps):
            self.evolved_time += dt
            self.checkpoint.emit(self)
        return None, self.psi


def pp_dummy_function(DL, *, kwarg_getting_m_key=None):
    # check that DL was instantiated and can return m_results
    m_result = DL.get_data_m(kwarg_getting_m_key)
    return m_result * 2


# making sure that the simulation is still running even when the dummy function raises an Exception
def broken_pp_dummy_function(*args, **kwargs):
    raise ValueError()


simulation_params = {
    'model_class':
    'XXZChain',
    'model_params': {
        'bc_MPS': 'finite',
        'L': 4,
        'sort_charge': True,
    },
    'algorithm_class':
    'DummyAlgorithmPP',
    'algorithm_params': {
        'N_steps': 2,
        'dt': 0.5,
    },
    'initial_state_params': {
        'method': 'lat_product_state',  # mandatory -> would complain if not passed on
        'product_state': [['up'], ['down']]
    },
    'connect_measurements': [('tenpy.simulations.measurement', 'm_onsite_expectation_value', {'opname': 'Sz'})],

    'post_processing': [(__name__, 'pp_dummy_function',
                         {'results_key': 'pp_result',
                          'kwarg_getting_m_key': '<Sz>'}),
                        (__name__, 'broken_pp_dummy_function'),
                        (__name__, 'pp_dummy_function',
                         {'results_key': 'pp_result',
                          'kwarg_getting_m_key': '<Sz>'})],
}


def test_Simulation_with_post_processing():
    sim_params = copy.deepcopy(simulation_params)
    sim = Simulation(sim_params)
    msg = "Error during post_process of test_post_processing broken_pp_dummy_function"
    with pytest.warns(UserWarning, match=msg):
        # should do exactly two measurements: one before and one after eng.run()
        results = sim.run()
    assert 'errors_during_run' in results, "we called broken_pp_dummy_function, so there should be an error"
    assert 'pp_result' in results
    assert 'pp_result_1' in results  # make sure pp_result was not overwritten

    Sz = results['measurements']['<Sz>']
    pp_result = results['pp_result']
    assert np.allclose(2*Sz, pp_result)


def test_init_of_DataLoader(tmp_path):
    sim_params = copy.deepcopy(simulation_params)
    sim_params['directory'] = tmp_path.as_posix()
    sim_params['output_filename'] = '_test.pkl'
    sim_params['max_errors_before_abort'] = None
    sim = Simulation(sim_params)
    msg = "Error during post_process of test_post_processing broken_pp_dummy_function"
    with pytest.warns(UserWarning, match=msg):
        results = sim.run()
    assert 'errors_during_run' in results, "we called broken_pp_dummy_function, so there should be an error"
    DL_1 = DataLoader(data=results)
    DL_2 = DataLoader(simulation=sim)
    DL_3 = DataLoader(filename=tmp_path / '_test.pkl')
    # check that model was correctly instantiated
    assert isinstance(DL_1.model, XXZChain)
    assert isinstance(DL_2.model, XXZChain)
    assert isinstance(DL_3.model, XXZChain)
    # check that getting measurement data worked
    Sz_1 = DL_1.get_data_m('<Sz>')
    Sz_2 = DL_2.get_data_m('<Sz>')
    Sz_3 = DL_3.get_data_m('<Sz>')
    assert np.allclose(Sz_1, Sz_2) and np.allclose(Sz_2, Sz_3)
