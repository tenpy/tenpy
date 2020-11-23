"""Test import and output with pickle."""

import io_test
import os
import pickle
import pytest
import warnings
import tempfile
import time

import tenpy

datadir_pkl = [f for f in io_test.datadir_files if f.endswith('.pkl')]


class DummyAlgorithmSleep(tenpy.algorithms.algorithm.Algorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sleep_time = self.options.get('sleep_time', 0.2)

    def run(self):
        N_steps = self.options.get('N_steps', 5)
        self.dummy_value = N_steps**2
        for i in range(N_steps):
            self.checkpoint.emit(self)
            time.sleep(self.sleep_time)


def export_to_datadir():
    filename = io_test.get_datadir_filename("exported_from_tenpy_{0}.pkl")
    data = io_test.gen_example_data()
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    # done


def test_pickle():
    """Try subsequent export and import to pickle."""
    data = io_test.gen_example_data()
    io_test.assert_event_handler_example_works(data)  #if this fails, it's not import/export
    with tempfile.TemporaryDirectory() as tdir:
        filename = 'test.pkl'
        with open(os.path.join(tdir, filename), 'wb') as f:
            pickle.dump(data, f)
        with open(os.path.join(tdir, filename), 'rb') as f:
            data_imported = pickle.load(f)
    io_test.assert_equal_data(data_imported, data)
    io_test.assert_event_handler_example_works(data_imported)


@pytest.mark.parametrize('fn', datadir_pkl)
def test_import_from_datadir(fn):
    print("import ", fn)
    filename = os.path.join(io_test.datadir, fn)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    if 'version' in data:
        data_expected = io_test.gen_example_data(data['version'])
    else:
        data_expected = io_test.gen_example_data('0.4.0')
    io_test.assert_equal_data(data, data_expected)
    io_test.assert_event_handler_example_works(data)


def test_simulation_export_import():
    """Try subsequent export and import to pickle."""
    sim_params = {
        'model_class':
        'XXZChain',
        'model_params': {
            'bc_MPS': 'infinite',
            'L': 2
        },
        'algorithm_class':
        'DummyAlgorithmSleep',
        'algorithm_params': {
            'N_steps': 3
        },  # only one step
        # 'initial_state_builder': 'KagomeInitialStateBuilder',
        'initial_state_params': {
            'method': 'lat_product_state',
            'product_state': [['up'], ['down']]
        },
        'connect_measurements': [
            ('tenpy.simulations.measurement', 'onsite_expectation_value', {
                'opname': 'Sz'
            }),
        ],
    }
    with tempfile.TemporaryDirectory() as tdir:
        sim_params['directory'] = tdir  # go into temporary directory to avoid leaving data behind
        sim_params['output_filename'] = filename = 'my_results.pkl'
        sim = tenpy.simulations.simulation.Simulation(sim_params)
        data_direct = sim.run()
        with open(os.path.join(tdir, filename), 'rb') as f:
            data_imported = pickle.load(f)
        assert 'psi' in data_direct
        assert '<Sz>' in data_direct['measurements']
        io_test.assert_equal_data(data_imported, data_direct)
        #TODO test shelving and resuming the simulation


if __name__ == "__main__":
    export_to_datadir()
