"""Test output to and import from hdf5."""

import io_test
import os
import pickle
import pytest
import warnings
import tempfile
from tenpy.tools import hdf5_io
import numpy as np

datadir_hdf5 = [f for f in io_test.datadir_files if f.endswith('.hdf5')]

try:
    import h5py
except ImportError:
    h5py = None


def export_to_datadir():
    filename = io_test.get_datadir_filename("exported_from_tenpy_{0}.hdf5")
    data = io_test.gen_example_data()
    with h5py.File(filename, 'w') as f:
        hdf5_io.save_to_hdf5(f, data)


@pytest.mark.filterwarnings(r'ignore:Hdf5Saver.* object of type.*:UserWarning')
@pytest.mark.skipif(h5py is None, reason="h5py not available")
def test_hdf5_export_import():
    """Try subsequent export and import to pickle."""
    data = io_test.gen_example_data()
    io_test.assert_event_handler_example_works(data)  #if this fails, it's not import/export
    with tempfile.TemporaryDirectory() as tdir:
        filename = 'test.hdf5'
        with h5py.File(os.path.join(tdir, filename), 'w') as f:
            hdf5_io.save_to_hdf5(f, data)
        with h5py.File(os.path.join(tdir, filename), 'r') as f:
            data_imported = hdf5_io.load_from_hdf5(f)
    io_test.assert_equal_data(data_imported, data)
    io_test.assert_event_handler_example_works(data_imported)


@pytest.mark.skipif(h5py is None, reason="h5py not available")
@pytest.mark.parametrize('fn', datadir_hdf5)
def test_import_from_datadir(fn):
    print("import ", fn)
    filename = os.path.join(io_test.datadir, fn)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        with h5py.File(filename, 'r') as f:
            data = hdf5_io.load_from_hdf5(f)
    if 'version' in data:
        data_expected = io_test.gen_example_data(data['version'])
    else:
        data_expected = io_test.gen_example_data('0.4.0')
    io_test.assert_equal_data(data, data_expected)
    io_test.assert_event_handler_example_works(data)


if __name__ == "__main__":
    export_to_datadir()
