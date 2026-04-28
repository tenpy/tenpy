"""Test output to and import from hdf5."""

import os
import warnings

import io_test
import pytest

from cyten.tools import hdf5_io

h5py = pytest.importorskip('h5py')

datadir_hdf5 = [f for f in io_test.datadir_files if f.endswith('.hdf5')]


def export_to_datadir():
    filename = io_test.get_datadir_filename('exported_from_tenpy_{0}.hdf5')
    data = io_test.gen_example_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.filterwarnings('ignore', category=UserWarning)
        with h5py.File(filename, 'w') as f:
            hdf5_io.save_to_hdf5(f, data)
    for w in caught:
        msg = str(w.message)
        expected = 'without explicit HDF5 format' in msg
        if expected:
            expected = any(
                t in msg
                for t in [
                    'io_test.DummyClass',
                    'tenpy.tools.events.EventHandler',
                    'tenpy.tools.events.Listener',
                    'method',
                ]
            )
        if not expected:
            warnings.showwarning(w.message, w.category, w.filename, w.lineno, w.file, w.line)


@pytest.mark.filterwarnings(r'ignore:Hdf5Saver.* object of type.*:UserWarning')
def test_hdf5_export_import(make_compatible_space, compatible_backend, tmp_path):
    """Try subsequent export and import to pickle."""
    data = io_test.gen_example_data()
    io_test.assert_event_handler_example_works(data)  # if this fails, it's not import/export
    filename = tmp_path / 'test.hdf5'
    with h5py.File(str(filename), 'w') as f:
        hdf5_io.save_to_hdf5(f, data)
    with h5py.File(str(filename), 'r') as f:
        data_imported = hdf5_io.load_from_hdf5(f)
    io_test.assert_equal_data(data_imported, data)
    io_test.assert_event_handler_example_works(data_imported)


@pytest.mark.filterwarnings(r'ignore:Hdf5Saver.* object of type.*:UserWarning')
def test_hdf5_tensor_io(tmp_path):
    """Try subsequent export and import to hdf5."""
    testU1 = io_test.U1_sym_test_tensor()
    testSU2 = io_test.SU2_sym_test_tensor()
    testrand = io_test.create_test_random_symmetric_tensor()
    testdiag = io_test.create_test_random_diagonal_tensor()

    for data in [testU1, testSU2, testrand, testdiag]:
        io_test.assert_event_handler_example_works(data)  # if this fails, it's not import/export
        filename = tmp_path / 'test.hdf5'
        with h5py.File(str(filename), 'w') as f:
            hdf5_io.save_to_hdf5(f, data)
        with h5py.File(str(filename), 'r') as f:
            data_imported = hdf5_io.load_from_hdf5(f)
        io_test.assert_equal_data(data_imported, data)
        io_test.assert_event_handler_example_works(data_imported)


@pytest.mark.parametrize('fn', datadir_hdf5)
@pytest.mark.filterwarnings('ignore::FutureWarning')
def test_import_from_datadir(fn):
    print('import ', fn)
    filename = os.path.join(io_test.datadir, fn)
    with h5py.File(filename, 'r') as f:
        data = hdf5_io.load_from_hdf5(f)
    if 'version' in data:
        data_expected = io_test.gen_example_data(data['version'])
    else:
        data_expected = io_test.gen_example_data('0.4.0')
    io_test.assert_equal_data(data, data_expected)
    io_test.assert_event_handler_example_works(data)
