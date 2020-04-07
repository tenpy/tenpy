"""Test import and output with pickle."""

import io_test
import os
import pickle
import pytest
import warnings
import tempfile

datadir_pkl = [f for f in io_test.datadir_files if f.endswith('.pkl')]


def export_to_datadir():
    filename = io_test.get_datadir_filename("exported_from_tenpy_{0}.pkl")
    data = io_test.gen_example_data()
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    # done


def test_pickle():
    """Try subsequent export and import to pickle."""
    data = io_test.gen_example_data()
    with tempfile.TemporaryDirectory() as tdir:
        filename = 'test.pkl'
        with open(os.path.join(tdir, filename), 'wb') as f:
            pickle.dump(data, f)
        with open(os.path.join(tdir, filename), 'rb') as f:
            data_imported = pickle.load(f)
    io_test.assert_equal_data(data_imported, data)


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


if __name__ == "__main__":
    export_to_datadir()
