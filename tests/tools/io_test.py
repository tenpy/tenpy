"""Provide functionality for import (input) and export (output) tests.

The subfolder "data/" (given by `datadir`) in the directory of this file is used for storage of
test files to be imported. Files in this subfolder are detected by tests and checked for correct
import. In that way, we can test the import of data files exported by old TeNPy versions. To
generate the files, you can call the ``test_*.py`` files in this folder manually, e.g., ``python
test_pickle.py``. This will generate the files with pre-defined data (see :func:`gen_example_data`)
and the tenpy version in the filename.
"""
# Copyright (C) TeNPy Developers, GNU GPLv3

import os
import tenpy

try:
    from packaging.version import parse as parse_version  # part of setuptools
except ImportError:

    def parse_version(version_str):
        """Hack to allow very basic version comparison.

        Does not handle cases of alpha, beta, pre and dev releases correctly.
        """
        return version_str.split('.')


__all__ = [
    'datadir', 'datadir_files', 'gen_example_data', 'assert_equal_data', 'get_datadir_filename'
]

datadir = os.path.join(os.path.dirname(__file__), 'data')
datadir_files = []
if os.path.isdir(datadir):
    datadir_files = os.listdir(datadir)


class DummyClass:
    """Used to test exporting a custom class."""
    def __init__(self):
        self.data = []

    def dummy_append(self, obj):
        self.data.append(obj)


_dummy_function_arg_memo = []


def dummy_function(obj):
    _dummy_function_arg_memo.append(obj)


def gen_example_data(version=tenpy.version.full_version):
    raise NotImplementedError


def assert_event_handler_example_works(data):
    if 'event_handler' not in data:
        return
    eh = data['event_handler']
    l1 = len(data['dummy'].data)
    l2 = len(_dummy_function_arg_memo)
    assert l1 == l2
    obj = DummyClass()
    eh.emit(obj)
    for memo in [data['dummy'].data, _dummy_function_arg_memo]:
        assert len(memo) == l1 + 1
        assert memo[-1] is obj
        memo.clear()


def assert_equal_data(data_imported, data_expected, max_recursion_depth=10):
    """Check that the imported data is as expected."""
    # assert isinstance(data_imported, type(data_expected))
    # if hasattr(data_expected, 'test_sanity'):
    #     data_imported.test_sanity()
    # if isinstance(data_expected, dict):
    #     assert set(data_imported.keys()) == set(data_expected.keys())
    #     if max_recursion_depth > 0:
    #         for ki in data_imported.keys():
    #             assert_equal_data(data_imported[ki], data_expected[ki], max_recursion_depth - 1)
    # elif isinstance(data_expected, (list, tuple)):
    #     if max_recursion_depth > 0:
    #         for vi, ve in zip(data_imported, data_expected):
    #             assert_equal_data(vi, ve, max_recursion_depth - 1)
    # elif isinstance(data_expected, npc.Array):
    #     assert npc.norm(data_imported - data_expected) == 0.  # should be exactly equal!
    # elif isinstance(data_expected, np.ndarray):
    #     np.testing.assert_array_equal(data_imported, data_expected)
    # elif isinstance(data_expected, (int, float, np.int64, np.float64, complex, str)):
    #     assert data_imported == data_expected
    # elif isinstance(data_expected, (types.FunctionType, type)):
    #     # global variables where no copy should be made
    #     assert data_imported is data_expected


def get_datadir_filename(template="pickled_from_tenpy_{0}.pkl"):
    """Determine filename for export to `datadir`."""
    if not os.path.isdir(datadir):
        os.mkdir(datadir)
    version = tenpy.version.full_version
    fn = template.format(version)
    filename = os.path.join(datadir, fn)
    if os.path.exists(filename):
        raise ValueError("File already exists: " + filename)
    print("export to datadir: ", fn)
    return filename