"""Provide functionality for import (input) and export (output) tests.

The subfolder "data/" (given by `datadir`) in the directory of this file is used for storage of
test files to be imported. Files in this subfolder are detected by tests and checked for correct
import. In that way, we can test the import of data files exported by old TeNPy versions. To
generate the files, you can call the ``test_*.py`` files in this folder manually, e.g., ``python
test_pickle.py``. This will generate the files with pre-defined data (see :func:`gen_example_data`)
and the tenpy version in the filename.
"""

# Copyright (C) TeNPy Developers, Apache license
import os
import types

import lasyten as lasy
import numpy as np

import tenpy
import tenpy.models.xxz_chain
from tenpy.networks.mpo import MPO
from tenpy.networks.mps import MPS

try:
    from packaging.version import parse as parse_version  # part of setuptools
except ImportError:

    def parse_version(version_str):
        """Hack to allow very basic version comparison.

        Does not handle cases of alpha, beta, pre and dev releases correctly.
        """
        return version_str.split('.')


__all__ = ['datadir', 'datadir_files', 'gen_example_data', 'assert_equal_data', 'get_datadir_filename']

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
    if '+' in version:
        version = version.split('+')[0]  # discard '+GITHASH' from version
    s = lasy.sites.SpinSite(S=0.5, conserve='Sz')
    Sz = s.get_op('Sz')
    data = {
        'SpinHalfSite': s,
        'trivial_tensor': Sz,
        'Sz': Sz,
    }
    if parse_version(version) >= parse_version('2.0.0-alpha.dev0'):
        psi = MPS.from_product_state([s] * 4, ['up', 'down', 'up', 'down'], bc='finite', unit_cell_width=4)
        psi.test_sanity()
        M = tenpy.models.xxz_chain.XXZChain({'L': 4, 'bc_MPS': 'finite'})
        data.update(
            {
                'version': version,
                'None': None,
                'scalars': [0, np.int64(1), 2.0, np.float64(3.0), 4.0j, 'five'],
                'arrays': [np.array([6, 66]), np.array([]), np.zeros([])],
                'iterables': [[], [11, 12], tuple([]), tuple([1, 2, 3]), set([]), set([1, 2, 3])],
                'recursive': [0, None, 2, [3, None, 5]],
                'dict_complicated': {0: 1, 'asdf': 2, (1, 2): '3'},
                'exportable': tenpy.tools.hdf5_io.Hdf5Exportable(),
                'range': range(2, 8, 3),
                'dtypes': [np.dtype('int64'), np.dtype([('a', np.int32, 8), ('b', np.float64, 5)])],
                'psi': psi,
                'H_mpo': M.get_H_MPO(),
                # Full CouplingModel is not picklable yet (lasyten Coupling); H_MPO covers I/O.
            }
        )
        data['recursive'][3][1] = data['recursive'][1] = data['recursive']
        data['exportable'].some_attr = 'something'
        dummy = DummyClass()
        data['dummy'] = dummy
        event_handler = tenpy.tools.events.EventHandler('obj')
        event_handler.connect(dummy_function)
        event_handler.connect(dummy.dummy_append)
        data['event_handler'] = event_handler
    assert len(data) > 3, 'version parsing worked and we included data for >=2.0.0-alpha'
    return data


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
    if data_expected is None:
        assert data_imported is None
        return
    assert isinstance(data_imported, type(data_expected))
    if hasattr(data_expected, 'test_sanity'):
        data_imported.test_sanity()
    if isinstance(data_expected, dict):
        assert set(data_imported.keys()) == set(data_expected.keys())
        if max_recursion_depth > 0:
            for ki in data_imported.keys():
                assert_equal_data(data_imported[ki], data_expected[ki], max_recursion_depth - 1)
    elif isinstance(data_expected, (list, tuple)):
        if max_recursion_depth > 0:
            for vi, ve in zip(data_imported, data_expected):
                assert_equal_data(vi, ve, max_recursion_depth - 1)
    elif isinstance(data_expected, set):
        assert data_imported == data_expected
    elif isinstance(data_expected, lasy.Tensor):
        assert lasy.almost_equal(data_imported, data_expected)
    elif isinstance(data_expected, MPS):
        assert data_imported.L == data_expected.L
        assert data_imported.bc == data_expected.bc
        for B_i, B_e in zip(data_imported._B, data_expected._B):
            assert lasy.almost_equal(B_i, B_e)
    elif isinstance(data_expected, MPO):
        assert data_imported.L == data_expected.L
        assert data_imported.bc == data_expected.bc
        for W_i, W_e in zip(data_imported._W, data_expected._W):
            assert lasy.almost_equal(W_i, W_e)
    elif isinstance(data_expected, lasy.Site):
        assert data_imported.leg == data_expected.leg
        assert data_imported.state_labels == data_expected.state_labels
    elif isinstance(data_expected, np.ndarray):
        np.testing.assert_array_equal(data_imported, data_expected)
    elif isinstance(
        data_expected, (bool, int, float, np.integer, np.floating, complex, str, range, np.dtype, lasy.Dtype)
    ):
        assert data_imported == data_expected
    elif isinstance(data_expected, (types.FunctionType, type)):
        # global variables where no copy should be made
        assert data_imported is data_expected
    elif isinstance(data_expected, (DummyClass, tenpy.tools.hdf5_io.Hdf5Exportable, tenpy.tools.events.EventHandler)):
        # structural equality is enough; event handler checked separately
        pass
    elif hasattr(data_expected, '__eq__'):
        assert data_imported == data_expected
    else:
        raise TypeError(f'unsupported type for assert_equal_data: {type(data_expected)!r}')


def get_datadir_filename(template='pickled_from_tenpy_{0}.pkl'):
    """Determine filename for export to `datadir`."""
    if not os.path.isdir(datadir):
        os.mkdir(datadir)
    version = tenpy.version.full_version
    fn = template.format(version)
    filename = os.path.join(datadir, fn)
    if os.path.exists(filename):
        raise ValueError('File already exists: ' + filename)
    print('export to datadir: ', fn)
    return filename
