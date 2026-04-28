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

import numpy as np

import cyten
from cyten import tensors
from cyten.symmetries import _symmetries, spaces

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


def gen_example_data(version=cyten.version.full_version):
    if '+' in version:
        version = version.split('+')[0]  # discard '+GITHASH' from version
    testU1 = U1_sym_test_tensor()
    testSU2 = SU2_sym_test_tensor()
    testrand = create_test_random_symmetric_tensor()
    testdiag = create_test_random_diagonal_tensor()
    testU1.test_sanity()
    testSU2.test_sanity()
    testrand.test_sanity()
    testdiag.test_sanity()

    data = {'TestU1': testU1, 'TestSU2': testSU2, 'Testrand': testrand, 'Testdiag': testdiag}
    if parse_version(version) >= parse_version('0.5.0.dev25'):
        data.update(
            {
                'version': version,
                'None': None,
                'scalars': [0, np.int64(1), 2.0, np.float64(3.0), 4.0j, 'five'],
                'arrays': [np.array([6, 66]), np.array([]), np.zeros([])],
                'iterables': [[], [11, 12], tuple([]), tuple([1, 2, 3]), set([]), set([1, 2, 3])],
                'recursive': [0, None, 2, [3, None, 5]],
                'dict_complicated': {0: 1, 'asdf': 2, (1, 2): '3'},
                'exportable': cyten.tools.hdf5_io.Hdf5Exportable(),
                'range': range(2, 8, 3),
                'dtypes': [np.dtype('int64'), np.dtype([('a', np.int32, 8), ('b', np.float64, 5)])],
                'psi': (),
            }
        )
        data['recursive'][3][1] = data['recursive'][1] = data['recursive']
        data['exportable'].some_attr = 'something'

    return data


def SU2_sym_test_tensor():
    sym = _symmetries.SU2Symmetry()
    spin_half = spaces.ElementarySpace(sym, np.array([[1]]))
    backend = cyten.get_backend(sym, 'numpy')

    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

    heisenberg_4 = sum(si[:, :, None, None] * si[None, None, :, :] for si in [sx, sy, sz])  # [p1, p1*, p2, p2*]
    heisenberg_4 = np.transpose(heisenberg_4, [0, 2, 3, 1])  # [p1, p2, p2*, p1*]

    tens = tensors.SymmetricTensor.from_dense_block(
        heisenberg_4,
        codomain=[spin_half, spin_half],
        domain=[spin_half, spin_half],
        backend=backend,
        labels=[['p1', 'p2'], ['p1*', 'p2*']],
        tol=10**-8,
    )

    return tens


def U1_sym_test_tensor():
    sym = _symmetries.U1Symmetry()
    spin_half = spaces.ElementarySpace(sym, np.array([[1]]))
    backend = cyten.get_backend(sym, 'numpy')

    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

    heisenberg_4 = sum(si[:, :, None, None] * si[None, None, :, :] for si in [sx, sy, sz])  # [p1, p1*, p2, p2*]
    heisenberg_4 = np.transpose(heisenberg_4, [0, 2, 3, 1])  # [p1, p2, p2*, p1*]

    tens = tensors.SymmetricTensor.from_dense_block(
        heisenberg_4,
        codomain=[spin_half, spin_half],
        domain=[spin_half, spin_half],
        backend=backend,
        labels=[['p1', 'p2'], ['p1*', 'p2*']],
        tol=10**-8,
    )

    return tens


def create_test_random_symmetric_tensor():
    sym = _symmetries.SU2Symmetry()
    sec = np.random.choice(int(1.3 * 3), replace=False, size=(3, 1))

    x1 = spaces.ElementarySpace.from_defining_sectors(sym, sec)
    xp1 = spaces.TensorProduct([x1] * 4)
    xp2 = spaces.TensorProduct([x1] * 2)
    dat = np.random.normal(size=(11, 11, 11, 11, 11, 11))

    tens = tensors.SymmetricTensor.from_dense_block(
        dat, xp1, xp2, tol=10**20, labels=[['p1', 'p2', 'p3', 'p4'], ['p1*', 'p2*']]
    )
    return tens


def create_test_random_diagonal_tensor():
    sym = _symmetries.SU2Symmetry()
    sec = np.random.choice(int(1.3 * 3), replace=False, size=(3, 1))

    x1 = spaces.ElementarySpace.from_defining_sectors(sym, sec)
    dat = np.diag(np.random.normal(size=2000))

    tens = tensors.DiagonalTensor.from_dense_block(dat, x1, tol=10**20)

    return tens


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
    elif isinstance(data_expected, np.ndarray):
        assert np.linalg.norm(data_imported - data_expected) == 0.0  # should be exactly equal!
    elif isinstance(data_expected, np.ndarray):
        np.testing.assert_array_equal(data_imported, data_expected)
    elif isinstance(data_expected, (int, float, np.int64, np.float64, complex, str)):
        assert data_imported == data_expected
    elif isinstance(data_expected, (types.FunctionType, type)):
        # global variables where no copy should be made
        assert data_imported is data_expected


def get_datadir_filename(template='pickled_from_tenpy_{0}.pkl'):
    """Determine filename for export to `datadir`."""
    if not os.path.isdir(datadir):
        os.mkdir(datadir)
    version = cyten.version.full_version
    fn = template.format(version)
    filename = os.path.join(datadir, fn)
    if os.path.exists(filename):
        raise ValueError('File already exists: ' + filename)
    print('export to datadir: ', fn)
    return filename
