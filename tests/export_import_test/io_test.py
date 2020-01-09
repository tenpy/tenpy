"""Provide functionality for import (input) and export (output) tests.

The subfolder "data/" (given by `datadir`) in the directory of this file is used for storage of
test files to be imported.
Files in this subfolder are detected by tests and checked for correct import.
In that way, we can test the import of data files exported by old TeNPy versions.
To generate the files, you can call the ``test_*.py`` files in this folder
manually, e.g. ``python test_pickle.py``. This will generate


If you call the ``test_*.py`` files in this folder manually, they exports data to files
in the subfolder `data`
in a subfolder `data`
inside the directory of this file itself.
"""
# Copyright 2019-2020 TeNPy Developers, GNU GPLv3

import numpy as np
import os
import tenpy
import tenpy.linalg.np_conserved as npc

try:
    from packaging.version import parse as parse_version  # part of setuptools
except ImportError:

    def parse_version(version_str):
        """Hack to allow very basic version comparison.

        (This does not handle correctly cases of alpha, beta, pre and development releases.)"""
        return version_str.split('.')


__all__ = [
    'datadir', 'datadir_files', 'gen_example_data', 'assert_equal_data', 'get_datadir_filename'
]

datadir = os.path.join(os.path.dirname(__file__), 'data')
datadir_files = []
if os.path.isdir(datadir):
    datadir_files = [fn for fn in os.listdir(datadir) if fn.endswith('.pkl')]


def gen_example_data(version=tenpy.version.full_version):
    if '+' in version:
        version = version.split('+')[0]  # discard '+GITHASH' from version
    if parse_version(version) < parse_version('0.5.0.dev25'):
        s = tenpy.networks.site.SpinHalfSite()
        data = {
            'SpinHalfSite': s,
            'trivial_array': npc.Array.from_ndarray_trivial(np.arange(20).reshape([4, 5])),
            'Sz': s.Sz
        }
    else:
        s = tenpy.networks.site.SpinHalfSite()
        data = {
            'SpinHalfSite': s,
            'trivial_array': npc.Array.from_ndarray_trivial(np.arange(20).reshape([4, 5])),
            'Sz': s.Sz,
            'version': version
        }
    return data


def assert_equal_data(data_imported, data_expected):
    """Check that the imported data is as expected."""
    keys_imported = sorted(list(data_imported.keys()))
    assert set(keys_imported) == set(data_expected.keys())
    for ki in keys_imported:
        print("key:", ki)
        vk = data_expected[ki]
        vi = data_imported[ki]
        if hasattr(vk, 'test_sanity'):  # if vk has vk.test_sanity(), vi should have it, too!
            vi.test_sanity()
        assert isinstance(vi, vk.__class__)
        if isinstance(vi, npc.Array):
            assert npc.norm(vi - vk) == 0.  # should be exactly equal!


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
