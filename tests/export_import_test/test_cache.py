# Copyright 2019-2020 TeNPy Developers, GNU GPLv3

import numpy as np
import tempfile
import os

import io_test
from tenpy.tools.cache import Hdf5CachedList

try:
    import h5py
except ImportError:
    h5py = None


def test_hdf5_export_import():
    """Try subsequent export and import to pickle."""
    L = 10
    data = [{'eye': np.eye(d), 'random': np.random.random([d])} for d in range(1, L + 1)]

    data_recoverd = []
    with tempfile.TemporaryDirectory() as tdir:
        filename = os.path.join(tdir, 'test_cache.hdf5')
        cache = Hdf5CachedList(L, filename)
        for i in range(L):
            cache[i] = data[i]
        for i in range(L):
            data_recoverd.append(cache[i])
    io_test.assert_equal_data(data_recoverd, data)
