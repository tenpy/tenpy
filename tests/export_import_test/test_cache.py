# Copyright 2019-2020 TeNPy Developers, GNU GPLv3

import numpy as np
import tempfile
import os
import pytest

from tenpy.tools.cache import Hdf5CacheFile

try:
    import h5py
except ImportError:
    h5py = None


@pytest.mark.skipif(h5py is None, reason="h5py not available")
def test_hdf5_cached_list():
    L = 10
    data = [{'eye': np.eye(d), 'random': np.random.random([d])} for d in range(1, L + 1)]
    result = [np.linalg.norm(d['random']) for d in data]

    with tempfile.TemporaryDirectory() as tdir:
        filename = os.path.join(tdir, 'hdf5_cached_list.hdf5')
        cache = Hdf5CacheFile(filename)
        lst = cache.make_ListCache(data, "/data")
        for i in range(L):
            data = lst[i]
            data['norm'] = np.linalg.norm(data['random'])
            lst[i] = data
            lst[i] = data
            assert lst[i]['norm'] == result[i]
    # done
