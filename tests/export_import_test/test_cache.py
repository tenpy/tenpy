# Copyright 2019-2020 TeNPy Developers, GNU GPLv3

import numpy as np
import numpy.testing as npt
import tempfile
import os
import pytest

from tenpy.tools.cache import Hdf5Cache, PickleCache, DictCache

try:
    import h5py
except ImportError:
    h5py = None


def test_DictCache(CacheClass=DictCache, **kwargs):
    data = dict([(f"x{i:d}", np.arange(i + 5)) for i in range(5)])
    datasub = dict([(f"x{i:d}", np.ones(i + 1)) for i in range(5)])  # same keys, different values
    with CacheClass.open(kwargs) as cache:
        for k in ['filename', 'directory']:
            if k in kwargs and kwargs[k] is not None:
                assert os.path.exists(kwargs[k])
        cache.set_short_term_keys("x0", "x1")
        for key in ["x0", "x4", "x3", "x2"]:
            cache[key] = data[key]
        cache.preload("x2")
        assert cache.get("x1", None) is None
        for key in ["x0", "x2"]:
            loaded = cache[key]
            npt.assert_equal(loaded, data[key])
        cache["x1"] = data["x1"]
        with cache.create_subcache("subcache") as sub:
            sub.set_short_term_keys("x4")
            keys = ["x1", "x0", "x4"]
            for key in keys:
                sub[key] = datasub[key]
            for key in reversed(keys):
                loaded = sub[key]
                npt.assert_equal(loaded, datasub[key])
        del cache["x0"]
        assert "x0" not in cache
        assert "x1" in cache
        cache.set_short_term_keys("x1", "x2")
        cache["x0"] = data["x0"]
        for k, v in data.items():
            loaded = cache[key]
            npt.assert_equal(loaded, data[key])
    for k in ['filename', 'directory']:
        if k in kwargs and kwargs[k] is not None:
            assert not os.path.exists(kwargs[k])  # did the cleanup work?


@pytest.mark.skipif(h5py is None, reason="h5py not available")
def test_Hdf5Cache():
    with tempfile.TemporaryDirectory() as tdir:
        filename = os.path.join(tdir, 'tmp_Hdf5Cache.h5')
        test_DictCache(CacheClass=Hdf5Cache, filename=filename)
    test_DictCache(CacheClass=Hdf5Cache)  # path = None -> tempfile use in tenpy.tools.cache


def test_PickleCache():
    with tempfile.TemporaryDirectory() as tdir:
        subdir = os.path.join(tdir, 'tmp_PickleCache')
        test_DictCache(CacheClass=PickleCache, directory=subdir)
    test_DictCache(CacheClass=PickleCache)  # path = None -> tempfile use in tenpy.tools.cache
