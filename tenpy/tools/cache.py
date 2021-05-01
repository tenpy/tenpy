"""Tools to temporarily cache parts of data to disk in order to free RAM.


"""

# Copyright 2020 TeNPy Developers, GNU GPLv3

import pickle

from . import hdf5_io
import collections
import os

try:
    import h5py
except ImportError:
    h5py = None

__all__ = ["Hdf5CacheFile", "CachedList"]


class Hdf5CacheFile:
    """Temporary HDF5 file that gets deleted when cleared out of Memory."""
    def __init__(self, filename="cache.h5"):
        self.h5file = h5py.File(filename, 'w')

    def make_ListCache(self, data, subgroup):
        subgr = self.h5file.create_group(subgroup)
        return CachedList.from_list(data, subgr)

    def __del__(self):
        fn = str(self.h5file.filename)
        self.h5file.close()
        os.remove(fn)


class CachedList(collections.abc.Sequence):
    """List-like container caching data to disc instead of keeping it in RAM.

    Instances of this class can replace lists.

    Parameters
    ----------
    L : int
        Desired length of the list, as returned by ``len(self)``.
    h5file : :class:`h5py.Group`
        The hdf5 file to be used for caching.
    keystring :
        Template for the keys of the different data sets

    Attributes
    ----------
    L : int
        Length.
    h5file : h5py.Group
        Hdf5 file/group to save the data in.
    keystring : str
        Template for `keys`.
    keys : list of str
        Keys for the different data sets.
    saver : :class:`~tenpy.tools.hdf_io.Hdf5Saver`
        Loading class.
    loader : :class:`~tenpy.tools.hdf_io.Hdf5Loader`
        Saving class.
    """
    def __init__(self, L, h5file, keystring="{0:d}"):
        L = int(L)
        self.L = L
        self.keystring = keystring
        self.keys = [keystring.format(i) for i in range(L)]
        self.h5file = h5file
        self.saver = hdf5_io.Hdf5Saver(self.h5file)
        self.loader = hdf5_io.Hdf5Loader(self.h5file)

    @classmethod
    def from_list(cls, data, h5file=None, keystring="{0:d}"):
        res = cls(len(data), h5file, keystring)
        for i, entry in enumerate(data):
            res[i] = entry
        return res

    def __getitem__(self, i):
        obj = self.loader.load(self.keys[i])
        print("get", "/".join([self.h5file.name, self.keys[i]]), repr(obj)[:30])
        self.loader.memo_load.clear()
        return obj

    def __setitem__(self, i, obj):
        key = self.keys[i]
        if key in self.h5file:
            del self.h5file[key]
        self.saver.save(obj, key)
        self.saver.memo_save.clear()
        print("set", "/".join([self.h5file.name, key]), repr(obj)[:30])

    def __delitem__(self, i):
        del self.h5file[self.keys[i]]

    def __len__(self):
        return self.L

    def __insert__(self, i, val):
        if i < 0:
            i = self.L + 1 + i
        self.keys.append(self.keystring.format(self.L))
        for j in range(self.L, i, -1):
            self.h5file[self.keys[j]] = self.h5file[self.keys[j - 1]]
        self.L = self.L + 1
        self.saver.save(self.keys[i], value)
        self.saver.memo_save.clear()
