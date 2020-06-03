"""Tools to temporarily cache parts of data to disk in order to free RAM.

"""

# Copyright 2020 TeNPy Developers, GNU GPLv3

import pickle

from . import hdf5_io
import collections

try:
    import h5py
except ImportError:
    h5py = None


class Hdf5CachedList(collections.abc.Sequence):
    """List-like container caching data to disc instead of keeping it in RAM.

    Instance of this class can replace lists.


    Parameters
    ----------
    L : int
        Desired length of the list, as returned by ``len(self)``.
    h5file : str | h5py.Group
        Filename for the file or directly the hdf5 file to be used for caching.
    keystring :
        Template for the keys of the different data sets

    Attributes
    ----------
    L : int
        Length.
    h5file : h5py.Group
        Hdf5 file/group to save the data in.

    """
    def __init__(self, L, h5file=None, keystring="data{0:d}"):
        L = int(L)
        self.L = L
        self.keystring = keystring
        self.keys = [keystring.format(i) for i in range(L)]
        if h5file is None:
            h5file = "cache.h5"
        if not isinstance(h5file, h5py.Group):
            h5file = h5py.File(h5file, 'w')
        self.h5file = h5file
        self.saver = hdf5_io.Hdf5Saver(self.h5file)
        self.loader = hdf5_io.Hdf5Loader(self.h5file)

    def __getitem__(self, i):
        obj = self.loader.load(self.keys[i])
        self.loader.memo_load.clear()
        return obj

    def __setitem__(self, i, obj):
        key = self.keys[i]
        if key in self.h5file:
            del self.h5file[key]
        self.saver.save(obj, key)
        self.saver.memo_save.clear()

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

    def __del__(self):
        fn = self.h5file.filename
        self.h5file.close()
        os.path.remove(filename)
