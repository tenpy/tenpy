"""Tools to temporarily cache parts of data to disk in order to free RAM.

The :class:`DictCache` provides a dictionary-like interface to handle saving some data to disk.
While the :class:`DictCache` itself actually keeps everything in memory,
the subclasses store the provided data to disk for future lookup in order to free memory.
Any cache should be handled like a file object that needs to be closed after use;
this is easiest done through a ``with`` statement, see the example in :class:`DictCache`.
"""

# Copyright 2021 TeNPy Developers, GNU GPLv3

import pickle
import shutil
import tempfile
import logging
logger = logging.getLogger(__name__)

from .misc import find_subclass
from .hdf5_io import load_from_hdf5, save_to_hdf5
import collections
import os
import pathlib
from .params import asConfig

__all__ = ["DictCache", "PickleCache", "Hdf5Cache"]


class DictCache(collections.abc.MutableMapping):
    """Cache with dict-like interface keeping everything in RAM.

    This class has a dict-like interface to allow storing data; it just keeps everything in RAM
    (actually in a dict).
    However, it also serves as a base class for :class:`PickleCache` and :class:`Hdf5Cache`,
    which allow to save data that isn't needed for a while (e.g. environments for DMRG) to disk,
    in order to free RAM. :meth:`set_short_term_keys` allows to define which keys should
    nevertheless (in addition to saving them to disk) be kept in RAM until the next updating call
    to :meth:`set_short_term_keys` in order to avoid unnecessary read/write cycles.


    Examples
    --------
    The cache has as dict-like interface accepting strings (acceptable as file names) as keys.

    >>> cache = DictCache()
    >>> cache['a'] = 1
    >>> cache['b'] = 2
    >>> assert cache['a'] == 1
    >>> assert cache.get('b') == 2
    >>> "b" in cache
    True
    >>> "c" in cache
    False
    >>> assert cache.get('c', default=None) is None

    Subclasses need to create a file, so you can use it as a context manager in a with statement

    >>> with PickleCache.open(dict(directory="temp_cache", delete=True)) as cache:
    ...     cache['a'] = 1  # use as before
    ... # cache.close() was called here, don't use the cache anymore
    """
    #: if True, the class actually keeps everything in RAM instead of saving things to disk
    dummy_cache = True

    def __init__(self):
        self._long_term_keys = set([])
        self._short_term_cache = {}
        self._short_term_keys = set([])
        self._long_term_cache = {}
        self._delete_directory = None
        self._delete_file = None

    @classmethod
    def open(cls, options):
        """Interface for opening the necessary file/create directory and creating a class instance.

        For the :class:`DictCache`, this is a trivial method just initializing it,
        but for the :class:`Hdf5Cache` or :class:`PickleCache` it creates an Hdf5 file / directory
        where data can be saved to disk.

        .. warning ::
            Make sure that you call the :meth:`close` again after opening a cache in order to make
            sure that the temporary data on disk gets removed again.
            One way to ensure this is to use the class in a ``with`` statement::

                with Hdf5Cache.open('cache_filename.h5') as cache:
                    cache['my_data'] = (1, 2, 3)
                    assert cache['my_data'] == (1, 2, 3)
                # cache is closed again here, don't use it anymore

        Options
        -------
        The :class:`DictCache` takes no options, but for compatibility with subclasses,
        it accepts options as a dictionary.

        .. cfg:config :: DictCache

        Parameters
        ----------
        options : dict-like
            Completely ignored for a :class:`DictCache`.
        """
        return cls()

    def close(self):
        self._long_term_cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def create_subcache(self, name):
        """Create another instance of the same class as `self` based on the same disk resource.

        Parameters
        ----------
        name : str
            Name of a subdirectory for the :class:`PickleCache` or of a hdf5 subgroup for the
            :class:`Hdf5Cache`.

        Returns
        -------
        cache : :class:`DictCache`
            Another class instance of the same type as `self`.
        """
        return DictCache()

    def get(self, key, default=None):
        """Same as ``self[key]``, but return `default` if `key` is not in `self`."""
        if key not in self._long_term_keys:
            return default
        return self.__getitem__(key)

    def __getitem__(self, key):
        if key in self._short_term_cache:
            return self._short_term_cache[key]
        logger.debug("Cache._load_long_term(%r)", key)
        data = self._load_long_term(key)
        if key in self._short_term_keys:
            self._short_term_cache[key] = data
        return data

    def _load_long_term(self, key):
        """Interface for loading ``self[key]`` from disk in subclasses."""
        return self._long_term_cache[key]

    def __setitem__(self, key, val):
        self._long_term_keys.add(key)
        logger.debug("Cache._save_long_term(%r)", key)
        self._save_long_term(key, val)
        if key in self._short_term_keys:
            self._short_term_cache[key] = val

    def _save_long_term(self, key, val):
        """Interface for saving ``self[key]`` to disk in subclasses."""
        self._long_term_cache[key] = val

    def __delitem__(self, key):
        if key in self._long_term_keys:
            self._long_term_keys.remove(key)
            self._del_long_term(key)

    def _del_long_term(self, key):
        del self._long_term_cache[key]

    def __contains__(self, key):
        return key in self._long_term_keys

    def __iter__(self):
        return iter(self._long_term_keys)

    def __len__(self):
        return len(self._long_term_keys)

    def set_short_term_keys(self, *keys):
        """Set keys for data which should be kept in RAM for a while.

        Disk input/output is slow, so we want to avoid unnecessary read/write cycles.
        This method allows to specify keys the data of which should be kept in RAM after setting/
        reading, until the keys are updated with the next call to :meth:`set_short_term_keys`.
        The data is still *written* to disk in each ``self[key] = data``,
        but subsequent *reading* ``data = self[key]`` will be fast for the given keys.

        Parameters
        ----------
        *keys : str
            The keys for which data should be kept in RAM for quick short-term lookup.
        """
        self._short_term_keys = keys = set(keys)
        sc = self._short_term_cache
        for key in list(sc.keys()):
            if key not in keys:
                del sc[key]

    def preload(self, *keys, raise_missing=False):
        """Pre-load the data for one or more keys from disk to RAM.

        Parameters
        ----------
        *keys : str
            The keys which should be pre-loaded.
        raise_missing : bool
            Whether to raise a KeyError if a given key does not exist in `self`.
        """
        for key in keys:
            self._short_term_keys.add(key)
        for key in keys:
            if key not in self._short_term_cache:
                if key in self._long_term_keys:
                    logger.debug("Cache._load_long_term(%r) in preload", key)
                    self._short_term_cache[key] = self._load_long_term(key)
                elif raise_missing:
                    raise KeyError("trying to preload missing key")
        # done


class PickleCache(DictCache):
    """Version of :class:`DictCache` which saves long-term data on disk with :mod:`pickle`.

    Parameters
    ----------
    directory : path-like
        An existing directory within which pickle files will be saved for each `key`.
    """
    dummy_cache = False

    def __init__(self, directory):
        super().__init__()
        self._long_term_cache = pathlib.Path(directory)
        self._subdirs = set([])

    @classmethod
    def open(cls, options):
        """Create a directory and use it to initialize a :class:`PickleCache`.

        Options
        -------
        .. cfg:config : PickleCache

            directory : path-like | None
                Name of a directory to be created, in which pickle files will be stored for
                each `key`.
                If `None`, create a temporary directory with :mod:`tempfile` tools.
            delete : bool
                Whether to automatically remove the directory when closing the cache.
        """
        options = asConfig(options, "PickleCache")
        directory = options.get("directory", None)
        if directory is None:
            directory = tempfile.mkdtemp(prefix='tenpy_PickleCache')
            exist_ok = True
        else:
            exist_ok = False
        directory = pathlib.Path(directory)
        logger.info("PickleCache: create directory %s", directory)
        directory.mkdir(exist_ok=exist_ok)
        res = cls(directory)
        if options.get("delete", True):
            res._delete_directory = directory.absolute()
        return res

    def close(self):
        delete_dir = self._delete_directory
        if delete_dir is not None:
            self._delete_directory = None
            logger.info("PickleCache: cleanup/remove directory %s", delete_dir)
            shutil.rmtree(delete_dir)

    def create_subcache(self, name):
        subdir = self._long_term_cache / name
        if name in self._subdirs:
            # should exists already, clean up
            assert subdir.exists()
            shutil.rmtree(subdir)
        self._subdirs.add(name)
        subdir.mkdir(exist_ok=False)
        return PickleCache(subdir)

    def _load_long_term(self, key):
        key = key + '.pkl'
        with open(self._long_term_cache / key, 'rb') as f:
            data = pickle.load(f)
        return data

    def _save_long_term(self, key, value):
        key = key + '.pkl'
        with open(self._long_term_cache / key, 'wb') as f:
            pickle.dump(value, f)

    def _del_long_term(self, key):
        fn = self._long_term_cache / key
        if fn.exists():
            fn.remove()


class Hdf5Cache(DictCache):
    """Version of :class:`DictCache` which saves long-term data on disk with :mod:`h5py`.

    Parameters
    ----------
    h5group : :class:`Group`
        The hdf5 group in which data will be saved using
        :func:`~tenpy.tools.hdf5_io.save_to_hdf5` under the specified keys.

    """
    dummy_cache = False

    def __init__(self, h5group):
        super().__init__()
        self.h5gr = h5group
        self._file_obj = None
        self._close = False  # whether to close the h5gr.file in :meth:`close`

    @classmethod
    def open(cls, options):
        """Create an hdf5 file and use it to initialize an :class:`Hdf5Cache`.

        Options
        -------
        .. cfg:config : Hdf5Cache

            filename : path-like | None
                Filename of the Hdf5 file to be created.
                If `None`, create a temporary file with :mod:`tempfile` tools.
            delete : bool
                Whether to automatically remove the corresponding file when closing the cache.
            mode : str
                Filemode for opening the Hdf5 file.
        """
        options = asConfig(options, "Hdf5Cache")
        import h5py
        filename = options.get("filename", None)
        mode = options.get("mode", "w-")
        subgroup = options.get("subgroup", None)
        delete = options.get("delete", True)
        if filename is None:
            # h5py supports file-like objects, but this gives a python overhead for I/O.
            # hence h5py doc recommends using a temporary directory
            # and creating an hdf5 file inside that
            directory = tempfile.mkdtemp(prefix='tenpy_Hdf5Cache')
            logger.info("Hdf5Cache: created temporary directory %s", directory)
            filename = os.path.join(directory, "cache.h5")
        else:
            directory = None
            logger.info("Hdf5Cache: create temporary file %s", filename)
        f = h5py.File(filename, mode=mode)
        if subgroup is not None:
            if subgroup in f:
                f = subgroup[f]
            else:
                f = f.create_group(subgroup)
        res = cls(f)
        if delete:
            if directory is not None:
                # created temp directory -> need to clean that up!
                res._delete_directory = os.path.abspath(directory)
            else:
                res._delete_file = os.path.abspath(filename)
        res._close = True
        return res

    def close(self):
        if self._close:
            f = self.h5gr.file
            if f:
                f.close()
            # else: already closed, c.f. h5py documentation
        delete_file = self._delete_file
        if delete_file is not None:
            self._delete_file = None
            logger.info("Hdf5Cache: cleanup/remove file %s", delete_file)
            os.remove(delete_file)
        delete_dir = self._delete_directory
        if delete_dir is not None:
            self._delete_directory = None
            logger.info("Hdf5Cache: cleanup/remove temp directory %s", delete_dir)
            shutil.rmtree(delete_dir)

    def create_subcache(self, name):
        assert "/" not in name
        if name in self.h5gr:
            import h5py
            assert isinstance(h5gr[name], h5py.Group)
            del h5gr[name]
        return Hdf5Cache(self.h5gr.create_group(name))

    def _load_long_term(self, key):
        return load_from_hdf5(self.h5gr, key)

    def _save_long_term(self, key, value):
        save_to_hdf5(self.h5gr, value, key)

    def _del_long_term(self, key):
        if key in self.h5gr:
            del self.h5gr[key]
