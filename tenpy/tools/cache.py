"""Tools to temporarily cache parts of data to disk in order to free RAM.

The :class:`DictCache` provides a dictionary-like interface to handle saving some data to disk.
While the :class:`DictCache` itself actually keeps everything in memory,
the subclasses store the provided data to disk for future lookup in order to free memory.
Any cache should be handled like a file object that needs to be closed after use;
this is easiest done through a ``with`` statement, see the example in :class:`DictCache`.
"""

# Copyright 2021 TeNPy Developers, GNU GPLv3

import pickle
import numpy as np
import shutil
import tempfile
import collections
import os
import pathlib
import warnings
import logging
logger = logging.getLogger(__name__)

from .misc import find_subclass
from .thread import Worker
from .hdf5_io import load_from_hdf5, save_to_hdf5
from .params import asConfig

__all__ = ["DictCache", "CacheFile", "Storage", "PickleStorage", "Hdf5Storage", "ThreadedStorage"]


class DictCache(collections.abc.MutableMapping):
    """Cache with dict-like interface.

    The idea of the Cache is to save data that isn't needed for a while in a long-term
    :class:`Storage` container in order to free RAM.
    While the default :class:`Storage` is just an interface around a plain dictionary and hence
    actually *does* keep everything in RAM, this class is designed to handle also the case of
    other storage classes like the :class:`PickleCache` or :class:`Hdf5Cache`.
    To avoid unnecessary read-write cycles, it keeps some values in a "short-term" cache in
    memory, see :meth:`set_short_term_keys`.

    Using the :meth:`preload` method allows to generalize to the :class:`ThreadedDictCache`,
    which can save/load data in parallel without blocking the main thread excution while waiting
    for disk input/output.

    .. note ::
        To allow a proper closing of opened storage, it is highly recommended to use the
        :class:`DictCache` as a context manager in a ``with`` statement, see :func:`open`.

    Parameters
    ----------
    storage : :class:`Storage`
        Container for saving the data long-term.

    Attributes
    ----------
    long_term_storage : :class:`Storage`
        The storage passed during initialization.
    long_term_keys : set
        Keys of `long_term_storage` for which we have data.
    short_term_cache : dict
        Dictionary for keeping a "short-term" memory of the keys in `short_term_keys`.
    short_term_keys : set
        Keys for which data should be kept in `short_term_cache`.


    Examples
    --------
    The :class:`DictCache` has as dict-like interface accepting strings as keys.
    The keys should be acceptable as filenames and not contain "/".

    .. testsetup :: DictCache

        from tenpy.tools.cache import *
        import os

    .. doctest :: DictCache

        >>> cache = DictCache.trivial()
        >>> cache['a'] = 1
        >>> cache['b'] = 2
        >>> assert cache['a'] == 1
        >>> assert cache.get('b') == 2
        >>> "b" in cache
        True
        >>> "c" in cache
        False
        >>> assert cache.get('c', default=None) is None

    """
    def __init__(self, storage):
        self.long_term_storage = storage
        self.long_term_keys = set()
        self.short_term_cache = {}
        self.short_term_keys = set()

    @classmethod
    def trivial(cls):
        """Create a trivial storage that keeps everything in RAM."""
        return cls(Storage.open())

    def create_subcache(self, name):
        """Create another :class:`DictCache` based on the same storage resource.

        Uses :meth:`Storage.subcontainer` to create another storage container for a new
        :class:`DictCache`. The data is still *completely* owned by the top-most
        :class:`Storage` (in turn owned by the :class:`CacheFile`).
        Hence, closing the parent :class:`CacheFile` will close all :class:`DictCache` instances
        generated with `create_subcache`; accessing the data is no longer possible afterwards.

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
        return DictCache(self.long_term_storage.subcontainer(name))

    def get(self, key, default=None):
        """Same as ``self[key]``, but return `default` if `key` is not in `self`."""
        if key not in self.long_term_keys:
            return default
        return self.__getitem__(key)

    def __getitem__(self, key):
        if key in self.short_term_cache:
            return self.short_term_cache[key]
        if key not in self.long_term_keys:
            raise KeyError(f"{key!r} not existent in cache")
        logger.debug("Cache.long_term_storage.load(%r)", key)
        data = self.long_term_storage.load(key)
        if key in self.short_term_keys:
            self.short_term_cache[key] = data
        return data

    def __setitem__(self, key, val):
        self.long_term_keys.add(key)
        logger.debug("Cache.long_term_storage.save(%r)", key)
        self.long_term_storage.save(key, val)
        if key in self.short_term_keys:
            self.short_term_cache[key] = val

    def __delitem__(self, key):
        if key in self.long_term_keys:
            self.long_term_keys.remove(key)
            self.long_term_storage.delete(key)

    def __contains__(self, key):
        return key in self.long_term_keys

    def __iter__(self):
        return iter(self.long_term_keys)

    def __len__(self):
        return len(self.long_term_keys)

    def __bool__(self):
        """Whether the cache is open and ready for read/write."""
        return bool(self.long_term_storage)

    def set_short_term_keys(self, *keys):
        """Set keys for data which should be kept in RAM for a while.

        Disk input/output is slow, so we want to avoid unnecessary read/write cycles.
        This method allows to specify keys the data of which should be kept in RAM after setting/
        reading, until the keys are updated with the next call to :meth:`set_short_term_keys`.
        The data is still *written* to disk in each ``self[key] = data``,
        but (subsequent) *reading* ``data = self[key]`` will be fast for the given keys.

        Parameters
        ----------
        *keys : str
            The keys for which data should be kept in RAM for quick short-term lookup.
        """
        self.short_term_keys = keys = set(keys)
        sc = self.short_term_cache
        for key in list(sc.keys()):
            if key not in keys:
                del sc[key]

    def preload(self, *keys, raise_missing=False):
        """Pre-load the data for one or more keys from disk to RAM.

        Parameters
        ----------
        *keys : str
            The keys which should be pre-loaded. Are added to the :attr:`short_term_keys`.
        raise_missing : bool
            Whether to raise a KeyError if a given key does not exist in `self`.
        """
        for key in keys:
            self.short_term_keys.add(key)
        for key in keys:
            if key not in self.long_term_keys:
                if raise_missing:
                    raise KeyError("trying to preload missing entry " + repr(key))
            else:
                self.long_term_storage.preload(key)


class CacheFile(DictCache):
    """Subclass of :class:`DictCache` to handle opening and closing resources.

    You should open this class with the :meth:`open` method (or :meth:`trivial`),
    and make sure that you call :meth:`close` after usage.
    The easiest way to ensure this is to use a ``with`` statement, see :meth:`open`.
    """
    @classmethod
    def open(cls,
             storage_class="Storage",
             use_threading=False,
             delete=True,
             max_queue_size=2,
             **storage_kwargs):
        """Interface for opening a :class:`Storage` and creating a :class:`DictCache` from it.

        .. warning ::
            Make sure that you call the :meth:`close` method of the returned :class:`CacheFile`
            to close opened files and clean up temporary files/directories.
            One way to ensure this is to use the class in a ``with`` statement like this::

                with CacheFile.open(...) as cache:
                    cache['my_data'] = (1, 2, 3)
                    assert cache['my_data'] == (1, 2, 3)
                # cache is closed again here, don't use it anymore

        Parameters
        ----------
        storage_class : str
            Name for a subclass of :class:`Storage` to define how data is saved.
            Use just :class:`Storage` to keep things in RAM, or, e.g., :class:`PickleStorage`
            to actually save things to disk.
        use_threading : bool
            If True, use the :class:`ThreadedStorage` wrapper for thread-parallel disk I/O.
            In that case, you *need* to use the cache in a `with` statement (or manually call
            :meth:`__enter__` and :meth:`__exit__`).
        delete : bool
            If True, delete the opened file/directory after closing the cache.
        max_queue_size : int
            Only used for `use_threading`. Needs to be positive to limit the number of
            environments kept in RAM in case the disk is much slower then the actual update.
        **storage_kwargs :
            Further keyword arguments given to the :meth:`Storage.open` method of the
            `storage_class`.
        """
        StorageClass = find_subclass(Storage, storage_class)
        storage = StorageClass.open(delete=delete, **storage_kwargs)
        if use_threading:
            storage = ThreadedStorage.open(storage, max_queue_size=max_queue_size)
        return CacheFile(storage)

    def close(self):
        """Close the associated storage container and shut down."""
        self.long_term_storage.close()
        self.short_term_cache.clear()

    def __enter__(self):
        self.long_term_storage = self.long_term_storage.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.long_term_storage.__exit__(exc_type, exc, tb)
        self.short_term_cache.clear()


class Storage:
    """A container interface for saving data to disk.

    Subclasses implement different methods for :meth:`save` and :meth:`load`.
    The vanilla :class:`Storage` class is "trivial" in the sense that it actually doesn't save
    the data to disk, but keeps explicit references in RAM.
    """
    #: Whether the storage is actually kept in memory, instead of saving to disk.
    trivial = True

    def __init__(self):
        self._opened = True  # check this with bool(self)
        self._owns_resources = False
        self._subcontainers = []

    @classmethod
    def open(cls, delete=None):
        res = cls()
        res._owns_resources = True
        res.data = {}
        return res

    def close(self):
        """Close opened files, free memory and clean up temporary files/directories."""
        self._common_close()
        if self._owns_resources:
            self.data.clear()

    def _common_close(self):
        if not self._opened:
            raise ValueError("storage was already closed")
        self._opened = False
        for storage in self._subcontainers:
            storage.close()

    def __bool__(self):
        """Indicator whether the file is open"""
        return self._opened

    def subcontainer(self, name):
        """Create another instance of the same class saving in a subdirectory/subgroup.

        This method allows multiple :class:`DictCache` instance re-using open resources.
        Subcontainers will explcitly be closed when any of the parent containers (on which
        `subcontainer()` was called) is closed.
        """
        if not self._opened:
            raise ValueError("Trying to access closed storage")
        res = Storage.open()
        self._subcontainers.append(res)
        return res

    def load(self, key):
        """Interface for loading data from disk in subclasses."""
        if not self._opened:
            raise ValueError("Trying to access closed storage")
        return self.data[key]

    def save(self, key, val):
        """Interface for saving data to disk in subclasses."""
        if not self._opened:
            raise ValueError("Trying to access closed storage")
        self.data[key] = val

    def delete(self, key):
        """Interface for cleaning up a previously saved data from disk in subclasses."""
        if not self._opened:
            raise ValueError("Trying to access closed storage")
        del self.data[key]

    def preload(self, key):
        """Interface for preloading data into the given dictionary `into`.

        Only overriden in :class:`ThreadedStorage` for thread-parallelized pre-loading;
        in other cases it does nothing.
        """
        if not self._opened:
            raise ValueError("Trying to access closed storage")
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __repr__(self):
        closed = "" if self._opened else ", closed"
        return f"<{self.__class__.__name__} in RAM{closed}>"


class PickleStorage(Storage):
    """Subclass of :class:`Storage` which saves long-term data on disk with :mod:`pickle`.

    Parameters
    ----------
    directory : path-like
        An existing directory within which pickle files will be saved for each `key`.
    """
    trivial = False

    #: filename extension
    extension = '.pkl'

    def __init__(self, directory):
        super().__init__()
        self.directory = pathlib.Path(directory)
        self._delete_directory = None
        self._opened = self.directory.is_dir()

    @classmethod
    def open(cls, directory=None, tmpdir=None, delete=True):
        """Create a directory and use it to initialize a :class:`PickleCache`.

        Parameters
        ----------
        directory : path-like | None
            Name of a directory to be created, in which pickle files will be stored.
            If `None`, create a temporary directory with :mod:`tempfile` tools.
        tmpdir : path-like | None
            Only used if `directory` is None. Used as base `dir` for :func:`tempfile.mkdtemp`,
            i.e., a temporary directory is created within this path.
        delete : bool
            Whether to automatically remove the directory in :meth:`close`.
        """
        if directory is None:
            directory = tempfile.mkdtemp(prefix='tenpy_cache_' + cls.__name__, dir=tmpdir)
            exist_ok = True
        else:
            exist_ok = False
        directory = pathlib.Path(directory)
        logger.info("%s: create directory %s", cls.__name__, directory)
        directory.mkdir(exist_ok=exist_ok)
        res = cls(directory)
        res._owns_resources = True
        if delete:
            res._delete_directory = directory.absolute()
        return res

    def close(self):
        self._common_close()
        if self._owns_resources:
            delete_dir = self._delete_directory
            if delete_dir is not None:
                logger.info("%s: cleanup/remove directory %s", self.__class__.__name__, delete_dir)
                shutil.rmtree(delete_dir)

    def subcontainer(self, name):
        if not self._opened:
            raise ValueError("Trying to access closed storage")
        subdir = self.directory / name
        if subdir.exists():
            raise ValueError("Subcontainer with that name already exists")
        subdir.mkdir(exist_ok=False)
        res = self.__class__(subdir)
        self._subcontainers.append(res)
        return res

    def load(self, key):
        if not self._opened:
            raise ValueError("Trying to access closed storage")
        with open(self.directory / (key + self.extension), 'rb') as f:
            data = pickle.load(f)
        return data

    def save(self, key, value):
        if not self._opened:
            raise ValueError("Trying to access closed storage")
        with open(self.directory / (key + self.extension), 'wb') as f:
            pickle.dump(value, f)

    def delete(self, key):
        if not self._opened:
            raise ValueError("Trying to access closed storage")
        fn = self.directory / (key + self.extension)
        if fn.exists():
            fn.unlink()

    def __repr__(self):
        closed = "" if self._opened else ", closed"
        return f"<{self.__class__.__name__} at {self.directory!s}{closed}>"


class _NumpyStorage(PickleStorage):
    """Subclass of :class:`Storage` which saves long-term data on disk with :func:`numpy.save`.

    This class can **only** accept numpy arrays to be stored.

    Parameters
    ----------
    directory : path-like
        An existing directory within which numpy files will be saved for each `key`.
    """
    extension = '.npy'

    def load(self, key):
        if not self._opened:
            raise ValueError("Trying to access closed storage")
        return np.load(self.directory / (key + self.extension))

    def save(self, key, value):
        if not self._opened:
            raise ValueError("Trying to access closed storage")
        np.save(self.directory / (key + self.extension), value)


class _NpcArrayStorage(PickleStorage):
    """Subclass of :class:`Storage` which saves long-term data on disk with :func:`numpy.save`.

    This class can **only** accept :class:`~tenpy.linalg.np_conserve.Array` objects to be stored.
    It does so by keeping the "metadata" like charges in RAM and only stores the actual dense
    tensors.

    Parameters
    ----------
    directory : path-like
        An existing directory within which numpy files will be saved for each `key`.
    """

    extension = '.npy'

    def __init__(self, directory):
        super().__init__(directory)
        self._array_except_data = {}

    def load(self, key):
        if not self._opened:
            raise ValueError("Trying to access closed storage")
        value = self._array_except_data[key].copy(deep=False)
        N = value._data
        data = value._data = []
        with open(self.directory / (key + self.extension), 'rb') as f:
            value._qdata = np.load(f)
            for _ in range(N):
                data.append(np.load(f))
        return value

    def save(self, key, value):
        if not self._opened:
            raise ValueError("Trying to access closed storage")
        value = value.copy(deep=False)
        data = value._data
        N = value._data = len(data)  # replace _data attribute with just the length
        with open(self.directory / (key + self.extension), 'wb') as f:
            np.save(f, value._qdata)
            for T in data:
                np.save(f, T)
        value._qdata = None
        self._array_except_data[key] = value

    def delete(self, key):
        super().delete(key)
        del self._array_except_data[key]


class Hdf5Storage(Storage):
    """Subclass of :class:`Storage` which saves long-term data on disk with :mod:`h5py`.

    .. warning ::
        Some benchmarks that I ran when implementing this indicate that :class:`PickleStorage`
        has a much lower overhead than the :class:`Hdf5Storage`.
        Unless you have benchmarks indicated the opposite,
        I highly recommend sticking to :class:`PickleStorage`.

    Parameters
    ----------
    h5group : :class:`Group`
        The hdf5 group in which data will be saved using
        :func:`~tenpy.tools.hdf5_io.save_to_hdf5` under the specified keys.
    """
    trivial = False

    def __init__(self, h5group):
        super().__init__()
        self.h5gr = h5group
        self._delete_directory = None
        self._delete_file = None

    @classmethod
    def open(cls, filename=None, subgroup=None, mode="w-", delete=True, tmpdir=None):
        """Create an hdf5 file and use it to initialize an :class:`Hdf5Cache`.

        Parameters
        ----------
        filename : path-like | None
            Filename of the Hdf5 file to be created.
            If `None`, create a temporary file with :mod:`tempfile` tools.
        tmpdir : path-like | None
            Only used if `filename` is None. Used as base `dir` for :func:`tempfile.mkdtemp`,
            i.e., a temporary directory is created within this path, and inside that the hdf5 file.
        mode : str
            Filemode for opening the Hdf5 file.
        delete : bool
            Whether to automatically remove the corresponding file when closing the cache.
        """
        warnings.warn("Benchmarks suggest that PickleStorage is faster than Hdf5Storage")
        import h5py
        if filename is None:
            # h5py supports file-like objects, but this gives a python overhead for I/O.
            # hence h5py doc recommends using a temporary directory
            # and creating an hdf5 file inside that
            directory = tempfile.mkdtemp(prefix='tenpy_Hdf5Cache', dir=tmpdir)
            logger.info("create temporary cache directory %s", directory)
            filename = os.path.join(directory, "cache.h5")
        else:
            directory = None
            logger.info("create temporary cache file %s", filename)
        f = h5py.File(filename, mode=mode)
        if subgroup is not None:
            if subgroup in f:
                f = subgroup[f]
            else:
                f = f.create_group(subgroup)
        res = cls(f)
        res._owns_resources = True
        if delete:
            if directory is not None:
                # created temp directory -> need to clean that up!
                res._delete_directory = os.path.abspath(directory)
            else:
                res._delete_file = os.path.abspath(filename)
        return res

    def close(self):
        self._common_close()
        if not self._owns_resources:
            return
        f = self.h5gr.file
        if f:  # not yet closed by other function
            f.close()
        delete_file = self._delete_file
        if delete_file is not None:
            logger.info("cleanup/remove cache file %s", delete_file)
            os.remove(delete_file)
        delete_dir = self._delete_directory
        if delete_dir is not None:
            logger.info("cleanup/remove cache directory %s", delete_dir)
            shutil.rmtree(delete_dir)

    def subcontainer(self, name):
        if not self._opened:
            raise ValueError("Trying to access closed storage")
        if name in self.h5gr:
            raise ValueError("Subcontainer with that name already exists")
        res = Hdf5Storage(self.h5gr.create_group(name))
        return res

    def load(self, key):
        if not self._opened:
            raise ValueError("Trying to access closed storage")
        return load_from_hdf5(self.h5gr, key)

    def save(self, key, value):
        if not self._opened:
            raise ValueError("Trying to access closed storage")
        save_to_hdf5(self.h5gr, value, key)

    def delete(self, key):
        if not self._opened:
            raise ValueError("Trying to access closed storage")
        if key in self.h5gr:
            del self.h5gr[key]

    def __repr__(self):
        if self._opened:
            return f"<Hdf5Storage in {self.h5gr.file.filename!s}[{self.h5gr.name!r}]>"
        else:
            return "<Hdf5Storage, closed>"


class ThreadedStorage(Storage):
    """Wrapper around a :class:`Storage` (or subclass) with thread-parallelization.

    Parameters
    ----------
    worker : :class:`Group`
        The hdf5 group in which data will be saved using
        :func:`~tenpy.tools.hdf5_io.save_to_hdf5` under the specified keys.
    disk_storage : :class:`Storage`
        Instance of one of the other storage classes to wrap around.
    """
    def __init__(self, worker, disk_storage):
        if disk_storage.trivial:
            raise ValueError("ThreadedStorage with trivial `disk_storage` doesn't make sense")
        super().__init__()
        self.worker = worker
        self.disk_storage = disk_storage
        self.trivial = disk_storage.trivial
        self._loaded = {}
        self._waiting_for_load = set()

    @classmethod
    def open(cls, disk_storage, max_queue_size=2):
        """Setup and start a :class:`Worker` subthread.

        Parameters
        ----------
        disk_storage : :class:`Storage`
            Instance with methods for the actual disk I/O handling.
        """
        worker = Worker(max_queue_size=max_queue_size)
        worker = worker.__enter__()
        res = cls(worker, disk_storage)
        res._owns_resources = True
        return res

    def close(self):
        self._common_close()
        if not self._owns_resources:
            return
        self.worker.__exit__(None, None, None)
        self.disk_storage.close()
        self._loaded.clear()
        self._waiting_for_load.clear()

    def __exit__(self, exc_type, exc, tb):
        # same as self.close(), but pass on exception to worker.__exit__
        self._common_close()
        if not self._owns_resources:
            return
        self.worker.__exit__(exc_type, exc, tb)
        self.disk_storage.close()
        self._loaded.clear()
        self._waiting_for_load.clear()

    def subcontainer(self, name):
        # share the *same* worker subthread, but different save/load methods from subcontainer
        return ThreadedStorage(self.worker, self.disk_storage.subcontainer(name))

    def __bool__(self):
        return bool(self.disk_storage) and self.worker.worker_thread.is_alive()

    def load(self, key):
        if key not in self._loaded and key not in self._waiting_for_load:
            logger.debug("ThreadedStorage.load %s", key)
            self._waiting_for_load.add(key)
            self.worker.put_task(self.disk_storage.load,
                                 key,
                                 return_dict=self._loaded,
                                 return_key=key)
        else:
            logger.debug("ThreadedStorage.load %s (have pre-loaded)", key)
        assert key in self._waiting_for_load
        if key not in self._loaded:
            self.worker.join_tasks()  # wait for the tasks to finish loading
        assert key in self._loaded
        val = self._loaded[key]
        self._waiting_for_load.remove(key)
        del self._loaded[key]
        return val

    def preload(self, key):
        logger.debug("ThreadedStorage.preload %s", key)
        if key in self._waiting_for_load or key in self._loaded:
            return
        self._waiting_for_load.add(key)
        self.worker.put_task(self.disk_storage.load, key, return_dict=self._loaded, return_key=key)

    def save(self, key, value):
        logger.debug("ThreadedStorage.save %s", key)
        if key in self._waiting_for_load:
            # overwriting a preloaded but not yet returned value
            # need to wait for the preload to finish to avoid that the preload overrides results
            self.worker.join_tasks()
            assert key in self._loaded
            self._loaded[key] = value  # overwrite with new value
            # now we're in a valid preloaded state again with the new val already loaded!
        self.worker.put_task(self.disk_storage.save, key, value)

    def delete(self, key):
        self.worker.put_task(self.disk_storage.delete, key)
