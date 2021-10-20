"""Tools to save and load data (from TeNPy) to disk.

.. note ::
    This module is maintained in the repository https://github.com/tenpy/hdf5_io.git

See :doc:`/intro/input_output` for a motivation and specification of the HDF5 format implemented
below.
.. online at https://tenpy.readthedocs.io/en/latest/intro/input_output.html

The functions :func:`save` and :func:`load` are convenience functions for saving and loading
quite general python objects (like dictionaries) to/from files, guessing the file type
(and hence protocol for reading/writing) from the file ending.

On top of that, this function provides support for saving python objects to [HDF5]_ files with
the :class:`Hdf5Saver` and :class:`Hdf5Loader` classes
and the wrapper functions :func:`save_to_hdf5`, :func:`load_from_hdf5`.

.. note ::
    To use the export/import features to HDF5, you need to install the
    `h5py <http://docs.h5py.org>`_ python package
    (and hence some version of the HDF5 library).

.. warning ::
    Like loading a pickle file, loading data from a manipulated HDF5 file with the functions
    provided below has the potential to cause arbitrary code execution.
    Only load data from trusted sources!

.. rubric:: Global module constants used for our HDF5 format

Names of HDF5 attributes:

.. autodata:: ATTR_TYPE
.. autodata:: ATTR_CLASS
.. autodata:: ATTR_MODULE
.. autodata:: ATTR_LEN
.. autodata:: ATTR_FORMAT

Names for the ``ATTR_TYPE`` attribute:

.. autodata:: REPR_HDF5EXPORTABLE

.. autodata:: REPR_ARRAY
.. autodata:: REPR_INT
.. autodata:: REPR_FLOAT
.. autodata:: REPR_STR
.. autodata:: REPR_COMPLEX
.. autodata:: REPR_INT64
.. autodata:: REPR_FLOAT64
.. autodata:: REPR_INT32
.. autodata:: REPR_FLOAT32
.. autodata:: REPR_BOOL

.. autodata:: REPR_NONE
.. autodata:: REPR_RANGE
.. autodata:: REPR_LIST
.. autodata:: REPR_TUPLE
.. autodata:: REPR_SET
.. autodata:: REPR_DICT_GENERAL
.. autodata:: REPR_DICT_SIMPLE
.. autodata:: REPR_DTYPE
.. autodata:: REPR_IGNORED

.. autodata:: TYPES_FOR_HDF5_DATASETS

.. todo ::
    For memory caching with big MPO environments,
    we need a Hdf5Cacher clearing the memo's every now and then (triggered by what?).
"""
# Copyright 2020-2021 TeNPy Developers, GNU GPLv3

import pickle
import gzip
import types
import numpy as np
import importlib
import warnings
import sys
try:
    from packaging.version import parse as parse_version
except:
    try:
        from setuptools._vendor.packaging.version import parse as parse_version
    except ImportError:

        def parse_version(version_str):
            return version_str.split('.')  # bad but better than nothing


try:
    import h5py
    h5py_version = h5py.version.version_tuple
except ImportError:
    h5py_version = (0, 0)

__all__ = [
    'save', 'load', 'find_global', 'valid_hdf5_path_component', 'Hdf5FormatError',
    'Hdf5ExportError', 'Hdf5ImportError', 'Hdf5Exportable', 'Hdf5Ignored', 'Hdf5Saver',
    'Hdf5Loader', 'save_to_hdf5', 'load_from_hdf5', 'REPR_IGNORED', 'REPR_HDF5EXPORTABLE',
    'REPR_REDUCE', 'REPR_ARRAY', 'REPR_INT', 'REPR_FLOAT', 'REPR_STR', 'REPR_COMPLEX',
    'REPR_INT64', 'REPR_FLOAT64', 'REPR_COMPLEX128', 'REPR_INT32', 'REPR_FLOAT32',
    'REPR_COMPLEX64', 'REPR_BOOL', 'REPR_NONE', 'REPR_RANGE', 'REPR_LIST', 'REPR_TUPLE',
    'REPR_SET', 'REPR_DICT_GENERAL', 'REPR_DICT_SIMPLE', 'REPR_DTYPE', 'REPR_FUNCTION',
    'REPR_CLASS', 'REPR_GLOBAL', 'TYPES_FOR_HDF5_DATASETS', 'ATTR_TYPE', 'ATTR_CLASS',
    'ATTR_MODULE', 'ATTR_LEN', 'ATTR_FORMAT'
]


def save(data, filename, mode='w'):
    """Save `data` to file with given `filename`.

    This function guesses the type of the file from the filename ending.
    Supported endings:

    ============ ===============================
    ending       description
    ============ ===============================
    .pkl         Pickle without compression
    ------------ -------------------------------
    .pklz        Pickle with gzip compression.
    ------------ -------------------------------
    .hdf5, .h5   HDF5 file (using `h5py`).
    ============ ===============================

    Parameters
    ----------
    filename : str
        The name of the file where to save the data.
    mode : str
        File mode for opening the file. ``'w'`` for write (discard existing file),
        ``'a'`` for append (add data to exisiting file).
        See :py:func:`open` for more details.
    """
    filename = str(filename)
    if filename.endswith('.pkl'):
        with open(filename, mode + 'b') as f:
            pickle.dump(data, f)
    elif filename.endswith('.pklz'):
        with gzip.open(filename, mode + 'b') as f:
            pickle.dump(data, f)
    elif filename.endswith('.hdf5') or filename.endswith('.h5'):
        with h5py.File(filename, mode) as f:
            save_to_hdf5(f, data)
    else:
        raise ValueError("Don't recognise file ending of " + repr(filename))


def load(filename):
    """Load data from file with given `filename`.

    Guess the type of the file from the filename ending, see :func:`save` for possible endings.

    Parameters
    ----------
    filename : str
        The name of the file to load.

    Returns
    -------
    data : obj
        The object loaded from the file.
    """
    filename = str(filename)
    if filename.endswith('.pkl'):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    elif filename.endswith('.pklz'):
        with gzip.open(filename, 'rb') as f:
            data = pickle.load(f)
    elif filename.endswith('.hdf5') or filename.endswith('.h5'):
        with h5py.File(filename, 'r') as f:
            data = load_from_hdf5(f)
    else:
        raise ValueError("Don't recognise file ending of " + repr(filename))
    return data


def find_global(module, qualified_name):
    """Get the object of the `qualified_name` in a given python `module`.

    Parameters
    ----------
    module : str
        Name of the module containing the object. The module gets imported.
    qualified_name : str
        Name of the object to be retrieved. May contain dots if the object is part of a class etc.
    """
    mod = importlib.import_module(module)
    obj = mod
    for subpath in qualified_name.split('.'):
        obj = getattr(obj, subpath)
    return obj


# =================================================================================
# everything below is for our export/import with our self-definded HDF5 format.
# =================================================================================

REPR_IGNORED = "ignore"  #: ignore the object/dataset during loading and saving

#: saved object is instance of a user-defined class following the :class:`Hdf5Exportable` style.
REPR_HDF5EXPORTABLE = "instance"

REPR_REDUCE = "reduce"  #: saved object had a __reduce__ method according to pickle protocol

REPR_ARRAY = "array"  #: saved object represents a numpy array
REPR_INT = "int"  #: saved object represents a (python) int
REPR_FLOAT = "float"  #: saved object represents a (python) float
REPR_STR = "str"  #: saved object represents a (python unicode) string
REPR_COMPLEX = "complex"  #: saved object represents a complex number
REPR_INT64 = "np.int64"  #: saved object represents a np.int64
REPR_FLOAT64 = "np.float64"  #: saved object represents a np.float64
REPR_COMPLEX128 = "np.complex128"  #: saved object represents a np.complex128
REPR_INT32 = "np.int32"  #: saved object represents a np.int32
REPR_FLOAT32 = "np.float32"  #: saved object represents a np.float32
REPR_COMPLEX64 = "np.complex64"  #: saved object represents a np.complex64
REPR_BOOL = "bool"  #: saved object represents a boolean

REPR_NONE = "None"  #: saved object is ``None``
REPR_RANGE = "range"  #: saved object is a range
REPR_LIST = "list"  #: saved object represents a list
REPR_TUPLE = "tuple"  #: saved object represents a tuple
REPR_SET = "set"  #: saved object represents a set
REPR_DICT_GENERAL = "dict"  #: saved object represents a dict with complicated keys
REPR_DICT_SIMPLE = "simple_dict"  #: saved object represents a dict with simple keys
REPR_DTYPE = "dtype"  #: saved object represents a np.dtype

REPR_FUNCTION = "function"  #: saved object represents a (global) function
REPR_CLASS = "class"  #: saved object is a (global) class
REPR_GLOBAL = "global"  #: saved object is a global variable (like a class or function)

#: tuple of (type, type_repr) which h5py can save as datasets; one entry for each type.
TYPES_FOR_HDF5_DATASETS = tuple([
    (np.ndarray, REPR_ARRAY),
    (int, REPR_INT),
    (float, REPR_FLOAT),
    (str, REPR_STR),
    (complex, REPR_COMPLEX),
    (np.int64, REPR_INT64),
    (np.float64, REPR_FLOAT64),
    (np.complex128, REPR_COMPLEX128),
    (np.int32, REPR_INT32),
    (np.float32, REPR_FLOAT32),
    (np.complex64, REPR_COMPLEX64),
    (np.bool_, REPR_BOOL),
    (bool, REPR_BOOL),
])

ATTR_TYPE = "type"  #: Attribute name for type of the saved object, should be one of the ``REPR_*``
ATTR_CLASS = "class"  #: Attribute name for the class name of an HDF5Exportable
ATTR_MODULE = "module"  #: Attribute name for the module where ATTR_CLASS can be retrieved
ATTR_LEN = "len"  #: Attribute name for the length of iterables, e.g, list, tuple
ATTR_FORMAT = "format"  #: indicates the `ATTR_TYPE` format used by :class:`Hdf5Exportable`


def valid_hdf5_path_component(name):
    """Determine if `name` is a valid HDF5 path component.

    Conditions: String, no ``'/'``, and overall ``name != '.'``.
    """
    # unicode is encoded correctly by h5py and works - amazing!
    return isinstance(name, str) and '/' not in name and name != '.'


class Hdf5FormatError(Exception):
    """Common base class for errors regarding our HDF5 format."""
    pass


class Hdf5ExportError(Hdf5FormatError):
    """This exception is raised when something went wrong during export to hdf5."""
    pass


class Hdf5ImportError(Hdf5FormatError):
    """This exception is raised when something went wrong during import from hdf5."""
    pass


class Hdf5Exportable:
    """Interface specification for a class to be exportable to our HDF5 format.

    To allow a class to be exported to HDF5 with :func:`save_to_hdf5`,
    it only needs to implement the :meth:`save_hdf5` method as documented below.
    To allow import, a class should implement the classmethod :meth:`from_hdf5`.
    During the import, the class already needs to be defined;
    loading can only initialize instances, not define classes.

    The implementation given works for sufficiently simple (sub-)classes, for which all data is
    stored in :attr:`~object.__dict__`.
    In particular, this works for python-defined classes which simply store data using
    ``self.data = data`` in their methods.
    """
    def save_hdf5(self, hdf5_saver, h5gr, subpath):
        """Export `self` into a HDF5 file.

        This method saves all the data it needs to reconstruct `self` with :meth:`from_hdf5`.

        This implementation saves the content of :attr:`~object.__dict__` with
        :meth:`~tenpy.tools.hdf5_io.Hdf5Saver.save_dict_content`,
        storing the format under the attribute ``'format'``.

        Parameters
        ----------
        hdf5_saver : :class:`~tenpy.tools.hdf5_io.Hdf5Saver`
            Instance of the saving engine.
        h5gr : :class`Group`
            HDF5 group which is supposed to represent `self`.
        subpath : str
            The `name` of `h5gr` with a ``'/'`` in the end.
        """
        # for new implementations, use:
        #   hdf5_saver.save(data, subpath + "key")  # for big content/data
        #   h5gr.attrs["name"] = info               # for metadata

        # here: assume all the data is given in self.__dict__
        type_repr = hdf5_saver.save_dict_content(self.__dict__, h5gr, subpath)
        h5gr.attrs[ATTR_FORMAT] = type_repr

    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):
        """Load instance from a HDF5 file.

        This method reconstructs a class instance from the data saved with :meth:`save_hdf5`.

        Parameters
        ----------
        hdf5_loader : :class:`~tenpy.tools.io.Hdf5Loader`
            Instance of the loading engine.
        h5gr : :class:`Group`
            HDF5 group which is represent the object to be constructed.
        subpath : str
            The `name` of `h5gr` with a ``'/'`` in the end.

        Returns
        -------
        obj : cls
            Newly generated class instance containing the required data.
        """
        # for new implementations, use:
        #   obj = cls.__new__(cls)                     # create class instance, no __init__() call
        #   hdf5_loader.memorize_load(h5gr, obj)       # call preferably before loading other data
        #   info = hdf5_loader.get_attr(h5gr, "name")  # for metadata
        #   data = hdf5_loader.load(subpath + "key")   # for big content/data

        dict_format = hdf5_loader.get_attr(h5gr, ATTR_FORMAT)
        obj = cls.__new__(cls)  # create class instance, no __init__() call
        hdf5_loader.memorize_load(h5gr, obj)  # call preferably before loading other data
        data = hdf5_loader.load_dict(h5gr, dict_format, subpath)  # specialized loading
        # (the `load_dict` did not overwrite the memo_load entry)
        obj.__dict__.update(data)  # store data in the object
        return obj


class Hdf5Ignored:
    """Placeholder for a dataset/group to be ignored during both loading and saving.

    Objects of this type are not saved.
    Moreover, if a saved dataset/group has the `type` attribute matching `REPR_IGNORED`,
    instance of this class are returned instead of loading the data.

    Parameters
    ----------
    name : str
        The name of the dataset during loading; just for reference.

    Attributes
    ----------
    name : str
        See above.
    """
    def __init__(self, name='unknown'):
        self.name = name


class Hdf5Saver:
    """Class to save simple enough objects into a HDF5 file.

    The intended use of this class is through :func:`save_to_hdf5`, which is simply an alias
    for ``Hdf5Saver(h5group).save(obj, path)``.

    It exports python objects to a HDF5 file such that they can be loaded with the
    :class:`Hdf5Loader`, or a call to :func:`load_from_hdf5`, respectively.

    The basic structure of this class is similar as the `Pickler` from :mod:`pickle`.

    See :doc:`/intro/input_output` for a specification of what can be saved and what the resulting
    datastructure is.

    Parameters
    ----------
    h5group : :class:`Group`
        The HDF5 group (or HDF5 :class:`File`) where to save the data.
    format_selection : dict
        This dictionary allows to set a output format selection for user-defined
        :meth:`Hdf5Exportable.save_hdf5` implementations.
        For example, :class:`~tenpy.linalg.LegCharge` checks it for the key ``"LegCharge"``.

    Attributes
    ----------
    h5group : :class:`Group`
        The HDF5 group (or HDF5 :class:`File`) where to save the data.
    dispatch_save : dict
        Mapping from a type `keytype` to methods `f` of this class.
        The method is called as ``f(self, obj, path, type_repr)``.
        The call to `f` should save the object `obj` in ``self.h5group[path]``,
        call :meth:`memorize_save`, and set ``h5gr.attr[ATTR_TYPE] = type_repr``
        to a string `type_repr` in order to allow loading with the dispatcher
        in ``Hdf5Loader.dispatch_save[type_repr]``.
    memo_save : dict
        A dictionary to remember all the objects which we already stored to :attr:`h5group`.
        The dictionary key is the object id; the value is a two-tuple of the hdf5 group or dataset
        where an object was stored, and the object itself. See :meth:`memorize_save`.
    format_selection : dict
        This dictionary allows to set a output format selection for user-defined
        :meth:`Hdf5Exportable.save_hdf5` implementations.
        For example, :class:`~tenpy.linalg.LegCharge` checks it for the key ``"LegCharge"``.
    """
    def __init__(self, h5group, format_selection=None):
        self.h5group = h5group
        self.memo_save = {}
        if format_selection is None:
            format_selection = {}
        self.format_selection = format_selection

    def save(self, obj, path='/'):
        """Save `obj` in ``self.h5group[path]``.

        Parameters
        ----------
        obj : object
            The object (=data) to be saved.
        path : str
            Path within `h5group` under which the `obj` should be saved.
            To avoid unwanted overwriting of important data, the group/object should not yet exist,
            except if `path` is the default ``'/'``.

        Returns
        -------
        h5gr : :class:`Group` | :class:`Dataset`
            The h5py group or dataset in which `obj` was saved.
        """
        obj_id = id(obj)
        in_memo = self.memo_save.get(obj_id)  # default=None
        if in_memo is not None:  # saved the object before
            h5gr, _ = in_memo
            self.h5group[path] = h5gr  # create hdf5 hard link
            # hard linked objects share an hdf5 id,
            # which we use in the loader to distinguish them
            return h5gr

        disp = self.dispatch_save.get(type(obj))
        if disp is not None:
            f, type_repr = disp
            # `f` is a dispatcher function, which should
            # - save the `obj` in self.h5group['path'],
            # - call :meth:`memorize_save`, and
            # - set ``h5gr.attr[ATTR_TYPE] = type_repr`` to a string `type_repr`
            #   to allow loading with the dispatcher ``Hdf5Loader.dispatch_load[type_repr]``
            # call unbound method `f` with explicit self
            h5gr = f(self, obj, path, type_repr)
            return h5gr

        # handle classes with `save_hdf5` method
        obj_save_hdf5 = getattr(obj, 'save_hdf5', None)
        if obj_save_hdf5 is not None:  # of Hdf5Exportable type
            # `obj_save_hdf5` should be the bound method `obj.save_hdf5`,
            # so it does not need an explicit reference of `obj`
            h5gr, subpath = self.create_group_for_obj(path, obj)
            h5gr.attrs[ATTR_TYPE] = REPR_HDF5EXPORTABLE
            h5gr.attrs[ATTR_CLASS] = obj.__class__.__qualname__
            h5gr.attrs[ATTR_MODULE] = obj.__class__.__module__
            obj_save_hdf5(self, h5gr, subpath)  # should save the actual data
            return h5gr

        warnings.warn(
            "Hdf5Saver: object of type {t!r} without explicit HDF5 format; "
            "fall back to pickle protocol".format(t=type(obj)), UserWarning)

        obj_reduce = getattr(obj, "__reduce__", None)
        if obj_reduce is not None:

            rv = obj_reduce()
            if isinstance(rv, str):
                h5gr = self.save_global(obj, REPR_GLOBAL)
                return h5gr
            if not isinstance(rv, tuple) or not 2 <= len(rv) < 7:
                raise Hdf5ExportError("Wrong return value of {0!r}".format(obj_reduce))

            h5gr = self.save_reduce(*rv, obj=obj, path=path)
            return h5gr

        # unknown case
        msg = "Don't know how to save object of type {0!r}:\n{1!r}".format(type(obj), obj)
        raise Hdf5ExportError(msg)

    def create_group_for_obj(self, path, obj):
        """Create an HDF5 group ``self.h5group[path]`` to store `obj`.

        Also handle ending of path with ``'/'``, and memorize `obj` in :attr:`memo_save`.

        Parameters
        ----------
        path : str
            Path within `h5group` under which the `obj` should be saved.
            To avoid unwanted overwriting of important data, the group/object should not yet exist,
            except if `path` is the default ``'/'``.
        obj : object
            The object (=data) to be saved.

        Returns
        -------
        h5group : :class:`Group`
            Newly created h5py (sub)group ``self.h5group[path]``, unless `path` is ``'/'``,
            in which case it is simply the existing ``self.h5group['/']``.
        subpath : str
            The `group.name` ending with ``'/'``, such that other names can be appended to
            get the path for subgroups or datasets in the group.

        Raises
        ------
        ValueError : if `self.h5group[path]`` already existed and `path` is not ``'/'``.
        """
        if path == '/':
            gr = self.h5group[path]
        else:
            gr = self.h5group.create_group(path)  # raises ValueError if path already exists.
        subpath = path if path[-1] == '/' else (path + '/')
        self.memorize_save(gr, obj)
        return gr, subpath

    def memorize_save(self, h5gr, obj):
        """Store objects already saved in the :attr:`memo_save`.

        This allows to avoid copies, if the same python object appears multiple times in the
        data of `obj`. Examples can be shared :class:`~tenpy.linalg.charges.LegCharge` objects
        or even shared :class:`~tenpy.linalg.np_conserved.Array`.
        Using the memo also avoids crashes from cyclic references,
        e.g., when a list contains a reference to itself.

        Parameters
        ----------
        h5gr : :class:`Group` | :class:`Dataset`
            The h5py group or dataset in which `obj` was saved.
        obj : :class:`object`
            The object saved.
        """
        obj_id = id(obj)
        assert obj_id not in self.memo_save
        self.memo_save[obj_id] = (h5gr, obj)

    def save_reduce(self,
                    func,
                    args,
                    state=None,
                    listitems=None,
                    dictitems=None,
                    state_setter=None,
                    obj=None,
                    path=None):
        """Save the return values of ``obj.__reduce__`` following the pickle protocol."""
        h5gr, subpath = self.create_group_for_obj(path, obj)
        h5gr.attrs[ATTR_TYPE] = REPR_REDUCE
        self.save(func, subpath + 'func')
        self.save(args, subpath + 'args')
        if state is not None:
            self.save(state, subpath + 'state')
        if listitems is not None:
            self.save(state, subpath + 'listitems')
        if dictitems is not None:
            self.save(state, subpath + 'dictitems')
        if state_setter is not None:
            self.save(state, subpath + 'state_setter')
        return h5gr

    # save_reduce is called directly from `save()`, not dispatched.

    dispatch_save = {}

    # the methods below are used in the dispatch table

    def save_none(self, obj, path, type_repr):
        """Save the None object as a string (dataset); in dispatch table."""
        self.h5group[path] = REPR_NONE
        h5gr = self.h5group[path]
        h5gr.attrs[ATTR_TYPE] = REPR_NONE
        self.memorize_save(h5gr, obj)
        return h5gr

    dispatch_save[type(None)] = (save_none, REPR_NONE)

    def save_dataset(self, obj, path, type_repr):
        """Save `obj` as a hdf5 dataset; in dispatch table."""
        self.h5group[path] = obj  # save as dataset
        h5gr = self.h5group[path]
        h5gr.attrs[ATTR_TYPE] = type_repr
        self.memorize_save(h5gr, obj)
        return h5gr

    for _t, _type_repr in TYPES_FOR_HDF5_DATASETS:
        dispatch_save[_t] = (save_dataset, _type_repr)

    def save_iterable(self, obj, path, type_repr):
        """Save an iterable `obj` like a list, tuple or set; in dispatch table."""
        h5gr, subpath = self.create_group_for_obj(path, obj)
        h5gr.attrs[ATTR_TYPE] = type_repr
        self.save_iterable_content(obj, h5gr, subpath)
        return h5gr

    dispatch_save[list] = (save_iterable, REPR_LIST)
    dispatch_save[tuple] = (save_iterable, REPR_TUPLE)
    dispatch_save[set] = (save_iterable, REPR_SET)

    def save_iterable_content(self, obj, h5gr, subpath):
        """Save contents of an iterable `obj` in the existing `h5gr`.

        Parameters
        ----------
        obj : dict
            The data to be saved
        h5gr : :class:`Group`
            h5py Group under which the keys and values of `obj` should be saved.
        subpath : str
            Name of h5gr with ``'/'`` in the end.
        """
        h5gr.attrs[ATTR_LEN] = len(obj)
        for i, elem in enumerate(obj):
            self.save(elem, subpath + str(i))

    def save_dict(self, obj, path, type_repr):
        """Save the dictionary `obj`; in dispatch table."""
        h5gr, subpath = self.create_group_for_obj(path, obj)
        type_repr = self.save_dict_content(obj, h5gr, subpath)
        h5gr.attrs[ATTR_TYPE] = type_repr
        return h5gr

    dispatch_save[dict] = (save_dict, REPR_DICT_GENERAL)

    def save_dict_content(self, obj, h5gr, subpath):
        """Save contents of a dictionary `obj` in the existing `h5gr`.

        The format depends on whether the dictionary `obj` has simple keys valid for hdf5 path
        components (see :func:`valid_hdf5_path_component`) or not.
        For simple keys: directly use the keys as path.
        For non-simple keys: save list of keys und ``"keys"`` and list of values und ``"values"``.

        Parameters
        ----------
        obj : dict
            The data to be saved
        h5gr : :class:`Group`
            h5py Group under which the keys and values of `obj` should be saved.
        subpath : str
            Name of h5gr with ``'/'`` in the end.

        Returns
        -------
        type_repr : REPR_DICT_SIMPLE | REPR_DICT_GENERAL
            Indicates whether the data was saved in the format for a dictionary with simple keys
            or general keys, see comment above.
        """
        # check if we have only simple keys, which we can use in `path`
        simple_keys = True
        for k in obj.keys():
            if not valid_hdf5_path_component(k):
                simple_keys = False
                break

        if simple_keys:
            for k, v in obj.items():
                self.save(v, subpath + k)
            return REPR_DICT_SIMPLE
        else:
            keys = obj.keys()
            values = obj.values()
            self.save_iterable(keys, subpath + "keys", REPR_LIST)
            self.save_iterable(values, subpath + "values", REPR_LIST)
            return REPR_DICT_GENERAL

    def save_range(self, obj, path, type_repr):
        """Save a range object; in dispatch table."""
        h5gr, subpath = self.create_group_for_obj(path, obj)
        h5gr.attrs[ATTR_TYPE] = REPR_RANGE
        self.save(obj.start, subpath + 'start')
        self.save(obj.stop, subpath + 'stop')
        self.save(obj.step, subpath + 'step')
        return h5gr

    dispatch_save[range] = (save_range, REPR_RANGE)

    def save_dtype(self, obj, path, type_repr):
        """Save a :class:`~numpy.dtype` object; in dispatch table."""
        h5gr, subpath = self.create_group_for_obj(path, obj)
        h5gr.attrs[ATTR_TYPE] = REPR_DTYPE
        name = getattr(obj, "name", "void")
        h5gr.attrs["name"] = name
        self.save(obj.descr, subpath + 'descr')
        return h5gr

    if parse_version(np.__version__) < parse_version('1.20.0'):
        dispatch_save[np.dtype] = (save_dtype, REPR_DTYPE)
    else:
        # numpy version 1.20 introduced separate subclasses of dtype for the standard types
        for t in np.dtype.__subclasses__():
            dispatch_save[t] = (save_dtype, REPR_DTYPE)

    def save_ignored(self, obj, path, type_repr):
        """Don't save the Hdf5Ignored object; just return None."""
        return None

    dispatch_save[Hdf5Ignored] = (save_ignored, REPR_IGNORED)

    def save_global(self, obj, path, type_repr):
        """Save a global object like a function or class."""
        module = obj.__module__
        qualname = obj.__qualname__
        try:
            obj2 = find_global(module, qualname)
        except (ImportError, KeyError, AttributeError):
            raise Hdf5ExportError(
                "Can't export `{0!r}`: it's not found as {1} in module {2}".format(
                    obj, module, classname)) from None
        else:
            if obj2 is not obj:
                raise Hdf5ExportError("Can't export `{0!r}`: it's not the same object"
                                      "as {1} in module {2}".format(obj, module, classname))
        full_name = qualname + " in " + module
        self.h5group[path] = full_name  # save as string dataset
        h5gr = self.h5group[path]
        h5gr.attrs[ATTR_TYPE] = type_repr
        h5gr.attrs[ATTR_CLASS] = qualname
        h5gr.attrs[ATTR_MODULE] = module
        self.memorize_save(h5gr, obj)
        return h5gr

    dispatch_save[types.FunctionType] = (save_global, REPR_FUNCTION)
    dispatch_save[types.BuiltinFunctionType] = (save_global, REPR_FUNCTION)
    dispatch_save[type] = (save_global, REPR_CLASS)

    # clean up temporary variables
    del _t
    del _type_repr


class Hdf5Loader:
    """Class to load and import object from a HDF5 file.

    The intended use of this class is through :func:`load_from_hdf5`, which is simply an alias
    for ``Hdf5Loader(h5group).load(path)``.

    It can load data exported with :func:`save_to_hdf5` or the :class:`Hdf5Saver`, respectively.

    The basic structure of this class is similar as the `Unpickler` from :mod:`pickle`.

    See :doc:`/intro/input_output` for a specification of what can be saved and what the resulting
    datastructure is.

    Parameters
    ----------
    h5group : :class:`Group`
        The HDF5 group (or file) where to save the data.
    ignore_unknown : bool
        Whether to just warn (True) or raise an Error (False) if a class to be loaded is not found.
    exclude : list of str
        List of paths (possibly relative to `h5group`) for objects to be excluded from loading.
        References to the corresponding object are replaced by an instance of :class:`Hdf5Ignored`.
        Of course, **this might break other functions** expecting correctly loaded data.

    Attributes
    ----------
    h5group : :class:`Group`
        The HDF5 group (or HDF5 :class:`File`) where to save the data.
    ignore_unknown : bool
        Whether to just warn (True) or raise an Error (False) if a class to be loaded is not found.
    dispatch_load : dict
        Mapping from one of the global ``REPR_*`` variables to (unbound) methods `f` of this class.
        The method is called as ``f(self, h5gr, type_info, subpath)``.
        The call to `f` should load and return an object `obj` from the h5py :class:`Group`
        or :class:`Dataset` `h5gr`; and memorize the loaded `obj` with :meth:`memorize_load`.
        `subpath` is just the name of `h5gr` with a guaranteed ``'/'`` in the end.
        `type_info` is often the ``REPR_*`` variable of the type or some other information about
        the type, which allows to use a single dispatch_load function for different datatypes.
    memo_load : dict
        A dictionary to remember all the objects which we already loaded from :attr:`h5group`.
        The dictionary key is a h5py group- or dataset ``id``;
        the value is the loaded object. See :meth:`memorize_load`.
    """
    def __init__(self, h5group, ignore_unknown=True, exclude=None):
        self.h5group = h5group
        self.ignore_unknown = ignore_unknown
        self.memo_load = {}
        if exclude:
            for path in exclude:
                try:
                    data = self.h5group[path]
                except KeyError:
                    warnings.warn(
                        "can't exclude {0!r} from loading: not existent in h5group".format(path))
                    continue
                self.memorize_load(data, Hdf5Ignored(path))

    def load(self, path=None):
        """Load a Python :class:`object` from the dataset.

        See :func:`load_from_hdf5` for more details.

        Parameters
        ----------
        path : None | str | :class:`Reference`
            Path within :attr:`h5group` to be used for loading.
            Defaults to the name of :attr:`h5group` itself.

        Returns
        -------
        obj : object
            The Python object loaded from `h5group` (specified by `path`).
        """
        # get dataset to be loaded
        if path is None:
            h5gr = self.h5group
            path = self.h5group.name
        else:
            h5gr = self.h5group[path]
        subpath = path if path[-1] == '/' else (path + '/')
        # check memo_load
        in_memo = self.memo_load.get(h5gr.id)  # default=None
        if in_memo is not None:  # loaded the object before
            return in_memo

        # determine type of object to be loaded.
        type_repr = self.get_attr(h5gr, ATTR_TYPE)
        disp = self.dispatch_load.get(type_repr)
        if disp is None:
            msg = "Unknown type {0!r} while loading hdf5 dataset {1!s}"
            raise Hdf5ImportError(msg.format(type_repr, h5gr.name))
        f, type_info = disp
        # `f` is a dispatcher function, which should do the following
        # (preferably in this order, if `obj` is mutable):
        # - generate an object `obj` of the described type
        # - call :meth:`memorize_load` for the generated `obj`,
        # - fill the object with the data from subgroups/subdatasets (everything under `subpath`)
        # - return the generated `obj`
        # call unbound method `f` with explicit self
        obj = f(self, h5gr, type_info, subpath)
        return obj

    def memorize_load(self, h5gr, obj):
        """Store objects already loaded in the :attr:`memo_load`.

        This allows to avoid copies, if the same dataset appears multiple times in the
        hdf5 group of `obj`.
        Examples can be shared :class:`~tenpy.linalg.charges.LegCharge` objects
        or even shared :class:`~tenpy.linalg.np_conserved.Array`.

        To handle cyclic references correctly, this function should be called *before* loading
        data from subgroups with new calls of :meth:`load`.
        """
        self.memo_load.setdefault(h5gr.id, obj)  # don't overwrite existing entries!

    @staticmethod
    def get_attr(h5gr, attr_name):
        """Return attribute ``h5gr.attrs[attr_name]``, if existent.

        Raises
        ------
        :class:`Hdf5ImportError`
            If the attribute does not exist.
        """
        res = h5gr.attrs.get(attr_name)
        if res is None:
            msg = "missing attribute {0!r} for dataset {1!s}"
            raise Hdf5ImportError(msg.format(attr_name, h5gr.name))
        if isinstance(res, bytes):
            res = res.decode()
        return res

    dispatch_load = {}

    # the methods below are used in the dispatch table

    def load_none(self, h5gr, type_info, subpath):
        """Load the ``None`` object from a dataset."""
        obj = None
        self.memorize_load(h5gr, obj)
        return obj

    dispatch_load[REPR_NONE] = (load_none, None)

    def load_dataset(self, h5gr, type_info, subpath):
        """Load a h5py :class:`Dataset` and convert it into the desired type."""
        if type_info is np.ndarray:
            obj = h5gr[...]
        else:
            obj = h5gr[()]  # load scalar from hdf5 Dataset
            # convert to desired type: type_info is simply the type
            obj = type_info(obj)
        self.memorize_load(h5gr, obj)
        return obj

    for _t, _type_repr in TYPES_FOR_HDF5_DATASETS:
        dispatch_load[_type_repr] = (load_dataset, _t)

    def load_str(self, h5gr, type_info, subpath):
        """Load a string from a h5py :class:`Dataset`."""
        # `asstr()` is a new method for handling strings introduced in h5py version 3.0
        # if asstr() is not used, the returned data is a raw bindary/ascii string.
        obj = h5gr.asstr()[()]
        self.memorize_load(h5gr, obj)
        return obj

    if h5py_version >= (3, 0):  # for older h5py versions, just read the dataset directly.
        dispatch_load[REPR_STR] = (load_str, str)

    def load_list(self, h5gr, type_info, subpath):
        """Load a list."""
        obj = []
        self.memorize_load(h5gr, obj)
        length = self.get_attr(h5gr, ATTR_LEN)
        for i in range(length):
            sub_obj = self.load(subpath + str(i))
            obj.append(sub_obj)
        return obj

    dispatch_load[REPR_LIST] = (load_list, REPR_LIST)

    def load_set(self, h5gr, type_info, subpath):
        """Load a set."""
        obj = set([])
        self.memorize_load(h5gr, obj)
        length = self.get_attr(h5gr, ATTR_LEN)
        for i in range(length):
            sub_obj = self.load(subpath + str(i))
            obj.add(sub_obj)
        return obj

    dispatch_load[REPR_SET] = (load_set, REPR_SET)

    def load_tuple(self, h5gr, type_info, subpath):
        """Load a tuple."""
        obj = []  # tuple is immutable: can't append to it
        # so we need to use a list during loading
        self.memorize_load(h5gr, obj)
        # BUG: for recursive tuples, the memorized object is a list instead of a tuple.
        # but I don't know how to circumvent this.
        # It's hopefully not relevant for our applications.
        length = self.get_attr(h5gr, ATTR_LEN)
        for i in range(length):
            sub_obj = self.load(subpath + str(i))
            obj.append(sub_obj)
        # now conjvert the list to tuple
        obj = tuple(obj)
        self.memo_load[h5gr.id] = obj  # overwrite the memo entry to point to the tuple,
        # not the list
        return obj

    dispatch_load[REPR_TUPLE] = (load_tuple, REPR_TUPLE)

    def load_dict(self, h5gr, type_info, subpath):
        """Load a dictionary in the format according to `type_info`."""
        if type_info == REPR_DICT_GENERAL:
            return self.load_general_dict(h5gr, type_info, subpath)
        elif type_info == REPR_DICT_SIMPLE:
            return self.load_simple_dict(h5gr, type_info, subpath)
        raise ValueError("can't interpret type_info {0!r}".format(type_info))

    def load_general_dict(self, h5gr, type_info, subpath):
        """Load a dictionary with general keys."""
        obj = {}
        self.memorize_load(h5gr, obj)
        keys = self.load_list(h5gr['keys'], REPR_LIST, subpath + 'keys/')
        values = self.load_list(h5gr['values'], REPR_LIST, subpath + 'values/')
        obj.update(zip(keys, values))
        return obj

    dispatch_load[REPR_DICT_GENERAL] = (load_general_dict, REPR_DICT_GENERAL)

    def load_simple_dict(self, h5gr, type_info, subpath):
        """Load a dictionary with simple keys."""
        obj = {}
        self.memorize_load(h5gr, obj)
        for k in h5gr.keys():
            v = self.load(subpath + k)
            obj[k] = v
        return obj

    dispatch_load[REPR_DICT_SIMPLE] = (load_simple_dict, REPR_DICT_SIMPLE)

    def load_range(self, h5gr, type_info, subpath):
        """Load a range."""
        start = self.load(subpath + 'start')
        stop = self.load(subpath + 'stop')
        step = self.load(subpath + 'step')
        obj = range(start, stop, step)
        self.memorize_load(h5gr, obj)  # late, but okay: no cyclic reference expected
        return obj

    dispatch_load[REPR_RANGE] = (load_range, REPR_RANGE)

    def load_dtype(self, h5gr, type_info, subpath):
        """Load a :class:`numpy.dtype`."""
        name = self.get_attr(h5gr, "name")
        if name.startswith("void"):
            descr = self.load(subpath + 'descr')
            obj = np.dtype(descr)
        else:
            obj = np.dtype(name)
        self.memorize_load(h5gr, obj)
        return obj

    dispatch_load[REPR_DTYPE] = (load_dtype, REPR_DTYPE)

    def load_hdf5exportable(self, h5gr, type_info, subpath):
        """Load an instance of a userdefined class."""
        module_name = self.get_attr(h5gr, ATTR_MODULE)
        class_name = self.get_attr(h5gr, ATTR_CLASS)
        try:
            cls = find_global(module_name, class_name)
        except (ImportError, AttributeError):
            msg = "Can't import class {0!s} from {1!s}".format(class_name, module_name)
            if self.ignore_unknown:
                warnings.warn(msg, UserWarning)
                return Hdf5Ignored(msg)
            else:
                raise
        return cls.from_hdf5(self, h5gr, subpath)

    dispatch_load[REPR_HDF5EXPORTABLE] = (load_hdf5exportable, REPR_HDF5EXPORTABLE)

    def load_ignored(self, h5gr, type_info, subpath):
        """Ignore the group to be loaded."""
        return Hdf5Ignored(h5gr.name)

    dispatch_load[REPR_IGNORED] = (load_ignored, REPR_IGNORED)

    def load_global(self, h5gr, type_info, subpath):
        """Load a global object like a class or function from its qualified name and module."""
        module_name = self.get_attr(h5gr, ATTR_MODULE)
        class_name = self.get_attr(h5gr, ATTR_CLASS)
        try:
            obj = find_global(module_name, class_name)
        except (ImportError, AttributeError):
            msg = "Can't import global {0!s} from {1!s}".format(class_name, module_name)
            if self.ignore_unknown:
                warnings.warn(msg, UserWarning)
                return Hdf5Ignored(msg)
            else:
                raise
        self.memorize_load(h5gr, obj)
        return obj

    dispatch_load[REPR_FUNCTION] = (load_global, REPR_FUNCTION)
    dispatch_load[REPR_CLASS] = (load_global, REPR_CLASS)
    dispatch_load[REPR_GLOBAL] = (load_global, REPR_GLOBAL)

    def load_reduce(self, h5gr, type_info, subpath):
        """Load an object where the return values of  ``obj.__reduce__`` has been exported."""
        func = self.load(subpath + 'func')
        args = self.load(subpath + 'args')
        obj = func(*args)
        self.memorize_load(h5gr, obj)
        if 'state' in h5gr:
            state = self.load(subpath + 'state')
            if 'state_setter' in h5gr:
                state_setter = self.load(subpath + 'state_setter')
                obj = state_setter(obj, state)
                self.memorize_load(h5gr, obj)  # overwrites old memo entry
            else:
                # see pickle._Unpickler.load_build
                setstate = getattr(obj, '__setstate__', None)
                if setstate is not None:
                    setstate(state)
                else:
                    slotstate = None
                    if isinstance(state, tuple) and len(state) == 2:
                        state, slotstate = state
                    if state:
                        obj_dict = obj.__dict__
                        for k, v in state.items():
                            if type(k) is str:
                                obj_dict[sys.intern(k)] = v
                            else:
                                obj_dict[k] = v
                    if slotstate:
                        for k, v in slotstate.items():
                            setattr(obj, k, v)
        if 'listitems' in h5gr:
            listitems = self.load(subpath + 'listitems')
            for item in listitems:
                obj.append(item)
        if 'dictitems' in h5gr:
            dictitems = self.load(subpath + 'dictitems')
            for key, val in dictitems:
                obj[key] = val
        return obj

    dispatch_load[REPR_REDUCE] = (load_reduce, REPR_REDUCE)

    # clean up temporary variables
    del _t
    del _type_repr


def save_to_hdf5(h5group, obj, path='/'):
    """Save an object `obj` into a hdf5 file or group.

    Roughly equivalent to ``h5group[path] = obj``, but handle different types of `obj`.
    For example, dictionaries are handled recursively.
    See :doc:`/intro/input_output` for a specification of what can be saved and what the resulting
    datastructure is.

    Parameters
    ----------
    h5group : :class:`Group`
        The HDF5 group (or h5py :class:`File`) to which `obj` should be saved.
    obj : object
        The object (=data) to be saved.
    path : str
        Path within `h5group` under which the `obj` should be saved.
        To avoid unwanted overwriting of important data, the group/object should not yet exist,
        except if `path` is the default ``'/'``.

    Returns
    -------
    h5obj : :class:`Group` | :class:`Dataset`
        The h5py group or dataset under which `obj` was saved.
    """
    return Hdf5Saver(h5group).save(obj, path)


def load_from_hdf5(h5group, path=None, ignore_unknown=True, exclude=None):
    """Load an object from hdf5 file or group.

    Roughly equivalent to ``obj = h5group[path][...]``, but handle more complicated objects saved
    as hdf5 groups and/or datasets with :func:`save_to_hdf5`.
    For example, dictionaries are handled recursively.
    See :doc:`/intro/input_output` for a specification of what can be saved/loaded and what the
    corresponding datastructure is.

    Parameters
    ----------
    h5group : :class:`Group`
        The HDF5 group (or h5py :class:`File`) to be loaded.
    path : None | str | :class:`Reference`
        Path within `h5group` to be used for loading. Defaults to the `h5group` itself specified.
    ignore_unknown : bool
        Whether to just warn (True) or raise an Error (False) if a class to be loaded is not found.
    exclude : list of str
        List of paths (possibly relative to `h5group`) for objects to be excluded from loading.
        References to the corresponding object are replaced by an instance of :class:`Hdf5Ignored`.
        For example, you could load a saved dictionary
        ``{'big_data': [...], 'small_data': small_data}`` with ``exclude=['/big_data']`` to get
        ``{'big_data': Hdf5Ignored('/big_data'), 'small_data': small_data}``.
        Of course, **this might break other functions** expecting correctly loaded data.

    Returns
    -------
    obj : object
        The Python object loaded from `h5group` (specified by `path`).
    """
    return Hdf5Loader(h5group, ignore_unknown, exclude).load(path)
