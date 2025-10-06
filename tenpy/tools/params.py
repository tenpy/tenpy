"""Tools to handle config options/parameters for algorithms.

See the doc-string of :class:`Config` for details.
"""
# Copyright (C) TeNPy Developers, Apache license

import warnings
import numpy
import numpy as np
import numbers
from collections.abc import MutableMapping
import pprint
import os
import logging
logger = logging.getLogger(__name__)

from .hdf5_io import ATTR_FORMAT

__all__ = ["Config", "asConfig", "load_yaml_with_py_eval"]


class Config(MutableMapping):
    """Dict-like wrapper class for parameter/configuration dictionaries.

    This class behaves mostly like a dictionary of option keys/values (together making the whole
    "config") with some additional features:

    - Logging of the options the first time they get used.
    - :meth:`get` acts more like :meth:`dict.setdefault` such that after the algorithm, all the
      used default values are known and can be saved for future reference.
    - Keeping track of unused options to detect typos in the keys.
    - Nicer formatting with ``print(config)``
    - Import/export to yaml and hdf5 files.

    .. cfg:config :: Config

    Parameters
    ----------
    config : dict
        Dictionary containing the actual option keys and values.
    name : str
        Descriptive name of the config used for logging.

    Attributes
    ----------
    name : str
        Name of the dictionary, for output statements. For example, when using
        a `Config` class for DMRG, ``name='DMRG'``.
    options : dict
        Dictionary containing the actual option keys and values.
    unused : set
        Keeps track of any :attr:`options` not yet used.
    """
    def __init__(self, config, name):
        self.options = config
        self.unused = set(config.keys())
        self.name = name

    def copy(self, share_unused=True):
        """Make a *shallow* copy, as for a dictionary.

        Parameters
        ----------
        share_unused : bool
            Whether the :attr:`unused` set should be shared.
        """
        res = Config(self.options.copy(), self.name)
        if share_unused:
            res.unused = self.unused
        return res

    def as_dict(self):
        """Return a copy of the options as a dictionary.

        Subconfigs are recursively converted to dict.
        """
        res = dict(self.options)
        for k, v in res.items():
            if isinstance(v, Config):
                res[k] = v.as_dict()
        return res

    def save_yaml(self, filename):
        """Save the parameters to `filename` as a YAML file.

        Parameters
        ----------
        filename : str
            Name of the resulting YAML file.
        """
        import yaml
        with open(filename, 'w') as stream:
            yaml.dump(self.as_dict(), stream)

    @classmethod
    def from_yaml(cls, filename, name=None):
        """Load a `Config` instance from a YAML file containing the :attr:`options`.

        The yaml file can have additional ``!py_eval`` tags, see :func:`load_yaml_with_py_eval`.

        .. warning ::
            Like pickle, it is not safe to load a yaml file from an untrusted source! A malicious
            file can call any Python function and should thus be treated with extreme caution.

        Parameters
        ----------
        filename : str
            Name of the YAML file
        name : str | None
            Name of the resulting :class:`Config` instance.
            If ``None``, default to (the basename of) `filename`.

        Returns
        -------
        obj : Config
            A `Config` object, loaded from file.
        """
        if name is None:
            name = os.path.basename(filename)
        config = load_yaml_with_py_eval(filename)
        return cls(config, name)

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
        type_repr = hdf5_saver.save_dict_content(self.options, h5gr, subpath)
        h5gr.attrs[ATTR_FORMAT] = type_repr
        h5gr.attrs["name"] = self.name
        h5gr.attrs["unused"] = [str(u) for u in self.unused]

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
        dict_format = hdf5_loader.get_attr(h5gr, ATTR_FORMAT)
        obj = cls.__new__(cls)  # create class instance, no __init__() call
        hdf5_loader.memorize_load(h5gr, obj)
        obj.options = hdf5_loader.load_dict(h5gr, dict_format, subpath)
        obj.name = hdf5_loader.get_attr(h5gr, "name")
        obj.unused = set(hdf5_loader.get_attr(h5gr, "unused"))
        return obj

    def __getitem__(self, key):
        val = self.options[key]
        self.log(key, "reading")
        self.unused.discard(key)
        return val

    def __setitem__(self, key, value):
        if key not in self.options.keys():
            self.unused.add(key)
        self.options[key] = value
        self.log(key, "setting")

    def __delitem__(self, key):
        self.log(key, "deleting")
        self.unused.discard(key)
        del self.options[key]

    def __iter__(self):
        return iter(self.options)

    def __len__(self):
        return len(self.options)

    def __str__(self):
        res = "Config, name={0!r}, options:\n".format(self.name)
        res += pprint.pformat(self.options)
        return res

    def __repr__(self):
        return "Config(<{0:d} options>, {1!r})".format(len(self.options), self.name)

    def __del__(self):
        self.warn_unused()

    def __ior__(self, other):
        self.update(other)
        return self

    def warn_unused(self, recursive=False):
        """Warn about (so far) unused options.

        This can help to detect typos in the option keys.
        It is automatically called upon deletion of `self`,
        but this might be a bit later than you intended.

        Parameters
        ----------
        recursive : bool
            If True, check the values of `self` for other :class:`Config` and warn in them as well.
        """
        unused = getattr(self, 'unused', None)
        if unused is None:
            return
        if len(unused) > 0:
            if len(unused) > 1:
                msg = "unused options for config {name!s}:\n{keys!s}"
            else:
                msg = "unused option {keys!s} for config {name!s}"
            warnings.warn(msg.format(keys=sorted(unused), name=self.name))
            self.unused.clear()  # don't warn twice about the same parameters
        if recursive:
            for val in self.options.values():
                if isinstance(val, Config):
                    val.warn_unused(True)

    def keys(self):
        return self.options.keys()

    def get(self, key, default, expect_type=None):
        """Find the value of `key`; really more like `setdefault` of a :class:`dict`.

        If no value is set, return `default` and set the value of `key` to `default` internally.

        Parameters
        ----------
        option : str
            Key for the option being read out.
        default :
            Default value for the parameter.
        expect_type : str | (sequence of) type
            If given, we check if the returned value is an instance of *any* of the given types.
            If it is not, we issue a ``UserWarning``.
            As an exception, the value ``None`` is always allowed and never triggers a warning.
            The following string short-hands are accepted as well::

                'real': ``numbers.Real``
                'complex': ``numbers.Complex``
                'array': ``[list, numpy.ndarray]``
                'real_or_array`: ``[numbers.Real, list, numpy.ndarray]``
                'complex_or_array`: ``[numbers.Complex, list, numpy.ndarray]``
        
        Returns
        -------
        val :
            The value for `option` if it existed, `default` otherwise.
        """
        use_default = key not in self.options.keys()
        val = self.options.setdefault(key, default)  # get & set default if not existent
        self.log(key, "reading", use_default)
        self.unused.discard(key)  # (does nothing if key not in set)
        if (expect_type is not None) and (val is not None):  # (val is None) => nothing to check
            # convert to sequence
            if expect_type == 'real':
                expect_type = [numbers.Real]
            if expect_type == 'complex':
                expect_type = [numbers.Complex]
            if expect_type == 'array':
                expect_type = [list, np.ndarray]
            if expect_type == 'real_or_array':
                expect_type = [numbers.Real, list, np.ndarray]
            if expect_type == 'complex_or_array':
                expect_type = [numbers.Complex, list, np.ndarray]
            try:
                iter(expect_type)
            except TypeError:
                expect_type = [expect_type]
            else:
                expect_type = list(expect_type)
            assert len(expect_type) > 0, 'Expected at least one type'
            type_ok = False
            for t in expect_type:
                if not isinstance(t, type):
                    raise ValueError(f'Not a type: {t}')
                if isinstance(val, t):
                    type_ok = True
                    break
            if not type_ok:
                if len(expect_type) == 1:
                    expected = f'Expected {expect_type[0]}'
                else:
                    expected = f'Expected one of {", ".join(t.__name__ for t in expect_type)}'
                msg = f'Invalid type for key "{key}". {expected}. Got {type(val).__name__}.'
                warnings.warn(msg, stacklevel=2)
        return val

    def silent_get(self, key, default):
        """Find the value of `key`, but don't set as default value and don't print.

        Same as ``dict.get``, i.e. just return `self[key]` if existent, else `default`, without
        memorizing/logging the access.
        Does not count as read-out for the :attr:`unused` parameters.
        """
        return self.options.get(key, default)

    def setdefault(self, key, default):
        """Set a default value without reading it out.

        Parameters
        ----------
        key : str
            Key name for the option being set.
        default :
            The value to be set by default if the option is not yet set.
        """
        use_default = key not in self.keys()
        self.options.setdefault(key, default)
        self.log(key, "set default", not use_default)
        self.unused.discard(key)  # (does nothing if key not in set)
        # do no return the value: not added to self.unused!

    def subconfig(self, key, default=None):
        """Get ``self[key]`` as a :class:`Config`."""
        use_default = key not in self.keys()
        if use_default:
            if default is None:
                subconfig = {}
            else:
                subconfig = default.copy()
        else:
            subconfig = self.options[key]
        subconfig = asConfig(subconfig, key)
        self.options[key] = subconfig
        self.log(key, "subconfig", use_default)
        self.unused.discard(key)  # (does nothing if key not in set)
        return subconfig

    def touch(self, *keys):
        """Mark `keys` as read out to suppress warnings about those keys being unused.

        Parameters
        ----------
        *keys : str
            Each key is marked as read out.
        """
        for key in keys:
            self.unused.discard(key)  # (does nothing if key not in set)

    def log(self, option, action="Option", use_default=False):
        """Print out `option` if verbosity and other conditions are met.

        Parameters
        ----------
        option : str
            Key/option name for the parameter being read out.
        action : str, optional
            Use to adapt log message to specific actions (e.g. "Deleting")
        """
        name = self.name
        new_key = option in self.unused or use_default
        val = self.options.get(option, "<not set>")
        if new_key:
            if use_default:
                logger.debug("%s: %s %r=%r (default)", name, action, option, val)
            else:
                logger.info("%s: %s %r=%r", name, action, option, val)

    def deprecated_alias(self, old_key, new_key, extra_msg=""):
        if old_key in self.options.keys():
            msg = "Deprecated option in {name!r}: {old!r} renamed to {new!r}"
            msg = msg.format(name=self.name, old=old_key, new=new_key)
            if extra_msg:
                msg = '\n'.join(msg, extra_msg)
            warnings.warn(msg, FutureWarning, stacklevel=3)
            self.options[new_key] = self.options[old_key]
            self.unused.discard(old_key)
            self.unused.add(new_key)

    def any_nonzero(self, keys, log_msg=None):
        """Check for any non-zero or non-equal entries in some parameters.

        Parameters
        ----------
        keys : list of {key | tuple of keys}
            For a single key, check ``self[key]`` for non-zero entries.
            For a tuple of keys, all the ``self[key]`` have to be equal (as numpy arrays).
            It is assumed that the default values for the keys are 0!
        log_msg : None | str
            If not None, `logger.debug` this message with the reason if `True` is returned.

        Returns
        -------
        match : bool
            False, if all ``self[key]`` are zero or `None` and
            True, if any of the ``self[key]`` for single `key` in `keys`,
            or if any of the entries for a tuple of `keys`
        """
        for k in keys:
            if isinstance(k, tuple):
                if len(k) == 0:
                    raise ValueError("got empty tuple, nothing to compare")
                # check equality
                nonzero = [self.has_nonzero(k0) for k0 in k]
                if not any(nonzero):
                    continue  # all zero, so equal
                if not all(nonzero):
                    if log_msg is not None:
                        logger.debug("%s: %r would need to be equal", log_msg, k)
                    return True
                val = self.options[k[0]]
                for k1 in k[1:]:
                    other_val = self.options[k1]
                    if not np.array_equal(val, other_val):
                        if log_msg is not None:
                            logger.debug("%s: %r and %r have different entries", log_msg, k, k1)
                        return True
            else:
                if self.has_nonzero(k):
                    if log_msg is not None:
                        logger.debug("%s: %r as nonzero entries", log_msg, k)
                    return True
        return False

    def has_nonzero(self, key):
        """Check whether `self` contains `key`, and if `self[key]` is nontrivial.

        Parameters
        ----------
        key : str
            Key for the parameter to check

        Returns
        -------
        bool
            True if `self` has key `key` with a nontrivial value. False otherwise.
        """
        return (key in self.keys() and self.options[key] is not None
                and np.any(np.array(self.options[key])) != 0)


def asConfig(config, name):
    """Convert a dict-like `config` to a :class:`Config`.

    Parameters
    ----------
    config : dict | :class:`Config`
        If this is a :class:`Config`, just return it.
        Otherwise, create a :class:`Config` from it and return that.
    name : str
        Name to be used for the :class:`Config`.

    Returns
    -------
    config : :class:`Config`
        Either directly `config` or ``Config(config, name)``.
    """
    if isinstance(config, Config):
        return config
    return Config(config, name)



def _yaml_eval_constructor(loader, node):
    """Yaml constructor to support `!py_eval` tag in yaml files."""
    cmd = loader.construct_scalar(node)
    if not isinstance(cmd, str):
        raise ValueError("expect string argument to `!py_eval`")
    try:
        res = eval(cmd, loader.eval_context)
    except:
        print("\nError while yaml parsing the following !py_eval command:\n", cmd, "\n")
        raise
    return res


try:
    import yaml
except ImportError:
    yaml = None

if yaml is None:
    _YamlLoaderWithPyEval = None
else:
    class _YamlLoaderWithPyEval(yaml.FullLoader):
        eval_context = {}

    yaml.add_constructor("!py_eval", _yaml_eval_constructor, Loader=_YamlLoaderWithPyEval)


def load_yaml_with_py_eval(filename=None, yaml_content=None, context={'np': numpy}):
    """Load a yaml file with support for an additional `!py_eval` tag.

    When defining yaml parameter files, it's sometimes convenient to just have python snippets
    in there, e.g. to get fractions of pi or expand last lists.

    This function loads a yaml file supporting such (short) python snippets
    that get evaluated by python's ``eval(snippet)``.

    It expects one string of python code following the ``!py_eval`` tag.
    The most reliable method to pass the python code is to use a literal
    string in yaml, as shown in the example below.

    .. code :: yaml

        a: !py_eval |
            2**np.arange(6, 10)
        b: !py_eval |
            [10, 15] + list(range(20, 31, 2)) + [35, 40]
        c: !py_eval "2*np.pi * 0.3"

    Note that a subsequent ``yaml.dump()`` might contain ugly parts if you construct
    generic python objects, e.g., a numpy array scalar like ``np.arange(10)[0]``.
    If you want to avoid this, you can explicitly convert back to lists before.


    .. warning ::

        Like pickle, it is not safe to load a yaml file from an untrusted source! A malicious
        file can call any Python function and should thus be treated with extreme caution.

    Parameters
    ----------
    filename : str | None
        Filename of the file to load.
    yaml_content : str | None
        Alternatively to filename directly the content of the yaml file.
        Pass either `filename` or `yaml_content`.
    context : dict
        The context of ``globals()`` passed to `eval`.

    Returns
    -------
    config :
        Data (typically nested dictionary) as defined in the yaml file.

    """
    if _YamlLoaderWithPyEval is None:
        raise RuntimeError('Could not import yaml. Consider installing the pyyaml package.')

    _YamlLoaderWithPyEval.eval_context = context

    if filename is not None:
        with open(filename, 'r') as stream:
            config = yaml.load(stream, Loader=_YamlLoaderWithPyEval)
    elif yaml_content is not None:
        config = yaml.load(yaml_content, Loader=_YamlLoaderWithPyEval)
    else:
        raise ValueError("pass either filename or yaml_content!")
    return config
