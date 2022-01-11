"""Tools to handle config options/paramters for algorithms.

See the doc-string of :class:`Config` for details.
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import warnings
import numpy as np
from collections.abc import MutableMapping
import pprint
import os
import logging
logger = logging.getLogger(__name__)

from .hdf5_io import ATTR_FORMAT

__all__ = ["Config", "asConfig", "get_parameter", "unused_parameters"]


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

    @property
    def verbose(self):
        warnings.warn(
            "verbose is deprecated, we're using logging now! \n"
            "See https://tenpy.readthedocs.io/en/latest/intro/logging.html", FutureWarning, 2)
        return self.options.get('verbose', 1.)

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

        Subconfigs are recursivley converted to dict.
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
        import yaml
        with open(filename, 'r') as stream:
            config = yaml.safe_load(stream)
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
                msg = "unused option {keys!s} for config {name!s}\n"
            warnings.warn(msg.format(keys=sorted(unused), name=self.name))
            self.unused.clear()  # don't warn twice about the same parameters
        if recursive:
            for val in self.options.values():
                if isinstance(val, Config):
                    val.warn_unused(True)

    def keys(self):
        return self.options.keys()

    def get(self, key, default):
        """Find the value of `key`; really more like `setdefault` of a :class:`dict`.

        If no value is set, return `default` and set the value of `key` to `default` internally.

        Parameters
        ----------
        option : str
            Key for the option being read out.
        default :
            Default value for the parameter.

        Returns
        -------
        val :
            The value for `option` if it existed, `default` otherwise.
        """
        use_default = key not in self.options.keys()
        val = self.options.setdefault(key, default)  # get & set default if not existent
        self.log(key, "reading", use_default)
        self.unused.discard(key)  # (does nothing if key not in set)
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
        """Mark `keys` as read out to supress warnings about those keys being unused.

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


def get_parameter(params, key, default, descr, asarray=False):
    """Read out a parameter from the dictionary and/or provide default values.

    .. deprecated :: 0.6.0
        Use the :class:`Config` instead.

    This function provides a similar functionality as ``params.get(key, default)``.
    *Unlike* `dict.get` this function writes the default value into the dictionary
    (i.e. in other words it's more similar to ``params.setdefault(key, default)``).

    This allows the user to save the modified dictionary as meta-data, which gives a
    concrete record of the actually used parameters and simplifies reproducing the results
    and restarting simulations.

    Moreover, a special entry with the key ``'verbose'`` *in* the `params`
    can trigger this function to also print the used value.
    A higer `verbose` level implies more output.
    If `verbose` >= 100, it is printed every time it's used.
    If `verbose` >= 2., its printed for the first time time its used.
    and for `verbose` >= 1, non-default values are printed the first time they are used.
    otherwise only for the first use.

    Internally, whether a parameter was used is saved in the set ``params['_used_param']``.
    This is used in :func:`unused_parameters` to print a warning if the key wasn't used
    at the end of the algorithm, to detect mis-spelled parameters.

    Parameters
    ----------
    params : dict
        A dicionary of the parameters as provided by the user.
        If `key` is not a valid key, ``params[key]`` is set to `default`.
    key : string
        The key for the parameter which should be read out from the dictionary.
    default :
        The default value for the parameter.
    descr : str
        A short description for verbose output, like 'TEBD', 'XXZ_model', 'truncation'.
    asarray : bool
        If True, convert the result to a numpy array with ``np.asarray(...)`` before returning.

    Returns
    -------
    value :
        ``params[key]`` if the key is in params, otherwise `default`.
        Converted to a numpy array, if `asarray`.

    Examples
    --------
    In the algorithm
    :class:`~tenpy.algorithms.tebd.TEBDEngine` gets a dictionary of parameters.
    Beside doing other stuff, it calls :meth:`tenpy.models.model.NearestNeighborModel.calc_U_bond`
    with the dictionary as argument, which looks similar like:

    >>> from tenpy.tools.params import get_parameter
    >>> def model_calc_U(params):
    ...    dt = get_parameter(params, 'dt', 0.01, 'TEBD')
    ...    order = get_parameter(params, 'order', 1, 'TEBD')
    ...    print("calc U with dt =", dt, "and order =", order )
    ...    # ... calculate exp(-i * dt* H) ....

    Then, when you call it without any parameters, it just uses the default value:

    >>> model_calc_U(dict())
    calc U with dt = 0.01 and order = 1

    Of course you can also provide the parameter to use a non-default value:

    >>> model_calc_U(dict(dt=0.02))
    calc U with dt = 0.02 and order = 1


    Increasing the special keyword ``'verbose'`` generally prints more:

    >>> model_calc_U(dict(dt=0.02, verbose=1))
    parameter 'dt'=0.02 for TEBD
    calc U with dt = 0.02 and order = 1
    >>> model_calc_U(dict(dt=0.02, verbose=2))
    parameter 'dt'=0.02 for TEBD
    parameter 'order'=1 (default) for TEBD
    calc U with dt = 0.02 and order = 1
    """
    msg = ("Old-style parameter dictionaries are deprecated in favor of `Config` class objects. "
           "Use `Config` methods to read out parameters. "
           "In particular, inside models just use `model_params.get(key, default)`.")
    warnings.warn(msg, category=FutureWarning, stacklevel=2)
    if isinstance(params, Config):
        return params.get(key, default)
    use_default = key not in params
    val = params.setdefault(key, default)  # get the value; set default if not existent
    used = params.setdefault('_used_param', set())
    verbose = params.get('verbose', 0)
    new_key = key not in used
    if verbose >= 100 or (new_key and verbose >= (2. if use_default else 1.)):
        defaultstring = "(default) " if use_default else ""
        print("parameter {key!r}={val!r} {defaultstring}for {descr!s}".format(
            descr=descr, key=key, val=val, defaultstring=defaultstring))
    used.add(key)  # (does nothing if already present)
    if asarray:
        val = np.asarray(val)
    return val


def unused_parameters(params, warn=None):
    """Returns a set of the parameters which have not been read out with `get_parameters`.

    This function might be useful to check for typos in the parameter keys.

    .. deprecated :: 0.6.0
        Use the :class:`Config` instead.

    Parameters
    ----------
    params : dict
        A dictionary of parameters which was given to (functions using) :func:`get_parameter`
    warn : None | str
        If given, print a warning "unused parameter for {warn!s}: {unused_keys!s}".

    Returns
    -------
    unused_keys : set
        The set of keys of the params which was not used
    """
    msg = ("Old-style parameter dictionaries are deprecated in favor of `Config` class objects. "
           "Using `unused_parameters` to warn about non-used parameters is no longer necessary; "
           "this is now done during garbage collection.")
    warnings.warn(msg, category=FutureWarning, stacklevel=2)
    if isinstance(params, Config):
        return params.unused
    used = params.get('_used_param', set())
    unused = set(params.keys()) - used
    unused.discard('_used_param')
    unused.discard('verbose')
    if warn is not None:
        if len(unused) > 0:
            if len(unused) > 1:
                msg = "unused parameters for {descr!s}:\n{keys!s}"
            else:
                msg = "unused parameter {keys!s} for {descr!s}\n"
            warnings.warn(msg.format(keys=sorted(unused), descr=warn))
    return unused
