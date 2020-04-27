"""Tools to handle config options/paramters for algorithms.

See the doc-string of :func:`get_parameter` for details.
"""
# Copyright 2018-2020 TeNPy Developers, GNU GPLv3

import warnings
import numpy as np
from collections.abc import MutableMapping

from .hdf5_io import Hdf5Exportable

__all__ = ["Config", "get_parameter", "unused_parameters"]


class Config(MutableMapping, Hdf5Exportable):
    """Wrapper class for parameter dictionaries.

    Parameters
    ----------
    params : dict
        Dictionary containing the actual parameters.
    name : str
        Descriptive name of the parameter set used for verbose printing.

    Attributes
    ----------
    documentation : dict
        Contains type and general information for parameters.
    name : str
        Name of the dictionary, for output statements. For example, when using
        a `Config` class for DMRG parameters, ``name='DMRG'``.
    paramss : dict
        Dictionary containing the actual parameters.
    unused : set
        Keeps track of any parameters not yet used.
    verbose : int
        Verbosity level for output statements.
    """
    def __init__(self, params, name):
        self.params = params
        self.unused = set(params.keys())
        self.verbose = params.get('verbose', 0)
        self.documentation = {}
        self.name = name

    def __getitem__(self, key):
        self.print_if_verbose(key, "Reading")
        self.unused.discard(key)
        return self.params[key]

    def __setitem__(self, key, value):
        self.print_if_verbose(key, "Setting")
        self.params[key] = value

    def __delitem__(self, key):
        self.print_if_verbose(key, "Deleting")
        self.unused.discard(key)
        del self.params[key]

    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self.params)

    def __str__(self):
        return repr(self)  # TODO This is not what we want

    def __repr__(self):
        return "<Config, {0!s} parameters>".format(len(self))

    def __del__(self):
        unused = self.unused
        if len(unused) > 0:
            if len(unused) > 1:
                msg = "unused parameters for config {name!s}:\n{keys!s}"
            else:
                msg = "unused parameter {keys!s} for config {name!s}\n"
            warnings.warn(msg.format(keys=sorted(unused), name=self.name))
        return unused

    def keys(self):
        return self.params.keys()

    def get(self, key, default):
        """Find the value of `key`; really more like `setdefault` of a :class:`dict`.

        If no value is set, return `default` and set the value of `key` to
        `default` internally.

        Parameters
        ----------
        key : str
            Key name for the parameter being read out.
        default :
            Default value for the parameter.

        Returns
        -------
        val : any type
            The value for `key` if it existed, `default` otherwise.
        """
        use_default = key not in self.keys()
        if use_default:
            self.unused.add(key)
        val = self.params.setdefault(key, default)  # get the value; set default if not existent
        self.print_if_verbose(key, "Reading", use_default)
        self.unused.discard(key)  # (does nothing if key not in set)
        return val

    def setdefault(self, key, default):
        """Set a default value without reading it out.

        Parameters
        ----------
        key : str
            Key name for the parameter being set.
        default :
            The value to be set by default if the parameter is not yet set.
        """
        use_default = key not in self.keys()
        self.params.setdefault(key, default)  # get the value; set default if not existent
        self.print_if_verbose(key, "Set default", not use_default)
        self.unused.discard(key)  # (does nothing if key not in set)
        # do no return the value: not added to self.unused!

    def print_if_verbose(self, key, action=None, use_default=False):
        """Print out `key` if verbosity and other conditions are met.

        Parameters
        ----------
        key : str
            Key name for the parameter being read out.
        action : str, optional
            Use to adapt printout message to specific actions (e.g. "Deleting")
        """
        val = self.params[key]
        name = self.name
        verbose = self.verbose
        new_key = key in self.unused
        if verbose >= 100 or (new_key and verbose >= (2. if use_default else 1.)):
            actionstring = "Parameter" if action is None else action
            defaultstring = "(default) " if use_default else ""
            print("{actionstring} {key!r}={val!r} {defaultstring}for {name!s}".format(
                actionstring=actionstring,
                name=name,
                key=key,
                val=val,
                defaultstring=defaultstring))

    def any_nonzero(self, keys, verbose_msg=None):
        """Check for any non-zero or non-equal entries in some parameters.

        .. todo ::
            Currently, if k is a tuple, only checks equality between k[0] and
            any other element. Should potentially be generalized.

        Parameters
        ----------
        keys : list of {key | tuple of keys}
            For a single key, check ``self[key]`` for non-zero entries.
            For a tuple of keys, all the ``self[key]`` have to be equal (as numpy arrays).
        verbose_msg : None | str
            If :attr:`verbose` >= 1, we print `verbose_msg` before checking,
            and a short notice with the `key`, if a non-zero entry is found.

        Returns
        -------
        match : bool
            False, if all ``self[key]`` are zero or `None` and
            True, if any of the ``self[key]`` for single `key` in `keys`,
            or if any of the entries for a tuple of `keys`
        """
        verbose = (self.verbose > 1.)
        for k in keys:
            if isinstance(k, tuple):
                # check equality
                if self.has_nonzero(k[0]):
                    val = self.params[k[0]]
                    for k1 in k[1:]:
                        if self.has_nonzero(k1):
                            param_val = self.params[k1]
                        if not np.array_equal(val, param_val):
                            if verbose:
                                print("{k0!r} and {k1!r} have different entries.".format(k0=k[0],
                                                                                         k1=k1))
                            return True
            else:
                if self.has_nonzero(k):
                    if verbose:
                        print(verbose_msg)
                        print(str(k) + " has nonzero entries")
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
        return (key in self.keys() and np.any(np.array(self.params[key])) != 0
                and self.params[key] is not None)

    def save_yaml(self, filename):
        """Save the parameters to `filename` as a YAML file.

        Parameters
        ----------
        filename : str
            Name of the resulting YAML file.
        """
        import yaml
        with open(filename, 'w') as stream:
            yaml.dump(self.params, stream)

    @classmethod
    def from_yaml(cls, filename, name):
        """Load a `Config` instance from a YAML file containing the `params`.

        .. warning ::
            Like pickle, it is not safe to load a yaml file from an untrusted source! A malicious
            file can call any Python function and should thus be treated with extreme caution.

        Parameters
        ----------
        filename : str
            Name of the YAML file
        name : str
            Name of the resulting :class:`Config` instance.

        Returns
        -------
        obj : Config
            A `Config` object, loaded from file.
        """
        import yaml
        with open(filename, 'r') as stream:
            params = yaml.safe_load(stream)
        return cls(params, name)


def asconfig(params, name):
    """Convert a dict-like `params` to a :class:`Config`.

    Parameters
    ----------
    params : dict | :class:`Config`
        If this is a :class:`Config`, just return it.
        Otherwise, create a :class:`Config` from it and return that.
    name : str
        Name to be used for the :class:`Config`.

    Returns
    -------
    config : :class:`Config`
        Either directly `params` or ``Config(params, name)``.
    """
    if isinstance(params, Config):
        return params
    return Config(params, name)


def get_parameter(params, key, default, descr, asarray=False):
    """Read out a parameter from the dictionary and/or provide default values.

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
    In the algorith
    :class:`~tenpy.algorithms.tebd.Engine` gets a dictionary of parameters.
    Beside doing other stuff, it calls :meth:`tenpy.models.model.NearestNeighborModel.calc_U_bond`
    with the dictionary as argument, which looks similar like:

    >>> def model_calc_U(U_param):
    >>>    dt = get_parameter(U_param, 'dt', 0.01, 'TEBD')
    >>>    # ... calculate exp(-i * dt* H) ....

    Then, when you call `time_evolution` without any parameters, it just uses the default value:

    >>> tenpy.algorithms.tebd.time_evolution(..., dict())  # uses dt=0.01

    If you provide the special keyword ``'verbose'`` you can triger this function to print the
    used parameter values:

    >>> tenpy.algorithms.tebd.time_evolution(..., dict(verbose=1))
    parameter 'dt'=0.01 (default) for TEBD

    Of course you can also provide the parameter to use a non-default value:

    >>> tenpy.algorithms.tebd.time_evolution(..., dict(dt=0.1, verbose=1))
    parameter 'dt'=0.1 for TEBD

    """
    msg = ("Old-style parameter dictionaries are deprecated in favor of "
           "`Config` class objects. Use `Config` methods to read out "
           "parameters.")
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

    Parameters
    ----------
    params : dict
        A dictionary of parameters which was given to (functions using) :meth:`get_parameter`
    warn : None | str
        If given, print a warning "unused parameter for {warn!s}: {unused_keys!s}".

    Returns
    -------
    unused_keys : set
        The set of keys of the params which was not used
    """
    msg = ("Old-style parameter dictionaries are deprecated in favor of "
           "`Config` class objects. Use `Config.unused` attribute to "
           "get unused parameters.")
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
