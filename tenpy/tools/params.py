"""Tools to handle paramters for algorithms.

See the doc-string of :func:`get_parameter` for details.
"""
# Copyright 2018 TeNPy Developers

import warnings
import numpy as np

__all__ = ["get_parameter", "unused_parameters"]


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
