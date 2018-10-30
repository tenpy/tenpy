"""Tools to handle paramters for algorithms.

See the doc-string of :func:`get_parameter` for details.
"""
# Copyright 2018 TeNPy Developers

import warnings
import numpy as np

__all__ = ["get_parameter", "unused_parameters"]


def get_parameter(par_dict, key, default, descr, asarray=False):
    """Read out a parameter from the dictionary and/or provide default values.

    This function provides a similar functionality as ``par_dict.get(key, default)``.
    *Unlike* `dict.get` this function writes the default value into the dictionary
    (i.e. in other words it's more similar to ``par_dict.setdefault(key, default)``).

    However, a special key ``'verbose'`` *in* the `par_dict` with value > 0 triggers this function
    to additionally print the used value. If verbose >= 10, it is printed every time its used,
    otherwise only for the first use.
    (Wheter a parameter was used is saved in the set ``par_dict['_used_param']``.)

    This function should be used in the algorithms to read out parameters.
    Then, when the algorithms are calleed by tenpy users,
    simply including ``verbose=1`` into a parameter dictionary will trigger the algorithms to
    print all the actually used parameters during runtime.

    In addition, the user can save the modified dictionary along with other data, which gives a
    concrete record of the actually used parameters and simplifies reproducing the results.

    Parameters
    ----------
    par_dict : dict
        A dicionary of the parameters as provided by the user.
        If `key` is not a valid key, ``par_dict[key]`` is set to `default`.
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
        ``par_dict[key]`` if the key is in par_dict, otherwise `default`.
        Converted to a numpy array, if `asarray`.

    Examples
    --------
    :func:`~tenpy.algorithms.tebd.time_evolution` gets a dictionary of parameters.
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
    defaultstring = "" if key in par_dict else "(default) "
    val = par_dict.setdefault(key, default)  # get the value; set default if not existent
    used = par_dict.setdefault('_used_param', set())
    verbose = par_dict.get('verbose', 0)
    if verbose >= 100 or (key not in used and verbose > 0):
        print("parameter {key!r}={val!r} {defaultstring}for {descr!s}".format(
            descr=descr, key=key, val=val, defaultstring=defaultstring))
    used.add(key)  # (does nothing if already present)
    if asarray:
        val = np.asarray(val)
    return val


def unused_parameters(par_dict, warn=None):
    """Returns a set of the parameters which have not been read out with `get_parameters`.

    This function might be useful to check for typos in the parameter keys.

    Parameters
    ----------
    par_dict : dict
        A dictionary of parameters which was given to (functions using) :meth:`get_parameter`
    warn : None | str
        If given, print a warning "unused parameter for {warn!s}: {unused_keys!s}".

    Returns
    -------
    unused_keys : set
        The set of keys of the par_dict which was not used
    """
    used = par_dict.get('_used_param', set())
    unused = set(par_dict.keys()) - used
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
