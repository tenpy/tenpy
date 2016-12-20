"""Tools to handle paramters for algorithms.

See the doc-string of :func:`get_parameter` for details.

.. todo :
    If a function (like `tenpy.algorithms.truncation` using `get_parameter` gets called very
    often, we might want a way to print the parameters only for the first time they are used.
    Maybe add a special keyword `_used_par` wich is a set?
    That would also allow for a check for miss-spelled parameters....
"""

__all__ = ["get_parameter"]

def get_parameter(par_dict, key, default, descr=''):
    """Read out a parameter from the dictionary and/or provide default values.

    This function provides a similar functionality as ``par_dict.get(key, default)``.
    *Unlike* `dict.get` this function writes the default value into the dictionary
    (i.e. in other words it's more similar to ``par_dict.set_default(key, default)``).

    However, a special key ``'verbose'`` (with value > 0) *in* the `par_dict`
    triggers this function to additionally print the used (default) value.

    This function should be used in the algorithms to read out parameters.
    Then, when the algorithms are calleed by TenPy users,
    simply including ``verbose=1`` into a parameter dictionary will trigger the algorithms to
    print all the actually used parameters during runtime.

    In addition, the user can save the modified dictionary along with other data, which gives a
    concrete record of the actually used parameters and simplifies reproducing the results.

    Parameters
    ----------
    par_dict : dict
        A dicionary of the parameters as provided by the user.
        If `key` is not a valid key, par_dict[key] is set to the `default`
    key : string
        The key for the parameter which should be read out from the dictionary.
    default :
        The default value for the parameter.
    descr : str
        A short description like 'TEBD', 'XXZ_model', which is used for verbose output.

    Returns
    -------
    value :
        ``par_dict[key]`` if the key is in par_dict, otherwise `default`.

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
    set parameter 'dt'=0.01 (default) for TEBD

    Of course you can also provide the parameter to use a non-default value:

    >>> tenpy.algorithms.tebd.time_evolution(..., dict(dt=0.1, verbose=1))
    set parameter 'dt'=0.1 for TEBD

    """
    verbose = par_dict.get('verbose', 0)
    val = par_dict.get(key, default)
    if verbose > 0:
        defaultstring = "" if key in par_dict else "(default) "
        print "set parameter {key!r}={val!s} {defaultstring}for {descr!s}".format(
            descr=descr, key=key, val=val, defaultstring=defaultstring)
    return val
