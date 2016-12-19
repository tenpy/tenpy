"""Tools to handle paramters for algorithms.

"""


def get_parameter(par_dict, key, default, descr=''):
    """read out a parameter from the dictionary or provide default value.

    Examples
    --------
    In a function of the algorithms, use this function for read-out of parameters:

    >>> def model_calc_U(U_param):
    >>>    dt = get_parameter(U_param, 'dt', 0.01, 'TEBD')
    >>>    # calculate exp(-i * dt* H) ....

    Then, when you call the algoithm without any parameters, they just use the default values:

    >>> tenpy.algorithms.tebd.time_evolution(..., dict())  # calls model_calc_U at some point

    If you provide a special keyword ``'verbose'`` which will triger this function to print the
    used parameter values.
    This is an easy way to get a list of all the possible parameters you can change.

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
