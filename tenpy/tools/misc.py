"""Miscellaneous tools, somewhat random mix yet often helpful."""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np
from .optimization import bottleneck
from .process import omp_set_nthreads
from .params import Config
import random
import os
import itertools
import argparse
import warnings

__all__ = [
    'to_iterable', 'to_iterable_of_len', 'to_array', 'anynan', 'argsort', 'lexsort',
    'inverse_permutation', 'list_to_dict_list', 'atleast_2d_pad', 'transpose_list_list',
    'zero_if_close', 'pad', 'any_nonzero', 'add_with_None_0', 'chi_list', 'group_by_degeneracy',
    'get_close', 'find_subclass', 'get_recursive', 'set_recursive', 'flatten',
    'build_initial_state', 'setup_executable'
]


def to_iterable(a):
    """If `a` is a not iterable or a string, return ``[a]``, else return ``a``."""
    if type(a) == str:
        return [a]
    try:
        iter(a)
    except TypeError:
        return [a]
    else:
        return a


def to_iterable_of_len(a, L):
    """If a is a non-string iterable of length `L`, return `a`, otherwise return [a]*L.

    Raises ValueError if `a` is already an iterable of different length.
    """
    if type(a) == str:
        return [a] * L
    try:
        iter(a)
    except TypeError:
        return [a] * L
    # else:
    if len(a) != L:
        raise ValueError("wrong length: got {0:d}, expected {1:d}".format(len(a), L))
    return a


def to_array(a, shape=(None, ), dtype=None):
    """Convert `a` to an numpy array and tile to matching dimension/shape.

    This function provides similar functionality as numpys broadcast, but not quite the same:
    Only scalars are broadcasted to higher dimensions,
    for a non-scalar, we require the number of dimension to match.
    If the shape does not match, we repeat periodically, e.g. we tile ``(3, 4) -> (6, 16)``,
    but ``(4, 4) -> (6, 16)`` will raise an error.

    Parameters
    ----------
    a : scalar | array_like
        The input to be converted to an array. A scalar is reshaped to the desired dimension.
    shape : tuple of {None | int}
        The desired shape of the array. An entry ``None`` indicates arbitrary len >=1.
        For int entries, tile the array periodically to fit the len.
    dtype :
        Optionally specifies the data type.

    Returns
    -------
    a_array : ndarray
        A copy of `a` converted to a numpy ndarray of desired dimension and shape.
    """
    a = np.array(a, dtype=dtype)  # copy
    if a.ndim != len(shape):
        if a.size == 1:
            a = np.reshape(a, [1] * len(shape))
        else:  # extending dimensions is ambiguous, so we better raise an Error.
            raise ValueError("don't know how to cast `a` to required dimensions.")
    reps = [1] * a.ndim
    for i in range(a.ndim):
        if shape[i] is None:
            continue
        if shape[i] % a.shape[i] != 0:
            raise ValueError("incomensurate len for tiling from {0:d} to {1:d}".format(
                a.shape[i], shape[i]))
        reps[i] = shape[i] // a.shape[i]
    return np.tile(a, reps)


if bottleneck is not None:
    anynan = bottleneck.anynan
else:

    def anynan(a):
        """check whether any entry of a ndarray `a` is 'NaN'."""
        return np.isnan(np.sum(a))  # still faster than 'np.isnan(a).any()'


def argsort(a, sort=None, **kwargs):
    """wrapper around np.argsort to allow sorting ascending/descending and by magnitude.

    Parameters
    ----------
    a : array_like
        The array to sort.
    sort : ``'m>', 'm<', '>', '<', None``
        Specify how the arguments should be sorted.

        ==================== =============================
        `sort`               order
        ==================== =============================
        ``'m>', 'LM'``       Largest magnitude first
        -------------------- -----------------------------
        ``'m<', 'SM'``       Smallest magnitude first
        -------------------- -----------------------------
        ``'>', 'LR', 'LA'``  Largest real part first
        -------------------- -----------------------------
        ``'<', 'SR', 'SA'``  Smallest real part first
        -------------------- -----------------------------
        ``'LI'``             Largest imaginary part first
        -------------------- -----------------------------
        ``'SI'``             Smallest imaginary part first
        -------------------- -----------------------------
        ``None``             numpy default: same as '<'
        ==================== =============================

    **kwargs :
        Further keyword arguments given directly to :func:`numpy.argsort`.

    Returns
    -------
    index_array : ndarray, int
        Same shape as `a`, such that ``a[index_array]`` is sorted in the specified way.
    """
    if sort is not None:
        if sort == 'm<' or sort == 'SM':
            a = np.abs(a)
        elif sort == 'm>' or sort == 'LM':
            a = -np.abs(a)
        elif sort == '<' or sort == 'SR' or sort == 'SA':
            a = np.real(a)
        elif sort == '>' or sort == 'LR' or sort == 'LA':
            a = -np.real(a)
        elif sort == 'SI':
            a = np.imag(a)
        elif sort == 'LI':
            a = -np.imag(a)
        else:
            raise ValueError("unknown sort option " + repr(sort))
    return np.argsort(a, **kwargs)


def lexsort(a, axis=-1):
    """wrapper around ``np.lexsort``: allow for trivial case ``a.shape[0] = 0`` without sorting"""
    if any([s == 0 for s in a.shape]):
        return np.arange(a.shape[axis], dtype=np.intp)
    return np.lexsort(a, axis=axis)


def inverse_permutation(perm):
    """reverse sorting indices.

    Sort functions (as :meth:`LegCharge.sort`) return a (1D) permutation `perm` array,
    such that ``sorted_array = old_array[perm]``.
    This function inverts the permutation `perm`,
    such that ``old_array = sorted_array[inverse_permutation(perm)]``.

    Parameters
    ----------
    perm : 1D array_like
        The permutation to be reversed. *Assumes* that it is a permutation with unique indices.
        If it is, ``inverse_permutation(inverse_permutation(perm)) == perm``.

    Returns
    -------
    inv_perm : 1D array (int)
        The inverse permutation of `perm` such that ``inv_perm[perm[j]] = j = perm[inv_perm[j]]``.
    """
    perm = np.asarray(perm, dtype=np.intp)
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(perm.shape[0], dtype=perm.dtype)
    return inv_perm
    # equivalently: return np.argsort(perm) # would be O(N log(N))


def list_to_dict_list(l):
    """Given a list `l` of objects, construct a lookup table.

    This function will handle duplicate entries in `l`.

    Parameters
    ----------
    l: iterable of iterabele of immutable
        A list of objects that can be converted to tuples to be used as keys for a dictionary.

    Returns
    -------
    lookup : dict
        A dictionary with (key, value) pairs ``(key):[i1,i2,...]``
        where ``i1, i2, ...`` are the indices where `key` is found in `l`:
        i.e. ``key == tuple(l[i1]) == tuple(l[i2]) == ...``
    """
    d = {}
    for i, r in enumerate(l):
        k = tuple(r)
        try:
            d[k].append(i)
        except KeyError:
            d[k] = [i]
    return d


def atleast_2d_pad(a, pad_item=0):
    """Transform `a` into a 2D array, filling missing places with `pad_item`.

    Given a list of lists, turn it to a 2D array (pad with 0), or turn a 1D list to 2D.

    Parameters
    ----------
    a : list of lists
        to be converted into ad 2D array.

    Returns
    -------
    a_2D : 2D ndarray
        a converted into a numpy array.

    Examples
    --------
    .. testsetup ::

        from tenpy.tools.misc import *


    >>> atleast_2d_pad([3, 4, 0])
    array([[3, 4, 0]])

    >>> atleast_2d_pad([[3, 4], [1, 6, 7]])
    array([[3., 4., 0.],
           [1., 6., 7.]])
    """
    iter(a)  # check that a is at least 1D iterable
    if len(a) == 0:
        return np.zeros([0, 0])
    # Check if every element of a is a list
    is_list_of_list = True
    for s in a:
        try:
            iter(s)
        except TypeError:
            is_list_of_list = False
            break
    if not is_list_of_list:
        return np.array([a])
    maxlen = max([len(s) for s in a])
    # Pad if necessary
    a = [np.hstack([s, [pad_item] * (maxlen - len(s))]) for s in a]
    return np.array(a)


def transpose_list_list(D, pad=None):
    """Returns a list of lists T, such that ``T[i][j] = D[j][i]``.

    Parameters
    ----------
    D : list of list
        to be transposed
    pad :
        Used to fill missing places, if D is not rectangular.

    Returns
    -------
    T : list of lists
        transposed, rectangular version of `D`.
        constructed such that ``T[i][j] = D[j][i] if i < len(D[j]) else pad``
    """
    nRow = len(D)
    if nRow == 0:
        return [[]]
    nCol = max([len(R) for R in D])
    T = [[pad] * nRow for i in range(nCol)]
    for j, R in enumerate(D):
        for i, e in enumerate(R):
            T[i][j] = e
    return T


def zero_if_close(a, tol=1.e-15):
    """set real and/or imaginary part to 0 if their absolute value is smaller than `tol`.

    Parameters
    ----------
    a : ndarray
        numpy array to be rounded
    tol : float
        the threashold which values to consider as '0'.
    """
    if a.dtype == np.complex128 or a.dtype == np.complex64:
        ar = np.choose(np.abs(a.real) < tol, [a.real, np.zeros(a.shape)])
        ai = np.choose(np.abs(a.imag) < tol, [a.imag, np.zeros(a.shape)])
        return ar + 1j * ai
    else:
        return np.choose(np.abs(a) < tol, [a, np.zeros_like(a)])


def pad(a, w_l=0, v_l=0, w_r=0, v_r=0, axis=0):
    """Pad an array along a given `axis`.

    Parameters
    ----------
    a : ndarray
        the array to be padded
    w_l : int
        the width to be padded in the front
    v_l : dtype
        the value to be inserted before `a`
    w_r : int
        the width to be padded after the last index
    v_r : dtype
        the value to be inserted after `a`
    axis : int
        the axis along which to pad

    Returns
    -------
    padded : ndarray
        a copy of `a` with enlarged `axis`, padded with the given values.
    """
    shp = list(a.shape)
    shp[axis] += w_r + w_l
    b = np.empty(shp, a.dtype)
    # tuple of full slices
    take = [slice(None) for j in range(len(shp))]
    # prepend
    take[axis] = slice(w_l)
    b[tuple(take)] = v_l
    # copy a
    take[axis] = slice(w_l, -w_r)
    b[tuple(take)] = a
    # append
    take[axis] = slice(-w_r, None)
    b[tuple(take)] = v_r
    return b


def any_nonzero(params, keys, verbose_msg=None):
    """Check for any non-zero or non-equal entries in some parameters.

    Parameters
    ----------
    params : dict | Config
        A dictionary of parameters, or a :class:`~tenpy.tools.params.Config`
        instance.
    keys : list of {key | tuple of keys}
        For a single key, check ``params[key]`` for non-zero entries.
        For a tuple of keys, all the ``params[key]`` have to be equal (as numpy arrays).
    verbose_msg : None | str
        If params['verbose'] >= 1, we print `verbose_msg` before checking,
        and a short notice with the `key`, if a non-zero entry is found.

    Returns
    -------
    match : bool
        False, if all params[key] are zero or `None` and
        True, if any of the params[key] for single `key` in `keys`,
        or if any of the entries for a tuple of `keys`
    """
    msg = ("tools.misc.any_nonzero() is deprecated in favor of "
           "tools.params.Config.any_nonzero().")
    warnings.warn(msg, category=FutureWarning, stacklevel=2)
    if isinstance(params, Config):
        return params.any_nonzero(keys, verbose_msg)
    verbose = (params.get('verbose', 0) > 1.)
    for k in keys:
        if isinstance(k, tuple):
            # check equality
            val = params.get(k[0], None)
            for k1 in k[1:]:
                if not np.array_equal(val, params.get(k1, None)):
                    if verbose:
                        print("{k0!r} and {k1!r} have different entries.".format(k0=k[0], k1=k1))
                    return True
        else:
            val = params.get(k, None)
            if val is not None and np.any(np.array(val) != 0.):  # count `None` as zero
                if verbose:
                    print(verbose_msg)
                    print(str(k) + " has nonzero entries")
                return True
    return False


def add_with_None_0(a, b):
    """Return ``a + b``, treating `None` as zero.

    Parameters
    ----------
    a, b :
        The two things to be added, or ``None``.

    Returns
    -------
    sum :
        ``a + b``, except if `a` or `b` is `None`, in which case the other variable is returned.
    """
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def chi_list(chi_max, dchi=20, nsweeps=20, verbose=0):
    warnings.warn("Deprecated: moved `chi_list` to `tenpy.algorithms.dmrg.chi_list`.",
                  category=FutureWarning,
                  stacklevel=2)
    from tenpy.algorithms import dmrg
    chi_list = dmrg.chi_list(chi_max, dchi, nsweeps)
    if verbose:
        import pprint
        print("chi_list = ")
        pprint.pprint(chi_list)
    return chi_list


def group_by_degeneracy(E, *args, subset=None, cutoff=1.e-12):
    """Find groups of indices for which (energy) values are degenerate.

    Parameters
    ----------
    values : 1D array
        Values (e.g. energies) which need to be close to count as degenerate.
    *args : 1D array
        Additional vectors (with same length as `values`),
        which also need to be close (up to cutoff) to count as degenerate.
    subset : 1D array
        Optionally selects a subset of the indices
    cutoff : float
        Precision up to which values still count as degenerate.

    Returns
    -------
    idx_groups : list of tuple of int
        Each tuple `group` contains indices ``i, j, k, ...`` for which the values are closer than
        `cutoff`, i.e., ``|E[j, k, ...] - E[i]| <= cutoff``.
        Each index appears exactly once (if it is containted in `subset`).

    .. testsetup ::

        from tenpy.tools.misc import *

    >>> E = [2., 2.4, 1.9999, 1.8, 2.3999, 5, 1.8]
    ... # -> 0   1    2       3    4       5  6
    >>> k = [0,  1,   2,      2,   1,      2, 1]
    >>> group_by_degeneracy(E, cutoff=0.001)
    [(0, 2), (1, 4), (3, 6), (5,)]
    >>> group_by_degeneracy(E, k, cutoff=0.001)  # k and E need to be close
    [(0,), (1, 4), (2,), (3,), (5,), (6,)]

    """
    assert cutoff >= 0.
    E = np.asarray(E)
    args = [np.asarray(arg) for arg in args]
    N, = E.shape
    groups = []
    if subset is None:
        subset = np.arange(N, dtype=np.intp)
    else:
        subset = np.asarray(subset, dtype=np.intp)
    while len(subset) > 0:
        x = subset[0]
        group = np.abs(E[subset] - E[x]) <= cutoff
        for arg in args:
            group = np.logical_and(group, np.abs(arg[subset] - arg[x]) <= cutoff)
        groups.append(tuple(subset[group]))
        subset = subset[np.logical_not(group)]
    return groups


def get_close(values, target, default=None, eps=1.e-13):
    """Iterate through `values` and return first entry closer than `eps`.

    Parameters
    ----------
    values : interable of float
        Values to compare to.
    target : float
        Value to find.
    default :
        Returned if no value close to `target` is found.
    eps : float
        Tolerance what counts as "close", namely everything with ``abs(val-target) < eps``.

    Returns
    -------
    value : float
        An entry of `values`, if one close to `target` is found, otherwise `default`.
    """
    for v in values:
        if abs(v - target) < eps:
            return v
    return default


def find_subclass(base_class, subclass_name):
    """For a given base class, recursively find the subclass with the given name.

    Parameters
    ----------
    base_class : class
        The base class of which `subclass_name` is supposed to be a subclass.
    subclass_name : str
        Name of the class to be found.

    Returns
    -------
    subclass : None | class
        Class with name `subclass_name` which is a subclass of the `base_class`.
        None, if no subclass of the given name is found.
    """
    if base_class.__name__ == subclass_name:
        return base_class
    subclasses = base_class.__subclasses__()
    for subcls in subclasses:
        if subcls.__name__ == subclass_name:
            return subcls
    for subcls in subclasses:
        found = find_subclass(subcls, subclass_name)  # recursion
        if found is not None:
            return found
    return None


def get_recursive(nested_data, recursive_key, separator="/"):
    """Extract specific value from a nested data structure.

    Parameters
    ----------
    nested_data : dict of dict (-like)
        Some nested data structure supporting a dict-like interface.
    recursive_key : str
        The key(-parts) to be extracted, separated by `separator`.
        A leading `separator` is ignored.
    separator : str
        Separator for splitting `recursive_key` into subkeys.

    Returns
    -------
    entry :
        For example, ``recursive_key="/some/sub/key"`` will result in extracing
        ``nested_data["some"]["sub"]["key"]``.

    See also
    --------
    set_recursive : same for changing/setting a value.
    flatten : Get a completely flat structure.
    """
    if recursive_key.startswith(separator):
        recursive_key = recursive_key[len(separator):]
    if not recursive_key:
        return nested_data  # return the original data if recursive_key is just "/"
    for subkey in recursive_key.split(separator):
        nested_data = nested_data[subkey]
    return nested_data


def set_recursive(nested_data, recursive_key, value, separator="/", insert_dicts=False):
    """Same as :func:`get_recursive`, but set the data entry to `value`."""
    if recursive_key.startswith(separator):
        recursive_key = recursive_key[len(separator):]
    subkeys = recursive_key.split(separator)
    for subkey in subkeys[:-1]:
        if insert_dicts and subkey not in nested_data:
            nested_data[subkey]
        nested_data = nested_data[subkey]
    nested_data[subkeys[-1]] = value


def flatten(mapping, separator='/'):
    """Obtain a flat dictionary with all key/value pairs of a nested data structure.

    Parameters
    ----------
    separator : str
        Separator for merging keys to a single string.

    Returns
    -------
    flat_config : dict
        A single dictionary with all key-value pairs.

    Examples
    --------
    .. testsetup ::

        from tenpy.tools.misc import *

    >>> sample_data = {'some': {'nested': {'entry': 100, 'structure': 200},
    ...                         'subkey': 10},
    ...                'topentry': 1}
    >>> flat = flatten(sample_data)
    >>> for k in sorted(flat):
    ...     print(repr(k), ':', flat[k])
    'some/nested/entry' : 100
    'some/nested/structure' : 200
    'some/subkey' : 10
    'topentry' : 1


    See also
    --------
    get_recursive : Useful to obtain a single entry from a nested data structure.
    """
    if isinstance(mapping, Config):
        mapping = mapping.as_dict()
    result = {}  #mapping.copy()
    for k1, v1 in mapping.items():
        if isinstance(v1, dict):
            flat_submapping = flatten(v1, separator)
            for k2, v2 in flat_submapping.items():
                new_key = separator.join((k1, k2))
                result[new_key] = v2
        else:
            result[k1] = v1
    return result


def build_initial_state(size, states, filling, mode='random', seed=None):
    warnings.warn(
        "Deprecated `build_initial_state`: Use `tenpy.networks.mps.InitialStateBuilder` instead.",
        category=FutureWarning,
        stacklevel=2)
    from tenpy.networks import mps
    return mps.build_initial_state(size, states, filling, mode, seed)


def setup_executable(mod, run_defaults, identifier_list=None):
    """Read command line arguments and turn into useable dicts.

    .. warning ::

        this is a deprecated interface. Use the :class:`~tenpy.simulations.simulation.Simulation`
        interface in combination with :func:`~tenpy.

    Uses default values defined at:
    - model class for model_par
    - here for sim_par
    - executable file for run_par
    Alternatively, a model_defaults dictionary and identifier_list can be supplied without the model

    NB: for setup_executable to work with a model class, the model class needs to define two things:
            - defaults, a static (class level) dictionary with (key, value) pairs that have the name
              of the parameter (as string) as key, and the default value as value.
            - identifier, a static (class level) list or other iterable with the names of the parameters
              to be used in filename identifiers.

    Parameters
    ----------
    mod : model | dict
        Model class (or instance) OR a dictionary containing model defaults
    run_defaults : dict
        default values for executable file parameters
    identifier_list : ieterable
        Used only if mod is a dict. Contains the identifier variables

    Returns
    -------
    model_par, sim_par, run_par : dict
        containing all parameters.
    args :
        namespace with raw arguments for some backwards compatibility with executables.
    """
    warnings.warn("Deprecated: use `tenpy.run_simulation` and `tenpy.console_main` instead.",
                  category=FutureWarning,
                  stacklevel=2)
    parser = argparse.ArgumentParser()

    # These deal with backwards compatibility (supplying a model)
    if type(mod) != dict and identifier_list == None:  # Assume we've been given a model class
        try:
            model_defaults = mod.defaults
            identifier_list = mod.identifier
        except AttributeError as err:
            print("Cannot get model defaults and identifer list from mod. Is mod a class/instance?")
            print(err)
            raise AttributeError
    elif type(mod) == dict and hasattr(identifier_list, '__iter__'):
        model_defaults = mod
    else:
        raise ValueError("If model_par are supplied as dict, identifier_list should be provided.")

    # The model_par bit (for all model parameters)
    for label, value in model_defaults.items():
        if type(value) == bool:  # For boolean defaults, we want a true/false flag
            if value:
                parser.add_argument('-' + label, action='store_false')
            else:
                parser.add_argument('-' + label, action='store_true')
        else:  # For non-boolean defaults, take the type of the default as type for the cmdline var
            parser.add_argument('-' + label, type=type(value), default=value)

    # The run_par bit (for executable-level parameters). These are defined in the executable file
    # but need to be included for argparse to work correctly.
    for label, value in run_defaults.items():
        if type(value) == bool:  # For boolean defaults, we want a true/false flag
            if value:
                parser.add_argument('-' + label, action='store_false')
            else:
                parser.add_argument('-' + label, action='store_true')
        else:  # For non-boolean defaults, take the type of the default as type for the cmdline var
            print('Adding argument', label)
            parser.add_argument('-' + label, type=type(value), default=value)
    # The following parameters are run-time but so general they're defined here
    parser.add_argument('-ncores', type=int, default=1)
    parser.add_argument('-dir', type=str, default=None)
    parser.add_argument('-plots', action='store_true')  # Generic flag to activate plotting
    parser.add_argument('-seed', default=None)  # For anything random

    # The sim_par bit (for DMRG-related parameters). These don't vary, so we'll just define here.
    parser.add_argument('-chi', type=int, default=100)
    parser.add_argument('-dchi', type=int, default=20)  # Step size for chi ramp
    parser.add_argument('-dsweeps', type=int, default=20)  # Number of sweeps for chi step
    parser.add_argument('-min_sweeps', type=int, default=30)
    parser.add_argument('-max_sweeps', type=int, default=1000)
    #parser.add_argument('-n_steps', type=int, default=10)
    #parser.add_argument('-max_steps', type=int, default=2400)
    parser.add_argument('-mixer', action='store_true')  # To activate mixer
    parser.add_argument('-mix_str', type=float, default=1.e-3)
    parser.add_argument('-mix_dec', type=float, default=1.5)
    parser.add_argument('-mix_len', type=int, default=80)
    parser.add_argument('-start_env', type=int, default=0)
    parser.add_argument('-update_env', type=int)

    # Now parse and turn into manageable dicts.
    args = parser.parse_args()
    par_dict = vars(args)  # Turns args (='Namespace' object) into dict.

    model_par = {}
    for label in model_defaults.keys():  # Select the model-relevant parts of par_dict
        model_par[label] = par_dict[label]

    run_par = {}
    for label in run_defaults.keys():  # Select the executable-relevant parts of par_dict
        run_par[label] = par_dict[label]

    try:
        sim_par = {
            'chi_list': chi_list(args.chi, args.dchi, args.dsweeps),
            'N_sweeps_check': 10,
            'min_sweeps': args.min_sweeps,
            'max_sweeps': args.max_sweeps,
            'verbose': args.verbose,  # Take this from the model
            'lanczos_params': {
                'N_min': 2,
                'N_max': 40,
                'E_tol': 10**(-12)
            }
        }
    except AttributeError as err:
        print(
            'sim_par parsing has failed, most likely because model does not define verbose parameter.'
        )
        print(err)
        raise AttributeError
    if args.mixer:
        sim_par['mixer'] = True
        sim_par['mixer_params'] = {
            'amplitude': args.mix_str,
            'decay': args.mix_dec,
            'disable_after': args.mix_len
        }

    # Having set up all dictionaries, we can now do some other setting up
    omp_set_nthreads(args.ncores)
    if not args.dir == None:
        os.chdir(args.dir)
    import matplotlib
    matplotlib.rcParams["savefig.directory"] = os.chdir(os.getcwd())

    # Build the identifier based on model-defined and general parameters
    identifier = "chi_{}_seed_{}_".format(args.chi, args.seed)  # Only use seed if supplied?
    for varname in identifier_list:
        if 'conserve' in varname:
            shortened = varname.replace('conserve',
                                        'cons').replace('number',
                                                        'num').replace('charge',
                                                                       'ch').replace('spin', 'S')
            identifier += shortened + "_"
        elif model_par[varname] != 0:  # Parameters that are 0 are ignored. Only want supplied?
            identifier += varname + "_" + str(model_par[varname]) + "_"
    if args.mixer:
        identifier += 'mix_({},{},{})'.format(args.mix_str, args.mix_dec, args.mix_len)
    if identifier[-1] == "_":
        identifier = identifier[:-1]
    # Attempt to shorten the identifier
    identifier = identifier.replace('periodic', 'inf').replace('finite', 'fin').replace('.0_', '_')
    if len(identifier) >= 144:
        print("Warning: identifier has a lenght longer than max filename on encrypted Ubuntu!")

    run_par.update({
        'ncores': args.ncores,
        'dir': args.dir,
        'plots': args.plots,
        'identifier': identifier,
        'seed': args.seed,
    })

    return model_par, sim_par, run_par, args
