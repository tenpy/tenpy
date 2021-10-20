"""Miscellaneous tools, somewhat random mix yet often helpful."""
# Copyright 2018-2020 TeNPy Developers, GNU GPLv3

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


def to_array(a, shape=(None, )):
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

    Returns
    -------
    a_array : ndarray
        A copy of `a` converted to a numpy ndarray of desired dimension and shape.
    """
    a = np.array(a)  # copy
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
    
def qramp_list(nqramp, last_qramp, qramp_op, qramp_site, qramp_move_right):
    """return a dictionary of sweep indices with entries corresponding to qramp_list 
    in :class:`~tenpy.algorithms.dmrg.DMRGEngine`
    Parameters
    ----------
    nqramp : int
        Number of operators to be inserted in total
    last_qrampp : int
        The largest sweep index where an insertion occurs
    qramp_op : str, npc.array
        Description of the operator to insert, appropriate for the chosen DMRG algorithm
    qramp_site : int
    	Site of MPS where operators should be inserted
    qramp_move_right : True | False
    	Direction of movement during DMRG when operators should be inserted

    Returns
    -------
    dictionary: {nsweep1: [i0, move_right, custom_op], nsweep2: [...]}
    """
    if (last_qramp<nqramp):
    	raise ValueError("Error: multiple operator insertions per sweep not currently supported.")
    if qramp_site is None:
    	qramp_site = 0
    if nqramp == 0:
    	return {}
    stride = int(last_qramp)//int(nqramp)
    nlarger=last_qramp-stride*nqramp # number of steps with size (stride +1)
    qramp_list = {}
    for i in range (nqramp - nlarger):
    	sweep = (i+1)*stride
    	if (qramp_op is not None):
    		qramp_list[sweep] = [qramp_site, qramp_move_right, qramp_op]
    	else:
            qramp_list[sweep] = [qramp_site, qramp_move_right]
    offset = stride*(nqramp - nlarger)
    for i in range (nlarger):
    	sweep = offset + (i+1)*(stride+1)
    	if (qramp_op is not None):
    	    qramp_list[sweep] = [qramp_site, qramp_move_right, qramp_op]
    	else:
    		qramp_list[sweep] = [qramp_site, qramp_move_right]
    print ("qramp_list=",qramp_list)
    return qramp_list

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


def build_initial_state(size, states, filling, mode='random', seed=None):
    warnings.warn(
        "Deprecated: moved `build_initial_state` to `tenpy.networks.mps.build_initial_state`.",
        category=FutureWarning,
        stacklevel=2)
    from tenpy.networks import mps
    return mps.build_initial_state(size, states, filling, mode, seed)


def setup_executable(mod, run_defaults, identifier_list=None, only_list_supplied=False):
    """Read command line arguments and turn into useable dicts.

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

    Args:
        mod (model | dict): Model class (or instance) OR a dictionary containing model defaults
        run_defaults (dict): default values for executable file parameters
        identifier_list (iterable, optional) | Used only if mod is a dict. Contains the identifier
                                                                                    variables

    Returns:
        model_par, sim_par, run_par (dicts) : containing all parameters.
        args | namespace with raw arguments for some backwards compatibility with executables.
    """
    warnings.warn(
        "Attention: `setup_executable` was developed for a previous version of tenpy and not all options may be operational.",
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
    if not 'active_sites' in run_defaults:
        parser.add_argument('-active_sites', type=int, default=2)
    parser.add_argument('-chi', type=int, default=100)
    parser.add_argument('-dchi', type=int, default=20)  # Step size for chi ramp
    parser.add_argument('-dsweeps', type=int, default=20)  # Number of sweeps for chi step
    parser.add_argument('-nqramp', type=int, default=0) # Number of insertions of qramp operator
    parser.add_argument('-last_qramp', type=int, default=50) # Last sweep at which an operator is inserted
    parser.add_argument('-qramp_op', type=str, default=None) # Operator to insert during ramp events
    parser.add_argument('-qramp_site', type=str, default='0', help='The site index where ramp operators should be inserted, or "R" for random sites') # Site on which to insert the ramp operator
    parser.add_argument('-qramp_move_left', action='store_true') # flag indicating the operators should be inserted while moving left
    if run_defaults.get('min_sweeps') is None:
        parser.add_argument('-min_sweeps', type=int, default=30)
    if run_defaults.get('max_sweeps') is None:
        parser.add_argument('-max_sweeps', type=int, default=1000)
    if run_defaults.get('N_sweeps_check') is None:
        parser.add_argument('-N_sweeps_check', type=int, default=10)
    
    # inputs for mixer parameters, as per https://tenpy.readthedocs.io/en/latest/reference/tenpy.algorithms.dmrg.Mixer.html#cfg-config-Mixer
    parser.add_argument('-mixer', action='store_true')  # To activate mixer
    parser.add_argument('-mix_str', type=float, default=1.e-3)
    parser.add_argument('-mix_dec', type=float, default=1.5)
    parser.add_argument('-mix_len', type=int, default=80)
            
    # control of tolerances:
    if run_defaults.get('max_E_err') is None:
        parser.add_argument('-max_E_err', type=float, default=1e-8) # DMRG Error tolerance
    if run_defaults.get('max_S_err') is None:
        parser.add_argument('-max_S_err', type=float, default=1e-5) # DMRG Entanglement tolerance

    # DMRG norm tolerance: https://tenpy.readthedocs.io/en/latest/reference/tenpy.algorithms.dmrg.DMRGEngine.html#cfg-option-DMRGEngine.norm_tol
    if not 'norm_tol' in run_defaults:        parser.add_argument('-norm_tol', type=float, default=1e-5) # After the DMRG run, update the environment with at most `norm_tol_iter` sweeps until ``np.linalg.norm(psi.norm_err()) < norm_tol``.
    if not 'norm_tol_iter' in run_defaults:
        parser.add_argument('-norm_tol_iter', type=float, default=5.0)
	#Perform at most `norm_tol_iter`*`update_env` sweeps to converge the norm error below `norm_tol`
    
    # parameters controlling sweeps to reconstruct the environment
    if run_defaults.get('start_env') is None:
        parser.add_argument('-start_env', type=int, default=0)
    if run_defaults.get('update_env') is None:
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
        from ..algorithms import dmrg
        sim_par = {
            'active_sites': args.active_sites,
            'chi_list': dmrg.chi_list(args.chi, args.dchi, args.dsweeps),
            'qramp_op': args.qramp_op,
            'qramp_list': qramp_list(args.nqramp, args.last_qramp, args.qramp_op, args.qramp_site, not args.qramp_move_left),
            'min_sweeps': args.min_sweeps,
            'max_sweeps': args.max_sweeps,
            'N_sweeps_check' : args.N_sweeps_check,
            'start_env': args.start_env,
            'max_E_err' : args.max_E_err,
            'max_S_err' : args.max_S_err,
            'norm_tol' : args.norm_tol,
            'norm_tol_iter' : args.norm_tol_iter,
            'verbose': args.verbose,  # Take this from the model
            'lanczos_params': {
                'N_min': 2,
                'N_max': 40,
                'E_tol': 10**(-12)
            }
         }
        if (args.update_env is None):
            sim_par['update_env'] = args.N_sweeps_check // 2
        else:
            sim_par['update_env'] = args.update_env
			
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
        else:
            if (only_list_supplied):
                if model_par[varname] != model_defaults[varname]:
                    identifier += varname + "_" + str(model_par[varname]) + "_"
            else:
                if model_par[varname] != 0:  # Parameters that are 0 are ignored. Only want supplied?
                    identifier += varname + "_" + str(model_par[varname]) + "_"
    if args.mixer:
        identifier += 'mix_({},{},{})'.format(args.mix_str, args.mix_dec, args.mix_len)
    if identifier[-1] == "_":
        identifier = identifier[:-1]
    # Attempt to shorten the identifier
    identifier = identifier.replace('periodic', 'inf').replace('finite', 'fin').replace('.0_', '_')
    identifier = identifier.replace('flux_p', 'p').replace('flux_q', 'q').replace('phi_ext_mode','pe-mode')
    if len(identifier) >= 144:
        print("Warning: identifier has a length longer than max filename on encrypted Ubuntu! Try argument 'only_list_supplied'")

    run_par.update({
        'ncores': args.ncores,
        'dir': args.dir,
        'plots': args.plots,
        'identifier': identifier,
        'seed': args.seed,
    })

    return model_par, sim_par, run_par, args
