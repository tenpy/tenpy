"""Differnt math functions needed at some point in the library.

.. todo :
    clean up, document, move to tools/misc, tools/fit
"""
import numpy as np
import scipy.sparse as sparse
import scipy.optimize as optimize
import warnings

try:
    from scipy.sparse.linalg import eigs as sparse_eigen
except ImportError:
    pass

int_I3 = np.eye(3, dtype=int)
LeviCivita3 = np.array([[np.cross(b, a) for a in int_I3] for b in int_I3])

##########################################################################
##########################################################################
# Random useful stuff
# TODO: this has nothing to do with `math`. Move this to tools/misc.py


def tonparray(o, array_size=None):
    """ Convert to an np.ndarray
            If array_size is set, then tile the array to the correct size.
            """
    try:
        iter(o)
    except TypeError:
        a = np.array([o])
    else:
        a = np.array(o)
    if array_size is not None:
        if array_size % len(a) != 0:
            raise ValueError("incomensurate len")
        a = np.tile(a, array_size // len(a))
    return a


def list_to_dict_list(l):
    """ Given a list (l) of objects, construct a lookup table.
            This function will handle duplicate entries in l.

            Parameters:
                    l: a list of objects that can be converted to tuples (keys for dictionary).

            Returns:
                    A dictionary with key:value pair (key):[i1,i2,...]
                    where i1, i2, ... are the places where key is found in l.
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
    """ Given a list of lists, turn it to a 2D array (pad with 0), or turn a 1D list to 2D.
                    Returns a numpy.ndarray.

                    Examples:
                    >>> atleast_2d_pad([3,4,0])
                            array([[3, 4, 0]])
                    >>> atleast_2d_pad([[3,4],[1,6,7]])
                            array([[ 3.,  4.,  0.],
                                   [ 1.,  6.,  7.]])
            """
    iter(a)
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
    if is_list_of_list:
        a2 = a
    else:
        try:
            return np.array([a])
        except ValueError:
            return [a]
    maxlen = max([len(s) for s in a2])
    # Pad if necessary
    for r in xrange(len(a2)):
        s = a2[r]
        a2[r] = np.hstack([s, [pad_item] * (maxlen - len(s))])
    return np.array(a2)


def transpose_list_list(D, pad=None):
    """ Returns a list of lists T, such that T[i][j] = D[j][i]
            If D is not rectangular, make it rectangular padding with pad
            """
    nRow = len(D)
    if nRow == 0:
        return [[]]
    c = [len(R) for R in D]
    nCol = max(c)
    T = [[pad] * nRow for i in range(nCol)]
    for j, R in enumerate(D):
        for i, e in enumerate(R):
            T[i][j] = e
    return T


def matvec_to_array(H):
    """ Given an object with a matvec (taking an np.array), return its corresponding dense matrix.
            """
    dim = H.dim
    X = np.zeros((dim, dim), H.dtype)
    v = np.zeros((dim), H.dtype)
    for i in range(dim):
        v[i] = 1
        X[i] = H.matvec(v)
        v[i] = 0
    return X



def zero_if_close(a, tol=1e-15):
    if a.dtype == np.complex128 or a.dtype == np.complex64:
        cr = np.array(np.abs(a.real) < tol, int)
        ci = np.array(np.abs(a.imag) < tol, int)
        ar = np.choose(cr, [a.real, np.zeros(a.shape)])
        ai = np.choose(ci, [a.imag, np.zeros(a.shape)])
        return ar + 1j * ai
    else:
        c = np.array(np.abs(a) < tol, int)
        return np.choose(c, [a, np.zeros_like(a)])


def pad(a, w_l=0, v_l=0, w_r=0, v_r=0, axis=0):
    """ Pad an array along 'axis.' w_l is the width of the pad added before index 0, w_r after last index, with values v_l, v_r.

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


def promote_types(*arg):
    """Example:
    >>> promote_types(np.float, np.complex64)
    dtype('complex128')
    """
    if len(arg) == 0:
        raise ValueError
    if len(arg) == 1:
        return arg[0]
    if len(arg) >= 2:
        t = np.promote_types(arg[0], arg[1])
        for a in arg[2:]:
            t = np.promote_types(t, a)
        return t


def assert_np_identical(a, b, name=''):
    """Assert that a, b are identical, otherwise print an error message and raise an error."""
    if np.all(a == b):
        return
    else:
        print "{} mismatch:  {}  !=  {}".format(name, a, b)
        raise ValueError, "{} mismatch".format(name)


def assert_np_match(a, b, tol=1e-15, name=''):
    """Assert that a, b are the same up to some tolerance, otherwise print an error message and raise an error."""
    try:
        err = np.max(np.abs(a - b))
    except:
        print "{} mismatch:  {}  and  {} are incompatible.".format(name, a, b)
        raise ValueError
    if err > tol:
        print "{} mismatch:  {}  -  {}  >  {}".format(name, a, b, tol)
        raise ValueError, "{} not within tolerance {}".format(name, tol)

##########################################################################
##########################################################################
# Actual Math functions


def gcd(a, b):
    """Computes the greatest common divisor (GCD) of two numbers.  Retrun 0 if both a,b are zero, otherwise always return a non-negative number."""
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def gcd_array(a):
    a = np.array(a).reshape(-1)
    if len(a) <= 0:
        raise ValueError
    t = a[0]
    for x in a[1:]:
        if t == 1:
            break
        t = gcd(t, x)
    return t


def lcm(a, b):
    """Returns the least common multiple (LCM) of two positive numbers."""
    a0, b0 = a, b
    while b:
        a, b = b, a % b
    return a0 * (b0 // a)


def speigs(A, k, *args, **kwargs):
    """Wrapper around scipy.sparse.linalg.eigs, lifting the restriction ``k < rank(A)-1``.

    Parameters
    ----------
    A : MxM ndarray or like scipy.sparse.linalg.LinearOperator
        the (square) linear operator for which the eigenvalues should be computed.
    k : int
        the number of eigenvalues to be computed.
    *args, **kwargs :
        further arguments are directly given to ``scipy.sparse.linalg.eigs``

    Returns
    -------
    w : ndarray
        array of min(`k`, A.shape[0]) eigenvalues
    v : ndarray
        array of min(`k`, A.shape[0]) eigenvectors, ``v[:, i]`` is the `i`th eigenvector.
        Only returned if ``kwargs['return_eigenvectors'] == True``.
    """
    d = A.shape[0]
    if A.shape != (d, d):
        raise ValueError("A.shape not a square matrix: " + str(A.shape))
    if k < d:
        return sparse_eigen(A, k, *args, **kwargs)
    else:
        if k > d:
            warnings.warn("trimming k={k:d} to d={d:d}".format(k=k, d=d))
        if isinstance(A, np.ndarray):
            Amat = A
        else:
            Amat = matvec_to_array(A)  # Constructs the matrix
        ret_eigv = kwargs.get('return_eigenvectors', args[7] if len(args) > 7 else True)
        if ret_eigv:
            return np.linalg.eig(Amat)
        else:
            return np.linalg.eigvals(Amat)


def perm_sign(p):
    """ Given a permutation of numbers, returns its sign. (+1 or -1)
            Assumes that all the elements are distinct, if not, you get crap.

            Example:
            >>> print '\n'.join(['  %s: %s' % (u, perm_sign(u)) for u in itertools.permutations(range(4))])
            """
    rp = np.argsort(p)
    p = np.argsort(rp)
    s = 1
    for i, v in enumerate(p):
        if i == v:
            continue
        # by the way we loop, i < v, so we find where i is.
        # loci = rp[i]
        # we don't have to do p[i] = p[locv] becasue we never need p[i] again
        #p[i] = p[rp[i]]
        p[rp[i]] = v
        rp[v] = rp[i]
        s = -s
    return s

##########################################################################
# Stuff for fitting to an algebraic decay


def alg_decay(x, a, b, c):
    return a * x**(-b) + c


def linear_fit(x, y):
    """Perform a linear fit of y to ax + b.
    Returns a, b, res."""
    assert x.ndim == 1 and y.ndim == 1
    fit = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y)
    return fit[0][0], fit[0][1], fit[1][0]


def lin_fit_res(x, y):
    """Returns the least-square residue of a linear fit y vs. x."""
    assert x.ndim == 1 and y.ndim == 1
    fit = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y)
    if len(fit[1]) < 1:
        # print 'lin_fit_res', x, y
        # print fit[1], type(fit)
        return np.max(y) - np.min(y)
    return fit[1][0]


def alg_decay_fit_res(log_b, x, y):
    """Returns the residue of an algebraic decay fit of the form x ** (-np.exp(log_b))."""
    return lin_fit_res(x**(-np.exp(log_b)), y)


def alg_decay_fit(x, y, npts=5, power_range=(0.01, 4.), power_mesh=[60, 10]):
    """Fit y to the form a * x**(-b) + c.
    Returns a triplet [a, b, c].

    npts specifies the maximum number of points to fit.  If npts < len(x), then alg_decay_fit() will only fit to the last npts points.
    power_range is a tuple that gives that restricts the possible ranges for b.
    power_mesh is a list of numbers, which specifies how fine to search for the optimal b.
            E.g., if power_mesh = [60,10], then it'll first divide the power_range into 60 intervals, and then divide those intervals by 10.
    """
    x = np.array(x)
    y = np.array(y)
    assert x.ndim == 1 and y.ndim == 1
    assert len(x) == len(y)
    if npts < 3:
        raise ValueError
    if len(x) > npts:
        x = x[-npts:]
        y = y[-npts:]
    global_log_power_range = (np.log(power_range[0]), np.log(power_range[1]))
    log_power_range = global_log_power_range
    for i in range(len(power_mesh)):
        # number of points inclusive
        brute_Ns = (power_mesh[i] if i == 0 else 2 * power_mesh[i]) + 1
        log_power_step = (log_power_range[1] - log_power_range[0]) / float(brute_Ns - 1)
        # print 'iteration', i, ":", np.exp(np.arange(brute_Ns) *
        # log_power_step + log_power_range[0])
        brute_fit = optimize.brute(
            alg_decay_fit_res, [log_power_range], (x, y), Ns=brute_Ns, finish=None)
        if brute_fit <= global_log_power_range[0] + 1e-6:
            return [0., 0., y[-1]]  # shit happened
        log_power_range = (brute_fit - log_power_step, brute_fit + log_power_step)
        # print "-->", np.exp(brute_fit), "new range", np.exp(log_power_range)
    l_fit = linear_fit(x**(-np.exp(brute_fit)), y)
    return [l_fit[0], np.exp(brute_fit), l_fit[1]]


def alg_decay_fits(x, ys, npts=5, power_range=(0.01, 4.), power_mesh=[60, 10]):
    """Fit arrays of y's to the form a * x**(-b) + c.
    Returns arrays of [a, b, c]."""
    x = np.array(x)
    if x.ndim != 1:
        raise ValueError
    ys = np.array(ys)
    y_shape = ys.shape
    assert y_shape[-1] == len(x)
    abc_flat = np.array([alg_decay_fit(
        x, yyy, npts=npts, power_range=power_range, power_mesh=power_mesh)
                         for yyy in ys.reshape(-1, len(x))])
    return abc_flat.reshape(y_shape[:-1] + (3, ))


def plot_alg_decay_fit(plot_module, x, y, fit_par, xfunc=None, kwargs={}, plot_fit_args={}):
    """Given x, y, and fit_par (output from alg_decay_fit), produces a plot of the algebraic decay fit.

    plot_module is matplotlib.pyplot, or a subplot.
    x, y are the data (real, 1-dimensional np.ndarray)
    fit_par is a triplet of numbers [a, b, c] that describes and algebraic decay (see alg_decay()).
    xfunc is an optional parameter that scales the x-axis in the resulting plot.
    kwargs is a dictionary, whoses key/items are passed to the plot function.
    plot_fit_args is a dictionary that controls how the fit is shown."""
    if xfunc is None:
        xfunc = lambda x: x
    if plot_fit_args.get('show_data_points', True):
        plot_module.plot(xfunc(x), y, 'o', **kwargs)
    n_interp = plot_fit_args.get('n_interp', 30)
    if len(x) > 1:
        interp_x = np.arange(-0.03, 1.1, 1. / n_interp) * \
            (np.max(x) - np.min(x)) + np.min(x)
        if plot_fit_args.get('show_fit', True):
            plot_module.plot(xfunc(interp_x), alg_decay(interp_x, *fit_par), '-', **kwargs)
        extrap_xrange = np.array([x[-2], np.max(interp_x)])
        if 'extrap_line_start' in plot_fit_args:
            try:
                extrap_xrange[0] = x[plot_fit_args['extrap_line_start']]
            except IndexError:
                if plot_fit_args['extrap_line_start'] >= len(x):
                    extrap_xrange[0] = np.max(interp_x)
                if plot_fit_args['extrap_line_start'] < -len(x):
                    extrap_xrange[0] = np.min(interp_x)
        if 'extrap_line_end' in plot_fit_args and plot_fit_args['extrap_line_end'] < len(x):
            try:
                extrap_xrange[1] = x[plot_fit_args['extrap_line_end']]
            except IndexError:
                extrap_xrange[1] = extrap_xrange[0]
        if extrap_xrange[0] < extrap_xrange[1]:
            plot_module.plot(xfunc(extrap_xrange), [fit_par[2], fit_par[2]], '--', **kwargs)
