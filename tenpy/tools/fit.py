"""tools to fit to an algebraic decay."""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np
import scipy.optimize as optimize

__all__ = [
    'alg_decay', 'linear_fit', 'lin_fit_res', 'alg_decay_fit_res', 'alg_decay_fit',
    'alg_decay_fits', 'plot_alg_decay_fit', 'fit_with_sum_of_exp', 'sum_of_exp',
    'entropy_profile_from_CFT', 'central_charge_from_S_profile'
]


def alg_decay(x, a, b, c):
    """define the algebraic decay."""
    return a * x**(-b) + c


def linear_fit(x, y):
    """Perform a linear fit of y to ax + b.

    Returns a, b, res.
    """
    assert x.ndim == 1 and y.ndim == 1
    fit = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)
    return fit[0][0], fit[0][1], fit[1][0]


def lin_fit_res(x, y):
    """Returns the least-square residue of a linear fit y vs x."""
    assert x.ndim == 1 and y.ndim == 1
    fit = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y)
    if len(fit[1]) < 1:
        return np.max(y) - np.min(y)
    return fit[1][0]


def alg_decay_fit_res(log_b, x, y):
    """Returns the residue of an algebraic decay fit of the form ``x**(-np.exp(log_b))``."""
    return lin_fit_res(x**(-np.exp(log_b)), y)


def alg_decay_fit(x, y, npts=5, power_range=(0.01, 4.), power_mesh=[60, 10]):
    """Fit y to the form ``a*x**(-b) + c``.

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
        brute_fit = optimize.brute(alg_decay_fit_res, [log_power_range], (x, y),
                                   Ns=brute_Ns,
                                   finish=None)
        if brute_fit <= global_log_power_range[0] + 1e-6:
            return [0., 0., y[-1]]  # shit happened
        log_power_range = (brute_fit - log_power_step, brute_fit + log_power_step)
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
    abc_flat = np.array([
        alg_decay_fit(x, yyy, npts=npts, power_range=power_range, power_mesh=power_mesh)
        for yyy in ys.reshape(-1, len(x))
    ])
    return abc_flat.reshape(y_shape[:-1] + (3, ))


def plot_alg_decay_fit(plot_module, x, y, fit_par, xfunc=None, kwargs={}, plot_fit_args={}):
    """Given x, y, and fit_par (output from alg_decay_fit), produces a plot of the algebraic decay
    fit.

    plot_module is matplotlib.pyplot, or a subplot. x, y are the data (real, 1-dimensional
    np.ndarray) fit_par is a triplet of numbers [a, b, c] that describes and algebraic decay (see
    alg_decay()). xfunc is an optional parameter that scales the x-axis in the resulting plot.
    kwargs is a dictionary, whoses key/items are passed to the plot function. plot_fit_args is a
    dictionary that controls how the fit is shown.
    """
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


def fit_with_sum_of_exp(f, n, N=50):
    r"""Approximate a decaying function `f` with a sum of exponentials.

    MPOs can naturally represent long-range interactions with an exponential decay.
    A common technique for other (e.g. powerlaw) long-range interactions is to approximate them
    by sums of exponentials and to include them into the MPOs.
    This funciton allows to do that.

    The algorithm/implementation follows the appendix of :cite:`murg2010`.

    Parameters
    ----------
    f : function
        Decaying function to be approximated. Needs to accept a 1D numpy array `x`
    n : int
        Number of exponentials to be used.
    N : int
        Number of points at which to evaluate/fit `f`;
        we evaluate and fit `f` at the points ``x = np.arange(1, N+1)``.

    Returns
    -------
    lambdas, prefactors: 1D arrays
        Such that :math:`f(k) \approx \sum_i x_i \lambda_i^k` for (integer) 1 <= `k` <= `N`.
        The function :func:`sum_of_exp` evaluates this for given `x`.
    """
    assert n < N
    x = np.arange(1, N + 1)
    f_x = f(x)
    F = np.zeros([N - n + 1, n])
    for i in range(n):
        F[:, i] = f_x[i:i + N - n + 1]

    U, V = np.linalg.qr(F)
    U1 = U[:-1, :]
    U2 = U[1:, :]
    M = np.dot(np.linalg.pinv(U1), U2)
    lam = np.linalg.eigvals(M)
    lam = np.sort(lam)[::-1]
    # least-square fit
    W = np.power.outer(lam, x).T
    pref, res, rank, s = np.linalg.lstsq(W, f_x, None)
    return lam, pref


def sum_of_exp(lambdas, prefactors, x):
    """Evaluate ``sum_i prefactor[i] * lambda[i]**x`` for different `x`.

    See :func:`fit_sum_of_exp` for more details.
    """
    return np.real_if_close(np.dot(np.power.outer(lambdas, x).T, prefactors))


def entropy_profile_from_CFT(size_A, L, central_charge, const):
    r"""Expected profile for the entanglement entropy at a critical point.

    Conformal field theory predicts the entanglement entropy for cutting
    a ground state of a finite, critical (i.e. gapless) system of length `L`
    into the left `l` and right `L-l` sites to be (eq. 2 of :cite:`calabrese2004`):

    .. math ::

        S(l, L) = \frac{c}{6} \log\left(\frac{2L}{\pi a} \sin\left(\frac{\pi l}{L}\right)\right)
                 + \textrm{const}

    Here, `c` is the central charge of the system, and `a` is the lattice spacing, which we set to
    1, and `const` is a non-universal constant.

    Returns exactly that formula.
    """
    return central_charge / 6 * np.log(2 * L / np.pi * np.sin(np.pi * size_A / L)) + const


def central_charge_from_S_profile(psi, exclude=None):
    """Fit the entanglement entropy of a finite MPS to the expected profile for critical models.

    See :func:`entropy_profile_from_CFT` for the function we fit to.

    Parameters
    ----------
    psi : :class:`~tenpy.networks.mps.MPS`
        Ground state of a *finite* system at a critical point (i.e. gapless!).
        The bond dimension should be large enough to be converged!
    exclude : int
        How many sites at the left (and at the right) boundary to exclude from the fit
        (to avoid boundary effects). Defaults to ``psi.L // 4``

    Returns
    -------
    central_charge, const : float
        Central charge and constant offset as in :func:`entropy_profile_from_CFT`.
    res : float
        Residuum of the error.
    """
    if not psi.bc == 'finite':
        raise ValueError("works only for finite MPS at a critical point")
    L = psi.L
    if exclude is None:
        exclude = L // 4
    if 2 * exclude >= L - 8:
        raise ValueError("Not enough points for a reasonable fit left")
    S = psi.entanglement_entropy()
    size_A = np.arange(1, psi.L)[exclude:-exclude]
    expected = entropy_profile_from_CFT(size_A, L, 1., 0.)
    # fit S ~=~ central_charge * expected + const
    c, const, res = linear_fit(expected, S[exclude:-exclude])
    return c, const, res
