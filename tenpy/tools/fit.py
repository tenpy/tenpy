"""tools to fit to an algebraic decay."""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np
import scipy.optimize as optimize

__all__ = [
    'alg_decay', 'linear_fit', 'lin_fit_res', 'alg_decay_fit_res', 'alg_decay_fit',
    'alg_decay_fits', 'plot_alg_decay_fit', 'fit_with_sum_of_exp', 'sum_of_exp',
    'entropy_profile_from_CFT', 'central_charge_from_S_profile'
]


def alg_decay(x, a, b, c):
    """Algebraic decay function :math :`a * x^{-b} + c`."""
    return a * x**(-b) + c


def linear_fit(x, y):
    """Perform a linear fit `y` ~ `a * x + b`.

    Parameters
    ----------
    x : array_like [M]
        The independent variable where the data is measured.
    y : array_like [M]
        The dependent data.

    Returns
    -------
    a : float
        The "slope" parameter of the fit function.
    b : float
        They "y-intercept" parameter of the fit function.
    res : float
        The (squared) residue, i.e. ``sum((y - (a * x + b)) ** 2)``.
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
    """Fit `y` to an algebraic decay of the form :math :`a * x^{-b} + c`.

    The exponent ``b`` is first determined via a brute-force search with a fixed search grid
    which is refined in multiple steps.
    Then, ``a`` and ``c`` are determined via least-squares linear fit with independent variable :math:`x^{-b}`.

    Parameters
    ----------
    x : array_like [M]
        The independent variable where the data is measured.
    y : array_like [M]
        The dependent data.
    npts : int
        The maximum number of points used for the fit.
        If ``npts < len(x)``, only the last `npts` datapoints, i.e. ``x[-npts:]`` and ``y[-npts:]`` are used.
    power_range : tuple(float, float)
        A range that restricts the possible values of the fit exponent ``b``
    power_mesh : list of float
        Number of points in the search grid for the fit exponent ``b``.
        The ``power_range`` is first divided into ``power_mesh[0]`` many intervals.
        Then, for each subsequent entry of ``power_mesh`` the smaller region around the best
        previous guess is further divided into as many intervals.

    Returns
    -------
    a : float
        The prefactor of the fitted algebraic decay.
    b : float
        The (negative) exponent of the fitted algebraic decay.
    c : float
        The y-intercept of the fitted algebraic decay.

    See Also
    --------
    alg_decay_fits
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
    """Batched version of :func:`~tenpy.tools.fit.alg_decay`.

    Parameters
    ----------
    x : array_like [M]
        The independent variable where the data is measured.
    y : array_like [M, N]
        ``N`` distinct sets of data for the dependent variable. ``N`` separate fits will be performed.
    *args :
        Same as for :func:`~tenpy.tools.fit.alg_decay`.

    Returns
    -------
    a : array [N]
        The prefactors of each fitted algebraic decay.
    b : array [N]
        The (negative) exponents of each fitted algebraic decay.
    c : array [N]
        The y-intercepts of each fitted algebraic decay.

    See Also
    --------
    alg_decay_fit
    """
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
    """Utility function used the plot an algebraic fit function next to the data.

    Parameters
    ----------
    plot_module
        This is either the module ``matplotlib.pyplot`` or an instance of ``matplotlib.pyplot.subplot``.
    x, y : array_like [M]
        The (real-valued) data.
    fit_par : tuple(float, float, float)
        The fit parameters ``(a, b, c)``, e.g. as returned by :func:`~tenpy.tools.fit.alg_decay`.
    xfunc : callable, optional
        If given, this function is used to scale the x-axis of the plot.
    kwargs : dict
        Keyword arguments that are passed to the ``plot_module.plot`` function.
    plot_fit_args : dict
        A dictionary that controls how the fit is shown via the following key value pairs::

        =================== ====== ========= =======================================================================
        key                 type   default   description
        =================== ====== ========= =======================================================================
        show_data_points    bool   True      If the datapoint `x`, `y` should be plotted.
        ------------------- ------ --------- -----------------------------------------------------------------------
        n_interp            int    30        The number of points to plot for the fit.
        ------------------- ------ --------- -----------------------------------------------------------------------
        show_fit            bool   True      If the fit should be plotted.
        ------------------- ------ --------- -----------------------------------------------------------------------
        extrap_line_start   int    -2        Define the start of the extrapolation line as ``x[extrap_line_start]``.
        ------------------- ------ --------- -----------------------------------------------------------------------
        extrap_line_end     int    ...       Define the end of the extrapolation as ``x[extrap_line_end]``.
                                             Per default, it ends at the end of the x-axis.
        =================== ====== ========= =======================================================================
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
    This function allows to do that.

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
