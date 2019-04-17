"""tools to fit to an algebraic decay"""
# Copyright 2018 TeNPy Developers

import numpy as np
import scipy.optimize as optimize

__all__ = [
    'alg_decay', 'linear_fit', 'lin_fit_res', 'alg_decay_fit_res', 'alg_decay_fit',
    'alg_decay_fits', 'plot_alg_decay_fit'
]


def alg_decay(x, a, b, c):
    """define the algebraic decay"""
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
