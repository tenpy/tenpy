"""Tools around spectral functions.

This includes (spatial) fourier transforms on a :class:`~tenpy.models.lattice.Lattice`,
a :func:`spectral_function` to compute the spectral function from time dependent correlations
on a lattice (which incorporates linear prediction and gaussian windowing directly).
Those functions are used for the classes :class:`~tenpy.simulations.time_evolution.TimeDependentCorrelation`
and :class:`~tenpy.simulations.time_evolution.SpectralFunction`.
However, they can also be used in a standalone way on available results (i.e. in an interactive ipython or
jupyter notebook session).
"""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np

from .prediction import linear_prediction

__all__ = [
    'spectral_function', 'fourier_transform_space', 'fourier_transform_time', 'to_mps_geometry',
    'apply_gaussian_windowing', 'plot_correlations_on_lattice'
]


def spectral_function(time_dep_corr,
                      lat,
                      dt: float,
                      gaussian_window: bool = False,
                      sigma: float = 0.4,
                      linear_predict: bool = False,
                      rel_prediction_time: float = 1,
                      rel_num_points: float = 0.3,
                      truncation_mode: str = 'renormalize',
                      rel_split: float = 0,
                      axis_time: int = 0,
                      axis_space: int = 1):
    r"""Given a time dependent correlation function C(t, r), calculate its
    Spectral Function.

    After a run of :class:`~tenpy.simulations.time_evolution.TimeDependentCorrelation`, a :class:`DataLoader` instance
    should be passed, from which the underlying lattice and additional parameters (e.g. ``dt``) can be extracted.
    The `correlation_key` must coincide with the key of the time-dep. correlation function in the output of the
    Simulation.

    Parameters
    ----------
    time_dep_corr : np.ndarray
        Time dependent correlation :math`C(t, r)`
    lat : :class:`~tenpy.models.lattice.Lattice`
        instance of a lattice
    dt : float
        time-step discretization of the ``t_dep_correlation``
    gaussian_window : bool
        boolean flag to apply gaussian windowing
    sigma : float
        standard-deviation used for the gaussian window
    linear_predict : bool
        boolean flag to apply linear prediction
    rel_prediction_time : float
        relative time to predict, defaults to 1
    rel_num_points : float
        relative percentage of last points to base linear prediction on
    truncation_mode : str
        truncation_mode of :func:`~tenpy.tools.prediction.get_alpha_and_c`
    rel_split : float
        percentage of the data to be discarded during training
    axis_time :
        time axis (default 0)
    axis_space :
        axis of mps tensors (default 1)

    Returns
    -------
    dict:
        dictionary of keys for `k`, `k_reduced`, `w` and for the spectral function `S`

    Notes
    -----
    The Spectral Function is given by the fourier transform in space and time of the (time-dep.) correlation function.
    For a e.g. translationally invariant system, this is
    .. math ::

        S(w, \mathbf{k}) = \int dt e^{-iwt} \int d\mathbf{r} e^{i \mathbf{k} \mathbf{r} C(t, \mathbf{r})
    """
    # first we fourier transform in space C(r, t) -> C(k, t)
    ft_space, k = fourier_transform_space(lat, time_dep_corr, axis=axis_space)
    k_reduced = lat.BZ.reduce_points(k)
    # optional linear prediction
    if linear_predict is True:
        ft_space = linear_prediction(ft_space,
                                     rel_prediction_time=rel_prediction_time,
                                     rel_num_points=rel_num_points,
                                     axis=axis_time,
                                     truncation_mode=truncation_mode,
                                     rel_split=rel_split)
    # optional gaussian windowing
    if gaussian_window is True:
        ft_space = apply_gaussian_windowing(ft_space, sigma, axis=axis_time)
    # fourier transform in time C(k, t) -> C(k, w) = S
    s_k_w, w = fourier_transform_time(ft_space, dt)
    return {'S': s_k_w, 'k': k, 'k_reduced': k_reduced, 'w': w}


def fourier_transform_space(lat, a, axis=1):
    # transform mps array to lattice array
    a = lat.mps2lat_values(a, axes=axis)  # axis is only an int, since MPS is always "flattened"
    if lat.dim == 1:
        ft_space = np.fft.fftn(a, axes=(1, ))
        k = np.fft.fftfreq(ft_space.shape[1])
        # shifting
        ft_space = np.fft.fftshift(ft_space, axes=1)
        k = np.fft.fftshift(k)
        # make sure k is returned in correct basis
        # use the norm for the reciprocal basis, in case the basis is 2d (e.g. the ladder lattice)
        k = k * np.linalg.norm(lat.reciprocal_basis)
    else:
        # only transform over dims (1, 2), since 3 could hold unit cell index
        ft_space = np.fft.fftn(a, axes=(1, 2))
        k_x = np.fft.fftfreq(ft_space.shape[1])
        k_y = np.fft.fftfreq(ft_space.shape[2])
        # shifting
        ft_space = np.fft.fftshift(ft_space, axes=(1, 2))
        k_x = np.fft.fftshift(k_x)
        k_y = np.fft.fftshift(k_y)
        # make sure k is returned in correct basis (-> transform into reciprocal basis)
        b1, b2 = lat.reciprocal_basis
        k_x = b1 * k_x.reshape(-1, 1)  # multiply k_x by its basis vector (b1)
        k_y = b2 * k_y.reshape(-1, 1)  # multiply k_y by its basis vector (b2)
        # if k is indexed like (kx, ky) a coordinate (2d) is returned.
        k = k_x[:, np.newaxis, :] + k_y[np.newaxis, :, :]
    return ft_space, k


def fourier_transform_time(a, dt, axis=0):
    # fourier transform in time
    # (note that ifft is used, resulting in a minus sign in the exponential)
    ft_time = np.fft.ifft(a, axis=axis) * a.shape[axis]  # renormalize
    w = np.fft.fftfreq(len(ft_time), dt / (2 * np.pi))
    # shifting
    ft_time = np.fft.fftshift(ft_time, axes=axis)
    w = np.fft.fftshift(w)
    return ft_time, w


def apply_gaussian_windowing(a, sigma: float = 0.4, axis=0):
    """Simple gaussian windowing function along an axes.

    Applying a windowing function avoids Gibbs oscillation. tn are time steps 0, 1, ..., N

    Parameters
    ----------
    a : ndarray
        a ndarray where the time series is along axis `axes`
    sigma : float
        standard-deviation used for the gaussian window
    axis : int
        axes along which to apply the gaussian window

    Returns
    -------
    np.ndarray
    """
    # extract number of time-steps
    n_tsteps = a.shape[axis]
    tn = np.arange(n_tsteps)
    # create gaussian windowing function with the right length
    gaussian_window = np.exp(-0.5 * (tn / (n_tsteps * sigma))**2)
    # swap dimension which should be weighted (window applied to) to last dim (for np broadcasting)
    swapped_a = np.swapaxes(a, -1, axis)
    weighted_arr = swapped_a * gaussian_window  # apply window
    return np.swapaxes(weighted_arr, axis, -1)


def to_mps_geometry(lat, a):
    """Bring measurement in lattice geometry to mps geometry.

    This assumes that the array a has shape (..., Lx, Ly, Lu),
    or if Lu = 1, (..., Lx, Ly)
    """
    mps_idx_flattened = np.ravel_multi_index(tuple(lat.order.T), lat.shape)
    dims_until_lat_dims = a.ndim - (lat.dim + 1)  # add unit cell dim
    if lat.Lu == 1:
        dims_until_lat_dims += 1
    a = a.reshape(a.shape[:dims_until_lat_dims] + (-1, ))
    a = np.take(a, mps_idx_flattened, axis=-1)
    return a


def plot_correlations_on_lattice(ax,
                                 lat,
                                 correlations,
                                 pairs='nearest_neighbors',
                                 scale=1,
                                 color_pos='r',
                                 color_neg='g',
                                 color=None,
                                 zorder=0):
    """Function to plot correlations on a lattice.

    The strength of the correlations is given by the thickness of connecting lines.

    Parameters
    ----------
    ax :
        `matplotlib.axes.Axes`
    lat :
        a (TeNPy) lattice :class:`~tenpy.lattice.Lattice` to plot the correlations on
    correlations : array-like
        an array of correlations (in mps_form)
    pairs: str
        Pairs as in :attr:`~tenpy.lattice.Lattice.pairs`
    scale: float
        scale of the lines defining the correlations
    color_pos: str
        color for positive correlations
    color_neg: str
        color for negative correlations
    color: str
        one color for both positive and negative correlations
    zorder: float
        zorder of lines defining the correlations (set a higher order to display them above
        the couplings)
    """
    from matplotlib.collections import LineCollection

    mps_is = list()
    mps_js = list()
    for pair in lat.pairs[pairs]:
        coupling = lat.possible_couplings(*pair)
        mps_i = coupling[0]
        mps_j = coupling[1]
        mps_is.append(mps_i)
        mps_js.append(mps_j)

    all_mps_js = np.concatenate(mps_js)
    all_mps_is = np.concatenate(mps_is)

    pos_i = lat.position(lat.mps2lat_idx(all_mps_is))
    pos_j = lat.position(lat.mps2lat_idx(all_mps_js))

    pos_x = np.array([pos_i[:, 0], pos_j[:, 0]])
    if lat.dim == 1:
        pos_y = np.zeros(pos_x.shape)
    else:
        pos_y = np.array([pos_i[:, 1], pos_j[:, 1]])

    strengths = correlations[all_mps_is, all_mps_js]
    scaled_strengths = strengths * scale

    # differentiate between correlations larger than zero
    where_pos = scaled_strengths >= 0
    where_neg = np.bitwise_not(where_pos)

    if color is not None:
        color_pos = color_neg = color

    lc_pos = LineCollection(np.array([pos_x, pos_y]).T[where_pos],
                            linewidths=np.abs(scaled_strengths)[where_pos],
                            color=color_pos,
                            zorder=zorder)
    lc_neg = LineCollection(np.array([pos_x, pos_y]).T[where_neg],
                            linewidths=np.abs(scaled_strengths)[where_neg],
                            color=color_neg,
                            zorder=zorder)

    ax.add_collection(lc_pos)
    ax.add_collection(lc_neg)
