"""This module contains functions for linear prediction."""
# Copyright 2020-2023 TeNPy Developers, GNU GPLv3

import numpy as np
import scipy
import logging
logger = logging.getLogger(__name__)

__all__ = ["linear_prediction", "autocorrelation", "get_lpc", "alpha_and_c"]


def linear_prediction(x: np.ndarray, m: int, p: int, split: float = 0, trunc_mode: str = 'renormalize',
                      two_d_mode: str = 'individual', cyclic: bool = False, epsilon: float = 10e-7) -> np.ndarray:
    """Linear prediction for m time steps, based on the last p data points of the time series x.

    Parameters
    ----------
    x : ndarray
        time series data (n_tsteps, ...)
    m : int
        number of timesteps to predict
    p : int
        number of last points to base linear prediction on
    split : float
        where 0<=split<1; this splits the dataset into a portion that is cut off in order to discard effects
        not relevant for the long term behaviour of the system
    trunc_mode : str
        the truncation mode (default is 'renormalize', meaning their absolute
        value will be set to 1) used for truncating the eigenvalues. Other options are 'cutoff'  and 'conjugate'
    two_d_mode : str
        the mode to use for 2D data (default is 'full', in which case the mse of the entire datapoints
        along the second (non-time) direction is taken). An alternative is 'individual', where the
        mse calculation is only base on the time series data points along each 'row' in the second dimension
    cyclic : bool
        whether to use the cyclic autocorrelation or not (see :meth:`autocorrelation`)
    epsilon : float
        regularization constant, in case matrix can not be inverted

    Returns
    -------
    ndarray
        with shape (n_tsteps + m, ...)
    """
    # data validation
    if not isinstance(m, int) or not isinstance(p, int):
        raise TypeError("Parameters 'm' and 'p' must be integers.")
    if not isinstance(cyclic, bool):
        raise TypeError("Cyclic must be a boolean value")
    if not isinstance(epsilon, float):
        raise TypeError("Parameter 'epsilon' must be a float.")
    if not isinstance(trunc_mode, str) or not isinstance(two_d_mode, str):
        raise TypeError("Parameter 'trunc_mode' and 'two_d_mode' must be a strings.")
    if trunc_mode.casefold() not in ('cutoff', 'renormalize', 'conjugate'):
        raise ValueError("Parameter 'trunc_mode' must be 'renormalize' (default), 'cutoff', or 'conjugate'.")
    if two_d_mode.casefold() not in ('full', 'individual'):
        raise ValueError("Parameter 'two_d_mode' must be 'individual' (default) or 'full'.")

    # get index of where to split dataset
    split_idx = int(len(x)*split)
    if two_d_mode == 'full':
        correlations = list()
        # get the last p correlations
        for j in range(p + 1):
            correlations.append(autocorrelation(x[split_idx:], 0, j, cyclic=cyclic))
        correlations = np.array(correlations)
        lpc = get_lpc(correlations, epsilon=epsilon)
        alpha, c = alpha_and_c(x[split_idx:], lpc, trunc_mode=trunc_mode)
        predictions = [np.tensordot(c, alpha ** m_i, axes=(0, 0)) for m_i in range(1, m + 1)]
        return np.concatenate([x, predictions])
    else:
        predictions = list()
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        for x_indiv_site in x.T:  # transpose array since it is given in (t_steps, site)
            correlations = list()
            # get the last p correlations
            for j in range(p + 1):
                correlations.append(autocorrelation(x_indiv_site[split_idx:], 0, j, cyclic=cyclic))
            correlations = np.array(correlations)
            lpc = get_lpc(correlations, epsilon=epsilon)
            alpha, c = alpha_and_c(x_indiv_site[split_idx:], lpc, trunc_mode=trunc_mode)
            predictions_site = [np.tensordot(c, alpha ** m_i, axes=(0, 0)) for m_i in range(1, m + 1)]
            predictions.append(np.concatenate([x_indiv_site, predictions_site]))
        return np.array(predictions).T  # transpose back


def autocorrelation(x, i, j, cyclic=False, cut_idx=0):  # TODO: introduce cut_pad_zero: bool = False
    """Compute the autocorrelation :math:`R_{XX}(i, j) = E[x(n-i) \cdot x(n-j)]`.

    Note that this can be rewritten as :math:`R_{XX}(i - j) = E[x(n) \cdot x(n-(j-i))]`

    Parameters
    ----------
    x : ndarray
        time series data, first dimension must be corresponding to the time
    i: delay of i steps of input signal
    j: delay of j steps of copy of input signal
    cyclic : bool
        whether the cyclic autocorrelation is used or not. If set to False (default), the
        data points with indices smaller than 0 are all set to zero. If set to True all indices
        are interpreted mod N (where N is the length of x) -> cyclic.
    cut_idx : int
        index at which data is cut off (e.g. to remove short time effects)
    Returns
    -------
    float
    """
    N = len(x)
    if cyclic is True:  # interprets the indices mod N
        # essentially we compute: np.sum([np.sum(x[n - i] * np.conj(x[n - j])) for n in range(N)])
        x_shift_i = np.roll(x[cut_idx:], -i, 0)
        x_shift_j = np.roll(x[cut_idx:], -j, 0)
        return np.sum(x_shift_i * np.conj(x_shift_j))
    else:
        # for 1D, this is equivalent to: np.correlate(x, x, mode='full')[x.size - 1 + (j - i)]
        # however, we allow a general input
        corrs = []
        for n in range(cut_idx, N):
            if (n - j + i) >= N:
                corrs.append(0)
            elif (n - j + i) < 0:
                corrs.append(0)
            else:
                corrs.append(np.sum(x[n] * np.conj(x[n - j + i])))
        return np.sum(corrs)


def get_lpc(correlations, epsilon=10e-7):
    """Function to obtain the linear prediction coefficients (lpc) of a time series x.

    Parameters
    ----------
    correlations : ndarray
        containing the last p+1 correlations of the time series
        [E{x(n)*x(n-0)}, E{x(n)*x(n-1)}, ..., E{x(n)*x(n-p)}
    epsilon : float
        regularization constant, in case matrix can not be inverted

    Returns
    -------
    ndarray
        1D array containing the p linear prediction coefficients [a_p, a_{p-1}, ..., a_1] from
        the correlations of the time series x
    """
    r = correlations[1:]
    R = scipy.linalg.toeplitz(correlations[:-1])
    # getting array of coefficients
    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        try:
            R_inv = np.linalg.inv(R + np.eye(len(R))*epsilon)
        except Exception:
            raise Exception(f"Matrix could not be inverted, even after applying regularization \
                            with epsilon = {epsilon}, maybe try a higher regularization parameter.")
    return R_inv @ r


def alpha_and_c(x, lpc, trunc_mode='cutoff'):
    """Get the eigenvalues and coefficients for linearly predicting the time series x.

    If necessary, truncate the eigenvalues.

    Parameters
    ----------
    x : ndarray
        time series data (n_tsteps, other)
    lpc : ndarray
        1D array containing the p linear prediction coefficients [a_p, a_{p-1}, ..., a_1] from the correlations of x
    trunc_mode : str
        the truncation mode (default is 'cutoff', which means that those eigenvalues will be cut)
        used for truncating the eigenvalues. Other options are 'renormalize' (meaning their absolute
        value will be set to 1) and 'conjugate'

    Returns
    -------
    evals : ndarray
    c : ndarray
    """
    A = np.diag(np.ones(len(lpc) - 1), -1).astype(lpc.dtype)
    A[0] = lpc

    evals, evects = np.linalg.eig(A)  # Note that A is not symmetric!
    if trunc_mode == 'renormalize':
        evals[np.abs(evals) > 1] = evals[np.abs(evals) > 1] / np.abs(evals[np.abs(evals) > 1])
    elif trunc_mode == 'cutoff':
        evals[np.abs(evals) > 1] = 0
    elif trunc_mode == 'conjugate':
        evals[np.abs(evals) > 1] = 1 / np.conj(evals[np.abs(evals) > 1])

    x_tilde_N = x[-len(lpc):][::-1]
    shape = (-1,) + (x.ndim - 1) * (1,)
    try:
        evects_inv = np.linalg.inv(evects)
    except np.linalg.LinAlgError as e:
        # Regularization is only done here to avoid an Exception for ill defined correlations (e.g. all zero)
        evects_inv = np.linalg.inv(evects + np.eye(len(evects))*10e-07)
        logger.warning(f"Matrix inversion failed: {e}. Linear prediction will probably fail.")

    c = np.tensordot(evects_inv, x_tilde_N, axes=(1, 0)) * evects[0, :].reshape(shape)
    return evals, c
