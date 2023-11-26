"""This module contains functions for linear prediction."""
# Copyright 2020-2023 TeNPy Developers, GNU GPLv3

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import correlate
import logging

logger = logging.getLogger(__name__)

__all__ = ["linear_prediction", "simple_linear_prediction_1D", "get_lpc", "get_alpha_and_c"]


def linear_prediction(x, *args, axis=0, **kwargs):
    """Apply linear prediction to a multidimensional time series along an axis.

    The results are the predictions appended to the original array.

    Parameters
    ----------
    x: ndarray
        (multi dim) time series data
    axis: int
        default 0, axis along which to apply the prediction
    *args
        arguments to :func:`simple_linear_prediction`
    **kwargs
        keyword-arguments to :func:`simple_linear_prediction`

    Returns
    -------
    np.ndarray
        Predictions along the given axis (default 0), concatenated with the original input
    """
    pred_along_axis = np.apply_along_axis(simple_linear_prediction_1D, axis, x, *args, **kwargs)
    pred_along_axis_concat = np.concatenate([x, pred_along_axis], axis=axis)
    return pred_along_axis_concat


def simple_linear_prediction_1D(x: np.ndarray, m: int, p: int, truncation_mode: str = 'renormalize', split: float = 0):
    """Linear prediction of a one-dimensional time series data.

    Parameters
    ----------
    x : ndarray
        one-dimensional time series data
    m : int
        number of time-steps to predict
    p : int
        number of last points to base linear prediction on
    truncation_mode : str
        the truncation mode (default is 'renormalize', meaning their absolute
        value will be set to 1) used for truncating the eigenvalues. Other options are 'cutoff'  and 'conjugate'
    split: float
        between [0, 1) which is the proportion of data that should be discarded, i.e. short timescale effects

    Returns
    -------
    np.ndarray
    """
    if not isinstance(m, int) or not isinstance(p, int):
        raise TypeError("Parameters 'm' and 'p' must be integers.")
    if truncation_mode.casefold() not in ('cutoff', 'renormalize', 'conjugate'):
        raise ValueError("Parameter 'truncation_mode' must be 'renormalize' (default), 'cutoff', or 'conjugate'.")

    assert x.ndim == 1, "This version assumes a one-dimensional time series"
    assert 0 <= split < 1, "split must be between 0 and 1"
    N = len(x)
    split_idx = int(N * split)
    x = x[split_idx:]
    N -= split_idx
    if not p <= N:
        raise ValueError("p must be less or equal to the length of the time series")

    # get coefficients
    lpc = get_lpc(x, p)
    # get eigenvalues and eigenvectors
    alpha, c = get_alpha_and_c(x, lpc, truncation_mode)
    # fast version of the equivalent
    # predictions = [np.tensordot(c, alpha ** m_i, axes=(0, 0)) for m_i in range(1, m + 1)]
    multi_power = alpha[:, np.newaxis] ** np.arange(1, m + 1)
    predictions = np.tensordot(c, multi_power, axes=(0, 0))
    return predictions


def get_lpc(x, p):
    r"""Function to obtain the linear prediction coefficients (lpc) from the (last p) correlations of a time series x.

    First, the last p correlations are obtained, then the system of equations R x = r (which is in toeplitz form)
    is solved.

    Parameters
    ----------
    x : ndarray
        one-dimensional time series data
    p : int
        Number of coefficients

    Returns
    -------
    ndarray
        1D array containing the p linear prediction coefficients [a_p, a_{p-1}, ..., a_1] from
        the last p correlations of the time series x

    Notes
    -----
    The last p+1 correlations of the one-dimensional time-series are given by their autocorrelation
    ..math ::

        [E\{x(n)*x(n-0)\}, E\{x(n)*x(n-1)\}, ..., E\{x(n)*x(n-p)\}]

    """
    N = len(x)
    correlations = correlate(x, x, mode='full')[N - 1:N + p]  # scipy version uses fft
    r = correlations[1:]
    R = correlations[:-1]
    # getting array of coefficients
    lpc = solve_toeplitz(R, r)
    # which is just solving:
    #    r = correlations[1:]
    #    R = scipy.linalg.toeplitz(correlations[:-1]) # Toeplitz matrix
    #    R_inv = np.linalg.inv(R)
    #    lpc = R_inv @ r
    return lpc


def get_alpha_and_c(x, lpc, truncation_mode='cutoff', epsilon=10e-07):
    r"""Get the eigenvalues and coefficients from a vector of linear prediction coefficients for the time series x.

    This follows the approach taken in :arxiv:`0901.2342`. If necessary, the eigenvalues are truncated
    according to the `truncation_mode`

    Parameters
    ----------
    x : ndarray
        one-dimensional time series data
    lpc : ndarray
        1D array containing the p linear prediction coefficients [a_p, a_{p-1}, ..., a_1] from the correlations of x
    truncation_mode : str
        the truncation mode (default is 'cutoff', which means that those eigenvalues will be cut)
        used for truncating the eigenvalues. Other options are 'renormalize' (meaning their absolute
        value will be set to 1) and 'conjugate'
    epsilon : float
        Regularization parameter for matrix inversion. However, if the matrix can't be inverted here,
        linear prediction is probably not applicable

    Returns
    -------
    evals : ndarray
    c : ndarray
    """
    A = np.diag(np.ones(len(lpc) - 1), -1).astype(lpc.dtype)
    A[0] = lpc

    evals, evects = np.linalg.eig(A)  # Note that A is not symmetric!
    if truncation_mode == 'renormalize':
        evals[np.abs(evals) > 1] = evals[np.abs(evals) > 1] / np.abs(evals[np.abs(evals) > 1])
    elif truncation_mode == 'cutoff':
        evals[np.abs(evals) > 1] = 0
    elif truncation_mode == 'conjugate':
        evals[np.abs(evals) > 1] = 1 / np.conj(evals[np.abs(evals) > 1])

    x_tilde_N = x[-len(lpc):][::-1]
    shape = (-1,) + (x.ndim - 1) * (1,)
    try:
        evects_inv = np.linalg.inv(evects)
    except np.linalg.LinAlgError as e:
        # Regularization is only done here to avoid an Exception for ill-defined correlations (e.g. all zero)
        evects_inv = np.linalg.inv(evects + np.eye(len(evects)) * epsilon)
        logger.warning(f"Matrix inversion failed: {e}. Linear prediction will probably fail.")

    c = np.tensordot(evects_inv, x_tilde_N, axes=(1, 0)) * evects[0, :].reshape(shape)
    return evals, c
