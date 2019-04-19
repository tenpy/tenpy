"""A test for tenpy.algorithms.truncation"""
# Copyright 2018 TeNPy Developers

import numpy as np
import numpy.testing as npt

from tenpy.algorithms import truncation


def test_truncate():
    S = np.exp(-np.arange(15) - 0.1 * np.random.rand(15))
    np.random.shuffle(S)
    # default arguments
    pars = dict(verbose=1)
    mask, norm_new, TE = truncation.truncate(S, pars)
    print(S[mask])
    assert np.all(mask)  # don't truncate by default
    pars['chi_max'] = 18
    pars['chi_min'] = 5
    mask, norm_new, TE = truncation.truncate(S, pars)
    print(S[mask])
    assert (pars['chi_min'] <= np.sum(mask) <= pars['chi_max'])
    pars['trunc_cut'] = 0.0005**2
    mask, norm_new, TE = truncation.truncate(S, pars)
    print(S[mask])
    assert (pars['chi_min'] <= np.sum(mask) <= pars['chi_max'])
    assert (TE.eps <= pars['trunc_cut'])
    pars['svd_min'] = 0.005  # 10 times as large as trunc_cut -> allows to discard more
    mask, norm_new, TE = truncation.truncate(S, pars)
    print(S[mask])
    print(S[~mask])
    assert (pars['chi_min'] <= np.sum(mask) <= pars['chi_max'])
    assert (np.all(S[~mask] < pars['svd_min']))
    assert (np.all(S[mask] >= pars['svd_min']))
