"""A test for tenpy.algorithms.truncation."""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np
import pytest

from tenpy.algorithms import truncation


def is_expected_S(S_expected, S_truncated):
    assert len(S_expected) == len(S_truncated)
    S_truncated = np.sort(S_truncated)[::-1]
    assert np.all(S_expected == S_truncated)


def test_truncate():
    # generate a test-S
    S = 10**(-np.arange(15.))  # exponentially decaying
    assert len(S) == 15  # 15 values
    # make artificial degeneracy
    S[5] = S[6] * (1. + 1.e-11)
    S[4] = S[5] * (1. + 1.e-9)
    # for degeneracy_tol = 1.e-10, S[4] != S[5] = S[6]
    # S is not normalized, but shouldn't matter for `truncate`
    S_shuffled = S.copy()
    np.random.shuffle(S_shuffled)

    # default arguments
    pars = dict(svd_min=None, trunc_cut=None)
    mask, norm_new, TE = truncation.truncate(S_shuffled, pars.copy())
    is_expected_S(S, S_shuffled[mask])  # don't truncate by default
    pars['chi_max'] = 12
    mask, norm_new, TE = truncation.truncate(S_shuffled, pars.copy())
    is_expected_S(S[:12], S_shuffled[mask])  # chi_max dominates
    pars['svd_min'] = 1.e-13  #  smaller than S[11], so chi_max should still dominate
    mask, norm_new, TE = truncation.truncate(S_shuffled, pars.copy())
    is_expected_S(S[:12], S_shuffled[mask])
    pars['svd_min'] = 2.e-10  #  between S[9] and S[10], should keep S[9], exclude S[10]
    mask, norm_new, TE = truncation.truncate(S_shuffled, pars.copy())
    is_expected_S(S[:10], S_shuffled[mask])
    # now with degeneracy: decrease chi_max to 7 to expect keeping S[6]
    pars['chi_max'] = 6  # not allowed to keep S[6]
    mask, norm_new, TE = truncation.truncate(S_shuffled, pars.copy())
    is_expected_S(S[:6], S_shuffled[mask])
    pars['degeneracy_tol'] = 1.e-10  # degeneracy: then also can't keep S[5]
    mask, norm_new, TE = truncation.truncate(S_shuffled, pars.copy())
    is_expected_S(S[:5], S_shuffled[mask])
    pars['degeneracy_tol'] = 1.e-8  # more degernacy: also can't keep S[4]
    mask, norm_new, TE = truncation.truncate(S_shuffled, pars.copy())
    is_expected_S(S[:4], S_shuffled[mask])
    pars['chi_min'] = 5  # want to keep S[4] nevertheless
    with pytest.warns(UserWarning):
        # can't satisfy the degeneracy criteria: S[4] == S[5] == S[6],
        # but chi_min requires S[4] to be kept, and chi_max excludes S[6]
        # so the degeneracy criteria shoulb be ignored and we should just keep 6 states
        mask, norm_new, TE = truncation.truncate(S_shuffled, pars.copy())
        is_expected_S(S[:6], S_shuffled[mask])
