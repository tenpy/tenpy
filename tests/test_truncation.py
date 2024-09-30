"""A test for tenpy.algorithms.truncation."""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np
import pytest

import tenpy.linalg.np_conserved as npc
from tenpy.linalg import truncation
from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.algorithms import mpo_evolution


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


@pytest.mark.parametrize('use_eig_based_svd', [False, True])
@pytest.mark.parametrize('conserve', [None, 'parity'])
@pytest.mark.filterwarnings("ignore:_eig_based_svd is nonsensical on CPU!!")
def test_decompose_theta_qr_based(use_eig_based_svd, conserve, tol=1e-12):
    # Evolve state to obtain evolved tensors
    model_params = {
        'J': 1. ,
        'g': 1.,
        'L': 16,
        'bc_MPS': 'finite',
        'conserve': conserve
        }
    M = TFIChain(model_params)
    psi = MPS.from_lat_product_state(M.lat, [['up']])
    options = {
        'dt': 0.1,
        'N_steps': 1,
        'order': 1,
        'approximation': 'I',
        'cbe_min_block_increase': 1,
        'cbe_expand': 0.1,
        'use_eig_based_svd': use_eig_based_svd,
        'compression_method': 'variationalQR',
        'trunc_params': {
            'chi_max': 50,
            'svd_min': 1e-12,
        }
    }
    eng = mpo_evolution.ExpMPOEvolution(psi, M, options)
    for i in range(15):
        eng.run()
    assert psi.chi[8] == 50  # make sure we actually saturated chi to 50.

    # Fabricate dummy theta (not actually updated)
    S = psi.get_SL(8)
    old_T_L = psi.get_B(8, 'B').ireplace_label('p','p0')
    old_T_R = psi.get_B(9, 'B').ireplace_label('p','p1')
    theta = npc.tensordot(old_T_L.scale_axis(S, axis='vL'), old_T_R, ['vL','vR'])
    theta = theta.combine_legs([['vL','p0'], ['p1','vR']])

    for move_right in [True, False]:
        for chi_max in [None, 10, 45]:
            options.update(trunc_params={
                'chi_max': chi_max, 'svd_min': 1e-12
            })

            # SVD decomposition and truncation
            U, S_svd, VH, err_svd, renormalize_svd = truncation.svd_theta(theta, trunc_par=options.get('trunc_params'))

            # QR decomposition and truncation
            T_Lc, S_qr, T_Rc, form, err_qr, renormalize_qr = truncation.decompose_theta_qr_based(
                old_qtotal_L=old_T_L.qtotal, old_qtotal_R=old_T_R.qtotal, old_bond_leg=old_T_R.get_leg('vL'),
                theta=theta, move_right=move_right,
                expand=options.get('cbe_expand'),
                min_block_increase=options.get('cbe_min_block_increase'),
                use_eig_based_svd=use_eig_based_svd,
                trunc_params=options.get('trunc_params'),
                compute_err=True,
                return_both_T=True
            )
            T_L = T_Lc.split_legs(['(vL.p)'])
            T_R = T_Rc.split_legs(['(p.vR)'])

            # Check singular value properties
            assert np.all(S_qr > -tol)  # non-negative up to tolerance
            assert abs(np.linalg.norm(S_qr) - 1) < tol  # normalize
            assert chi_max is None or len(S_qr) <= chi_max
            # Compare singular values to SVD
            k = min(len(S_qr), len(S_svd))  # compare only the k largest singular values
            np.testing.assert_almost_equal(np.sort(S_qr)[-k:], np.sort(S_svd)[-k:])

            # Check that approximated theta reproduces the original theta, up to error given by err_qr
            if list(form) == ['A', 'B']:
                theta_approx = renormalize_qr * npc.tensordot(T_L.scale_axis(S_qr, 'vR'), T_R, ['vR', 'vL'])
            elif list(form) in [['Th', 'B'], ['A', 'Th']]:
                theta_approx = renormalize_qr * npc.tensordot(T_L, T_R, ['vR', 'vL'])
            else:
                raise NotImplementedError
            norm_err = npc.norm(theta_approx - theta.split_legs())
            expect_err = np.sqrt(err_qr.eps) * npc.norm(theta)
            assert abs(norm_err - expect_err) < tol

            # Check that QR truncated theta has the same accuracy as SVD truncated theta
            assert abs(err_svd.eps - err_qr.eps) < max(err_svd.eps * 0.01, 1e-15)

            # Check expected form of the QR tensors T_L and T_R:
            if form[0] == 'A':
                expect_eye = npc.tensordot(T_L.conj(), T_L, [['vL*','p*'],['vL','p']])
                assert npc.norm(expect_eye - npc.eye_like(expect_eye)) < tol
            elif form[0] == 'Th':
                assert abs(npc.norm(T_L) - 1) < tol
            else:
                raise NotImplementedError
            #
            if form[1] == 'Th':
                assert abs(npc.norm(T_R) - 1) < tol
            elif form[1] == 'B':
                expect_eye = npc.tensordot(T_R.conj(), T_R, [['vR*','p*'],['vR','p']])
                assert npc.norm(expect_eye - npc.eye_like(expect_eye)) < tol
            else:
                raise NotImplementedError
