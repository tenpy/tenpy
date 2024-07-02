"""A test for tenpy.algorithms.truncation."""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np
import pytest

import tenpy.linalg.np_conserved as npc
from tenpy.algorithms import truncation
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


@pytest.mark.parametrize('use_eig_based_svd', [
    (False),
    (True),
])
@pytest.mark.filterwarnings("ignore:_eig_based_svd is nonsensical on CPU!!")
def test_decompose_theta_qr_based(use_eig_based_svd):
    # Evolve state to obtain evolved tensors
    model_params = {
        'J': 1. , 'g': 1.,
        'L': 16,
        'bc_MPS': 'finite',    
        }
    M = TFIChain(model_params)
    psi = MPS.from_lat_product_state(M.lat, [['up']])
    options = {
        'dt': 0.1,
        'N_steps': 1,
        'order': 1,
        'approximation': 'I',
        'cbe_min_block_increase': 1,
        'cbe_expand': float(0.1),
        'use_eig_based_svd': use_eig_based_svd,
        'compression_method': 'variationalQR',
        'trunc_params': {
            'chi_max': 50,
            'svd_min': 10**(-12),
        }
    }
    eng = mpo_evolution.ExpMPOEvolution(psi, M, options)
    for i in range(15):
        eng.run()
    
    # Fabricate dummy theta (not actually updated)
    S = psi.get_SL(8)
    old_T_L = psi.get_B(8, 'B').ireplace_label('p','p0')
    old_T_R = psi.get_B(9, 'B').ireplace_label('p','p1')
    theta = npc.tensordot(old_T_L.scale_axis(S, axis='vL'), old_T_R, ['vL','vR'])
    theta = theta.combine_legs([['vL','p0'],['p1','vR']])

    for move_right in [True, False]:
        for chi_max in [None, 10]:
            options.update(trunc_params={
                'chi_max': chi_max,
            })

            # SVD decomposition and truncation
            U, S, VH, SVD_err, renormalization = truncation.svd_theta(theta, trunc_par=options.get('trunc_params'))

            # QR decomposition and truncation
            T_Lc, S, T_Rc, form, QR_err, renormalization = truncation.decompose_theta_qr_based(
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

            # Check that approximated theta is the original theta
            if chi_max is None:
                assert SVD_err.eps < 1e-15
                assert QR_err.eps < 1e-15
            
            # Check that QR truncated theta has the same accuracy as SVD truncated theta
            assert abs(SVD_err.eps-QR_err.eps) < max(SVD_err.eps*0.01 ,1e-15)

            # Check canonical form of the QR tensors T_L and T_R by computing the norm error of the orthogonality condition
            L_l = npc.tensordot(T_L.conj(), T_L, [['vL*','p*'],['vL','p']])
            L_r = npc.tensordot(T_L.conj(), T_L, [['vR*','p*'],['vR','p']])
            L_norm = npc.inner(T_L.conj(), T_L, [['vL*','p*','vR*'],['vL','p','vR']])
            R_l = npc.tensordot(T_R.conj(), T_R, [['vL*','p*'],['vL','p']])
            R_r = npc.tensordot(T_R.conj(), T_R, [['vR*','p*'],['vR','p']])
            R_norm = npc.inner(T_R.conj(), T_R, [['vL*','p*','vR*'],['vL','p','vR']])

            L_err = {'A': npc.norm(L_l-npc.eye_like(L_l)), 'B': npc.norm(L_r-npc.eye_like(L_r)), 'Th': abs(L_norm-1)}
            R_err = {'A': npc.norm(R_l-npc.eye_like(R_l)), 'B': npc.norm(R_r-npc.eye_like(R_r)), 'Th': abs(R_norm-1)}

            if move_right:
                if use_eig_based_svd:
                    assert L_err['A'] < 1e-12 and R_err['Th'] < 1e-12 # form should be theta = A @ Th
                else:
                    assert L_err['A'] < 1e-12 and R_err['B'] < 1e-12 # form should be theta = A @ S @ B
            else:
                if use_eig_based_svd:
                    assert L_err['Th'] < 1e-12 and R_err['B'] < 1e-12 # form should be theta = Th @ B
                else:
                    assert L_err['A'] < 1e-12 and R_err['B'] < 1e-12 # form should be theta = A @ S @ B