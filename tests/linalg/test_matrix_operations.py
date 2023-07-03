"""A collection of tests for tenpy.linalg.matrix_operations."""
# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
import numpy as np
import numpy.testing as npt
import pytest
import warnings

from tenpy.linalg import tensors, matrix_operations


@pytest.mark.parametrize('new_vh_leg_dual', [True, False])
def test_svd(tensor_rng, new_vh_leg_dual):
    # also covers truncated_svd and truncate_singular_values
    
    T: tensors.Tensor = tensor_rng(labels=['l1', 'r2', 'l2', 'r1'], max_block_size=3)
    #  T_dense = T.to_numpy_ndarray()

    print('check svd without truncation')
    U, S, Vd = matrix_operations.svd(
        T, ['l1', 'l2'], ['r1', 'r2'], new_labels=['cr', 'cl'], new_vh_leg_dual=new_vh_leg_dual
    )
    U.test_sanity()
    S.test_sanity()
    Vd.test_sanity()
    assert U.labels_are('l1', 'l2', 'cr')
    assert S.labels_are('cl', 'cr')
    assert Vd.labels_are('cl', 'r1', 'r2')
    assert Vd.legs[0].is_dual == new_vh_leg_dual
    assert isinstance(S, tensors.DiagonalTensor)
    # check that U @ S @ Vd recovers the original tensor
    U_S_Vd = tensors.tdot(U, tensors.tdot(S, Vd, 'cr', 'cl'), 'cr', 'cl')
    U_S_Vd.test_sanity()
    assert tensors.almost_equal(T, U_S_Vd, atol=1.e-10)
    # check that U, Vd are isometries
    Ud_U = tensors.tdot(U.conj(), U, ['l1*', 'l2*'], ['l1', 'l2'])
    Vd_V = tensors.tdot(Vd, Vd.conj(), ['r1', 'r2'], ['r1*', 'r2*'])
    assert tensors.almost_equal(tensors.eye_like(Ud_U), Ud_U)
    assert tensors.almost_equal(tensors.eye_like(Vd_V), Vd_V)
    # check singular values
    T_np = T.to_numpy_ndarray(leg_order=['l1', 'l2', 'r1', 'r2'])
    _l1, _l2, _r1, _r2 = T_np.shape
    S_np = np.linalg.svd(T_np.reshape(_l1 * _l2, _r1 * _r2), compute_uv=False, full_matrices=False)
    npt.assert_array_almost_equal(S_np[:S.legs[0].dim], np.sort(S.diag_numpy)[::-1])

    for comment, options in [
        ('weak', dict(svd_min=1e-14)),
        ('strong', dict(svd_min=1e-4)),
    ]:
        print(f'check truncated_svd with {comment} truncation_options')
        U, S, Vd, err, renormalize = matrix_operations.truncated_svd(
            T, ['l1', 'l2'], ['r1', 'r2'], new_labels=['cr', 'cl'], new_vh_leg_dual=new_vh_leg_dual,
            truncation_options=options
        )
        U.test_sanity()
        S.test_sanity()
        Vd.test_sanity()
        npt.assert_array_almost_equal(np.sort(S.diag_numpy)[::-1], S_np[:S.shape[0]])
        # check that U @ S @ Vd recovers the original tensor up to the error incurred
        T_approx = tensors.tdot(U, tensors.tdot(S, Vd, 'cr', 'cl'), 'cr', 'cl')
        diff = tensors.norm(T - T_approx)
        npt.assert_allclose(diff, tensors.norm(T) * np.sqrt(err.eps), atol=1e-12)
        # check that U, Vd are isometries
        Ud_U = tensors.tdot(U.conj(), U, ['l1*', 'l2*'], ['l1', 'l2'])
        Vd_V = tensors.tdot(Vd, Vd.conj(), ['r1', 'r2'], ['r1*', 'r2*'])
        assert tensors.almost_equal(tensors.eye_like(Ud_U), Ud_U)
        assert tensors.almost_equal(tensors.eye_like(Vd_V), Vd_V)
