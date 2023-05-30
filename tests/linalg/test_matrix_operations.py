"""A collection of tests for tenpy.linalg.matrix_operations."""
# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
import numpy as np
import numpy.testing as npt
import pytest
import warnings

from tenpy.linalg import tensors, matrix_operations


@pytest.mark.parametrize('new_vh_leg_dual', [True, False])
def test_svd(tensor_rng, new_vh_leg_dual):
    pytest.xfail("TODO: need to implement tdot for DiagonalTensor first")
    # TODO test multiple transpose
    T = tensor_rng(labels=['l1', 'r2', 'l2', 'r1'], max_block_size=3)
    #  T_dense = T.to_numpy_ndarray()

    U, S, Vd = matrix_operations.svd(T, ['l1', 'l2'], ['r1', 'r2'],
                                     new_labels=['cr', 'cl'],
                                     new_vh_leg_dual=new_vh_leg_dual)
    U.test_sanity()
    S.test_sanity()
    Vd.test_sanity()
    assert U.labels_are('l1', 'l2', 'cU*')
    assert S.labels_are('cU', 'cV*')
    assert Vd.labels_are('cV', 'r1', 'r2')
    assert Vd.legs[0].is_dual == new_vh_leg_dual
    assert isinstance(S, tensors.DiagonalTensor)
    U_S_Vd = tensors.tdot(U, tensors.tdot(S, Vd, 'cr', 'cl'), 'cr', 'cl')
    U_S_Vd.test_sanity()

    # check that U, Vd are isometries
    assert tensors.almost_equal(T, U_S_Vd, atol=1.e-10)
    Ud_U = tensors.tdot(U.conj(), U, ['l1*', 'l2*'], ['l1', 'l2'])
    Vd_V = tensors.tdot(Vd, Ud.conj(), ['r1', 'r2'], ['r1*', 'r2*'])
    pytest.xfail("TODO need more checks")  # TODO compare with eye_like()

