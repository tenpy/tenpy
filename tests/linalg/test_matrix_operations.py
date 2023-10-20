"""A collection of tests for tenpy.linalg.matrix_operations."""
# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
import numpy as np
import scipy
import numpy.testing as npt
import pytest
import warnings

from tenpy.linalg import tensors, matrix_operations, spaces, ProductSymmetry


@pytest.mark.parametrize('new_vh_leg_dual', [True, False])
def test_svd(tensor_rng, new_vh_leg_dual):
    # also covers truncated_svd and truncate_singular_values

    for labels, l_labels, r_labels in [
        (['l1', 'r2', 'l2', 'r1'], ['l1', 'l2'], ['r1', 'r2']),
        (['l1', 'r1', 'l2'], ['l2', 'l1'], ['r1']),
        (['l', 'r'], ['l'], ['r']),
    ]:
        assert set(l_labels + r_labels) == set(labels)
        print(f'leg bipartition {labels} -> {l_labels} & {r_labels}')

        T: tensors.Tensor = tensor_rng(labels=labels, max_block_size=3)
        #  T_dense = T.to_numpy_ndarray()

        U, S, Vd = matrix_operations.svd(
            T, l_labels, r_labels, new_labels=['cr', 'cl'], new_vh_leg_dual=new_vh_leg_dual
        )
        U.test_sanity()
        S.test_sanity()
        Vd.test_sanity()
        assert U.labels_are(*l_labels, 'cr')
        assert S.labels_are('cl', 'cr')
        assert Vd.labels_are('cl', *r_labels)
        assert Vd.legs[0].is_dual == new_vh_leg_dual
        assert isinstance(S, tensors.DiagonalTensor)
        # check that U @ S @ Vd recovers the original tensor
        U_S_Vd = tensors.tdot(U, tensors.tdot(S, Vd, 'cr', 'cl'), 'cr', 'cl')
        U_S_Vd.test_sanity()
        assert tensors.almost_equal(T, U_S_Vd, atol=1.e-10)
        # check that U, Vd are isometries
        Ud_U = tensors.tdot(U.conj(), U, [f'{l}*' for l in l_labels], l_labels)
        Vd_V = tensors.tdot(Vd, Vd.conj(), r_labels, [f'{r}*' for r in r_labels])
        assert tensors.almost_equal(tensors.eye_like(Ud_U), Ud_U)
        assert tensors.almost_equal(tensors.eye_like(Vd_V), Vd_V)
        # check singular values
        T_np = T.to_numpy_ndarray(leg_order=l_labels + r_labels)
        l_dim = np.prod(T_np.shape[:len(l_labels)])
        r_dim = np.prod(T_np.shape[len(l_labels):])
        S_np = np.linalg.svd(T_np.reshape(l_dim, r_dim), compute_uv=False, full_matrices=False)
        npt.assert_array_almost_equal(S_np[:S.legs[0].dim], np.sort(S.diag_numpy)[::-1])


@pytest.mark.parametrize(['new_vh_leg_dual', 'truncation'], list(zip([True, False], ['strong', 'weak', 'normalize'])))
def test_truncated_svd(tensor_rng, new_vh_leg_dual, truncation):
    T: tensors.Tensor = tensor_rng(labels=['l1', 'r2', 'l2', 'r1'], max_block_size=3)
    
    if truncation == 'weak':
        options = dict(svd_min=1e-14)
        normalize_to = None
    elif truncation == 'strong':
        options = dict(svd_min=1e-4)
        normalize_to = None
    elif truncation == 'normalize':
        options = dict(svd_min=1e-4)
        normalize_to = 2.7
    else:
        raise ValueError
        
    U, S, Vd, err, renormalize = matrix_operations.truncated_svd(
        T, ['l1', 'l2'], ['r1', 'r2'], new_labels=['cr', 'cl'], new_vh_leg_dual=new_vh_leg_dual,
        truncation_options=options, normalize_to=normalize_to
    )
    U.test_sanity()
    S.test_sanity()
    Vd.test_sanity()
    T_np = T.to_numpy_ndarray(leg_order=['l1', 'l2', 'r1', 'r2'])
    l_dim = np.prod(T_np.shape[:2])
    r_dim = np.prod(T_np.shape[2:])
    S_np = np.linalg.svd(T_np.reshape(l_dim, r_dim), compute_uv=False, full_matrices=False)
    if normalize_to is None:
        npt.assert_array_almost_equal(np.sort(S.diag_numpy)[::-1], S_np[:S.shape[0]])
    else:
        npt.assert_array_almost_equal(renormalize * np.sort(S.diag_numpy)[::-1], S_np[:S.shape[0]])
        npt.assert_almost_equal(tensors.norm(S), normalize_to)
    # check that U @ S @ Vd recovers the original tensor up to the error incurred
    T_approx = tensors.tdot(U, tensors.tdot(S, Vd, 'cr', 'cl'), 'cr', 'cl')
    npt.assert_allclose(err, tensors.norm(T - T_approx), atol=1e-12)
    # check that U, Vd are isometries
    Ud_U = tensors.tdot(U.conj(), U, ['l1*', 'l2*'], ['l1', 'l2'])
    Vd_V = tensors.tdot(Vd, Vd.conj(), ['r1', 'r2'], ['r1*', 'r2*'])
    assert tensors.almost_equal(tensors.eye_like(Ud_U), Ud_U)
    assert tensors.almost_equal(tensors.eye_like(Vd_V), Vd_V)


@pytest.mark.parametrize(['new_r_leg_dual', 'full'],
                         list(zip([True, False], [True, False])))
def test_qr(tensor_rng, new_r_leg_dual, full):
    T: tensors.Tensor = tensor_rng(labels=['l1', 'r2', 'l2', 'r1'], max_block_size=3)

    for comment, q_legs, r_legs in [
        ('all labelled', ['l1', 'l2'], ['r1', 'r2']),
        ('all numbered', [2, 0], [1, 3]),
        ('Q labelled', ['l1', 'l2'], None),
        ('R numbered', None, [1, 3]),
    ]:
        print(comment)
        Q, R = matrix_operations.qr(T, q_legs=q_legs, r_legs=r_legs, new_labels=['q', 'q*'],
                                    new_r_leg_dual=new_r_leg_dual, full=full)
        Q.test_sanity()
        R.test_sanity()
        if q_legs is None:
            expect_Q_labels = ['l1', 'l2', 'q']  # in order of appearance on T
        else:
            expect_Q_labels = [T.labels[n] for n in T.get_leg_idcs(q_legs)] + ['q']
        assert Q.labels == expect_Q_labels
        if r_legs is None:
            expect_R_labels = ['q*', 'r2', 'r1']  # in order of appearance on T
        else:
            expect_R_labels = ['q*'] + [T.labels[n] for n in T.get_leg_idcs(r_legs)]
        assert R.labels == expect_R_labels
        assert tensors.almost_equal(T, tensors.tdot(Q, R, 'q', 'q*'))
        Qd_Q = tensors.tdot(Q.conj(), Q, ['l1*', 'l2*'], ['l1', 'l2'])
        assert tensors.almost_equal(Qd_Q, tensors.eye_like(Qd_Q))
        # TODO should we check properties of R...?


@pytest.mark.parametrize(['new_l_leg_dual', 'full'],
                         list(zip([True, False], [True, False])))
def test_lq(tensor_rng, new_l_leg_dual, full):
    T: tensors.Tensor = tensor_rng(labels=['l1', 'r2', 'l2', 'r1'], max_block_size=3)

    for comment, l_legs, q_legs in [
        ('all labelled', ['l1', 'l2'], ['r1', 'r2']),
        ('all numbered', [2, 0], [1, 3]),
        ('Q labelled', ['l1', 'l2'], None),
        ('R numbered', None, [1, 3]),
    ]:
        print(comment)
        L, Q = matrix_operations.lq(T, l_legs=l_legs, q_legs=q_legs, new_labels=['q*', 'q'],
                                    new_l_leg_dual=new_l_leg_dual, full=full)
        L.test_sanity()
        Q.test_sanity()
        if l_legs is None:
            expect_L_labels = ['l1', 'l2', 'q*']  # in order of appearance on T
        else:
            expect_L_labels = [T.labels[n] for n in T.get_leg_idcs(l_legs)] + ['q*']
        if q_legs is None:
            expect_Q_labels = ['q', 'r2', 'r1']  # in order of appearance on T
        else:
            expect_Q_labels = ['q'] + [T.labels[n] for n in T.get_leg_idcs(q_legs)]
        assert L.labels == expect_L_labels
        assert Q.labels == expect_Q_labels
        # T == L @ Q ?
        assert tensors.almost_equal(T, tensors.tdot(L, Q, 'q*', 'q'))
        # Q isometric?
        Qd_Q = tensors.tdot(Q.conj(), Q, ['r1*', 'r2*'], ['r1', 'r2'])
        assert tensors.almost_equal(Qd_Q, tensors.eye_like(Qd_Q))
        if full:
            # Q unitary?
            Q_Qd = tensors.tdot(Q, Q.conj(), 'q', 'q*')
            expect = tensors.Tensor.eye(Q.get_legs(['r1', 'r2']), backend=T.backend,
                                        labels=['r1', 'r2', 'r1*', 'r2*'])
            assert tensors.almost_equal(Q_Qd, expect)


@pytest.mark.parametrize('real', [True, False])
def test_eigh(tensor_rng, vector_space_rng, real):
    a = vector_space_rng()
    b = vector_space_rng()
    T: tensors.Tensor = tensor_rng(legs=[a, b.dual, b, a.dual], real=real, labels=['a', 'b*', 'b', 'a*'])
    T = .5 * (T + T.conj())

    print('check that we have constructed a hermitian tensor')
    T_np = T.to_numpy_ndarray(leg_order=['a', 'b', 'a*', 'b*'])
    npt.assert_allclose(T_np, T_np.conj().transpose([2, 3, 0, 1]))

    print('perform eigh and test_sanity')
    D, U = matrix_operations.eigh(T, legs1=['a', 'b'], legs2=['a*', 'b*'], new_labels='c')
    D.test_sanity()
    U.test_sanity()
    assert D.dtype.is_real
    assert U.dtype == T.dtype

    print('checking legs and labels')
    new_leg = spaces.ProductSpace([a, b])
    assert D.legs == [new_leg, new_leg.dual]
    assert D.labels == ['c*', 'c']
    assert U.legs == [a, b, new_leg.dual]
    assert U.labels == ['a', 'b', 'c']

    print('checking eigen property')
    T_v = tensors.tdot(T, U, ['b*', 'a*'], ['b', 'a'])
    D_v = tensors.tdot(D, U, 'c*', 'c')
    assert tensors.almost_equal(T_v, D_v)

    print('checking normalization and completeness of eigenvectors (i.e. unitarity of U)')
    U_Ud = tensors.tdot(U, U.conj(), 'c', 'c*').combine_legs(['a', 'b'], ['a*', 'b*'])
    assert tensors.almost_equal(U_Ud, tensors.eye_like(U_Ud))
    Ud_U = tensors.tdot(U.conj(), U, ['a*', 'b*'], ['a', 'b'])
    assert tensors.almost_equal(Ud_U, tensors.eye_like(Ud_U))


@pytest.mark.parametrize('real', [True, False])
@pytest.mark.parametrize('mode', ['tensor', 'matrix', 'diagonal', 'scalar'])
@pytest.mark.parametrize('func', ['exp', 'log'])
def test_power_series_funcs(vector_space_rng, tensor_rng, np_random, func, mode, real):
    # common tests for matrix power series functions, such as exp, log etc
    tp_func = getattr(matrix_operations, func)
    np_func = dict(exp=scipy.linalg.expm, log=scipy.linalg.logm)[func]
    leg = vector_space_rng()
    leg2 = vector_space_rng()
    if mode == 'tensor':
        tens = tensor_rng(legs=[leg, leg2.dual, leg2, leg.dual], real=real, all_blocks=True)
        d1, d2 = leg.dim, leg2.dim
        d = leg.dim * leg2.dim
        res = tp_func(tens, legs1=[0, 2], legs2=[3, 1])
        res.test_sanity()
        res = res.to_numpy_ndarray()
        np_matrix = tens.to_numpy_ndarray().transpose([0, 2, 3, 1]).reshape([d, d])
        expect = np_func(np_matrix).reshape([d1, d2, d1, d2]).transpose([0, 3, 1, 2])
    elif mode == 'matrix':
        tens = tensor_rng(legs=[leg, leg.dual], real=real, all_blocks=True)
        res = tp_func(tens)
        res.test_sanity()
        res = res.to_numpy_ndarray()
        expect = np_func(tens.to_numpy_ndarray())
    elif mode == 'diagonal':
        data = np_random.random((leg.dim,))
        if not real:
            data = data + 1.j * np_random.random((leg.dim,))
        tens = tensors.DiagonalTensor.from_diag(diag=data, first_leg=leg)
        res = tp_func(tens)
        res.test_sanity()
        res = res.to_numpy_ndarray()
        expect = np_func(np.diag(data))
    elif mode == 'scalar':
        data = np_random.random((1, 1))
        if not real:
            data = data + 1.j * np_random.random((1, 1))
        res = tp_func(data.item())
        expect = np_func(data).item()
    else:
        raise RuntimeError

    if mode == 'tensor' and isinstance(tens.symmetry, ProductSymmetry) and func == 'log' and real:
        pytest.xfail()  # TODO no idea whats going on here...?
    
    npt.assert_allclose(res, expect, rtol=1e-7, atol=1e-12)
