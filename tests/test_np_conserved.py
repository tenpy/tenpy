"""A collection of tests for tenpy.linalg.np_conserved."""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import tenpy.linalg.np_conserved as npc
import numpy as np
import numpy.testing as npt
import itertools as it
from tenpy.tools.misc import inverse_permutation
import warnings
import pytest

from random_test import gen_random_legcharge, random_Array

chinfo = npc.ChargeInfo([1, 2], ['number', 'parity'])
# parity can be derived from number. Yet, this should all work...
qflat = np.array([[0, 0], [1, 1], [2, 0], [-2, 0], [1, 1]])
lc = npc.LegCharge.from_qflat(chinfo, qflat)
arr = np.zeros((5, 5))
arr[0, 0] = 1.
arr[1, 1] = arr[4, 1] = arr[1, 4] = 2.
arr[2, 2] = 3.
# don't fill all sectors compatible with charges: [3, 3] and [4, 4] are empty
# arr[3, 3] = 4.

qflat_add = qflat + np.array([[1, 0]])  # for checking non-zero total charge
lc_add = npc.LegCharge.from_qflat(chinfo, qflat_add)

chinfo2 = npc.ChargeInfo([1, 3, 1, 2])  # more charges for better testing
chinfo3 = npc.ChargeInfo([3])  # for larger blocks with random arrays of the same shape
chinfoTr = npc.ChargeInfo()  # trivial charge

lcTr = npc.LegCharge.from_qind(chinfoTr, [0, 2, 3, 5, 8], [[]] * 4)

EPS = np.finfo(np.float_).eps


def project_multiple_axes(flat_array, perms, axes):
    for p, a in it.izip(perms, axes):
        idx = [slice(None)] * flat_array.ndim
        idx[a] = p
        flat_array = flat_array[tuple(idx)]
    return flat_array


# ------- test functions -------------------


def test_npc_Array_conversion():
    print("trivial")
    a = npc.Array.from_ndarray_trivial(arr)
    npt.assert_equal(a.to_ndarray(), arr)
    print("non-trivial charges")
    a = npc.Array.from_ndarray(arr, [lc, lc.conj()])
    npt.assert_equal(a._get_block_charge([0, 2]), [-2, 0])
    npt.assert_equal(a.to_ndarray(), arr)
    npt.assert_equal(a.qtotal, [0, 0])
    print("empty")
    a = npc.Array([lc, lc.conj()])
    a.test_sanity()
    npt.assert_equal(a.to_ndarray(), np.zeros([lc.ind_len, lc.ind_len]))
    print("non-zero total charge")
    a = npc.Array.from_ndarray(arr, [lc, lc_add.conj()])
    npt.assert_equal(a.qtotal, [-1, 0])
    npt.assert_equal(a.to_ndarray(), arr)
    a = a.gauge_total_charge(1)
    npt.assert_equal(a.qtotal, [0, 0])
    print("type conversion")
    a_clx = a.astype(np.complex128)
    assert a_clx.dtype == np.complex128
    npt.assert_equal(a_clx.to_ndarray(), arr.astype(np.complex128))
    print("from_func")
    a = npc.Array.from_func(np.ones, [lc, lc.conj()])
    a.test_sanity()
    aflat = np.zeros((5, 5))
    for ind in [(0, 0), (1, 1), (1, 4), (4, 1), (2, 2), (3, 3), (4, 4)]:
        aflat[ind] = 1.
    npt.assert_equal(a.to_ndarray(), aflat)
    a = npc.ones([lc, lc.conj()])
    npt.assert_equal(a.to_ndarray(), aflat)
    print("random array")
    a = random_Array((20, 15, 10), chinfo2, sort=False)
    a.test_sanity()
    a = random_Array((20, 15, 10), chinfoTr, sort=False)
    a = npc.Array.from_func(np.random.random, [lcTr, lcTr.conj()], shape_kw='size')
    a.test_sanity()


def test_npc_Array_sort():
    print("sort a square matrix")
    a = npc.Array.from_ndarray(arr, [lc, lc.conj()])
    print(lc)
    p_flat, a_s = a.sort_legcharge(True, False)
    npt.assert_equal(p_flat[0], [3, 0, 2, 1, 4])
    arr_s = arr[np.ix_(*p_flat)]  # what a_s should be
    npt.assert_equal(a_s.to_ndarray(), arr_s)  # sort without bunch
    _, a_sb = a_s.sort_legcharge(False, True)
    npt.assert_equal(a_sb.to_ndarray(), arr_s)  # bunch after sort
    # after re-implementation of sort_legcharge, this is automatically _qdata_sorted...
    # npt.assert_equal(a_sb._qdata_sorted, False)
    a_sb.isort_qdata()
    npt.assert_equal(a_sb.to_ndarray(), arr_s)  # sort_qdata

    print("sort a for larger random array")
    a = random_Array((20, 15, 10), chinfo2, sort=False)
    p_flat, a_s = a.sort_legcharge(True, False)
    arr_s = a.to_ndarray()[np.ix_(*p_flat)]  # what a_s should be
    npt.assert_equal(a_s.to_ndarray(), arr_s)  # sort without bunch
    _, a_sb = a_s.sort_legcharge(False, True)
    npt.assert_equal(a_sb.to_ndarray(), arr_s)  # bunch after sort
    # npt.assert_equal(a_sb._qdata_sorted, False)
    a_sb.isort_qdata()
    npt.assert_equal(a_sb.to_ndarray(), arr_s)  # sort_qdata
    print("sort trivial charge data")
    a = random_Array((10, 5), chinfoTr, sort=False)
    p_flat, a_sb = a.sort_legcharge(False, True)

    print("'sort' trivial charge")
    a = npc.Array.from_func(np.random.random, [lcTr, lcTr.conj()], shape_kw='size')
    p_flat, a_s = a.sort_legcharge(True, False)
    a_s.test_sanity()
    npt.assert_equal(a_s.to_ndarray(), a.to_ndarray())  # p_flat should be trivial permuations...


def test_npc_Array_labels():
    a = npc.Array.from_ndarray(arr, [lc, lc.conj()])
    for t in [['x', None], [None, 'y'], ['x', 'y']]:
        a.iset_leg_labels(t)
        assert a.get_leg_labels() == t
        axes = (0, 1, 1, 0, 1, 0)
        axes_l = list(axes)  # replace with labels, where available
        for i, l in enumerate(axes[:4]):
            if t[l] is not None:
                axes_l[i] = t[l]
        assert tuple(a.get_leg_indices(axes_l)) == axes
    assert a.get_leg_index(-1) == 1  # negative indice


def test_npc_Array_project():
    a = npc.Array.from_ndarray(arr, [lc, lc.conj()])
    p1 = np.array([True, True, False, True, True])
    p2 = np.array([0, 1, 3])

    b = a.copy(True)
    b.iproject([p1, p2], (0, 1))
    b.test_sanity()
    bflat = a.to_ndarray()[np.ix_(p1, p2)]
    npt.assert_equal(b.to_ndarray(), bflat)
    # and again for a being blocked before: can we split the blocks
    print("for blocked")
    _, a = a.sort_legcharge()
    b = a.copy(True)
    b.iproject([p1, p2], (0, 1))
    b.test_sanity()
    bflat = a.to_ndarray()[np.ix_(p1, p2)]
    npt.assert_equal(b.to_ndarray(), bflat)

    print("for trivial charge")
    a = npc.Array.from_func(np.random.random, [lcTr, lcTr.conj()], shape_kw='size')
    p1 = (np.arange(lcTr.ind_len) % 3 == 0)
    b = a.copy(True)
    b.iproject([p1, p2], (0, 1))
    b.test_sanity()
    bflat = a.to_ndarray()[np.ix_(p1, p2)]
    npt.assert_equal(b.to_ndarray(), bflat)


def test_npc_Array_extend():
    a = npc.Array.from_ndarray(arr, [lc, lc.conj()])
    a = a.extend(0, 8)
    a.test_sanity()
    a.extend(1, 9)
    a.test_sanity()
    aflat = a.to_ndarray()
    aflat[:5, :5] -= arr
    assert np.all(aflat == 0)


def test_npc_Array_permute():
    sh = (20, 15, 10)
    a = random_Array(sh, chinfo)
    aflat = a.to_ndarray().copy()
    for ax in range(len(sh)):
        p = np.arange(sh[ax], dtype=np.intp)
        np.random.shuffle(p)
        a = a.permute(p, axis=ax)
        a.test_sanity()
        aflat = np.take(aflat, p, axis=ax)
        npt.assert_equal(a.to_ndarray(), aflat)


def test_npc_Array_transpose():
    a = random_Array((20, 15, 10), chinfo)
    aflat = a.to_ndarray()
    for tr in [None, [2, 1, 0], (1, 2, 0), (0, 2, 1)]:
        atr = a.transpose(tr)
        atr.test_sanity()
        npt.assert_equal(atr.to_ndarray(), aflat.transpose(tr))
    ax1, ax2 = -1, 0
    a.iswapaxes(ax1, ax2)
    npt.assert_equal(a.to_ndarray(), aflat.swapaxes(ax1, ax2))


def test_npc_Array_itemacces():
    a = npc.Array.from_ndarray(arr, [lc, lc.conj()])
    aflat = a.to_ndarray().copy()
    for i, j in it.product(range(5), range(5)):  # access all elements
        assert a[i, j] == aflat[i, j]
    for i, j in [(0, 0), (2, 2), (1, 4), (4, 1), (3, 3), (4, 4)]:  # sets also emtpy blocks
        val = np.random.rand()
        aflat[i, j] = val
        a[i, j] = val
    npt.assert_equal(a.to_ndarray(), aflat)
    # again for array with larger blocks
    a = random_Array((10, 10), chinfo3)
    aflat = a.to_ndarray().copy()
    for i, j in it.product(range(10), range(10)):  # access all elements
        assert a[i, j] == aflat[i, j]
    # take_slice and add_leg
    a = random_Array((20, 10, 5), chinfo3)
    aflat = a.to_ndarray().copy()
    for idx, axes in [(0, 0), (4, 1), ([3, -2], [-1, 0])]:
        a_sl = a.take_slice(idx, axes)
        a_sl.test_sanity()
        sl = [slice(None)] * a.rank
        try:
            for i, ax in zip(idx, axes):
                sl[ax] = i
        except:
            sl[axes] = idx
        sl = tuple(sl)
        npt.assert_equal(a_sl.to_ndarray(), aflat[sl])
        npt.assert_equal(a[sl].to_ndarray(), aflat[sl])
        # NOTE: interally uses advanced indexing notation, but only with slices.
        if type(axes) == int:
            a_ext = a_sl.add_leg(a.legs[axes], idx, axes)
            npt.assert_equal(a.qtotal, a_ext.qtotal)
            npt.assert_equal(a_ext.to_ndarray()[sl], aflat[sl])
    # advanced indexing with slices and projection/mask
    # NOTE: for bflat[idx] = aflat[idx] to work, non-slices may not be separated by slices.
    check_idx = [(2, Ellipsis, 1),
                 (slice(None), 3, np.array([True, True, False, True, False])),
                 (slice(3, 4), np.array([2, 4, 5]), slice(1, 4, 2)),
                 (slice(4, 2, -1), 2, np.array([3, 1, 4, 2]))]  # yapf: disable
    for idx in check_idx:
        print("take slice for ", idx)
        b = a[idx]
        b.test_sanity()
        bflat = aflat[idx]  # idx may only contain a single array for this to work
        npt.assert_equal(b.to_ndarray(), bflat)
    # create another random array to check copying with c[inds] = a[inds]
    b = npc.Array.from_func(np.random.random, a.legs, a.dtype, a.qtotal, shape_kw='size')
    # remove half of the blocks to check copying to empty sites as well
    keep = (np.arange(b.stored_blocks) % 2 == 0)
    try:
        b._data = [d for d, k in zip(b._data, keep) if k]
        b._qdata = b._qdata[keep]
    except AttributeError:
        pass  # for cython version, we can't write _data and _qdata...
    bflat = b.to_ndarray().copy()
    aflat = a.to_ndarray().copy()
    for idx in check_idx:
        print("copy slice for ", idx)
        b[idx] = a[idx]
        b.test_sanity()
        bflat[idx] = aflat[idx]  # idx may only contain a single array
        npt.assert_equal(b.to_ndarray(), bflat)


def test_npc_Array_reshape():
    a = random_Array((20, 15, 10), chinfo, sort=False)
    aflat = a.to_ndarray()
    for comb_legs, transpose in [([[1]], [0, 1, 2]), ([[1], [2]], [0, 1, 2]),
                                 ([[0], [1], [2]], [0, 1, 2]), ([[2, 0]], [1, 2, 0]),
                                 ([[2, 0, 1]], [2, 0, 1])]:
        print('combine legs', comb_legs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            acomb = a.combine_legs(comb_legs)  # just sorts second leg
        print("=> labels: ", acomb.get_leg_labels())
        acomb.test_sanity()
        asplit = acomb.split_legs()
        asplit.test_sanity()
        npt.assert_equal(asplit.to_ndarray(), aflat.transpose(transpose))
    print("test squeeze")
    b = random_Array((10, 1, 5, 1), chinfo3, sort=True)
    bflat = b.to_ndarray()
    bs = b.squeeze()
    bs.test_sanity()
    npt.assert_equal(bs.to_ndarray(), np.squeeze(bflat))
    bs.test_sanity()
    if b.stored_blocks > 0:
        # find a index with non-zero entry
        idx = tuple([l.slices[qi] for l, qi in zip(b.legs, b._qdata[0])])
    else:
        idx = tuple([0] * b.rank)
    assert b[idx[0], :, idx[2], :].squeeze() == bflat[idx]
    print("test add_trivial_leg")
    be = bs.copy(deep=True).add_trivial_leg(1, 'tr1', +1).add_trivial_leg(3, 'tr2', -1)
    be.test_sanity()
    npt.assert_equal(be.to_ndarray(), bflat)
    print("test concatenate")
    # create array `c` to concatenate with b along axis 2
    legs = b.legs[:]
    legs[1] = gen_random_legcharge(b.chinfo, 5)
    c1 = npc.Array.from_func(np.random.random, legs, qtotal=b.qtotal, shape_kw='size')
    c1flat = c1.to_ndarray()
    legs[1] = gen_random_legcharge(b.chinfo, 3)
    c2 = npc.Array.from_func(np.random.random, legs, qtotal=b.qtotal, shape_kw='size')
    c2flat = c2.to_ndarray()
    bc1c2 = npc.concatenate([b, c1, c2], axis=1)
    bc1c2.test_sanity()
    npt.assert_equal(bc1c2.to_ndarray(), np.concatenate([bflat, c1flat, c2flat], axis=1))

    print("trivial charges")
    a = npc.Array.from_func(np.random.random, [lcTr, lcTr.conj()], shape_kw='size')
    aflat = a.to_ndarray()
    acomb = a.combine_legs([0, 1])
    acomb.test_sanity()
    asplit = acomb.split_legs([0])
    asplit.test_sanity()
    npt.assert_equal(asplit.to_ndarray(), aflat)


def test_npc_Array_reshape_2():
    # check that combine_leg is compatible with pipe.map_incoming_flat
    shape = (2, 5, 2)
    a = random_Array(shape, chinfo3, sort=True)
    aflat = a.to_ndarray()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        acomb = a.combine_legs([[0, 1]])
    acombflat = acomb.to_ndarray()
    pipe = acomb.legs[0]
    print(a)
    print(acomb)
    print(pipe.q_map)
    print(pipe.slices)
    # expensive: compare all entries
    for i, j, k in it.product(*[list(range(s)) for s in shape]):
        ij = pipe.map_incoming_flat([i, j])
        print(i, j, k, ij)
        assert (acombflat[ij, k] == aflat[i, j, k])
    # done


def test_npc_grid_concat():
    # this is also a heavy test of Array.__getitem__
    ci = chinfo3
    legs = [gen_random_legcharge(ci, l) for l in [5, 4, 3]]
    A = npc.Array.from_func(np.random.random, legs, qtotal=[0], shape_kw='size')
    print("orig legs")
    for l in legs:
        print(l)
    print('---')
    Aflat = A.to_ndarray()
    grid = [A[..., :2, :], A[:, 2:3, ...], A[:, 3:]]
    A_full = npc.grid_concat(grid, [1])
    npt.assert_equal(Aflat, A_full.to_ndarray())
    grid = [[A[:, :1, 2:3], A[:, :1, 0:2]],
            [A[:, 2:, 2:3], A[:, 2:, 0:2]]]  # yapf: disable
    A_part = npc.grid_concat(grid, [1, 2]).to_ndarray()
    A_part_exact = Aflat[:, [0, 2, 3], :][:, :, [2, 0, 1]]
    npt.assert_equal(A_part, A_part_exact)
    grid[1][0] = None
    A_part = npc.grid_concat(grid, [1, 2]).to_ndarray()
    A_part_exact[:, 1:, 0] = 0.
    npt.assert_equal(A_part, A_part_exact)


def test_npc_grid_outer():
    ci = chinfo3
    p_leg = gen_random_legcharge(ci, 4)
    legs_op = [p_leg, p_leg.conj()]
    op_0 = 1.j * npc.Array.from_func(np.random.random, legs_op, qtotal=[0], shape_kw='size')
    op_pl = npc.Array.from_func(np.random.random, legs_op, qtotal=[1], shape_kw='size')
    op_min = npc.Array.from_func(np.random.random, legs_op, qtotal=[-1], shape_kw='size')
    op_id = npc.eye_like(op_0)
    grid = [[op_id, op_pl, op_min, op_0, None],
            [None, None, None, None, op_min],
            [None, None, None, None, op_pl],
            [None, None, None, None, op_0],
            [None, None, None, None, op_id]]  # yapf: disable
    leg_WL = npc.LegCharge.from_qflat(ci, ci.make_valid([[0], [1], [-1], [0], [0]]))
    leg_WR = npc.LegCharge.from_qflat(ci, ci.make_valid([[0], [1], [-1], [0], [0]]), -1)
    leg_WR_calc = npc.detect_grid_outer_legcharge(grid, [leg_WL, None], qconj=-1)[1]
    leg_WR.test_equal(leg_WR_calc)

    W = npc.grid_outer(grid, [leg_WL, leg_WR])
    W.test_sanity()
    Wflat = np.zeros([5, 5, 4, 4], dtype=W.dtype)
    for idx, op in [[(0, 0), op_id], [(0, 1), op_pl], [(0, 2), op_min], [(0, 3), op_0],
                    [(1, 4), op_min], [(2, 4), op_pl], [(3, 4), op_0], [(4, 4), op_id]]:
        Wflat[idx] = op.to_ndarray()
    npt.assert_equal(W.to_ndarray(), Wflat)


def test_npc_Array_scale_axis():
    a = random_Array((5, 15, 10), chinfo3, sort=True)
    aflat = a.to_ndarray()
    s0 = np.random.random((a.shape[0], 1, 1))
    s1 = np.random.random((1, a.shape[1], 1))
    s2 = np.random.random((1, 1, a.shape[2]))
    b = a.scale_axis(s0.flatten(), 0)
    b.test_sanity()
    npt.assert_equal(b.to_ndarray(), aflat * s0)
    a.iscale_axis(s1.flatten(), 1)
    a.test_sanity()
    aflat = aflat * s1
    npt.assert_equal(a.to_ndarray(), aflat)
    c = a.scale_axis(s2.flatten(), -1)
    c.test_sanity()
    npt.assert_equal(c.to_ndarray(), aflat * s2)


def test_npc_Array_conj():
    a = random_Array((15, 10), chinfo3, sort=True)
    a.iset_leg_labels(['a', 'b*'])
    aflat = a.to_ndarray()
    b = a.conj()
    b.test_sanity()
    npt.assert_equal(a.to_ndarray(), aflat)
    npt.assert_equal(b.to_ndarray(), aflat.conj())
    a.iconj()
    npt.assert_equal(a.to_ndarray(), aflat.conj())
    a.test_sanity()
    print(a.get_leg_labels())
    assert a._conj_leg_label('(a*.(b.c*).(d*.e))') == '(a.(b*.c).(d.e*))'
    print("conjugate Trivial charges")
    a = npc.Array.from_func(np.random.random, [lcTr, lcTr.conj()], shape_kw='size')
    aflat = a.to_ndarray()
    a.iconj()
    npt.assert_equal(a.to_ndarray(), aflat.conj())
    a.test_sanity()


def test_npc_Array_norm():
    a = random_Array((15, 10), chinfo3, sort=True)
    aflat = a.to_ndarray()
    for ord in [np.inf, -np.inf, 0, 1, 2, 3.]:  # divides by 0 for neg. ord
        print("ord = ", ord)
        anorm = a.norm(ord)
        aflnorm = npc.norm(aflat, ord)
        print(abs(anorm - aflnorm))
        assert (abs(anorm - aflnorm) < 100 * EPS)


def test_npc_Array_ops():
    a = random_Array((15, 10), chinfo3, sort=True)
    b = npc.Array.from_func(np.random.random, a.legs, qtotal=a.qtotal, shape_kw='size')
    s = 3.12
    aflat = a.to_ndarray()
    bflat = b.to_ndarray()
    import operator as Op
    # addition / subtraction
    for op in [Op.add, Op.sub, Op.iadd, Op.isub]:
        print(op.__name__)
        a2 = op(a, b)
        a.test_sanity()
        b.test_sanity()
        a2.test_sanity()
        aflat2 = op(aflat, bflat)
        npt.assert_equal(a.to_ndarray(), aflat)
        npt.assert_equal(a2.to_ndarray(), aflat2)
    npt.assert_equal(b.to_ndarray(), bflat)  # should not have been modified...
    # multiplication
    for op in [Op.mul, Op.imul]:
        print(op.__name__)
        a2 = op(a, s)
        a.test_sanity()
        b.test_sanity()
        a2.test_sanity()
        aflat2 = op(aflat, s)
        npt.assert_equal(a.to_ndarray(), aflat)
        npt.assert_equal(a2.to_ndarray(), aflat2)
    # reversed multiplication
    print("rmul")
    a2 = s * a
    a.test_sanity()
    a2.test_sanity()
    aflat2 = s * aflat
    npt.assert_equal(a.to_ndarray(), aflat)
    npt.assert_equal(a2.to_ndarray(), aflat2)
    # division
    for op in [Op.truediv, Op.itruediv]:
        # may differ by machine precision due to rounding errors
        print(op.__name__)
        a2 = op(a, s)
        a.test_sanity()
        a2.test_sanity()
        aflat2 = op(aflat, s)
        assert (np.max(np.abs(a.to_ndarray() - aflat)) < EPS)
        assert (np.max(np.abs(a.to_ndarray() - aflat)) < EPS)
    # equality
    assert a == a
    a2 = a.copy(deep=False)
    b = a.copy(deep=True)
    assert b == a == a2
    assert len(b._data) > 0
    b._data[-1][0, -1] += 1.e-13  # change
    assert not b == a  # not exactly equal
    assert a.__eq__(b, 1.e-12)  # but to high precision


def test_npc_addition_transpose():
    # addition with labels and transposed axes
    a1 = np.random.random([3, 3, 4])
    a2 = np.swapaxes(a1, 0, 1)
    t1 = npc.Array.from_ndarray_trivial(a1, labels=['a', 'b', 'c'])
    t2 = npc.Array.from_ndarray_trivial(a2, labels=['b', 'a', 'c'])
    # TODO: for now warning
    with pytest.warns(FutureWarning):
        diff = npc.norm(t1 - t2)
    # TODO: when the behaviour is changed do
    #  diff = npc.norm(t1 - t2)
    #  assert diff < 1.e-10


def test_npc_tensordot():
    for sort in [True, False]:
        print("sort =", sort)
        a = random_Array((10, 12, 15), chinfo3, qtotal=[0], sort=sort)
        aflat = a.to_ndarray()
        legs_b = [l.conj() for l in a.legs[::-1]]
        b = npc.Array.from_func(np.random.random, legs_b, qtotal=[1], shape_kw='size')
        b = b * (1 + 1.j)  # make second array complex: check that different dtypes work
        bflat = b.to_ndarray()
        print("axes = 1")  # start simple: only one axes
        c = npc.tensordot(a, b, axes=1)
        c.test_sanity()
        a.test_sanity()
        b.test_sanity()
        npt.assert_array_almost_equal_nulp(a.to_ndarray(), aflat, 1)
        npt.assert_array_almost_equal_nulp(b.to_ndarray(), bflat, 1)
        cflat = np.tensordot(aflat, bflat, axes=1)
        npt.assert_array_almost_equal_nulp(c.to_ndarray(), cflat, sum(a.shape))
        print("axes = 2")  # second: more than one axis
        c = npc.tensordot(a, b, axes=([1, 2], [1, 0]))
        a.test_sanity()
        b.test_sanity()
        npt.assert_array_almost_equal_nulp(a.to_ndarray(), aflat, 1)
        npt.assert_array_almost_equal_nulp(b.to_ndarray(), bflat, 1)
        c.test_sanity()
        cflat = np.tensordot(aflat, bflat, axes=([1, 2], [1, 0]))
        npt.assert_array_almost_equal_nulp(c.to_ndarray(), cflat, sum(a.shape))
        for i in range(b.shape[0]):
            b2 = b[i, :, :]
            if b2.stored_blocks > 0:
                break
        b2flat = b2.to_ndarray()
        print("right tensor fully contracted")
        print(a.shape, b2.shape)
        d = npc.tensordot(a, b2, axes=([0, 1], [1, 0]))
        d.test_sanity()
        dflat = np.tensordot(aflat, b2flat, axes=([0, 1], [1, 0]))
        npt.assert_array_almost_equal_nulp(d.to_ndarray(), dflat, sum(a.shape))
        print("left tensor fully contracted")
        d = npc.tensordot(b2, a, axes=([0, 1], [1, 0]))
        d.test_sanity()
        dflat = np.tensordot(b2flat, aflat, axes=([0, 1], [1, 0]))
        npt.assert_array_almost_equal_nulp(d.to_ndarray(), dflat, sum(a.shape))
    # full/no contraction is tested in test_npc_inner/test_npc_outer
    print("for trivial charge")
    a = npc.Array.from_func(np.random.random, [lcTr, lcTr.conj()], shape_kw='size')
    aflat = a.to_ndarray()
    b = npc.tensordot(a, a, axes=1)
    bflat = np.tensordot(aflat, aflat, axes=1)
    npt.assert_array_almost_equal_nulp(b.to_ndarray(), bflat, sum(a.shape))


def test_npc_tensordot_extra():
    # check that the sorting of charges is fine with special test matrices
    # which gave me some headaches at some point :/
    chinfo = npc.ChargeInfo([1], ['Sz'])
    leg = npc.LegCharge.from_qflat(chinfo, [-1, 1])
    legs = [leg, leg, leg.conj(), leg.conj()]
    idx = [(0, 0, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 1, 1, 1)]
    Uflat = np.eye(4).reshape([2, 2, 2, 2])  # up to numerical rubbish the identity
    Uflat[0, 1, 1, 0] = Uflat[1, 0, 0, 1] = 1.e-20
    U = npc.Array.from_ndarray(Uflat, legs, cutoff=0.)
    theta_flat = np.zeros([2, 2, 2, 2])
    vals = np.random.random(len(idx))
    vals /= np.linalg.norm(vals)
    for i, val in zip(idx, vals):
        theta_flat[i] = val
    theta = npc.Array.from_ndarray(theta_flat, [leg, leg, leg.conj(), leg.conj()], cutoff=0.)
    assert abs(np.linalg.norm(theta_flat) - npc.norm(theta)) < 1.e-14
    Utheta_flat = np.tensordot(Uflat, theta_flat, axes=2)
    Utheta = npc.tensordot(U, theta, axes=2)
    npt.assert_array_almost_equal_nulp(Utheta.to_ndarray(), Utheta_flat, 10)
    assert abs(np.linalg.norm(theta_flat) - npc.norm(Utheta)) < 1.e-10


def test_npc_inner(tol=1.e-13):
    for sort in [True, False]:
        print("sort =", sort)
        a = random_Array((10, 7, 5), chinfo3, sort=sort)
        a.iset_leg_labels(['x', 'y', 'z'])
        aflat = a.to_ndarray()
        b = npc.Array.from_func(np.random.random, a.legs, qtotal=a.qtotal, shape_kw='size')
        b.iset_leg_labels(['x', 'y', 'z'])
        b_conj = b.conj()
        b_conj_flat = b.to_ndarray()
        cflat = np.tensordot(aflat, b_conj_flat, axes=[[0, 1, 2], [0, 1, 2]])
        c = npc.inner(a, b_conj, axes='range')  # no transpose
        assert type(c) == np.dtype(float)
        assert (abs(c - cflat) < tol)
        c = npc.inner(a, b_conj, axes=[[0, 1, 2], [0, 1, 2]])
        assert (abs(c - cflat) < tol)
        c = npc.inner(a, b, axes='range', do_conj=True)
        assert (abs(c - cflat) < tol)
        # now transpose
        b.itranspose([2, 1, 0])
        b_conj.itranspose([2, 1, 0])
        c = npc.inner(a, b_conj, axes=[[2, 0, 1], [0, 2, 1]])  # unordered axes!
        assert (abs(c - cflat) < tol)
        c = npc.inner(a, b_conj, axes='labels')
        assert (abs(c - cflat) < tol)
        c = npc.inner(a, b, axes='labels', do_conj=True)
        assert (abs(c - cflat) < tol)

    print("for trivial charge")
    a = npc.Array.from_func(np.random.random, [lcTr, lcTr.conj()], shape_kw='size')
    aflat = a.to_ndarray()
    b = npc.tensordot(a, a, axes=2)
    bflat = np.tensordot(aflat, aflat, axes=2)
    npt.assert_array_almost_equal_nulp(b, bflat, sum(a.shape))


def test_npc_outer():
    for sort in [True, False]:
        print("sort =", sort)
        a = random_Array((6, 7), chinfo3, sort=sort)
        b = random_Array((5, 5), chinfo3, sort=sort)
        aflat = a.to_ndarray()
        bflat = b.to_ndarray()
        c = npc.outer(a, b)
        c.test_sanity()
        cflat = np.tensordot(aflat, bflat, axes=0)
        npt.assert_equal(c.to_ndarray(), cflat)
        c = npc.tensordot(a, b, axes=0)  # (should as well call npc.outer)
        npt.assert_equal(c.to_ndarray(), cflat)

    print("for trivial charge")
    a = npc.Array.from_func(np.random.random, [lcTr, lcTr.conj()], shape_kw='size')
    aflat = a.to_ndarray()
    b = npc.tensordot(a, a, axes=0)
    bflat = np.tensordot(aflat, aflat, axes=0)
    npt.assert_array_almost_equal_nulp(b.to_ndarray(), bflat, sum(a.shape))


def test_npc_svd():
    for m, n in [(1, 1), (1, 10), (10, 1), (10, 10), (10, 20)]:
        print("m, n = ", m, n)
        tol_NULP = max(20 * max(m, n)**3, 1000)
        for i in range(1000):
            A = random_Array((m, n), chinfo3, sort=True)
            if A.stored_blocks > 0:
                break
        Aflat = A.to_ndarray()
        Sonly = npc.svd(A, compute_uv=False)
        U, S, VH = npc.svd(A, full_matrices=False, compute_uv=True)
        assert (U.shape[1] == S.shape[0] == VH.shape[0])
        U.test_sanity()
        VH.test_sanity()
        npt.assert_array_almost_equal_nulp(Sonly, S, tol_NULP)
        recalc = npc.tensordot(U.scale_axis(S, axis=-1), VH, axes=1)
        npt.assert_array_almost_equal_nulp(recalc.to_ndarray(), Aflat, tol_NULP)
        # compare with flat SVD
        Uflat, Sflat, VHflat = np.linalg.svd(Aflat, False, True)
        perm = np.argsort(-S)  # sort descending
        print(S[perm])
        iperm = inverse_permutation(perm)
        for i in range(len(Sflat)):
            if i not in iperm:  # dopped it in npc.svd()
                assert (Sflat[i] < EPS * 10)
        Sflat = Sflat[iperm]
        npt.assert_array_almost_equal_nulp(Sonly, Sflat, tol_NULP)
        # comparing U and Uflat is hard: U columns can change by a phase...
    print("with full_matrices")
    Ufull, Sfull, VHfull = npc.svd(A, full_matrices=True, compute_uv=True)
    Ufull.test_sanity()
    VHfull.test_sanity()
    npt.assert_array_almost_equal_nulp(Sfull, S, tol_NULP)

    print("for trivial charges")
    A = npc.Array.from_func(np.random.random, [lcTr, lcTr.conj()], shape_kw='size')
    Aflat = A.to_ndarray()
    U, S, VH = npc.svd(A)
    recalc = npc.tensordot(U.scale_axis(S, axis=-1), VH, axes=1)
    tol_NULP = max(20 * max(A.shape)**3, 1000)
    npt.assert_array_almost_equal_nulp(recalc.to_ndarray(), Aflat, tol_NULP)


def test_npc_pinv():
    m, n = (10, 20)
    A = random_Array((m, n), chinfo3)
    tol_NULP = max(max(m, n)**3, 1000)
    Aflat = A.to_ndarray()
    P = npc.pinv(A, 1.e-13)
    P.test_sanity()
    Pflat = np.linalg.pinv(Aflat, 1.e-13)
    assert (np.max(np.abs(P.to_ndarray() - Pflat)) < tol_NULP * EPS)


def test_trace():
    chinfo = chinfo3
    legs = [gen_random_legcharge(chinfo, s) for s in (7, 8, 9)]
    legs.append(legs[1].conj())
    A = npc.Array.from_func(np.random.random, legs, qtotal=[1], shape_kw='size')
    Aflat = A.to_ndarray()
    Atr = npc.trace(A, leg1=1, leg2=-1)
    Atr.test_sanity()
    Aflattr = np.trace(Aflat, axis1=1, axis2=-1)
    npt.assert_array_almost_equal_nulp(Atr.to_ndarray(), Aflattr, A.shape[1])


def test_eig():
    size = 10
    max_nulp = 10 * size**3
    ci = chinfo3
    l = gen_random_legcharge(ci, size)
    A = npc.Array.from_func(np.random.random, [l, l.conj()], qtotal=None, shape_kw='size')
    print("hermitian A")
    A += A.conj().itranspose()
    Aflat = A.to_ndarray()
    W, V = npc.eigh(A, sort='m>')
    V.test_sanity()
    V_W = V.scale_axis(W, axis=-1)
    recalc = npc.tensordot(V_W, V.conj(), axes=[1, 1])
    npt.assert_array_almost_equal_nulp(Aflat, recalc.to_ndarray(), max_nulp)
    Wflat, Vflat = np.linalg.eigh(Aflat)
    npt.assert_array_almost_equal_nulp(np.sort(W), Wflat, max_nulp)
    W2 = npc.eigvalsh(A, sort='m>')
    npt.assert_array_almost_equal_nulp(W, W2, max_nulp)

    print("check complex B")
    B = 1.j * npc.Array.from_func(np.random.random, [l, l.conj()], shape_kw='size')
    B += B.conj().itranspose()
    B = A + B
    Bflat = B.to_ndarray()
    W, V = npc.eigh(B, sort='m>')
    V.test_sanity()
    recalc = npc.tensordot(V.scale_axis(W, axis=-1), V.conj(), axes=[1, 1])
    npt.assert_array_almost_equal_nulp(Bflat, recalc.to_ndarray(), max_nulp)
    Wflat, Vflat = np.linalg.eigh(Bflat)
    npt.assert_array_almost_equal_nulp(np.sort(W), Wflat, max_nulp)

    print("calculate without 'hermitian' knownledge")
    W, V = npc.eig(B, sort='m>')
    assert (np.max(np.abs(W.imag)) < EPS * max_nulp)
    npt.assert_array_almost_equal_nulp(np.sort(W.real), Wflat, max_nulp)

    print("sparse speigs")
    qi = 1
    ch_sect = B.legs[0].get_charge(qi)
    k = min(3, B.legs[0].slices[qi + 1] - B.legs[0].slices[qi])
    Wsp, Vsp = npc.speigs(B, ch_sect, k=k, which='LM')
    for W_i, V_i in zip(Wsp, Vsp):
        V_i.test_sanity()
        diff = npc.tensordot(B, V_i, axes=1) - V_i * W_i
        assert (npc.norm(diff, np.inf) < EPS * max_nulp)

    print("for trivial charges")
    A = npc.Array.from_func(np.random.random, [lcTr, lcTr.conj()], shape_kw='size')
    A = A + A.conj().itranspose()
    Aflat = A.to_ndarray()
    W, V = npc.eigh(A)
    recalc = npc.tensordot(V.scale_axis(W, axis=-1), V.conj(), axes=[1, 1])
    npt.assert_array_almost_equal_nulp(Aflat, recalc.to_ndarray(), 10 * A.shape[0]**3)


def test_expm(size=10):
    ci = chinfo3
    l = gen_random_legcharge(ci, size)
    A = npc.Array.from_func(np.random.random, [l, l.conj()], qtotal=None, shape_kw='size')
    A_flat = A.to_ndarray()
    exp_A = npc.expm(A)
    exp_A.test_sanity()
    from scipy.linalg import expm
    npt.assert_array_almost_equal_nulp(expm(A_flat), exp_A.to_ndarray(), size * size)


def test_qr():
    for shape in [(4, 4), (6, 8), (8, 6)]:
        for qtotal in [None, [1]]:
            print("qtotal=", qtotal, "shape =", shape)
            A = random_Array(shape, chinfo3, qtotal=qtotal, sort=False)
            A_flat = A.to_ndarray()
            q, r = npc.qr(A, 'reduced')
            print(q._qdata)
            q.test_sanity()
            r.test_sanity()
            qr = npc.tensordot(q, r, axes=1)
            npt.assert_array_almost_equal_nulp(A_flat, qr.to_ndarray(), shape[0] * shape[1] * 100)


def test_charge_detection():
    chinfo = chinfo3
    for qtotal in [[0], [1], None]:
        print("qtotal=", qtotal)
        shape = (8, 6, 5)
        A = random_Array(shape, chinfo3, qtotal=qtotal)
        Aflat = A.to_ndarray()
        legs = A.legs[:]
        print(A)
        if not np.any(Aflat > 1.e-8):
            print("skip test: no non-zero entry")
            continue
        qt = npc.detect_qtotal(Aflat, legs)
        npt.assert_equal(qt, chinfo.make_valid(qtotal))
        for i in range(len(shape)):
            correct_leg = legs[i]
            legs[i] = None
            legs = npc.detect_legcharge(Aflat, chinfo, legs, A.qtotal, correct_leg.qconj)
            res_leg = legs[i]
            assert res_leg.qconj == correct_leg.qconj
            legs[i].bunch()[1].test_equal(correct_leg.bunch()[1])
    # done


def test_drop_add_change_charge():
    chinfo14 = npc.ChargeInfo([1, 4], ['U1', 'Z4'])
    chinfo41 = npc.ChargeInfo([4, 1], ['Z4', 'U1'])
    chinfo1 = npc.ChargeInfo([1], ['U1'])
    chinfo4 = npc.ChargeInfo([4], ['Z4'])
    chinfo12 = npc.ChargeInfo([1, 2], ['U1', 'Z2'])
    for shape in [(50, ), (10, 4), (1, 1, 2)]:
        A14 = random_Array(shape, chinfo14)
        A14_flat = A14.to_ndarray()
        A = A14.drop_charge()
        A.test_sanity()
        npt.assert_equal(A.to_ndarray(), A14_flat)
        assert A.chinfo == chinfoTr
        A1 = A14.drop_charge(1)
        A1.test_sanity()
        npt.assert_equal(A1.to_ndarray(), A14_flat)
        assert A1.chinfo == chinfo1
        A4 = A14.drop_charge('U1', chinfo4)
        npt.assert_equal(A4.to_ndarray(), A14_flat)
        assert A4.chinfo is chinfo4
        A12 = A14.change_charge('Z4', 2, 'Z2', chinfo12)
        A12.test_sanity()
        npt.assert_equal(A4.to_ndarray(), A14_flat)
        assert A12.chinfo is chinfo12
        A14_new = A1.add_charge(A4.legs, qtotal=A4.qtotal)
        A14_new.test_sanity()
        npt.assert_equal(A14_new.to_ndarray(), A14_flat)
        assert A14_new.chinfo == chinfo14
        A41_new = A4.add_charge(A1.legs, chinfo41, qtotal=A1.qtotal)
        A41_new.test_sanity()
        npt.assert_equal(A41_new.to_ndarray(), A14_flat)
        assert A41_new.chinfo is chinfo41


def test_pickle():
    import pickle
    a = npc.Array.from_ndarray(arr, [lc, lc_add.conj()])
    b = random_Array((20, 15, 10), chinfo2, sort=False)
    a.test_sanity()
    b.test_sanity()
    aflat = a.to_ndarray()
    bflat = b.to_ndarray()
    data = {'a': a, 'b': b}
    stream = pickle.dumps(data)
    data2 = pickle.loads(stream)
    a2 = data2['a']
    b2 = data2['b']
    a.test_sanity()
    b.test_sanity()
    a2.test_sanity()
    b2.test_sanity()
    a2flat = a2.to_ndarray()
    b2flat = b2.to_ndarray()
    npt.assert_array_equal(aflat, a2flat)
    npt.assert_array_equal(bflat, b2flat)
