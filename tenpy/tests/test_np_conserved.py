"""A collection of tests for tenpy.linalg.np_conserved"""
from __future__ import division

import tenpy.linalg.np_conserved as npc
import numpy as np
import numpy.testing as npt
import nose.tools as nst
import itertools as it

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

# fix the random number generator such that tests are reproducible
np.random.seed(3141592)  # (it should work for any seed)

from test_charges import gen_random_legcharge


def random_Array(shape, chinfo, func=np.random.random, shape_kw='size', qtotal=None, sort=True):
    """generates a random npc.Array of given shape with random legcharges and entries."""
    legs = [gen_random_legcharge(s, chinfo) for s in shape]
    a = npc.Array.from_func(func, chinfo, legs, qtotal=qtotal, shape_kw=shape_kw)
    a.set_leg_labels([chr(i+ord('a')) for i in range(a.rank)])
    if sort:
        _, a = a.sort_legcharge(True, True)  # increase the probability for larger blocks
    return a


def project_multiple_axes(flat_array, perms, axes):
    for p, a in it.izip(perms, axes):
        idx = [slice(None)]*flat_array.ndim
        idx[a] = p
        flat_array = flat_array[tuple(idx)]
    return flat_array


# ------- test functions -------------------

def test_npc_Array_conversion():
    # trivial
    a = npc.Array.from_ndarray_trivial(arr)
    npt.assert_equal(a.to_ndarray(), arr)
    # non-trivial charges
    a = npc.Array.from_ndarray(arr, chinfo, [lc, lc.conj()])
    npt.assert_equal(a._get_block_charge([0, 2]), [-2, 0])
    npt.assert_equal(a.to_ndarray(), arr)
    npt.assert_equal(a.qtotal, np.array([0, 0], npc.QDTYPE))
    # check non-zero total charge
    a = npc.Array.from_ndarray(arr, chinfo, [lc, lc_add.conj()])
    npt.assert_equal(a.qtotal, np.array([-1, 0], npc.QDTYPE))
    npt.assert_equal(a.to_ndarray(), arr)
    a.gauge_total_charge(1)
    npt.assert_equal(a.qtotal, np.array([0, 0], npc.QDTYPE))
    # check type conversion
    a_clx = a.astype(np.complex128)
    nst.eq_(a_clx.dtype, np.complex128)
    npt.assert_equal(a_clx.to_ndarray(), arr.astype(np.complex128))
    # from_func
    a = npc.Array.from_func(np.ones, chinfo, [lc, lc.conj()])
    a.test_sanity()
    aflat = np.zeros((5, 5))
    for ind in [(0, 0), (1, 1), (1, 4), (4, 1), (2, 2), (3, 3), (4, 4)]:
        aflat[ind] = 1.
    npt.assert_equal(a.to_ndarray(), aflat)
    # random array
    a = random_Array((20, 15, 10), chinfo2, sort=False)
    a.test_sanity()


def test_npc_Array_sort():
    a = npc.Array.from_ndarray(arr, chinfo, [lc, lc.conj()])
    p_flat, a_s = a.sort_legcharge(True, False)
    npt.assert_equal(p_flat[0], [3, 0, 2, 1, 4])
    arr_s = arr[np.ix_(*p_flat)]  # what a_s should be
    npt.assert_equal(a_s.to_ndarray(), arr_s)  # sort without bunch
    _, a_sb = a_s.sort_legcharge(False, True)
    npt.assert_equal(a_sb.to_ndarray(), arr_s)  # bunch after sort
    npt.assert_equal(a_sb._qdata_sorted, False)
    a_sb.isort_qdata()
    npt.assert_equal(a_sb.to_ndarray(), arr_s)  # sort_qdata
    # and for larger random array
    a = random_Array((20, 15, 10), chinfo2, sort=False)
    p_flat, a_s = a.sort_legcharge(True, False)
    arr_s = a.to_ndarray()[np.ix_(*p_flat)]  # what a_s should be
    npt.assert_equal(a_s.to_ndarray(), arr_s)  # sort without bunch
    _, a_sb = a_s.sort_legcharge(False, True)
    npt.assert_equal(a_sb.to_ndarray(), arr_s)  # bunch after sort
    npt.assert_equal(a_sb._qdata_sorted, False)
    a_sb.isort_qdata()
    npt.assert_equal(a_sb.to_ndarray(), arr_s)  # sort_qdata


def test_npc_Array_labels():
    a = npc.Array.from_ndarray(arr, chinfo, [lc, lc.conj()])
    for t in [('x', None), (None, 'y'), ('x', 'y')]:
        a.set_leg_labels(t)
        nst.eq_(a.get_leg_labels(), t)
        axes = (0, 1, 1, 0, 1, 0)
        axes_l = list(axes)  # replace with labels, where available
        for i, l in enumerate(axes[:4]):
            if t[l] is not None:
                axes_l[i] = t[l]
        nst.eq_(tuple(a.get_leg_indices(axes_l)), axes)
    nst.eq_(a.get_leg_index(-1), 1)     # negative indices


def test_npc_Array_project():
    a = npc.Array.from_ndarray(arr, chinfo, [lc, lc.conj()])
    p1 = np.array([True, True, False, True, True])
    p2 = np.array([0, 1, 3])

    b = a.copy(True)
    b.iproject([p1, p2], (0, 1))
    b.test_sanity()
    bflat = project_multiple_axes(a.to_ndarray(), [p1, p2], (0, 1))
    npt.assert_equal(b.to_ndarray(), bflat)
    # and again for a being blocked before: can we split the blocks
    _, a = a.sort_legcharge()
    b = a.copy(True)
    b.iproject([p1, p2], (0, 1))
    b.test_sanity()
    bflat = project_multiple_axes(a.to_ndarray(), [p1, p2], (0, 1))
    npt.assert_equal(b.to_ndarray(), bflat)


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
    for tr in [None, [2, 1, 0], (1, 2, 0), (0, 2, 1)]:
        a = random_Array((20, 15, 10), chinfo)
        atr = a.transpose(tr)
        atr.test_sanity()
        npt.assert_equal(atr.to_ndarray(), a.to_ndarray().transpose(tr))


def test_npc_Array_itemacces():
    a = npc.Array.from_ndarray(arr, chinfo, [lc, lc.conj()])
    aflat = a.to_ndarray().copy()
    for i, j in it.product(xrange(5), xrange(5)):  # access all elements
        nst.eq_(a[i, j], aflat[i, j])
    for i, j in [(0, 0), (2, 2), (1, 4), (4, 1), (3, 3), (4, 4)]:  # sets also emtpy blocks
        val = np.random.rand()
        aflat[i, j] = val
        a[i, j] = val
    npt.assert_equal(a.to_ndarray(), aflat)
    # again for array with larger blocks
    a = random_Array((10, 10), chinfo3)
    aflat = a.to_ndarray().copy()
    for i, j in it.product(xrange(10), xrange(10)):  # access all elements
        nst.eq_(a[i, j], aflat[i, j])
    # take_slice
    a = random_Array((20, 10, 5), chinfo3)
    aflat = a.to_ndarray().copy()
    for idx, axes in [(0, 0), (4, 1), ([3, -2], [-1, 0])]:
        a_sl = a.take_slice(idx, axes)
        a_sl.test_sanity()
        sl = [slice(None)]*a.rank
        try:
            for i, ax in zip(idx, axes):
                sl[ax] = i
        except:
            sl[axes] = idx
        sl = tuple(sl)
        npt.assert_equal(a_sl.to_ndarray(), aflat[sl])
        npt.assert_equal(a[sl].to_ndarray(), aflat[sl])
    # advanced indexing with slices and projection/mask
    # NOTE: for blat[idx] = aflat[idx] to work, non-slices may not be separated by slices.
    check_idx = [(2, Ellipsis, 1),
                 (slice(None), 3, np.array([True, True, False, True, False])),
                 (slice(3, 4), np.array([2, 4, 5]), slice(1, 4, 2)),
                 (slice(4, 2, -1), 2, np.array([3, 1, 4, 2]))]  # yapf: disable
    for idx in check_idx:
        print "take slice for ", idx
        b = a[idx]
        b.test_sanity()
        bflat = aflat[idx]  # idx may only contain a single array for this to work
        npt.assert_equal(b.to_ndarray(), bflat)
    # create another random array to check copying with c[inds] = a[inds]
    b = npc.Array.from_func(np.random.random, a.chinfo, a.legs, a.dtype, a.qtotal, shape_kw='size')
    # remove half of the blocks to check copying to empty sites as well
    keep = (np.arange(b.stored_blocks) % 2 == 0)
    b._data = [d for d, k in zip(b._data, keep) if k]
    b._qdata = b._qdata[keep]
    bflat = b.to_ndarray().copy()
    aflat = a.to_ndarray().copy()
    for idx in check_idx:
        print "copy slice for ", idx
        b[idx] = a[idx]
        b.test_sanity()
        bflat[idx] = aflat[idx]  # idx may only contain a single array
        npt.assert_equal(b.to_ndarray(), bflat)


def test_npc_Array_reshape():
    a = random_Array((20, 15, 10), chinfo, sort=False)
    aflat = a.to_ndarray()
    for comb_legs, transpose in [([[1]], [0, 1, 2]),
                                 ([[1], [2]], [0, 1, 2]),
                                 ([[0], [1], [2]], [0, 1, 2]),
                                 ([[2, 0]], [1, 2, 0]),
                                 ([[2, 0, 1]], [2, 0, 1])]:
        print 'combine legs', comb_legs
        acomb = a.combine_legs(comb_legs)  # just sorts second leg
        print "=> labels: ", acomb.get_leg_labels()
        acomb.test_sanity()
        asplit = acomb.split_legs()
        asplit.test_sanity()
        npt.assert_equal(asplit.to_ndarray(), aflat.transpose(transpose))
    b = random_Array((10, 1, 5, 1), chinfo3, sort=True)
    bflat = b.to_ndarray()
    bs = b.squeeze()
    bs.test_sanity()
    npt.assert_equal(bs.to_ndarray(), np.squeeze(bflat))
    bs.test_sanity()
    if b.stored_blocks > 0:
        # find a index with non-zero entry
        idx = tuple([l.qind[i, 0] for l, i in zip(b.legs, b._qdata[0])])
    else:
        idx = tuple([0]*b.rank)
    nst.eq_(b[idx[0], :, idx[2], :].squeeze(), bflat[idx])


def test_npc_Array_scale_axis():
    a = random_Array((5, 15, 10), chinfo3, sort=True)
    aflat = a.to_ndarray()
    s0 = np.random.random((a.shape[0], 1, 1))
    s1 = np.random.random((1, a.shape[1], 1))
    s2 = np.random.random((1, 1, a.shape[2]))
    b = a.scale_axis(s0.flatten(), 0)
    b.test_sanity()
    npt.assert_equal(b.to_ndarray(), aflat*s0)
    a.iscale_axis(s1.flatten(), 1)
    a.test_sanity()
    aflat = aflat * s1
    npt.assert_equal(a.to_ndarray(), aflat)
    c = a.scale_axis(s2.flatten(), -1)
    c.test_sanity()
    npt.assert_equal(c.to_ndarray(), aflat*s2)


def test_npc_Array_conj():
    a = random_Array((15, 10), chinfo3, sort=True)
    a.set_leg_labels(['a', 'b*'])
    aflat = a.to_ndarray()
    b = a.conj()
    b.test_sanity()
    npt.assert_equal(a.to_ndarray(), aflat)
    npt.assert_equal(b.to_ndarray(), aflat.conj())
    a.iconj()
    npt.assert_equal(a.to_ndarray(), aflat.conj())
    a.test_sanity()
    print a.get_leg_labels()
    nst.eq_(a._conj_leg_label('(a*.(b.c*).(d*.e))'), '(a.(b*.c).(d.e*))')

def test_npc_Array_norm():
    a = random_Array((15, 10), chinfo3, sort=True)
    aflat = a.to_ndarray().flatten()
    for ord in [np.inf, -np.inf, 0, 1, -1, 2, -2, 3.]:
        print "ord = ", ord
        nst.eq_(a.norm(), np.linalg.norm(aflat))


def test_npc_Array_ops():
    a = random_Array((15, 10), chinfo3, sort=True)
    b = npc.Array.from_func(np.random.random, a.chinfo, a.legs, qtotal=a.qtotal, shape_kw='size')
    s = 3.12
    aflat = a.to_ndarray()
    bflat = b.to_ndarray()
    import operator as Op
    # addition / subtraction
    for op in [Op.add, Op.sub, Op.iadd, Op.isub]:
        print op.__name__
        a2 = op(a, b)
        a.test_sanity()
        a2.test_sanity()
        aflat2 = op(aflat, bflat)
        npt.assert_equal(a.to_ndarray(), aflat)
        npt.assert_equal(a2.to_ndarray(), aflat2)
    npt.assert_equal(b.to_ndarray(), bflat)  # should not have been modified...
    # multiplication
    for op in [Op.mul, Op.imul]:
        print op.__name__
        a2 = op(a, s)
        a.test_sanity()
        a2.test_sanity()
        aflat2 = op(aflat, s)
        npt.assert_equal(a.to_ndarray(), aflat)
        npt.assert_equal(a2.to_ndarray(), aflat2)
    # reversed multiplication
    print "rmul"
    a2 = s * a
    a.test_sanity()
    a2.test_sanity()
    aflat2 = s * aflat
    npt.assert_equal(a.to_ndarray(), aflat)
    npt.assert_equal(a2.to_ndarray(), aflat2)
    # division
    for op in [Op.truediv, Op.itruediv]:
        # may differ by machine precision due to rounding errors
        print op.__name__
        a2 = op(a, s)
        a.test_sanity()
        a2.test_sanity()
        aflat2 = op(aflat, s)
        eps = np.finfo(a.dtype).eps
        assert(np.max(np.abs(a.to_ndarray() - aflat)) < eps)
        assert(np.max(np.abs(a.to_ndarray() - aflat)) < eps)


if __name__ == "__main__":
    test_npc_Array_conversion()
    test_npc_Array_sort()
    test_npc_Array_labels()
    test_npc_Array_project()
    test_npc_Array_permute()
    test_npc_Array_transpose()
    test_npc_Array_itemacces()
    test_npc_Array_reshape()
    test_npc_Array_scale_axis()
    test_npc_Array_conj()
    test_npc_Array_norm()
    test_npc_Array_ops()
