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


def gen_random_legcharge(n, chinfo):
    """returns a random legcharge with index len `n`"""
    qflat = []
    for mod in chinfo.mod:
        if mod > 1:
            qflat.append(np.asarray(np.random.randint(0, mod, size=n), dtype=npc.QDTYPE))
        else:
            r = max(3, n//3)
            qflat.append(np.asarray(np.random.randint(-r, r, size=n), dtype=npc.QDTYPE))
    qflat = np.array(qflat, dtype=npc.QDTYPE).T
    qconj = np.random.randint(0, 1, 1) * 2 - 1
    return npc.LegCharge.from_qflat(chinfo, qflat, qconj)


def random_Array(shape, chinfo, func=np.random.random, shape_kw='size', qtotal=None, sort=True):
    """generates a random npc.Array of given shape with random legcharges and entries."""
    legs = [gen_random_legcharge(s, chinfo) for s in shape]
    a = npc.Array.from_func(func, chinfo, legs, qtotal=qtotal, shape_kw=shape_kw)
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
    # from_ndfunc
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
    a_sb.sort_qdata()
    npt.assert_equal(a_sb.to_ndarray(), arr_s)  # sort_qdata
    # and for larger random array
    a = random_Array((20, 15, 10), chinfo2, sort=False)
    p_flat, a_s = a.sort_legcharge(True, False)
    arr_s = a.to_ndarray()[np.ix_(*p_flat)]  # what a_s should be
    npt.assert_equal(a_s.to_ndarray(), arr_s)  # sort without bunch
    _, a_sb = a_s.sort_legcharge(False, True)
    npt.assert_equal(a_sb.to_ndarray(), arr_s)  # bunch after sort
    npt.assert_equal(a_sb._qdata_sorted, False)
    a_sb.sort_qdata()
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
