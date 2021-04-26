"""A collection of tests for tenpy.linalg.charges."""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import tenpy.linalg.charges as charges
import numpy as np
import numpy.testing as npt
import itertools as it
from random_test import gen_random_legcharge

# charges for comparison, unsorted (*_us) and sorted (*_s)
qflat_us = np.array([  #   v  v  <-- note the missing minus below
    -6, -6, -6, -4, -4, -4, 4, 4, -4, -4, -4, -4, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -2, -2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4,
    4, 4, 4, 6, 6
]).reshape((-1, 1))
slices_us = np.array([0, 3, 6, 8, 12, 23, 34, 37, 41, 55, 59, 61])
charges_us = np.array([[-6], [-4], [4], [-4], [-2], [0], [-2], [0], [2], [4], [6]])
# sorted
qflat_s = np.sort(qflat_us, axis=0)
slices_s = np.array([0, 3, 10, 24, 39, 53, 59, 61])
charges_s = np.array([[-6], [-4], [-2], [0], [2], [4], [6]])

qdict_s = {
    (-6, ): slice(0, 3),
    (-4, ): slice(3, 10),
    (-2, ): slice(10, 24),
    (0, ): slice(24, 39),
    (2, ): slice(39, 53),
    (4, ): slice(53, 59),
    (6, ): slice(59, 61)
}

ch_1 = charges.ChargeInfo([1])


def test_ChargeInfo():
    trivial = charges.ChargeInfo()
    trivial.test_sanity()
    print("trivial: ", trivial)
    assert trivial.qnumber == 0
    chinfo = charges.ChargeInfo([3, 1], ['some', ''])
    print("nontrivial chinfo: ", chinfo)
    assert chinfo.qnumber == 2
    qs = [[0, 2], [2, 0], [5, 3], [-2, -3]]
    is_valid = [True, True, False, False]
    for q, expect in zip(qs, is_valid):
        check = chinfo.check_valid(np.array([q], dtype=charges.QTYPE))
        assert check == expect
    qs_valid = np.array([chinfo.make_valid(q) for q in qs])
    npt.assert_equal(qs_valid, chinfo.make_valid(qs))
    chinfo2 = charges.ChargeInfo([3, 1], ['some', ''])
    assert (chinfo2 == chinfo)
    chinfo3 = charges.ChargeInfo([3, 1], ['other', ''])
    assert (chinfo3 != chinfo)


def test__find_row_differences():
    for qflat in [qflat_us, qflat_s]:
        qflat = np.array(qflat, dtype=charges.QTYPE)
        diff = charges._find_row_differences(qflat)
        comp = [0] + [i for i in range(1, len(qflat)) if np.any(qflat[i - 1] != qflat[i])
                      ] + [len(qflat)]
        npt.assert_equal(diff, comp)


def test_LegCharge():
    lcs = charges.LegCharge.from_qflat(ch_1, qflat_s).bunch()[1]
    npt.assert_equal(lcs.charges, charges_s)  # check from_qflat
    npt.assert_equal(lcs.slices, slices_s)  # check from_qflat
    npt.assert_equal(lcs.to_qflat(), qflat_s)  # check to_qflat
    lcus = charges.LegCharge.from_qflat(ch_1, qflat_us).bunch()[1]
    npt.assert_equal(lcus.charges, charges_us)  # check from_qflat
    npt.assert_equal(lcus.slices, slices_us)  # check from_qflat
    npt.assert_equal(lcus.to_qflat(), qflat_us)  # check to_qflat
    lcus_flipped = lcus.flip_charges_qconj()
    lcus_flipped.test_equal(lcus)

    lc = charges.LegCharge.from_qdict(ch_1, qdict_s)
    npt.assert_equal(lc.charges, charges_s)  # check from_qdict
    npt.assert_equal(lc.slices, slices_s)  # check from_dict
    npt.assert_equal(lc.to_qdict(), qdict_s)  # chec to_qdict
    assert lcs.is_sorted() == True
    assert lcs.is_blocked() == True
    assert lcus.is_sorted() == False
    assert lcus.is_blocked() == False

    # test sort & bunch
    lcus_charges = lcus.charges.copy()
    pqind, lcus_s = lcus.sort(bunch=False)
    lcus_s.test_sanity()
    npt.assert_equal(lcus_charges, lcus.charges)  # don't change the old instance
    npt.assert_equal(lcus_s.charges, lcus.charges[pqind])  # permutation returned by sort ok?
    assert lcus_s.is_sorted() == True == lcus_s.sorted
    assert lcus_s.is_bunched() == False == lcus_s.bunched
    assert lcus_s.is_blocked() == False
    assert lcus_s.ind_len == lcus.ind_len
    assert lcus_s.block_number == lcus.block_number
    idx, lcus_sb = lcus.sort(bunch=True)
    assert lcus_sb.is_sorted() == True == lcus_sb.sorted
    assert lcus_sb.is_bunched() == True == lcus_sb.bunched
    assert lcus_sb.is_blocked() == True
    assert lcus_sb.ind_len == lcus.ind_len

    # test get_qindex
    for i in range(lcs.ind_len):
        qidx, idx_in_block = lcs.get_qindex(i)
        assert (lcs.slices[qidx] <= i < lcs.slices[qidx + 1])
        assert (lcs.slices[qidx] + idx_in_block == i)


def test_LegPipe():
    shape = (20, 10, 8)
    legs = [gen_random_legcharge(ch_1, s) for s in shape]
    for sort, bunch in it.product([True, False], repeat=2):
        pipe = charges.LegPipe(legs, sort=sort, bunch=bunch)
        pipe.test_sanity()
        pipe_conj = pipe.conj()
        pipe.test_contractible(pipe.conj())
        pipe_equal = pipe.flip_charges_qconj()
        pipe_equal.test_equal(pipe)

        assert (pipe.ind_len == np.prod(shape))
        print(pipe.q_map)
        # test pipe._map_incoming_qind
        qind_inc = pipe.q_map[:, 3:].copy()  # all possible qindices
        np.random.shuffle(qind_inc)  # different order to make the test non-trivial
        qmap_ind = pipe._map_incoming_qind(qind_inc)
        for i in range(len(qind_inc)):
            npt.assert_equal(pipe.q_map[qmap_ind[i], 3:], qind_inc[i])
            size = np.prod([l.slices[j + 1] - l.slices[j] for l, j in zip(legs, qind_inc[i])])
            assert size == pipe.q_map[qmap_ind[i], 1] - pipe.q_map[qmap_ind[i], 0]
        # pipe.map_incoming_flat is tested by test_np_conserved.


def test__sliced_copy():
    x = np.random.random([20, 10, 4])  # c-contiguous!
    x_cpy = x.copy()
    y = np.random.random([5, 6, 7])
    y_cpy = y.copy()
    shape = np.array([4, 3, 2], dtype=np.intp)
    z = 2. * np.ones(shape)
    x_beg = np.array([3, 7, 1], dtype=np.intp)
    y_beg = np.array([1, 0, 4], dtype=np.intp)
    z_beg = np.array([0, 0, 0], dtype=np.intp)
    charges._sliced_copy(z, z_beg, x, x_beg, shape)
    npt.assert_equal(x, x_cpy)
    assert (not np.any(z == 2.))
    npt.assert_equal(x[3:7, 7:10, 1:3], z)
    charges._sliced_copy(y, y_beg, x, x_beg, shape)
    npt.assert_equal(y[1:5, 0:3, 4:6], z)
    charges._sliced_copy(y, y_beg, y_cpy, y_beg, shape)
    npt.assert_equal(y, y_cpy)
