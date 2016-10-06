"""A collection of tests for tenpy.linalg.charges"""

import tenpy.linalg.charges as charges
import numpy as np
import numpy.testing as npt
import nose.tools as nst
import itertools as it

# charges for comparison, unsorted (*_us) and sorted (*_s)
qflat_us = np.array([-6, -6, -6, -4, -4, -4, 4, 4, -4, -4, -4, -4, -2, -2, -2, -2, -2, -2, -2, -2,
                     -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -2, -2, 0, 0, 0, 0, 2, 2, 2,
                     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 6, 6]).reshape((-1, 1))
qind_us = np.array([[0, 3, -6],
                    [3, 6, -4],
                    [6, 8, 4],
                    [8, 12, -4],
                    [12, 23, -2],
                    [23, 34, 0],
                    [34, 37, -2],
                    [37, 41, 0],
                    [41, 55, 2],
                    [55, 59, 4],
                    [59, 61, 6]])   # yapf: disable
# sorted
qflat_s = np.sort(qflat_us, axis=0)
qind_s = np.array([[ 0,  3, -6],
                   [ 3, 10, -4],
                   [10, 24, -2],
                   [24, 39,  0],
                   [39, 53,  2],
                   [53, 59,  4],
                   [59, 61,  6]])    # yapf: disable

qdict_s = {(-6,): slice(0,  3),
           (-4,): slice(3,  10),
           (-2,): slice(10, 24),
           (0,) : slice(24, 39),
           (2,) : slice(39, 53),
           (4,) : slice(53, 59),
           (6,) : slice(59, 61)}    # yapf: disable

ch_1 = charges.ChargeInfo([1])


def test_ChargeInfo():
    trivial = charges.ChargeInfo()
    trivial.test_sanity()
    print "trivial: ", trivial
    nst.eq_(trivial.qnumber, 0)
    chinfo = charges.ChargeInfo([3, 1], ['some', ''])
    print "nontrivial chinfo: ", chinfo
    nst.eq_(chinfo.qnumber, 2)
    qs = [[0, 2], [2, 0], [5, 3], [-2, -3]]
    is_valid = [True, True, False, False]
    for q, valid in zip(qs, is_valid):
        nst.eq_(chinfo.check_valid(q), valid)
    qs_valid = np.array([chinfo.make_valid(q) for q in qs])
    npt.assert_equal(qs_valid, chinfo.make_valid(qs))


def test__find_row_differences():
    for qflat in [qflat_us, qflat_s]:
        diff = charges._find_row_differences(qflat)
        comp = [0] + [i for i in range(1, len(qflat))
                      if np.any(qflat[i - 1] != qflat[i])] + [len(qflat)]
        npt.assert_equal(diff, comp)


def test_LegCharge():
    lcs = charges.LegCharge.from_qflat(ch_1, qflat_s)
    npt.assert_equal(lcs.qind, qind_s)  # check qflat_s -> qind_s
    npt.assert_equal(lcs.to_qflat(), qflat_s)  # check qind_s -> qflat_s
    lcus = charges.LegCharge.from_qflat(ch_1, qflat_us)
    npt.assert_equal(lcus.qind, qind_us)  # check qflat_us -> qind_us
    npt.assert_equal(lcus.to_qflat(), qflat_us)  # check qind_us -> qflat_us

    lc = charges.LegCharge.from_qdict(ch_1, qdict_s)
    npt.assert_equal(lc.qind, qind_s)  # qdict -> qflat
    npt.assert_equal(lc.to_qdict(), qdict_s)  # qflat -> qdict
    nst.eq_(lcs.is_sorted(), True)
    nst.eq_(lcs.is_blocked(), True)
    nst.eq_(lcus.is_sorted(), False)
    nst.eq_(lcus.is_blocked(), False)

    # test sort & bunch
    lcus_qind = lcus.qind.copy()
    pflat, pqind, lcus_s = lcus.sort(bunch=False)
    lcus_s.test_sanity()
    npt.assert_equal(lcus_qind, lcus.qind)      # don't change the old instance
    npt.assert_equal(lcus_s.qind[:, 2:], lcus.qind[pqind, 2:])     # permutation ok?
    nst.eq_(lcus_s.is_sorted(), True)
    nst.eq_(lcus_s.is_bunched(), False)
    nst.eq_(lcus_s.is_blocked(), False)
    nst.eq_(lcus_s.ind_len, lcus.ind_len)
    nst.eq_(lcus_s.block_number, lcus.block_number)
    idx, lcus_sb = lcus_s.bunch()
    lcus_sb.test_sanity()
    lcus_sb.sorted = False  # to ensure that is_blocked really runs the check
    nst.eq_(lcus_sb.is_sorted(), True)
    nst.eq_(lcus_sb.is_bunched(), True)
    nst.eq_(lcus_sb.is_blocked(), True)
    nst.eq_(lcus_sb.ind_len, lcus.ind_len)


def test_reverse_sort_perm(N=10):
    x = np.random.random(N)
    p = np.arange(N)
    np.random.shuffle(np.arange(N))
    xnew = x[p]
    pinv = charges.reverse_sort_perm(p)
    npt.assert_equal(x, xnew[pinv])
