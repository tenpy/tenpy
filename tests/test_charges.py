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
slices_us = np.array([0, 3, 6, 8, 12, 23, 34, 37, 41, 55, 59, 61])
charges_us = np.array([[-6], [-4], [4], [-4], [-2], [0], [-2], [0], [2], [4], [6]])
# sorted
qflat_s = np.sort(qflat_us, axis=0)
slices_s = np.array([0, 3, 10, 24, 39, 53, 59, 61])
charges_s = np.array([[-6], [-4], [-2], [0], [2], [4], [6]])

qdict_s = {(-6,): slice(0, 3),
           (-4,): slice(3, 10),
           (-2,): slice(10, 24),
           (0,): slice(24, 39),
           (2,): slice(39, 53),
           (4,): slice(53, 59),
           (6,): slice(59, 61)}

ch_1 = charges.ChargeInfo([1])

# fix the random number generator such that tests are reproducible
np.random.seed(3141592)  # (it should work for any seed)


def rand_permutation(n):
    perm = range(n)
    np.random.shuffle(perm)
    return perm


def rand_distinct_int(a, b, n):
    """ returns n distinct integers from a to b inclusive"""
    if n < 0:
        raise ValueError
    if n > b - a + 1:
        raise ValueError
    return np.sort((np.random.random_integers(a, b - n + 1, size=n))) + np.arange(n)


def rand_partitions(a, b, n):
    """return [a] + `cuts` + [b], where `cuts` are ``n-1`` (strictly ordered) values inbetween."""
    if b - a <= n:
        return np.array(range(a, b + 1))
    else:
        return np.concatenate(([a], rand_distinct_int(a + 1, b - 1, n - 1), [b]))


def gen_random_legcharge_nq(chinfo, ind_len, n_qsector):
    """return a random (unsorted) LegCharge with a given number of charge sectors.
    `nqsector` gives the (desired) number of sectors for each of the charges."""
    if np.isscalar(n_qsector):
        n_qsector = [n_qsector] * chinfo.qnumber
    n_qsector = np.asarray(n_qsector, dtype=np.intp)
    if n_qsector.shape != (chinfo.qnumber,):
        raise ValueError
    slices = rand_partitions(0, ind_len, np.prod(n_qsector, dtype=int))
    qs = np.zeros((len(slices) - 1, len(n_qsector)), int)
    q_combos = [a for a in it.product(*[range(-(nq // 2), nq // 2 + 1) for nq in n_qsector])]
    qs = np.array(q_combos)[rand_distinct_int(0, len(q_combos) - 1, len(slices) - 1), :]
    qs = chinfo.make_valid(qs)
    return charges.LegCharge.from_qind(chinfo, slices, qs)


def gen_random_legcharge(chinfo, ind_len):
    """returns a random (unsorted) legcharge with index_len `n`."""
    qflat = []
    for mod in chinfo.mod:
        if mod > 1:
            qflat.append(np.asarray(np.random.randint(0, mod, size=ind_len)))
        else:
            r = max(3, ind_len // 3)
            qflat.append(np.asarray(np.random.randint(-r, r, size=ind_len)))
    qflat = np.array(qflat, dtype=chinfo.qtype).T.reshape(ind_len, chinfo.qnumber)
    qconj = np.random.randint(0, 1, 1) * 2 - 1
    return charges.LegCharge.from_qflat(chinfo, qflat, qconj).bunch()[1]


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
    lcs = charges.LegCharge.from_qflat(ch_1, qflat_s).bunch()[1]
    npt.assert_equal(lcs.charges, charges_s)  # check from_qflat
    npt.assert_equal(lcs.slices, slices_s)  # check from_qflat
    npt.assert_equal(lcs.to_qflat(), qflat_s)  # check to_qflat
    lcus = charges.LegCharge.from_qflat(ch_1, qflat_us).bunch()[1]
    npt.assert_equal(lcus.charges, charges_us)  # check from_qflat
    npt.assert_equal(lcus.slices, slices_us)  # check from_qflat
    npt.assert_equal(lcus.to_qflat(), qflat_us)  # check to_qflat

    lc = charges.LegCharge.from_qdict(ch_1, qdict_s)
    npt.assert_equal(lc.charges, charges_s)  # check from_qdict
    npt.assert_equal(lc.slices, slices_s)  # check from_dict
    npt.assert_equal(lc.to_qdict(), qdict_s)  # chec to_qdict
    nst.eq_(lcs.is_sorted(), True)
    nst.eq_(lcs.is_blocked(), True)
    nst.eq_(lcus.is_sorted(), False)
    nst.eq_(lcus.is_blocked(), False)

    # test sort & bunch
    lcus_charges = lcus.charges.copy()
    pqind, lcus_s = lcus.sort(bunch=False)
    lcus_s.test_sanity()
    npt.assert_equal(lcus_charges, lcus.charges)  # don't change the old instance
    npt.assert_equal(lcus_s.charges, lcus.charges[pqind])  # permutation returned by sort ok?
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

    # test get_qindex
    for i in xrange(lcs.ind_len):
        qidx, idx_in_block = lcs.get_qindex(i)
        assert (lcs.slices[qidx] <= i < lcs.slices[qidx+1])
        assert (lcs.slices[qidx] + idx_in_block == i)


def test_LegPipe():
    shape = (20, 10, 8)
    legs = [gen_random_legcharge(ch_1, s) for s in shape]
    for sort, bunch in it.product([True, False], repeat=2):
        pipe = charges.LegPipe(legs, sort=sort, bunch=bunch)
        pipe.test_sanity()
        assert (pipe.ind_len == np.prod(shape))
        print pipe.q_map
        # test pipe.map_incoming_qind
        qind_inc = pipe.q_map[:, 2:-1].copy()  # all possible qindices
        np.random.shuffle(qind_inc)  # different order to make the test non-trivial
        qmap_ind = pipe._map_incoming_qind(qind_inc)
        for i in range(len(qind_inc)):
            npt.assert_equal(pipe.q_map[qmap_ind[i], 2:-1], qind_inc[i])
            size = np.prod([l.slices[j+1] - l.slices[j] for l, j in zip(legs, qind_inc[i])])
            nst.eq_(size, pipe.q_map[qmap_ind[i], 1] - pipe.q_map[qmap_ind[i], 0])


if __name__ == "__main__":
    test_ChargeInfo()
    test__find_row_differences()
    test_LegCharge()
    test_LegPipe()
