"""To be used in the `-m` argument of benchmark.py."""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np
import tenpy.linalg.np_conserved as npc

import tenpy.tools.optimization as optimization
import itertools as it


def rand_permutation(n):
    """return a random permutation of length n."""
    perm = list(range(n))
    np.random.shuffle(perm)
    return perm


def rand_distinct_int(a, b, n):
    """returns n distinct integers from a to b inclusive."""
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

    `nqsector` gives the (desired) number of sectors for each of the charges.
    """
    if np.isscalar(n_qsector):
        n_qsector = [n_qsector] * chinfo.qnumber
    n_qsector = np.asarray(n_qsector, dtype=np.intp)
    if n_qsector.shape != (chinfo.qnumber, ):
        raise ValueError
    slices = rand_partitions(0, ind_len, np.prod(n_qsector, dtype=int))
    qs = np.zeros((len(slices) - 1, len(n_qsector)), int)
    q_combos = [a for a in it.product(*[range(-(nq // 2), nq // 2 + 1) for nq in n_qsector])]
    qs = np.array(q_combos)[rand_distinct_int(0, len(q_combos) - 1, len(slices) - 1), :]
    qs = chinfo.make_valid(qs)
    return npc.LegCharge.from_qind(chinfo, slices, qs)


def setup_benchmark(mod_q=[1],
                    sectors=3,
                    size=20,
                    legs=2,
                    select_frac=1.,
                    dtype=np.float64,
                    **kwargs):
    """Returns ``a, b, axes`` for timing of ``npc.tensordot(a, b, axes)``

    Constructed such that leg_contract legs are contracted, with
        a.rank = leg_a_out + leg_contract
        b.rank = leg_b_out + leg_contract
    If `select_frac` < 1, select only the given fraction of blocks compared to what is possible by
    charge requirements.
    """
    chinfo = npc.ChargeInfo(mod_q)
    legs_contr = [gen_random_legcharge_nq(chinfo, size, sectors) for i in range(legs)]
    legs_a = legs_contr + \
        [gen_random_legcharge_nq(chinfo, size, sectors) for i in range(legs)]
    legs_b = [l.conj() for l in legs_contr] + \
        [gen_random_legcharge_nq(chinfo, size, sectors) for i in range(legs)]
    a = npc.Array.from_func(np.random.random, legs_a, dtype, shape_kw='size')
    b = npc.Array.from_func(np.random.random, legs_b, dtype, shape_kw='size')
    a.ipurge_zeros()
    b.ipurge_zeros()
    if chinfo.qnumber > 0 and select_frac < 1.:
        a_bl = a.stored_blocks
        if a_bl > 0:
            a_subset = rand_distinct_int(0, a_bl - 1, max(int(a_bl * select_frac), 1))
            a._qdata = a._qdata[a_subset, :]
            a._data = [a._data[i] for i in a_subset]
        b_bl = a.stored_blocks
        if b_bl > 0:
            b_subset = rand_distinct_int(0, b_bl - 1, max(int(b_bl * select_frac), 1))
            b._qdata = b._qdata[b_subset, :]
            b._data = [b._data[i] for i in b_subset]

    labs = ["l{i:d}".format(i=i) for i in range(2 * legs)]
    a.iset_leg_labels(labs[:a.rank])
    b.iset_leg_labels(labs[:b.rank])
    a.itranspose(rand_permutation(a.rank))
    b.itranspose(rand_permutation(b.rank))
    axes = [a.get_leg_indices(labs[:legs]), b.get_leg_indices(labs[:legs])]
    optimization.set_level(3)
    return a, b, axes


def benchmark(data):
    a, b, axes = data
    npc.tensordot(a, b, axes)
