#!/usr/bin/env python

import numpy as np
import itertools as it
import tenpy.linalg.charges as charges
import tenpy.linalg.np_conserved as npc

# fix the random number generator such that tests are reproducible
np.random.seed(3141592)  # (it should work for any seed)

__all__ = ['rand_permutation', 'rand_distinct_int', 'rand_partitions', 'gen_random_legcharge_nq',
           'gen_random_legcharge', 'random_Array']


def rand_permutation(n):
    """return a random permutation of length n"""
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
    if n_qsector.shape != (chinfo.qnumber, ):
        raise ValueError
    slices = rand_partitions(0, ind_len, np.prod(n_qsector, dtype=int))
    qs = np.zeros((len(slices) - 1, len(n_qsector)), int)
    q_combos = [a for a in it.product(* [range(-(nq // 2), nq // 2 + 1) for nq in n_qsector])]
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
    qflat = np.array(qflat, dtype=charges.QTYPE).T.reshape(ind_len, chinfo.qnumber)
    qconj = np.random.randint(0, 1, 1) * 2 - 1
    return charges.LegCharge.from_qflat(chinfo, qflat, qconj).bunch()[1]


def random_Array(shape, chinfo, func=np.random.random, shape_kw='size', qtotal=None, sort=True):
    """generates a random npc.Array of given shape with random legcharges and entries."""
    legs = [gen_random_legcharge(chinfo, s) for s in shape]
    a = npc.Array.from_func(func, legs, qtotal=qtotal, shape_kw=shape_kw)
    a.set_leg_labels([chr(i + ord('a')) for i in range(a.rank)])
    if sort:
        _, a = a.sort_legcharge(True, True)  # increase the probability for larger blocks
    return a
