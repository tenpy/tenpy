#!/usr/bin/env python
"""Benchmark comparison:

new TenPyLight npc -vs.- old TenPy npc -vs.- flat numpy.
"""

import numpy as np

import test_old_np_conserved as old
import tenpy.algorithms.linalg.np_conserved as old_npc
import tenpy.linalg.np_conserved as new_npc
import itertools
import time


def tensordot_timing(verbose=1, rep=1,
                     num_q=0,
                     seed=0,
                     select_frac=1.,
                     size=20,
                     type=np.float,
                     d_ao=1,   # dimension a out
                     d_bo=1,   # dimension b out
                     d_contract=1,  # dimension contracted
                     ):
    took_flat = took_old_npc = took_new_npc = 0.  # adding the times needed
    if d_ao + d_contract <= 0 or d_bo + d_contract <= 0:
        return 0, 0
    # setup of old a, b
    q_ind_a, q_ind_b = old.generate_q_ind_pair(
        d_ao, d_bo, d_contract, size=size, n_qsector=[2] * num_q, rand_var=3, rand_seed=seed)
    axesa = range(d_ao, d_ao + d_contract)
    axesb = range(d_contract)
    # randomize the axes in a and b
    perma = old.rand_permutation(d_ao + d_contract)
    rev_perma = np.argsort(perma)
    permb = old.rand_permutation(d_bo + d_contract)
    rev_permb = np.argsort(permb)
    q_ind_a = [q_ind_a[i] for i in perma]
    q_ind_b = [q_ind_b[i] for i in permb]
    axesa = [rev_perma[i] for i in axesa]
    axesb = [rev_permb[i] for i in axesb]

    shape_a = old_npc.shape_from_q_ind(q_ind_a)
    total_el_a = np.prod(np.array(shape_a))
    shape_b = old_npc.shape_from_q_ind(q_ind_b)
    total_el_b = np.prod(np.array(shape_b))

    a = old_npc.array.from_ndarray(  # TODO: use type?
        np.sqrt(np.arange(total_el_a, dtype=np.float)).reshape(shape_a), q_ind_a)
    b = old_npc.array.from_ndarray(
        1. / np.sqrt(np.arange(
            1, total_el_b + 1, dtype=np.float)).reshape(shape_b), q_ind_b)
    a.ipurge_zeros()
    b.ipurge_zeros()

    if num_q > 0 and select_frac < 1.:
        a_subset = old.rand_distinct_int(0, len(a.dat) - 1, int(len(a.dat) * select_frac))
        b_subset = old.rand_distinct_int(0, len(b.dat) - 1, int(len(b.dat) * select_frac))
        if len(a.q_dat) > 0: a.q_dat = a.q_dat[a_subset, :]
        if len(b.q_dat) > 0: b.q_dat = b.q_dat[b_subset, :]
        a.dat = [a.dat[i] for i in a_subset]
        b.dat = [b.dat[i] for i in b_subset]

    old_a = a
    old_b = b

    # ------ convert old a, b to flat
    flat_a = a.to_ndarray()
    flat_b = b.to_ndarray()

    # ------ convert flat_a, flat_b to new_a, new_b
    chinfo = new_npc.ChargeInfo([1]*num_q)
    a_legs = [new_npc.LegCharge(chinfo, qind, +1) for qind in q_ind_a]
    b_legs = [new_npc.LegCharge(chinfo, qind, +1) for qind in q_ind_b]
    new_a = new_npc.Array.from_ndarray(
        flat_a, chinfo, a_legs, type, a.charge, 0.)
    new_b = new_npc.Array.from_ndarray(
        flat_b, chinfo, b_legs, type, b.charge, 0.)
    new_a.test_sanity()
    new_b.test_sanity()

    # TODO

    # ------ Timing
    t0 = time.time()
    for i in range(rep):
        old_c = old_npc.tensordot(old_a, old_b, axes=(axesa, axesb),
                                  verbose=verbose - 2)
    took_old_npc += time.time() - t0

    t0 = time.time()
    for i in range(rep):
        flat_c = np.tensordot(flat_a, flat_b, axes=[axesa, axesb])[()]
    took_flat += time.time() - t0

    t0 = time.time()
    for i in range(rep):
        new_c = new_npc.tensordot(new_a, new_b, axes=[axesa, axesb])[()]
    took_new_npc += time.time() - t0

    if d_ao + d_bo > 0:
        E_flat_old = np.linalg.norm((old_c.to_ndarray() - flat_c).reshape(-1))
    else:  # d_ao + d_bo == 0
        E_flat_old = abs(old_c - flat_c.item())
    thr = 1.e-12
    if E_flat_old > thr: # or E_flat_new > thr or E_new_old > thr
        print "error flat-old", E_flat_old,
        # print "flat-new", E_flat_new,
        # print "new-old", E_new_old,
        print

    return took_flat / rep, took_old_npc / rep, took_new_npc / rep


def run_tensordot_timing(verbose=0):
    print "------ tensordot_timing ------"
    print "benchmark in  ( total_time/ size**3 [microseconds] )"
    for num_q in range(0, 3):
        print "num_q:", num_q
        for size in range(5, 60, 5):
            total = np.zeros(3.)
            total_flat = total_old = total_new = 0.
            for d_ao, d_bo, d_contract, seed in itertools.product(
                    range(2, 3), range(1, 2), range(2, 3), range(3)):
                kwargs = {
                    'verbose': verbose,
                    'rep': 3,
                    'num_q': num_q,
                    'd_ao': d_ao,
                    'd_bo': d_bo,
                    'd_contract': d_contract,
                    'seed': seed,
                    'size': size,
                    'type': np.float,
                }
                took = tensordot_timing(**kwargs)
                total += np.array(took)
            total /= size**3  # compare scaling vs size**3
            total /= 1.e-6 # microseconds
            print "Size {0:3d}, flat_np, old_npc, new_npc = " \
                "{1: 10.4f} {2: 10.4f} {3: 10.4f}".format(size, *total)
        print "="*80


if __name__ == "__main__":
    run_tensordot_timing()
