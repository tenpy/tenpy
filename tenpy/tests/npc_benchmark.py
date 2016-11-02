#!/usr/bin/env python
"""Benchmark comparison:

new TenPyLight npc -vs.- old TenPy npc -vs.- flat numpy.
"""
import timeit

import numpy as np
import tenpy.linalg.np_conserved as npc
try:
    import algorithms.linalg.np_conserved as old_npc
    has_old_npc = True
except:
    has_old_npc = False

from test_charges import gen_random_legcharge_nq, rand_permutation, rand_distinct_int

# setup code for timeit
setup_code = """
import {mod}
a, b, axes = {mod}.{setup_fct}(**{kwargs!r})
tdot = {mod}.{tensordot}
"""  # .format(mod=__name__, setup_fct='setup_npc', kwargs=kwargs, tensordot='npc.tensordot')

# the code for the actual timing.
timing_code = "tdot(a, b, axes)"


def setup_npc(mod_q=[1], n_qsectors=3, size=20, dim_a_out=2, dim_b_out=2, dim_contract=2,
              select_frac=1., dtype=np.float, seed=123):
    """Returns ``a, b, axes`` for timing of ``npc.tensordot(a, b, axes)``

    Constructed such that dim_contract legs are contracted, with
        a.rank = dim_a_out + dim_contract
        b.rank = dim_b_out + dim_contract
    If `select_frac` < 1, select only the given fraction of blocks compared to what is possible by
    charge requirements.
    """
    chinfo = npc.ChargeInfo(mod_q)
    np.random.seed(seed)
    legs_contr = [gen_random_legcharge_nq(chinfo, size, n_qsectors) for i in range(dim_contract)]
    legs_a = legs_contr + \
        [gen_random_legcharge_nq(chinfo, size, n_qsectors) for i in range(dim_a_out)]
    legs_b = [l.conj() for l in legs_contr] + \
        [gen_random_legcharge_nq(chinfo, size, n_qsectors) for i in range(dim_b_out)]
    a = npc.Array.from_func(np.random.random, chinfo, legs_a, dtype, shape_kw='size')
    b = npc.Array.from_func(np.random.random, chinfo, legs_b, dtype, shape_kw='size')
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

    labs = ["l{i:d}".format(i=i) for i in range(max(dim_a_out, dim_b_out) + dim_contract)]
    a.set_leg_labels(labs[:a.rank])
    b.set_leg_labels(labs[:b.rank])
    a.itranspose(rand_permutation(a.rank))
    b.itranspose(rand_permutation(b.rank))
    axes = [labs[:dim_contract]]*2
    return a, b, axes


def setup_flat(**kwargs):
    a, b, axes = setup_npc(**kwargs)
    return convert_flat(a, b, axes)


def setup_old_npc(**kwargs):
    a, b, axes = setup_npc(**kwargs)
    return convert_old_npc(a, b, axes)


def convert_flat(a, b, axes):
    "convert result of `setup_npc()` to numpy arrays for `np.tensordot(a, b, axes)`"
    axes_a, axes_b = axes
    axes_a = a.get_leg_indices(axes_a)
    axes_b = b.get_leg_indices(axes_b)
    return a.to_ndarray(), b.to_ndarray(), (axes_a, axes_b)


def convert_old_npc(a, b, axes):
    "given the result of setup_npc, convert for ``npc_old.tensordot(a, b, axes)``"
    axes_a, axes_b = axes
    axes_a = a.get_leg_indices(axes_a)
    axes_b = b.get_leg_indices(axes_b)

    mod_q = a.chinfo.mod.copy()
    a_qind = [l.qind for l in a.legs]
    a_qconj = [l.qconj for l in a.legs]
    a2 = old_npc.zeros(a_qind, a.dtype, a_qconj, a.qtotal, mod_q)
    a2.sorted = a._qdata_sorted
    a2.dat = a._data[:]
    a2.q_dat = a._qdata.copy()
    b_qind = [l.qind for l in b.legs]
    b_qconj = [l.qconj for l in b.legs]
    b2 = old_npc.zeros(b_qind, b.dtype, b_qconj, b.qtotal, mod_q)
    b2.sorted = b._qdata_sorted
    b2.dat = b._data[:]
    b2.q_dat = b._qdata.copy()
    return a2, b2, (axes_a, axes_b)


def tensordot_timing(do_flat=True, do_old_npc=True,
                     rep_bestof=3, rep_tdot=3, seed_range=range(3),
                     **kwargs):
    """run tensordot timing for given kwargs of ``setup_npc``.

    Always time `npc`.
    If `do_numpy`, time flat ``np.tensordot``.
    If `do_old_npc`, time ``old_npc.tensordot``.

    Returns (time_npc, time_old_npc, time_flat)
    In units of seconds per execution of a single tensordot.
    """
    time_flat = time_old_npc = 0.

    npc_setup = setup_code.format(
        mod=__name__, setup_fct='setup_npc', kwargs=kwargs, tensordot='npc.tensordot')
    T = timeit.Timer(timing_code, npc_setup)
    time_npc = min(T.repeat(rep_bestof, rep_tdot))/rep_tdot

    if do_flat:
        npc_setup = setup_code.format(
            mod=__name__, setup_fct='setup_flat', kwargs=kwargs, tensordot='np.tensordot')
        T = timeit.Timer(timing_code, npc_setup)
        time_flat = min(T.repeat(rep_bestof, rep_tdot))/rep_tdot

    if has_old_npc and do_old_npc:
        npc_setup = setup_code.format(
            mod=__name__, setup_fct='setup_old_npc', kwargs=kwargs, tensordot='old_npc.tensordot')
        T = timeit.Timer(timing_code, npc_setup)
        time_old_npc = min(T.repeat(rep_bestof, rep_tdot))/rep_tdot

    return time_npc, time_old_npc, time_flat


def run_tensordot_timing(sizes=range(5, 60, 5),
                         num_qs=range(3),
                         seeds=range(5),
                         **kwargs):
    print "------ tensordot_timing ------"
    data = {}
    data['seeds'] = seeds
    data['sizes'] = sizes
    data['num_qs'] = num_qs
    all_timings = []
    for num_q in num_qs:
        mod_q = [1]*num_q
        print "num_q:", num_q
        num_q_timings = []
        for size in sizes:
            print size  # just to notice that we're still running
            dims = [kwargs.get(k, 2) for k in ['dim_a_out', 'dim_b_out', 'dim_contract']]
            mat_shape = [size**d for d in dims]  # flat requires to perform matrix (M,N) dot (N,K)
            do_flat = (np.prod(mat_shape) <= 70**3) and (num_q == num_qs[0])
            timing = np.zeros(3)  # average over seeds
            for seed in seeds:
                kwargs.update(mod_q=mod_q, size=size, seed=seed)
                res = tensordot_timing(do_flat, True, 3, 3, **kwargs)
                timing += res
            num_q_timings.append(timing/len(seeds))
        print "-"*80
        all_timings.append(num_q_timings)
    data['timings'] = np.array(all_timings, dtype=np.float)
    return data


def run_save(fn_t='npc_benchmark_res_{dim}_{n_qsectors:d}.pkl'):
    """get a collection of different timings...."""
    import pickle
    sizes = range(5, 50, 5) + range(50, 200, 25)
    for n_qsectors, dim in [(2, 1), (2, 2), (5, 1), (5, 2), (5, 3)]:
        print "+"*100
        kwargs = dict(n_qsectors=n_qsectors, dim_a_out=dim, dim_b_out=dim, dim_contract=dim)
        data = run_tensordot_timing(sizes=sizes, **kwargs)
        fn = fn_t.format(n_qsectors=n_qsectors, dim=dim)
        print "save to ", fn
        with open(fn, 'w') as f:
            pickle.dump(data, f)


def print_timing_res(data):
    num_qs = data['num_qs']
    sizes = data['sizes']
    timed = data['timings']
    print "="*80
    print "qnum| size |     flat|      old|      new|  new-old|"
    row = "{qn: 4d}{s: 5d}{flat: 10.4f}{old: 10.4f}{new: 10.4f}{new_old: 10.4f}|"
    for qnumber, timed_qn in zip(num_qs, timed):
        for size, timed_size in zip(sizes, timed_qn):
            new, old, flat = timed_size
            print row.format(qn=qnumber, s=size, flat=flat, new=new, old=old,
                             new_old=new-old)
    print "="*80


def plot_timing_res(data, fn=None):
    """plot the timing results.
    markers = num_q
    colors = method
    """
    import pylab as pl
    num_qs = data['num_qs']
    sizes = data['sizes']
    timed = data['timings']
    markers = ['o', 's', 'v', 'x']  # num_q
    pl.figure(figsize=(10, 7))
    for qn, t_qn, m in zip(num_qs, timed, markers):
        for t, lab, col in [(t_qn[:, 0], 'npc', 'r'),
                            (t_qn[:, 1], 'old_npc', 'g'),
                            (t_qn[:, 2], 'numpy', 'b'),
                            # (t_qn[:, 2]-t_qn[:, 1], 'diff old_npc-npc', 'k')
                            ]:
            pl.plot(sizes, t, col+m+'-', markersize=8, label=lab)
    pl.xlabel('size')
    pl.ylabel('total time')
    pl.loglog()
    pl.legend(loc='upper left')
    if fn is None:
        pl.show()
    else:
        pl.save(fn+'.png')
    pl.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        import pickle
        if sys.argv[1] == 'run_save':
            run_save()
            sys.exit(0)
        for fn in sys.argv[1:]:
            print "loading results from ", fn+'.pklz'
            with open(fn+'.pkl', 'r') as f:
                timing_res = pickle.load(f)  # load the timing results from the file
            print_timing_res(timing_res)
            plot_timing_res(timing_res, fn)
        sys.exit(0)
    timing_res = run_tensordot_timing()
    print_timing_res(timing_res)
    plot_timing_res(timing_res)
