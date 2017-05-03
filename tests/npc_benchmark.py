#!/usr/bin/env python
"""Benchmark comparison:

New TenPyLight npc --vs.-- old TenPy npc --vs.-- flat basic numpy.
Run ``python npc_benchmark.py --help`` for usage options.
"""

from __future__ import division

import timeit
import cProfile
import time

import numpy as np
import tenpy.linalg.np_conserved as npc
import tenpy
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


def setup_npc(mod_q=[1],
              n_qsectors=3,
              size=20,
              leg_a_out=2,
              leg_b_out=2,
              leg_contract=2,
              select_frac=1.,
              dtype=np.float,
              seed=123):
    """Returns ``a, b, axes`` for timing of ``npc.tensordot(a, b, axes)``

    Constructed such that leg_contract legs are contracted, with
        a.rank = leg_a_out + leg_contract
        b.rank = leg_b_out + leg_contract
    If `select_frac` < 1, select only the given fraction of blocks compared to what is possible by
    charge requirements.
    """
    chinfo = npc.ChargeInfo(mod_q)
    np.random.seed(seed)
    legs_contr = [gen_random_legcharge_nq(chinfo, size, n_qsectors) for i in range(leg_contract)]
    legs_a = legs_contr + \
        [gen_random_legcharge_nq(chinfo, size, n_qsectors) for i in range(leg_a_out)]
    legs_b = [l.conj() for l in legs_contr] + \
        [gen_random_legcharge_nq(chinfo, size, n_qsectors) for i in range(leg_b_out)]
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

    labs = ["l{i:d}".format(i=i) for i in range(max(leg_a_out, leg_b_out) + leg_contract)]
    a.set_leg_labels(labs[:a.rank])
    b.set_leg_labels(labs[:b.rank])
    a.itranspose(rand_permutation(a.rank))
    b.itranspose(rand_permutation(b.rank))
    axes = [labs[:leg_contract]] * 2
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


def convert_leg2oldqind(leg):
    """return `qind` to describe a new :class:`npc.LegCharge` in an old `npc_old.array`"""
    return np.hstack([leg.slices[:-1].reshape(-1, 1), leg.slices[1:].reshape(-1, 1), leg.charges])


def convert_old_npc(a, b, axes):
    "given the result of setup_npc, convert for ``npc_old.tensordot(a, b, axes)``"
    axes_a, axes_b = axes
    axes_a = a.get_leg_indices(axes_a)
    axes_b = b.get_leg_indices(axes_b)

    mod_q = a.chinfo.mod.copy()
    a_qind = [convert_leg2oldqind(l) for l in a.legs]
    a_qconj = [l.qconj for l in a.legs]
    a2 = old_npc.zeros(a_qind, a.dtype, a_qconj, a.qtotal, mod_q)
    a2.sorted = a._qdata_sorted
    a2.dat = a._data[:]
    a2.q_dat = np.array(a._qdata, dtype=np.uint)
    b_qind = [convert_leg2oldqind(l) for l in b.legs]
    b_qconj = [l.qconj for l in b.legs]
    b2 = old_npc.zeros(b_qind, b.dtype, b_qconj, b.qtotal, mod_q)
    b2.sorted = b._qdata_sorted
    b2.dat = b._data[:]
    b2.q_dat = np.array(b._qdata, dtype=np.uint)
    a2.check_sanity()
    b2.check_sanity()
    return a2, b2, (axes_a, axes_b)


def tensordot_timing(do_flat=True,
                     do_old_npc=True,
                     rep_bestof=3,
                     rep_tdot=3,
                     seed_range=range(3),
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
    time_npc = min(T.repeat(rep_bestof, rep_tdot)) / rep_tdot

    if do_flat:
        npc_setup = setup_code.format(
            mod=__name__, setup_fct='setup_flat', kwargs=kwargs, tensordot='np.tensordot')
        T = timeit.Timer(timing_code, npc_setup)
        time_flat = min(T.repeat(rep_bestof, rep_tdot)) / rep_tdot

    if has_old_npc and do_old_npc:
        npc_setup = setup_code.format(
            mod=__name__, setup_fct='setup_old_npc', kwargs=kwargs, tensordot='old_npc.tensordot')
        T = timeit.Timer(timing_code, npc_setup)
        time_old_npc = min(T.repeat(rep_bestof, rep_tdot)) / rep_tdot

    return time_npc, time_old_npc, time_flat


def tensordot_profile(fn=None, dmax=None, **kwargs):
    """profile the tensordot"""
    a, b, axes = setup_npc(**kwargs)
    print "profile tensordot(a, b, axes)"
    print "a: {a!r}\nb: {b!r}\naxes {axes!r}".format(a=a, b=b, axes=axes)
    print "sparse stats:"
    print a.sparse_stats()
    print b.sparse_stats()
    cProfile.runctx("npc.tensordot(a, b, axes)", globals(), locals(), fn)


def skip_timing(dmax, size=20, mod_q=[1], n_qsectors=3, **kwargs):
    """use dmax to determine whether to skip the timing for the given parameters"""
    legs = [kwargs.get(k, 2) for k in ['leg_a_out', 'leg_b_out', 'leg_contract']]
    flat_shapes = [size**d for d in legs]  # flat requires to perform matrix (M,N) dot (N,K)
    do_flat = (np.prod(flat_shapes) <= dmax**3) and (len(mod_q) == 0)
    # estimate the number of charge sectors on each leg
    sectors = min(size, n_qsectors**len(mod_q))
    sectors_shape = [sectors**d for d in legs]
    blocks = np.prod(sectors_shape)
    block_dot_flops = np.prod([f / s for f, s in zip(flat_shapes, sectors_shape)])
    # roughly the number of flops for a `dot` of blocks
    matmul = sectors_shape[0] * sectors_shape[2]  # roughly the number of `dot` of blocks,
    # assuming full charge blocking (fails for legs > 1)
    skip_all = blocks > dmax**3 or matmul * block_dot_flops > dmax**3
    return skip_all, do_flat


def run_tensordot_timing(sizes=range(5, 80, 5),
                         num_qs=range(3),
                         seeds=range(5),
                         dmax=2000,
                         **kwargs):
    """call `tensordot_timing` for different `sizes` and `num_qs`.
    """
    print "------ tensordot_timing ------"
    data = {}
    data['seeds'] = seeds
    data['sizes'] = sizes
    data['num_qs'] = num_qs
    all_timings = []
    for num_q in num_qs:
        mod_q = [1] * num_q
        print "num_q:", num_q
        num_q_timings = []
        for size in sizes:
            print size  # just to notice that we're still running
            timing = np.zeros(3)  # average over seeds
            kwargs.update(mod_q=mod_q, size=size)
            skip_all, do_flat = skip_timing(dmax=dmax, **kwargs)
            if not skip_all:
                for seed in seeds:
                    kwargs.update(seed=seed)
                    res = tensordot_timing(do_flat, True, 3, 3, **kwargs)
                    timing += res
            num_q_timings.append(timing / len(seeds))
        print "-" * 80
        all_timings.append(num_q_timings)
    all_timings = np.array(all_timings, dtype=np.float)
    data['timings'] = all_timings
    return data


def run_save(fn_t='npc_benchmark_timeit_{legs}_{n_qsectors:d}.pkl', dmax=2000, **kwargs):
    """get a fairly exhaustive collection of timings for different n_qsectros and leg_*....
    """
    sizes_all = [3, 5, 8, 10, 12] + range(15, 50, 5) + range(50, 200, 25) + \
        range(200, 500, 100) + range(500, 3001, 250)
    for n_qsectors, legs in [(2, 1), (2, 2), (5, 1), (5, 2), (5, 3), (20, 1)]:
        print "+" * 100
        print "n_qsectors = {nq:d}, legs ={legs:d}".format(nq=n_qsectors, legs=legs)
        print "sizes = ", sizes_all
        kwargs.update(n_qsectors=n_qsectors, leg_a_out=legs, leg_b_out=legs, leg_contract=legs)
        data = run_tensordot_timing(sizes=sizes_all, dmax=dmax, **kwargs)
        data['kwargs'] = kwargs.copy()
        data['version'] = tenpy.version.full_version
        fn = fn_t.format(n_qsectors=n_qsectors, legs=legs)
        save(data, fn)


def print_timing_res(data):
    num_qs = data['num_qs']
    sizes = data['sizes']
    timed = data['timings']
    print "=" * 80
    if 'version' in data:
        print "version", data['version']
    # print "kwargs:", data['kwargs']
    print "num_q size      flat       old       new   new-old"
    row = "{qn: 5d}{s: 5d}{flat: 10.6f}{old: 10.6f}{new: 10.6f}{new_old: 10.6f}"
    for qnumber, timed_qn in zip(num_qs, timed):
        for size, timed_size in zip(sizes, timed_qn):
            new, old, flat = timed_size
            print row.format(qn=qnumber, s=size, flat=flat, new=new, old=old, new_old=new - old)
    print "=" * 80


def plot_timing_res(data, fn=None):
    """plot the timing results.
    markers = num_q
    colors = method
    """
    import pylab as pl
    num_qs = data['num_qs']
    sizes = data['sizes']
    timed = data['timings']
    timed = np.ma.array(timed, mask=(timed == 0.))
    markers = ['o', 's', 'v', 'x']  # num_q
    pl.figure(figsize=(10, 7))
    for qn, t_qn, m in zip(num_qs, timed, markers):
        for t, lab, col in [(t_qn[:, 0], 'npc', 'r'), (t_qn[:, 1], 'old_npc', 'g'),
                            (t_qn[:, 2], 'numpy', 'b')]:
            lab = "qnumber {qn:d}, {lab}".format(lab=lab, qn=qn)
            if np.any(t != 0.):  # only if we have data
                pl.plot(sizes, t, col + m + '-', markersize=8, label=lab)
    if 'kwargs' in data:
        pl.title(
            ', '.join([k + "=" + str(data['kwargs'][k]) for k in sorted(data['kwargs'].keys())]))
    pl.xlabel('size (of each leg)')
    pl.ylabel('total time [s]')
    pl.loglog()
    pl.legend(loc='upper left')
    if fn is None:
        pl.show()
    else:
        pl.savefig(fn)
    pl.close()


def load(fn):
    import pickle
    print "loading ", fn
    with open(fn, 'r') as f:
        return pickle.load(f)


def save(data, fn):
    import pickle
    print "save to ", fn
    with open(fn, 'w') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    # ``python npc_benchmark.py --help`` prints a summary of the options
    import argparse
    import os.path
    parser = argparse.ArgumentParser(description="""obtianing benchmarks of np_conserved.
                                     Without any arguments, just run a quick test.""")

    parser.add_argument(
        '-t',
        '--timing',
        action='store_true',
        help="""run the function run_timing_save() to perform an extensive timing for
        scaling analysis. Duration with a single CPU may vary from 10 minutes to 2 hours...""")
    parser.add_argument(
        '-p',
        '--plot',
        action='store_true',
        help='print and plot the timing results saved in given files.')
    parser.add_argument(
        '--profile', action='store_true', help='profile tensordot. Save to file, if one is given.')
    parser.add_argument(
        '--dmax',
        type=int,
        default=2000,
        help="""maximum dimension of matrices to multiply for timing""")
    parser.add_argument(
        '--n_qsectors', type=int, default=3, help="""number of charge sectors per leg""")
    parser.add_argument(
        '--legs',
        type=int,
        default=2,
        help="""number of legs to keep for each tensor and to contract""")
    parser.add_argument(
        'files', nargs='*', help='Specify filenames used depending on other options.')
    args = parser.parse_args()
    kwargs = dict(
        n_qsectors=args.n_qsectors,
        leg_a_out=args.legs,
        leg_b_out=args.legs,
        leg_contract=args.legs,
        dmax=args.dmax)
    if args.timing:
        t0 = time.time()
        run_save(**kwargs)
        print "=" * 80
        print "finished timing after", time.time() - t0, "seconds in total"
    if args.plot:
        for fn in args.files:
            data = load(fn)
            print_timing_res(data)
            plot_timing_res(data, os.path.splitext(fn)[0] + '.png')
    if args.profile:
        fn = None if len(args.files) == 0 else args.files[0]
        tensordot_profile(fn, mod_q=[1], size=50, seed=2, **kwargs)
    if not any([args.timing, args.plot, args.profile]):
        data = run_tensordot_timing(**kwargs)
        print_timing_res(data)
        plot_timing_res(data)
