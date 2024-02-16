"""Perform profiling of the `benchmark` function in given modules.

Call this file with arguments, e.g::

    python profiling.py -m tensordot_npc tensordot_numpy -l 2 -s 20 -S 50 -q 1 1

Afterwards, you can print the produced statistics::

    python profiling.py -p tensordot_*_profile_*.prof --sort time --limit 10
"""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np
import cProfile
import pstats
import sys
import time

fn_template = '{mod_name!s}_profile_S_{size:d}_s_{sectors:d}_l_{legs:d}_mod_q_{mod_q_str}.prof'


def perform_profiling(mod_name, repeat=1, seed=0, filename=fn_template, **kwargs):
    """Run profiling of the `benchmark` function in the given module.

    Parameters
    ----------
    mod_name : str
        The name of a module containing the benchmark.
        Must define the functions ``data = setup_benchmark(size, **kwargs)``,
        which is followed by multiple ``benchmark(data)``, which should be benchmarked.
    repeat : int
        Repeat the `benchmark` function to be profiled that many times.
    seed : int
        Seed of the random number generator with this number to enhance reproducability
    filename : str
        Template for the filename.
    **kwargs :
        Further arguments given to the `setup_benchmark` function.
        Note: is formated to a string with ``repr(kwargs)``. Don't use too complicated arguements!
    """
    kwargs['mod_name'] = mod_name
    filename = filename.format(mod_q_str='_'.join([str(q) for q in kwargs['mod_q']]), **kwargs)
    np.random.seed(seed)
    setup_code = "import {mod_name!s}\ndata = {mod_name!s}.setup_benchmark(**{kwargs!r})"
    setup_code = setup_code.format(mod_name=mod_name, kwargs=kwargs)
    namespace = {}
    exec(setup_code, namespace, namespace)
    timing_code = "{mod_name}.benchmark(data)".format(mod_name=mod_name)
    if repeat > 1:
        timing_code = "for _ in range({repeat:d}): ".format(repeat=repeat) + timing_code
    if sys.version_info > (3, 3):
        prof = cProfile.Profile(time.perf_counter)
    else:
        prof = cProfile.Profile()
    prof.runctx(timing_code, namespace, namespace)
    prof.dump_stats(filename)

    #  cProfile.runctx(timing_code, namespace, namespace, filename)
    print("saved profiling to", filename)
    return filename


def print_profiling(filename, sort=[], limit=[], callees=None, callers=None):
    stats = pstats.Stats(filename)
    stats.strip_dirs()
    stats.sort_stats(*sort)
    if callees is not None:
        print("callees", callees)
        stats.print_callees(callees)
    elif callers is not None:
        print("callers", callers)
        stats.print_callers(callers)
    else:
        stats.print_stats(*limit)
    return stats


if __name__ == "__main__":
    # ``python benchmark.py --help`` prints a summary of the options
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-q',
        '--mod_q',
        type=int,
        nargs='*',
        default=[],
        help="Nature of the charge, ``Charges.mod``. The length determines the number of charges.")
    parser.add_argument('-l',
                        '--legs',
                        type=int,
                        default=2,
                        help="Number of legs to be contracted.")
    parser.add_argument('-s',
                        '--sectors',
                        type=int,
                        default=5,
                        help="(Maximal) number of sectors in each leg.")
    parser.add_argument('-S', '--size', type=int, default=50, help="Size of each leg.")
    parser.add_argument('-m',
                        '--modules',
                        nargs='+',
                        default=None,
                        help='Perform profiling for the given modules.')
    parser.add_argument('-p',
                        '--print_stats',
                        nargs='*',
                        default=None,
                        help='Print the produced profiling results (saved in the given files).')
    parser.add_argument('--sort',
                        default=['cumtime'],
                        choices=['cumtime', 'time', 'ncalls', 'name', 'filename', 'tottime'],
                        nargs='*',
                        help="Defines sorting for printing.")
    parser.add_argument(
        '--limit',
        default=[50],
        nargs='*',
        help="Limit for printing the stats. You can enter an in to limit the number of lines or" \
             " a regex to match the function name."
    )
    parser.add_argument('--callees',
                        default=None,
                        help="Print the functions called from inside the given function")
    parser.add_argument('--callers',
                        default=None,
                        help="Print the functions calling the given function")
    args = parser.parse_args()
    kwargs = dict(mod_q=args.mod_q, legs=args.legs, sectors=args.sectors, size=args.size)
    files = []
    if args.modules is not None:
        for mod_name in args.modules:
            if mod_name.endswith(".py"):
                mod_name = mod_name[:-3]
            kwargs['mod_name'] = mod_name
            fn = perform_profiling(**kwargs)
            files.append(fn)
    if args.print_stats is not None:
        limits = list(args.limit)
        for i in range(len(limits)):
            try:
                limits[i] = int(limits[i])
            except:
                pass
        for fn in files + args.print_stats:
            print_profiling(fn, args.sort, limits, args.callees, args.callers)
    if args.modules is None and args.print_stats is None:
        raise ValueError("Specify -m or -p arguments! (Help: call with -h)")
