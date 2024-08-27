"""Perform a single run of the "benchmark" function of a given module.

Call this file with arguments, e.g:     python singlerun.py -m tensordot_npc tensordot_numpy -l 2
-s 20 -S 50 -q 1 1
"""

# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np
import time


def single_run(mod_name, repeat=1, seed=0, **kwargs):
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
        Seed of the random number generator with this number to enhance reproducibility
    **kwargs :
        Further arguments given to the `setup_benchmark` function.
        Note: is formatted to a string with ``repr(kwargs)``. Don't use too complicated arguments!
    """
    kwargs['mod_name'] = mod_name
    np.random.seed(seed)
    setup_code = "import {mod_name!s}\ndata = {mod_name!s}.setup_benchmark(**{kwargs!r})"
    setup_code = setup_code.format(mod_name=mod_name, kwargs=kwargs)
    namespace = {}
    exec(setup_code, namespace, namespace)
    timing_code = "{mod_name}.benchmark(data)".format(mod_name=mod_name)
    if repeat > 1:
        timing_code = "for _ in range({repeat:d}): ".format(repeat=repeat) + timing_code
    t0 = time.time()
    exec(timing_code, namespace, namespace)
    t1 = time.time()
    print("finished single run for module", mod_name, "after {0:.2e} seconds".format(t1 - t0))


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
                        help='Perform profiling for the given modules.')
    args = parser.parse_args()
    kwargs = dict(mod_q=args.mod_q, legs=args.legs, sectors=args.sectors, size=args.size)
    for mod_name in args.modules:
        if mod_name.endswith(".py"):
            mod_name = mod_name[:-3]
        kwargs['mod_name'] = mod_name
        single_run(**kwargs)
