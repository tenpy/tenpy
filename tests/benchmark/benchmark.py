"""Perform benchmarks of the `benchmark` function in given modules.

Each of the specified modules needs to implement a function `setup_benchmark(**kwargs)`.
This function may return `data`, which is used in calls to the `benchmark(data)` function.
We benchmark (i.e. measure the execution time of) the latter function.
Calls to the `benchmark` function are repeated a few times for better statistics.
The `setup_benchmark` may use `np.random`, which is initialized with different seeds.

Call this file with arguments, e.g,::
    python benchmark.py -m tensordot_npc tensordot_numpy -l 2 -s 20 -q 1 1
Afterwards, you can plot the produced files::
    python benchmark.py -p tensordot_*_benchmark_*.txt
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

from __future__ import print_function  # (backwards compatibility to python 2)

import timeit
import time
import numpy as np
import sys

fn_template = '{mod_name!s}_benchmark_s_{sectors:d}_l_{legs:d}_mod_q_{mod_q_str}.txt'

sizes_choices = {
    'default': [1, 2, 3, 5, 7, 10, 12] + list(range(15, 50, 5)) + list(range(50, 200, 25)) + \
    list(range(200, 500, 100)) + list(range(500, 3001, 250)),
    'exp' : [2**L for L in range(12)]  # up to 2048
}

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
linestyles = ['-', '--', ':', '-.']


def perform_benchmark(mod_name,
                      sizes=sizes_choices['default'],
                      max_time=0.1,
                      seeds=list(range(1)),
                      repeat_average=1,
                      repeat_bestof=3,
                      **kwargs):
    """Perform a benchmark for a given module and given arguments.

    Parameters
    ----------
    mod_name : str
        The name of a module containing the benchmark.
        Must define the functions ``data = setup_benchmark(size, **kwargs)``,
        which is followed by multiple ``benchmark(data)``, which should be benchmarked.
    sizes : list of int
        The sizes for which benchmarks should be performed. We update ``kwargs['size']`` to the
        different values of this.
    max_time : float
        If the benchmark result is larger than this (or if the setup takes more than a factor 10
        longer than this), skip the following sizes.
    seeds : list of int
        Seeds of the random number generator. We average over the results for the different seeds.
    repeat_average : int
        Get the average time per call of the benchmark function for `repeat_average` repetitions.
    repeat_bestof : int
        Perform `rep_bestof` benchmarks and take the best of the them.
        Reduces noice due to other factors during the benchmark.
    **kwargs :
        Further arguments given to the `setup_benchmark` function.
        Note: is formated to a string with ``repr(kwargs)``. Don't use too complicated arguements!

    Returns
    -------
    times : list of float
        Average time [in seconds] to call the function `benchmark` in the module `mod_name`
        after calling `setup_benchmark` in this module with the given `kwargs`.
        Different entries are for the different `sizes`, averaged over the the random `seeds`.
        Sizes where the benchmark was skipped contain entries ``-1``.
    """
    print("-" * 80)
    print("module ", mod_name)
    namespace = {}
    exec("import {mod_name} as benchmark_mod".format(mod_name=mod_name), namespace, namespace)
    benchmark_mod = namespace['benchmark_mod']
    used_sizes = []
    results = []
    for size in sorted(sizes):
        kwargs_cpy = kwargs.copy()
        kwargs_cpy['size'] = size
        t0 = time.time()
        results_seeds = []
        for seed in seeds:
            np.random.seed(seed)
            setup_code = "import {mod_name!s}\ndata = {mod_name!s}.setup_benchmark(**{kwargs!r})"
            setup_code = setup_code.format(mod_name=mod_name, kwargs=kwargs_cpy)
            timing_code = "{mod_name}.benchmark(data)".format(mod_name=mod_name)
            T = timeit.Timer(timing_code, setup_code)
            res = T.repeat(repeat_bestof, repeat_average)
            results_seeds.append(min(res) / repeat_average)
        used_sizes.append(size)
        results.append(np.mean(results_seeds))
        print("size {size: 4d}: {res:.2e}".format(size=size, res=results[-1]))
        count = repeat_bestof * repeat_average * len(seeds)
        if (results[-1] > max_time or  # benchmark time
                time.time() - t0 >
                count * max_time * 11):  # setup time shouldn't be too much longer
            break
    return used_sizes, results


def save_results(sizes, benchmark_results, filename=fn_template, **kwargs):
    """Save the results into a file with the specified filename."""
    filename = filename.format(mod_q_str='_'.join([str(q) for q in kwargs['mod_q']]), **kwargs)
    header = []
    for kw, arg in kwargs.items():
        header.append(kw + " = " + repr(arg))
    header.append("")
    header.append("size benchmark_time")
    header = '\n'.join(header)
    data = np.stack([sizes, benchmark_results]).T
    np.savetxt(filename, data, fmt=['%d', '%.4e'], header=header)
    print("saved to", filename)
    return filename


def load_results(filename):
    """Load the results saved to a file with `save_results`."""
    # read header
    kwargs = {}
    with open(filename) as f:
        for line in f:
            if not line.startswith("# "):
                raise ValueError("Unexpected format")
            if line.startswith("# size"):
                break
            exec(line.strip()[2:], locals(), kwargs)
    sizes, results = np.loadtxt(filename, unpack=True)
    return sizes, results, kwargs


# plotting stuff


def map_plot_style(styles, style_map, key):
    return style_map.setdefault(key, styles[len(style_map) % len(styles)])


def plot_result(filename, axes, color_map, linestyle_map):
    """Plot the results saved in a given file."""
    sizes, results, kwargs = load_results(filename)
    col = map_plot_style(colors, color_map, kwargs['mod_name'])
    ls = map_plot_style(linestyles, linestyle_map, kwargs['sectors'])
    axes.plot(sizes, results, color=col, linestyle=ls, marker='x')


def plot_many_results(filenames, fn_beg_until="_", fn_end_from="_l_", save=True):
    """Plot files with similar beginning and ending filenames together."""
    import matplotlib.pyplot as plt
    figs = {}
    for fn in filenames:
        fn_beg = fn[:fn.find(fn_beg_until)]
        fn_end = fn[fn.find(fn_end_from):]
        fig_key = fn_beg, fn_end
        if fig_key not in figs:
            fig = plt.figure()
            ax = plt.gca()
            figs[fig_key] = (fig, {}, {})
        fig, color_map, linestyle_map = figs[fig_key]
        plot_result(fn, fig.axes[0], color_map, linestyle_map)
    for fn_key, (fig, color_map, linestyle_map) in figs.items():
        ax = fig.axes[0]
        ax.set_title(fn_key)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("size")
        ax.set_ylabel("wallclock time (s)")
        # add legend
        patches = []
        labels = []
        import matplotlib.patches as mpatches
        for key in sorted(color_map):
            patches.append(mpatches.Patch(color=color_map[key]))
            labels.append(key)
        import matplotlib.lines as mlines
        for key in sorted(linestyle_map):
            patches.append(mlines.Line2D([], [], linestyle=linestyle_map[key], color='k'))
            labels.append("{s:d} sectors".format(s=key))
        ax.legend(patches, labels)
    if save:
        for key, (fig, _, _) in figs.items():
            fn_beg, fn_end = key
            fn = fn_beg + '_plot' + fn_end[:-4] + '.png'
            fig.savefig(fn)
    else:
        plt.show()


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
    parser.add_argument('-m',
                        '--modules',
                        nargs='*',
                        default=None,
                        help='Perform benchmarks for the given modules.')
    parser.add_argument('-t',
                        '--max_time',
                        type=float,
                        default=0.1,
                        help='Maximum time after which we skip larger sizes.')
    parser.add_argument('--sizes',
                        default='default',
                        choices=list(sizes_choices.keys()),
                        help='What sizes to benchmark.')
    parser.add_argument('--bestof',
                        type=int,
                        default=3,
                        help='How often to repeat each benchmark to reduce the noice.')
    parser.add_argument('-p',
                        '--plot',
                        nargs='*',
                        default=None,
                        help='Plot the produced benchmark results (saved in the given files).')
    args = parser.parse_args()
    kwargs = dict(mod_q=args.mod_q,
                  legs=args.legs,
                  sectors=args.sectors,
                  max_time=args.max_time,
                  sizes=sizes_choices[args.sizes],
                  repeat_bestof=args.bestof)
    kwargs["python_version"] = sys.version
    files = []
    if args.modules is not None:
        for mod_name in args.modules:
            if mod_name.endswith(".py"):
                mod_name = mod_name[:-3]
            kwargs2 = kwargs.copy()
            kwargs2['mod_name'] = mod_name
            sizes, results = perform_benchmark(**kwargs2)
            del kwargs2['sizes']
            if len(sizes) > 0:
                fn = save_results(sizes, results, **kwargs2)
                files.append(fn)
    if args.plot is not None:
        plot_many_results(files + args.plot, save=(args.modules is None))
    if args.modules is None and args.plot is None:
        raise ValueError("Specify -m or -p arguments! (Help: call with -h)")
