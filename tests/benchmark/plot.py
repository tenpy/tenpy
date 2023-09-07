"""Utility function to plot the output of the three `run_benchmark_*.sh` scripts."""

from benchmark import fn_template, load_results

all_packages = [  # (module_suffix, backend_str, label)
    ('tenpy', 'abelian_numpy', 'v2.0 [CPU]'),
    ('tenpy', 'abelian_gpu', 'v2.0 [GPU]'),
    ('torch', 'abelian_gpu', 'pytorch [GPU]'),
    ('numpy', 'abelian_numpy', 'numpy'),
    ('old', 'abelian_numpy', 'v0.10'),
    ('old_uncompiled', 'abelian_numpy', 'v0.10 [uncompiled]'),
]
colormap = {
    'v2.0 [CPU]': 'C0',
    'v2.0 [GPU]': 'C1',
    'pytorch [GPU]': 'C2',
    'numpy': 'C3',
    'v0.10': 'C4',
    'v0.10 [uncompiled]': 'C5',
    'default': 'C7',
}
lsmap = {
    'v2.0 [CPU]': '-',
    'v2.0 [GPU]': '-',
    'pytorch [GPU]': '--',
    'numpy': '--',
    'v0.10': ':',
    'v0.10 [uncompiled]': ':',
    'default': '-',
}
markermap = {
    'v2.0 [CPU]': 'x',
    'v2.0 [GPU]': '+',
    'pytorch [GPU]': '+',
    'numpy': 'x',
    'v0.10': 'x',
    'v0.10 [uncompiled]': 'x',
    'default': 'x',
}

default_q_l_s = [
    ('U1_U1_Z2', 1, 2), ('U1_U1_Z2', 1, 5), ('U1_U1_Z2', 1, 20),
    ('U1', 1, 2), ('U1', 1, 5), ('U1', 1, 20),
    ('Z2', 1, 2),
    ('no_symmetry', 1, 5),  # s=5 is the default in the kwargs, so it is used when not specified
    ('U1_U1_Z2', 2, 2), ('U1_U1_Z2', 2, 5), ('U1_U1_Z2', 2, 20),
    ('U1', 2, 2), ('U1', 2, 5), ('U1', 2, 20),
    ('Z2', 2, 2),
    ('no_symmetry', 2, 5),  # s=5 is the default in the kwargs, so it is used when not specified
]


def plot_results(module_prefix='tdot',
                 packages=all_packages,
                 q_l_s=default_q_l_s):
    """Make plots for a given module_prefix, e.g. `tdot` or `svd`.

    For the given different packages, we plot different lines in the same plot.
    For the given combinations of q, l, s, make separate plots.
    Additionally, for every l, make a summary plot with subplots for every combination of q, s.

    Parameters
    ----------
    module_prefix : str
        e.g. from: tdot, svd, qr, combine, split, tebd_infinite, qrtebd_infinite, dmrg_infinite
    packages : list of (str, str, str)
        List of tuples (module_suffix, backend_str, label_for_leged)
    q_l_s : list of (str, int, int)

    """
    import matplotlib.pyplot as plt

    all_l = []
    all_q = []
    all_s = []
    missing_files = []
    for q, l, s in q_l_s:
        if q not in all_q:
            all_q.append(q)
        if l not in all_l:
            all_l.append(l)
        if s not in all_s:
            all_s.append(s)

    for l in all_l:
        # Setup summary figure
        l_fig, l_axs = plt.subplots(len(all_q), len(all_s), figsize=(10, 10), sharex=True, sharey=True)
        l_fig.suptitle(f'{module_prefix} with {l} legs')
        for ax in l_axs.flat:
            ax.set_xscale('log')
            ax.set_yscale('log')
        for ax in l_axs[-1, :]:
            ax.set_ylabel('Wallclock time (s)')
        for ax in l_axs[:, 0]:
            ax.set_xlabel('Size')
        for n_s, s in enumerate(all_s):
            l_axs[0, n_s].set_title(f'{s} sectors')

        # iterate over q, s
        for n_q, q in enumerate(all_q):
            q_str = 'No Symm.' if q == 'no_symmetry' else q.replace('_', ' * ')
            l_axs[n_q, 0].text(0.05, 0.9, q_str, transform=l_axs[n_q, 0].transAxes)

            for n_s, s in enumerate(all_s):
                if (q, l, s) not in q_l_s:
                    continue
                
                fig, ax = plt.subplots()
                fig.suptitle(f'{module_prefix} with {l} legs, {q} conservation and {s} sectors.')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlabel('Size')
                ax.set_ylabel('Wallclock time (s)')

                for module_suffix, backend_str, label in packages:
                    filename = fn_template.format(mod_name=f'{module_prefix}_{module_suffix}',
                                                  backend_str=backend_str, symm_str=q, legs=l,
                                                  sectors=s)
                    try:
                        sizes, times, kwargs = load_results(filename)
                    except FileNotFoundError:
                        missing_files.append(filename)
                        continue
                    is_gpu = 'GPU' in label
                    col = colormap.get(label, colormap['default'])
                    ls = lsmap.get(label, lsmap['default'])
                    marker = markermap.get(label, markermap['default'])
                    ax.plot(sizes, times, color=col, linestyle=ls, marker=marker, label=label)
                    l_axs[n_q, n_s].plot(sizes, times, color=col, linestyle=ls, marker=marker, label=label)

                ax.legend()
                fig.savefig(f'plots/{module_prefix}_plot_q_{q}_l_{l}_s_{s}.png')

        # Finish up summary fig
        # use last non-skipped ax for labels
        handles, labels = ax.get_legend_handles_labels()
        l_axs[-1, -1].legend(handles, labels)
        
        l_fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        l_fig.savefig(f'plots/{module_prefix}_plot_summary_l_{l}.png', bbox_inches='tight')
            
    if missing_files:
        num = 5
        print(f'{len(missing_files)} files were not found. Showing the first {num}::')
        for f in missing_files[:num]:
            print(f)


if __name__ == "__main__":
    # ``python benchmark.py --help`` prints a summary of the options
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-m',
        '--module',
        type=str,
        default='tdot',
        help='The module prefix for which plots should be made'
    )
    args = parser.parse_args()
    plot_results(args.module)
    