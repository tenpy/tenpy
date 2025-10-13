"""An example determining the phase diagram of the 2D transverse field Ising model."""
# Copyright (C) TeNPy Developers, Apache license
import pickle
import numpy as np
import os
import argparse
#
import tenpy


def main():
    """Main function that is executed if this script is called."""
    tenpy.show_config()
    
    # arg-parsing
    parser = argparse.ArgumentParser(
        description='Runs a phase diagrams sweep for the TFIM, i.e. DMRG for a grid of values g.'
    )
    parser.add_argument(
        '--plot', metavar='FOLDER', type=str, default=None,
        help='Plot results in the given folder instead of running the simulation'
    )
    parser.add_argument(
        '-o', type=str, default=None,
        help='Folder for output. Defaults to a subfolder of playground in the repo root.'
    )
    parser.add_argument(
        '--conserve', type=str, default='None', help='What should be conserved'
    )
    parser.add_argument(
        '--chi', type=int, default=8, help='Bond dimension'
    )
    parser.add_argument(
        '--Ly', type=int, default=4, help='Cylinder circumference'
    )
    parser.add_argument(
        '--num_g_points', type=int, default=101,
        help='Resolution of the phase diagram. Recommend round number plus 1.'
    )
    args = parser.parse_args()

    if args.plot:
        make_plot(folder=args.plot)
        return
    
    outfolder = args.o
    # default outfolder
    if outfolder is None:
        # use /path/to/repo_root/playground/tfi_cylinder
        package_folder = os.path.dirname(tenpy.__file__)  # parent folder of tenpy/__init__.py
        repo_root = os.path.abspath(os.path.join(package_folder, os.pardir))
        playground = os.path.join(repo_root, 'playground')
        assert os.path.exists(playground), 'Need to clone the github repo for default outfolder behavior.'
        outfolder = os.path.join(playground, 'tfi_cylinder')
        if not os.path.exists(outfolder):
            os.mkdir(outfolder)
    
    g_list = np.linspace(2, 4, args.num_g_points, endpoint=True)
    logfile = os.path.join(outfolder, f'tfi_2D_conserve_{args.conserve}_Ly_{args.Ly}_chi_{args.chi}.log')
    file = os.path.join(outfolder, f'tfi_2D_conserve_{args.conserve}_Ly_{args.Ly}_chi_{args.chi}.pkl')
    tenpy.setup_logging(filename=logfile, to_stdout='ERROR', to_file='INFO')
    
    res = sweep_phase_diagram(g_list, conserve=args.conserve, chi=args.chi, Ly=args.Ly)
    
    with open(file, 'wb') as f:
        pickle.dump(res, f)
    print(f'Wrote results to {file}')


def sweep_phase_diagram(g_list, conserve, chi: int, Ly: int):
    """Perform a sweep through the phase diagram.

    Do DMRG at each g point, than use the resulting state as an initial guess for the next g.

    Parameters
    ----------
    g_list : iterable of float
        A grid of g values to simulate at
    conserve : {'best', 'None'}
        What symmetry to conserve, if any
    chi : int
        The maximum bond dimension.
    Ly : int
        The cylinder circumference

    Returns
    -------
    A dictionary with the parameters repeated and with observables.
    """
    psi = None
    g_list=g_list
    energy_list = []
    entropies = []
    mag_z_list = []
    mag_x_list = []
    corr_xx_list = []
    corr_length_list = []

    for g in g_list:
        print(f'Optimizing at {g=}...')
        # re-use the psi from the previous g as initial guess
        psi, energy, entropy, mag_z, mag_x, corr_xx, corr_length = single_run(
            g, Ly, conserve=conserve, chi_max=chi, psi=psi
        )
        energy_list.append(energy)
        entropies.append(entropy)
        mag_z_list.append(mag_z)
        mag_x_list.append(mag_x)
        corr_xx_list.append(corr_xx)
        corr_length_list.append(corr_length)

    return dict(
        g_list=g_list, chi=chi, Ly=Ly, conserve=conserve,
        energy_list=energy_list, entropies=entropies, mag_z_list=mag_z_list,
        mag_x_list=mag_x_list, corr_xx_list=corr_xx_list, corr_length_list=corr_length_list,
    )


def single_run(g: float, Ly: int, conserve, chi_max: int, psi=None):
    """Perform a single DMRG run, at a single point in the phase diagram.

    Parameters
    ----------
    g : float
        The model parameter.
    conserve : {'best', 'None'}
        What symmetry to conserve, if any
    chi_max : int
        The maximum bond dimension.
    Ly : int
        The cylinder circumference
    psi : MPS, optional
        An initial guess

    Returns
    -------
    psi : MPS
        The groundstate approximation
    energy : float
        Its energy
    entropy : float
        Its bipartite von Neumann entanglement entropy for a cut between unit cells.
    mag_z, mag_x : float
        magnetization in z and x directions respectively, averaged over a unit cell.
    corr_xx : float
        Correlation function over a horizontal distance of 10 unit cells
    correlation_length : float
        The correlation length of the MPS in units of horizontal lattice sites.
    """
    # 2) Initialize the model
    # For a full list of options see
    # https://tenpy.readthedocs.io/en/latest/reference/tenpy.models.spins.SpinModel.html
    model_params = dict(
        lattice='Square',  # select the lattice from predefined classes in tenpy.models.lattice
        bc_y='cylinder',  # y boundary conditions: infinite cylinder
        bc_MPS='infinite',  # use iMPS, fixes x boundary conditions
        J=1,
        g=g,
        Lx=2,  # size in x direction *of a unit cell*, system is infinite in x direction
        Ly=Ly,  # size in y direction, i.e. circumference of the cylinder
        conserve=conserve,  # 'best' or None
    )
    model = tenpy.TFIModel(model_params)

    # 3) Initialize an MPS (initial guess). Note that this selects the charge sector.
    if psi is None:
        psi = tenpy.MPS.from_lat_product_state(model.lat, [[['up'], ['down']]])

    # 4) Initialize / Configure the DMRG engine
    # For a full list of options see
    # https://tenpy.readthedocs.io/en/latest/reference/tenpy.algorithms.dmrg.TwoSiteDMRGEngine.html
    dmrg_params = dict(
        mixer=True,  # enable the mixer
        trunc_params=dict(
            chi_max=chi_max,  # maximum bond dimension
            svd_min=1e-10,  # threshold for discarding small singular values
        ),
        combine=True,
        min_sweeps=10,
    )
    engine = tenpy.TwoSiteDMRGEngine(psi, model, dmrg_params)

    # 5) Run DMRG
    energy, psi = engine.run()

    # 6) Extract observables
    mag_z = np.average(psi.expectation_value('Sigmaz'))
    mag_x = np.average(psi.expectation_value('Sigmax'))
    corr_xx = psi.correlation_function('Sigmax', 'Sigmax', [0], [10 * model.lat.N_sites]).item()
    # See note in MPS.correlation_length for the units and why we need to divide
    correlation_length = psi.correlation_length() / model.lat.N_sites_per_ring
    entropy = psi.entanglement_entropy()[0]

    return psi, energy, entropy, mag_z, mag_x, corr_xx, correlation_length


def make_plot(folder):
    """Make the production plots.

    Expect multiple ``.pkl`` files generated by :func:`sweep_phase_diagram` in the `folder`.
    """
    print()
    
    results_None = {}
    results_best = {}
    for fn in os.listdir(folder):
        if not str(fn).startswith('tfi_2D'):
            continue
        if not str(fn).endswith('.pkl'):
            continue
        _, rest = fn.split('tfi_2D_conserve_')
        conserve, rest = rest.split('_Ly_')
        Ly, rest = rest.split('_chi_')
        chi, _ = rest.split('.pkl')
        Ly = int(Ly)
        chi = int(chi)
        with open(os.path.join(folder, fn), 'rb') as f:
            res = pickle.load(f)
        if conserve == 'best':
            results_best[Ly, chi] = res
        else:
            assert conserve == 'None'
            results_None[Ly, chi] = res
    
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    linewidth = 5.90666
    mpl.rcParams.update({'font.size': 10})
    mpl.rcParams.update({'legend.fontsize': 8})
    mpl.rcParams.update({'legend.title_fontsize': 8})

    for conserve, results in zip(['best', 'None'], [results_best, results_None]):
        _g_lists = [res['g_list'] for res in results.values()]
        g_list = _g_lists[0]
        for other in _g_lists[1:]:
            assert np.allclose(other, g_list)
        
        fig_width = linewidth
        aspect = .7
        fig, axs = plt.subplots(2, 2, figsize=(fig_width, aspect * fig_width), sharex=True)
        for ax in axs[1]:
            ax.set_xlabel('$g/J$')
            ax.set_xlim(min(g_list), max(g_list))

        # define line styles
        min_chi = 2
        max_chi = 500
        plot_styles = {
            (Ly, chi): dict(
                label=f'${Ly},~{chi}$',
                color=mpl.colormaps['Reds' if Ly == 8 else 'Blues'](
                    np.log(chi / min_chi) / np.log(max_chi / min_chi)
                ),
                ls={20: '-', 50: '--', 200: '-.'}[chi],
                lw={4: 1.5, 8: 2}[Ly]
            )
            for Ly, chi in results
        }  # plot_styles[Ly, chi] == kwargs_for_plot_function

        sorted_keys = [(Ly, chi) for Ly, chi in sorted(results.keys()) if Ly != 6]

        # subplot: <Z> magnetization
        axs[0, 0].set_ylabel(r'$\langle \sigma^z \rangle$')
        for key in sorted_keys:
            axs[0, 0].plot(g_list, results[key]['mag_z_list'], **plot_styles[key])

        # subplot: S_vN entropy
        axs[0, 1].set_ylabel(r'$S_{\text{vN}} ~/~ \mathrm{log} 2$')
        for key in sorted_keys:
            axs[0, 1].plot(g_list, results[key]['entropies'] / np.log(2), **plot_styles[key])

        # subplot: <X> magnetization
        axs[1, 0].set_ylabel(r'$\langle \sigma^x \rangle$')
        for key in sorted_keys:
            axs[1, 0].plot(g_list, np.abs(results[key]['mag_x_list']), **plot_styles[key])

        # subplot: <XX> correlations
        axs[1, 1].set_ylabel(r'$\langle \sigma^x_i \sigma^x_j \rangle$')
        for key in sorted_keys:
            axs[1, 1].plot(g_list, results[key]['corr_xx_list'], **plot_styles[key])

        axs[1, 1].legend(title=('     $L_y,~\\chi$'))
        fig.suptitle(f'conserve="{conserve}"', y=1.02)
        fig.tight_layout(pad=0.1)
        file = os.path.join(folder, f'dmrg_conserve_{conserve}.pdf')
        plt.savefig(file, bbox_inches='tight')
        print(f'saved plot to {file}')


if __name__ == '__main__':
    main()
