"""An example determining the phase diagram of the 2D transverse field Ising model."""
# Copyright (C) TeNPy Developers, GNU GPLv3
import matplotlib.pyplot as plt
import pickle
import numpy as np
import tenpy


def single_run(g=3.05, psi=None):
    # 2) Initialize the model
    # For a full list of options see
    # https://tenpy.readthedocs.io/en/latest/reference/tenpy.models.spins.SpinModel.html
    model_params = dict(
        bc_x='periodic', bc_y='cylinder',  # boundary conditions: infinite cylinder
        lattice='Square',  # select the lattice from predefined classes
        bc_MPS='infinite',  # use iMPS
        S=.5,  # spin-1/2
        Jx=-1,  # add NN coupling :math:`- \sum_{<i,j>} S^x_i S^x_j`.
        hz=g,  # add transverse field :math:`-g \sum_i S_x_i`
        Lx=2,  # size in x direction *of a unit cell*, system is infinite in x direction
        Ly=4,  # size in y direction, system is periodic in y direction (cylinder)
    )
    
    # We could also use the specialized ``tp.TFIModel`` here, but we want to showcase the
    # more general ``SpinModel`` here.
    model = tenpy.SpinModel(model_params)

    # 3) Initialize an MPS (initial guess). Note that this selects the charge sector.
    if psi is None:
        psi = tenpy.MPS.from_lat_product_state(model.lat, [[['up']]])

    # 4) Initialize / Configure the DMRG engine
    # For a full list of options see
    # https://tenpy.readthedocs.io/en/latest/reference/tenpy.algorithms.dmrg.TwoSiteDMRGEngine.html
    dmrg_params = dict(
        mixer=True,  # enable the mixer
        max_E_err=1e-10,  # convergence criterion
        trunc_params=dict(
            chi_max=200,  # maximum bond dimension
            svd_min=1e-10,  # cutoff for singular values, smaller ones are discarded.
        ),
        combine=True,
    )
    engine = tenpy.TwoSiteDMRGEngine(psi, model, dmrg_params)

    # 5) Run DMRG
    energy, psi = engine.run()

    # 6) Extract observables
    mag_z = np.average(psi.expectation_value('Sigmaz'))
    # See note in MPS.correlation_length for the units and why we need to divide
    correlation_length = psi.correlation_length() / model.lat.N_sites_per_ring
    entropy = psi.entanglement_entropy()[0]

    return psi, energy, entropy, mag_z, correlation_length


def sweep_phase_diagram(g_list, results_file=None):
    tenpy.setup_logging(to_stdout='INFO')
    psi = None
    g_list=g_list
    energy_list = []
    entropies = []
    mag_z_list = []
    corr_length_list = []

    for g in g_list:
        tenpy.logger.info(f'Optimizing at {g=}' + '\n' + '^' * 80)
        # re-use the psi from the previous g as initial guess
        psi, energy, entropy, mag_z, correlation_length = single_run(g, psi)
        energy_list.append(energy)
        entropies.append(entropy)
        mag_z_list.append(mag_z)
        corr_length_list.append(correlation_length)

    if results_file is not None:
        results = dict(g_list=g_list, energy_list=energy_list, entropies=entropies,
                    mag_z_list=mag_z_list, corr_length_list=corr_length_list)
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
    return results


def plot_results(results, plot_file=None):
    g_list = results['g_list']
    energy_list = results['energy_list']
    entropies = results['entropies']
    mag_z_list = results['mag_z_list']
    corr_length_list = results['corr_length_list']

    # plot results:
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    for ax, data, ylabel in zip(axs.flatten(),
                                [energy_list, entropies, mag_z_list, corr_length_list],
                                ['$E$', r'$S_{\text{vN}}$', r'$\langle Z \rangle$', r'$\xi$']):
        ax.set_xlabel('$g/J$')
        ax.set_ylabel(ylabel)
        ax.set_xlim(min(g_list), max(g_list))
        ax.plot(g_list, data)

    if plot_file is not None:
        plt.savefig(plot_file)
        print(f'saved plot to {plot_file}')
    else:
        plt.show()


if __name__ == '__main__':
    g_rough = np.linspace(0, 5, 21, endpoint=True)
    g_fine = np.linspace(2.5, 3.0, 21, endpoint=True)
    g_list = np.unique(np.concatenate([g_rough, g_fine]))
    sweep_phase_diagram(g_list, 'tfi_2D.pdf')
