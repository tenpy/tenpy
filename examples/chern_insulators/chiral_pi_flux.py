"""Chiral pi flux model - Chern insulator example

Based on the model in :arxiv:`1012.4723`
"""

# Copyright 2019 TeNPy Developers

import numpy as np

from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.fermions_pi_flux import FermionicPiFluxModel, BipartiteSquare
from tenpy.networks import site


def plot_lattice():

    import matplotlib.pyplot as plt
    ax = plt.gca()
    fs = site.FermionSite()
    lat = BipartiteSquare(3, 3, fs, basis=[[2, 0], [0, 2]])
    lat.plot_sites(ax)
    lat.plot_coupling(ax, lat.NN, linestyle='-', color='green')
    lat.plot_coupling(ax, lat.nNNdashed, linestyle='--', color='black')
    lat.plot_coupling(ax, lat.nNNdotted, linestyle='--', color='red')
    ax.set_aspect('equal')
    plt.show()


def run(phi_ext=np.linspace(0, 1.0, 7)):

    data = dict(phi_ext=phi_ext, QL=[], ent_spectrum=[])

    model_params = dict(conserve='N', t=-1, mu=0, V=0, Lx=1, Ly=3, verbose=1)

    dmrg_params = {
        'mixer': True,  # setting this to True helps to escape local minima
        'mixer_params': {
            'amplitude': 1.e-5,
            'decay': 1.2,
            'disable_after': 30
        },
        'trunc_params': {
            'svd_min': 1.e-10,
        },
        'lanczos_params': {
            'N_min': 5,
            'N_max': 20
        },
        'chi_list': {0: 9, 10: 49, 20: 100},
        'max_E_err': 1.e-10,
        'max_S_err': 1.e-6,
        'max_sweeps': 150,
        'verbose': 1.,
    }

    prod_state = ['empty', 'full'] * (model_params['Lx'] * model_params['Ly'] * 2)

    eng = None

    for phi in phi_ext:

        print("=" * 100)
        print("phi_ext = ", phi)

        model_params['phi_ext'] = phi

        if eng is None:  # first time in the loop
            M = FermionicPiFluxModel(model_params)
            psi = MPS.from_product_state(M.lat.mps_sites(), prod_state, bc=M.lat.bc_MPS)
            eng = dmrg.EngineCombine(psi, M, dmrg_params)
        else:
            del eng.DMRG_params['chi_list']
            M = FermionicPiFluxModel(model_params)
            eng.init_env(model=M)

        E, psi = eng.run()

        data['QL'].append(psi.average_charge(bond=0)[0])
        data['ent_spectrum'].append(psi.entanglement_spectrum(by_charge=True)[0])

    return data


def plot_results(data):

    import matplotlib.pyplot as plt

    plt.figure()
    ax = plt.gca()
    ax.plot(data['phi_ext'], data['QL'], marker='o')
    ax.set_xlabel(r"$\Phi_y / 2 \pi$")
    ax.set_ylabel(r"$ \langle Q^L(\Phi_y) \rangle$")
    plt.savefig("chiral_pi_flux_charge_pump.pdf")

    plt.figure()
    ax = plt.gca()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    color_by_charge = {}
    for phi_ext, spectrum in zip(data['phi_ext'], data['ent_spectrum']):
        for q, s in spectrum:
            q = q[0]
            label = ""
            if q not in color_by_charge:
                label = "{q:d}".format(q=q)
                color_by_charge[q] = colors[len(color_by_charge) % len(colors)]
            color = color_by_charge[q]
            ax.plot(phi_ext * np.ones(s.shape), s, linestyle='', marker='_', color=color, label=label)
    ax.set_xlabel(r"$\Phi_y / 2 \pi$")
    ax.set_ylabel(r"$ \epsilon_\alpha $")
    ax.set_ylim(0., 8.)
    ax.legend(loc='upper right')
    plt.savefig("chiral_pi_flux_ent_spec_flow.pdf")


if __name__ == "__main__":

    # plot_lattice()
    data = run()
    plot_results(data)
