"""Spinless fermion Haldane model - Chern insulator example

Reproduces Fig. 2.a,b) in [Grushin2015]_
"""

# Copyright 2019-2021 TeNPy Developers, GNU GPLv3

import numpy as np

from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.haldane import FermionicHaldaneModel


def plot_model(model_params, phi_ext=0.1):

    model_params['phi_ext'] = phi_ext
    M = FermionicHaldaneModel(model_params)
    import matplotlib.pyplot as plt
    plt.figure()
    ax = plt.gca()
    M.lat.plot_sites(ax)
    M.coupling_terms['t1 Cd_i C_j'].plot_coupling_terms(ax, M.lat)
    M.coupling_terms['t2 Cd_i C_j'].plot_coupling_terms(ax,
                                                        M.lat,
                                                        text='{op_j!s} {strength_angle:.2f}',
                                                        text_pos=0.9)
    print(M.coupling_terms['t1 Cd_i C_j'].to_TermList())
    ax.set_aspect(1.)
    plt.show()


def run(model_params, phi_ext=np.linspace(0, 1.0, 7)):

    data = dict(phi_ext=phi_ext, QL=[], ent_spectrum=[])

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
        'chi_list': {
            0: 9,
            10: 49,
            20: 100
        },
        'max_E_err': 1.e-10,
        'max_S_err': 1.e-6,
        'max_sweeps': 150,
        'verbose': 1.,
    }

    prod_state = ['empty', 'full'] * (model_params['Lx'] * model_params['Ly'])

    eng = None

    for phi in phi_ext:

        print("=" * 100)
        print("phi_ext = ", phi)

        model_params['phi_ext'] = phi

        if eng is None:  # first time in the loop
            M = FermionicHaldaneModel(model_params)
            psi = MPS.from_product_state(M.lat.mps_sites(), prod_state, bc=M.lat.bc_MPS)
            eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
        else:
            del eng.options['chi_list']
            M = FermionicHaldaneModel(model_params)
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
    plt.savefig("haldane_charge_pump.pdf")

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
            ax.plot(phi_ext * np.ones(s.shape),
                    s,
                    linestyle='',
                    marker='_',
                    color=color,
                    label=label)
    ax.set_xlabel(r"$\Phi_y / 2 \pi$")
    ax.set_ylabel(r"$ \epsilon_\alpha $")
    ax.set_ylim(0., 8.)
    ax.legend(loc='upper right')
    plt.savefig("haldane_ent_spec_flow.pdf")


if __name__ == "__main__":

    t1_value = -1

    phi = np.arccos(3 * np.sqrt(3 / 43))
    t2_value = (np.sqrt(129) / 36) * t1_value * np.exp(1j * phi)  # optimal band flatness

    model_params = dict(conserve='N',
                        t1=t1_value,
                        t2=t2_value,
                        mu=0,
                        V=0,
                        bc_MPS='infinite',
                        order='default',
                        Lx=1,
                        Ly=3,
                        bc_y='cylinder',
                        verbose=0)

    #  plot_model(model_params)
    data = run(model_params)
    plot_results(data)
