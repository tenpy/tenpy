"""Chiral pi flux model - Chern insulator example

Based on the model in :arxiv:`1012.4723`
"""

# Copyright 2019 TeNPy Developers

import numpy as np

from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS

# TODO: move model in tenpy/models/
from tenpy.models.model import CouplingMPOModel
from tenpy.tools.params import get_parameter
from tenpy.networks.site import FermionSite

from tenpy.models import lattice
from tenpy.networks import site
import sys


class BipartiteSquare(lattice.Lattice):

    def __init__(self, Lx, Ly, siteA, **kwargs):

        basis = np.array(([2, 0.], [0, 2]))

        pos = np.array(([0, 0], [1, 0], [0, 1], [1, 1]))

        kwargs.setdefault('order', 'default')
        kwargs.setdefault('bc', 'periodic')
        kwargs.setdefault('bc_MPS', 'infinite')
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)

        super().__init__([Lx, Ly], [siteA, siteA, siteA, siteA], **kwargs)

        self.NN = [(0, 1, np.array([0, 0])), (1, 3, np.array([0, 0])),
                   (3, 2, np.array([0, 0])), (2, 0, np.array([0, 0])),
                   (2, 0, np.array([0, 1])), (1, 3, np.array([0, -1])),
                   (0, 1, np.array([-1, 0])), (3, 2, np.array([1, 0]))]
        self.nNNdashed = [(0, 3, np.array([0, 0])), (2, 1, np.array([0, 0])),
                          (3, 0, np.array([1, 1])), (1, 2, np.array([1, -1]))]
        self.nNNdotted = [(1, 2, np.array([1, 0])), (3, 0, np.array([1, 0])),
                          (2, 1, np.array([0, 1])), (3, 0, np.array([0, 1]))]


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


class FermionicPiFluxModel(CouplingMPOModel):

    def __init__(self, model_params):

        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):

        conserve = get_parameter(model_params, 'conserve', 'N', self.name)
        site = FermionSite(conserve=conserve)
        return site

    def init_lattice(self, model_params):

        choice = get_parameter(model_params, 'lattice', 'BipartiteSquare', self.name)

        if choice != 'BipartiteSquare':
            sys.exit("Error: Please choose the BipartiteSquare for pi_flux.")

        Lx = get_parameter(model_params, 'Lx', 1, self.name)
        Ly = get_parameter(model_params, 'Ly', 3, self.name)

        fs = self.init_sites(model_params)

        lat = BipartiteSquare(Lx, Ly, fs)

        print(lat.N_sites)

        return lat

    def init_terms(self, model_params):

        t = get_parameter(model_params, 't', -1., self.name, True)
        V = get_parameter(model_params, 'V', 0, self.name, True)
        mu = get_parameter(model_params, 'mu', 0., self.name, True)
        phi_ext = 2*np.pi*get_parameter(model_params, 'phi_ext', 0., self.name)

        t1 = t * np.exp(1j * np.pi/4)
        t2 = t / np.sqrt(2)

        for u in range(len(self.lat.unit_cell)):

            self.add_onsite(mu, 0, 'N', category='mu N')
            self.add_onsite(-mu, 1, 'N', category='mu N')

        for u1, u2, dx in self.lat.NN:

            t1_phi = self.coupling_strength_add_ext_flux(t1, dx, [0, phi_ext])
            self.add_coupling(t1_phi, u1, 'Cd', u2, 'C', dx, 'JW', True, category='t1 Cd_i C_j')
            self.add_coupling(np.conj(t1_phi), u2, 'Cd', u1, 'C', -dx, 'JW', True, category='t1 Cd_i C_j h.c.')

        for u1, u2, dx in self.lat.nNNdashed:

            t2_phi = self.coupling_strength_add_ext_flux(t2, dx, [0, phi_ext])
            self.add_coupling(t2_phi, u1, 'Cd', u2, 'C', dx, 'JW', True, category='t2 Cd_i C_j')
            self.add_coupling(np.conj(t2_phi), u2, 'Cd', u1, 'C', -dx, 'JW', True, category='t2 Cd_i C_j h.c.')

        for u1, u2, dx in self.lat.nNNdotted:

            t2_phi = self.coupling_strength_add_ext_flux(t2, dx, [0, phi_ext])
            self.add_coupling(-t2_phi, u1, 'Cd', u2, 'C', dx, 'JW', True, category='-t2 Cd_i C_j')
            self.add_coupling(-np.conj(t2_phi), u2, 'Cd', u1, 'C', -dx, 'JW', True, category='-t2 Cd_i C_j h.c.')


def run(phi_ext=np.linspace(0, 1.0, 3)):

    data = dict(phi_ext=phi_ext, QL=[], ent_spectrum=[])

    model_params = dict(conserve='N', t=-1, mu=0, V=0, lattice='BipartiteSquare', Lx=1, Ly=3, verbose=1)

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

    prod_state = ['full', 'empty'] * (model_params['Lx'] * model_params['Ly'] * 2)

    print(prod_state)

    eng = None

    for phi in phi_ext:

        print("=" * 100)
        print("phi_ext = ", phi)

        model_params['phi_ext'] = phi

        if eng is None:  # first time in the loop
            M = FermionicPiFluxModel(model_params)

            print("sites = ", M.lat.mps_sites())

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
    ax.set_ylabel(r"$ \langle Q^L \rangle$")
    plt.show()

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
    plt.show()


if __name__ == "__main__":

    # plot_lattice()
    data = run()
    plot_results(data)
