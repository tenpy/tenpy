"""Generalized (C=3) Haldane model - Chern insulator example

Based on the model in :arxiv:`1205.5792`
"""

# Copyright 2019 TeNPy Developers

import numpy as np

from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS

# TODO: move model in tenpy/models/
from tenpy.models.model import CouplingMPOModel
from tenpy.tools.params import get_parameter
from tenpy.networks.site import FermionSite, GroupedSite

from tenpy.models import lattice
from tenpy.networks import site
import sys


class TripartiteTriangular(lattice.Lattice):

    def __init__(self, Lx, Ly, siteA, **kwargs):

        basis = np.array(([3., 0.], [0.5, 0.5*np.sqrt(3)]))

        pos = np.array(([0., 0.], [1., 0.], [2., 0.]))

        kwargs.setdefault('order', 'default')
        kwargs.setdefault('bc', 'periodic')
        kwargs.setdefault('bc_MPS', 'infinite')
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)

        super().__init__([Lx, Ly], [siteA, siteA, siteA], **kwargs)

        self.NN = [(0, 2, np.array([-1, 1])), (0, 1, np.array([0, 0])), (0, 0, np.array([0, -1])),
                   (1, 0, np.array([0, 1])), (1, 2, np.array([0, 0])), (1, 1, np.array([0, -1])),
                   (2, 1, np.array([0, 1])), (2, 0, np.array([1, 0])), (2, 2, np.array([0, -1]))]

        self.nNNA = [(0, 2, np.array([-1, 2])), (0, 2, np.array([0, -1])), (0, 2, np.array([-1, -1])),
                     (1, 0, np.array([0, 2])), (1, 0, np.array([1, -1])), (1, 0, np.array([0, -1])),
                     (2, 1, np.array([0, 2])), (2, 1, np.array([1, -1])), (2, 1, np.array([0, -1]))]

        self.nNNB = [(0, 1, np.array([0, 1])), (0, 1, np.array([-1, 1])), (0, 1, np.array([0, -2])),
                     (1, 2, np.array([0, 1])), (1, 2, np.array([-1, 1])), (1, 2, np.array([0, -2])),
                     (2, 0, np.array([1, 1])), (2, 0, np.array([0, 1])), (2, 0, np.array([1, -2]))]

        self.nnNN = [(0, 1, np.array([-1, 2])), (0, 2, np.array([0, 0])), (0, 0, np.array([0, -2])),
                     (1, 2, np.array([-1, 2])), (1, 0, np.array([1, 0])), (1, 1, np.array([0, -2])),
                     (2, 0, np.array([0, 2])), (2, 1, np.array([1, 0])), (2, 2, np.array([0, -2]))]


def plot_lattice():

    import matplotlib.pyplot as plt
    ax = plt.gca()
    fs = site.FermionSite()
    lat = TripartiteTriangular(3, 3, fs)
    lat.plot_sites(ax)
    lat.plot_coupling(ax, lat.NN, linestyle='--', color='green')
    lat.plot_coupling(ax, lat.nNNA, linestyle='--', color='red')
    lat.plot_coupling(ax, lat.nNNB, linestyle='--', color='blue')
    lat.plot_coupling(ax, lat.nnNN, linestyle='--', color='black')
    ax.set_aspect('equal')
    plt.show()


class FermionicC3HaldaneModel(CouplingMPOModel):

    def __init__(self, model_params):

        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):

        conserve = get_parameter(model_params, 'conserve', 'N', self.name)
        fs = FermionSite(conserve=conserve)

        gs = GroupedSite([fs, fs], labels=['A', 'B'], charges='same')
        gs.add_op('Ntot', gs.NA + gs.NB, False)

        return gs

    def init_lattice(self, model_params):

        choice = get_parameter(model_params, 'lattice', 'TripartiteTriangular', self.name)

        if choice != 'TripartiteTriangular':
            sys.exit("Error: Please choose the TripartiteTriangular for C3_haldane.")

        Lx = get_parameter(model_params, 'Lx', 1, self.name)
        Ly = get_parameter(model_params, 'Ly', 3, self.name)

        fs = self.init_sites(model_params)

        lat = TripartiteTriangular(Lx, Ly, fs)

        print(lat.N_sites)

        return lat

    def init_terms(self, model_params):

        t = get_parameter(model_params, 't', -1., self.name, True)
        V = get_parameter(model_params, 'V', 0, self.name, True)
        mu = get_parameter(model_params, 'mu', 0., self.name, True)
        phi_ext = 2*np.pi*get_parameter(model_params, 'phi_ext', 0., self.name)

        t1 = t
        t2 = 0.39*t*1j
        t3 = -0.34*t

        for u in range(len(self.lat.unit_cell)):

            self.add_onsite(mu, 0, 'N', category='mu N')
            self.add_onsite(-mu, 1, 'N', category='mu N')

        for u1, u2, dx in self.lat.NN:

            t1_phi = self.coupling_strength_add_ext_flux(t1, dx, [0, phi_ext])
            self.add_coupling(t1_phi, u1, 'CdA', u2, 'CB', dx, 'JW', True)
            self.add_coupling(np.conj(t1_phi), u2, 'CdB', u1, 'CA', -dx, 'JW', True)

        for u1, u2, dx in self.lat.nNNA:

            t2_phi = self.coupling_strength_add_ext_flux(t2, dx, [0, phi_ext])
            self.add_coupling(t2_phi, u1, 'CdA', u2, 'CA', dx, 'JW', True)
            self.add_coupling(np.conj(t2_phi), u2, 'CdA', u1, 'CA', -dx, 'JW', True)

        for u1, u2, dx in self.lat.nNNB:

            t2_phi = self.coupling_strength_add_ext_flux(t2, dx, [0, phi_ext])
            self.add_coupling(t2_phi, u1, 'CdB', u2, 'CB', dx, 'JW', True)
            self.add_coupling(np.conj(t2_phi), u2, 'CdB', u1, 'CB', -dx, 'JW', True)

        for u1, u2, dx in self.lat.nnNN:

            t3_phi = self.coupling_strength_add_ext_flux(t3, dx, [0, phi_ext])
            self.add_coupling(t3_phi, u1, 'CdA', u2, 'CB', dx, 'JW', True)
            self.add_coupling(np.conj(t3_phi), u2, 'CdB', u1, 'CA', -dx, 'JW', True)


def run(phi_ext=np.linspace(0, 1.0, 5)):

    data = dict(phi_ext=phi_ext, QL=[], ent_spectrum=[])

    model_params = dict(conserve='N', t=-1, mu=0, V=0, lattice='TripartiteTriangular', Lx=1, Ly=3, verbose=1)

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
        'chi_list': {0: 9, 10: 49, 20: 200},
        'max_E_err': 1.e-10,
        'max_S_err': 1.e-6,
        'max_sweeps': 150,
        'verbose': 1.,
    }

    prod_state = ['full_A empty_B', 'empty_A full_B', 'full_A empty_B'] * (model_params['Lx'] * model_params['Ly'])

    print(prod_state)

    eng = None

    for phi in phi_ext:

        print("=" * 100)
        print("phi_ext = ", phi)

        model_params['phi_ext'] = phi

        if eng is None:  # first time in the loop
            M = FermionicC3HaldaneModel(model_params)

            print("sites = ", M.lat.mps_sites())

            psi = MPS.from_product_state(M.lat.mps_sites(), prod_state, bc=M.lat.bc_MPS)
            eng = dmrg.EngineCombine(psi, M, dmrg_params)
        else:
            del eng.DMRG_params['chi_list']
            M = FermionicC3HaldaneModel(model_params)
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
