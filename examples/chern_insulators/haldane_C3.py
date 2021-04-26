"""Generalized (C=3) Haldane model - Chern insulator example

Based on the model in [Yang2012]_
"""

# Copyright 2019-2021 TeNPy Developers, GNU GPLv3

import numpy as np

from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.networks import site

from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import FermionSite, GroupedSite
from tenpy.models import lattice


class TripartiteTriangular(lattice.Lattice):
    def __init__(self, Lx, Ly, siteA, **kwargs):
        basis = np.array(([3., 0.], [0.5, 0.5 * np.sqrt(3)]))
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

        self.nNNA = [(0, 2, np.array([-1, 2])), (0, 2, np.array([0, -1])),
                     (0, 2, np.array([-1, -1])), (1, 0, np.array([0, 2])), (1, 0, np.array([1,
                                                                                            -1])),
                     (1, 0, np.array([0, -1])), (2, 1, np.array([0, 2])), (2, 1, np.array([1,
                                                                                           -1])),
                     (2, 1, np.array([0, -1]))]

        self.nNNB = [(0, 1, np.array([0, 1])), (0, 1, np.array([-1, 1])), (0, 1, np.array([0,
                                                                                           -2])),
                     (1, 2, np.array([0, 1])), (1, 2, np.array([-1, 1])), (1, 2, np.array([0,
                                                                                           -2])),
                     (2, 0, np.array([1, 1])), (2, 0, np.array([0, 1])), (2, 0, np.array([1, -2]))]

        self.nnNN = [(0, 1, np.array([-1, 2])), (0, 2, np.array([0, 0])), (0, 0, np.array([0,
                                                                                           -2])),
                     (1, 2, np.array([-1, 2])), (1, 0, np.array([1, 0])), (1, 1, np.array([0,
                                                                                           -2])),
                     (2, 0, np.array([0, 2])), (2, 1, np.array([1, 0])), (2, 2, np.array([0, -2]))]


class FermionicC3HaldaneModel(CouplingMPOModel):
    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'N')
        fs = FermionSite(conserve=conserve)
        gs = GroupedSite([fs, fs], labels=['A', 'B'], charges='same')
        gs.add_op('Ntot', gs.NA + gs.NB, False)
        return gs

    def init_lattice(self, model_params):
        Lx = model_params.get('Lx', 1)
        Ly = model_params.get('Ly', 3)
        fs = self.init_sites(model_params)
        lat = TripartiteTriangular(Lx, Ly, fs)
        return lat

    def init_terms(self, model_params):
        t = np.asarray(model_params.get('t', -1.))
        V = np.asarray(model_params.get('V', 0))
        phi_ext = 2 * np.pi * model_params.get('phi_ext', 0.)

        t1 = t
        t2 = 0.39 * t * 1j
        t3 = -0.34 * t

        for u1, u2, dx in self.lat.NN:
            t1_phi = self.coupling_strength_add_ext_flux(t1, dx, [0, phi_ext])
            self.add_coupling(t1_phi, u1, 'CdA', u2, 'CB', dx, 'JW', True)
            self.add_coupling(np.conj(t1_phi), u2, 'CdB', u1, 'CA', -dx, 'JW', True)
            self.add_coupling(V, u1, 'Ntot', u2, 'Ntot', dx)

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


def run(phi_ext=np.linspace(0, 1.0, 7)):

    data = dict(phi_ext=phi_ext, QL=[], ent_spectrum=[])

    model_params = dict(conserve='N', t=-1, V=0, Lx=1, Ly=3, verbose=1)

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

    prod_state = ['full_A empty_B', 'empty_A full_B', 'full_A empty_B'
                  ] * (model_params['Lx'] * model_params['Ly'])

    eng = None

    for phi in phi_ext:

        print("=" * 100)
        print("phi_ext = ", phi)

        model_params['phi_ext'] = phi

        if eng is None:  # first time in the loop
            M = FermionicC3HaldaneModel(model_params)
            psi = MPS.from_product_state(M.lat.mps_sites(), prod_state, bc=M.lat.bc_MPS)
            eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
        else:
            del eng.options['chi_list']
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
    ax.set_ylabel(r"$ \langle Q^L(\Phi_y) \rangle$")
    plt.savefig("haldane_C3_charge_pump.pdf")

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
    plt.savefig("haldane_C3_ent_spec_flow.pdf")


if __name__ == "__main__":

    # plot_lattice()
    data = run()
    plot_results(data)
