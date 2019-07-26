"""Spinless fermion chiral-pi-flux model.

Hamiltonian based on: "Fractional Quantum Hall States at Zero Magnetic Field", :arxiv:`1012.4723`.
"""

# Copyright 2019 TeNPy Developers

import numpy as np

from tenpy.models.model import CouplingMPOModel
from tenpy.tools.params import get_parameter
from tenpy.networks.site import FermionSite
from tenpy.models import lattice


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


class FermionicPiFluxModel(CouplingMPOModel):

    def __init__(self, model_params):

        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):

        conserve = get_parameter(model_params, 'conserve', 'N', self.name)
        site = FermionSite(conserve=conserve)
        return site

    def init_lattice(self, model_params):

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
            self.add_onsite(-mu, 0, 'N', category='mu N')

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
