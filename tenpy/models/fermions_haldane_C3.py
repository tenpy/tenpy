"""Spinless fermion generalized C=3 Haldane model.

Hamiltonian based on: "Topological flat band models with arbitrary Chern numbers", :arxiv:`1205.5792`.
"""

# Copyright 2019 TeNPy Developers

import numpy as np

from tenpy.models.model import CouplingMPOModel
from tenpy.tools.params import get_parameter
from tenpy.networks.site import FermionSite, GroupedSite
from tenpy.models import lattice


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
        Lx = get_parameter(model_params, 'Lx', 1, self.name)
        Ly = get_parameter(model_params, 'Ly', 3, self.name)
        fs = self.init_sites(model_params)
        lat = TripartiteTriangular(Lx, Ly, fs)
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
            self.add_onsite(-mu, 0, 'N', category='mu N')

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
