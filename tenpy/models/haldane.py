"""Bosonic and fermionic Haldane models."""
# Copyright 2019 TeNPy Developers, GNU GPLv3

import numpy as np

from tenpy.models.model import CouplingMPOModel
from tenpy.tools.params import get_parameter
from tenpy.networks.site import BosonSite, FermionSite

__all__ = ['BosonicHaldaneModel', 'FermionicHaldaneModel']


class BosonicHaldaneModel(CouplingMPOModel):
    r"""Hardcore bosonic Haldane model.

    The Hamiltonian reads:

    .. math ::
        H = \sum_{ij} t_{ij} b_i^\dagger b_j + \sum_i \mu (n_{A, i} - n_{B, i})
        + V \sum_{\langle ij \rangle, i<j} n_{A, i} n_{B, j}


    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs and :math:`n_A, n_B` are the number operators
    on the A and B sites. Hopping is allowed to nearest and next-nearest neighbor sites with amplitudes
    :math:`t_{\langle ij \rangle}=t_1 \in \mathbb{R}` and
    :math:`t_{\langle\langle ij \rangle\rangle}=t_2 e^{\pm\mathrm{i}\phi} \in \mathbb{C}` respectively, where
    :math:`\pm\phi` is the phase acquired by a boson hopping between atoms in the same sublattice with a sign
    given by the direction of the hopping. This Hamiltonian is translated from [Grushin2015]_.
    All parameters are collected in a single dictionary `model_params` and read out with
    :func:`~tenpy.tools.params.get_parameter`.

    Parameters
    ----------
    conserve : 'best' | 'N' | 'parity' | None
        What should be conserved. See :class:`~tenpy.networks.Site.BosonSite`.
        For ``'best'``, we check the parameters that can be preserved.
    t1, t2, V, mu : float | array
        Hopping, interaction and chemical potential as defined for the Hamiltonian above.
        The default value for t2 is chosen to achieve the optimal band flatness ratio.
    bc_MPS : {'finite' | 'infinte'}
        MPS boundary conditions along the x-direction.
        For 'infinite' boundary conditions, repeat the unit cell in x-direction.
        Coupling boundary conditions in x-direction are chosen accordingly.
        Only used if `lattice` is a string.
    order : string
        Ordering of the sites in the MPS, e.g. 'default', 'snake';
        see :meth:`~tenpy.models.lattice.Lattice.ordering`.
        Only used if `lattice` is a string.
    L : int
        Lenght of the lattice.
        Only used if `lattice` is the name of a 1D Lattice.
    Lx, Ly : int
        Length of the lattice in x- and y-direction.
        Only used if `lattice` is the name of a 2D Lattice.
    bc_y : 'ladder' | 'cylinder'
        Boundary conditions in y-direction.
        Only used if `lattice` is the name of a 2D Lattice.
    """
    def __init__(self, model_params):
        model_params.setdefault('lattice', 'Honeycomb')
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        conserve = get_parameter(model_params, 'conserve', 'N', self.name)
        site = BosonSite(conserve=conserve)
        return site

    def init_terms(self, model_params):
        t1 = get_parameter(model_params, 't1', -1., self.name, True)
        t2 = get_parameter(model_params, 't2',
                           (np.sqrt(129) / 36) * t1 * np.exp(1j * np.arccos(3 * np.sqrt(3 / 43))),
                           self.name, True)
        V = get_parameter(model_params, 'V', 0, self.name, True)
        mu = get_parameter(model_params, 'mu', 0., self.name, True)
        phi_ext = 2 * np.pi * get_parameter(model_params, 'phi_ext', 0., self.name)

        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(mu, 0, 'N', category='mu N')
            self.add_onsite(-mu, 1, 'N', category='mu N')

        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            t1_phi = self.coupling_strength_add_ext_flux(t1, dx, [0, phi_ext])
            self.add_coupling(t1_phi, u1, 'Bd', u2, 'B', dx, category='t1 Bd_i B_j')
            self.add_coupling(np.conj(t1_phi), u2, 'Bd', u1, 'B', -dx,
                              category='t1 Bd_i B_j h.c.')  # h.c.
            self.add_coupling(V, u1, 'N', u2, 'N', dx, category='V N_i N_j')

        for u1, u2, dx in [(0, 0, np.array([-1, 1])), (0, 0, np.array([1, 0])),
                           (0, 0, np.array([0, -1])), (1, 1, np.array([0, 1])),
                           (1, 1, np.array([1, -1])), (1, 1, np.array([-1, 0]))]:
            t2_phi = self.coupling_strength_add_ext_flux(t2, dx, [0, phi_ext])
            self.add_coupling(t2_phi, u1, 'Bd', u2, 'B', dx, category='t2 Bd_i B_j')
            self.add_coupling(np.conj(t2_phi), u2, 'Bd', u1, 'B', -dx,
                              category='t2 Bd_i B_j h.c.')  # h.c.


class FermionicHaldaneModel(CouplingMPOModel):
    r"""Spinless fermionic Haldane model.

    The Hamiltonian reads:

    .. math ::
        H = \sum_{ij} t_{ij} c_i^\dagger c_j + \sum_i \mu (n_{A, i} - n_{B, i})
        + V \sum_{\langle ij \rangle, i<j} n_{A, i} n_{B, j}


    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs and :math:`n_A, n_B` are the number operators
    on the A and B sites. Hopping is allowed to nearest and next-nearest neighbor sites with amplitudes
    :math:`t_{\langle ij \rangle}=t_1 \in \mathbb{R}` and
    :math:`t_{\langle\langle ij \rangle\rangle}=t_2 e^{\pm\mathrm{i}\phi} \in \mathbb{C}` respectively, where
    :math:`\pm\phi` is the phase acquired by an electron hopping between atoms in the same sublattice with a sign
    given by the direction of the hopping. This Hamiltonian is described in [Grushin2015]_.
    All parameters are collected in a single dictionary `model_params` and read out with
    :func:`~tenpy.tools.params.get_parameter`.

    .. warning ::
        Using the Jordan-Wigner string (``JW``) is crucial to get correct results!
        See :doc:`/intro_JordanWigner` for details.

    Parameters
    ----------
    conserve : 'best' | 'N' | 'parity' | None
        What should be conserved. See :class:`~tenpy.networks.Site.FermionSite`.
        For ``'best'``, we check the parameters what can be preserved.
    t1, t2, V, mu : float | array
        Hopping, interaction and chemical potential as defined for the Hamiltonian above.
        The default value for t2 is chosen to achieve the optimal band flatness ratio.
    bc_MPS : {'finite' | 'infinte'}
        MPS boundary conditions along the x-direction.
        For 'infinite' boundary conditions, repeat the unit cell in x-direction.
        Coupling boundary conditions in x-direction are chosen accordingly.
        Only used if `lattice` is a string.
    order : string
        Ordering of the sites in the MPS, e.g. 'default', 'snake';
        see :meth:`~tenpy.models.lattice.Lattice.ordering`.
        Only used if `lattice` is a string.
    L : int
        Lenght of the lattice.
        Only used if `lattice` is the name of a 1D Lattice.
    Lx, Ly : int
        Length of the lattice in x- and y-direction.
        Only used if `lattice` is the name of a 2D Lattice.
    bc_y : 'ladder' | 'cylinder'
        Boundary conditions in y-direction.
        Only used if `lattice` is the name of a 2D Lattice.
    """
    def __init__(self, model_params):
        model_params.setdefault('lattice', 'Honeycomb')
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        conserve = get_parameter(model_params, 'conserve', 'N', self.name)
        site = FermionSite(conserve=conserve)
        return site

    def init_terms(self, model_params):
        t1 = get_parameter(model_params, 't1', -1., self.name, True)
        t2 = get_parameter(model_params, 't2',
                           (np.sqrt(129) / 36) * t1 * np.exp(1j * np.arccos(3 * np.sqrt(3 / 43))),
                           self.name, True)
        V = get_parameter(model_params, 'V', 0, self.name, True)
        mu = get_parameter(model_params, 'mu', 0., self.name, True)
        phi_ext = 2 * np.pi * get_parameter(model_params, 'phi_ext', 0., self.name)

        for u in range(len(self.lat.unit_cell)):

            self.add_onsite(mu, 0, 'N', category='mu N')
            self.add_onsite(-mu, 1, 'N', category='mu N')

        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            t1_phi = self.coupling_strength_add_ext_flux(t1, dx, [0, phi_ext])
            self.add_coupling(t1_phi, u1, 'Cd', u2, 'C', dx, 'JW', True, category='t1 Cd_i C_j')
            self.add_coupling(np.conj(t1_phi),
                              u2,
                              'Cd',
                              u1,
                              'C',
                              -dx,
                              'JW',
                              True,
                              category='t1 Cd_i C_j h.c.')  # h.c.
            self.add_coupling(V, u1, 'N', u2, 'N', dx, category='V N_i N_j')

        for u1, u2, dx in [(0, 0, np.array([-1, 1])), (0, 0, np.array([1, 0])),
                           (0, 0, np.array([0, -1])), (1, 1, np.array([0, 1])),
                           (1, 1, np.array([1, -1])), (1, 1, np.array([-1, 0]))]:
            t2_phi = self.coupling_strength_add_ext_flux(t2, dx, [0, phi_ext])
            self.add_coupling(t2_phi, u1, 'Cd', u2, 'C', dx, 'JW', True, category='t2 Cd_i C_j')
            self.add_coupling(np.conj(t2_phi),
                              u2,
                              'Cd',
                              u1,
                              'C',
                              -dx,
                              'JW',
                              True,
                              category='t2 Cd_i C_j h.c.')  # h.c.
