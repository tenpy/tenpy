"""Bosonic and fermionic Haldane models."""

# Copyright (C) TeNPy Developers, Apache license
from typing import Literal

import lasyten as lasy
import numpy as np

from .lattice import Honeycomb
from .model import ModelBuilder

__all__ = ['BosonicHaldaneModel', 'FermionicHaldaneModel']


class BosonicHaldaneModel(ModelBuilder):
    r"""Hardcore bosonic Haldane model.

    The Hamiltonian reads:

    .. math ::
        H = \sum_{ij} t_{ij} b_i^\dagger b_j + \sum_i \mu (n_{A, i} - n_{B, i})
        + V \sum_{\langle ij \rangle, i<j} n_{A, i} n_{B, j}


    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs and :math:`n_A, n_B`
    are the number operators on the A and B sites.
    Hopping is allowed to nearest and next-nearest neighbor sites with amplitudes
    :math:`t_{\langle ij \rangle}=t_1 \in \mathbb{R}` and
    :math:`t_{\langle\langle ij \rangle\rangle}=t_2 e^{\pm\mathrm{i}\phi} \in \mathbb{C}`
    respectively, where :math:`\pm\phi` is the phase acquired by a boson hopping between atoms
    in the same sublattice with a sign given by the direction of the hopping.
    This Hamiltonian is translated from :cite:`grushin2015`.

    Parameters
    ----------
    model_params : dict
        Parameters for the model. See :cfg:config:`BosonicHaldaneModel` below.

    Options
    -------
    .. cfg:config :: BosonicHaldaneModel
        :include: CouplingMPOModel

        conserve : 'best' | 'N' | 'parity' | None
            What should be conserved. See :class:`~tenpy.networks.Site.BosonSite`.
            For ``'best'``, we check the parameters that can be preserved.
        t1, t2, V, mu : float | array
            Hopping, interaction and chemical potential as defined for the Hamiltonian above.
            The default value for t2 is chosen to achieve the optimal band flatness ratio.
        phi_ext : float
            External magnetic flux 'threaded' through the cylinder. Hopping amplitudes for bonds
            'across' the periodic boundary are modified such that particles hopping around the
            circumference of the cylinder acquire a phase ``2 pi phi_ext``.

    """

    default_lattice = Honeycomb
    force_default_lattice = True
    conserve: Literal['N', 'parity', None]

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'N', str)
        if conserve == 'best':
            conserve = 'N'
        self.conserve = conserve
        return lasy.sites.SpinlessBosonSite(conserve=conserve)

    def init_couplings(self, model_params):
        site, *_ = self.lattice.unit_cell
        self.define_coupling(
            lasy.models.hopping([site, site], t_left=1, t_right=0, name='Bd.B'), hc='B.Bd', permute='swap_hc'
        )
        self.define_coupling(
            lasy.models.density_density_interaction([site, site], name='N.N'), hc='N.N', permute='invariant'
        )

    def init_terms(self, model_params):
        t1 = np.asarray(model_params.get('t1', -1.0, 'real_or_array'))
        t2_default = (np.sqrt(129) / 36) * t1 * np.exp(1j * np.arccos(3 * np.sqrt(3 / 43)))
        t2 = np.asarray(model_params.get('t2', t2_default, 'complex_or_array'))
        V = np.asarray(model_params.get('V', 0, 'real_or_array'))
        mu = np.asarray(model_params.get('mu', 0.0, 'real_or_array'))
        phi_ext = model_params.get('phi_ext', 0.0, 'real')

        self.add_onsite_operator(mu, 'N', u=0)
        self.add_onsite_operator(-mu, 'N', u=1)

        for u1, u2, dx in self.lattice.pairs['nearest_neighbors']:
            t1_phi = self.coupling_strength_add_ext_flux(t1, dx, [0, 2 * np.pi * phi_ext])
            self.add_two_site_coupling(t1_phi, 'Bd.B', dx, x1='sum_over', u1=u1, u2=u2, plus_hc=True)
            self.add_two_site_coupling(V, 'N.N', dx, x1='sum_over', u1=u1, u2=u2)

        for u1, u2, dx in [
            (0, 0, np.array([-1, 1])),
            (0, 0, np.array([1, 0])),
            (0, 0, np.array([0, -1])),
            (1, 1, np.array([0, 1])),
            (1, 1, np.array([1, -1])),
            (1, 1, np.array([-1, 0])),
        ]:
            t2_phi = self.coupling_strength_add_ext_flux(t2, dx, [0, 2 * np.pi * phi_ext])
            self.add_two_site_coupling(t2_phi, 'Bd.B', dx, x1='sum_over', u1=u1, u2=u2, plus_hc=True)


class FermionicHaldaneModel(ModelBuilder):
    r"""Spinless fermionic Haldane model.

    The Hamiltonian reads:

    .. math ::
        H = \sum_{ij} t_{ij} c_i^\dagger c_j + \sum_i \mu (n_{A, i} - n_{B, i})
        + V \sum_{\langle ij \rangle, i<j} n_{A, i} n_{B, j}


    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs and :math:`n_A, n_B`
    are the number operators on the A and B sites.
    Hopping is allowed to nearest and next-nearest neighbor sites with amplitudes
    :math:`t_{\langle ij \rangle}=t_1 \in \mathbb{R}` and
    :math:`t_{\langle\langle ij \rangle\rangle}=t_2 e^{\pm\mathrm{i}\phi} \in \mathbb{C}`
    respectively, where :math:`\pm\phi` is the phase acquired by an electron hopping
    between atoms in the same sublattice with a sign
    given by the direction of the hopping. This Hamiltonian is described in :cite:`grushin2015`.

    .. warning ::
        Using the Jordan-Wigner string (``JW``) is crucial to get correct results!
        See :doc:`/intro/JordanWigner` for details.

    Parameters
    ----------
    model_params : dict
        Parameters for the model. See :cfg:config:`FermionicHaldaneModel` below.

    Options
    -------
    .. cfg:config :: FermionicHaldaneModel
        :include: CouplingMPOModel

        conserve : 'best' | 'N' | 'parity'
            What should be conserved. See :class:`~tenpy.networks.Site.FermionSite`.
            For ``'best'``, we check the parameters what can be preserved.
        t1, t2, V, mu : float | array
            Hopping, interaction and chemical potential as defined for the Hamiltonian above.
            The default value for t2 is chosen to achieve the optimal band flatness ratio.
        phi_ext : float
            External magnetic flux 'threaded' through the cylinder. Hopping amplitudes for bonds
            'across' the periodic boundary are modified such that particles hopping around the
            circumference of the cylinder acquire a phase ``2 pi phi_ext``.

    """

    default_lattice = Honeycomb
    force_default_lattice = True
    conserve: Literal['N', 'parity']

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'N', str)
        if conserve == 'best':
            conserve = 'N'
        self.conserve = conserve
        self._check_fermion_conserve(conserve)
        return lasy.sites.SpinlessFermionSite(conserve=conserve)

    def init_couplings(self, model_params):
        site, *_ = self.lattice.unit_cell
        self.define_coupling(
            lasy.models.hopping([site, site], t_left=1, t_right=0, name='Cd.C'), hc='C.Cd', permute='swap_hc'
        )
        self.define_coupling(
            lasy.models.density_density_interaction([site, site], name='N.N'), hc='N.N', permute='invariant'
        )

    def init_terms(self, model_params):
        t1 = np.asarray(model_params.get('t1', -1.0, 'real_or_array'))
        t2_default = np.sqrt(129) / 36 * t1 * np.exp(1j * np.arccos(3 * np.sqrt(3 / 43)))
        t2 = np.asarray(model_params.get('t2', t2_default, 'complex_or_array'))
        V = np.asarray(model_params.get('V', 0, 'real_or_array'))
        mu = np.asarray(model_params.get('mu', 0.0, 'real_or_array'))
        phi_ext = model_params.get('phi_ext', 0.0, 'real')

        self.add_onsite_operator(mu, 'N', u=0)
        self.add_onsite_operator(-mu, 'N', u=1)

        for u1, u2, dx in self.lattice.pairs['nearest_neighbors']:
            t1_phi = self.coupling_strength_add_ext_flux(t1, dx, [0, 2 * np.pi * phi_ext])
            self.add_two_site_coupling(t1_phi, 'Cd.C', dx, x1='sum_over', u1=u1, u2=u2, plus_hc=True)
            self.add_two_site_coupling(V, 'N.N', dx, x1='sum_over', u1=u1, u2=u2)

        for u1, u2, dx in [
            (0, 0, np.array([-1, 1])),
            (0, 0, np.array([1, 0])),
            (0, 0, np.array([0, -1])),
            (1, 1, np.array([0, 1])),
            (1, 1, np.array([1, -1])),
            (1, 1, np.array([-1, 0])),
        ]:
            t2_phi = self.coupling_strength_add_ext_flux(t2, dx, [0, 2 * np.pi * phi_ext])
            self.add_two_site_coupling(t2_phi, 'Cd.C', dx, x1='sum_over', u1=u1, u2=u2, plus_hc=True)
