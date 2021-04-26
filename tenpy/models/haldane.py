"""Bosonic and fermionic Haldane models."""
# Copyright 2019-2021 TeNPy Developers, GNU GPLv3

import numpy as np

from .model import CouplingMPOModel
from ..tools.params import asConfig
from ..networks.site import BosonSite, FermionSite
from .lattice import Honeycomb

__all__ = ['BosonicHaldaneModel', 'FermionicHaldaneModel']


class BosonicHaldaneModel(CouplingMPOModel):
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
    """
    default_lattice = Honeycomb
    force_default_lattice = True

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'N')
        site = BosonSite(conserve=conserve)
        return site

    def init_terms(self, model_params):
        t1 = np.asarray(model_params.get('t1', -1.))
        t2_default = (np.sqrt(129) / 36) * t1 * np.exp(1j * np.arccos(3 * np.sqrt(3 / 43)))
        t2 = np.asarray(model_params.get('t2', t2_default))
        V = np.asarray(model_params.get('V', 0))
        mu = np.asarray(model_params.get('mu', 0.))
        phi_ext = 2 * np.pi * model_params.get('phi_ext', 0.)

        self.add_onsite(mu, 0, 'N', category='mu N')
        self.add_onsite(-mu, 1, 'N', category='mu N')

        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            t1_phi = self.coupling_strength_add_ext_flux(t1, dx, [0, phi_ext])
            self.add_coupling(t1_phi, u1, 'Bd', u2, 'B', dx, category='t1 Bd_i B_j', plus_hc=True)
            self.add_coupling(V, u1, 'N', u2, 'N', dx, category='V N_i N_j')

        for u1, u2, dx in [(0, 0, np.array([-1, 1])), (0, 0, np.array([1, 0])),
                           (0, 0, np.array([0, -1])), (1, 1, np.array([0, 1])),
                           (1, 1, np.array([1, -1])), (1, 1, np.array([-1, 0]))]:
            t2_phi = self.coupling_strength_add_ext_flux(t2, dx, [0, phi_ext])
            self.add_coupling(t2_phi, u1, 'Bd', u2, 'B', dx, category='t2 Bd_i B_j', plus_hc=True)


class FermionicHaldaneModel(CouplingMPOModel):
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

        conserve : 'best' | 'N' | 'parity' | None
            What should be conserved. See :class:`~tenpy.networks.Site.FermionSite`.
            For ``'best'``, we check the parameters what can be preserved.
        t1, t2, V, mu : float | array
            Hopping, interaction and chemical potential as defined for the Hamiltonian above.
            The default value for t2 is chosen to achieve the optimal band flatness ratio.

    """
    default_lattice = Honeycomb
    force_default_lattice = True

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'N')
        site = FermionSite(conserve=conserve)
        return site

    def init_terms(self, model_params):
        t1 = np.asarray(model_params.get('t1', -1.))
        t2_default = np.sqrt(129) / 36 * t1 * np.exp(1j * np.arccos(3 * np.sqrt(3 / 43)))
        t2 = np.asarray(model_params.get('t2', t2_default))
        V = np.asarray(model_params.get('V', 0))
        mu = np.asarray(model_params.get('mu', 0.))
        phi_ext = 2 * np.pi * model_params.get('phi_ext', 0.)

        self.add_onsite(mu, 0, 'N', category='mu N')
        self.add_onsite(-mu, 1, 'N', category='mu N')

        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            t1_phi = self.coupling_strength_add_ext_flux(t1, dx, [0, phi_ext])
            self.add_coupling(t1_phi, u1, 'Cd', u2, 'C', dx, category='t1 Cd_i C_j', plus_hc=True)
            self.add_coupling(V, u1, 'N', u2, 'N', dx, category='V N_i N_j')

        for u1, u2, dx in [(0, 0, np.array([-1, 1])), (0, 0, np.array([1, 0])),
                           (0, 0, np.array([0, -1])), (1, 1, np.array([0, 1])),
                           (1, 1, np.array([1, -1])), (1, 1, np.array([-1, 0]))]:
            t2_phi = self.coupling_strength_add_ext_flux(t2, dx, [0, phi_ext])
            self.add_coupling(t2_phi, u1, 'Cd', u2, 'C', dx, category='t2 Cd_i C_j', plus_hc=True)
