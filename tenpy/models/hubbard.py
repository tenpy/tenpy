"""Bosonic and fermionic Hubbard models."""

# Copyright (C) TeNPy Developers, Apache license
from typing import Literal

import lasyten as lasy
import numpy as np

from .lattice import Chain
from .model import ModelBuilder

__all__ = [
    'BoseHubbardModel',
    'BoseHubbardChain',
    'FermiHubbardModel',
    'FermiHubbardChain',
    'FermiHubbardModel2',
    'DipolarBoseHubbardChain',
]


class BoseHubbardModel(ModelBuilder):
    r"""Spinless Bose-Hubbard model.

    The Hamiltonian is:

    .. math ::
        H = - t \sum_{\langle i, j \rangle, i < j} (b_i^{\dagger} b_j + b_j^{\dagger} b_i)
            + V \sum_{\langle i, j \rangle, i < j} n_i n_j
            + \frac{U}{2} \sum_i n_i (n_i - 1) - \mu \sum_i n_i

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`BoseHubbardModel` below.

    Options
    -------
    .. cfg:config :: BoseHubbardModel
        :include: ModelBuilder

        n_max : int
            Maximum number of bosons per site.
        filling : float
            Average filling.
        conserve: {'best' | 'N' | 'parity' | None}
            What should be conserved. See :class:`~lasyten.sites.SpinlessBosonSite`.
        t, U, V, mu: float | array
            Couplings as defined in the Hamiltonian above. Note the signs!
            Defaults to `t=1.` without interaction ``V=U=0`` and ``mu=0``.
        phi_ext : float
            For 2D lattices and periodic y boundary conditions only.
            External magnetic flux 'threaded' through the cylinder. Hopping amplitudes for bonds
            'across' the periodic boundary are modified such that particles hopping around the
            circumference of the cylinder acquire a phase ``2 pi phi_ext``.

    """

    conserve: Literal['N', 'parity', None]

    def init_sites(self, model_params):
        n_max = model_params.get('n_max', 3, int)
        filling = model_params.get('filling', 0.5, 'real')
        conserve = model_params.get('conserve', 'N', str)
        if conserve == 'best':
            conserve = 'N'
            self.logger.info('%s: set conserve to %s', self.name, conserve)
        self.conserve = conserve
        return lasy.sites.SpinlessBosonSite(Nmax=n_max, conserve=conserve, filling=filling, backend=self.backend)

    def init_couplings(self, model_params):
        site, *_ = self.lattice.unit_cell
        self.define_coupling(
            lasy.models.hopping([site, site], t_left=1, t_right=0, species=None, name='Bd.B'),
            hc='sum_B.Bd',
            permute='swap_hc',
        )
        self.define_coupling(
            lasy.models.density_density_interaction([site, site], name='N.N'), hc='Ntot.Ntot', permute='invariant'
        )

    def init_terms(self, model_params):
        # 0) Read and set parameters.
        t = model_params.get('t', 1.0, 'real_or_array')
        U = model_params.get('U', 0.0, 'real_or_array')
        V = model_params.get('V', 0.0, 'real_or_array')
        mu = model_params.get('mu', 0, 'real_or_array')
        phi_ext = model_params.get('phi_ext', None, 'real')

        self.add_onsite_operator(-mu - U / 2, 'N')
        self.add_onsite_operator(U / 2, 'NN')
        for u1, u2, dx in self.lattice.pairs['nearest_neighbors']:
            if phi_ext is None:
                hop = -t
            else:
                hop = self.coupling_strength_add_ext_flux(-t, dx, [0, 2 * np.pi * phi_ext])
            self.add_two_site_coupling(hop, 'Bd.B', dx, u1=u1, u2=u2, plus_hc=True)
            self.add_two_site_coupling(V, 'N.N', dx, u1=u1, u2=u2)


class BoseHubbardChain(BoseHubbardModel):
    """The :class:`BoseHubbardModel` on a Chain, suitable for TEBD.

    See the :class:`BoseHubbardModel` for the documentation of parameters.
    """

    default_lattice = Chain
    force_default_lattice = True


class FermiHubbardModel(ModelBuilder):
    r"""Spin-1/2 Fermi-Hubbard model.

    The Hamiltonian reads:

    .. math ::
        H = - \sum_{\langle i, j \rangle, i < j, \sigma} t (c^{\dagger}_{\sigma, i} c_{\sigma j} + h.c.)
            + \sum_i U n_{\uparrow, i} n_{\downarrow, i}
            - \sum_i \mu ( n_{\uparrow, i} + n_{\downarrow, i} )
            +  \sum_{\langle i, j \rangle, i< j, \sigma} V
                       (n_{\uparrow,i} + n_{\downarrow,i})(n_{\uparrow,j} + n_{\downarrow,j})


    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    .. warning ::
        Using the Jordan-Wigner string (``JW``) is crucial to get correct results!
        See :doc:`/intro/JordanWigner` for details.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`FermiHubbardModel` below.

    Options
    -------
    .. cfg:config :: FermiHubbardModel
        :include: ModelBuilder

        conserve_N : {'N' | 'parity' | None}
            Whether particle number is conserved,
            see :class:`~lasyten.sites.SpinHalfFermionSite` for details.
        conserve_S : {'SU2' | 'Sz' | 'parity' | None}
            Whether spin is conserved,
            see :class:`~lasyten.sites.SpinHalfFermionSite` for details.
        t, U, mu : float | array
            Couplings as defined for the Hamiltonian above. Note the signs!
            Defaults to ``t=1`` without interaction ``U=0`` and ``mu=0``.
        phi_ext : float
            For 2D lattices and periodic y boundary conditions only.
            External magnetic flux 'threaded' through the cylinder. Hopping amplitudes for bonds
            'across' the periodic boundary are modified such that particles hopping around the
            circumference of the cylinder acquire a phase ``2 pi phi_ext``.

    """

    conserve_N: Literal['N', 'parity', None]
    conserve_S: Literal['SU2', 'Sz', 'parity', None]

    def init_sites(self, model_params):
        self.conserve_N = conserve_N = model_params.get('conserve_N', 'N', str)
        self.conserve_S = conserve_S = model_params.get('conserve_S', 'SU2', str)
        model_params.deprecated_ignore('cons_N', 'cons_Sz')
        return lasy.sites.SpinHalfFermionSite(conserve_N=conserve_N, conserve_S=conserve_S, backend=self.backend)

    def init_couplings(self, model_params):
        site, *_ = self.lattice.unit_cell
        self.define_coupling(
            lasy.models.hopping([site, site], t_left=1, t_right=0, species=None, name='sum_Cd.C'),
            hc='sum_C.Cd',
            permute='swap_hc',
        )  # summed over all species, i.e. Cud.Cu + Cdd.Cd
        self.define_coupling(
            lasy.models.density_density_interaction([site, site], name='Ntot.Ntot'), hc='Ntot.Ntot', permute='invariant'
        )  # Ntot, i.e. summing over all species on each site is default

    def init_terms(self, model_params):
        # 0) Read out/set default parameters.
        t = model_params.get('t', 1.0, 'real_or_array')
        U = model_params.get('U', 0, 'real_or_array')
        V = model_params.get('V', 0, 'real_or_array')
        mu = model_params.get('mu', 0.0, 'real_or_array')
        phi_ext = model_params.get('phi_ext', None, 'real')

        self.add_onsite_operator(-mu, 'Ntot')
        self.add_onsite_operator(U, 'NupNdown')
        for u1, u2, dx in self.lattice.pairs['nearest_neighbors']:
            if phi_ext is None:
                hop = -t
            else:
                hop = self.coupling_strength_add_ext_flux(-t, dx, [0, 2 * np.pi * phi_ext])
            self.add_two_site_coupling(hop, 'sum_Cd.C', dx, u1=u1, u2=u2, plus_hc=True)
            self.add_two_site_coupling(V, 'Ntot.Ntot', dx, u1=u1, u2=u2)


class FermiHubbardChain(FermiHubbardModel):
    """The :class:`FermiHubbardModel` on a Chain, suitable for TEBD.

    See the :class:`FermiHubbardModel` for the documentation of parameters.
    """

    default_lattice = Chain
    force_default_lattice = True


class FermiHubbardModel2(ModelBuilder):
    """Another implementation of the :class:`FermiHubbardModel`, but with local dimension 2.

    This class implements the same Hamiltonian as :class:`FermiHubbardModel`:


    However, it does not use the :class:`~lasyten.sites.SpinHalfFermionSite`, but two plain
    :class:`lasyten.sites.SpinlessFermionSite` for individual spin-up/down fermions, combined in the
    :class:`~tenpy.models.lattice.MultiSpeciesLattice`.

    Formally, not grouping the Sites leads to a better scaling of DMRG;
    yet, it can sometimes lead to ergodicity issues in practice.
    When you :meth:`group_sites` in this model, you will end up with the same MPO as the
    :class:`FermiHubbardModel`.


    .. warning ::
        Using the Jordan-Wigner string (``JW``) is crucial to get correct results!
        See :doc:`/intro/JordanWigner` for details.

    Options
    -------
    .. cfg:config :: FermiHubbardModel2
        include: FermiHubbardModel

    """

    def init_sites(self, model_params):
        raise NotImplementedError('spin_half_species not ported')
        conserve_N = model_params.get('conserve_N', 'N', str)
        conserve_S = model_params.get('conserve_S', 'Sz', str)
        return spin_half_species(FermionSite, conserve_N=conserve_N, conserve_S=conserve_S)
        # special syntax: returns tuple (sites, species_names) to cause
        # CouplingMPOModel.init_lattice to initialize a MultiSpeciesLattice
        # based on the lattice specified in the model parameters

    def init_terms(self, model_params):
        t = model_params.get('t', 1.0, 'real_or_array')
        U = model_params.get('U', 0, 'real_or_array')
        V = model_params.get('V', 0, 'real_or_array')
        mu = model_params.get('mu', 0.0, 'real_or_array')
        phi_ext = model_params.get('phi_ext', None, 'real')

        for u in range(len(self.lattice.unit_cell)):
            self.add_onsite(-mu, u, 'N')
        for u1, u2, dx in self.lattice.pairs['onsite_up-down']:
            self.add_coupling(U, u1, 'N', u2, 'N', dx)

        for u1, u2, dx in self.lattice.pairs['nearest_neighbors_diag']:
            if phi_ext is None:
                hop = -t
            else:
                hop = self.coupling_strength_add_ext_flux(-t, dx, [0, 2 * np.pi * phi_ext])
            self.add_coupling(hop, u1, 'Cd', u2, 'C', dx, plus_hc=True)

        for u1, u2, dx in self.lattice.pairs['nearest_neighbors_all-all']:
            self.add_coupling(V, u1, 'N', u2, 'N', dx)


class DipolarBoseHubbardChain(ModelBuilder):
    r"""Dipole-conserving spinless Bose-Hubbard model.

    The Hamiltonian is:

    .. math ::
        H = - \mathtt{t} \sum_{i} (b_i^{\dagger} b_{i + 1}^2 b_{i + 2}^{\dagger} + \mathrm{h.c.})
            - \mathtt{t4} \sum_{i} (b_i^{\dagger} b_{i + 1} b_{i + 2} b_{i + 3}^{\dagger} + \mathrm{h.c.})
            + \frac{\mathtt{U}}{2} \sum_i n_i (n_i - 1)
            - \mathtt{mu} \sum_i n_i

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`DipolarBoseHubbardChain` below.

    Options
    -------
    .. cfg:config :: DipolarBoseHubbardChain
        :include: ModelBuilder

        Nmax : int
            Maximum number of bosons per site.
        conserve : {'best' | 'dipole' | 'N' | 'parity' | None}
            What should be conserved. See :class:`~lasyten.sites.SpinlessBosonSite`.
        t, t4, U, mu : float | array
            Couplings as defined in the Hamiltonian above. Note the signs!

    """

    def init_lattice(self, model_params):
        """Initialize a 1D lattice"""
        L = model_params.get('L', 64)
        Nmax = model_params.get('Nmax', 2)
        conserve = model_params.get('conserve', 'best')
        if conserve == 'best':
            conserve = 'dipole'
            self.logger.info('%s: set conserve to %s', self.name, conserve)
        if conserve == 'dipole':
            raise NotImplementedError('shift-symmetry')
        self.conserve = conserve
        bc_MPS = model_params.get('bc_MPS', 'finite')
        bc = 'periodic' if bc_MPS in ['infinite', 'segment'] else 'open'
        bc = model_params.get('bc', bc)
        site = lasy.sites.SpinlessBosonSite(Nmax=Nmax, conserve=conserve)
        lattice = Chain(L, site, bc=bc, bc_MPS=bc_MPS)
        return lattice

    def init_couplings(self, model_params):
        site, *_ = self.lattice.unit_cell
        Bd_B = lasy.models.hopping([site, site], t_left=1, t_right=0).to_tensor()
        B_Bd = Bd_B.hc
        h3 = lasy.planar_contraction(Bd_B, B_Bd, 'p1', 'p0*', relabel2={'p0': 'p1', 'p1': 'p2', 'p1*': 'p2*'})
        h4 = lasy.outer(Bd_B, B_Bd, relabel2={'p0': 'p2', 'p0*': 'p2*', 'p1': 'p3', 'p1*': 'p3*'})
        self.define_coupling(
            lasy.Coupling.from_tensor(h3, [site, site, site], name='Bd.BB.Bd'), hc='B.BdBd.B', permute=None
        )
        self.define_coupling(
            lasy.Coupling.from_tensor(h4, [site, site, site, site], name='Bd.B.B.Bd'), hc='B.Bd.Bd.B', permute=None
        )

    def init_terms(self, model_params):
        """Add the onsite and coupling terms to the model"""
        U = model_params.get('U', 1)
        t = model_params.get('t', 1)
        t4 = model_params.get('t4', 0)
        mu = model_params.get('mu', 0)

        # dipole hopping
        self.add_multi_site_coupling(-t, 'Bd.BB.Bd', [[0], [1], [2]], plus_hc=True)
        self.add_multi_site_coupling(-t4, 'Bd.B.B.Bd', [[0], [1], [2], [3]], plus_hc=True)

        # on-site interactions and chemical potential
        self.add_onsite_operator(U / 2, 'NN')
        self.add_onsite_operator(-mu - U / 2, 'N')
