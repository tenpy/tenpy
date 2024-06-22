"""Bosonic and fermionic Hubbard models."""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np

from .model import CouplingMPOModel, NearestNeighborModel
from .lattice import Chain
from ..tools.params import asConfig
from ..networks.site import FermionSite, BosonSite, SpinHalfFermionSite, spin_half_species

__all__ = ['BoseHubbardModel', 'BoseHubbardChain', 'FermiHubbardModel', 'FermiHubbardChain',
           'FermiHubbardModel2']


class BoseHubbardModel(CouplingMPOModel):
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
        :include: CouplingMPOModel

        n_max : int
            Maximum number of bosons per site.
        filling : float
            Average filling.
        conserve: {'best' | 'N' | 'parity' | None}
            What should be conserved. See :class:`~tenpy.networks.Site.BosonSite`.
        t, U, V, mu: float | array
            Couplings as defined in the Hamiltonian above. Note the signs!
        phi_ext : float
            For 2D lattices and periodic y boundary conditions only.
            External magnetic flux 'threaded' through the cylinder. Hopping amplitudes for bonds
            'across' the periodic boundary are modified such that particles hopping around the
            circumference of the cylinder acquire a phase ``2 pi phi_ext``.
    """
    def init_sites(self, model_params):
        n_max = model_params.get('n_max', 3, int)
        filling = model_params.get('filling', 0.5, 'real')
        conserve = model_params.get('conserve', 'N', str)
        if conserve == 'best':
            conserve = 'N'
            self.logger.info("%s: set conserve to %s", self.name, conserve)
        site = BosonSite(Nmax=n_max, conserve=conserve, filling=filling)
        return site

    def init_terms(self, model_params):
        # 0) Read and set parameters.
        t = model_params.get('t', 1., 'real_or_array')
        U = model_params.get('U', 0., 'real_or_array')
        V = model_params.get('V', 0., 'real_or_array')
        mu = model_params.get('mu', 0, 'real_or_array')
        phi_ext = model_params.get('phi_ext', None, 'real')
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-mu - U / 2., u, 'N')
            self.add_onsite(U / 2., u, 'NN')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            if phi_ext is None:
                hop = -t
            else:
                hop = self.coupling_strength_add_ext_flux(-t, dx, [0, 2 * np.pi * phi_ext])
            self.add_coupling(hop, u1, 'Bd', u2, 'B', dx, plus_hc=True)
            self.add_coupling(V, u1, 'N', u2, 'N', dx)


class BoseHubbardChain(BoseHubbardModel, NearestNeighborModel):
    """The :class:`BoseHubbardModel` on a Chain, suitable for TEBD.

    See the :class:`BoseHubbardModel` for the documentation of parameters.
    """
    def __init__(self, model_params):
        model_params = asConfig(model_params, self.__class__.__name__)
        model_params.setdefault('lattice', "Chain")
        CouplingMPOModel.__init__(self, model_params)

    def estimate_RAM_saving_factor(self):
        """Returns the expected saving factor for RAM based on charge conservation.

        For the BoseHubbardChain this factor was found to be between 1/7 and 1/10,
        therefore we let it default to 1/8 (for particle number conservation).

        Returns
        -------
        factor : int
            saving factor, due to conservation

        Options
        -------
        .. cfg:configoptions :: Model

            mem_saving_factor :: None | int
                Quantizes the RAM saving, due to conservation laws.
                By default it is 1/8 for the BoseHubbardChain.
                However, this factor might be overwritten, if a better approximation is known.
                In this case one can pass it via the argument `mem_saving_factor` to the model.

        """
        chinfo = self.lat.unit_cell[0].leg.chinfo
        savings = 1.
        for mod in chinfo.mod:
            if mod == 1:
                savings *= 1/8. # this is what we found empirically
        return self.options.get("mem_saving_factor", savings, 'real')


class FermiHubbardModel(CouplingMPOModel):
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
        :include: CouplingMPOModel

        cons_N : {'N' | 'parity' | None}
            Whether particle number is conserved,
            see :class:`~tenpy.networks.site.SpinHalfFermionSite` for details.
        cons_Sz : {'Sz' | 'parity' | None}
            Whether spin is conserved,
            see :class:`~tenpy.networks.site.SpinHalfFermionSite` for details.
        t, U, mu : float | array
            Couplings as defined for the Hamiltonian above. Note the signs!
        phi_ext : float
            For 2D lattices and periodic y boundary conditions only.
            External magnetic flux 'threaded' through the cylinder. Hopping amplitudes for bonds
            'across' the periodic boundary are modified such that particles hopping around the
            circumference of the cylinder acquire a phase ``2 pi phi_ext``.
    """
    def init_sites(self, model_params):
        cons_N = model_params.get('cons_N', 'N', str)
        cons_Sz = model_params.get('cons_Sz', 'Sz', str)
        site = SpinHalfFermionSite(cons_N=cons_N, cons_Sz=cons_Sz)
        return site

    def init_terms(self, model_params):
        # 0) Read out/set default parameters.
        t = model_params.get('t', 1., 'real_or_array')
        U = model_params.get('U', 0, 'real_or_array')
        V = model_params.get('V', 0, 'real_or_array')
        mu = model_params.get('mu', 0., 'real_or_array')
        phi_ext = model_params.get('phi_ext', None, 'real')

        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-mu, u, 'Ntot')
            self.add_onsite(U, u, 'NuNd')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            if phi_ext is None:
                hop = -t
            else:
                hop = self.coupling_strength_add_ext_flux(-t, dx, [0, 2 * np.pi * phi_ext])
            self.add_coupling(hop, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
            self.add_coupling(hop, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)
            self.add_coupling(V, u1, 'Ntot', u2, 'Ntot', dx)


class FermiHubbardChain(FermiHubbardModel, NearestNeighborModel):
    """The :class:`FermiHubbardModel` on a Chain, suitable for TEBD.

    See the :class:`FermiHubbardModel` for the documentation of parameters.
    """
    default_lattice = Chain
    force_default_lattice = True


class FermiHubbardModel2(CouplingMPOModel):
    """Another implementation of the :class:`FermiHubbardModel`, but with local dimension 2.

    This class implements the same Hamiltonian as :class:`FermiHubbardModel`:


    However, it does not use the :class:`~tenpy.networks.site.SpinHalfFermionSite`, but two plain
    :class:`~tenpy.networks.site.FermionSite` for individual spin-up/down fermions, combined in the
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
        cons_N = model_params.get('cons_N', 'N', str)
        cons_Sz = model_params.get('cons_Sz', 'Sz', str)
        return spin_half_species(FermionSite, cons_N=cons_N, cons_Sz=cons_Sz)
        # special syntax: returns tuple (sites, species_names) to cause
        # CouplingMPOModel.init_lattice to initialize a MultiSpeciesLattice
        # based on the lattice specified in the model parameters

    def init_terms(self, model_params):
        t = model_params.get('t', 1., 'real_or_array')
        U = model_params.get('U', 0, 'real_or_array')
        V = model_params.get('V', 0, 'real_or_array')
        mu = model_params.get('mu', 0., 'real_or_array')
        phi_ext = model_params.get('phi_ext', None, 'real')

        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-mu, u, 'N')
        for u1, u2, dx in self.lat.pairs['onsite_up-down']:
            self.add_coupling(U, u1, 'N', u2, 'N', dx)

        for u1, u2, dx in self.lat.pairs['nearest_neighbors_diag']:
            if phi_ext is None:
                hop = -t
            else:
                hop = self.coupling_strength_add_ext_flux(-t, dx, [0, 2 * np.pi * phi_ext])
            self.add_coupling(hop, u1, 'Cd', u2, 'C', dx, plus_hc=True)

        for u1, u2, dx in self.lat.pairs['nearest_neighbors_all-all']:
            self.add_coupling(V, u1, 'N', u2, 'N', dx)
