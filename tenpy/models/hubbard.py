"""Bosonic and fermionic Hubbard models."""
# Copyright 2019 TeNPy Developers, GNU GPLv3

import numpy as np

from .model import CouplingMPOModel, NearestNeighborModel
from ..tools.params import get_parameter
from ..networks.site import BosonSite, SpinHalfFermionSite

__all__ = ['BoseHubbardModel', 'BoseHubbardChain', 'FermiHubbardModel', 'FermiHubbardChain']


class BoseHubbardModel(CouplingMPOModel):
    r"""Spinless Bose-Hubbard model.

    The Hamiltonian is:

    .. math ::
        H = - t \sum_{\langle i, j \rangle, i < j} (b_i^{\dagger} b_j + b_j^{\dagger} b_i)
            + V \sum_{\langle i, j \rangle, i < j} n_i n_j
            + \frac{U}{2} \sum_i n_i (n_i - 1) - \mu \sum_i n_i

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs.
    All parameters are collected in a single dictionary `model_params` and read out with
    :func:`~tenpy.tools.params.get_parameter`.


    Parameters
    ----------
    n_max : int
        Maximum number of bosons per site.
    filling : float
        Average filling.
    conserve: {'best' | 'N' | 'parity' | None}
        What should be conserved. See :class:`~tenpy.networks.Site.BosonSite`.
    t, U, V, mu : float | array
        Couplings as defined in the Hamiltonian above. Note the signs!
    lattice : str | :class:`~tenpy.models.lattice.Lattice`
        Instance of a lattice class for the underlaying geometry.
        Alternatively a string being the name of one of the Lattices defined in
        :mod:`~tenpy.models.lattice`, e.g. ``"Chain", "Square", "HoneyComb", ...``.
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
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        n_max = get_parameter(model_params, 'n_max', 3, self.name)
        filling = get_parameter(model_params, 'filling', 0.5, self.name)
        conserve = get_parameter(model_params, 'conserve', 'N', self.name)
        if conserve == 'best':
            conserve = 'N'
            if self.verbose >= 1.:
                print(self.name + ": set conserve to", conserve)
        site = BosonSite(Nmax=n_max, conserve=conserve, filling=filling)
        return site

    def init_terms(self, model_params):
        # 0) Read and set parameters.
        t = get_parameter(model_params, 't', 1., self.name, True)
        U = get_parameter(model_params, 'U', 0., self.name, True)
        V = get_parameter(model_params, 'V', 0., self.name, True)
        mu = get_parameter(model_params, 'mu', 0, self.name, True)
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-mu - U / 2., u, 'N')
            self.add_onsite(U / 2., u, 'NN')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-t, u1, 'Bd', u2, 'B', dx)
            self.add_coupling(-np.conj(t), u2, 'Bd', u1, 'B', -dx)  # h.c.
            self.add_coupling(V, u1, 'N', u2, 'N', dx)


class BoseHubbardChain(BoseHubbardModel, NearestNeighborModel):
    """The :class:`BoseHubbardModel` on a Chain, suitable for TEBD.

    See the :class:`BoseHubbardModel` for the documentation of parameters.
    """
    def __init__(self, model_params):
        model_params.setdefault('lattice', "Chain")
        CouplingMPOModel.__init__(self, model_params)


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
    All parameters are collected in a single dictionary `model_params` and read out with
    :func:`~tenpy.tools.params.get_parameter`.

    .. warning ::
        Using the Jordan-Wigner string (``JW``) is crucial to get correct results!
        See :doc:`/intro_JordanWigner` for details.

    Parameters
    ----------
    cons_N : {'N' | 'parity' | None}
        Whether particle number is conserved,
        see :class:`~tenpy.networks.site.SpinHalfFermionSite` for details.
    cons_Sz : {'Sz' | 'parity' | None}
        Whether spin is conserved,
        see :class:`~tenpy.networks.site.SpinHalfFermionSite` for details.
    t, U, mu : float | array
        Parameters as defined for the Hamiltonian above. Note the signs!
    lattice : str | :class:`~tenpy.models.lattice.Lattice`
        Instance of a lattice class for the underlaying geometry.
        Alternatively a string being the name of one of the Lattices defined in
        :mod:`~tenpy.models.lattice`, e.g. ``"Chain", "Square", "HoneyComb", ...``.
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
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        cons_N = get_parameter(model_params, 'cons_N', 'N', self.name)
        cons_Sz = get_parameter(model_params, 'cons_Sz', 'Sz', self.name)
        site = SpinHalfFermionSite(cons_N=cons_N, cons_Sz=cons_Sz)
        return site

    def init_terms(self, model_params):
        # 0) Read out/set default parameters.
        t = get_parameter(model_params, 't', 1., self.name, True)
        U = get_parameter(model_params, 'U', 0, self.name, True)
        V = get_parameter(model_params, 'V', 0, self.name, True)
        mu = get_parameter(model_params, 'mu', 0., self.name, True)

        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-mu, u, 'Ntot')
            self.add_onsite(U, u, 'NuNd')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-t, u1, 'Cdu', u2, 'Cu', dx)
            self.add_coupling(-np.conj(t), u2, 'Cdu', u1, 'Cu', -dx)  # h.c.
            self.add_coupling(-t, u1, 'Cdd', u2, 'Cd', dx)
            self.add_coupling(-np.conj(t), u2, 'Cdd', u1, 'Cd', -dx)  # h.c.
            self.add_coupling(V, u1, 'Ntot', u2, 'Ntot', dx)


class FermiHubbardChain(FermiHubbardModel, NearestNeighborModel):
    """The :class:`FermiHubbardModel` on a Chain, suitable for TEBD.

    See the :class:`FermiHubbardModel` for the documentation of parameters.
    """
    def __init__(self, model_params):
        model_params.setdefault('lattice', "Chain")
        CouplingMPOModel.__init__(self, model_params)
