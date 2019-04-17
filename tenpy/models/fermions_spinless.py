"""Spinless fermions with hopping and interaction.

.. todo ::
    add further terms (e.g. c^dagger c^dagger + h.c.) to the Hamiltonian.
"""
# Copyright 2018 TeNPy Developers

import numpy as np

from .model import CouplingMPOModel, NearestNeighborModel
from ..tools.params import get_parameter
from ..networks.site import FermionSite

__all__ = ['FermionModel', 'FermionChain']


class FermionModel(CouplingMPOModel):
    r"""Spinless fermions with particle number conservation.

    The Hamiltonian reads:

    .. math ::
        H = \sum_{\langle i,j\rangle, i<j}
              - \mathtt{J} (c^{\dagger}_i c_j + c^{\dagger}_j c_i) + \mathtt{V} n_i n_j \\
            - \sum_i
              \mathtt{mu} n_{i}

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs.
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
    J, V, mu : float | array
        Hopping, interaction and chemical potential as defined for the
        Hamiltonian above.
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
        conserve = get_parameter(model_params, 'conserve', 'N', self.name)
        if conserve == 'best':
            conserve = 'N'
            if self.verbose >= 1.:
                print(self.name + ": set conserve to", conserve)
        site = FermionSite(conserve=conserve)
        return site

    def init_terms(self, model_params):
        J = get_parameter(model_params, 'J', 1., self.name, True)
        V = get_parameter(model_params, 'V', 1., self.name, True)
        mu = get_parameter(model_params, 'mu', 0., self.name, True)
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-mu, u, 'N')
        for u1, u2, dx in self.lat.nearest_neighbors:
            self.add_coupling(-J, u1, 'Cd', u2, 'C', dx)
            self.add_coupling(np.conj(-J), u2, 'Cd', u1, 'C', -dx)  # h.c.
            self.add_coupling(V, u1, 'N', u2, 'N', dx)


class FermionChain(FermionModel, NearestNeighborModel):
    """The :class:`FermionModel` on a Chain, suitable for TEBD.

    See the :class:`FermionModel` for the documentation of parameters.
    """

    def __init__(self, model_params):
        model_params.setdefault('lattice', "Chain")
        CouplingMPOModel.__init__(self, model_params)
