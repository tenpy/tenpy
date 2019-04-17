"""Simple Bose-Hubbard model.

"""
# Copyright 2018 TeNPy Developers

import numpy as np

from ..networks.site import BosonSite
from .model import CouplingMPOModel, NearestNeighborModel
from ..tools.params import get_parameter

__all__ = ["BoseHubbardModel", "BoseHubbardChain"]


class BoseHubbardModel(CouplingMPOModel):
    r"""Spinless Bose-Hubbard model on a chain.

    The Hamiltonian is:

    .. math ::
        H = t \sum_{\langle i, j \rangle, i < j} (b_i^{\dagger} b_j + b_j^{\dagger} b_i)
            + \frac{U}{2} \sum_i n_i (n_i - 1) + \mu \sum_i n_i

    Note that the signs of all parameters as defined in the Hamiltonian are positive.

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
    t, U, mu : float | array
        Couplings as defined in the Hamiltonian above.
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
        mu = get_parameter(model_params, 'mu', 0, self.name, True)
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(mu - U / 2., u, 'N')
            self.add_onsite(U / 2., u, 'NN')
        for u1, u2, dx in self.lat.nearest_neighbors:
            self.add_coupling(t, u1, 'Bd', u2, 'B', dx)
            self.add_coupling(np.conj(t), u2, 'Bd', u1, 'B', -dx)  # h.c.


class BoseHubbardChain(BoseHubbardModel, NearestNeighborModel):
    """The :class:`BoseHubbardModel` on a Chain, suitable for TEBD.

    See the :class:`BoseHubbardModel` for the documentation of parameters.
    """

    def __init__(self, model_params):
        model_params.setdefault('lattice', "Chain")
        CouplingMPOModel.__init__(self, model_params)
