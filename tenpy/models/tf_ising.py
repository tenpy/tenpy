"""Prototypical example of a quantum model: the transverse field Ising model.

Like the :class:`~tenpy.models.xxz_chain.XXZChain`, the transverse field ising chain
:class:`TFIChain` is contained in the more general :class:`~tenpy.models.spins.SpinChain`;
the idea is more to serve as a pedagogical example for a 'model'.
Choosing the field along z allow to preserve parity, if desired, at the expense of a larger MPO
bond dimension for the Hamiltion.

The :class:`TFIModel2D` contains the same couplings but on a square lattice in 2D
(with chooseable boundary conditions).
As such, it illustrates the correct usage of the :class:`~tenpy.models.lattice.Lattice` classes.
"""
# Copyright 2018 TeNPy Developers

import numpy as np

from .lattice import Chain, SquareLattice
from .model import CouplingModel, NearestNeighborModel, MPOModel
from ..tools.params import get_parameter, unused_parameters
from ..networks.site import SpinHalfSite

__all__ = ['TFIChain', 'TFIModel2D']


class TFIChain(CouplingModel, NearestNeighborModel, MPOModel):
    r"""Transverse field Ising chain.

    The Hamiltonian reads:

    .. math ::
        H = - \sum_{i} \mathtt{J} \sigma^x_i \sigma^x_{i+1}
            - \sum_{i} \mathtt{g} \sigma^z_i

    All parameters are collected in a single dictionary `model_param` and read out with
    :func:`~tenpy.tools.params.get_parameter`.

    Parameters
    ----------
    L : int
        Length of the chain
    J, g : float | array
        Couplings as defined for the Hamiltonian above.
    bc_MPS : {'finite' | 'infinte'}
        MPS boundary conditions. Coupling boundary conditions are chosen appropriately.
    conserve : 'parity' | None
        What should be conserved. See :class:`~tenpy.networks.Site.SpinSite`.
    """

    def __init__(self, model_param):
        # 0) read out/set default parameters
        L = get_parameter(model_param, 'L', 2, self.__class__)
        J = get_parameter(model_param, 'J', 1., self.__class__)
        g = get_parameter(model_param, 'g', 1., self.__class__)  # critical!
        bc_MPS = get_parameter(model_param, 'bc_MPS', 'finite', self.__class__)
        conserve = get_parameter(model_param, 'conserve', 'parity', self.__class__)
        assert conserve != 'Sz'
        unused_parameters(model_param, self.__class__)  # checks for mistyped parameters
        # 1-3)
        site = SpinHalfSite(conserve=conserve)
        # 4) lattice
        lat = Chain(L, site, bc_MPS=bc_MPS)
        bc_coupling = 'periodic' if bc_MPS == 'infinite' else 'open'
        # 5) initialize CouplingModel
        CouplingModel.__init__(self, lat, bc_coupling)
        # 6) add terms of the Hamiltonian
        # (u is always 0 as we have only one site in the unit cell)
        self.add_onsite(-np.asarray(g), 0, 'Sigmaz')
        J = np.asarray(J)
        if conserve is None:
            self.add_coupling(-J, 0, 'Sigmax', 0, 'Sigmax', 1)
        else:
            # individual 'Sigmax' does not conserve parity; rewrite in terms of Sp and Sm
            self.add_coupling(-J, 0, 'Sp', 0, 'Sp', 1)
            self.add_coupling(-J, 0, 'Sp', 0, 'Sm', 1)
            self.add_coupling(-J, 0, 'Sm', 0, 'Sp', 1)
            self.add_coupling(-J, 0, 'Sm', 0, 'Sm', 1)
        # 7) initialize MPO
        MPOModel.__init__(self, lat, self.calc_H_MPO())
        # 8) initialize bonds (the order of 7/8 doesn't matter)
        NearestNeighborModel.__init__(self, lat, self.calc_H_bond())


class TFIModel2D(CouplingModel, MPOModel):
    r"""Transverse field Ising model on a square lattice.

    The Hamiltonian reads:

    .. math ::
        H = - \sum_{\langle i,j\rangle, i < j} \mathtt{J} \sigma^x_i \sigma^x_{j}
            - \sum_{i} \mathtt{g} \sigma^z_i

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs, each pair appearing
    exactly once.
    All parameters are collected in a single dictionary `model_param` and read out with
    :func:`~tenpy.tools.params.get_parameter`.

    Parameters
    ----------
    Lx, Ly : int
        Length of the chain in x- and y-direction.
    J, g : float | array
        Couplings as defined for the Hamiltonian above.
    bc_MPS : {'finite' | 'infinte'}
        MPS boundary conditions along the x-direction.
        For 'infinite' boundary conditions, repeat the unit cell in x-direction.
        Coupling boundary conditions in x-direction are chosen accordingly.
    bc_y : 'ladder' | 'cylinder'
        Boundary conditions in y-direction.
    conserve : None | 'parity'
        What should be conserved. See :class:`~tenpy.networks.Site.SpinSite`.
    order : string
        Ordering of the sites in the MPS, e.g. 'default', 'snake';
        see :meth:`~tenpy.models.lattice.Lattice.ordering`.
    """

    def __init__(self, model_param):
        # 0) read out/set default parameters
        Lx = get_parameter(model_param, 'Lx', 1, self.__class__)
        Ly = get_parameter(model_param, 'Ly', 4, self.__class__)
        J = get_parameter(model_param, 'J', 1., self.__class__)
        g = get_parameter(model_param, 'g', 1., self.__class__)
        bc_MPS = get_parameter(model_param, 'bc_MPS', 'infinite', self.__class__)
        bc_y = get_parameter(model_param, 'bc_y', 'cylinder', self.__class__)
        order = get_parameter(model_param, 'order', 'default', self.__class__)
        conserve = get_parameter(model_param, 'conserve', None, self.__class__)
        assert conserve != 'Sz'  # invalid!
        assert bc_y in ['cylinder', 'ladder']
        unused_parameters(model_param, self.__class__)  # checks for mistyped parameters
        # 1-3)
        site = SpinHalfSite(conserve=conserve)
        # 4) lattice
        lat = SquareLattice(Lx, Ly, site, order, bc_MPS=bc_MPS)
        bc_coupling_x = 'periodic' if bc_MPS == 'infinite' else 'open'
        bc_coupling_y = 'periodic' if bc_y == 'cylinder' else 'open'
        # 5) initialize CouplingModel
        CouplingModel.__init__(self, lat, [bc_coupling_x, bc_coupling_y])
        # 6) add terms of the Hamiltonian
        # (u is always 0 as we have only one site in the unit cell)
        self.add_onsite(-np.asarray(g), 0, 'Sigmaz')
        J = -np.asarray(J)
        if conserve is None:
            self.add_coupling(-J, 0, 'Sigmax', 0, 'Sigmax', [1, 0])
            self.add_coupling(-J, 0, 'Sigmax', 0, 'Sigmax', [0, 1])
        else:
            for op1, op2 in [('Sp', 'Sp'), ('Sp', 'Sm'), ('Sm', 'Sp'), ('Sm', 'Sm')]:
                self.add_coupling(-J, 0, op1, 0, op2, [1, 0])
                self.add_coupling(-J, 0, op1, 0, op2, [0, 1])
        # 7) initialize MPO
        MPOModel.__init__(self, lat, self.calc_H_MPO())
        # skip 8): not a NearestNeighborModel...
