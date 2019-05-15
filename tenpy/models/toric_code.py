"""Kitaev's exactly solvable toric code model.

As we put the model on a cylinder, the name "toric code" is a bit misleading,
but it is the established name for this model...
"""
# Copyright 2018 TeNPy Developers

import numpy as np

from .lattice import Lattice, _parse_sites
from ..networks.site import SpinHalfSite
from .model import MultiCouplingModel, CouplingMPOModel
from ..tools.params import get_parameter, unused_parameters
from ..tools.misc import any_nonzero

__all__ = ['DualSquare', 'ToricCode']


class DualSquare(Lattice):
    """The dual lattice of the square lattice (again square).

    The sites in this lattice correspond to the vertical and horizontal (nearest neighbor) bonds
    of a common :class:`~tenpy.models.lattice.Square` lattice with the same dimensions `Lx, Ly`.

    .. image :: /images/lattices/DualSquare.*

    Parameters
    ----------
    Lx, Ly : int
        Dimensions of the original lattice. This lattice has `2*Lx*Ly` sites.
    sites : :class:`~tenpy.networks.site.Site`
        The sites for the horizontal (first entry) and vertical (second entry) bonds.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `[[next_]next_]nearest_neighbors` are set accordingly.
    """

    def __init__(self, Lx, Ly, sites, **kwargs):
        sites = _parse_sites(sites, 2)
        basis = np.eye(2)
        pos = np.array([[0.5, 0.], [0., 0.5]])
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        NN = [(0, 1, np.array([0, 0])), (0, 1, np.array([1, 0])), (1, 0, np.array([-1, 1])),
              (1, 0, np.array([0, 1]))]
        nNN = [(i, i, dx) for i in [0, 1] for dx in [np.array([1, 0]), np.array([0, 1])]]
        nnNN = [(i, i, dx) for i in [0, 1] for dx in [np.array([1, 1]), np.array([-1, 1])]]
        kwargs.setdefault('nearest_neighbors', NN)
        kwargs.setdefault('next_nearest_neighbors', nNN)
        kwargs.setdefault('next_nearest_neighbors', nnNN)
        kwargs.setdefault('next_next_nearest_neighbors', nnNN)
        super().__init__([Lx, Ly], sites, **kwargs)


class ToricCode(CouplingMPOModel, MultiCouplingModel):
    r"""Spin-S sites coupled by nearest neighbour interactions.

    The Hamiltonian reads:

    .. math ::
        H = - \mathtt{Jv} \sum_{vertices v} \prod_{i \in v}  \sigma^x_i
            - \mathtt{Jp} \sum_{plaquettes p} \prod_{i \in p} \sigma^z_i

    (Note that this are Pauli matrices, not spin-1/2 operators.)
    All parameters are collected in a single dictionary `model_params` and read out with
    :func:`~tenpy.tools.params.get_parameter`.

    Parameters
    ----------
    Lx, Ly : int
        Dimension of the lattice, number of plaquettes around the cylinder.
    conserve : 'parity' | None
        What should be conserved. See :class:`~tenpy.networks.Site.SpinHalfSite`.
    Jc, Jp: float | array
        Couplings as defined for the Hamiltonian above.
    bc_MPS : {'finite' | 'infinte'}
        MPS boundary conditions. Coupling boundary conditions are chosen appropriately.
    order : str
        The order of the lattice sites in the lattice, see :class:`DualSquare`.
    """

    def __init__(self, model_params):
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        conserve = get_parameter(model_params, 'conserve', 'parity', self.name)
        site = SpinHalfSite(conserve)
        return site

    def init_lattice(self, model_params):
        site = self.init_sites(model_params)
        Lx = get_parameter(model_params, 'Lx', 2, self.name)
        Ly = get_parameter(model_params, 'Ly', 2, self.name)
        order = get_parameter(model_params, 'order', 'default', self.name)
        bc_MPS = get_parameter(model_params, 'bc_MPS', 'infinite', self.name)
        bc = [None, 'periodic']
        bc[0] = 'periodic' if bc_MPS == 'infinite' else 'open'
        lat = DualSquare(Lx, Ly, site, order=order, bc=bc, bc_MPS=bc_MPS)
        return lat

    def init_terms(self, model_params):
        Jv = get_parameter(model_params, 'Jv', 1., self.name, True)
        Jp = get_parameter(model_params, 'Jp', 1., self.name, True)
        # vertex/star term
        self.add_multi_coupling(Jv, 0, 'Sigmax', [(1, 'Sigmax', [0, 0]), (0, 'Sigmax', [-1, 0]),
                                                  (1, 'Sigmax', [0, -1])])
        # plaquette term
        self.add_multi_coupling(Jp, 0, 'Sigmaz', [(1, 'Sigmaz', [0, 0]), (0, 'Sigmaz', [0, 1]),
                                                  (1, 'Sigmaz', [1, 0])])
        # done
