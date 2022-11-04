"""Kitaev's exactly solvable toric code model.

As we put the model on a cylinder, the name "toric code" is a bit misleading, but it is the
established name for this model...
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np

from .lattice import Lattice, get_order, _parse_sites
from ..networks.site import SpinHalfSite
from .model import CouplingMPOModel
from ..tools.params import asConfig
from ..tools.misc import any_nonzero

__all__ = ['DualSquare', 'ToricCode']


class DualSquare(Lattice):
    """The dual lattice of the square lattice (again square).

    The sites in this lattice correspond to the vertical and horizontal (nearest neighbor) bonds
    of a common :class:`~tenpy.models.lattice.Square` lattice with the same dimensions `Lx, Ly`.

    .. plot ::

        import matplotlib.pyplot as plt
        from tenpy.models.toric_code import DualSquare
        from tenpy.models.lattice import Square
        plt.figure(figsize=(5, 5))
        ax = plt.gca()
        lat = DualSquare(4, 4, None, bc='periodic')
        sq = Square(4, 4, None, bc='periodic')
        sq.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.5*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()

    Parameters
    ----------
    Lx, Ly : int
        Dimensions of the original lattice. This lattice has `2*Lx*Ly` sites.
    sites : :class:`~tenpy.networks.site.Site`
        The sites for the horizontal (first entry) and vertical (second entry) bonds.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 2  #: the dimension of the lattice

    def __init__(self, Lx, Ly, sites, **kwargs):
        sites = _parse_sites(sites, 2)
        basis = np.eye(2)
        pos = np.array([[0., 0.5], [0.5, 0.]])
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        NN = [(1, 0, np.array([0, 0])), (1, 0, np.array([1, 0])), (0, 1, np.array([-1, 1])),
              (0, 1, np.array([0, 1]))]
        nNN = [(i, i, dx) for i in [0, 1] for dx in [np.array([1, 0]), np.array([0, 1])]]
        nnNN = [(i, i, dx) for i in [0, 1] for dx in [np.array([1, 1]), np.array([-1, 1])]]
        kwargs.setdefault('pairs', {})
        kwargs['pairs'].setdefault('nearest_neighbors', NN)
        kwargs['pairs'].setdefault('next_nearest_neighbors', nNN)
        kwargs['pairs'].setdefault('next_next_nearest_neighbors', nnNN)
        super().__init__([Lx, Ly], sites, **kwargs)

    def ordering(self, order):
        """Provide possible orderings of the `N` lattice sites.

        The following orders are defined in this method compared to
        :meth:`tenpy.models.lattice.Lattice.ordering`:

        ================== =========================== =============================
        `order`            equivalent `priority`       equivalent ``snake_winding``
        ================== =========================== =============================
        ``'default'``      (0, 2, 1)                   (False, False, False)
        ================== =========================== =============================
        """
        if isinstance(order, str):
            if order == "default":
                priority = (0, 2, 1)
                snake_winding = (False, False, False)
                return get_order(self.shape, snake_winding, priority)
        return super().ordering(order)


class ToricCode(CouplingMPOModel):
    r"""Toric code model.

    The Hamiltonian reads:

    .. math ::
        H = - \mathtt{Jv} \sum_{vertices v} \prod_{i \in v}  \sigma^x_i
            - \mathtt{Jp} \sum_{plaquettes p} \prod_{i \in p} \sigma^z_i

    (Note that this are Pauli matrices, not spin-1/2 operators.)
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    .. versionchanged :: 0.7.2-98
        There was a bug that the terms for Jv and Jp were added with a positive instead of
        a negative sign.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`ToricCode` below.

    Options
    -------
    .. cfg:config :: ToricCode
        :include: CouplingMPOModel

        Lx, Ly: int
            Dimension of the lattice, number of plaquettes around the cylinder.
        conserve : 'parity' | None
            What should be conserved. See :class:`~tenpy.networks.Site.SpinHalfSite`.
        sort_charge : bool | None
            Whether to sort by charges of physical legs.
            See change comment in :class:`~tenpy.networks.site.Site`.
        Jv, Jp : float | array
            Couplings as defined for the Hamiltonian above.
        order : str
            The order of the lattice sites in the lattice, see :class:`DualSquare`.
        bc_y : ``"open" | "periodic"``
            The boundary conditions in y-direction.
        bc_x : ``"open" | "periodic"``
            Can be used to force "periodic" boundaries for the lattice,
            i.e., for the couplings in the Hamiltonian, even if the MPS is finite.
            Defaults to ``"open"`` for ``bc_MPS="finite"`` and
            ``"periodic"`` for ``bc_MPS="infinite``.
            If you are not aware of the consequences, you should probably
            *not* use "periodic" boundary conditions:
            The MPS is still "open", so this will introduce long-range couplings between the
            first and last sites of the MPS, and require **squared** MPS bond-dimensions.
    """
    default_lattice = DualSquare
    force_default_lattice = True

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'parity')
        sort_charge = model_params.get('sort_charge', None)
        site = SpinHalfSite(conserve, sort_charge=sort_charge)
        return site

    def init_terms(self, model_params):
        Jv = np.asarray(model_params.get('Jv', 1.))
        Jp = np.asarray(model_params.get('Jp', 1.))
        # vertex/star term
        self.add_multi_coupling(-Jv, [('Sigmax', [0, 0], 1), ('Sigmax', [0, 0], 0),
                                      ('Sigmax', [-1, 0], 1), ('Sigmax', [0, -1], 0)])
        # plaquette term
        self.add_multi_coupling(-Jp, [('Sigmaz', [0, 0], 1), ('Sigmaz', [0, 0], 0),
                                      ('Sigmaz', [0, 1], 1), ('Sigmaz', [1, 0], 0)])
        # done
