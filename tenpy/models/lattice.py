"""Classes to define the lattice structure of a model.

.. todo :
    documentation, examples, ...
"""

import numpy as np

from ..linalg import np_conserved as npc


class Site(object):
    """Collects necessary information about a single local site of a lattice.

    It has the on-site operators as attributes, e.g. ``self.Id`` is the identy.

    Parameters
    ----------
    charges : :class:`npc.LegCharge`
        Charges of the physical state.
    state_labels : None | list
        List of labels for the local basis states. ``None`` entries are ignored / not set.
    **site_ops :
        Additional keyword arguments of the form ``name=op`` given to :meth:`add_op`.
        The identity operator 'Id' is always included.

    Attributes
    ----------
    dim
    onsite_ops
    leg : :class:`npc.LegCharge`
        Charges of the local basis states.
    state_labels : dict
        (Optional) labels for the local basis states.
    opnames : set
        Labels of all onsite operators (i.e. ``self.op`` exists if ``'op'`` in ``self.opnames``).
    ops : :class:`npc.Array`
        Onsite operators are added directly as attributes to self.
        For example after ``self.add_op('Sz', Sz)`` you can use ``self.Sz`` for the `Sz` operator.

    Examples
    --------
    The following generates a site for spin-1/2 with Sz conservation.
    Note that ``Sx = (Sp + Sm)/2`` violates Sz conservation and is thus not a valid
    on-site operator.

    >>> chinfo = npc.ChargeInfo([1], ['Sz'])
    >>> ch = npc.LegCharge.from_qflat(chinfo, [1, -1])
    >>> Sp = [[0, 1.], [0, 0]]
    >>> Sm = [[0, 0], [1., 0]]
    >>> Sz = [[0.5, 0], [0, -0.5]]
    >>> site = Site(ch, ['up', 'down'], Splus=Sp, Sminus=Sm, Sz=Sz)
    >>> print site.Splus.to_ndarray()
    array([[ 0.,  1.],
           [ 0.,  0.]])

    .. todo :
        Problem: what if we later want to remove the charges / add new charges?!?
        Some onsite op's might not be compatible with charges, although the resulting
        Hamiltonian might be?
    .. todo :
        add option to sort by charges and save the resulting permutation.
    .. todo :
        need clever way to handle Jordan-Wigner strings for fermions...
    """
    def __init__(self, charges, state_labels=None, **site_ops):
        self.leg = charges
        self.state_labels = dict(state_labels)
        self.opnames = set()
        self.add_op('Id', npc.diag(1., self.leg))
        for name, op in site_ops:
            self.add_op(name, op)
        self.test_sanity()

    def test_sanity(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
        for lab, ind in self.state_labels:
            if type(lab) != str:
                raise ValueError("wrong type of state label")
            if not 0 <= ind < self.dim:
                raise ValueError("index of state label out of bounds")
        for name in self.opnames:
            if not hasattr(self, name):
                raise ValueError("missing onsite operator " + name)
        for op in self.onsite_ops.values():
            if op.rank != 2:
                raise ValueError("only rank-2 onsite operators allowed")
            op.legs[0].test_equal(self.leg)
            op.legs[1].test_contractible(self.leg)

    @property
    def dim(self):
        """Dimension of the local Hilbert space"""
        return self.leg.ind_len

    @property
    def onsite_ops(self):
        """dictionary of on-site operators for iteration.

        (single operators are accessible as attributes.)"""
        return dict([(name, getattr(self, name)) for name in sorted(self.opnames)])

    def add_op(self, name, op):
        """add one or multiple on-site operators

        Parameters
        ----------
        name : str
            A valid python variable name, used to label the operator.
            The name under which `op` is added as attribute to self.
        op : np.ndarray | npc.Array
            A matrix acting on the local hilbert space representing the local operator.
            Dense numpy arrays are automatically converted to :class:`npc.Array`.
        """
        name = str(name)
        if name in self.opnames:
            raise ValueError("operator with that name already existent: " + name)
        if hasattr(self, name):
            raise ValueError("Site already has that attribute name: " + name)
        if not isinstance(op, npc.Array):
            op = np.asarray(op)
            if op.shape != (self.dim, self.dim):
                raise ValueError("wrong shape of on-site operator")
            # try to convert op into npc.Array
            op = npc.Array.from_ndarray(op, self.leg.chinfo, [self.leg, self.leg.conj()])
        if op.rank != 2:
            raise ValueError("only rank-2 on-site operators allowed")
        op.legs[0].test_equal(self.leg)
        op.legs[1].test_contractible(self.leg)
        setattr(self, name, op)
        self.opnames.add(name)

    def get_state_index(self, label):
        """return index of a basis state from its label.

        Parameters
        ----------
        label : int | string
            eather the index directly or a label (string) set before.

        Returns
        -------
        state_index : int
            the index of the basis state associated with the label.
        """
        res = self.state_labels.get(label, label)
        try:
            res = int(res)
        except:
            raise KeyError("label not found: " + repr(label))
        return res


class Lattice(object):
    r"""Collects lattice sites into a lattice.

    The lattice consists of a `unit_cell` which is repeated in `D` different directions.
    A site of the lattice is thus identified by (1) `D` indices ``a = (a_0, ...,  a_{D-1})`` with
    ``0 <= a_i < Ls[i]`` picking the unit cell in combination with (2) an index ``j``
    picking the site within the unit cell. It is located in 'real space' at
    ``sum_i a_i*basis[i] + unit_cell_positions[j]``.

    In addition to the pure geometry, this class also defines an 'order' of all sites.
    This order maps the lattice to a finite 1D chain and defines the geometry of MPSs and MPOs.

    Parameters
    ----------
    Ls : list of int
        the length in each direction
    unit_cell : list of :class:`Site`
        the lattice sites making up a unit cell of the lattice.
    order : str
        a string specifying the order, given to :meth:`ordering`.
        Defaults ``'default'``: First direction changing slowest, within the unit cell fastest.
    bc_MPS : {'finite', 'segment', 'infinite'}
        boundary conditions for an MPS/MPO living on the ordered lattice. Default 'finite'.
    basis : ndarray, shape (D, D)
        translation vectors shifting the unit cell. The ``i``th row gives the vector shifting in
        direction ``i``. Defaults to the standard orthonormal basis ``np.eye(D)``.
    positions : ndarray, shape (len(unit_cell), D)
        for each site of the unit cell a vector giving its position within the unit cell.
        Defaults to ``np.zeros``.

    Attributes
    ----------
    dim
    N_sites
    chinfo : :class:`npc.ChargeInfo`
        The nature of the charge (which is the same for all sites).
    unit_cell : list of :class:`Site`
        the lattice sites making up a unit cell of the lattice.
    site_labels : dict(str -> int)
        optional names of the sites in the unit cell.
    Ls : list of int
        the length in each direction.
    order : ndarray, shape (N_sites, dim+1)
        Defines an ordering of the lattice sites, thus mapping the lattice to a 1D chain.
        This order defines how an MPS/MPO winds through the lattice.
    bc_MPS : {'finite', 'segment', 'infinite'}
        boundary conditions for an MPS/MPO living on the ordered lattice.
    basis: ndarray, shape (dim, dim)
        translation vectors shifting the unit cell. The ``i``th row gives the vector shifting in
        direction ``i``.
    unit_cell_positions : ndarray, shape (len(unit_cell), dim)
        for each site in the unit cell a vector giving its position within the unit cell.

    .. todo :
        what are valid values for MPS boundary conditions? -> need to define MPS class first...

    .. todo :
        some way to define what are the 'nearest neighbours'/'next nearest neighbours'?
    """
    def __init__(self, Ls, unit_cell, order='default', bc_MPS='finite', basis=None, positions=None):
        self.Ls = [int(L) for L in Ls]
        self.unit_cell = list(unit_cell)
        self.chinfo = self.unit_cell[0].leg.chinfo
        for site in unit_cell:
            if site.chinfo != self.chinfo:
                raise ValueError("All sites must have the same ChargeInfo!")
        if positions is None:
            positions = np.zeros((len(self.unit_cell), self.dim))
        if basis is None:
            basis = np.eye(self.dim)
        self.unit_cell_positions = positions
        self.basis = basis
        self.order = self.ordering(order)

    def test_sanity(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
        for site in self.unit_cell:
            if not isinstance(site, Site):
                raise ValueError("element of Unit cell is not Site.")
            if site.leg.chinfo != self.chinfo:
                raise ValueError("All sites must have the same ChargeInfo!")
        if self.basis.shape[0] != self.dim:
            raise ValueError("Need one basis vector for each direction!")
        if self.unit_cell_positions.shape[0] != len(self.unit_cell):
            raise ValueError("Need one position for each site in the unit cell.")

    @property
    def dim(self):
        """the dimension of the lattice."""
        return len(self.Ls)

    @property
    def N_sites(self):
        """the number of sites in the lattice"""
        N = self.Ls[0]
        for L in self.Ls[1:]:
            N *= L
        return N*len(self.unit_cell)

    def ordering(self, name):
        """Provide possible orderings of the `N` lattice sites.

        This function can be overwritten by derived lattices to define additional orderings.
        The following orders are devined in this function:

        =========== =============================================================================
        name        ordering
        =========== =============================================================================
        Cstyle      First ascending within the unit cell, then through the lattice in
                    C-style array order, i.e., the first direction changes slowest.
        Fstyle      Similar as default, but use Fortran-style array order for the lattice sites,
                    i.e., the last dimension changes slowest, unit cell fastest.
        snakeCstyle like Cstyle but always alternate ascending/descending.
        snakeFstyle like Fstyle but always alternate ascending/descending.
        default     same as Cstyle
        snake       same as snakeCstyle
        =========== =============================================================================

        Parameters
        ----------
        name : str
            specifies the desired ordering, see table above.

        Returns
        -------
        order : array, shape (N, D+1), dtype np.intp
            the order to be used for ``self.order``.

        See also
        --------
        plot_ordering : visualizes the ordering
        """
        res = np.empty((self.N_sites, self.dim+1), np.intp)
        if name in ["default", "Cstyle"]:
            shapes = self.Ls + [len(self.unit_cell)]
            res = np.mgrid[tuple([slice(0, L) for L in shapes])]
            return res.reshape((self.dim+1, self.N_sites)).T
        elif name == "Fstyle":
            shapes = self.Ls[::-1] + [len(self.unit_cell)]
            res = np.mgrid[tuple([slice(0, L) for L in shapes])]
            res = res.transpose([0] + range(1, self.dim-1)[::-1] + [self.dim+2])
            return res.reshape((self.dim+1, self.N_sites)).T
        elif name in ["snake", "snakeCstyle"]:
            return _ordering_snake(self.Ls+[len(self.unit_cell)])
        elif name == "snakeFstyle":
            res = _ordering_snake(self.Ls[::-1] + [len(self.unit_cell)])
            reorder = np.array(range(self.dim)[::-1]+[-1])
            return res[:, reorder]
        # in a derived lattice ``class DerivedLattice(Lattice)``, use:
        # return super(DerivedLattice, self).ordering(name)
        # such that the derived lattece also has the orderings defined in this function.
        raise ValueError("unknown ordering name" + str(name))

    def plot_ordering(self, order=None, axes=None):
        """Vizualize the ordering by plotting the lattice.

        .. todo :
            implement
        """
        # if order is None:
        #     order = self.order
        # import pylab as pl
        # if axes is None:
        #     axes = pl.gca()
        raise NotImplementedError() # TODO

    def position(self, site_indices):
        """return 'real space' position of one or multiple sites.

        Parameters
        ----------
        site_indices : array
        """
        idx = np.asarray(site_indices, dtype=np.intp)
        if idx.shape[-1] != self.dim + 1:
            raise ValueError("wrong number of indices")
        res = np.take(self.unit_cell_positions, idx[..., -1], axis=0)
        for i in range(self.dim):
            res += idx[..., i, np.newaxis] * self.basis[i]
        return res

    def mps2lat(self, i):
        """translate mps index `i` to lattice site index ``(x, y, ... , u)``"""
        return tuple(self.order[i])

    def lat2mps(self, site_index):
        """translate site_index ``(x, y, ..., u)`` to mps index `i`.

        .. todo :
            implement: i = perm[sum(site_index*strides)]
            perm and strides have to be calculated in __init__.
        """
        raise NotImplementedError()

    def site(self, i):
        """return :class:`Site` instance corresponding to an mps index"""
        return self.unit_cell[self.order[i, -1]]


class Chain(Lattice):
    """A simple chain of L equal sites."""
    def __init__(self, L, site, bc_MPS='finite'):
        super(Chain, self).__init__([L], [site], bc_MPS=bc_MPS)  # and otherwise default values.


def _ordering_snake(Ls):
    """built the order of a snake winding through a (hyper-)cubic lattice in Cstyle order."""
    order = np.emtpy((1, 0), dtype=np.intp)
    while len(Ls) > 0:
        L = Ls.pop()
        L0, D = order.shape
        new_order = np.empty((L*L0, D+1), dtype=np.intp)
        new_order[:, 0] = np.repeat(np.arange(L), order.shape[0])
        new_order[:L, 1:] = order
        if L > 1:
            # reverse order to go back for second index
            new_order[L:2*L, 1:] = order[::-1]
        if L > 2:
            # repeat (ascending, descending) up to length L
            rep = L // 2 - 1
            new_order[2*L:(rep+1)*2*L, :] = np.tile(new_order[:2*L, :], [rep, 1])
            if L % 2 == 1:
                new_order[-2*L:, :] = order
        order = new_order
    return order
