"""Classes to define the lattice structure of a model.

.. todo :
    documentation, how to generate new lattices, examples, ...
"""

from __future__ import division

import numpy as np

from ..linalg import np_conserved as npc
from ..tools.misc import to_iterable


class Site(object):
    """Collects necessary information about a single local site of a lattice.

    It has the on-site operators as attributes, e.g. ``self.Id`` is the identy.

    Parameters
    ----------
    leg : :class:`npc.LegCharge`
        Charges of the physical states, to be used for the physical leg of MPS & co).
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
    state_labels : iterable of (str, int)
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
        if state_labels is None:
            state_labels = []
        self.state_labels = dict([(k, v % self.dim) for k, v in state_labels])
        self.opnames = set()
        self.add_op('Id', npc.diag(1., self.leg))
        for name, op in site_ops.iteritems():
            self.add_op(name, op)
        self.test_sanity()

    def test_sanity(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
        for lab, ind in self.state_labels.iteritems():
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
            op.test_sanity()

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
            LegCharges have to be [leg, leg.conj()].
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
        op.test_sanity()
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
    r"""A general lattice.

    The lattice consists of a **unit cell** which is repeated in `dim` different directions.
    A site of the lattice is thus identified by **lattice indices** ``(x_0, ..., x_{dim-1}, u)``,
    where ``0 <= x_l < Ls[l]`` pick the position of the unit cell in the lattice and
    ``0 <= u < len(unit_cell)`` picks the site within the unit cell. The site is located
    in 'space' at ``sum_l x_l*basis[l] + unit_cell_positions[u]`` (see :meth:`position`).

    In addition to the pure geometry, this class also defines an 'order' of all sites.
    This order maps the lattice to a finite 1D chain and defines the geometry of MPSs and MPOs.
    The **MPS index** `i` corresponds thus to the lattice sites given by
    ``(a_0, ..., a_{D-1}, u) = tuple(self.order[i])``.
    Use :meth:`mps2lat_idx` and :meth:`lat2mps_idx` for conversion of indices.

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
    basis : iterable of 1D arrays
        for each direction one translation vectors shifting the unit cell.
        Defaults to the standard ONB ``np.eye(dim)``.
    positions : iterable of 1D arrays
        for each site of the unit cell the position within the unit cell.
        Defaults to ``np.zeros((len(unit_cell), dim))``.

    Attributes
    ----------
    dim
    N_cells
    N_sites
    Ls : tuple of int
        the length in each direction.
    shape : tuple of int
        the 'shape' of the lattice, same as ``Ls + (len(unit_cell), )``
    chinfo : :class:`npc.ChargeInfo`
        The nature of the charge (which is the same for all sites).
    unit_cell : list of :class:`Site`
        the lattice sites making up a unit cell of the lattice.
    order : ndarray (N_sites, dim+1)
        Defines an ordering of the lattice sites, thus mapping the lattice to a 1D chain.
        This order defines how an MPS/MPO winds through the lattice.
    bc_MPS : {'finite', 'segment', 'infinite'}
        boundary conditions for an MPS/MPO living on the ordered lattice.
    basis: ndarray (dim, dim)
        translation vectors shifting the unit cell. The ``i``th row gives the vector shifting in
        direction ``i``.
    unit_cell_positions : ndarray, shape (len(unit_cell), dim)
        for each site in the unit cell a vector giving its position within the unit cell.
    _strides : ndarray (dim, )
        necessary for :meth:`mps2lat`
    _perm : ndarray (N, )
        permutation needed to make `order` lexsorted.
    _mps2lat_vals_idx : ndarray `shape`
        index array for reshape/reordering in :meth:`mps2lat_vals`
    _mps_fix_u : tuple of ndarray (N_cells, ) np.intp
        for each site of the unit cell an index array selecting the mps indices of that site.
    _mps_fix_u_None : ndarray (N_sites, )
        just np.arange(N_sites, np.intp)
    _mps2lat_vals_idx_fix_u : tuple of ndarray of shape `Ls`
        similar as `_mps2lat_vals_idx`, but for a fixed `u` picking a site from the unit cell.

    .. todo :
        what are valid values for MPS boundary conditions? -> need to define MPS class first...
    .. todo :
        some way to define what are the 'nearest neighbours'/'next nearest neighbours'?
    """
    def __init__(self,
                 Ls,
                 unit_cell,
                 order='default',
                 bc_MPS='finite',
                 basis=None,
                 positions=None):
        self.Ls = tuple([int(L) for L in Ls])
        self.unit_cell = list(unit_cell)
        self.N_cells = int(np.prod(self.Ls))
        self.shape = self.Ls + (len(unit_cell), )
        self.chinfo = self.unit_cell[0].leg.chinfo
        self.N_sites = int(np.prod(self.shape))
        if positions is None:
            positions = np.zeros((len(self.unit_cell), self.dim))
        if basis is None:
            basis = np.eye(self.dim)
        self.unit_cell_positions = np.asarray(positions)
        self.basis = np.asarray(basis)
        # calculate order for MPS
        self.order = self.ordering(order)
        # from order, calc necessary stuff for mps2lat and lat2mps
        self._perm = np.lexsort(self.order.T)
        # use advanced numpy indexing...
        self._mps2lat_vals_idx = np.empty(self.shape, np.intp)
        self._mps2lat_vals_idx[tuple(self.order.T)] = np.arange(self.N_sites)
        # versions for fixed u
        self._mps_fix_u = []
        self._mps2lat_vals_idx_fix_u = []
        for u in range(len(self.unit_cell)):
            mps_fix_u = np.nonzero(self.order[:, -1] == u)[0]
            self._mps_fix_u.append(mps_fix_u)
            mps2lat_vals_idx = np.empty(self.Ls, np.intp)
            mps2lat_vals_idx[tuple(self.order[mps_fix_u, :-1].T)] = np.arange(self.N_cells)
            self._mps2lat_vals_idx_fix_u.append(mps2lat_vals_idx)
        self._mps_fix_u = tuple(self._mps_fix_u)
        # calculate _strides
        strides = [1]
        for L in self.Ls:
            strides.append(strides[-1]*L)
        self._strides = np.array(strides, np.intp)
        self.test_sanity()  # check consistency

    def test_sanity(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
        assert self.shape == self.Ls + (len(self.unit_cell), )
        assert self.N_cells == np.prod(self.Ls)
        assert self.N_sites == np.prod(self.shape)
        for site in self.unit_cell:
            if not isinstance(site, Site):
                raise ValueError("element of Unit cell is not Site.")
            if site.leg.chinfo != self.chinfo:
                raise ValueError("All sites must have the same ChargeInfo!")
            site.test_sanity()
        if self.basis.shape[0] != self.dim:
            raise ValueError("Need one basis vector for each direction!")
        if self.unit_cell_positions.shape[0] != len(self.unit_cell):
            raise ValueError("Need one position for each site in the unit cell.")
        if self.basis.shape[1] != self.unit_cell_positions.shape[1]:
            raise ValueError("Different space dimensions of `basis` and `unit_cell_positions`")
        # if one of the following assert fails, the `ordering` function returned an invalid array
        assert np.all(self.order >= 0) and np.all(self.order <= self.shape)  # entries of `order`
        assert np.all(np.sum(self.order * self._strides, axis=1)[self._perm]
                      == np.arange(self.N_sites))  # rows of `order` unique?

    @property
    def dim(self):
        """the dimension of the lattice."""
        return len(self.Ls)

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
            res = np.mgrid[tuple([slice(0, L) for L in self.shape])]
            return res.reshape((self.dim+1, self.N_sites)).T
        elif name == "Fstyle":
            shape = self.Ls[::-1] + (len(self.unit_cell), )
            res = np.mgrid[tuple([slice(0, L) for L in shape])]
            res = res.reshape((self.dim+1, self.N_sites)).T
            perm = np.array(range(self.dim)[::-1] + [-1])
            return res[:, perm]
        elif name in ["snake", "snakeCstyle"]:
            return _ordering_snake(self.shape)
        elif name == "snakeFstyle":
            res = _ordering_snake(self.Ls[::-1] + (len(self.unit_cell), ))
            perm = np.array(range(self.dim)[::-1]+[-1])
            return res[:, perm]
        # in a derived lattice ``class DerivedLattice(Lattice)``, use:
        # return super(DerivedLattice, self).ordering(name)
        # such that the derived lattece also has the orderings defined in this function.
        raise ValueError("unknown ordering name" + str(name))

    def plot_ordering(self, order=None, ax=None):
        """Vizualize the ordering by plotting the lattice.

        Parameters
        ----------
        order : None | 2D array (self.N_sites, self.dim+1)
            An order array as returned by :meth:`ordering`. ``None`` defaults to ``self.order``.
        ax : matplotlib.pyplot.Axes
            The axes on which the ordering should be plotted. Defaults to ``pylab.gca()``.
        """
        if order is None:
            order = self.order
        import pylab as pl
        if ax is None:
            ax = pl.gca()
        pos = self.position(order)
        D = pos.shape[1]
        styles = ['o', '^', 's', 'p', 'h', 'D', 'd', 'v', '<', '>']
        if D == 1:
            ax.plot(pos[:, 0], np.zeros(len(pos)), 'r-')
            for u in range(len(self.unit_cell)):
                p = pos[self.mps_idx_fix_u(u), 0]
                ax.plot(p, np.zeros(len(p)), styles[u % len(styles)])
            for i, p in enumerate(pos):
                ax.text(p[0], 0.1, str(i))
        elif D == 2:
            ax.plot(pos[:, 0], pos[:, 1], 'r-')
            for u in range(len(self.unit_cell)):
                p = pos[self.mps_idx_fix_u(u), :]
                ax.plot(p[:, 0], p[:, 1], styles[u % len(styles)])
            for i, p in enumerate(pos):
                ax.text(p[0], p[1], str(i))
        else:
            raise NotImplementedError()  # D >= 3

    def position(self, lat_idx):
        """return 'space' position of one or multiple sites.

        Parameters
        ----------
        lat_idx : ndarray, ``(... , dim+1)``
            lattice indices

        Returns
        -------
        pos : ndarray, ``(..., dim)``
        """
        idx = self._asvalid_latidx(lat_idx)
        res = np.take(self.unit_cell_positions, idx[..., -1], axis=0)
        for i in range(self.dim):
            res += idx[..., i, np.newaxis] * self.basis[i]
        return res

    def site(self, i):
        """return :class:`Site` instance corresponding to an MPS index `i`"""
        return self.unit_cell[self.order[i, -1]]

    def mps2lat_idx(self, i):
        """translate MPS index `i` to lattice indices ``(x_0, ..., x_{D_1}, u)``"""
        return tuple(self.order[i])

    def lat2mps_idx(self, lat_idx):
        """translate lattice indices ``(x_0, ..., x_{D-1}, u)`` to MPS index `i`."""
        i = np.sum(self._asvalid_latidx(lat_idx)*self._strides, axis=-1)
        return self._perm[i]

    def mps_idx_fix_u(self, u=None):
        """return an index array of MPS indices for which the site within the unit cell is `u`.

        If you have multiple sites in your unit-cell, an onsite operator is in general not defined
        for all sites. This functions returns an index array of the mps indices which belong to
        sites given by ``self.unit_cell[u]``.

        Parameters
        ----------
        u : None | int
            Selects a site of the unit cell. ``None`` (default) means all sites.

        Returns
        -------
        mps_idx : array
            MPS indices for which ``self.site(i) is self.unit_cell[u]``.
        """
        if u is not None:
            return self._mps_fix_u[u]
        return np.arange(self.N_sites, dtype=np.intp)

    def mps_lat_idx_fix_u(self, u=None):
        """Similar as :meth:`mps_idx_fix_u`, but return also the corresponding lattice indices.

        Parameters
        ----------
        u : None | int
            Selects a site of the unit cell. ``None`` (default) means all sites.

        Returns
        -------
        mps_idx : array
            MPS indices `i` for which ``self.site(i) is self.unit_cell[u]``.
        lat_idx : 2D array
            The `j`th row contains the lattice index (without `u`) corresponding to ``mps_idx[j]``.
        """
        mps_idx = self.mps_idx_fix_u(self)
        return mps_idx, self.order[mps_idx, :-1]

    def mps2lat_values(self, A, axes=0, u=None):
        """reshape/reorder A to replace an MPS index by lattice indices.

        Parameters
        ----------
        A : ndarray
            some values. Must have ``A.shape[axes] = self.N_sites`` if `u` is ``None``, or
            ``A.shape[axes] = self.N_cells`` if `u` is an int.
        axes : (iterable of) int
            chooses the axis which should be replaced.
        u : ``None`` | int
            Optionally choose a subset of MPS indices present in the axes of `A`, namely the
            indices corresponding to ``self.unit_cell[u]``, as returned by :meth:`mps_idx_fix_u`.
            The resulting array will not have the additional dimension(s) of `u`.

        Returns
        -------
        res_A : ndarray
            reshaped and reordered verions of A. Such that an MPS index `j` is replaced by
            ``res_A[..., self.order, ...] = A[..., np.arange(self.N_sites), ...]``

        Example
        -------
        Say you measure expection values of an onsite term for an MPS, which gives you an 1D array
        `A`, where `A[i]` is the expectation value of the site given by ``self.mps2lat_idx(i)``.
        Then this function gives you the expectation values ordered by the lattice:

        >>> print lat.shape, A.shape
        (10, 3, 2) (60,)
        >>> A_res = lat.mps2lat_values(A)
        >>> A_res.shape
        (10, 3, 2)
        >>> A_res[lat.mps2lat_idx(5)] == A[5]
        True

        If you have a correlation function ``C[i, j]``, it gets just slightly more complicated:

        >>> print lat.shape, C.shape
        (10, 3, 2) (60, 60)
        >>> lat.mps2lat_values(C, axes=[0, 1]).shape
        (10, 3, 2, 10, 3, 2)

        If the unit cell consists of different physical sites, an onsite operator might be defined
        only on one of the sites in the unit cell. Then you can use :meth:`mps_idx_fix_u` to get
        the indices of sites it is defined on, measure the operator on these sites, and use
        the argument `u` of this function. say y

        >>> u = 0
        >>> idx_subset = lat.mps_idx_fix_u(u)
        >>> A_u = A[idx_subset]
        >>> A_u_res = lat.mps2lat_values(A_u, u=u)
        >>> A_u_res.shape
        (10, 3)
        >>> np.all(A_res[:, :, u] == A_u_res[:, :])
        True

        .. todo :
            make sure this function is used for expectation values...
        """
        axes = to_iterable(axes)
        if len(axes) > 1:
            axes = [(ax + A.ndim if ax < 0 else ax) for ax in axes]
            for ax in reversed(sorted(axes)):  # need to start with largest axis!
                A = self.mps2lat_values(A, ax, u)  # recursion with single axis
            return A
        # choose the appropriate index arrays calcuated in __init__
        if u is None:
            idx = self._mps2lat_vals_idx
        else:
            idx = self._mps2lat_vals_idx_fix_u[u]
        return np.take(A, idx, axis=axes[0])

    def _asvalid_latidx(self, lat_idx):
        """convert lat_idx to ndarray with valid entries >=0."""
        lat_idx = np.asarray(lat_idx, dtype=np.intp)
        if lat_idx.shape[-1] != len(self.shape):
            raise ValueError("wrong len of last dimension of lat_idx: " + str(lat_idx.shape))
        lat_idx = np.choose(lat_idx < 0, [lat_idx, lat_idx + self.shape])
        if np.any(lat_idx < 0) or np.any(lat_idx >= self.shape):
            raise IndexError("lattice index out of bonds")
        return lat_idx


class SimpleLattice(Lattice):
    """A lattice with a unit cell consiting of just a single site.

    In many cases, the unit cell consists just of a single site, such that the the last entry of
    `u` of an 'lattice index' can only be ``0``.
    From the point of internal algorithms, we handle this class like a :class:`Lattice` --
    in that way we don't need to distinguish special cases in the algorithms.

    Yet, from the point of a tenpy user, for example if you measure and expectation value
    on each site in a `SimpleLattice`, you expect to get an ndarray of dimensions ``self.Ls``,
    not ``self.shape``. To avoid that problem, `SimpleLattice` overwrites just the meaning of
    ``u=None`` in :meth:`mps2lat_values` to be the same as ``u=0``.

    Parameters
    ----------
    Ls : list of int
        the length in each direction
    site : :class:`Site`
        the lattice site. The `unit_cell` of the :class:`Lattice` is just ``[site]``.
    order : str
        A string specifying the order, given to :meth:`ordering`.
        Defaults ``'default'``: First direction changing slowest.
    bc_MPS : {'finite', 'segment', 'infinite'}
        boundary conditions for an MPS/MPO living on the ordered lattice. Default 'finite'.
    basis : iterable of 1D arrays
        for each direction one translation vectors shifting the unit cell.
        Defaults to the standard ONB ``np.eye(dim)``.
    position : 1D array
        The position of the site within the unit cell. Defaults to ``np.zeros(dim))``.
    """
    def __init__(self, Ls, site, order='default', bc_MPS='finite', basis=None, position=None):
        if position is not None:
            position = [position]
        super(SimpleLattice, self).__init__(Ls, [site], order, bc_MPS, basis, position)

    def mps2lat_values(self, A, axes=0, u=None):
        """same as :meth:`Lattice.mps2lat_values`, but ignore ``u``, setting it to ``0``."""
        super(SimpleLattice, self).mps2lat_values(A, axes, 0)


class Chain(SimpleLattice):
    """A simple uniform chain of L equal sites."""
    def __init__(self, L, site, order='default', bc_MPS='finite'):
        super(Chain, self).__init__([L], site, order, bc_MPS)  # and otherwise default values.


class SquareLattice(SimpleLattice):
    """A simple uniform square lattice of `Lx` by `Lx` sites."""
    def __init__(self, Lx, Ly, site, order='default', bc_MPS='finite'):
        super(SquareLattice, self).__init__([Lx, Ly], site, order, bc_MPS)


def _ordering_snake(Ls):
    """built the order of a snake winding through a (hyper-)cubic lattice in Cstyle order."""
    Ls = list(Ls)
    order = np.empty((1, 0), dtype=np.intp)
    while len(Ls) > 0:
        L = Ls.pop()
        L0, D = order.shape
        new_order = np.empty((L*L0, D+1), dtype=np.intp)
        print order.shape, "- L =", L, "-->", new_order.shape
        new_order[:, 0] = np.repeat(np.arange(L), L0)
        new_order[:L0, 1:] = order
        if L > 1:
            # reverse order to go back for second index
            new_order[L0:2*L0, 1:] = order[::-1]
        if L > 2:
            # repeat (ascending, descending) up to length L
            rep = L // 2 - 1
            new_order[2*L0:(rep+1)*2*L0, 1:] = np.tile(new_order[:2*L0, 1:], [rep, 1])
            if L % 2 == 1:
                new_order[-L0:, 1:] = order
        order = new_order
    return order
