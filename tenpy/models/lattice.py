"""Classes to define the lattice structure of a model.

The base class :class:`lattice` defines the general structure of a lattice,
you can subclass this to define you own lattice.
Further, we have the predefined lattices, namely :class:`Chain`, :class:`Square`, and
:class:`Honeycomb`.

.. todo ::
    documentation, how to generate new lattices, examples, ...
    implement some __repr__ and/or __str__...
    equality tests?

.. todo ::
    Above, make table with pictures of them (-> use Lattice.plot_ordering)
"""
# Copyright 2018 TeNPy Developers

import numpy as np
import itertools

from ..networks.site import Site
from ..tools.misc import to_iterable, inverse_permutation
from ..networks.mps import MPS  # only to check boundary conditions

__all__ = ['Lattice', 'SimpleLattice', 'Chain', 'Square', 'Honeycomb', 'get_order']

# (update module doc string if you add further lattices)


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
    :meth:`mps2lat_values` perform the necessary reshaping and re-ordering from arrays indexed in
    MPS form to arrays indexed in lattice form.

    Parameters
    ----------
    Ls : list of int
        the length in each direction
    unit_cell : list of :class:`~tenpy.networks.Site`
        the lattice sites making up a unit cell of the lattice.
    order : str | (priority, snake_winding)
        A string or tuple specifying the order, given to :meth:`ordering`.
    bc_MPS : {'finite' | 'segment' | 'infinite'}
        boundary conditions for an MPS/MPO living on the ordered lattice. Default 'finite'.
        If the system is ``'infinite'``, the infinite direction is always along the first basis
        vector (justifying the definition of `N_rings` and `N_sites_per_ring`).
    basis : iterable of 1D arrays
        for each direction one translation vectors shifting the unit cell.
        Defaults to the standard ONB ``np.eye(dim)``.
    positions : iterable of 1D arrays
        for each site of the unit cell the position within the unit cell.
        Defaults to ``np.zeros((len(unit_cell), dim))``.

    Attributes
    ----------
    dim
    N_cells : int
        the number of unit cells in the lattice, ``np.prod(self.Ls)``.
    N_sites : int
        the number of sites in the lattice, ``np.prod(self.shape)``.
    N_sites_per_ring : int
        Defined as ``N_sites / Ls[0]``, for an infinite system the number of cites per "ring".
    N_rings : int
        Alias for ``Ls[0]``, for an infinite system the number of "rings" in the unit cell.
    Ls : tuple of int
        the length in each direction.
    shape : tuple of int
        the 'shape' of the lattice, same as ``Ls + (len(unit_cell), )``
    chinfo : :class:`~tenpy.linalg.charges.ChargeInfo`
        The nature of the charge (which is the same for all sites).
    unit_cell : list of :class:`~tenpy.networks.Site`
        the lattice sites making up a unit cell of the lattice.
    order : ndarray (N_sites, dim+1)
        Defines an ordering of the lattice sites, thus mapping the lattice to a 1D chain.
        This order defines how an MPS/MPO winds through the lattice.
    bc_MPS : {'finite' | 'segment' | 'infinite'}
        boundary conditions for an MPS/MPO living on the ordered lattice.
    basis: ndarray (dim, dim)
        translation vectors shifting the unit cell. The row `i` gives the vector shifting in
        direction `i`.
    unit_cell_positions : ndarray, shape (len(unit_cell), dim)
        for each site in the unit cell a vector giving its position within the unit cell.
    nearest_neighbors : ``None`` | list of ``(u1, u2, dx)``
        May be unspecified (``None``), otherwise it gives a list of parameters `u1`, `u2`, `dx`
        as needed for the :meth:`~tenpy.models.model.CouplingModel` to generate nearest-neighbor
        couplings.
        Note that we include each coupling only in one direction; to get both directions, use
        ``nearest_neighbors + [(u1, u2, -dx) for (u1, u2, dx) in nearest_neighbors]``.
    next_nearest_neighbors : ``None`` | list of ``(u1, u2, dx)``
        Same as :attr:`nearest_neighbors`, but for the next-nearest neigbhors.
    _strides : ndarray (dim, )
        necessary for :meth:`mps2lat_idx`
    _perm : ndarray (N, )
        permutation needed to make `order` lexsorted.
    _mps2lat_vals_idx : ndarray `shape`
        index array for reshape/reordering in :meth:`mps2lat_vals`
    _mps_fix_u : tuple of ndarray (N_cells, ) np.intp
        for each site of the unit cell an index array selecting the mps indices of that site.
    _mps2lat_vals_idx_fix_u : tuple of ndarray of shape `Ls`
        similar as `_mps2lat_vals_idx`, but for a fixed `u` picking a site from the unit cell.
    """

    def __init__(self, Ls, unit_cell, order='default', bc_MPS='finite', basis=None,
                 positions=None):
        self.Ls = tuple([int(L) for L in Ls])
        self.unit_cell = list(unit_cell)
        self.N_cells = int(np.prod(self.Ls))
        self.shape = self.Ls + (len(unit_cell), )
        self.chinfo = self.unit_cell[0].leg.chinfo
        self.N_sites = int(np.prod(self.shape))
        self.N_sites_per_ring = int(self.N_sites // self.Ls[0])
        self.N_rings = self.Ls[0]
        if positions is None:
            positions = np.zeros((len(self.unit_cell), self.dim))
        if basis is None:
            basis = np.eye(self.dim)
        self.unit_cell_positions = np.asarray(positions)
        self.basis = np.asarray(basis)
        self.bc_MPS = bc_MPS
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
            strides.append(strides[-1] * L)
        self._strides = np.array(strides, np.intp)
        for attr in ('nearest_neighbors', 'next_nearest_neighbors'):
            if not hasattr(self, attr):
                setattr(self, attr, None)
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
        assert np.all(
            np.sum(self.order * self._strides,
                   axis=1)[self._perm] == np.arange(self.N_sites))  # rows of `order` unique?
        if self.bc_MPS not in MPS._valid_bc:
            raise ValueError("invalid MPS boundary conditions")

    @property
    def dim(self):
        """the dimension of the lattice."""
        return len(self.Ls)

    def ordering(self, order):
        """Provide possible orderings of the `N` lattice sites.

        This function can be overwritten by derived lattices to define additional orderings.
        The following orders are defined in this method:

        ================== =========================== =============================
        `order`            equivalent `priority`       equivalent ``snake_winding``
        ================== =========================== =============================
        ``'Cstyle'``       (0, 1, ..., dim-1, dim)     (False, ..., False, False)
        ``'default'``
        ``'snake'``        (0, 1, ..., dim-1, dim)     (True, ..., True, True)
        ``'snakeCstyle'``
        ``'Fstyle'``       (dim-1, ..., 1, 0, dim)     (False, ..., False, False)
        ``'snakeFstyle'``  (dim-1, ..., 1, 0, dim)     (False, ..., False, False)
        ================== =========================== =============================

        Parameters
        ----------
        order : str | (priority, snake_winding)
            Specifies the desired ordering using one of the strings of the above tables.
            Alternatively, an ordering is specified by the
            `priority` (one value for each direction, winding along the highest value first)
            and the `snake_winding` (True/False for each direction).
            Further explanations of these tuples in :func:`get_order`.

        Returns
        -------
        order : array, shape (N, D+1), dtype np.intp
            the order to be used for :attr:`order`.

        See also
        --------
        :func:`get_order` : generates the ordering from the equivalent `priority` and `snake_winding`.
        :meth:`plot_ordering` : visualizes the ordering
        """
        if isinstance(order, str):
            if order in ["default", "Cstyle"]:
                priority = None
                snake_winding = (False,) * (self.dim+1)
            elif order == "Fstyle":
                priority = range(self.dim, -1, -1)
                snake_winding = (False,) * (self.dim+1)
            elif order in ["snake", "snakeCstyle"]:
                priority = None
                snake_winding = (True,) * (self.dim+1)
            elif order == "snakeFstyle":
                priority = range(self.dim, -1, -1)
                snake_winding = (True,) * (self.dim+1)
            else:
                # in a derived lattice use ``return super().ordering(order)`` as last option
                # such that the derived lattice also has the orderings defined in this function.
                raise ValueError("unknown ordering " + repr(order))
        else:
            priority, snake_winding = order
        return get_order(self.shape, snake_winding, priority)

    def position(self, lat_idx):
        """return 'space' position of one or multiple sites.

        Parameters
        ----------
        lat_idx : ndarray, ``(... , dim+1)``
            lattice indices

        Returns
        -------
        pos : ndarray, ``(..., dim)``
            The position of the lattice sites specified by `lat_idx` in real-space.
        """
        idx = self._asvalid_latidx(lat_idx)
        res = np.take(self.unit_cell_positions, idx[..., -1], axis=0)
        for i in range(self.dim):
            res += idx[..., i, np.newaxis] * self.basis[i]
        return res

    def site(self, i):
        """return :class:`~tenpy.networks.Site` instance corresponding to an MPS index `i`"""
        return self.unit_cell[self.order[i, -1]]

    def mps_sites(self):
        """Return a list [self.site(i) for i in range(self.N_sites)].

        This should be used for `sites` of 1D tensor networks (MPS, MPO,...)."""
        return [self.unit_cell[u] for u in self.order[:, -1]]

    def mps2lat_idx(self, i):
        """translate MPS index `i` to lattice indices ``(x_0, ..., x_{D_1}, u)``"""
        return tuple(self.order[i])

    def lat2mps_idx(self, lat_idx):
        """translate lattice indices ``(x_0, ..., x_{D-1}, u)`` to MPS index `i`."""
        i = np.sum(self._asvalid_latidx(lat_idx) * self._strides, axis=-1)
        return np.take(self._perm, i)

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
            The row `j` contains the lattice index (without `u`) corresponding to ``mps_idx[j]``.
        """
        mps_idx = self.mps_idx_fix_u(u)
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

        Examples
        --------
        Say you measure expection values of an onsite term for an MPS, which gives you an 1D array
        `A`, where `A[i]` is the expectation value of the site given by ``self.mps2lat_idx(i)``.
        Then this function gives you the expectation values ordered by the lattice:

        >>> print(lat.shape, A.shape)
        (10, 3, 2) (60,)
        >>> A_res = lat.mps2lat_values(A)
        >>> A_res.shape
        (10, 3, 2)
        >>> A_res[lat.mps2lat_idx(5)] == A[5]
        True

        If you have a correlation function ``C[i, j]``, it gets just slightly more complicated:

        >>> print(lat.shape, C.shape)
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

        .. todo ::
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

    def number_nearest_neighbors(self, u=0):
        """Count the number of nearest neighbors for a site in the bulk.

        Requires :attr:`nearest_neighbors` to be set.

        Parameters
        ----------
        u : int
            Specifies the site in the unit cell.

        Returns
        -------
        number_NN : int
            Number of nearest neighbors of the `u`-th site in the unit cell in the bulk of the
            lattice, not that it might be different at the edges of the lattice for open boundary
            conditions.
        """
        if self.nearest_neighbors is None:
            raise ValueError("self.nearest_neighbors were not specified")
        count = 0
        for u1, u2, dx in self.nearest_neighbors:
            if u1 == u:
                count += 1
            if u2 == u:
                count += 1
        return count

    def number_next_nearest_neighbors(self, u=0):
        """Count the number of next nearest neighbors for a site in the bulk.

        Requires :attr:`next_nearest_neighbors` to be set.

        Parameters
        ----------
        u : int
            Specifies the site in the unit cell.

        Returns
        -------
        number_NNN : int
            Number of next nearest neighbors of the `u`-th site in the unit cell in the bulk of the
            lattice, not that it might be different at the edges of the lattice for open boundary
            conditions.
        """
        if self.next_nearest_neighbors is None:
            raise ValueError("self.next_nearest_neighbors were not specified")
        count = 0
        for u1, u2, dx in self.next_nearest_neighbors:
            if u1 == u:
                count += 1
            if u2 == u:
                count += 1
        return count

    def plot_sites(self, ax, markers=['o', '^', 's', 'p', 'h', 'D'], **kwargs):
        """Plot the sites of the lattice with markers.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes
            The axes on which we should plot.
        markers : list
            List of values for the keywork `marker` of ``ax.plot()`` to distinguish the different
            sites in the unit cell, a site `u` in the unit cell is plotted with a marker
            ``markers[u % len(markers)]``.
        **kwargs :
            Further keyword arguments given to ``ax.plot()``.
        """
        kwargs.setdefault("linestyle", 'None')
        use_marker = ('marker' not in kwargs)
        for u in range(len(self.unit_cell)):
            pos = self.position(self.order[self.mps_idx_fix_u(u), :])
            if pos.shape[1] == 1:
                pos = pos * np.array([[1., 0]])  # use broadcasting to add a column with zeros
            if pos.shape[1] != 2:
                raise ValueError("can only plot in 2 dimensions.")
            if use_marker:
                kwargs['marker'] = markers[u % len(markers)]
            ax.plot(pos[:, 0], pos[:, 1], **kwargs)

    def plot_order(self, ax, order=None, textkwargs={}, **kwargs):
        """Plot a line connecting sites in the specified "order" and text labels enumerating them.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes
            The axes on which we should plot.
        order : None | 2D array (self.N_sites, self.dim+1)
            The order as returned by :meth:`ordering`; by default (``None``) use :attr:`order`.
        textkwargs: ``None`` | dict
            If not ``None``, we add text labels enumerating the sites in the plot. The dictionary
            can contain keyword arguments for ``ax.text()``.
        **kwargs :
            Further keyword arguments given to ``ax.plot()``.
        """
        if order is None:
            order = self.order
        pos = self.position(order)
        kwargs.setdefault('color', 'r')
        if pos.shape[1] == 1:
            pos = pos * np.array([[1., 0]])  # use broadcasting to add a column with zeros
        if pos.shape[1] != 2:
            raise ValueError("can only plot in 2 dimensions.")
        ax.plot(pos[:, 0], pos[:, 1], **kwargs)
        if textkwargs is not None:
            for i, p in enumerate(pos):
                ax.text(p[0], p[1], str(i), **textkwargs)

    def plot_coupling(self, ax, coupling=None, **kwargs):
        """Plot lines connecting nearest neighbors of the lattice.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes
            The axes on which we should plot.
        coupling : list of (u1, u2, dx)
            Specifies the connections to be plotted; iteating over lattice indices `(i0, i1, ...)`,
            we plot a connection from the site ``(i0, i1, ..., u1)`` to the site
            ``(i0+dx[0], i1+dx[1], ..., u1)``.
            By default (``None``), use :attr:``nearest_neighbors``.
        **kwargs :
            Further keyword arguments given to ``ax.plot()``.
        """
        if coupling is None:
            coupling = self.nearest_neighbors
        kwargs.setdefault('color', 'k')
        for u1, u2, dx in coupling:
            dx = np.r_[np.array(dx), 0]  # append a 0 to dx
            lat_idx_1 = self.order[self._mps_fix_u[u1], :]
            lat_idx_2 = self.order[self._mps_fix_u[u2], :] + dx[np.newaxis, :]
            pos1 = self.position(lat_idx_1)
            pos2 = self.position(lat_idx_2)
            pos = np.stack((pos1, pos2), axis=0)
            # ax.plot connects columns of 2D array by lines
            if pos.shape[2] == 1:
                pos = pos * np.array([[[1., 0]]])  # use broadcasting to add a column with zeros
            if pos.shape[2] != 2:
                raise ValueError("can only plot in 2 dimensions.")
            ax.plot(pos[:, :, 0], pos[:, :, 1], **kwargs)

    def plot_basis(self, ax, **kwargs):
        """Plot arrows indicating the basis vectors of the lattice

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes
            The axes on which we should plot.
        **kwargs :
            Keyword arguments specifying the "arrowprops" of ``ax.annotate``.
        """
        kwargs.setdefault("arrowstyle", "->")
        for i in range(self.dim):
            vec = self.basis[i]
            if vec.shape[0] == 1:
                vec = vec * np.array([1., 0])
            if vec.shape[0] != 2:
                raise ValueError("can only plot in 2 dimensions.")
            ax.annotate("", vec, [0., 0.], arrowprops=kwargs)

    def _asvalid_latidx(self, lat_idx):
        """convert lat_idx to an ndarray with correct last dimension."""
        lat_idx = np.asarray(lat_idx, dtype=np.intp)
        if lat_idx.shape[-1] != len(self.shape):
            raise ValueError("wrong len of last dimension of lat_idx: " + str(lat_idx.shape))
        return lat_idx


class SimpleLattice(Lattice):
    """A lattice with a unit cell consiting of just a single site.

    In many cases, the unit cell consists just of a single site, such that the the last entry of
    `u` of an 'lattice index' can only be ``0``.
    From the point of internal algorithms, we handle this class like a :class:`Lattice` --
    in that way we don't need to distinguish special cases in the algorithms.

    Yet, from the point of a tenpy user, for example if you measure an expectation value
    on each site in a `SimpleLattice`, you expect to get an ndarray of dimensions ``self.Ls``,
    not ``self.shape``. To avoid that problem, `SimpleLattice` overwrites just the meaning of
    ``u=None`` in :meth:`mps2lat_values` to be the same as ``u=0``.

    Parameters
    ----------
    Ls : list of int
        the length in each direction
    site : :class:`~tenpy.networks.Site`
        the lattice site. The `unit_cell` of the :class:`Lattice` is just ``[site]``.
    order : str | (priority, snake_winding)
        A string or tuple specifying the order, given to :meth:`ordering`.
        If a tuple, the priority and snake_winding should only be specified for the lattice
        directions.
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
        if not isinstance(order, str):
            priority, snake_winding = order
            priority = tuple(priority) +  (max(priority) + 1., )
            snake_winding = tuple(snake_winding) + (False, )
            order = priority, snake_winding
        super(SimpleLattice, self).__init__(Ls, [site], order, bc_MPS, basis, position)

    def mps2lat_values(self, A, axes=0, u=None):
        """same as :meth:`Lattice.mps2lat_values`, but ignore ``u``, setting it to ``0``."""
        super(SimpleLattice, self).mps2lat_values(A, axes, 0)


class Chain(SimpleLattice):
    """A simple uniform chain of L equal sites.

    Parameters
    ----------
    L : int
        The lenght of the chain.
    site : :class:`~tenpy.networks.Site`
        Definition of local Hilbert space.
    bc_MPS : {'finite', 'segment', 'infinite'}
        MPS boundary conditions.
    """

    def __init__(self, L, site, bc_MPS='finite'):
        self.nearest_neighbors = [(0, 0, np.array([1,]))]
        self.next_nearest_neighbors = [(0, 0, np.array([2,]))]
        super(Chain, self).__init__([L], site, bc_MPS=bc_MPS)  # and otherwise default values.


class Square(SimpleLattice):
    """A simple uniform square lattice of `Lx` by `Ly` sites."""

    def __init__(self, Lx, Ly, site, order='default', bc_MPS='finite'):
        self.nearest_neighbors = [(0, 0, np.array([1, 0])),
                                  (0, 0, np.array([0, 1]))]
        self.next_nearest_neighbors = [(0, 0, np.array([1, 1])),
                                       (0, 0, np.array([1, -1]))]
        super(Square, self).__init__([Lx, Ly], site, order, bc_MPS)


class Honeycomb(Lattice):
    """A honeycomb lattice."""

    def __init__(self, Lx, Ly, siteA, siteB, order='default', bc_MPS='finite'):
        basis = np.array(([0.5*np.sqrt(3), 0.5], [0., 1]))
        delta = np.array([1/(2.*np.sqrt(3.)), 0.5])
        pos = (-delta/2., delta/2)
        self.nearest_neighbors = [(0, 1, np.array([0, 0])),
                                  (1, 0, np.array([1, 0])),
                                  (1, 0, np.array([0, 1]))]
        self.next_nearest_neighbors = [(0, 0, np.array([1, 0])),
                                       (1, 1, np.array([1, 0])),
                                       (0, 0, np.array([0, 1])),
                                       (1, 1, np.array([0, 1])),
                                       (0, 0, np.array([1, -1])),
                                       (1, 1, np.array([1, -1]))]
        super(Honeycomb, self).__init__([Lx, Ly], [siteA, siteB], order, bc_MPS, basis, pos)

    def ordering(self, order):
        """Provide possible orderings of the `N` lattice sites.

        The following orders are defined in this method compared to :meth:`Lattice.ordering`:

        ================== =========================== =============================
        `order`            equivalent `priority`       equivalent ``snake_winding``
        ================== =========================== =============================
        ``'default'``      (0, 2, 1)                   (False, False, False)
        ``'snake'``        (0, 2, 1)                   (False, True, False)
        ================== =========================== =============================

        Parameters
        ----------
        order : str | (priority, snake_winding)
            Specifies the desired ordering using one of the strings of the above tables.
            Alternatively, an ordering is specified by the
            `priority` (one value for each direction, winding along the highest value first)
            and the `snake_winding` (True/False for each direction).
            Further explanations of these tuples in :func:`get_order`.

        Returns
        -------
        order : array, shape (N, D+1), dtype np.intp
            the order to be used for :attr:`order`.

        See also
        --------
        :func:`get_order` : generates the ordering from the equivalent `priority` and `snake_winding`.
        :meth:`plot_ordering` : visualizes the ordering
        """
        if isinstance(order, str):
            if order == "default":
                priority = (0, 2, 1)
                snake_winding = (False, False, False)
                return get_order(self.shape, snake_winding, priority)
            elif order == "snake":
                priority = (0, 2, 1)
                snake_winding = (False, False, True)
                return get_order(self.shape, snake_winding, priority)
        return super().ordering(order)


def get_order(shape, snake_winding, priority=None):
    """Built the :attr:`Lattice.order` in (Snake-) C-Style for a given lattice shape.

    In this function, the word 'direction' referst to a physical direction of the lattice or the
    index `u` of the unit cell as an "artificial direction".

    Parameters
    ----------
    shape : tuple of int
        The shape of the lattice, i.e., the length in each direction.
    snake_winding : tuple of bool
        For each direction one bool, whether we should wind as a "snake" (True) in that direction
        (i.e., going forth and back) or simply repeat ascending (False)
    priority: ``None`` | tuple of float
        If ``None`` (default), use C-Style ordering.
        Otherwise, this defines the priority along which direction to wind first;
        the direction with the highest priority increases fastest.
        For example, "C-Style" order is enforced by ``priority=(0, 1, 2, ...)``,
        and Fortrans F-style order is enforced by ``priority=(dim, dim-1, ..., 1, 0)``

    Returns
    -------
    order : ndarray (np.prod(shape), len(shape))
        An order of the sites for :attr:`Lattice.order` in the specified `ordering`.
    """
    if priority is not None:
        # reduce this case to C-style order and a few permutations
        perm = np.argsort(priority)
        inv_perm = inverse_permutation(perm)
        transp_shape = np.array(shape)[perm]
        transp_snake = np.array(snake_winding)[perm]
        order = get_order(transp_shape, transp_snake, None)  # in plain C-style
        order = order[:, inv_perm]
        return order
    # simpler case: generate C-style order
    shape = tuple(shape)
    if not any(snake_winding):
        # optimize: can use np.mgrid
        res = np.mgrid[tuple([slice(0, L) for L in shape])]
        return res.reshape((len(shape), np.prod(shape))).T
    # some snake: generate direction by direction, each time adding a new column to `order`
    snake_winding = tuple(snake_winding) + (False, )
    dim = len(shape)
    order = np.empty((1, 0), dtype=np.intp)
    for i in range(dim):
        L = shape[dim-1-i]
        snake = snake_winding[dim-i] # previous direction snake?
        L0, D = order.shape
        # insert a new first column into order
        new_order = np.empty((L * L0, D + 1), dtype=np.intp)
        new_order[:, 0] = np.repeat(np.arange(L), L0)
        if not snake:
            new_order[:, 1:] = np.tile(order, (L, 1))
        else:  # snake
            new_order[:L0, 1:] = order
            L0_2 = 2 * L0
            if L > 1:
                # reverse order to go back for second index
                new_order[L0:L0_2, 1:] = order[::-1, :]
            if L > 2:
                # repeat (ascending, descending) up to length L
                rep = L // 2 - 1
                if rep > 0:
                    new_order[L0_2:(rep + 1) * L0_2, 1:] = np.tile(new_order[:L0_2, 1:], (rep, 1))
                if L % 2 == 1:
                    new_order[-L0:, 1:] = order
        order = new_order
    return order
