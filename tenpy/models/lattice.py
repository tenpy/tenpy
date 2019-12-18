"""Classes to define the lattice structure of a model.

The base class :class:`Lattice` defines the general structure of a lattice,
you can subclass this to define you own lattice.
The :class:`SimpleLattice` is a slight simplification for lattices with a single-site unit cell.
Further, we have some predefined lattices, namely
:class:`Chain`, :class:`Ladder` in 1D and
:class:`Square`, :class:`Triangular`, :class:`Honeycomb`, and :class:`Kagome` in 2D.

See also the :doc:`/intro_model`.
"""
# Copyright 2018-2019 TeNPy Developers, GNU GPLv3

import numpy as np
import itertools
import warnings

from ..networks.site import Site
from ..tools.misc import to_iterable, inverse_permutation
from ..networks.mps import MPS  # only to check boundary conditions

__all__ = [
    'Lattice', 'TrivialLattice', 'IrregularLattice', 'SimpleLattice', 'Chain', 'Ladder', 'Square',
    'Triangular', 'Honeycomb', 'Kagome', 'get_lattice', 'get_order', 'get_order_grouped'
]

# (update module doc string if you add further lattices)

bc_choices = {'open': True, 'periodic': False}
"""dict: maps possible choices of boundary conditions in a lattice to bool/int."""


class Lattice:
    r"""A general, regular lattice.

    The lattice consists of a **unit cell** which is repeated in `dim` different directions.
    A site of the lattice is thus identified by **lattice indices** ``(x_0, ..., x_{dim-1}, u)``,
    where ``0 <= x_l < Ls[l]`` pick the position of the unit cell in the lattice and
    ``0 <= u < len(unit_cell)`` picks the site within the unit cell. The site is located
    in 'space' at ``sum_l x_l*basis[l] + unit_cell_positions[u]`` (see :meth:`position`).
    (Note that the position in space is only used for plotting, not for defining the couplings.)

    In addition to the pure geometry, this class also defines an `order` of all sites.
    This order maps the lattice to a finite 1D chain and defines the geometry of MPSs and MPOs.
    The **MPS index** `i` corresponds thus to the lattice sites given by
    ``(x_0, ..., x_{dim-1}, u) = tuple(self.order[i])``.
    Infinite boundary conditions of the MPS repeat in the first spatial direction of the lattice,
    i.e., if the site at (x_0, x_1, ..., x_{dim-1},u)`` has MPS index `i`, the site at
    at ``(x_0 + a*Ls[0], x_1 ..., x_{dim-1}, u)`` corresponds to MPS index ``i + N_sites``.
    Use :meth:`mps2lat_idx` and :meth:`lat2mps_idx` for conversion of indices.
    The function :meth:`mps2lat_values` performs the necessary reshaping and re-ordering from
    arrays indexed in MPS form to arrays indexed in lattice form.

    Parameters
    ----------
    Ls : list of int
        the length in each direction
    unit_cell : list of :class:`~tenpy.networks.site.Site`
        The sites making up a unit cell of the lattice.
        If you want to specify it only after initialization, use ``None`` entries in the list.
    order : str | ``('standard', snake_winding, priority)`` | ``('grouped', groups)``
        A string or tuple specifying the order, given to :meth:`ordering`.
    bc : (iterable of) {'open' | 'periodic' | int}
        Boundary conditions in each direction of the lattice.
        A single string holds for all directions.
        An integer `shift` means that we have periodic boundary conditions along this direction,
        but shift/tilt by ``-shift*lattice.basis[0]`` (~cylinder axis for ``bc_MPS='infinite'``)
        when going around the boundary along this direction.
    bc_MPS : 'finite' | 'segment' | 'infinite'
        Boundary conditions for an MPS/MPO living on the ordered lattice.
        If the system is ``'infinite'``, the infinite direction is always along the first basis
        vector (justifying the definition of `N_rings` and `N_sites_per_ring`).
    basis : iterable of 1D arrays
        For each direction one translation vectors shifting the unit cell.
        Defaults to the standard ONB ``np.eye(dim)``.
    positions : iterable of 1D arrays
        For each site of the unit cell the position within the unit cell.
        Defaults to ``np.zeros((len(unit_cell), dim))``.
    nearest_neighbors : ``None`` | list of ``(u1, u2, dx)``
        Deprecated. Specify as ``pairs['nearest_neighbors']`` instead.
    next_nearest_neighbors : ``None`` | list of ``(u1, u2, dx)``
        Deprecated. Specify as ``pairs['next_nearest_neighbors']`` instead.
    next_next_nearest_neighbors : ``None`` | list of ``(u1, u2, dx)``
        Deprecated. Specify as ``pairs['next_next_nearest_neighbors']`` instead.
    pairs : dict
        Of the form ``{'nearest_neighbors': [(u1, u2, dx), ...], ...}``.
        Typical keys are ``'nearest_neighbors', 'next_nearest_neighbors'``.
        For each of them, it specifies a list of tuples ``(u1, u2, dx)`` which can
        be used as parameters for :meth:`~tenpy.models.model.CouplingModel.add_coupling`
        to generate couplings over each pair of e.g. ``'nearest_neighbors'``.
        Note that this adds couplings for each pair *only in one direction*!

    Attributes
    ----------
    dim : int
    order : ndarray (N_sites, dim+1)
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
    unit_cell : list of :class:`~tenpy.networks.site.Site`
        the lattice sites making up a unit cell of the lattice.
    bc : bool ndarray
        Boundary conditions of the couplings in each direction of the lattice,
        translated into a bool array with the global `bc_choices`.
    bc_shift : None | ndarray(int)
        The shift in x-direction when going around periodic boundaries in other directions.
    bc_MPS : 'finite' | 'segment' | 'infinite'
        Boundary conditions for an MPS/MPO living on the ordered lattice.
        If the system is ``'infinite'``, the infinite direction is always along the first basis
        vector (justifying the definition of `N_rings` and `N_sites_per_ring`).
    basis : ndarray (dim, Dim)
        translation vectors shifting the unit cell. The row `i` gives the vector shifting in
        direction `i`.
    unit_cell_positions : ndarray, shape (len(unit_cell), Dim)
        for each site in the unit cell a vector giving its position within the unit cell.
    pairs : dict
        See above.
    _order : ndarray (N_sites, dim+1)
        The place where :attr:`order` is stored.
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
    def __init__(self,
                 Ls,
                 unit_cell,
                 order='default',
                 bc='open',
                 bc_MPS='finite',
                 basis=None,
                 positions=None,
                 nearest_neighbors=None,
                 next_nearest_neighbors=None,
                 next_next_nearest_neighbors=None,
                 pairs=None):
        self.Ls = tuple([int(L) for L in Ls])
        self.unit_cell = list(unit_cell)
        self.N_cells = int(np.prod(self.Ls))
        self.shape = self.Ls + (len(unit_cell), )
        self.N_sites = int(np.prod(self.shape))
        self.N_rings = self.Ls[0]
        self.N_sites_per_ring = int(self.N_sites // self.N_rings)
        if positions is None:
            positions = np.zeros((len(self.unit_cell), self.dim))
        if basis is None:
            basis = np.eye(self.dim)
        self.unit_cell_positions = np.array(positions)
        self.basis = np.array(basis)
        self._set_bc(bc)
        self.bc_MPS = bc_MPS
        # calculate order for MPS
        self.order = self.ordering(order)
        # uses attribute setter to calculte _mps2lat_vals_idx_fix_u etc and lat2mps
        # calculate _strides
        strides = [1]
        for L in self.Ls:
            strides.append(strides[-1] * L)
        self._strides = np.array(strides, np.intp)
        self.pairs = pairs if pairs is not None else {}
        for name, NN in [('nearest_neighbors', nearest_neighbors),
                         ('next_nearest_neighbors', next_nearest_neighbors),
                         ('next_next_nearest_neighbors', next_next_nearest_neighbors)]:
            if NN is None:
                continue  # no value set
            msg = "Lattice.__init__() got argument `{0!s}`.\nSet as `neighbors['{0!s}'] instead!"
            msg = msg.format(name)
            warnings.warn(msg, FutureWarning)
            if name in self.pairs:
                raise ValueError("{0!s} sepcified twice!".format(name))
            self.pairs[name] = NN
        self.test_sanity()  # check consistency

    def test_sanity(self):
        """Sanity check.

        Raises ValueErrors, if something is wrong.
        """
        assert self.dim == len(self.Ls)
        assert self.shape == self.Ls + (len(self.unit_cell), )
        assert self.N_cells == np.prod(self.Ls)
        assert self.N_sites == np.prod(self.shape)
        if self.bc.shape != (self.dim, ):
            raise ValueError("Wrong len of bc")
        assert self.bc.dtype == np.bool
        chinfo = None
        for site in self.unit_cell:
            if site is None:
                continue
            if chinfo is None:
                chinfo = site.leg.chinfo
            if not isinstance(site, Site):
                raise ValueError("element of Unit cell is not Site.")
            if site.leg.chinfo != chinfo:
                raise ValueError("All sites must have the same ChargeInfo!")
        if self.basis.shape[0] != self.dim:
            raise ValueError("Need one basis vector for each direction!")
        if self.unit_cell_positions.shape[0] != len(self.unit_cell):
            raise ValueError("Need one position for each site in the unit cell.")
        if self.basis.shape[1] != self.unit_cell_positions.shape[1]:
            raise ValueError("Different space dimensions of `basis` and `unit_cell_positions`")
        # if one of the following assert fails, the `ordering` function returned an invalid array
        assert np.all(self._order >= 0) and np.all(self._order <= self.shape)  # entries of `order`
        assert np.all(
            np.sum(self._order * self._strides,
                   axis=1)[self._perm] == np.arange(self.N_sites))  # rows of `order` unique?
        if self.bc_MPS not in MPS._valid_bc:
            raise ValueError("invalid MPS boundary conditions")
        if self.bc[0] and self.bc_MPS == 'infinite':
            raise ValueError("Need periodic boundary conditions along the x-direction "
                             "for 'infinite' `bc_MPS`")

    @property
    def dim(self):
        """The dimension of the lattice."""
        return len(self.Ls)

    @property
    def order(self):
        """Defines an ordering of the lattice sites, thus mapping the lattice to a 1D chain.

        This order defines how an MPS/MPO winds through the lattice.
        """
        return self._order

    @order.setter
    def order(self, order_):
        # update the value itself
        self._order = order_
        # and the other stuff which is cached
        self._perm = np.lexsort(order_.T)
        # use advanced numpy indexing...
        self._mps2lat_vals_idx = np.empty(self.shape, np.intp)
        self._mps2lat_vals_idx[tuple(order_.T)] = np.arange(self.N_sites)
        # versions for fixed u
        self._mps_fix_u = []
        self._mps2lat_vals_idx_fix_u = []
        for u in range(len(self.unit_cell)):
            mps_fix_u = np.nonzero(order_[:, -1] == u)[0]
            self._mps_fix_u.append(mps_fix_u)
            mps2lat_vals_idx = np.empty(self.Ls, np.intp)
            mps2lat_vals_idx[tuple(order_[mps_fix_u, :-1].T)] = np.arange(self.N_cells)
            self._mps2lat_vals_idx_fix_u.append(mps2lat_vals_idx)
        self._mps_fix_u = tuple(self._mps_fix_u)

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
        order : str | ``('standard', snake_winding, priority)`` | ``('grouped', groups)``
            Specifies the desired ordering using one of the strings of the above tables.
            Alternatively, an ordering is specified by a tuple with first entry specifying a
            function, ``'standard'`` for :func:`get_order` and ``'grouped'`` for
            :func:`get_order_grouped`, and other arguments in the tuple as specified in the
            documentation of these functions.

        Returns
        -------
        order : array, shape (N, D+1), dtype np.intp
            the order to be used for :attr:`order`.

        See also
        --------
        get_order : generates the `order` from equivalent `priority` and `snake_winding`.
        get_order_grouped : variant of `get_order`.
        plot_order : visualizes the resulting `order`.
        """
        if isinstance(order, str):
            if order in ["default", "Cstyle"]:
                priority = None
                snake_winding = (False, ) * (self.dim + 1)
            elif order == "Fstyle":
                priority = range(self.dim, -1, -1)
                snake_winding = (False, ) * (self.dim + 1)
            elif order in ["snake", "snakeCstyle"]:
                priority = None
                snake_winding = (True, ) * (self.dim + 1)
            elif order == "snakeFstyle":
                priority = range(self.dim, -1, -1)
                snake_winding = (True, ) * (self.dim + 1)
            else:
                # in a derived lattice use ``return super().ordering(order)`` as last option
                # such that the derived lattice also has the orderings defined in this function.
                raise ValueError("unknown ordering " + repr(order))
        else:
            descr = order[0]
            if descr == 'standard':
                snake_winding, priority = order[1], order[2]
            elif descr == 'grouped':
                return get_order_grouped(self.shape, order[1])
            else:
                raise ValueError("unknown ordering " + repr(order))
        return get_order(self.shape, snake_winding, priority)

    def position(self, lat_idx):
        """return 'space' position of one or multiple sites.

        Parameters
        ----------
        lat_idx : ndarray, ``(... , dim+1)``
            Lattice indices.

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
        """return :class:`~tenpy.networks.site.Site` instance corresponding to an MPS index `i`"""
        return self.unit_cell[self.order[i, -1]]

    def mps_sites(self):
        """Return a list of sites for all MPS indices.

        Equivalent to ``[self.site(i) for i in range(self.N_sites)]``.

        This should be used for `sites` of 1D tensor networks (MPS, MPO,...).
        """
        return [self.unit_cell[u] for u in self.order[:, -1]]

    def mps2lat_idx(self, i):
        """Translate MPS index `i` to lattice indices ``(x_0, ..., x_{dim-1}, u)``.

        Parameters
        ----------
        i : int | array_like of int
            MPS index/indices.

        Returns
        -------
        lat_idx : array
            First dimensions like `i`, last dimension has len `dim`+1 and contains the lattice
            indices ``(x_0, ..., x_{dim-1}, u)`` corresponding to `i`.
            For `i` accross the MPS unit cell and "infinite" `bc_MPS`, we shift `x_0` accordingly.
        """
        if self.bc_MPS == 'infinite':
            # allow `i` outsit of MPS unit cell for bc_MPS infinite
            i0 = i
            i = np.mod(i, self.N_sites)
            if np.any(i0 != i):
                lat = self.order[i].copy()
                lat[..., 0] += (i0 - i) // self.N_sites * self.N_rings
                return lat
        return self.order[i].copy()

    def lat2mps_idx(self, lat_idx):
        """Translate lattice indices ``(x_0, ..., x_{D-1}, u)`` to MPS index `i`.

        Parameters
        ----------
        lat_idx : array_like [..., dim+1]
            The last dimension corresponds to lattice indices ``(x_0, ..., x_{D-1}, u)``.
            All lattice indices should be positive and smaller than the corresponding entry in
            ``self.shape``. Exception: for "infinite" `bc_MPS`, an `x_0` outside indicates shifts
            accross the boundary.

        Returns
        -------
        i : array_like
            MPS index/indices corresponding to `lat_idx`.
            Has the same shape as `lat_idx` without the last dimension.
        """
        idx = self._asvalid_latidx(lat_idx)
        if self.bc_MPS == 'infinite':
            i_shift = idx[..., 0] - np.mod(idx[..., 0], self.N_rings)
            idx[..., 0] -= i_shift
        i = np.sum(np.mod(idx, self.shape) * self._strides, axis=-1)  # before permutation
        i = np.take(self._perm, i)  # after permutation
        if self.bc_MPS == 'infinite':
            i += i_shift * (self.N_sites // self.N_rings)
        return i

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
            MPS indices for which ``self.site(i) is self.unit_cell[u]``. Ordered ascending.
        """
        if u is not None:
            return self._mps_fix_u[u]
        return self._perm

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
        """Reshape/reorder `A` to replace an MPS index by lattice indices.

        Parameters
        ----------
        A : ndarray
            Some values. Must have ``A.shape[axes] = self.N_sites`` if `u` is ``None``, or
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
            Reshaped and reordered verions of A. Such that an MPS index `j` is replaced by
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
        the argument `u` of this function.

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
        # choose the appropriate index arrays
        if u is None:
            idx = self._mps2lat_vals_idx
        else:
            idx = self._mps2lat_vals_idx_fix_u[u]
        return np.take(A, idx, axis=axes[0])

    def count_neighbors(self, u=0, key='nearest_neighbors'):
        """Count e.g. the number of nearest neighbors for a site in the bulk.

        Parameters
        ----------
        u : int
            Specifies the site in the unit cell, for which we should count the number
            of neighbors (or whatever `key` specifies).
        key : str
            Key of :attr:`pairs` to select what to count.

        Returns
        -------
        number : int
            Number of nearest neighbors (or whatever `key` specified) for the `u`-th site in the
            unit cell, somewhere in the bulk of the lattice. Note that it might not be the correct
            value at the edges of a lattice with open boundary conditions.
        """
        pairs = self.pairs[key]
        count = 0
        for u1, u2, dx in pairs:
            if u1 == u:
                count += 1
            if u2 == u:
                count += 1
        return count

    def number_nearest_neighbors(self, u=0):
        """Deprecated.

        Use :meth:`count_neighbors` instead.
        """
        msg = "Use ``count_neighbors(u, 'nearest_neighbors')`` instead."
        warnings.warn(msg, FutureWarning)
        return self.count_neighbors(u, 'nearest_neighbors')

    def number_next_nearest_neighbors(self, u=0):
        """Deprecated.

        Use :meth:`count_neighbors` instead.
        """
        msg = "Use ``count_neighbors(u, 'next_nearest_neighbors')`` instead."
        warnings.warn(msg, FutureWarning)
        return self.count_neighbors(u, 'next_nearest_neighbors')

    def possible_couplings(self, u1, u2, dx):
        """Find possible MPS indices for two-site couplings.

        For periodic boundary conditions (``bc[a] == False``)
        the index ``x_a`` is taken modulo ``Ls[a]`` and runs through ``range(Ls[a])``.
        For open boundary conditions, ``x_a`` is limited to ``0 <= x_a < Ls[a]`` and
        ``0 <= x_a+dx[a] < lat.Ls[a]``.

        Parameters
        ----------
        u1, u2 : int
            Indices within the unit cell; the `u1` and `u2` of
            :meth:`~tenpy.models.model.CouplingModel.add_coupling`
        dx : array
            Length :attr:`dim`. The translation in terms of basis vectors for the coupling.

        Returns
        -------
        mps1, mps2 : array
            For each possible two-site coupling the MPS indices for the `u1` and `u2`.
        lat_indices : 2D int array
            Rows of `lat_indices` correspond to rows of `mps_ijkl` and contain the lattice indices
            of the "lower left corner" of the box containing the coupling.
        coupling_shape : tuple of int
            Len :attr:`dim`. The correct shape for an array specifying the coupling strength.
            `lat_indices` has only rows within this shape.
        """
        coupling_shape, shift_lat_indices = self.coupling_shape(dx)
        if any([s == 0 for s in coupling_shape]):
            return [], [], np.zeros([0, self.dim]), coupling_shape
        Ls = np.array(self.Ls)
        N_sites = self.N_sites
        mps_i, lat_i = self.mps_lat_idx_fix_u(u1)
        lat_j_shifted = lat_i + dx
        lat_j = np.mod(lat_j_shifted, Ls)  # assuming PBC
        if self.bc_shift is not None:
            shift = np.sum(((lat_j_shifted - lat_j) // Ls)[:, 1:] * self.bc_shift, axis=1)
            lat_j_shifted[:, 0] -= shift
            lat_j[:, 0] = np.mod(lat_j_shifted[:, 0], Ls[0])
        keep = np.all(
            np.logical_or(
                lat_j_shifted == lat_j,  # not accross the boundary
                np.logical_not(self.bc)),  # direction has PBC
            axis=1)
        mps_i = mps_i[keep]
        lat_indices = lat_i[keep] + shift_lat_indices[np.newaxis, :]
        lat_indices = np.mod(lat_indices, coupling_shape)
        lat_j = lat_j[keep]
        lat_j_shifted = lat_j_shifted[keep]
        mps_j = self.lat2mps_idx(np.concatenate([lat_j, [[u2]] * len(lat_j)], axis=1))
        if self.bc_MPS == 'infinite':
            # shift j by whole MPS unit cells for couplings along the infinite direction
            mps_j_shift = (lat_j_shifted[:, 0] - lat_j[:, 0]) * (N_sites // Ls[0])
            mps_j += mps_j_shift
            # finally, ensure 0 <= min(i, j) < N_sites.
            mps_ij_shift = np.where(mps_j_shift < 0, -mps_j_shift, 0)
            mps_i += mps_ij_shift
            mps_j += mps_ij_shift
        return mps_i, mps_j, lat_indices, coupling_shape

    def possible_multi_couplings(self, u0, other_us, dx):
        """Generalization of :meth:`possible_couplings` to couplings with more than 2 sites.

        Given the arguments of :meth:`~tenpy.models.model.MultiCouplingModel.add_coupling`
        determine the necessary shape of `strength`.

        Parameters
        ----------
        u0 : int
            Argument `u0` of :meth:`~tenpy.models.model.MultiCouplingModel.add_multi_coupling`.
        other_us : list of int
            The `u` of the `other_ops` in
            :meth:`~tenpy.models.model.MultiCouplingModel.add_multi_coupling`.
        dx : array, shape (len(other_us), lat.dim+1)
            The `dx` specifying relative operator positions of the `other_ops` in
            :meth:`~tenpy.models.model.MultiCouplingModel.add_multi_coupling`.

        Returns
        -------
        mps_ijkl : 2D int array
            Each row contains MPS indices `i,j,k,l,...`` for each of the operators positions.
            The positions are defined by `dx` (j,k,l,... relative to `i`) and boundary coundary
            conditions of `self` (how much the `box` for given `dx` can be shifted around without
            hitting a boundary - these are the different rows).
        lat_indices : 2D int array
            Rows of `lat_indices` correspond to rows of `mps_ijkl` and contain the lattice indices
            of the "lower left corner" of the box containing the coupling.
        coupling_shape : tuple of int
            Len :attr:`dim`. The correct shape for an array specifying the coupling strength.
            `lat_indices` has only rows within this shape.
        """
        coupling_shape, shift_lat_indices = self.multi_coupling_shape(dx)
        if any([s == 0 for s in coupling_shape]):
            return [], [], [], coupling_shape
        Ls = np.array(self.Ls)
        N_sites = self.N_sites
        mps_i, lat_i = self.mps_lat_idx_fix_u(u0)
        lat_jkl_shifted = lat_i[:, np.newaxis, :] + dx[np.newaxis, :, :]
        # lat_jkl* has 3 axes "initial site", "other_op", "spatial directions"
        lat_jkl = np.mod(lat_jkl_shifted, Ls)  # assuming PBC
        if self.bc_shift is not None:
            shift = np.sum(((lat_jkl_shifted - lat_jkl) // Ls)[:, :, 1:] * self.bc_shift, axis=2)
            lat_jkl_shifted[:, :, 0] -= shift
            lat_jkl[:, :, 0] = np.mod(lat_jkl_shifted[:, :, 0], Ls[0])
        keep = np.all(
            np.logical_or(
                lat_jkl_shifted == lat_jkl,  # not accross the boundary
                np.logical_not(self.bc)),  # direction has PBC
            axis=(1, 2))
        mps_i = mps_i[keep]
        lat_indices = lat_i[keep, :] + shift_lat_indices[np.newaxis, :]
        lat_indices = np.mod(lat_indices, coupling_shape)
        lat_jkl = lat_jkl[keep, :, :]
        lat_jkl_shifted = lat_jkl_shifted[keep, :, :]
        latu_jkl = np.concatenate((lat_jkl, np.array([other_us] * len(lat_jkl))[:, :, np.newaxis]),
                                  axis=2)
        mps_jkl = self.lat2mps_idx(latu_jkl)
        if self.bc_MPS == 'infinite':
            # shift by whole MPS unit cells for couplings along the infinite direction
            mps_jkl += (lat_jkl_shifted[:, :, 0] - lat_jkl[:, :, 0]) * (N_sites // Ls[0])
        mps_ijkl = np.concatenate((mps_i[:, np.newaxis], mps_jkl), axis=1)
        return mps_ijkl, lat_indices, coupling_shape

    def coupling_shape(self, dx):
        """Calculate correct shape of the `strengths` for a coupling.

        Parameters
        ----------
        dx : tuple of int
            Translation vector in the lattice for a coupling of two operators.

        Returns
        -------
        coupling_shape : tuple of int
            Len :attr:`dim`. The correct shape for an array specifying the coupling strength.
            `lat_indices` has only rows within this shape.
        shift_lat_indices : array
            Translation vector from lower left corner of box spanned by `dx` to the origin.
        """
        shape = [La - abs(dxa) * int(bca) for La, dxa, bca in zip(self.Ls, dx, self.bc)]
        shift_strength = [min(0, dxa) for dxa in dx]
        return tuple(shape), np.array(shift_strength)

    def multi_coupling_shape(self, dx):
        """Calculate correct shape of the `strengths` for a multi_coupling.

        Parameters
        ----------
        dx : tuple of int
            Translation vector in the lattice for a coupling of two operators.

        Returns
        -------
        coupling_shape : tuple of int
            Len :attr:`dim`. The correct shape for an array specifying the coupling strength.
            `lat_indices` has only rows within this shape.
        shift_lat_indices : array
            Translation vector from lower left corner of box spanned by `dx` to the origin.
        """
        Ls = self.Ls
        shape = [None] * len(Ls)
        shift_strength = [None] * len(Ls)
        for a in range(len(Ls)):
            max_dx, min_dx = np.max(dx[:, a]), np.min(dx[:, a])
            box_dx = max(max_dx, 0) - min(min_dx, 0)
            shape[a] = Ls[a] - box_dx * int(self.bc[a])
            shift_strength[a] = min(0, min_dx)
        return tuple(shape), np.array(shift_strength)

    def plot_sites(self, ax, markers=['o', '^', 's', 'p', 'h', 'D'], **kwargs):
        """Plot the sites of the lattice with markers.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
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
        ax : :class:`matplotlib.axes.Axes`
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
            textkwargs.setdefault('color', kwargs['color'])
            for i, p in enumerate(pos):
                ax.text(p[0], p[1], str(i), **textkwargs)

    def plot_coupling(self, ax, coupling=None, **kwargs):
        """Plot lines connecting nearest neighbors of the lattice.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            The axes on which we should plot.
        coupling : list of (u1, u2, dx)
            By default (``None``), use ``self.pairs['nearest_neighbors']``.
            Specifies the connections to be plotted; iteating over lattice indices `(i0, i1, ...)`,
            we plot a connection from the site ``(i0, i1, ..., u1)`` to the site
            ``(i0+dx[0], i1+dx[1], ..., u2)``, taking into account the boundary conditions.
        **kwargs :
            Further keyword arguments given to ``ax.plot()``.
        """
        if coupling is None:
            coupling = self.pairs['nearest_neighbors']
        kwargs.setdefault('color', 'k')
        Ls = np.array(self.Ls)
        for u1, u2, dx in coupling:
            # TODO: should use `possible_couplings` somehow,
            # but then beriodic boundary conditions screw up the image
            # should plot couplings of periodic boundary conditions
            dx = np.r_[np.array(dx), u2 - u1]  # append the difference in u to dx
            lat_idx_1 = self.order[self._mps_fix_u[u1], :]
            lat_idx_2 = lat_idx_1 + dx[np.newaxis, :]
            lat_idx_2_mod = np.mod(lat_idx_2[:, :-1], Ls)
            # handle boundary conditions
            if self.bc_shift is not None:
                shift = np.sum(((lat_idx_2[:, :-1] - lat_idx_2_mod) // Ls)[:, 1:] * self.bc_shift,
                               axis=1)
                lat_idx_2[:, 0] -= shift
                lat_idx_2_mod[:, 0] = np.mod(lat_idx_2[:, 0], self.Ls[0])
            keep = np.all(
                np.logical_or(
                    lat_idx_2_mod == lat_idx_2[:, :-1],  # not accross the boundary
                    np.logical_not(self.bc)),  # direction has PBC
                axis=1)
            # get positions
            pos1 = self.position(lat_idx_1[keep, :])
            pos2 = self.position(lat_idx_2[keep, :])
            pos = np.stack((pos1, pos2), axis=0)
            # ax.plot connects columns of 2D array by lines
            if pos.shape[2] == 1:
                pos = pos * np.array([[[1., 0]]])  # use broadcasting to add a column with zeros
            if pos.shape[2] != 2:
                raise ValueError("can only plot in 2 dimensions.")
            ax.plot(pos[:, :, 0], pos[:, :, 1], **kwargs)

    def plot_basis(self, ax, **kwargs):
        """Plot arrows indicating the basis vectors of the lattice.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
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

    def plot_bc_identified(self, ax, direction=-1, shift=None, **kwargs):
        """Mark two sites indified by periodic boundary conditions.

        Works only for lattice with a 2-dimensional basis.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            The axes on which we should plot.
        direction : int
            The direction of the lattice along which we should mark the idenitified sites.
            If ``None``, mark it along all directions with periodic boundary conditions.
        shift : None | np.ndarray
            The origin starting from where we mark the identified sites.
            Defaults to the first entry of :attr:`unit_cell_positions`.
        **kwargs :
            Keyword arguments for the used ``ax.plot``.
        """
        if direction is None:
            dirs = [i for i in range(self.dim) if not self.bc[i]]
        else:
            if direction < 0:
                direction += self.dim
            dirs = [direction]
        shift = self.unit_cell_positions[0]
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 10)
        kwargs.setdefault("color", "orange")
        x_y = []
        for i in dirs:
            if self.bc[i]:
                raise ValueError("Boundary conditons are not periodic for given direction")
            x_y.append(shift)
            x_y.append(shift + self.Ls[i] * self.basis[i])
            if self.bc_shift is not None and i > 0:
                x_y[-1] = x_y[-1] - self.bc_shift[i - 1] * self.basis[0]
        x_y = np.array(x_y)
        if x_y.shape[1] == 1:
            x_y = np.hstack([x_y, np.zeros_like(x_y)])
        if x_y.shape[1] != 2:
            raise ValueError("can only plot in 2D")
        ax.plot(x_y[:, 0], x_y[:, 1], **kwargs)

    def _asvalid_latidx(self, lat_idx):
        """convert lat_idx to an ndarray with correct last dimension."""
        lat_idx = np.asarray(lat_idx, dtype=np.intp)
        if lat_idx.shape[-1] != len(self.shape):
            raise ValueError("wrong len of last dimension of lat_idx: " + str(lat_idx.shape))
        return lat_idx

    def _set_bc(self, bc):
        global bc_choices
        if bc in list(bc_choices.keys()):
            bc = [bc_choices[bc]] * self.dim
            self.bc_shift = None
        else:
            bc = list(bc)  # we modify entries...
            self.bc_shift = np.zeros(self.dim - 1, np.int_)
            for i, bc_i in enumerate(bc):
                if isinstance(bc_i, int):
                    if i == 0:
                        raise ValueError("Invalid bc: first entry can't be a shift")
                    self.bc_shift[i - 1] = bc_i
                    bc[i] = bc_choices['periodic']
                else:
                    bc[i] = bc_choices[bc_i]
            if not np.any(self.bc_shift != 0):
                self.bc_shift = None
        self.bc = np.array(bc)

    @property
    def nearest_neighbors(self):
        msg = ("Deprecated access with ``lattice.nearest_neighbors``.\n"
               "Use ``lattice.pairs['nearest_neighbors']`` instead.")
        warnings.warn(msg, FutureWarning)
        return self.pairs['nearest_neighbors']

    @property
    def next_nearest_neighbors(self):
        msg = ("Deprecated access with ``lattice.next_nearest_neighbors``.\n"
               "Use ``lattice.pairs['next_nearest_neighbors']`` instead.")
        warnings.warn(msg, FutureWarning)
        return self.pairs['next_nearest_neighbors']

    @property
    def next_next_nearest_neighbors(self):
        msg = ("Deprecated access with ``lattice.next_next_nearest_neighbors``.\n"
               "Use ``lattice.pairs['next_next_nearest_neighbors']`` instead.")
        warnings.warn(msg, FutureWarning)
        return self.pairs['next_next_nearest_neighbors']


class TrivialLattice(Lattice):
    """Trivial lattice consisting of a single (possibly large) unit cell in 1D.

    This is usefull if you need a valid :class:`Lattice` given just the :meth:`mps_sites`.

    Parameters
    ----------
    mps_sites : list of :class:`~tenpy.networks.site.Site`
        The sites making up a unit cell of the lattice.
    **kwargs :
        Further keyword arguments given to :class:`Lattice`.
    """
    def __init__(self, mps_sites, **kwargs):
        Lattice.__init__(self, [1], mps_sites, **kwargs)


class IrregularLattice(Lattice):
    """A variant of a regular lattice, where we might have extra sites or sites missing.

    .. todo ::
        - this doesn't fully work yet...
    """
    def __init__(self, mps_sites, based_on, order=None):
        self.based_on = based_on
        self._mps_sites = mps_sites
        Lattice.__init__(self,
                         based_on.Ls,
                         based_on.unit_cell,
                         order='default',
                         bc=based_on.bc,
                         bc_MPS=based_on.bc_MPS)
        # don't copy nearest_neighbors, basis, positions etc: no longer valid
        self.N_sites = len(mps_sites)
        self._order = order

    @classmethod
    def from_mps_sites(cls, mps_sites, based_on=None):
        if based_on is None:
            based_on = k
        return cls(mps_sites, based_on)

    @classmethod
    def from_add_sites(self, M):
        raise NotImplementedError()

    def mps_sites(self):
        return self._mps_sites


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
    site : :class:`~tenpy.networks.site.Site`
        the lattice site. The `unit_cell` of the :class:`Lattice` is just ``[site]``.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        If `order` is specified in the form ``('standard', snake_windingi, priority)``,
        the `snake_winding` and `priority` should only be specified for the spatial directions.
        Similarly, `positions` can be specified as a single vector.
    """
    def __init__(self, Ls, site, **kwargs):
        if 'positions' in kwargs:
            Dim = len(kwargs['basis'][0]) if 'basis' in kwargs else len(Ls)
            kwargs['positions'] = np.reshape(kwargs['positions'], (1, Dim))
        if 'order' in kwargs and not isinstance(kwargs['order'], str):
            descr, snake_winding, priority = kwargs['order']
            assert descr == 'standard'
            snake_winding = tuple(snake_winding) + (False, )
            priority = tuple(priority) + (max(priority) + 1., )
            kwargs['order'] = descr, snake_winding, priority
        Lattice.__init__(self, Ls, [site], **kwargs)

    def mps2lat_values(self, A, axes=0, u=None):
        """same as :meth:`Lattice.mps2lat_values`, but ignore ``u``, setting it to ``0``."""
        return super().mps2lat_values(A, axes, 0)


class Chain(SimpleLattice):
    """A chain of L equal sites.

    .. image :: /images/lattices/Chain.*

    Parameters
    ----------
    L : int
        The lenght of the chain.
    site : :class:`~tenpy.networks.site.Site`
        The local lattice site. The `unit_cell` of the :class:`Lattice` is just ``[site]``.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `pairs` are initialize with ``[next_]next_]nearest_neighbors``.
        `positions` can be specified as a single vector.
    """
    dim = 1

    def __init__(self, L, site, **kwargs):
        kwargs.setdefault('pairs', {})
        kwargs['pairs'].setdefault('nearest_neighbors', [(0, 0, np.array([1]))])
        kwargs['pairs'].setdefault('next_nearest_neighbors', [(0, 0, np.array([2]))])
        kwargs['pairs'].setdefault('next_next_nearest_neighbors', [(0, 0, np.array([3]))])
        # and otherwise default values.
        SimpleLattice.__init__(self, [L], site, **kwargs)

    def ordering(self, order):
        """Provide possible orderings of the `N` lattice sites.

        The following orders are defined in this method compared to :meth:`Lattice.ordering`:

        ================== ============================================================
        `order`            Resulting order
        ================== ============================================================
        ``'default'``      ``0, 1, 2, 3, 4, ... ,L-1``
        ------------------ ------------------------------------------------------------
        ``'folded'``       ``0, L-1, 1, L-2, ... , L//2``.
                           This order might be usefull if you want to consider a
                           ring with periodic boundary conditions with a finite MPS:
                           It avoids the ultra-long range of the coupling from site
                           0 to L present in the default order.
        ================== ============================================================
        """
        if isinstance(order, str) and order == 'default' or order == 'folded':
            (L, u) = self.shape
            assert u == 1
            ordering = np.zeros([L, 2], dtype=np.intp)
            if order == 'default':
                ordering[:, 0] = np.arange(L, dtype=np.intp)
            elif order == 'folded':
                order = []
                for i in range(L // 2):
                    order.append(i)
                    order.append(L - i - 1)
                if L % 2 == 1:
                    order.append(L // 2)
                assert len(order) == L
                ordering[:, 0] = np.array(order, dtype=np.intp)
            else:
                assert (False)  # should not be possible
            return ordering
        return super().ordering(order)


class Ladder(Lattice):
    """A ladder coupling two chains.

    .. image :: /images/lattices/Ladder.*

    Parameters
    ----------
    L : int
        The length of each chain, we have 2*L sites in total.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both chains.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `[[next_]next_]nearest_neighbors` are set accordingly.
    """
    dim = 1

    def __init__(self, L, sites, **kwargs):
        sites = _parse_sites(sites, 2)
        basis = np.array([[1., 0.]])
        pos = np.array([[0., 0.], [0., 1.]])
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        NN = [(0, 0, np.array([1])), (1, 1, np.array([1])), (0, 1, np.array([0]))]
        nNN = [(0, 1, np.array([1])), (1, 0, np.array([1]))]
        nnNN = [(0, 0, np.array([2])), (1, 1, np.array([2]))]
        kwargs.setdefault('pairs', {})
        kwargs['pairs'].setdefault('nearest_neighbors', NN)
        kwargs['pairs'].setdefault('next_nearest_neighbors', nNN)
        kwargs['pairs'].setdefault('next_next_nearest_neighbors', nnNN)
        Lattice.__init__(self, [L], sites, **kwargs)


class Square(SimpleLattice):
    """A square lattice.

    .. image :: /images/lattices/Square.*

    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    site : :class:`~tenpy.networks.site.Site`
        The local lattice site. The `unit_cell` of the :class:`Lattice` is just ``[site]``.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `[[next_]next_]nearest_neighbors` are set accordingly.
        If `order` is specified in the form ``('standard', snake_windingi, priority)``,
        the `snake_winding` and `priority` should only be specified for the spatial directions.
        Similarly, `positions` can be specified as a single vector.
    """
    dim = 2

    def __init__(self, Lx, Ly, site, **kwargs):
        NN = [(0, 0, np.array([1, 0])), (0, 0, np.array([0, 1]))]
        nNN = [(0, 0, np.array([1, 1])), (0, 0, np.array([1, -1]))]
        nnNN = [(0, 0, np.array([2, 0])), (0, 0, np.array([0, 2]))]
        kwargs.setdefault('pairs', {})
        kwargs['pairs'].setdefault('nearest_neighbors', NN)
        kwargs['pairs'].setdefault('next_nearest_neighbors', nNN)
        kwargs['pairs'].setdefault('next_next_nearest_neighbors', nnNN)
        SimpleLattice.__init__(self, [Lx, Ly], site, **kwargs)


class Triangular(SimpleLattice):
    """A triangular lattice.

    .. image :: /images/lattices/Triangular.*

    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    site : :class:`~tenpy.networks.site.Site`
        The local lattice site. The `unit_cell` of the :class:`Lattice` is just ``[site]``.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `[[next_]next_]nearest_neighbors` are set accordingly.
        If `order` is specified in the form ``('standard', snake_windingi, priority)``,
        the `snake_winding` and `priority` should only be specified for the spatial directions.
        Similarly, `positions` can be specified as a single vector.
    """
    dim = 2

    def __init__(self, Lx, Ly, site, **kwargs):
        sqrt3_half = 0.5 * np.sqrt(3)  # = cos(pi/6)
        basis = np.array([[sqrt3_half, 0.5], [0., 1.]])
        NN = [(0, 0, np.array([1, 0])), (0, 0, np.array([-1, 1])), (0, 0, np.array([0, -1]))]
        nNN = [(0, 0, np.array([2, -1])), (0, 0, np.array([1, 1])), (0, 0, np.array([-1, 2]))]
        nnNN = [(0, 0, np.array([2, 0])), (0, 0, np.array([0, 2])), (0, 0, np.array([-2, 2]))]
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('pairs', {})
        kwargs['pairs'].setdefault('nearest_neighbors', NN)
        kwargs['pairs'].setdefault('next_nearest_neighbors', nNN)
        kwargs['pairs'].setdefault('next_next_nearest_neighbors', nnNN)
        SimpleLattice.__init__(self, [Lx, Ly], site, **kwargs)


class Honeycomb(Lattice):
    """A honeycomb lattice.

    .. image :: /images/lattices/Honeycomb.*

    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `[[next_]next_]nearest_neighbors` are set accordingly.
        For the Honeycomb lattice ``'fourth_nearest_neighbors', 'fifth_nearest_neighbors'``
        are set in :attr:`pairs`.
    """
    dim = 2

    def __init__(self, Lx, Ly, sites, **kwargs):
        sites = _parse_sites(sites, 2)
        basis = np.array(([0.5 * np.sqrt(3), 0.5], [0., 1]))
        delta = np.array([1 / (2. * np.sqrt(3.)), 0.5])
        pos = (-delta / 2., delta / 2)
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        NN = [(0, 1, np.array([0, 0])), (1, 0, np.array([1, 0])), (1, 0, np.array([0, 1]))]
        nNN = [(0, 0, np.array([1, 0])), (0, 0, np.array([0, 1])), (0, 0, np.array([1, -1])),
               (1, 1, np.array([1, 0])), (1, 1, np.array([0, 1])), (1, 1, np.array([1, -1]))]
        nnNN = [(1, 0, np.array([1, 1])), (0, 1, np.array([-1, 1])), (0, 1, np.array([1, -1]))]
        NN4 = [(0, 1, np.array([0, 1])), (0, 1, np.array([1, 0])), (0, 1, np.array([1, -2])),
               (0, 1, np.array([0, -2])), (0, 1, np.array([-2, 0])), (0, 1, np.array([-2, 1]))]
        NN5 = [(0, 0, np.array([1, 1])), (0, 0, np.array([2, -1])), (0, 0, np.array([1, -2])),
               (0, 0, np.array([-1, -1])), (0, 0, np.array([-2, 1])), (0, 0, np.array([-1, 2]))]
        kwargs.setdefault('pairs', {})
        kwargs['pairs'].setdefault('nearest_neighbors', NN)
        kwargs['pairs'].setdefault('next_nearest_neighbors', nNN)
        kwargs['pairs'].setdefault('next_next_nearest_neighbors', nnNN)
        kwargs['pairs'].setdefault('fourth_nearest_neighbors', NN4)
        kwargs['pairs'].setdefault('fifth_nearest_neighbors', NN5)
        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)

    def ordering(self, order):
        """Provide possible orderings of the `N` lattice sites.

        The following orders are defined in this method compared to :meth:`Lattice.ordering`:

        ================== =========================== =============================
        `order`            equivalent `priority`       equivalent ``snake_winding``
        ================== =========================== =============================
        ``'default'``      (0, 2, 1)                   (False, False, False)
        ``'snake'``        (0, 2, 1)                   (False, True, False)
        ================== =========================== =============================
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

    @property
    def fourth_nearest_neighbors(self):
        msg = ("Deprecated access with ``lattice.fourth_nearest_neighbors``.\n"
               "Use ``lattice.pairs['fourth_nearest_neighbors']`` instead.")
        warnings.warn(msg, FutureWarning)
        return self.pairs['fourth_nearest_neighbors']

    @property
    def fifth_nearest_neighbors(self):
        msg = ("Deprecated access with ``lattice.fifth_nearest_neighbors``.\n"
               "Use ``lattice.pairs['fifth_nearest_neighbors']`` instead.")
        warnings.warn(msg, FutureWarning)
        return self.pairs['fifth_nearest_neighbors']


class Kagome(Lattice):
    """A Kagome lattice.

    .. image :: /images/lattices/Kagome.*

    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `[[next_]next_]nearest_neighbors` are set accordingly.
    """
    dim = 2

    def __init__(self, Lx, Ly, sites, **kwargs):
        sites = _parse_sites(sites, 3)
        #     2
        #    / \
        #   /   \
        #  0-----1
        pos = np.array([[0, 0], [1, 0], [0.5, 0.5 * 3**0.5]])
        basis = [2 * pos[1], 2 * pos[2]]
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        NN = [(0, 1, np.array([0, 0])), (0, 2, np.array([0, 0])), (1, 2, np.array([0, 0])),
              (1, 0, np.array([1, 0])), (2, 0, np.array([0, 1])), (2, 1, np.array([-1, 1]))]
        nNN = [(0, 1, np.array([0, -1])), (0, 2, np.array([1, -1])), (1, 0, np.array([1, -1])),
               (1, 2, np.array([1, 0])), (2, 0, np.array([1, 0])), (2, 1, np.array([0, 1]))]
        nnNN = [(0, 0, np.array([1, -1])), (1, 1, np.array([0, 1])), (2, 2, np.array([1, 0]))]
        kwargs.setdefault('pairs', {})
        kwargs['pairs'].setdefault('nearest_neighbors', NN)
        kwargs['pairs'].setdefault('next_nearest_neighbors', nNN)
        kwargs['pairs'].setdefault('next_next_nearest_neighbors', nnNN)
        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


def get_lattice(lattice_name):
    """Given the name of a :class:`Lattice` class, create an instance of it with gi.

    Parameters
    ----------
    lattice_name : str
        Name of a :class:`Lattice` class defined in the module :mod:`~tenpy.models.lattice`,
        for example ``"Chain", "Square", "Honeycomb", ...``.
    *args, **kwargs
        Arguments and keyword-arguments for the initialization of the specified lattice class.

    Returns
    -------
    LatticeClass : (subclass of) :class:`Lattice`
        An instance of the lattice class specified by `lattice_name`.
    """
    LatticeClass = globals()[lattice_name]
    assert issubclass(LatticeClass, Lattice)
    return LatticeClass


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
    priority : ``None`` | tuple of float
        If ``None`` (default), use C-Style ordering.
        Otherwise, this defines the priority along which direction to wind first;
        the direction with the highest priority increases fastest.
        For example, "C-Style" order is enforced by ``priority=(0, 1, 2, ...)``,
        and Fortrans F-style order is enforced by ``priority=(dim, dim-1, ..., 1, 0)``
    group : ``None`` | tuple of tuple
        If ``None`` (default), ignore it.
        Otherwise, it specifies that we group the fastests changing dimension

    Returns
    -------
    order : ndarray (np.prod(shape), len(shape))
        An order of the sites for :attr:`Lattice.order` in the specified `ordering`.

    See also
    --------
    Lattice.ordering : method in :class:`Lattice` to obtain the order from parameters.
    Lattice.plot_order : visualizes the resulting order in a :class:`Lattice`.
    get_order_grouped : a variant grouping sites of the unit cell.
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
        L = shape[dim - 1 - i]
        snake = snake_winding[dim - i]  # previous direction snake?
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


def get_order_grouped(shape, groups):
    """Variant of :func:`get_order`, grouping some sites of the unit cell.

    In this function, the word 'direction' referst to a physical direction of the lattice or the
    index `u` of the unit cell as an "artificial direction".
    This function is usefull for lattices with a unit cell of more than 2 sites (e.g. Kagome).
    The argument `group` is a
    To explain the order, assume we have a 3-site unit cell in a 2D lattice with shape
    (Lx, Ly, Lu).
    Calling this function with groups=((1,), (2, 0)) returns an order of the following form::

        # columns: [x, y, u]
        [0, 0, 1]  # first for u = 1 along y
        [0, 1, 1]
            :
        [0, Ly-1, 1]
        [0, 0, 2]  # then for u = 2 and 0
        [0, 0, 0]
        [0, 1, 2]
        [0, 1, 0]
            :
        [0, Ly-1, 2]
        [0, Ly-1, 0]
        # and then repeat the above for increasing `x`.



    Parameters
    ----------
    shape : tuple of int
        The shape of the lattice, i.e., the length in each direction.
    groups : tuple of tuple of int
        A partition and reordering of range(shape[-1]) into smaller groups.
        The ordering goes first within a group, then along the last spatial dimensions, then
        changing between different groups and finally in Cstyle order along the remaining spatial
        dimensions.

    Returns
    -------
    order : ndarray (np.prod(shape), len(shape))
        An order of the sites for :attr:`Lattice.order` in the specified `ordering`.

    See also
    --------
    :meth:`Lattice.ordering` : method in :class:`Lattice` to obtain the order from parameters.
    :meth:`Lattice.plot_order` : visualizes the resulting order in a :class:`Lattice`.
    """
    Ly = shape[-2]
    Lu = shape[-1]
    N_sites = np.prod(shape)
    # sanity check for argument group
    groups = list(groups)
    all = [g for gr in groups for g in gr]
    all_set = set(all)
    assert all_set == set(range(Lu))  # does every number appear?
    assert len(all) == len(all_set) == Lu  # exactly once?
    assert len(shape) > 1
    rLy = np.arange(Ly)
    pre_order = np.empty((Ly * Lu, 2), dtype=np.intp)
    start = 0
    for gr in groups:
        gr = np.array(gr)
        Lgr = len(gr)
        end = start + Lgr * Ly
        pre_order[start:end, 0] = np.repeat(rLy, Lgr)
        pre_order[start:end, 1] = np.tile(gr, Ly)
        start = end
    other_order = get_order(shape[:-2], [False])
    order = np.empty((N_sites, len(shape)), dtype=np.intp)
    order[:, :-2] = np.repeat(other_order, Ly * Lu, axis=0)
    order[:, -2:] = np.tile(pre_order, (N_sites // (Ly * Lu), 1))
    return order


def _parse_sites(sites, expected_number):
    try:  # allow to specify a single site
        iter(sites)
    except TypeError:
        return [sites] * expected_number
    if len(sites) != expected_number:
        raise ValueError("need to specify a single site or exactly {0:d}, got {1:d}".format(
            expected_number, len(sites)))
    return sites
