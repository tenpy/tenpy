"""Classes to define the lattice structure of a model.

The base class :class:`Lattice` defines the general structure of a lattice,
you can subclass this to define you own lattice.
The :class:`SimpleLattice` is a slight simplification for lattices with a single-site unit cell.
Further, we have some predefined lattices, namely
:class:`Chain`, :class:`Ladder` in 1D and
:class:`Square`, :class:`Triangular`, :class:`Honeycomb`, and :class:`Kagome` in 2D.

The :class:`IrregularLattice` provides a way to remove or add sites to an existing, regular
lattice.

See also the :doc:`/intro/model` and :doc:`/intro/lattices`.
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np
import itertools
import warnings
import copy

from ..networks.site import Site
from ..tools.misc import (to_iterable, to_array, to_iterable_of_len, inverse_permutation,
                          get_close, find_subclass)
from ..networks.mps import MPS  # only to check boundary conditions

__all__ = [
    'Lattice', 'TrivialLattice', 'IrregularLattice', 'HelicalLattice', 'SimpleLattice', 'Chain',
    'Ladder', 'Square', 'Triangular', 'Honeycomb', 'Kagome', 'get_lattice', 'get_order',
    'get_order_grouped'
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
    i.e., if the site at ``(x_0, x_1, ..., x_{dim-1},u)`` has MPS index `i`, the site at
    at ``(x_0 + Ls[0], x_1, ..., x_{dim-1}, u)`` corresponds to MPS index ``i + N_sites``.
    Use :meth:`mps2lat_idx` and :meth:`lat2mps_idx` for conversion of indices.
    The function :meth:`mps2lat_values` performs the necessary reshaping and re-ordering from
    arrays indexed in MPS form to arrays indexed in lattice form.

    .. deprecated :: 0.5.0
        The parameters and attributes `nearest_neighbors`, `next_nearest_neighbors` and
        `next_next_nearest_neighbors` are deprecated. Instead, we use a dictionary `pairs`
        with those names as keys and the corresponding values as specified before.

    Parameters
    ----------
    Ls : list of int
        the length in each direction
    unit_cell : list of :class:`~tenpy.networks.site.Site`
        The sites making up a unit cell of the lattice.
        If you want to specify it only after initialization, use ``None`` entries in the list.
    order : str | ``('standard', snake_winding, priority)`` | ``('grouped', groups, ...)``
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
        to generate couplings over each pair of ,e.g., ``'nearest_neighbors'``.
        Note that this adds couplings for each pair *only in one direction*!

    Attributes
    ----------
    Ls : tuple of int
        the length in each direction.
    shape : tuple of int
        the 'shape' of the lattice, same as ``Ls + (len(unit_cell), )``
    N_cells : int
        the number of unit cells in the lattice, ``np.prod(self.Ls)``.
    N_sites : int
        the number of sites in the lattice, ``np.prod(self.shape)``.
    N_sites_per_ring : int
        Defined as ``N_sites / Ls[0]``, for an infinite system the number of cites per "ring".
    N_rings : int
        Alias for ``Ls[0]``, for an infinite system the number of "rings" in the unit cell.
    unit_cell : list of :class:`~tenpy.networks.site.Site`
        the sites making up a unit cell of the lattice.
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
    segement_first_last : tuple of int
        The `first` and `last` MPS sites for "segment" :attr:`bc_MPS`; not set otherwise.
    _order : ndarray (N_sites, dim+1)
        The place where :attr:`order` is stored.
    _strides : ndarray (dim, )
        necessary for :meth:`lat2mps_idx`.
    _perm : ndarray (N, )
        permutation needed to make `order` lexsorted, ``_perm = np.lexsort(_order.T)``.
    _mps2lat_vals_idx : ndarray `shape`
        index array for reshape/reordering in :meth:`mps2lat_vals`
    _mps_fix_u : tuple of ndarray (N_cells, ) np.intp
        for each site of the unit cell an index array selecting the mps indices of that site.
    _mps2lat_vals_idx_fix_u : tuple of ndarray of shape `Ls`
        similar as `_mps2lat_vals_idx`, but for a fixed `u` picking a site from the unit cell.
    """
    Lu = None  #: the (expected) number of sites in the unit cell, ``len(unit_cell)``.
    dim = None  #: the dimension of the lattice

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
        self.unit_cell = list(unit_cell)
        self._set_Ls(Ls)  # after setting unit_cell
        if positions is None:
            positions = np.zeros((len(self.unit_cell), self.dim))
        if basis is None:
            basis = np.eye(self.dim)
        self.unit_cell_positions = np.array(positions)
        self.basis = np.array(basis)
        self.boundary_conditions = bc  # property setter for self.bc and self.bc_shift
        self.bc_MPS = bc_MPS
        # calculate order for MPS
        self.order = self.ordering(order)
        # uses attribute setter to calculte _mps2lat_vals_idx_fix_u etc and lat2mps
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
        if not isinstance(self, HelicalLattice):
            assert self.N_cells == np.prod(self.Ls)
        if self.bc.shape != (self.dim, ):
            raise ValueError("Wrong len of bc")
        assert self.bc.dtype == bool
        chinfo = None
        for site in self.unit_cell:
            if not isinstance(site, Site):
                continue
            if chinfo is None:
                chinfo = site.leg.chinfo
            if site.leg.chinfo != chinfo:
                raise ValueError("All sites in the lattice must have the same ChargeInfo!"
                                 " Call tenpy.networks.site.set_common_charges() before "
                                 "giving them to the lattice!")
        if self.basis.shape[0] != self.dim:
            raise ValueError("Need one basis vector for each direction!")
        if self.unit_cell_positions.shape[0] != len(self.unit_cell):
            raise ValueError("Need one position for each site in the unit cell.")
        if self.basis.shape[1] != self.unit_cell_positions.shape[1]:
            raise ValueError("Different space dimensions of `basis` and `unit_cell_positions`")
        if self.bc_MPS not in MPS._valid_bc:
            raise ValueError("invalid MPS boundary conditions")
        if self.bc[0] and self.bc_MPS == 'infinite':
            raise ValueError("Need periodic boundary conditions along the x-direction "
                             "for 'infinite' `bc_MPS`")
        if not isinstance(self, (IrregularLattice, HelicalLattice)):
            assert self.N_sites == np.prod(self.shape)
            # if one of the following assert fails,
            # the `ordering` function might have returned an invalid array
            assert np.all(self._order >= 0) and np.all(self._order <= self.shape)
            # rows of `order` unique and _perm correct?
            assert np.all(
                np.sum(self._order * self._strides, axis=1)[self._perm] == np.arange(self.N_sites))

    def copy(self):
        """Shallow copy of `self`."""
        return copy.copy(self)

    def save_hdf5(self, hdf5_saver, h5gr, subpath):
        """Export `self` into a HDF5 file.

        This method saves all the data it needs to reconstruct `self` with :meth:`from_hdf5`.

        Specifically, it saves
        :attr:`unit_cell`, :attr:`Ls`, :attr:`unit_cell_positions`, :attr:`basis`,
        :attr:`boundary_conditions`, :attr:`pairs` under their name,
        :attr:`bc_MPS` as ``"boundary_conditions_MPS"``, and
        :attr:`order` as ``"order_for_MPS"``.
        Moreover, it saves :attr:`dim` and :attr:`N_sites` as HDF5 attributes.

        Parameters
        ----------
        hdf5_saver : :class:`~tenpy.tools.hdf5_io.Hdf5Saver`
            Instance of the saving engine.
        h5gr : :class`Group`
            HDF5 group which is supposed to represent `self`.
        subpath : str
            The `name` of `h5gr` with a ``'/'`` in the end.
        """
        hdf5_saver.save(self.unit_cell, subpath + "unit_cell")
        hdf5_saver.save(np.array(self.Ls, int), subpath + "lengths")
        hdf5_saver.save(self.unit_cell_positions, subpath + "unit_cell_positions")
        hdf5_saver.save(self.basis, subpath + "basis")
        hdf5_saver.save(self.boundary_conditions, subpath + "boundary_conditions")
        hdf5_saver.save(self.bc_MPS, subpath + "boundary_condition_MPS")
        hdf5_saver.save(self.order, subpath + "order_for_MPS")
        hdf5_saver.save(self.pairs, subpath + "pairs")
        # not necessary for loading, but still usefull
        h5gr.attrs["dim"] = self.dim
        h5gr.attrs["N_sites"] = self.N_sites
        if hasattr(self, 'segement_first_last'):
            first, last = self.segment_first_last
            h5gr.attrs['segment_first'] = first
            h5gr.attrs['segment_last'] = last

    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):
        """Load instance from a HDF5 file.

        This method reconstructs a class instance from the data saved with :meth:`save_hdf5`.

        Parameters
        ----------
        hdf5_loader : :class:`~tenpy.tools.hdf5_io.Hdf5Loader`
            Instance of the loading engine.
        h5gr : :class:`Group`
            HDF5 group which is represent the object to be constructed.
        subpath : str
            The `name` of `h5gr` with a ``'/'`` in the end.

        Returns
        -------
        obj : cls
            Newly generated class instance containing the required data.
        """
        obj = cls.__new__(cls)  # create class instance, no __init__() call
        hdf5_loader.memorize_load(h5gr, obj)

        obj.unit_cell = hdf5_loader.load(subpath + "unit_cell")
        Ls = hdf5_loader.load(subpath + "lengths")
        obj._set_Ls(Ls)
        obj.unit_cell_positions = hdf5_loader.load(subpath + "unit_cell_positions")
        obj.basis = hdf5_loader.load(subpath + "basis")
        obj.boundary_conditions = hdf5_loader.load(subpath + "boundary_conditions")
        obj.bc_MPS = hdf5_loader.load(subpath + "boundary_condition_MPS")
        obj.order = hdf5_loader.load(subpath + "order_for_MPS")  # property setter!
        obj.pairs = hdf5_loader.load(subpath + "pairs")
        if 'segment_first' in h5gr.attrs:
            first = h5gr.attrs['segment_first']
            last = h5gr.attrs['segment_last']
            obj.segment_first_last = first, last
        obj.test_sanity()
        return obj

    @property
    def dim(self):
        """The dimension of the lattice."""
        return len(self.Ls)

    @property
    def order(self):
        """Defines an ordering of the lattice sites, thus mapping the lattice to a 1D chain.

        Each row of the array contains the lattice indices for one site,
        the order of the rows thus specifies a path through the lattice,
        along which an MPS will wind through through the lattice.

        You can visualize the order with :meth:`plot_order`.
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

        .. plot ::

            import matplotlib.pyplot as plt
            from tenpy.models import lattice
            fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 6))
            orders = ['Cstyle', 'snakeCstyle', 'Fstyle', 'snakeFstyle']
            lat = lattice.Square(5, 3, None, bc='periodic')
            for order, ax in zip(orders, axes.flatten()):
                lat.order = lat.ordering(order)
                lat.plot_order(ax, linestyle=':')
                lat.plot_sites(ax)
                lat.plot_basis(ax, origin=-0.25*(lat.basis[0] + lat.basis[1]))
                ax.set_title(repr(order))
                ax.set_aspect('equal')
                ax.set_xlim(-1)
                ax.set_ylim(-1)
            plt.show()

        .. note ::
            For lattices with a non-trivial unit cell (e.g. Honeycomb, Kagome), the
            grouped order might be more appropriate, see :func:`get_order_grouped`

        Parameters
        ----------
        order : str | ``('standard', snake_winding, priority)`` | ``('grouped', groups, ...)``
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
                snake_winding, priority = order[1:]
            elif descr == 'grouped':
                return get_order_grouped(self.shape, *order[1:])
            else:
                raise ValueError("unknown ordering " + repr(order))
        return get_order(self.shape, snake_winding, priority)

    @property
    def boundary_conditions(self):
        """Human-readable list of boundary conditions from :attr:`bc` and :attr:`bc_shift`.

        Returns
        -------
        boundary_conditions : list of str
            List of ``"open"`` or ``"periodic"``, one entry for each direction of the lattice.
        """
        global bc_choices
        bc_choices_reverse = dict([(v, k) for (k, v) in bc_choices.items()])
        bc = [bc_choices_reverse[bc] for bc in self.bc]
        if self.bc_shift is not None:
            for i, shift in enumerate(self.bc_shift):
                assert bc[i + 1] == "periodic"
                bc[i + 1] = int(shift)
        return bc

    @boundary_conditions.setter
    def boundary_conditions(self, bc):
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

    def extract_segment(self, first=0, last=None, enlarge=None):
        """Extract a finite segment from an infinite/large system.

        Parameters
        ----------
        first, last : int
            The first and last site to *include* into the segment.
            `last` defaults to :attr:`L` - 1, i.e., the MPS unit cell for infinite MPS.
        enlarge : int
            Instead of specifying the `first` and `last` site, you can specify this factor
            by how much the MPS unit cell should be enlarged.

        Returns
        -------
        copy : :class:`Lattice`
            A copy of `self` with "segment" :attr:`bc_MPS` and :attr:`segment_first_last` set.
        """
        cp = self.copy()
        L = cp.N_sites
        assert first >= 0
        if enlarge is not None:
            if cp.bc_MPS != 'infinite':
                raise ValueError("enlarge only possible for infinite MPS")
            if last is not None or first != 0:
                raise ValueError("specifiy either `first`+`last` or `enlarge`!")
            assert enlarge > 0
            last = enlarge * L - 1
        elif last is None:
            last = L - 1
            enlarge = 1
        else:
            enlarge = last + 1 // L
        assert enlarge > 0
        if enlarge > 1:
            cp.enlarge_mps_unit_cell(enlarge)
        if first >= last:
            raise ValueError(f"need first < last, got {first:d}, {last:d}")
        if first > 0 or last < cp.N_sites - 1:
            # take out some parts of the lattice
            remove = list(range(0, first)) + list(range(last + 1, cp.N_sites))
            cp = IrregularLattice(cp, remove=remove)
        cp.bc_MPS = 'segment'
        cp.segment_first_last = first, last
        return cp

    def enlarge_mps_unit_cell(self, factor=2):
        """Repeat the unit cell for infinite MPS boundary conditions; in place.

        Parameters
        ----------
        factor : int
            The new number of sites in the MPS unit cell will be increased from `N_sites` to
            ``factor*N_sites_per_ring``. Since MPS unit cells are repeated in the `x`-direction
            in our convetion, the lattice shape goes from
            ``(Lx, Ly, ..., Lu)`` to ``(Lx*factor, Ly, ..., Lu)``.
        """
        if self.bc_MPS != "infinite":
            raise ValueError("can only enlarge the MPS unit cell for infinite MPS.")
        new_Ls = list(self.Ls)
        old_Lx = new_Ls[0]
        new_Ls[0] = old_Lx * factor
        old_order = self.order
        new_order = []
        for i in range(factor):
            order = old_order.copy()
            shift_x = i * old_Lx
            order[:, 0] += shift_x
            new_order.append(order)
        new_order = np.vstack(new_order)
        # now update the contents of `self`
        self._set_Ls(new_Ls)
        self.order = new_order  # property setter
        self.test_sanity()

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
                lat[..., 0] += (i0 - i) * self.N_rings // self.N_sites
                # N_sites_per_ring might not be set for IrregularLattice
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
            i += i_shift * self.N_sites // self.N_rings
            # N_sites_per_ring might not be set for IrregularLattice
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
            Reshaped and reordered verions of A. Such that MPS indices along the specified axes
            are replaced by lattice indices, i.e., if MPS index `j` maps to lattice site
            `(x0, x1, x2)`, then ``res_A[..., x0, x1, x2, ...] = A[..., j, ...]``.

        Examples
        --------
        Say you measure expection values of an onsite term for an MPS, which gives you an 1D array
        `A`, where `A[i]` is the expectation value of the site given by ``self.mps2lat_idx(i)``.
        Then this function gives you the expectation values ordered by the lattice:

        .. testsetup :: mps2lat_values

            lat = tenpy.models.lattice.Honeycomb(10, 3, None)
            A = np.arange(60)
            C = np.arange(60*60).reshape(60, 60)
            A_res = lat.mps2lat_values(A)

        .. doctest :: mps2lat_values

            >>> print(lat.shape, A.shape)
            (10, 3, 2) (60,)
            >>> A_res = lat.mps2lat_values(A)
            >>> A_res.shape
            (10, 3, 2)
            >>> A_res[tuple(lat.mps2lat_idx(5))] == A[5]
            True

        If you have a correlation function ``C[i, j]``, it gets just slightly more complicated:

        .. doctest :: mps2lat_values

            >>> print(lat.shape, C.shape)
            (10, 3, 2) (60, 60)
            >>> lat.mps2lat_values(C, axes=[0, 1]).shape
            (10, 3, 2, 10, 3, 2)

        If the unit cell consists of different physical sites, an onsite operator might be defined
        only on one of the sites in the unit cell. Then you can use :meth:`mps_idx_fix_u` to get
        the indices of sites it is defined on, measure the operator on these sites, and use
        the argument `u` of this function.

        .. doctest :: mps2lat_values

            >>> u = 0
            >>> idx_subset = lat.mps_idx_fix_u(u)
            >>> A_u = A[idx_subset]
            >>> A_u_res = lat.mps2lat_values(A_u, u=u)
            >>> A_u_res.shape
            (10, 3)
            >>> np.all(A_res[:, :, u] == A_u_res[:, :])
            True

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

    def mps2lat_values_masked(self, A, axes=-1, mps_inds=None, include_u=None):
        """Reshape/reorder an array `A` to replace an MPS index by lattice indices.

        This is a generalization of :meth:`mps2lat_values` allowing for the case of an arbitrary
        set of MPS indices present in each axis of `A`.

        Parameters
        ----------
        A : ndarray
            Some values.
        axes : (iterable of) int
            Chooses the axis of `A` which should be replaced.
            If multiple axes are given, you also need to give multiple index arrays as `mps_inds`.
        mps_inds : (list of) 1D ndarray
            Specifies for each `axis` in `axes`, for which MPS indices we have values in the
            corresponding `axis` of `A`.
            Defaults to ``[np.arange(A.shape[ax]) for ax in axes]``.
            For indices accross the MPS unit cell and "infinite" `bc_MPS`,
            we shift `x_0` accordingly.
        include_u : (list of) bool
            Specifies for each `axis` in `axes`, whether the `u` index of the lattice should be
            included into the output array `res_A`. Defaults to ``len(self.unit_cell) > 1``.

        Returns
        -------
        res_A : np.ma.MaskedArray
            Reshaped and reordered copy of A. Such that MPS indices along the specified axes
            are replaced by lattice indices, i.e., if MPS index `j` maps to lattice site
            `(x0, x1, x2)`, then ``res_A[..., x0, x1, x2, ...] = A[..., mps_inds[j], ...]``.
        """
        try:
            iter(axes)
        except TypeError:  # axes is single int
            axes = [axes]
            mps_inds = [mps_inds]
            include_u = [include_u]
        else:  # iterable axes
            if mps_inds is None:
                mps_inds = [None] * len(axes)
            if include_u is None:
                include_u = [None] * len(axes)
            if len(axes) != len(mps_inds) or len(axes) != len(include_u):
                raise ValueError("Lenght of `axes`, `mps_inds` and `include_u` different")
        # sort axes ascending
        axes = [(ax + A.ndim if ax < 0 else ax) for ax in axes]

        # goal: use numpy advanced indexing for the data copy
        lat_inds = []  # lattice indices to be used
        new_shapes = []  # shape to be
        for ax, mps_inds_ax, include_u_ax in zip(axes, mps_inds, include_u):
            if mps_inds_ax is None:  # default
                mps_inds_ax = np.arange(A.shape[ax])
            if include_u_ax is None:  # default
                include_u_ax = (len(self.unit_cell) > 1)
            if mps_inds_ax.ndim != 1:
                raise ValueError("got non-1D array in `mps_inds` " + str(mps_inds_ax.shape))
            lat_inds_ax = self.mps2lat_idx(mps_inds_ax)
            shape = list(self.shape)
            max_i = np.max(mps_inds_ax)
            if max_i >= self.N_sites:
                shape[0] += (max_i - self.N_sites) * self.N_rings // self.N_sites + 1
            min_i = np.min(mps_inds_ax)
            if min_i < 0:
                # we use numpy indexing to simply wrap around negative indices
                shape[0] += (abs(min_i) - 1) * self.N_rings // self.N_sites + 1
            if not include_u_ax:
                shape = shape[:-1]
                lat_inds_ax = lat_inds_ax[:, :-1]
            new_shapes.append(shape)
            lat_inds.append(lat_inds_ax)

        res_A_ndim = A.ndim - len(axes) + sum([len(s) for s in new_shapes])
        res_A_shape = []
        res_A_inds = []
        dim_before = 0
        dim_after = A.ndim - 1
        for ax in range(A.ndim):
            if ax in axes:
                i = axes.index(ax)
                inds_ax = lat_inds[i].T
                res_A_shape.extend(new_shapes[i])
                for inds in lat_inds[i].T:
                    inds = inds.reshape([1] * dim_before + [len(inds)] + [1] * dim_after)
                    res_A_inds.append(inds)
            else:
                d = A.shape[ax]
                res_A_shape.append(d)
                inds = np.arange(d).reshape([1] * dim_before + [d] + [1] * dim_after)
                res_A_inds.append(inds)
            dim_before += 1
            dim_after -= 1

        # and finally we are in the position to create the masked Array
        fill_value = np.ma.array([0, 1], dtype=A.dtype).get_fill_value()
        res_A_data = np.full(res_A_shape, fill_value, dtype=A.dtype)
        res_A = np.ma.array(res_A_data, mask=True, fill_value=fill_value)

        res_A[tuple(res_A_inds)] = A  # copy data, automatically unmasks entries
        return res_A

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

        .. deprecated :: 0.5.0
            Use :meth:`count_neighbors` instead.
        """
        msg = "Use ``count_neighbors(u, 'nearest_neighbors')`` instead."
        warnings.warn(msg, FutureWarning)
        return self.count_neighbors(u, 'nearest_neighbors')

    def number_next_nearest_neighbors(self, u=0):
        """Deprecated.

        .. deprecated :: 0.5.0
            Use :meth:`count_neighbors` instead.
        """
        msg = "Use ``count_neighbors(u, 'next_nearest_neighbors')`` instead."
        warnings.warn(msg, FutureWarning)
        return self.count_neighbors(u, 'next_nearest_neighbors')

    def distance(self, u1, u2, dx):
        """Get the distance for a given coupling between two sites in the lattice.

        The `u1`, `u2`, `dx` parameters are defined in analogy with
        :meth:`~tenpy.models.model.CouplingModel.add_coupling`, i.e., this function
        calculates the distance between a pair of operators added with `add_coupling` (using the
        :attr:`basis` and :attr:`unit_cell_positions` of the lattice).

        .. warning ::
            This function ignores "wrapping" arround the cylinder in the case of periodic boundary
            conditions.

        Parameters
        ----------
        u1, u2 : int
            Indices within the unit cell; the `u1` and `u2` of
            :meth:`~tenpy.models.model.CouplingModel.add_coupling`
        dx : array
            Length :attr:`dim`. The translation in terms of basis vectors for the coupling.

        Returns
        -------
        distance : float
            The distance between site at lattice indices ``[x, y, u1]`` and
            ``[x + dx[0], y + dx[1], u2]``, **ignoring** any boundary effects.
        """
        vec_dist = self.unit_cell_positions[u2] - self.unit_cell_positions[u1]
        for ax in range(self.dim):
            vec_dist = vec_dist + dx[..., ax, np.newaxis] * self.basis[ax]
        return np.linalg.norm(vec_dist, axis=-1)

    def find_coupling_pairs(self, max_dx=3, cutoff=None, eps=1.e-10):
        """Automatically find coupling pairs grouped by distances.

        Given the :attr:`unit_cell_positions` and :attr:`basis`, the coupling :attr:`pairs` of
        `nearest_neighbors`, `next_nearest_neighbors` etc at a given distance are basically
        fixed (although not uniquely, since we take out half of them to avoid double-counting
        couplings in both directions ``A_i B_j + B_i A_i``).
        This function iterates through all possible couplings up to a given `cutoff` distance and
        then determines the possible :attr:`pairs` at fixed distances (up to round-off errors).

        Parameters
        ----------
        max_dx : int
            Maximal index for each index of `dx` to iterate over. You need large enough values
            to include every possible coupling up to the desired distance, but choosing it too
            large might make this function run for a long time.
        cutoff : float
            Maximal distance (in the units in which :attr:`basis` and :attr:`unit_cell_positions`
            is given).
        eps : float
            Tolerance up to which to distances are considered the same.

        Returns
        -------
        coupling_pairs : dict
            Keys are distances of nearest-neighbors, next-nearest-neighbors etc.
            Values are ``[(u1, u2, dx), ...]`` as in :attr:`pairs`.
        """
        if cutoff is None:
            cutoff = max_dx - eps
        assert cutoff < max_dx * min(np.linalg.norm(self.basis, axis=-1))
        Lu = len(self.unit_cell)
        dist_pairs = {}
        for u1, u2 in itertools.product(range(Lu), repeat=2):
            for dx in itertools.product(range(max_dx, -max_dx - 1, -1), repeat=self.dim):
                dist = self.distance(u1, u2, np.array(dx))
                if dist > cutoff or dist < eps:
                    continue
                d0 = get_close(dist_pairs.keys(), dist)
                if d0 is None:
                    dist_pairs[dist] = []
                    d0 = dist
                pairs = dist_pairs[d0]
                if (u2, u1, tuple(-i for i in dx)) in pairs:
                    continue  # avoid double-counting for existing pairs
                pairs.append((u1, u2, dx))
        # finally sort the keys of the dict by distances
        result = {}
        for key in sorted(dist_pairs.keys()):
            result[key] = [(u1, u2, np.array(dx)) for u1, u2, dx in dist_pairs[key]]
        return result

    def coupling_shape(self, dx):
        """Calculate correct shape of the `strengths` for a coupling.

        Parameters
        ----------
        dx : tuple of int
            Translation vector in the lattice for a coupling of two operators.
            Corresponds to `dx` argument of
            :meth:`tenpy.models.model.CouplingModel.add_multi_coupling`.

        Returns
        -------
        coupling_shape : tuple of int
            Len :attr:`dim`. The correct shape for an array specifying the coupling strength.
            `lat_indices` has only rows within this shape.
        shift_lat_indices : array
            Translation vector from origin to the lower left corner of box spanned by `dx`.
        """
        shape = [La - abs(dxa) * int(bca) for La, dxa, bca in zip(self.Ls, dx, self.bc)]
        shift_strength = [min(0, dxa) for dxa in dx]
        return tuple(shape), np.array(shift_strength)

    def possible_couplings(self, u1, u2, dx, strength=None):
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
        strength : array_like | None
            If given, instead of returning `lat_indices` and `coupling_shape`
            directly return the correct `strength_12`.

        Returns
        -------
        mps1, mps2 : 1D array
            For each possible two-site coupling the MPS indices for the `u1` and `u2`.
        strength_vals : 1D array
            (Only returend if `strength` is not None.)
            Such that ``for (i, j, s) in zip(mps1, mps2, strength_vals):`` iterates over all
            possible couplings with `s` being the strength of that coupling.
            Couplings where ``strength_vals == 0.`` are filtered out.
        lat_indices : 2D int array
            (Only returend if `strength` is None.)
            Rows of `lat_indices` correspond to entries of `mps1` and `mps2` and contain the
            lattice indices of the "lower left corner" of the box containing the coupling.
        coupling_shape : tuple of int
            (Only returend if `strength` is None.)
            Len :attr:`dim`. The correct shape for an array specifying the coupling strength.
            `lat_indices` has only rows within this shape.
        """
        coupling_shape, shift_lat_indices = self.coupling_shape(dx)
        if any([s == 0 for s in coupling_shape]):
            if strength is None:
                return [], [], np.zeros([0, self.dim]), coupling_shape
            else:
                return [], [], np.array([])
        Ls = np.array(self.Ls)
        mps_i, lat_i = self.mps_lat_idx_fix_u(u1)
        lat_j_shifted = lat_i + dx
        lat_j = np.mod(lat_j_shifted, Ls)  # assuming PBC
        if self.bc_shift is not None:
            shift = np.sum(((lat_j_shifted - lat_j) // Ls)[:, 1:] * self.bc_shift, axis=1)
            lat_j_shifted[:, 0] -= shift
            lat_j[:, 0] = np.mod(lat_j_shifted[:, 0], Ls[0])
        keep = self._keep_possible_couplings(lat_j, lat_j_shifted, u2)
        mps_i = mps_i[keep]
        lat_indices = lat_i[keep] + shift_lat_indices[np.newaxis, :]
        lat_indices = np.mod(lat_indices, coupling_shape)
        lat_j = lat_j[keep]
        lat_j_shifted = lat_j_shifted[keep]
        mps_j = self.lat2mps_idx(np.concatenate([lat_j, [[u2]] * len(lat_j)], axis=1))
        if self.bc_MPS == 'infinite':
            # shift j by whole MPS unit cells for couplings along the infinite direction
            # N_sites_per_ring might not be set for IrregularLattice
            mps_j_shift = (lat_j_shifted[:, 0] - lat_j[:, 0]) * self.N_sites // self.N_rings
            mps_j += mps_j_shift
            # finally, ensure 0 <= min(i, j) < N_sites.
            # (so far, 0 <= mps_i < N_sites)
            mps_ij_shift = np.where(mps_j_shift < 0, -mps_j_shift, 0)
            mps_i += mps_ij_shift
            mps_j += mps_ij_shift
        if strength is None:
            return mps_i, mps_j, lat_indices, coupling_shape
        else:
            strength = to_array(strength, coupling_shape)  # tile to correct shape
            strength_vals = strength[tuple(lat_indices.T)]
            keep_nonzero = (strength_vals != 0.)  # filter out couplings with strength 0
            return mps_i[keep_nonzero], mps_j[keep_nonzero], strength_vals[keep_nonzero]

    def _keep_possible_couplings(self, lat_j, lat_j_shifted, u2):
        """filter possible j sites of a coupling from :meth:`possible_couplings`"""
        return np.all(
            np.logical_or(
                lat_j_shifted == lat_j,  # not accross the boundary
                np.logical_not(self.bc)),  # direction has PBC
            axis=1)

    def multi_coupling_shape(self, dx):
        """Calculate correct shape of the `strengths` for a multi_coupling.

        Parameters
        ----------
        dx : 2D array, shape (N_ops, :attr:`dim`)
            ``dx[i, :]`` is the translation vector in the lattice for the `i`-th operator.
            Corresponds to the `dx` of each operator given in the argument `ops` of
            :meth:`tenpy.models.model.CouplingModel.add_multi_coupling`.

        Returns
        -------
        coupling_shape : tuple of int
            Len :attr:`dim`. The correct shape for an array specifying the coupling strength.
            `lat_indices` has only rows within this shape.
        shift_lat_indices : array
            Translation vector from origin to the lower left corner of box spanned by `dx`.
            (Unlike for :meth:`coupling_shape` it can also contain entries > 0)
        """
        # coupling_shape(dx) is equivalent to
        # multi_coupling_shape(np.array([[0]*self.dim, dx]))
        Ls = self.Ls
        shape = [None] * len(Ls)
        shift_strength = [None] * len(Ls)
        for a in range(len(Ls)):
            max_dx, min_dx = np.max(dx[:, a]), np.min(dx[:, a])
            box_dx = max_dx - min_dx
            shape[a] = Ls[a] - box_dx * int(self.bc[a])
            shift_strength[a] = min_dx  # note: can be positive!
        return tuple(shape), np.array(shift_strength)

    def possible_multi_couplings(self, ops, strength=None):
        """Generalization of :meth:`possible_couplings` to couplings with more than 2 sites.

        Parameters
        ----------
        ops : list of ``(opname, dx, u)``
            Same as the argument `ops` of
            :meth:`~tenpy.models.model.CouplingModel.add_multi_coupling`.

        Returns
        -------
        mps_ijkl : 2D int array
            Each row contains MPS indices `i,j,k,l,...`` for each of the operators positions.
            The positions are defined by `dx` (j,k,l,... relative to `i`) and boundary coundary
            conditions of `self` (how much the `box` for given `dx` can be shifted around without
            hitting a boundary - these are the different rows).
        strength_vals : 1D array
            (Only returend if `strength` is not None.)
            Such that ``for  (ijkl, s) in zip(mps_ijkl, strength_vals):`` iterates over all
            possible couplings with `s` being the strength of that coupling.
            Couplings where ``strength_vals == 0.`` are filtered out.
        lat_indices : 2D int array
            (Only returend if `strength` is None.)
            Rows of `lat_indices` correspond to rows of `mps_ijkl` and contain the lattice indices
            of the "lower left corner" of the box containing the coupling.
        coupling_shape : tuple of int
            (Only returend if `strength` is None.)
            Len :attr:`dim`. The correct shape for an array specifying the coupling strength.
            `lat_indices` has only rows within this shape.
        """
        D = self.dim
        Nops = len(ops)
        Ls = np.array(self.Ls)
        # make 3D arrays ["iteration over lattice", "operator", "spatial direction"]
        # recall numpy broadcasing: 1D equivalent to [np.newaxis, np.newaxis, :]
        dx = np.array([op_dx for _, op_dx, op_u in ops], dtype=np.int_).reshape([1, Nops, D])
        u = np.array([op_u for _, op_dx, op_u in ops], dtype=np.int_).reshape([1, Nops, 1])
        coupling_shape, shift_lat_indices = self.multi_coupling_shape(dx[0, :, :])
        if any([s == 0 for s in coupling_shape]):
            if strength is None:
                return [], [], coupling_shape
            else:
                return [], np.array([])
        lat_indices = np.indices(coupling_shape).reshape([1, self.dim, -1]).transpose([2, 0, 1])
        lat_ijkl_shifted = lat_indices + (dx - shift_lat_indices)
        lat_ijkl = np.mod(lat_ijkl_shifted, Ls)
        if self.bc_shift is not None:
            shift = np.sum(((lat_ijkl_shifted - lat_ijkl) // Ls)[:, :, 1:] * self.bc_shift, axis=2)
            lat_ijkl_shifted[:, :, 0] -= shift
            lat_ijkl[:, :, 0] = np.mod(lat_ijkl_shifted[:, :, 0], Ls[0])
        keep = self._keep_possible_multi_couplings(lat_ijkl, lat_ijkl_shifted, u)
        lat_indices = lat_indices[keep, 0, :]  # make 2D as to be returned
        lat_ijkl = lat_ijkl[keep, :, :]
        u = np.broadcast_to(u, lat_ijkl.shape[:2] + (1, ))
        mps_ijkl = self.lat2mps_idx(np.concatenate([lat_ijkl, u], axis=2))
        if self.bc_MPS == 'infinite':
            # shift by whole MPS unit cells for couplings along the infinite direction
            # N_sites_per_ring might not be set for IrregularLattice
            mps_ijkl += ((lat_ijkl_shifted[keep, :, 0] - lat_ijkl[:, :, 0]) * self.N_sites //
                         self.N_rings)
            # but ensure that  0 <= min(i,j,...) < N_sites
            min_ijkl = np.min(mps_ijkl, axis=1)
            mps_ijkl += (np.mod(min_ijkl, self.N_sites) - min_ijkl)[:, np.newaxis]
        if strength is None:
            return mps_ijkl, lat_indices, coupling_shape
        else:
            strength = to_array(strength, coupling_shape)  # tile to correct shape
            strength_vals = strength[tuple(lat_indices.T)]  # extract correct entries
            keep_nonzero = (strength_vals != 0.)  # filter out couplings with strength 0
            return mps_ijkl[keep_nonzero], strength_vals[keep_nonzero]

    def _keep_possible_multi_couplings(self, lat_ijkl, lat_ijkl_shifted, u_ijkl):
        """Filter possible couplings from :meth:`possible_multi_couplings`"""
        return np.all(
            np.logical_or(
                lat_ijkl_shifted == lat_ijkl,  # not accross the boundary
                np.logical_not(self.bc)),  # direction has PBC
            axis=(1, 2))

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

    def plot_coupling(self, ax, coupling=None, wrap=False, **kwargs):
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
        wrap : bool
            If ``True``, plot couplings going around the boundary by directly connecting the sites
            it connects. This might be hard to see, as this puts lines from one end of the lattice
            to the other.
            If ``False``, plot the couplings as dangling lines.
        **kwargs :
            Further keyword arguments given to ``ax.plot()``.
        """
        if coupling is None:
            coupling = self.pairs['nearest_neighbors']
        kwargs.setdefault('color', 'k')
        Ls = np.array(self.Ls)
        for u1, u2, dx in coupling:
            if wrap:
                mps_i, mps_j, _, _ = self.possible_couplings(u1, u2, dx)
                pos1 = self.position(self.mps2lat_idx(mps_i))
                pos2 = self.position(self.mps2lat_idx(mps_j))
            else:
                dx = np.r_[np.array(dx), u2 - u1]  # append the difference in u to dx
                lat_idx_1 = self.order[self._mps_fix_u[u1], :]
                lat_idx_2 = lat_idx_1 + dx[np.newaxis, :]
                lat_idx_2_mod = np.mod(lat_idx_2[:, :-1], Ls)
                keep = self._keep_possible_couplings(lat_idx_2_mod, lat_idx_2[:, :-1], u2)
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

    def plot_basis(self, ax, origin=(0., 0.), shade=None, **kwargs):
        """Plot arrows indicating the basis vectors of the lattice.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            The axes on which we should plot.
        **kwargs :
            Keyword arguments for ``ax.arrow``.
        """
        kwargs.setdefault("length_includes_head", True)
        kwargs.setdefault("width", 0.03)
        kwargs.setdefault("color", 'g')
        origin = np.array(origin)
        basis = np.array([self.basis[i] for i in range(self.dim)])
        if basis.shape[1] == 1:
            basis = basis * np.array([[1., 0]])
            if basis.shape[1] != 2:
                raise ValueError("can only plot in 2 dimensions.")
        if shade is None:
            shade = True if self.dim == 2 else False
        if shade:
            from matplotlib.patches import Polygon
            xy = [origin, origin + basis[0], origin + basis[0] + basis[1], origin + basis[1]]
            ax.add_patch(Polygon(xy, fill=True, color='palegreen'))
        for i in range(self.dim):
            vec = basis[i]
            ax.arrow(origin[0], origin[1], vec[0], vec[1], **kwargs)

    def plot_bc_identified(self, ax, direction=-1, origin=None, cylinder_axis=False, **kwargs):
        """Mark two sites indified by periodic boundary conditions.

        Works only for lattice with a 2-dimensional basis.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            The axes on which we should plot.
        direction : int
            The direction of the lattice along which we should mark the idenitified sites.
            If ``None``, mark it along all directions with periodic boundary conditions.
        cylinder_axis : bool
            Whether to plot the cylinder axis as well.
        origin : None | np.ndarray
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
        if origin is None:
            origin = self.unit_cell_positions[0]
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 10)
        kwargs.setdefault("color", "orange")
        x_y = []
        for i in dirs:
            if self.bc[i]:
                raise ValueError("Boundary conditons are not periodic for given direction")
            x_y.append(origin)
            x_y.append(origin + self.Ls[i] * self.basis[i])
            if self.bc_shift is not None and i > 0:
                x_y[-1] = x_y[-1] + self.bc_shift[i - 1] * self.basis[0]
        x_y = np.array(x_y)
        if x_y.shape[1] != 2:
            raise ValueError("can only plot in 2D")
        ax.plot(x_y[:, 0], x_y[:, 1], **kwargs)
        if cylinder_axis:
            if len(x_y) != 2 or self.dim != 2:
                raise ValueError("can't plot cylinder axis for multiple directions")
            center = np.mean(x_y, axis=0)
            diff = x_y[1, :] - x_y[0]
            perp = np.array([diff[1], -diff[0]])
            x_y_cyl = np.array([center - perp, center + perp])
            kwargs.setdefault('linestyle', '--')
            kwargs['marker'] = None
            ax.plot(x_y_cyl[:, 0], x_y_cyl[:, 1], **kwargs)

    def _asvalid_latidx(self, lat_idx):
        """convert lat_idx to an ndarray with correct last dimension."""
        lat_idx = np.asarray(lat_idx, dtype=np.intp)
        if lat_idx.shape[-1] != len(self.shape):
            raise ValueError("wrong len of last dimension of lat_idx: " + str(lat_idx.shape))
        return lat_idx

    def _set_Ls(self, Ls):
        self.Ls = tuple([int(L) for L in Ls])
        self.N_cells = int(np.prod(self.Ls))
        self.shape = self.Ls + (len(self.unit_cell), )
        self.N_sites = int(np.prod(self.shape))
        self.N_rings = self.Ls[0]
        self.N_sites_per_ring = int(self.N_sites // self.N_rings)
        strides = [1]
        for L in self.Ls:
            strides.append(strides[-1] * L)
        self._strides = np.array(strides, np.intp)

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

    This is usefull if you need a valid :class:`Lattice` with given :meth:`mps_sites`
    and don't care about the actual geometry, e.g, because you don't intend to use the
    :class:`~tenpy.models.model.CouplingModel`.

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

    .. note ::
        The lattice defines only the geometry of the sites, not the couplings;
        you can have position-dependent couplings/onsite terms despite having a regular lattice.

    By adjusting the :attr:`order` and a few private attributes and methods, we can
    make functions like :meth:`possible_couplings` work with a more "irregular" lattice structure,
    where some of the sites are missing and other sites added instead.

    Parameters
    ----------
    regular_lattice : :class:`Lattice`
        The lattice this is based on.
    remove : 2D array | None
        Each row is a lattice index ``(x_0, ..., x_{dim-1}, u)`` of a site to be removed.
        If ``None``, don't remove something.
    add : Tuple[2D array, list] | None
        Each row of the 2D array is a lattice index ``(x_0, ..., x_{dim-1}, u)`` specifiying
        where a site is to be added; `u` is the index of the site within the final
        :attr:`unit_cell` of the irregular lattice.
        For each row of the 2D array, there is one entry in the list specifying where the site
        is inserted in the MPS; the values are compared to the MPS indices of the *regular* lattice
        and sorted into it, so "2.5" goes between what was site 2 and 3 in the regular lattice.
        An entry `None` indicates that the site should be inserted after the lattice site
        ``(x_0, ..., x_{dim-1}, -1)`` of the `regular_lattice`.
    add_unit_cell : list of :class:`~tenpy.networks.site.Site`
        Extra sites to be added to the unit cell.
    add_positions : iterable of 1D arrays
        For each extra site in `add_unit_cell` the position within the unit cell.
        Defaults to ``np.zeros((len(add_unit_cell), dim))``.

    Attributes
    ----------
    regular_lattice : :class:`Lattice`
        The lattice this is based on.
    remove, add : 2D array | None
        See above.  Used in :meth:`ordering` only.

    Examples
    --------

    .. testsetup :: IrregularLattice

        from tenpy.models.lattice import *

    Let's imagine that we have two different sites; for concreteness we can thing of a
    fermion site, which we represent with ``'F'``, and a spin site ``'S'``.
    If you want to preserve charges, take a look at
    :func:`~tenpy.networks.site.set_common_charges` for the proper way to initialize the sites.


    You could now imagine that to have fermion chain with spins on the "bonds".
    In the periodic/infinite case, you would simply define

    .. doctest :: IrregularLattice

        >>> lat = Lattice([2], unit_cell=['F', 'S'], bc='periodic', bc_MPS='infinite')
        >>> lat.mps_sites()
        ['F', 'S', 'F', 'S']

    For a finite system, you don't want to terminate with a spin on the right, so you need to
    remove the very last site by specifying the lattice index ``[L-1, 1]`` of that site:

    .. doctest :: IrregularLattice

        >>> L = 4
        >>> reg_lat = Lattice([L], unit_cell=['F', 'S'], bc='open', bc_MPS='finite')
        >>> irr_lat = IrregularLattice(reg_lat, remove=[[L - 1, 1]])
        >>> irr_lat.mps_sites()
        ['F', 'S', 'F', 'S', 'F', 'S', 'F']

    Another simple example would be to add a spin in the center of a fermion chain.
    In that case, we add another site to the unit cell and specify the lattice index as
    ``[(L-1)//2, 1]`` (where the 1 is the index of ``'S'`` in the unit cell ``['F', 'S']`` of the
    irregular lattice).
    The `None` for the MPS index is equivalent to ``(L-1)/2`` in this case.

    .. doctest :: IrregularLattice

        >>> reg_lat = Lattice([L], unit_cell=['F'])
        >>> irr_lat = IrregularLattice(reg_lat, add=([[(L - 1)//2, 1]], [None]),
        ...                            add_unit_cell=['S'])
        >>> irr_lat.mps_sites()
        ['F', 'F', 'S', 'F', 'F']

    """
    _REMOVED = -123456  # value in self._perm indicating removed sites.

    def __init__(self,
                 regular_lattice,
                 remove=None,
                 add=None,
                 add_unit_cell=[],
                 add_positions=None):
        if add_positions is None:
            add_positions = np.zeros((len(add_unit_cell), regular_lattice.dim))
        elif len(add_unit_cell) != len(add_positions):
            raise ValueError("length of add_unit_cell and add_positions need to be the same")
        self.regular_lattice = regular_lattice
        if remove is not None:
            remove = np.array(remove, np.intp)
        self.remove = remove
        self.add = add
        unit_cell = list(regular_lattice.unit_cell) + list(add_unit_cell)
        positions = list(regular_lattice.unit_cell_positions) + list(add_positions)
        Lattice.__init__(
            self,
            regular_lattice.Ls,
            unit_cell,
            bc=regular_lattice.boundary_conditions,
            bc_MPS=regular_lattice.bc_MPS,
            basis=regular_lattice.basis,
            positions=positions,
            pairs=regular_lattice.pairs,
        )
        self.order = self._ordering_irreg(regular_lattice.order)
        # done

    def save_hdf5(self, hdf5_saver, h5gr, subpath):
        super().save_hdf5(hdf5_saver, h5gr, subpath)
        hdf5_saver.save(self.regular_lattice, subpath + "regular_lattice")
        hdf5_saver.save(self.remove, subpath + "remove")
        hdf5_saver.save(self.add[0], subpath + "add_lat_idx")
        hdf5_saver.save(self.add[1], subpath + "add_mps_idx")
        add_unit_cell = self.unit_cell[len(self.regular_lattice.unit_cell):]
        add_positions = self.unit_cell_positions[len(self.regular_lattice.unit_cell_positions):]
        hdf5_saver.save(add_unit_cell, subpath + "add_unit_cell")
        hdf5_saver.save(add_positions, subpath + "add_positions")

    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):
        obj = super().from_hdf5(hdf5_loader, h5gr, subpath)
        obj.regular_lattice = hdf5_loader.load(subpath + "regular_lattice")
        lat_idx = hdf5_loader.load(subpath + "add_lat_idx")
        mps_idx = hdf5_loader.load(subpath + "add_mps_idx")
        obj.add = (lat_idx, mps_idx)
        obj.remove = hdf5_loader.load(subpath + "remove")
        return obj

    def ordering(self, order):
        """Provide possible orderings of the lattice sites.

        Parameters
        ----------
        order :
            Argument for the :meth:`Lattice.ordering` of the :attr:`regular_lattice`, or
            2D ndarray providing the order of the *regular lattice*.

        Returns
        -------
        order : array, shape (N, D+1)
            The order to be used for :attr:`order`, i.e. `order` with added/removed sites
            as specified by :attr:`remove` and :attr:`add`.
        """
        order_reg = self.regular_lattice.ordering(order)
        return self._ordering_irreg(order_reg)

    def _ordering_irreg(self, order):
        """Remove and add irregular sites to the order of the regular lattice."""
        mps_reg = np.arange(len(order))
        if self.remove is not None:
            self._perm = np.lexsort(order.T)  # allow to temporarily use lat2mps_idx for lattice
            # indices with u from regular lattice
            keep = np.ones([len(order)], np.bool_)
            keep[self.lat2mps_idx(self.remove)] = False
            order = order[keep]
            mps_reg = mps_reg[keep]
        if self.add is not None:
            # sort such that MPS indices are ascending
            lat_idx, mps_add = self.add
            mps_add = list(mps_add)
            for i in range(len(mps_add)):
                if mps_add[i] is None:
                    close_to = np.array(lat_idx[i])
                    close_to[-1] = len(self.regular_lattice.unit_cell) - 1
                    mps_add[i] = self.regular_lattice.lat2mps_idx(close_to)
            mps_add = np.array(mps_add)
            sort = np.argsort(np.concatenate((mps_reg, mps_add)), kind="stable")
            order = np.concatenate((order, np.array(lat_idx)), axis=0)
            order = order[sort, :]
        return order

    @Lattice.order.setter
    def order(self, order_):
        # very similar to HelicalLattice.order setter
        self._order = np.array(order_, dtype=np.intp)

        # this defines `self._perm`
        perm = np.full([np.prod(self.shape)], self._REMOVED)
        perm[np.sum(self._order * self._strides, axis=1)] = np.arange(len(order_))
        self._perm = perm

        # and the other stuff which is cached
        # versions for fixed u
        self._mps_fix_u = []
        for u in range(len(self.unit_cell)):
            mps_fix_u = np.nonzero(order_[:, -1] == u)[0]
            self._mps_fix_u.append(mps_fix_u)
        self._mps_fix_u = tuple(self._mps_fix_u)
        self.N_sites = len(order_)
        _, counts = np.unique(order_[:, 0], return_counts=True)
        if np.all(counts == counts[0]):
            self.N_sites_per_ring = counts[0]
        else:
            self.N_sites_per_ring = None

    # mps2lat_idx and lat2mps_idx work thanks to the way _perm is defined,
    # mps2lat_values, mps2lat_values_masked work as well

    def mps_idx_fix_u(self, u=None):
        if u is not None:
            return self._mps_fix_u[u]
        return self._perm[self._perm != self._REMOVED]

    # make possible_couplings and possible_multi_couplings work

    def _keep_possible_couplings(self, lat_j, lat_j_shifted, u2):
        """filter possible j sites of a coupling from :meth:`possible_couplings`"""
        keep = super()._keep_possible_couplings(lat_j, lat_j_shifted, u2)
        lat_j_u = np.concatenate([lat_j, np.full([len(lat_j), 1], u2)], axis=1)
        i = np.sum(lat_j_u * self._strides, axis=-1)
        i = np.take(self._perm, i, 0)
        return np.logical_and(keep, i != self._REMOVED)

    def _keep_possible_multi_couplings(self, lat_ijkl, lat_ijkl_shifted, u_ijkl):
        """filter possible j sites of a coupling from :meth:`possible_couplings`"""
        keep = super()._keep_possible_multi_couplings(lat_ijkl, lat_ijkl_shifted, u_ijkl)
        u_ijkl = np.broadcast_to(u_ijkl, lat_ijkl.shape[:2] + (1, ))
        i = np.sum(np.concatenate([lat_ijkl, u_ijkl], axis=2) * self._strides, axis=-1)
        i = np.take(self._perm, i, 0)
        return np.logical_and(keep, np.all(i != self._REMOVED, axis=1))

    def _set_Ls(self, Ls):
        super()._set_Ls(Ls)
        # N_sites, N_sites_per_ring set by order setter
        # self.N_sites = None
        self.N_sites_per_ring = None


class HelicalLattice(Lattice):
    """Translation invariant version of a tilted, regular 2D lattice.

    A 2D lattice on an infinite cylinder becomes translation invariant by a single *lattice* unit
    cell if we tilt/shift the boundary conditions around the cylinder such that the unit cell
    at ``(x, y=Ly-1)`` is neighbored by ``(x+1, y=0)``, and the MPS winds as a helix
    around the cylinder.
    Let's illustrate this for the Square lattice with a single-site unit cell - for a multi-site
    unit cell, imagine it being inserted at each of the sites of a Square lattice.

    .. plot ::

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(7, 5))
        ax = fig.gca(projection='3d')
        Lx, Ly, r = 6, 6, 1.
        x = np.arange(0., Lx - 0.001, 1./Ly)
        theta = 2*np.pi* x - np.pi/6.
        y = r * np.sin(theta)
        z = r * np.cos(theta)
        ax.plot(x, y, z, 'k-', label='neighbors')
        ax.plot(x, y, z, 'bo', label='sites')
        N = Lx*Ly //2
        for j in range(Ly):
            j2 = j + (Lx-1)*Ly
            ax.plot([x[j], x[j2]], [y[j], y[j2]], [z[j], z[j2]], 'k-')
        ax.plot(x[:N], y[:N], z[:N], 'r--', linewidth=3., label='MPS')
        ax.legend()
        ax.view_init(elev=30, azim=-77)  # adjust view-point

    .. warning ::
        Some assumptions of the regular lattice like "the number of the sites in the MPS unit
        cell is ``product(lat.shape)``" no longer hold for this model!
        Be very careful e.g. for getting the units of the
        :meth:`~tenpy.networks.MPS.correlation_length` right.

    Parameters
    ----------
    N_unit_cells : int
        Number of *lattice* unit cells to include into the MPS unit cell.
        The total number of sites will be ``N_unit_cells * len(regular_lattice.unit_cell)``.

    """
    _REMOVED = IrregularLattice._REMOVED

    def __init__(self, regular_lattice, N_unit_cells):
        assert not isinstance(regular_lattice, HelicalLattice)
        if isinstance(regular_lattice, IrregularLattice):
            raise ValueError("regular_lattice can't be irregular: we want translation invariance!")
        self.regular_lattice = regular_lattice
        if regular_lattice.dim != 2:
            raise ValueError("Works only for 2D lattices")
        if regular_lattice.bc_shift is None or tuple(regular_lattice.bc_shift) != (-1, ):
            raise ValueError("To keep the coding simpler, we require that you initialize the "
                             "regular lattice with the shifted `bc=['periodic', -1]`")
        if regular_lattice.bc_MPS != 'infinite':
            raise ValueError("Require `bc_MPS='infinite'` for the regular lattice. "
                             "For finite systems, just take a regular lattice!")
        assert regular_lattice.bc[1] == bc_choices['periodic']  # require cylinder
        if N_unit_cells > regular_lattice.N_cells:
            raise ValueError("N_unit_cells larger than regular_lattice.N_cells: "
                             "increase Lx of regular_lattice!")
        if regular_lattice.N_cells % N_unit_cells != 0:
            raise ValueError("N_unit_cells incommensurate with regular_lattice.N_cells: "
                             "increase Lx of regular_lattice!")
        self._N_cells = N_unit_cells
        Lattice.__init__(
            self,
            regular_lattice.Ls,
            regular_lattice.unit_cell,
            order='Cstyle',  # temporary
            bc=regular_lattice.boundary_conditions,
            bc_MPS=regular_lattice.bc_MPS,
            basis=regular_lattice.basis,
            positions=regular_lattice.unit_cell_positions,
            pairs=regular_lattice.pairs,
        )
        self.order = self._ordering_helical(regular_lattice.order)
        # done

    def save_hdf5(self, hdf5_saver, h5gr, subpath):
        super().save_hdf5(hdf5_saver, h5gr, subpath)
        hdf5_saver.save(self.regular_lattice, subpath + "regular_lattice")
        h5gr.attrs["N_unit_cells"] = self.N_sites

    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):
        obj = super().from_hdf5(hdf5_loader, h5gr, subpath)
        obj._N_cells = hdf5_loader.get_attr(h5gr, "N_unit_cells")
        return obj

    def ordering(self, order):
        """Provide possible orderings of the lattice sites.

        Parameters
        ----------
        order :
            Argument for the :meth:`Lattice.ordering` of the :attr:`regular_lattice`, or
            2D ndarray providing the order of the *regular lattice*.
            Note that really the only freedom is the order of the sites in the :attr:`unit_cell`.

        Returns
        -------
        order : array, shape (N, D+1)
            The order to be used for :attr:`order`, i.e. `order` with added/removed sites
            as specified by :attr:`remove` and :attr:`add`.
        """
        order_reg = self.regular_lattice.ordering(order)
        return self._ordering_helical(order_reg)

    def _ordering_helical(self, order):
        """extract relevant sites from the `order` of the full 2D lattice."""
        Lx, Ly, Lu = self.regular_lattice.shape
        N_sites = self._N_cells * Lu
        errmsg = ("order of regular lattice incompatible with tilting. "
                  "Must be C-style up to an overall permutation inside the unit cell.")
        assert np.all(order[:Lu, :-1] == 0), errmsg
        order_within_unit_cell = order[:Lu, -1]
        assert np.all(order[:, -1] == np.tile(order_within_unit_cell, [Lx * Ly])), errmsg
        cstyle_xy = get_order([Lx, Ly], [False, False])
        assert np.all(order[:, :-1] == np.repeat(cstyle_xy, Lu, axis=0))
        return order[:N_sites, :]

    @Lattice.order.setter
    def order(self, order_):
        # very similar to IrregularLattice.order setter
        self._order = np.array(order_, dtype=np.intp)
        assert len(order_) == len(self.unit_cell) * self._N_cells

        # this defines `self._perm`
        perm = np.full([np.prod(self.shape)], self._REMOVED)
        perm[np.sum(self._order * self._strides, axis=1)] = np.arange(len(order_))
        self._perm = perm

        # and the other stuff which is cached
        # versions for fixed u
        self._mps_fix_u = []
        for u in range(len(self.unit_cell)):
            mps_fix_u = np.nonzero(order_[:, -1] == u)[0]
            self._mps_fix_u.append(mps_fix_u)
        self._mps_fix_u = tuple(self._mps_fix_u)

    # the regular lattice has the same order for the MPS,
    # only the division into unit cells is different
    # hence we can just use the versions of the regular lattice.

    def mps2lat_idx(self, i):
        # doc: see Lattice
        return self.regular_lattice.mps2lat_idx(i)

    def lat2mps_idx(self, lat_idx):
        # doc: see Lattice
        return self.regular_lattice.lat2mps_idx(lat_idx)

    def mps2lat_values(self, *args, **kwargs):
        """Not implemented, use :meth:`mps2lat_values_masked` instead."""
        raise NotImplementedError("Use mps2lat_values_masked instead")

    def mps2lat_values_masked(self, *args, **kwargs):
        # doc: see Lattice
        return self.regular_lattice.mps2lat_values_masked(*args, **kwargs)

    def enlarge_mps_unit_cell(self, factor=2):
        # doc: see Lattice
        if (self._N_cells * factor > self.regular_lattice.N_cells
                or self.regular_lattice.N_cells % (self._N_cells * factor) != 0):
            self.regular_lattice.enlarge_mps_unit_cell(factor)
        self._N_cells = factor * self._N_cells

        self._set_Ls(self.regular_lattice.Ls)
        order_reg = self.regular_lattice.order
        self.order = self._ordering_helical(order_reg)  # use property setter
        self.test_sanity()


    # strategy for possible_[multi_]couplings:
    # since everything is translation invariant along the MPS, we can just extract it
    # from the couplings of the larger lattice
    # by restricting to 0 <= min(i,j,...) < self.N_sites
    # instead of the `0 <= min(i,j,...) < self.regular_lattice.N_sites`
    # guaranteed by self.possible_[multi_]couplings

    def possible_couplings(self, u1, u2, dx, strength=None):
        reg = self.regular_lattice
        if strength is None:
            mps_i, mps_j, lat_ind, coupl_sh = reg.possible_couplings(u1, u2, dx)
            keep = (np.min([mps_i, mps_j], axis=0) < self.N_sites)
            return mps_i[keep], mps_j[keep], lat_ind[keep], coupl_sh
        else:
            mps_i, mps_j, strength_vals = reg.possible_couplings(u1, u2, dx, strength)
            # we can actually check that everything is translation invariant!
            self._check_transl_invar_strength(np.stack([mps_i, mps_j]).T, strength_vals)
            keep = (np.min([mps_i, mps_j], axis=0) < self.N_sites)
            return mps_i[keep], mps_j[keep], strength_vals[keep]

    def possible_multi_couplings(self, ops, strength=None):
        reg = self.regular_lattice
        if strength is None:
            mps_ijkl, lat_inds, coupl_shape = reg.possible_multi_couplings(ops)
            keep = np.min(mps_ijkl, axis=1) < self.N_sites
            return mps_ijkl[keep, :], lat_inds[keep, :], coupl_shape
        else:
            mps_ijkl, strength_vals = reg.possible_multi_couplings(ops, strength)
            # we can actually check that everything is translation invariant!
            self._check_transl_invar_strength(mps_ijkl, strength_vals)
            keep = (np.min(mps_ijkl, axis=1) < self.N_sites)
            return mps_ijkl[keep, :], strength_vals[keep]

    def _check_transl_invar_strength(self, mps_ijkl, strength_vals):
        sort = np.lexsort(mps_ijkl.T)
        mps_ijkl = mps_ijkl[sort]
        strength_vals = strength_vals[sort]
        min_ijkl = np.min(mps_ijkl, axis=1)
        for cell_start in range(0, self.regular_lattice.N_sites, self.N_sites):
            keep_cell = np.logical_and(cell_start <= min_ijkl,
                                       min_ijkl < cell_start + self.N_sites)
            if cell_start == 0:
                ijkl_compare = mps_ijkl[keep_cell]
                strength_compare = strength_vals[keep_cell]
            else:
                assert np.all(mps_ijkl[keep_cell] - cell_start == ijkl_compare)
                if not np.all(np.abs(strength_vals[keep_cell] - strength_compare) < 1e-10):
                    raise ValueError("Not translation invariant: can't use HelicalLattice")

    # most plot_* functions work, except:

    def plot_coupling(self, ax, coupling=None, wrap=True, **kwargs):
        if not wrap:
            raise NotImplementedError("wrap=False not implemented for the HelicalLattice")
        super().plot_coupling(ax, coupling, wrap, **kwargs)

    def _set_Ls(self, Ls):
        super()._set_Ls(Ls)
        self.N_cells = self._N_cells
        self.N_sites = len(self.unit_cell) * self._N_cells
        self.N_sites_per_ring = None  # shouldn't be used
        self.N_rings = None  # shouldn't be used - pointless for this case.


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
    Lu = 1  #: the (expected) number of sites in the unit cell, ``len(unit_cell)``.

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

    .. plot ::

        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 1.4))
        ax = plt.gca()
        lat = lattice.Chain(4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=(-0.5, -0.25), shade=False)
        ax.set_xlim(-1.)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect('equal')
        plt.show()

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
    dim = 1  #: the dimension of the lattice

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

    .. plot ::

        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 1.4))
        ax = plt.gca()
        lat = lattice.Ladder(4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=[-0.5, -0.25], shade=False)
        ax.set_aspect('equal')
        ax.set_xlim(-1.)
        ax.set_ylim(-1.)
        plt.show()

    Parameters
    ----------
    L : int
        The length of each chain, we have 2*L sites in total.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both chains.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    Lu = 2  #: the (expected) number of sites in the unit cell, ``len(unit_cell)``.
    dim = 1  #: the dimension of the lattice

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

    .. plot ::

        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 5))
        ax = plt.gca()
        lat = lattice.Square(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
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
        The length in each direction.
    site : :class:`~tenpy.networks.site.Site`
        The local lattice site. The `unit_cell` of the :class:`Lattice` is just ``[site]``.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `pairs` are set accordingly.
        If `order` is specified in the form ``('standard', snake_winding, priority)``,
        the `snake_winding` and `priority` should only be specified for the spatial directions.
        Similarly, `positions` can be specified as a single vector.
    """
    dim = 2  #: the dimension of the lattice

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

    .. plot ::

        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(4, 5))
        ax = plt.gca()
        lat = lattice.Triangular(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.5*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        plt.show()

    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    site : :class:`~tenpy.networks.site.Site`
        The local lattice site. The `unit_cell` of the :class:`Lattice` is just ``[site]``.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `pairs` are set accordingly.
        If `order` is specified in the form ``('standard', snake_windingi, priority)``,
        the `snake_winding` and `priority` should only be specified for the spatial directions.
        Similarly, `positions` can be specified as a single vector.
    """
    dim = 2  #: the dimension of the lattice

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

    .. plot ::

        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 6))
        ax = plt.gca()
        lat = lattice.Honeycomb(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
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
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
        For the Honeycomb lattice ``'fourth_nearest_neighbors', 'fifth_nearest_neighbors'``
        are set in :attr:`pairs`.
    """
    dim = 2  #: the dimension of the lattice
    Lu = 2  #: the (expected) number of sites in the unit cell, ``len(unit_cell)``.

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
        NN5 = [(0, 0, np.array([1, 1])), (0, 0, np.array([2, -1])), (0, 0, np.array([-1, 2])),
               (1, 1, np.array([1, 1])), (1, 1, np.array([2, -1])), (1, 1, np.array([-1, 2]))]
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

        .. plot ::

            import matplotlib.pyplot as plt
            from tenpy.models import lattice
            fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(5, 6))
            orders = ['default', 'snake']
            lat = lattice.Honeycomb(4, 3, None, bc='periodic')
            for order, ax in zip(orders, axes.flatten()):
                lat.order = lat.ordering(order)
                lat.plot_order(ax, linestyle=':')
                lat.plot_sites(ax)
                lat.plot_basis(ax, origin=-0.25*(lat.basis[0] + lat.basis[1]))
                ax.set_title(repr(order))
                ax.set_aspect('equal')
                ax.set_xlim(-1)
                ax.set_ylim(-1)
            plt.show()

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

    .. plot ::

        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 4))
        ax = plt.gca()
        lat = lattice.Kagome(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.25*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()

    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 2  #: the dimension of the lattice
    Lu = 3  #: the (expected) number of sites in the unit cell, ``len(unit_cell)``.

    def __init__(self, Lx, Ly, sites, **kwargs):
        sites = _parse_sites(sites, 3)
        #   \   /
        #    \ /
        #     2
        #    / \
        #   /   \
        #  0-----1-----
        pos = np.array([[0, 0], [1, 0], [0.5, 0.5 * 3**0.5]])
        basis = [2 * pos[1], 2 * pos[2]]
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        NN = [(0, 1, np.array([0, 0])), (0, 2, np.array([0, 0])), (1, 2, np.array([0, 0])),
              (1, 0, np.array([1, 0])), (2, 0, np.array([0, 1])), (2, 1, np.array([-1, 1]))]
        nNN = [(0, 1, np.array([0, -1])), (0, 2, np.array([1, -1])), (1, 0, np.array([1, -1])),
               (1, 2, np.array([1, 0])), (2, 0, np.array([1, 0])), (2, 1, np.array([0, 1]))]
        nnNN = [(0, 0, np.array([1, -1])), (0, 0, np.array([0, 1])), (0, 0, np.array([1, 0])),
                (1, 1, np.array([1, -1])), (1, 1, np.array([0, 1])), (1, 1, np.array([1, 0])),
                (2, 2, np.array([1, -1])), (2, 2, np.array([0, 1])), (2, 2, np.array([1, 0]))]
        kwargs.setdefault('pairs', {})
        kwargs['pairs'].setdefault('nearest_neighbors', NN)
        kwargs['pairs'].setdefault('next_nearest_neighbors', nNN)
        kwargs['pairs'].setdefault('next_next_nearest_neighbors', nnNN)
        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


def get_lattice(lattice_name):
    """Given the name of a :class:`Lattice` class, get the lattice class itself.

    Parameters
    ----------
    lattice_name : str | type
        Name of a :class:`Lattice` class defined in the module :mod:`~tenpy.models.lattice`,
        for example ``"Chain", "Square", "Honeycomb", ...``.
        Alternatively, instead of the name directly the class itself can be given.

    Returns
    -------
    LatticeClass : :class:`Lattice`
        The lattice class (type, not instance) specified by `lattice_name`.
    """
    return find_subclass(Lattice, lattice_name)


def get_order(shape, snake_winding, priority=None):
    """Built the :attr:`Lattice.order` in (Snake-) C-Style for a given lattice shape.

    .. note ::
        In this doc-string, the word 'direction' referst to a physical direction of the lattice
        or the index `u` of the unit cell as an "artificial direction".

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


def get_order_grouped(shape, groups, priority=None):
    """Variant of :func:`get_order`, grouping some sites of the unit cell.

    This function is usefull for lattices with a unit cell of more than 2 sites (e.g. Kagome).
    For 2D lattices with a unit cell, the ordering goes
    first within a group , then along y,
    then the next group (for the same x-value), again along y,
    and finally along x when all groups are done.

    As an example, consider the Kagome lattice.

    .. plot ::

        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 6))
        groups = [[(0, 1, 2)], [(0, 2, 1)],
                [(0, 1), (2,)], [(0, 2), (1,)]]
        priorities = [None, None, None, [1, 0, 2]]
        lat = lattice.Kagome(3, 3, None, bc='periodic')
        for gr, prio, ax in zip(groups, priorities, axes.flatten()):
            order = lattice.get_order_grouped(lat.shape, gr, prio)
            lat.order = order
            lat.plot_order(ax, linestyle=':')
            lat.plot_sites(ax)
            lat.plot_basis(ax, origin=-0.25*(lat.basis[0] + lat.basis[1]))
            ax.set_title(', '.join(['("grouped"', str(gr), str(prio) + ')']))
            ax.set_aspect('equal')
            ax.set_xlim(-1)
            ax.set_ylim(-1)
        plt.show()

    .. note ::
        In this doc-string, the word 'direction' referst to a physical direction of the lattice
        or the index `u` of the unit cell as an "artificial direction".

    Parameters
    ----------
    shape : tuple of int
        The shape of the lattice, i.e., the length in each direction.
    groups : tuple of tuple of int
        A partition and reordering of ``range(shape[-1])`` into smaller groups.
        The ordering goes first within a group, then along the last spatial dimensions, then
        changing between different groups and finally in Cstyle order along the remaining spatial
        dimensions.
    priority : None | tuple of ints
        By default (`None`), use C-style order for everything except the unit cell, as shown above.
        If a tuple, it should have length ``len(shape)`` and specifies which order to go first,
        similarly as in :func:`get_order`. To group sites in the unit cell, you should make the
        last entry of `priority` the largest. However, you can also choose to group along another
        direction - in that case `groups` should be a partitioning of
        ``range(shape(argmax(priority)))``. Try and plot it, if you need it!

    Returns
    -------
    order : ndarray (np.prod(shape), len(shape))
        An order of the sites for :attr:`Lattice.order` in the specified `ordering`.

    See also
    --------
    :meth:`Lattice.ordering` : method in :class:`Lattice` to obtain the order from parameters.
    :meth:`Lattice.plot_order` : visualizes the resulting order in a :class:`Lattice`.
    """
    if priority is not None:
        # reduce this case to C-style order and a few permutations
        perm = np.argsort(priority)
        inv_perm = inverse_permutation(perm)
        transp_shape = np.array(shape)[perm]
        order = get_order_grouped(transp_shape, groups, None)  # in plain C-style
        order = order[:, inv_perm]
        return order
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
