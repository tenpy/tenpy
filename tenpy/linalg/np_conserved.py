r"""A module to handle charge conservation in tensor networks.

A detailed introduction (including notations) can be found in :doc:`../IntroNpc`.

This module `np_conserved` implements an class :class:`Array`
designed to make use of charge conservation in tensor networks.
The idea is that the `Array` class is used in a fashion very similar to
the `numpy.ndarray`, e.g you can call the functions :func:`tensordot` or :func:`svd`
(of this module) on them.
The structure of the algorithms (as DMRG) is thus the same as with basic numpy ndarrays.

Internally, an :class:`Array` saves charge meta data to keep track of blocks which are nonzero.
All possible operations (e.g. tensordot, svd, ...) on such arrays preserve the total charge
structure. In addition, these operations make use of the charges to figure out which of the blocks
it hase to use/combine - this is the basis for the speed-up.


See also
--------
:mod:`tenpy.linalg.charges` : Implementation of :class:`~tenpy.linalg.charges.ChargeInfo`
and :class:`~tenpy.linalg.charges.LegCharge` with additional documentation.


.. todo ::
   Routine listing,
   update ``from charges import``,
   write example section
"""
# Examples
# --------
# >>> import numpy as np
# >>> import tenpy.linalg.np_conserved as npc
# >>> Sz = np.array([[0., 1.], [1., 0.]])
# >>> Sz_c = npc.Array.from_ndarray_trivial(Sz)  # convert to npc array with trivial charge
# >>> Sz_c
# <npc.Array shape=(2, 2)>
# >>> sx = npc.ndarray.from_ndarray([[0., 1.], [1., 0.]])  # trivial charge conservation
# >>> b = npc.ndarray.from_ndarray([[0., 1.], [1., 0.]])  # trivial charge conservation
# >>>
# >>> print a[0, -1]
# >>> c = npc.tensordot(a, b, axes=([1], [0]))

from __future__ import division

import numpy as np
import copy as copy_
import warnings
import itertools

# import public API from charges
from .charges import (QDTYPE, ChargeInfo, LegCharge, LegPipe,
                      reverse_sort_perm)
from . import charges   # for private functions

from ..tools.math import toiterable

#: A cutoff to ignore machine precision rounding errors when determining charges
QCUTOFF = np.finfo(np.float64).eps * 10


class Array(object):
    r"""A multidimensional array (=tensor) for using charge conservation.

    An `Array` represents a multi-dimensional tensor,
    together with the charge structure of its legs (for abelian charges).
    Further information can be found in :doc:`../IntroNpc`.

    The default :meth:`__init__` (i.e. ``Array(...)``) does not insert any data,
    and thus yields an Array 'full' of zeros, equivalent to :func:`zeros()`.
    Further, new arrays can be created with one of :meth:`from_ndarray_trivial`,
    :meth:`from_ndarray`, or :meth:`from_npfunc`, and of course by copying/tensordot/svd etc.

    Parameters
    ----------
    chargeinfo : :class:`~tenpy.linalg.charges.ChargeInfo`
        the nature of the charge, used as self.chinfo.
    legs : list of :class:`~tenpy.linalg.charges.LegCharge`
        the leg charges for each of the legs.
    dtype : type or string
        the data type of the array entries. Defaults to np.float64.


    Attributes
    ----------
    shape : tuple(int)
        the number of indices for each of the legs
    dtype : np.dtype
        the data type of the entries
    chinfo : :class:`~tenpy.linalg.charges.ChargeInfo`
        the nature of the charge
    qtotal : charge values
        the total charge of the tensor.
    legs : list of :class:`~tenpy.linalg.charges.LegCharge`
        the leg charges for each of the legs.
    labels : dict (string -> int)
        labels for the different legs
    _data : list of arrays
        the actual entries of the tensor
    _qdata : 2D array (len(_data), rank)
        for each of the _data entries the qind of the different legs.
    _qdata_sorted : Bool
        whether self._qdata is lexsorted. Defaults to `True`,
        but *must* be set to `False` by algorithms changing _qdata.

    Notes
    -----
    The Array

    .. todo ::

        - Somehow, including `rank` to the list of attributes breaks
          the sphinx build for this class...
        - What about size and stored_blocks?
        - Methods section
        - should _qdata be of type np.intp instead of QDTYPE?
    """
    def __init__(self, chargeinfo, legcharges, dtype=np.float64, qtotal=None):
        """see help(self)"""
        self.chinfo = chargeinfo
        self.legs = list(legcharges)
        self._set_shape()
        self.dtype = np.dtype(dtype)
        self.qtotal = self.chinfo.make_valid(qtotal)
        self.labels = {}
        self._data = []
        self._qdata = np.empty((0, self.rank), QDTYPE)
        self._qdata_sorted = True
        self.test_sanity()

    def copy(self, deep=False):
        """Return a (deep or shallow) copy of self.

        **Both** deep and shallow copies will share ``chinfo`` and the `LegCharges` in ``legs``.
        In contrast to a deep copy, the shallow copy will also share the tensor entries.
        """
        if deep:
            cp = copy_.deepcopy(self)
        else:
            cp = copy_.copy(self)
            # some things should be copied even for shallow copies
            cp._set_shape()
            cp.qtotal = cp.qtotal.copy()
        # even deep copies can share chargeinfo and legs
        cp.chinfo = self.chinfo
        cp.legs = self.legs[:]
        return cp

    @classmethod
    def from_ndarray_trivial(cls, data_flat, dtype=np.float64):
        """convert a flat numpy ndarray to an Array with trivial charge conservation.

        Parameters
        ----------
        data_flat : array_like
            the data to be converted to a Array
        dtype : type | string
            the data type of the array entries. Defaults to ``np.float64``.

        Returns
        -------
        res : :class:`Array`
            an Array with data of data_flat
        """
        data_flat = np.array(data_flat, dtype)
        chinfo = ChargeInfo()
        legs = [LegCharge.from_trivial(s, chinfo) for s in data_flat.shape]
        res = cls(chinfo, legs, dtype)
        res._data = [data_flat]
        res._qdata = np.zeros((1, res.rank), QDTYPE)
        res._qdata_sorted = True
        res.test_sanity()
        return res

    @classmethod
    def from_ndarray(cls, data_flat, chargeinfo, legcharges, dtype=np.float64, qtotal=None,
                     cutoff=None):
        """convert a flat (numpy) ndarray to an Array.

        Parameters
        ----------
        data_flat : array_like
            the flat ndarray which should be converted to a npc `Array`.
            The shape has to be compatible with legcharges.
        chargeinfo : ChargeInfo
            the nature of the charge
        legcharges : list of LegCharge
            a LegCharge for each of the legs.
        dtype : type | string
            the data type of the array entries. Defaults to np.float64.
        qtotal : None | charges
            the total charge of the new array.
        cutoff : float
            A cutoff to exclude rounding errors of machine precision. Defaults to :data:`QCUTOFF`.

        Returns
        -------
        res : :class:`Array`
            an Array with data of `data_flat`.

        See also
        --------
        detect_ndarray_qtotal : used to detect the total charge of the flat array.
        """
        if cutoff is None:
            cutoff = QCUTOFF
        res = cls(chargeinfo, legcharges, dtype, qtotal)  # without any data
        data_flat = np.asarray(data_flat, dtype=res.dtype)
        if res.shape != data_flat.shape:
            raise ValueError("Incompatible shapes: legcharges {0!s} vs flat {1!s} ".format(
                res.shape, data_flat.shape))
        if qtotal is None:
            res.qtotal = qtotal = res.detect_ndarray_qtotal(data_flat, cutoff)
        data = []
        qdata = []
        for qindices in res._iter_all_blocks():
            sl = res._get_block_slices(qindices)
            if np.all(res._get_block_charge(qindices) == qtotal):
                data.append(np.array(data_flat[sl], dtype=res.dtype))   # copy data
                qdata.append(qindices)
            elif np.any(np.abs(data_flat[sl]) > cutoff):
                warnings.warn("flat array has non-zero entries in blocks incompatible with charge")
        res._data = data
        if len(qdata) == 0:
            res._qdata = np.empty((0, res.rank), dtype=QDTYPE)
        else:
            res._qdata = np.array(qdata, dtype=QDTYPE)
        res._qdata_sorted = True
        res.test_sanity()
        return res

    @classmethod
    def from_func(cls, func, chargeinfo, legcharges, dtype=np.float64, qtotal=None,
                  func_args=(), func_kwargs={}, shape_kw=None):
        """Create an Array from a numpy func.

        This function creates an array and fills the blocks *compatible* with the charges
        using `func`, where `func` is a function returning a `array_like` when given a shape,
        e.g. one of ``np.ones`` or ``np.random.standard_normal``.

        Parameters
        ----------
        func : callable
            a function-like object which is called to generate the data blocks.
            We expect that `func` returns a flat array of the given `shape` convertible to `dtype`.
            If no `shape_kw` is given, it is called like ``func(shape, *fargs, **fkwargs)``,
            otherwise as ``func(*fargs, `shape_kw`=shape, **fkwargs)``.
            `shape` is a tuple of int.
        chargeinfo : ChargeInfo
            the nature of the charge
        legcharges : list of LegCharge
            a LegCharge for each of the legs.
        dtype : type | string
            the data type of the output entries. Defaults to np.float64.
            Note that this argument is not given to func, but rather a type conversion
            is performed afterwards. You might want to set a `dtype` in `func_kwargs` as well.
        qtotal : None | charges
            the total charge of the new array. Defaults to charge 0.
        func_args : iterable
            additional arguments given to `func`
        func_kwargs : dict
            additional keyword arguments given to `func`
        shape_kw : None | str
            If given, the keyword with which shape is given to `func`.

        Returns
        -------
        res : :class:`Array`
            an Array with blocks filled using `func`.
        """
        res = cls(chargeinfo, legcharges, dtype, qtotal)  # without any data yet.
        data = []
        qdata = []
        for qindices in res._iter_all_blocks():
            if np.any(res._get_block_charge(qindices) != res.qtotal):
                continue
            shape = res._get_block_shape(qindices)
            if shape_kw is None:
                block = func(shape, *func_args, **func_kwargs)
            else:
                kws = func_kwargs.copy()
                kws[shape_kw] = shape
                block = func(*func_args, **kws)
            block = np.asarray(block, dtype=res.dtype)
            data.append(block)
            qdata.append(qindices)
        res._data = data
        if len(qdata) == 0:
            res._qdata = np.empty((0, res.rank), QDTYPE)
        else:
            res._qdata = np.array(qdata, dtype=QDTYPE)
        res._qdata_sorted = True  # _iter_all_blocks is in lexiographic order
        res.test_sanity()
        return res

    def zeros_like(self):
        """return a shallow copy of self with only zeros as entries, containing no `_data`"""
        res = self.copy(deep=False)
        res._data = []
        res._qdata = np.empty((0, res.rank), QDTYPE)
        res._qdata_sorted = True
        return res

    def test_sanity(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
        if self.shape != tuple([lc.ind_len for lc in self.legs]):
            raise ValueError("shape mismatch with LegCharges\n self.shape={0!s} != {1!s}".format(
                self.shape, tuple([lc.ind_len for lc in self.legs])))
        if any([self.dtype != d.dtype for d in self._data]):
            raise ValueError("wrong dtype: {0!s} vs\n {1!s}".format(
                self.dtype, [self.dtype != d.dtype for d in self._data]))
        for l in self.legs:
            l.test_sanity()
            if l.chinfo != self.chinfo:
                raise ValueError("leg has different ChargeInfo:\n{0!s}\n vs {1!s}".format(
                    l.chinfo, self.chinfo))
        if self._qdata.shape != (self.stored_blocks, self.rank):
            raise ValueError("_qdata shape wrong")
        if np.any(self._qdata < 0) or np.any(self._qdata >= [l.block_number for l in self.legs]):
            raise ValueError("invalid qind in _qdata")
        if self._qdata_sorted:
            perm = np.lexsort(self._qdata.T)
            if np.any(perm != np.arange(len(perm))):
                raise ValueError("_qdata_sorted == True, but _qdata is not sorted")

    # properties ==============================================================

    @property
    def rank(self):
        """the number of legs"""
        return len(self.shape)

    @property
    def size(self):
        """the number of dtype-objects stored"""
        return np.sum([t.size for t in self._data], dtype=np.int_)

    @property
    def stored_blocks(self):
        """the number of (non-zero) blocks stored in self._data"""
        return len(self._data)

    # accessing entries =======================================================

    def to_ndarray(self):
        """convert self to a dense numpy ndarray."""
        res = np.zeros(self.shape, self.dtype)
        for block, slices, _, _ in self:  # that's elegant! :)
            res[slices] = block
        return res

    def __iter__(self):
        """Allow to iterate over the non-zero blocks, giving all `_data`.

        Yields
        ------
        block : ndarray
            the actual entries of a charge block
        blockslices : tuple of slices
            a slice giving the range of the block in the original tensor for each of the legs
        charges : list of charges
            the charge value(s) for each of the legs (takink `qconj` into account)
        qdat : ndarray
            the qindex for each of the legs
        """
        for block, qdat in itertools.izip(self._data, self._qdata):
            blockslices = []
            qs = []
            for (qi, l) in itertools.izip(qdat, self.legs):
                blockslices.append(l.get_slice(qi))
                qs.append(l.get_charge(qi))
            yield block, tuple(blockslices), qs, qdat

    def __getitem__(self, inds):
        """acces entries with ``self[inds]``

        Parameters
        ----------
        inds : tuple
            A tuple specifying the `index` for each leg.
            An ``Ellipsis`` (written as ``...``) replaces ``slice(None)`` for missing axes.
            For a single `index`, we currently support:

            - A single integer, choosing an index of the axis,
              reducing the dimension of the resulting array.
            - A ``slice(None)`` specifying the complete axis.
            - A ``slice``, which acts like a `mask` in :meth:`iproject`.
            - A 1D array_like(bool): acts like a `mask` in :meth:`iproject`.
            - A 1D array_like(int): acts like a `mask` in :meth:`iproject`,
              and if not orderd, a subsequent permuation with :meth:`permute`

        Returns
        -------
        res : `dtype`
            only returned, if a single integer is given for all legs.
            It is the entry specified by `inds`, giving ``0.`` for non-saved blocks.
        or
        sliced : :class:`Array`
            a copy with some of the data removed by :meth:`take_slice` and/or :meth:`project`

        Notes
        -----
        ``self[i]`` is equivalent to ``self[i, ...]``.
        ``self[i, ..., j]`` is syntactic sugar for ``self[(i, Ellipsis, i2)]``

        Raises
        ------
        IndexError
            If the number of indices is too large, or
            if an index is out of range.
        """
        int_only, inds = self._pre_indexing(inds)
        if int_only:
            pos = np.array([l.get_qindex(i) for i, l in zip(inds, self.legs)])
            block = self._get_block(pos[:, 0])
            if block is None:
                return self.dtype.type(0)
            else:
                return block[tuple(pos[:, 1])]
        # advanced indexing
        return self._advanced_getitem(inds)

    def __setitem__(self, inds, other):
        """assign ``self[inds] = other``.

        Should work as expected for both basic and advanced indexing as described in
        :meth:`__getitem__`.
        `other` can be:
        - a single value (if all of `inds` are integer)
        or for slicing/advanced indexing:
        - a :class:`Array`, with charges as ``self[inds]`` returned by :meth:`__getitem__`.
        - or a flat numpy array, assuming the charges as with ``self[inds]``.
        """
        int_only, inds = self._pre_indexing(inds)
        if int_only:
            pos = np.array([l.get_qindex(i) for i, l in zip(inds, self.legs)])
            block = self._get_block(pos[:, 0], insert=True, raise_incomp_q=True)
            block[tuple(pos[:, 1])] = other
            return
        # advanced indexing
        if not isinstance(other, Array):
            # if other is a flat array, convert it to an npc Array
            like_other = self.zeros_like()._advanced_getitem(inds)
            other = Array.from_ndarray(other, self.chinfo, like_other.legs, self.dtype,
                                       like_other.qtotal)
        self._advanced_setitem_npc(inds, other)

    def take_slice(self, indices, axes):
        """Return a copy of self fixing `indices` along one or multiple `axes`.

        For a rank-4 Array ``A.take_slice([i, j], [1,2])`` is equivalent to ``A[:, i, j, :]``.

        Parameters
        ----------
        indices : (iterable of) int
            the (flat) index for each of the legs specified by `axes`
        axes : (iterable of) str/int
            leg labels or indices to specify the legs for which the indices are given.

        Returns
        -------
        slided_self : :class:`Array`
            a copy of self, equivalent to taking slices with indices inserted in axes.
        """
        axes = self.get_leg_indices(toiterable(axes))
        indices = np.asarray(toiterable(indices), dtype=np.intp)
        if len(axes) != len(indices):
            raise ValueError("len(axes) != len(indices)")
        if indices.ndim != 1:
            raise ValueError("indices may only contain ints")
        res = self.copy(deep=True)
        if len(axes) == 0:
            return res  # nothing to do
        # qindex and index_within_block for each of the axes
        pos = np.array([self.legs[a].get_qindex(i) for a, i in zip(axes, indices)])
        # which axes to keep
        keep_axes = [a for a in xrange(self.rank) if a not in axes]
        res.legs = [self.legs[a] for a in keep_axes]
        res._set_shape()
        labels = self.get_leg_labels()
        res.set_leg_labels([labels[a] for a in keep_axes])
        # calculate new total charge
        for a, (qi, _) in zip(axes, pos):
            res.qtotal -= self.legs[a].get_charge(qi)
        res.qtotal = self.chinfo.make_valid(res.qtotal)
        # which blocks to keep
        axes = np.array(axes, dtype=np.intp)
        keep_axes = np.array(keep_axes, dtype=np.intp)
        keep_blocks = np.all(self._qdata[:, axes] == pos[:, 0], axis=1)
        res._qdata = self._qdata[np.ix_(keep_blocks, keep_axes)].copy()
        # res._qdata_sorted is not changed
        # determine the slices to take on _data
        sl = [slice(None)] * self.rank
        for a, ri in zip(axes, pos[:, 1]):
            sl[a] = ri  # the indices within the blocks
        sl = tuple(sl)
        # finally take slices on _data
        res._data = [block[sl] for block, k in itertools.izip(res._data, keep_blocks) if k]
        return res

    # handling of charges =====================================================

    def detect_ndarray_qtotal(self, flat_array, cutoff=None):
        """ Returns the total charge of first non-zero sector found in `a`.

        Charge information is taken from self.
        If you have only the charge data, create an empty Array(chinf, legcharges).

        Parameters
        ----------
        flat_array : array
            the flat numpy array from which you want to detect the charges
        chinfo : ChargeInfo
            the nature of the charge
        legcharges : list of LegCharge
            for each leg the LegCharge
        cutoff : float
            defaults to :data:`QCUTOFF`

        Returns
        -------
        qtotal : charge
            the total charge fo the first non-zero (i.e. > cutoff) charge block
        """
        if cutoff is None:
            cutoff = QCUTOFF
        for qindices in self._iter_all_blocks():
            sl = self._get_block_slices(qindices)
            if np.any(np.abs(flat_array[sl]) > cutoff):
                return self._get_block_charge(qindices)
        warnings.warn("can't detect total charge: no entry larger than cutoff. Return 0 charge.")
        return self.chinfo.make_valid()

    def gauge_total_charge(self, leg, newqtotal=None):
        """changes the total charge of an Array `A` inplace by adjusting the charge on a certain leg.

        The total charge is given by finding a nonzero entry [i1, i2, ...] and calculating::

            qtotal = sum([l.qind[qi, 2:] * l.conj for i, l in zip([i1,i2,...], self.legs)])

        Thus, the total charge can be changed by redefining the leg charge of a given leg.
        This is exaclty what this function does.

        Parameters
        ----------
        leg : int or string
            the new leg (index or label), for which the charge is changed
        newqtotal : charge values, defaults to 0
            the new total charge
        """
        leg = self.get_leg_index(leg)
        newqtotal = self.chinfo.make_valid(newqtotal)  # converts to array, default zero
        chdiff = newqtotal - self.qtotal
        if isinstance(leg, LegPipe):
            raise ValueError("not possible for a LegPipe. Convert to a LegCharge first!")
        newleg = copy_.copy(self.legs[leg])  # shallow copy of the LegCharge
        newleg.qind = newleg.qind.copy()
        newleg.qind[:, 2:] = self.chinfo.make_valid(newleg.qind[:, 2:] + newleg.qconj * chdiff)
        self.legs[leg] = newleg
        self.qtotal = newqtotal

    def is_completely_blocked(self):
        """returns bool wheter all legs are blocked by charge"""
        return all([l.is_blocked() for l in self.legs])

    def sort_legcharge(self, sort=True, bunch=True):
        """Return a copy with one ore all legs sorted by charges.

        Sort/bunch one or multiple of the LegCharges.
        Legs which are sorted *and* bunched are guaranteed to be blocked by charge.

        Parameters
        ----------
        sort : True | False | list of {True, False, perm}
            A single bool holds for all legs, default=True.
            Else, `sort` should contain one entry for each leg, with a bool for sort/don't sort,
            or a 1D array perm for a given permuation to apply to a leg.
        bunch : True | False | list of {True, False}
            A single bool holds for all legs, default=True.
            whether or not to bunch at each leg, i.e. combine contiguous blocks with equal charges.

        Returns
        -------
        perm : tuple of 1D arrays
            the permutation applied to each of the legs.
            cp.to_ndarray() = self.to_ndarray(perm)
        result : Array
            a shallow copy of self, with legs sorted/bunched
        """
        if sort is False or sort is True:  # ``sort in [False, True]`` doesn't work...
            sort = [sort]*self.rank
        if bunch is False or bunch is True:
            bunch = [bunch]*self.rank
        if not len(sort) == len(bunch) == self.rank:
            raise ValueError("Wrong len for bunch or sort")
        cp = self.copy(deep=False)
        cp._qdata = cp._qdata.copy(self)
        for li in xrange(self.rank):
            if sort[li] is not False:
                if sort[li] is True:
                    if cp.legs[li].sorted:  # optimization if the leg is sorted already
                        sort[li] = np.arange(cp.shape[li])
                        continue
                    p_qind, newleg = cp.legs[li].sort(bunch=False)
                    sort[li] = cp.legs[li].perm_flat_from_qind(p_qind)  # called for the old leg
                    cp.legs[li] = newleg
                else:
                    try:
                        p_qind = self.legs[li].perm_qind_from_perm_flat(sort[li])
                    except ValueError:  # permutation mixes qindices
                        cp = cp.permute(sort[li], axes=[li])
                        continue
                cp._perm_qind(p_qind, li)
            else:
                sort[li] = np.arange(cp.shape[li])
        if any(bunch):
            cp = cp._bunch(bunch)  # bunch does not permute...
        return tuple(sort), cp

    def sort_qdata(self):
        """(lex)sort ``self._qdata``. In place.

        Lexsort ``self._qdata`` and ``self._data`` and set ``self._qdata_sorted = True``.
        """
        if self._qdata_sorted:
            return
        if len(self._qdata) < 2:
            self._qdata_sorted = True
            return
        perm = np.lexsort(self._qdata.T)
        self._qdata = self._qdata[perm, :]
        self._data = [self._data[p] for p in perm]
        self._qdata_sorted = True

    def make_pipe(self, axes, **kwargs):
        """generates a :class:`~tenpy.linalg.charges.LegPipe` for specified axes.

        Parameters
        ----------
        axes : iterable of str|int
            the leg labels for the axes which should be combined. Order matters!
        **kwargs :
            additional keyword arguments given to :class:`~tenpy.linalg.charges.LegPipe`

        Returns
        -------
        pipe : :class:`~tenpy.linalg.charges.LegPipe`
            A pipe of the legs specified by axes.
        """
        axes = self.get_leg_indices(axes)
        legs = [self.legs[a] for a in axes]
        return charges.LegPipe(legs, **kwargs)

    def combine_legs(self, combine_legs, new_axes=None, pipes=None, qconj=None):
        """combine multiple legs into pipes. If necessary, transpose before.

        Parameters
        ----------
        combine_legs : (iterable of) iterable of {str|int}
            bundles of leg indices or labels, which should be combined into a new output pipes.
            If multiple pipes should be created, use a list fore each new pipe.
        new_axes : None | (iterable of) int
            The leg-indices, at which the combined legs should appear in the resulting array.
            Default: for each pipe the position of its first pipe in the original array,
            (taking into account that some axes are 'removed' by combining).
            Thus no transposition is perfomed if `combine_legs` contains only contiguous ranges.
        pipes : None | (iterable of) {:class:`LegPipes` | None}
            optional: provide one or multiple of the resulting LegPipes to avoid overhead of
            computing new leg pipes for the same legs multiple times.
            The LegPipes are conjugated, if that is necessary for compatibility with the legs.
        qconj : (iterable of) {+1, -1}
            specify whether new created charges point inward or outward. Defaults to +1.
            Ignored for given `pipes`, which are not newly calculated.

        Returns
        -------
        res : :class:`Array`
            A copy of self, whith some legs combined into pipes as specified by the arguments.

        Notes
        -----
        Labels are inherited from self.
        New pipe labels are generated as `` '(' + '.'.join(*leglabels) + ')' ``.
        For these new labels, previously unlabeled legs are replaced by '?#',
        where '#' is the leg-index in the original tensor `self`.

        Examples
        --------
        >>> oldarray.set_leg_labels(['a', 'b', 'c', 'd', 'e'])
        >>> c1 = oldarray.combine_legs([2, 3], qconj=-1)  # only single output pipe
        >>> c1.get_leg_labels()
        ['a', '(b.c)', 'd', 'e']

        Indices of `combine_legs` refer to the original array.
        Necessary permutations are performed automatically:

        >>> c2 = oldarray.combine_legs([[1, 4], [5, 2]], qconj=[+1, -1]) # two output pipes
        >>> c2.get_leg_labels()
        ['(a.d)', 'c', '(e.b)']
        >>> c3 = oldarray.combine_legs([['a', 'd'], ['e', 'b']], new_axes=[2, 1],
        >>>                            pipes=[c2.legs[0], c2.legs[2]])
        >>> c3.get_leg_labels()
        ['b', '(e.b)', '(a.d)']

        .. todo ::

            test this function

        """
        # bring arguments into a standard form
        combine_legs = list(combine_legs)  # convert iterable to list
        # check: is combine_legs `iterable(iterable(int|str))` or `iterable(int|str)` ?
        if [combine_legs[0]] == toiterable(combine_legs[0]):
            # the first entry is (int|str) -> only a single new pipe
            combine_legs = [combine_legs]
            if new_axes is not None:
                new_axes = toiterable(new_axes)
            if pipes is not None:
                pipes = toiterable(pipes)
        npipes = len(combine_legs)
        # default arguments for pipes and qconj
        if pipes is None:
            pipes = [None]*npipes
        elif len(pipes) != npipes:
            raise ValueError("wrong len of `pipes`")
        qconj = list(toiterable(qconj if qconj is not None else +1))
        if len(qconj) == 1 and 1 < npipes:
            qconj = [qconj[0]]*npipes  # same qconj for all pipes
        if len(qconj) != npipes:
            raise ValueError("wrong len of `qconj`")

        # good for index tricks: convert combine_legs into arrays
        combine_legs = [np.asarray(self.get_leg_indices(cl), dtype=np.intp) for cl in combine_legs]
        all_combine_legs = np.concatenate(combine_legs)
        if len(set(all_combine_legs)) != len(all_combine_legs):
            raise ValueError("got a leg multiple times: " + str(combine_legs))
        # make pipes as necessary
        for i, pipe in enumerate(pipes):
            if pipe is None:
                pipes[i] = self.make_pipe(axes=combine_legs[i], qconj=qconj[i])
            else:
                # test for compatibility
                legs = [self.legs[a] for a in combine_legs[i]]
                if pipe.nlegs != len(legs):
                    raise ValueError("pipe has wrong number of legs")
                if legs[0].qconj != pipe.legs[0].qconj:
                    pipes[i] = pipe = pipe.conj()  # need opposite qind
                for self_leg, pipe_leg in zip(legs, pipe.legs):
                    self_leg.test_contractible(pipe_leg.conj())
        # (now we can forget about `qconj`)

        # figure out new order of axes, i.e. how to transpose the axes
        # (first without the pipes, later insert the pipes at `new_axes`)
        transp = [i for i in range(self.rank) if i not in all_combine_legs]
        non_combined_legs = np.array(transp)
        if new_axes is None:  # figure out default new_legs
            first_cl = np.array([cl[0] for cl in combine_legs])
            new_axes = [(np.sum(non_combined_legs < a) + np.sum(first_cl < a))
                        for a in first_cl]
        else:
            # test compatibility
            if len(new_axes) != npipes:
                raise ValueError("wrong len of `new_axes`")
            na_max = len(pipes) + len(non_combined_legs)
            for i, na in enumerate(new_axes):
                if na < 0:
                    new_axes[i] = na + na_max
                elif na >= na_max:
                    raise ValueError("new_axis larger than the new number of legs")

        # permute arguments sucht that new_axes is sorted ascending
        perm_args = np.argsort(new_axes)
        combine_legs = [combine_legs[p] for p in perm_args]
        pipes = [pipes[p] for p in perm_args]
        new_axes = [new_axes[p] for p in perm_args]
        # insert the combined legs into `transp` at `new_axes`
        for na, cl in reversed(zip(new_axes, combine_legs)):
            # reversed: insert from the back, otherwise we would need to shift
            transp[na:na] = cl
        # now, `transp` has again len(self.rank) and gives the necessary transposition.
        # labels: replace non-set labels with '?#' before transpose
        labels = [(l if l is not None else '?'+str(i))
                  for i, l in enumerate(self.get_leg_labels())]
        # transpose if necessary
        if transp != range(self.rank):
            res = self.copy(deep=False)
            res.set_leg_labels(labels)
            res = res.transpose(transp)
            tr_combine_legs = [range(na, na+len(cl)) for na, cl in zip(new_axes, combine_legs)]
            return res.combine_legs(tr_combine_legs, new_axes=new_axes, pipes=pipes)
        # if we come here, combine_legs has the form of `tr_combine_legs`.

        # the **main work** of copying the data is sourced out, now that we have the
        # standard form of our arguments
        res = self._combine_legs_worker(combine_legs, non_combined_legs, new_axes, pipes)

        # get new labels
        pipe_labels = [('(' + '.'.join([labels[c] for c in cl]) + ')') for cl in combine_legs]
        for na, p, plab in zip(new_axes, pipes, pipe_labels):
            labels[na:na+p.nlegs] = plab
        res.set_leg_labels(labels)
        return res

    def split_legs(self, axes):
        """split legs, which were previously combined. They have a LegCharge

        .. todo ::
            implement
        """
        raise NotImplementedError()

    # data manipulation =======================================================

    def astype(self, dtype):
        """Return (deep) copy with new dtype, upcasting all blocks in ``_data``.

        Parameters
        ----------
        dtype : convertible to a np.dtype
            the new data type.
            If None, deduce the new dtype as common type of ``self._data``.

        Returns
        -------
        copy : :class:`Array`
            deep copy of self with new dtype
        """
        cp = self.copy(deep=False)  # manual deep copy: don't copy every block twice
        cp._qdata = cp._qdata.copy()
        if dtype is None:
            dtype = np.common_dtype(*self._data)
        cp.dtype = np.dtype(dtype)
        cp._data = [d.astype(self.dtype, copy=True) for d in self._data]
        return cp

    def ipurge_zeros(self, cutoff=QCUTOFF, norm_order=None):
        """Removes ``self._data`` blocks with norm less than cutoff. In place.

        Parameters
        ----------
        cutoff : float
            blocks with norm <= `cutoff` are removed. defaults to :data:`QCUTOFF`.
        norm_order :
            a valid `ord` argument for np.linalg.norm.
        """
        if len(self._data) == 0:
            return
        norm = np.array([np.linalg.norm(t, ord=norm_order) for t in self._data])
        keep = (norm > cutoff)  # bool array
        self._data = [t for t, k in itertools.izip(self._data, keep) if k]
        self._qdata = self._qdata[keep]
        # self._qdata_sorted is preserved
        return self

    def iproject(self, mask, axes):
        """Applying masks to one or multiple axes. In place.

        This function is similar as `np.compress` with boolean arrays
        For each specified axis, a boolean 1D array `mask` can be given,
        which chooses the indices to keep.

        .. warning ::
            Although it is possible to use an 1D int array as a mask, the order is ignored!
            If you really need to permute an axis, use :meth:`permute`.

        Parameters
        ----------
        mask : (list of) 1D array(bool|int)
            for each axis specified by `axes` a mask, which indices of the axes should be kept.
            If `mask` is a bool array, keep the indices where `mask` is True.
            If `mask` is an int array, keep the indices listed in the mask, *ignoring* the
            order or multiplicity.
        axes : (list of) int | string
            The `i`th entry in this list specifies the axis for the `i`th entry of `mask`,
            either as an int, or with a leg label.
            If axes is just a single int/string, specify just one mask.

        Returns
        -------
        map_qind : list of 1D arrays
            the mapping of qindices for each of the specified axes.
        block_masks: list of lists of 1D bool arrays
            ``block_masks[a][qind]`` is a boolen mask which indices to keep
            in block ``qindex`` of ``axes[a]``
        """
        axes = self.get_leg_indices(toiterable(axes))
        mask = [np.asarray(m) for m in toiterable(mask)]
        if len(axes) != len(mask):
            raise ValueError("len(axes) != len(mask)")
        if len(axes) == 0:
            return [], []  # nothing to do.
        for i, m in enumerate(mask):
            # convert integer masks to bool masks
            if m.dtype != np.bool_:
                mask[i] = np.zeros(self.shape[axes[i]], dtype=np.bool_)
                np.put(mask[i], m, True)
        block_masks = []
        proj_data = np.arange(self.stored_blocks)
        map_qind = []
        for m, a in zip(mask, axes):
            l = self.legs[a]
            m_qind, bm, self.legs[a] = l.project(m)
            map_qind.append(m_qind)
            block_masks.append(bm)
            q = self._qdata[:, a] = m_qind[self._qdata[:, a]]
            piv = (q >= 0)
            self._qdata = self._qdata[piv]
            # self._qdata_sorted is preserved
            proj_data = proj_data[piv]
        self._set_shape()
        # finally project out the blocks
        data = []
        for i, iold in enumerate(proj_data):
            block = self._data[iold]
            subidx = [slice(d) for d in block.shape]
            for m, a in zip(block_masks, axes):
                subidx[a] = m[self._qdata[i, a]]
                block = np.compress(m[self._qdata[i, a]], block, axis=a)
            data.append(block)
        self._data = data
        return map_qind, block_masks

    def permute(self, perm, axis):
        """Apply a permutation in the indices of an axis.

        Similar as np.take with a 1D array.
        Roughly equivalent to ``res[:, ...] = self[perm, ...]`` for the corresponding `axis`.
        .. warning ::

            This function is quite slow, and usually not needed during core algorithms.

        Parameters
        ----------
        perm : array_like 1D int
            The permutation which should be applied to the leg given by `axis`
        axis : str | int
            a leg label or index specifying on which leg to take the permutation.

        Returns
        -------
        res : :class:`Array`
            a copy of self with leg `axis` permuted, such that
            ``res[i, ...] = self[perm[i], ...]`` for ``i`` along `axis`

        See also
        --------
        sort_legcharge : can also be used to perform a general permutation.
            However, it is faster for permutations which don't mix blocks.
        """
        axis = self.get_leg_index(axis)
        perm = np.asarray(perm, dtype=np.intp)
        oldleg = self.legs[axis]
        if len(perm) != oldleg.ind_len:
            raise ValueError("permutation has wrong length")
        rev_perm = reverse_sort_perm(perm)
        newleg = LegCharge.from_qflat(self.chinfo, oldleg.to_qflat()[perm], oldleg.qconj)
        res = self.copy(deep=False)  # data is replaced afterwards
        res.legs[axis] = newleg
        qdata_axis = self._qdata[:, axis]
        new_block_idx = [slice(None)]*self.rank
        old_block_idx = [slice(None)]*self.rank
        data = []
        qdata = {}  # dict for fast look up: tuple(indices) -> _data index
        for old_qind, old_qind_row in enumerate(oldleg.qind):
            old_range = xrange(old_qind_row[0], old_qind_row[1])
            for old_data_index in np.nonzero(qdata_axis == old_qind)[0]:
                old_block = self._data[old_data_index]
                old_qindices = self._qdata[old_data_index]
                new_qindices = old_qindices.copy()
                for i_old in old_range:
                    i_new = rev_perm[i_old]
                    qi_new, within_new = newleg.get_qindex(i_new)
                    new_qindices[axis] = qi_new
                    # look up new_qindices in `qdata`, insert them if necessary
                    new_data_ind = qdata.setdefault(tuple(new_qindices), len(data))
                    if new_data_ind == len(data):
                        # insert new block
                        data.append(np.zeros(res._get_block_shape(new_qindices)))
                    new_block = data[new_data_ind]
                    # copy data
                    new_block_idx[axis] = within_new
                    old_block_idx[axis] = i_old - old_qind_row[0]
                    new_block[tuple(new_block_idx)] = old_block[tuple(old_block_idx)]
        # data blocks copied
        res._data = data
        res._qdata_sorted = False
        res_qdata = res._qdata = np.empty((len(data), self.rank), dtype=QDTYPE)
        for qindices, i in qdata.iteritems():
            res_qdata[i] = qindices
        return res

    def itranspose(self, axes=None):
        """Transpose axes like `np.transpose`. In place.

        Parameters
        ----------
        axes: iterable (int|string), len ``rank`` | None
            the new order of the axes. By default (None), reverse axes.
        """
        if axes is None:
            axes = tuple(reversed(xrange(self.rank)))
        else:
            axes = tuple(self.get_leg_indices(axes))
            if len(axes) != self.rank or len(set(axes)) != self.rank:
                raise ValueError("axes has wrong length: " + str(axes))
        axes_arr = np.array(axes)
        self.legs = [self.legs[a] for a in axes]
        self._set_shape()
        labs = self.get_leg_labels()
        self.set_leg_labels([labs[a] for a in axes])
        self._qdata = self._qdata[:, axes_arr]
        self._qdata_sorted = False
        self._data = [np.transpose(block, axes) for block in self._data]
        return self

    def transpose(self, axes=None):
        """Like :meth:`itranspose`, but on a deep copy."""
        cp = self.copy(deep=True)
        cp.itranspose(axes)
        return cp

    # labels ==================================================================

    def get_leg_index(self, label):
        """translate a leg-index or leg-label to a leg-index.

        Parameters
        ----------
        label : int | string
            eather the leg-index directly or a label (string) set before.

        Returns
        -------
        leg_index : int
            the index of the label

        See also
        --------
        get_leg_indices : calls get_leg_index for a list of labels
        set_leg_labels : set the labels of different legs.
        """
        res = self.labels.get(label, label)
        if res > self.rank:
            raise ValueError("axis {0:d} out of rank {1:d}".format(res, self.rank))
        elif res < 0:
            res += self.rank
        return res

    def get_leg_indices(self, labels):
        """Translate a list of leg-indices or leg-labels to leg indices.

        Parameters
        ----------
        labels : iterable of string/int
            The leg-labels (or directly indices) to be translated in leg-indices

        Returns
        -------
        leg_indices : list of int
            the translated labels.

        See also
        --------
        get_leg_index : used to translate each of the single entries.
        set_leg_labels : set the labels of different legs.
        """
        return [self.get_leg_index(l) for l in labels]

    def set_leg_labels(self, labels):
        """Return labels for the legs.

        Introduction to leg labeling can be found in :doc:`../IntroNpc`.

        Parameters
        ----------
        labels : iterable (strings | None), len=self.rank
            One label for each of the legs.
            An entry can be None for an anonymous leg.

        See also
        --------
        get_leg: translate the labels to indices
        get_legs: calls get_legs for an iterable of labels
        """
        if len(labels) != self.rank:
            raise ValueError("Need one leg label for each of the legs.")
        self.labels = {}
        for i, l in enumerate(labels):
            if l is not None:
                self.labels[l] = i

    def get_leg_labels(self):
        """Return tuple of the leg labels, with `None` for anonymous legs."""
        lb = [None] * self.rank
        for k, v in self.labels.iteritems():
            lb[v] = k
        return tuple(lb)

    # string output ===========================================================

    def __repr__(self):
        return "<npc.array shape={0!s} charge={1!s} labels={2!s}>".format(
            self.shape, self.chinfo, self.get_leg_labels())

    def __str__(self):
        res = "\n".join([repr(self)[:-1], str(self.to_ndarray()), ">"])
        return res

    def sparse_stats(self):
        """Returns a string detailing the sparse statistics"""
        total = np.prod(self.shape)
        if total is 0:
            return "Array without entries, one axis is empty."
        nblocks = self.stored_blocks
        stored = self.size
        nonzero = np.sum([np.count_nonzero(t) for t in self._data], dtype=np.int_)
        bs = np.array([t.size for t in self._data], dtype=np.float)
        if nblocks > 0:
            bs1 = (np.sum(bs**0.5)/nblocks)**2
            bs2 = np.sum(bs)/nblocks
            bs3 = (np.sum(bs**2.0)/nblocks)**0.5
            captsparse = float(nonzero) / stored
        else:
            captsparse = 1.
            bs1, bs2, bs3 = 0, 0, 0
        res = "{nonzero:d} of {total:d} entries (={nztotal:g}) nonzero,\n" \
            "stored in {nblocks:d} blocks with {stored:d} entries.\n" \
            "Captured sparsity: {captsparse:g}\n"  \
            "Effective block sizes (second entry=mean): [{bs1:.2f}, {bs2:.2f}, {bs3:.2f}]"

        return res.format(nonzero=nonzero, total=total, nztotal=nonzero/total, nblocks=nblocks,
                          stored=stored, captsparse=captsparse, bs1=bs1, bs2=bs2, bs3=bs3)

    # private functions =======================================================

    def _set_shape(self):
        """deduce self.shape from self.legs"""
        self.shape = tuple([lc.ind_len for lc in self.legs])

    def _iter_all_blocks(self):
        """generator to iterate over all combinations of qindices in lexiographic order.

        Yields
        ------
        qindices : tuple of int
            a qindex for each of the legs
        """
        for block_inds in itertools.product(*[xrange(l.block_number)
                                              for l in reversed(self.legs)]):
            # loop over all charge sectors in lex order (last leg most siginificant)
            yield tuple(block_inds[::-1])   # back to legs in correct order

    def _get_block_charge(self, qindices):
        """returns the charge of a block selected by `qindices`

        The charge of a single block is defined as ::

            qtotal = sum_{legs l} legs[l].qind[qindices[l], 2:] * legs[l].qconj() modulo qmod
        """
        q = np.sum([l.get_charge(qi) for l, qi in itertools.izip(self.legs, qindices)],
                   axis=0)
        return self.chinfo.make_valid(q)

    def _get_block_slices(self, qindices):
        """returns tuple of slices for a block selected by `qindices`"""
        return tuple([l.get_slice(qi) for l, qi in itertools.izip(self.legs, qindices)])

    def _get_block_shape(self, qindices):
        """return shape for the block given by qindices"""
        return tuple([(l.qind[qi, 1] - l.qind[qi, 0]) for l, qi in
                      itertools.izip(self.legs, qindices)])

    def _get_block(self, qindices, insert=False, raise_incomp_q=False):
        """return the ndarray in ``_data`` representing the block corresponding to `qindices`.

        Parameters
        ----------
        qindices : 1D array of QDTYPE
            the qindices, for which we need to look in _qdata
        insert : bool
            If True, insert a new (zero) block, if `qindices` is not existent in ``self._data``.
            Else: just return ``None`` in that case.
        raise_incomp_q : bool
            Raise an IndexError if the charge is incompatible.

        Returns
        -------
        block: ndarray
            the block in ``_data`` corresponding to qindices
            If `insert`=False and there is not block with qindices, return ``False``

        Raises
        ------
        IndexError
            If qindices are incompatible with charge and `raise_incomp_q`
        """
        if not np.all(self._get_block_charge(qindices) == self.qtotal):
            if raise_incomp_q:
                raise IndexError("trying to get block for qindices incompatible with charges")
            return None
        # find qindices in self._qdata
        match = np.argwhere(np.all(self._qdata == qindices, axis=1))[:, 0]
        if len(match) == 0:
            if insert:
                res = np.zeros(self._get_block_shape(qindices), dtype=self.dtype)
                self._data.append(res)
                self._qdata = np.append(self._qdata, [qindices], axis=0)
                self._qdata_sorted = False
                return res
            else:
                return None
        return self._data[match[0]]

    def _bunch(self, bunch_legs):
        """Return copy and bunch the qind for one or multiple legs

        Parameters
        ----------
        bunch : list of {True, False}
            one entry for each leg, whether the leg should be bunched.

        See also
        --------
        sort_legcharge: public API calling this function.
        """
        cp = self.copy(deep=False)
        # lists for each leg:
        new_to_old_idx = [None]*cp.rank     # the `idx` returned by cp.legs[li].bunch()
        map_qindex = [None]*cp.rank         # array mapping old qindex to new qindex, such that
        # new_leg.qind[m_qindex[i]] == old_leg.qind[i]  # (except the second column entry)
        bunch_qindex = [None]*cp.rank       # bool array wheter the *new* qind was bunched
        for li, bunch in enumerate(bunch_legs):
            idx, new_leg = cp.legs[li].bunch()
            cp.legs[li] = new_leg
            new_to_old_idx[li] = idx
            # generate entries in map_qindex and bunch_qdindex
            idx = np.append(idx, [self.shape[li]])
            m_qindex = []
            bunch_qindex[li] = b_qindex = np.empty(idx.shape, dtype=np.bool_)
            for inew in xrange(len(idx)-1):
                old_blocks = idx[inew+1] - idx[inew]
                m_qindex.append([inew]*old_blocks)
                b_qindex[inew] = (old_blocks > 1)
            map_qindex[li] = np.concatenate(m_qindex, axis=0)

        # now map _data and _qdata
        bunched_blocks = {}     # new qindices -> index in new _data
        new_data = []
        new_qdata = []
        for old_block, old_qindices in itertools.izip(self._data, self._qdata):
            new_qindices = tuple([m[qi] for m, qi in itertools.izip(map_qindex, old_qindices)])
            bunch = any([b[qi] for b, qi in itertools.izip(bunch_qindex, new_qindices)])
            if bunch:
                if new_qindices not in bunched_blocks:
                    # create enlarged block
                    bunched_blocks[new_qindices] = len(new_data)
                    # cp has new legs and thus gives the new shape
                    new_block = np.zeros(cp._get_block_shape(new_qindices), dtype=cp.dtype)
                    new_data.append(new_block)
                    new_qdata.append(new_qindices)
                else:
                    new_block = new_data[bunched_blocks[new_qindices]]
                # figure out where to insert the in the new bunched_blocks
                old_slbeg = [l.qind[qi, 0] for l, qi in itertools.izip(self.legs, old_qindices)]
                new_slbeg = [l.qind[qi, 0] for l, qi in itertools.izip(cp.legs, new_qindices)]
                slbeg = [(o-n) for o, n in itertools.izip(old_slbeg, new_slbeg)]
                sl = [slice(beg, beg+l) for beg, l in itertools.izip(slbeg, old_block.shape)]
                # insert the old block into larger new block
                new_block[tuple(sl)] = old_block
            else:
                # just copy the old block
                new_data.append(old_block.copy())
                new_qdata.append(new_qindices)
        cp._data = new_data
        cp._qdata = np.array(new_qdata, dtype=QDTYPE)
        cp._qsorted = False
        return cp

    def _perm_qind(self, p_qind, leg):
        """Apply a permutation `p_qind` of the qindices in leg `leg` to _qdata. In place."""
        # entry ``b`` of of old old._qdata[:, leg] refers to old ``old.legs[leg][b]``.
        # since new ``new.legs[leg][i] == old.legs[leg][p_qind[i]]``,
        # we have new ``new.legs[leg][reverse_sort_perm(p_qind)[b]] == old.legs[leg][b]``
        # thus we replace an entry `b` in ``_qdata[:, leg]``with reverse_sort_perm(q_ind)[b].
        p_qind_r = reverse_sort_perm(p_qind)
        self._qdata[:, leg] = p_qind_r[self._qdata[:, leg]]  # equivalent to
        # self._qdata[:, leg] = [p_qind_r[i] for i in self._qdata[:, leg]]
        self._qdata_sorted = False

    def _pre_indexing(self, inds):
        """check if `inds` are valid indices for ``self[inds]`` and replaces Ellipsis by slices.

        Returns
        -------
        only_integer : bool
            whether all of `inds` are (convertible to) np.intp
        inds : tuple, len=self.rank
            `inds`, where ``Ellipsis`` is replaced by the correct number of slice(None).
        """
        if type(inds) != tuple:  # for rank 1
            inds = tuple(inds)
        if len(inds) < self.rank:
            inds = inds + (Ellipsis, )
        if any([(i is Ellipsis) for i in inds]):
            fill = tuple([slice(None)] * (self.rank - len(inds)+1))
            e = inds.index(Ellipsis)
            inds = inds[:e] + fill + inds[e+1:]
        if len(inds) > self.rank:
            raise IndexError("too many indices for Array")
        # do we have only integer entries in `inds`?
        try:
            np.array(inds, dtype=np.intp)
        except:
            return False, inds
        else:
            return True, inds

    def _advanced_getitem(self, inds, calc_map_qind=False, permute=True):
        """calculate self[inds] for non-integer `inds`.

        This function is called by self.__getitem__(inds).
        and from _advanced_setitem_npc with ``calc_map_qind=True``.

        Parameters
        ----------
        inds : tuple
            indices for the different axes, as returned by :meth:`_pre_indexing`
        calc_map_qind :
            whether to calculate and return the additional `map_qind` and `axes` tuple

        Returns
        -------
        map_qind_part2self : function
            Only returned if `calc_map_qind` is True.
            This function takes qindices from `res` as arguments
            and returns ``(qindices, block_mask)`` such that
            ``res._get_block(part_qindices) = self._get_block(qindices)[block_mask]``.
            permutation are ignored for this.
        permutations : list((int, 1D array(int)))
            Only returned if `calc_map_qind` is True.
            Collects (axes, permutation) applied to `res` *after* `take_slice` and `iproject`.
        res : :class:`Array`
            an copy with the data ``self[inds]``.
        """
        # non-integer inds -> slicing / projection
        slice_inds = []  # arguments for `take_slice`
        slice_axes = []
        project_masks = []  # arguments for `iproject`
        project_axes = []
        permutations = []  # [axis, mask] for all axes for which we need to call `permute`
        for a, i in enumerate(inds):
            if isinstance(i, slice):
                if i != slice(None):
                    m = np.zeros(self.shape[a], dtype=np.bool_)
                    m[i] = True
                    project_masks.append(m)
                    project_axes.append(a)
                    if i.step is not None and i.step < 0:
                        permutations.append((a,
                                             np.arange(np.count_nonzero(m), dtype=np.intp)[::-1]))
            else:
                try:
                    iter(i)
                except:  # not iterable: single index
                    slice_inds.append(int(i))
                    slice_axes.append(a)
                else:  # iterable
                    i = np.asarray(i)
                    project_masks.append(i)
                    project_axes.append(a)
                    if i.dtype != np.bool_:  # should be integer indexing
                        perm = np.argsort(i)  # check if maks is sorted
                        if np.any(perm != np.arange(len(perm))):
                            # np.argsort(i) gives the reverse permutation, so reverse it again.
                            # In that way, we get the permuation within the projected indices.
                            permutations.append((a, reverse_sort_perm(perm)))
        res = self.take_slice(slice_inds, slice_axes)
        res_axes = np.cumsum([(a not in slice_axes) for a in xrange(self.rank)]) - 1
        p_map_qinds, p_masks = res.iproject(project_masks, [res_axes[p] for p in project_axes])
        permutations = [(res_axes[a], p) for a, p in permutations]
        if permute:
            for a, perm in permutations:
                res = res.permute(perm, a)
        if not calc_map_qind:
            return res
        part2self = self._advanced_getitem_map_qind(inds, slice_axes, slice_inds,
                                                    project_axes, p_map_qinds, p_masks, res_axes)
        return part2self, permutations, res

    def _advanced_getitem_map_qind(self, inds, slice_axes, slice_inds,
                                   project_axes, p_map_qinds, p_masks, res_axes):
        """generate a function mapping from qindices of `self[inds]` back to qindices of self

        This function is called only by `_advanced_getitem(calc_map_qind=True)`
        to obtain the function `map_qind_part2self`,
        which in turn in needed in `_advanced_setitem_npc` for ``self[inds] = other``.
        This function returns a function `part2self`, see doc string in the source for details.
        Note: the function ignores permutations introduced by `inds` - they are handled separately.
        """
        map_qinds = [None] * self.rank
        map_blocks = [None] * self.rank
        for a, i in zip(slice_axes, slice_inds):
            qi, within_block = self.legs[a].get_qindex(inds[a])
            map_qinds[a] = qi
            map_blocks[a] = within_block
        for a, m_qind in zip(project_axes, p_map_qinds):
            map_qinds[a] = np.nonzero(m_qind >= 0)[0]  # revert m_qind
        # keep_axes = neither in slice_axes nor in project_axes
        keep_axes = [a for a, i in enumerate(map_qinds) if i is None]
        not_slice_axes = sorted(project_axes + keep_axes)
        bsizes = [l._get_block_sizes() for l in self.legs]

        def part2self(part_qindices):
            """given `part_qindices` of ``res = self[inds]``,
            return (`qindices`, `block_mask`) such that
            ``res._get_block(part_qindices) == self._get_block(qindices)``.
            """
            qindices = map_qinds[:]  # copy
            block_mask = map_blocks[:]  # copy
            for a in keep_axes:
                qindices[a] = qi = part_qindices[res_axes[a]]
                block_mask[a] = np.arange(bsizes[a][qi], dtype=np.intp)
            for a, bmask in zip(project_axes, p_masks):
                old_qi = part_qindices[res_axes[a]]
                qindices[a] = map_qinds[a][old_qi]
                block_mask[a] = bmask[old_qi]
            # advanced indexing in numpy is tricky ^_^
            # np.ix_ can't handle integer entries reducing the dimension.
            # we have to call it only on the entries with arrays
            ix_block_mask = np.ix_(*[block_mask[a] for a in not_slice_axes])
            # and put the result back into block_mask
            for a, bm in zip(not_slice_axes, ix_block_mask):
                block_mask[a] = bm
            return qindices, tuple(block_mask)

        return part2self

    def _advanced_setitem_npc(self, inds, other):
        """self[inds] = other for non-integer `inds` and :class:`Array` `other`.
        This function is called by self.__setitem__(inds, other)."""
        map_part2self, permutations, self_part = self._advanced_getitem(inds,
                                                                        calc_map_qind=True,
                                                                        permute=False)
        # permuations are ignored by map_part2self.
        # instead of figuring out permuations in self, apply the *reversed* permutations ot other
        for ax, perm in permutations:
            other = other.permute(reverse_sort_perm(perm), ax)
        # now test compatibility of self_part with `other`
        if self_part.rank != other.rank:
            raise IndexError("wrong number of indices")
        for pl, ol in zip(self_part.legs, other.legs):
            pl.test_contractible(ol.conj())
        if np.any(self_part.qtotal != other.qtotal):
            raise ValueError("wrong charge for assinging self[inds] = other")
        # note: a block exists in self_part, if and only if its extended version exists in self.
        # by definition, non-existent blocks in `other` are zero.
        # instead of checking which blocks are non-existent,
        # we first set self[inds] completely to zero
        for p_qindices in self_part._qdata:
            qindices, block_mask = map_part2self(p_qindices)
            block = self._get_block(qindices)
            block[block_mask] = 0.  # overwrite data in self
        # now we copy blocks from other
        for o_block, o_qindices in zip(other._data, other._qdata):
            qindices, block_mask = map_part2self(o_qindices)
            block = self._get_block(qindices, insert=True)
            block[block_mask] = o_block  # overwrite data in self
        self.ipurge_zeros(0.)  # remove blocks identically zero

    def _combine_legs_worker(self, combine_legs, non_combined_legs, new_axes, pipes):
        """the main work of combine_legs: create a copy and reshape the data blocks.

        Assumes standard form of parameters.

        Parameters
        ----------
        combine_legs : list(1D np.array)
            axes of self which are collected into pipes.
        non_combined_legs : 1D array
            axes of self which are not collected into pipes
        new_axes : 1D array
            the axes of the pipes in the new array. Ascending.
        pipes : list of :class:`LegPipe`
            all the correct output pipes, already generated.

        Returns
        -------
        res : :class:`Array`
            copy of self with combined legs
        """
        legs = [self.legs[i] for i in non_combined_legs]
        for na, p in zip(new_axes, pipes):  # not reversed
            legs.insert(na, p)
        non_new_axes = [i for i in range(len(legs)) if i not in new_axes]
        non_new_axes = np.array(non_new_axes, dtype=np.intp)  # for index tricks

        res = self.copy(deep=False)  # TODO: deep?
        res.legs = legs
        res._set_shape
        res.labels = {}
        # map `self._qdata[:, combine_leg]` to `pipe.q_map` indices for each new pipe
        qmap_inds = [p._map_incoming_qind(self._qdata[:, cl])
                     for p, cl in zip(pipes, combine_legs)]

        # get new qdata
        qdata = np.empty((self.stored_blocks, res.rank), dtype=self._qdata.dtype)
        qdata[:, non_new_axes] = self._qdata[:, non_combined_legs]
        for na, p, qmap_ind in zip(new_axes, pipes, qmap_inds):
            np.take(p.q_map[:, -1],  # last column of q_map maps to qindex of the pipe
                    qmap_ind,
                    out=qdata[:, na])  # write the result directly into qdata
        # now we have probably many duplicate rows in qdata,
        # since for the pipes many `qmap_ind` map to the same `qindex`
        # find unique entries by sorting qdata
        sort = np.lexsort(qdata)
        qdata_s = qdata[sort]
        qmap_inds = [qm[sort] for qm in qmap_inds]

        diffs = charges._find_row_differences(qdata_s)  # including the first and last row

        # now the hard part: map data
        old_data = self._data
        data = []
        # get slices for the old blocks
        # the slices for the new
        slices = [slice(None)]*res.rank  # for selecting the slices in the new blocks
        # iterate over all different
        for beg, end in itertools.izip(diffs[:-1], diffs[1:]):
            qindices = qdata_s[beg]
            new_block = np.zeros(res._get_block_shape(qindices), dtype=res.dtype)
            data.append(new_block)
            # copy blocks
            for old_i in sort[beg:end]:
                for na, qmi in zip(new_axes, qmap_inds):
                    slices[na] = qm[sort]
                sl = tuple(slices)
                new_block_view = new_block[sl]
                # reshape block while copying
                new_block_view[:] = old_data[old_i].reshape(new_block_view.shape)
        res._qdata = qdata_s[diffs[:-1]]
        res._qdata_sorted = True
        res._data = data
        return res


# functions ====================================================================

def zeros(*args, **kwargs):
    """create a npc array full of zeros (with no _data).

    This is just a wrapper around ``Array(...)``,
    detailed documentation can be found in the class doc-string of :class:`Array`."""
    return Array(*args, **kwargs)
