r"""Basic definitions of a charge.

Contains implementation of classes
:class:`ChargeInfo`,
:class:`LegCharge` and
:class:`LegPipe`.

.. note ::
    The contents of this module are imported in :mod:`~tenpy.linalg.np_conserved`,
    so you usually don't need to import this module in your application.

A detailed introduction to `np_conserved` can be found in :doc:`../IntroNpc`.
"""

from __future__ import division

import numpy as np
import copy
import itertools
import bisect
import warnings


class ChargeInfo(object):
    """Meta-data about the charge of a tensor.

    Saves info about the nature of the charge of a tensor.
    Provides :meth:`make_valid` for taking modulo `m`.

    Parameters
    ----------
    mod : iterable of `qtype`
        The len gives the number of charges, `qnumber`.
        For each charge one entry `m`: the charge is conserved modulo `m`.
        Defaults to trivial, i.e., no charge.
    names : list of str
        Descriptive names for the charges.  Defaults to ``['']*qnumber``.
    qtype : type
        the data type for the numpy arrays. Defaults to `np.int_`.

    Attributes
    ----------
    qnumber
    mod
    qtype
    names : list of strings
        A descriptive name for each of the charges.  May have '' entries.
    _mask_mod1 : 1D array bool
        mask ``(mod == 1)``, to speed up `make_valid`
    _mod_masked : 1D arry qtype
        equivalent to ``self.mod[self._maks_mod1]``

    Notes
    -----
    Instances of this class can (should) be shared between different `LegCharge` and `Array`'s.
    """

    def __init__(self, mod=[], names=None, qtype=np.int_):
        """see help(self)"""
        mod = np.array(mod, dtype=qtype)
        self._mask = np.not_equal(mod, 1)  # where we need to take modulo in :meth:`make_valid`
        self._mod_masked = mod[self._mask].copy()  # only where mod != 1
        if names is None:
            names = [''] * self.qnumber
        self.names = [str(n) for n in names]
        self.test_sanity()  # checks for invalid arguments

    def test_sanity(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
        if self._mod_masked.ndim != 1:
            raise ValueError("mod has wrong shape")
        if np.any(self._mod_masked <= 0):
            raise ValueError("mod should be > 0")
        if len(self.names) != self.qnumber:
            raise ValueError("names has incompatible length with mod")

    @property
    def qnumber(self):
        """the number of charges, also refered to as qnumber."""
        return len(self._mask)

    @property
    def mod(self):
        """modulo how much each of the charges is taken."""
        res = np.ones(self.qnumber, dtype=self.qtype)
        res[self._mask] = self._mod_masked
        return res

    @property
    def qtype(self):
        """the data type of the charges"""
        return self._mod_masked.dtype

    def make_valid(self, charges=None):
        """Take charges modulo self.mod.

        Acts in-place, if charges is an array.

        Parameters
        ----------
        charges : array_like or None
            1D or 2D array of charges, last dimension `self.qnumber`
            None defaults to np.zeros(qnumber).

        Returns
        -------
        charges :
            `charges` taken modulo `mod`, but with ``x % 1 := x``
        """
        if charges is None:
            return np.zeros((self.qnumber, ), dtype=self.qtype)
        charges = np.array(charges, dtype=self.qtype)
        charges[..., self._mask] = np.mod(charges[..., self._mask], self._mod_masked)
        return charges

    def check_valid(self, charges):
        r"""Check, if `charges` has all entries as expected from self.mod.

        Returns
        -------
        res : bool
            True, if all 0 <= charges <= self.mod (wherever self.mod != 1)
        """
        charges = np.asarray(charges, dtype=self.qtype)[..., self._mask]
        return np.all(np.logical_and(0 <= charges, charges < self._mod_masked))

    def __repr__(self):
        """full string representation"""
        return "ChargeInfo({0!s}, {1!s})".format(list(self.mod), self.names)

    def __eq__(self, other):
        r"""compare self.mod and self.names for equality, ignoring missin names."""
        if self is other:
            return True
        if not np.all(self.mod == other.mod):
            return False
        for l, r in itertools.izip(self.names, other.names):
            if r != l and l != '' and r != '':
                return False
        return True

    def __ne__(self, other):
        r"""Define `self != other` as `not (self == other)`"""
        return not self.__eq__(other)


class LegCharge(object):
    r"""Save the charge data associated to a leg of a tensor.

    In the end, this class is more a wrapper around a 2D numpy array `qind`.
    See :doc:`../IntroNpc` for more details.

    Parameters
    ----------
    chargeinfo : :class:`ChargeInfo`
        the nature of the charge
    qind : 2D array_like
        first dimension = qindex,
        second dimension: first two entries define the block as `slice(beg, end)`,
        remaining entries = charge.
    qconj : {+1, -1}
        A flag telling whether the charge points inwards (+1, default) or outwards (-1).

    Attributes
    ----------
    ind_len
    block_number
    chinfo : :class:`ChargeInfo` instance
        the nature of the charge. Can be shared between LegCharges
    qind : np.array, shape(block_number, 2+qnumber)
        first dimension = qindex,
        second dimension: first two entries define the block as `slice(beg, end)`,
        remaining entries = charge.
    qconj : {-1, 1}
        A flag telling whether the charge points inwards (+1) or outwards (-1).
        When charges are added, they are multiplied with their qconj value.
    sorted : bool
        whether the charges are guaranteed to be sorted
    bunched : bool
        whether the charges are guaranteed to be bunched

    Notes
    -----
    Instances of this class can be shared between different `npc.Array`s.
    Thus, functions changing self.qind *must* always make copies;
    further they *must* set `sorted` and `bunched` to false, if not guaranteeing to preserve them.
    """

    def __init__(self, chargeinfo, qind, qconj=1):
        """see help(self)"""
        self.chinfo = chargeinfo
        self.qind = np.array(qind, dtype=chargeinfo.qtype)
        self.qconj = int(qconj)
        self.sorted = False
        self.bunched = False
        self.test_sanity()

    @classmethod
    def from_trivial(cls, ind_len, chargeinfo=None, qconj=1):
        """create trivial (qnumber=0) LegCharge for given len of indices `ind_len`"""
        if chargeinfo is None:
            chargeinfo = ChargeInfo()
        return cls(chargeinfo, [[0, ind_len]], qconj)

    @classmethod
    def from_qflat(cls, chargeinfo, qflat, qconj=1):
        """create a LegCharge from qflat form.

        Does *neither* bunch *nor* sort. We recommend to sort (and bunch) afterwards,
        if you expect that tensors using the LegCharge have entries at all positions compatible
        with the charges.

        Parameters
        ----------
        chargeinfo : :class:`ChargeInfo`
            the nature of the charge
        qflat : array_like (ind_len, `qnumber`)
            `qnumber` charges for each index of the leg on entry
        qconj : {-1, 1}
            A flag telling whether the charge points inwards (+1) or outwards (-1).

        See also
        --------
        :meth:`sort` : sorts by charges
        :meth:`bunch` : bunches contiguous blocks of the same charge.
        """
        qflat = np.asarray(qflat, dtype=chargeinfo.qtype)
        if qflat.ndim == 1 and chargeinfo.qnumber == 1:
            # accept also 1D arrays, if the qnumber is 1
            qflat = qflat.reshape(-1, 1)
        ind_len, qnum = qflat.shape
        if qnum != chargeinfo.qnumber:
            raise ValueError("qflat has wrong shape!")
        qind = np.empty((ind_len, 2 + qnum), dtype=chargeinfo.qtype)
        qind[:, 0] = np.arange(ind_len)
        qind[:, 1] = np.arange(1, ind_len + 1)
        qind[:, 2:] = chargeinfo.make_valid(qflat)
        res = cls(chargeinfo, qind, qconj)
        res.sorted = res.is_sorted()
        res.bunched = res.is_bunched()
        return res

    @classmethod
    def from_qind(cls, chargeinfo, qind, qconj=1):
        """just a wrapper around self.__init__(), see class doc-string for parameters.

        See also
        --------
        sort : sorts by charges
        block : blocks by charges
        """
        res = cls(chargeinfo, qind, qconj)
        res.sorted = res.is_sorted()
        res.bunched = res.is_bunched()
        return res

    @classmethod
    def from_qdict(cls, chargeinfo, qdict, qconj=1):
        """create a LegCharge from qdict form.

        """
        qind = [[sl.start, sl.stop] + list(ch) for (ch, sl) in qdict.iteritems()]
        qind = np.array(qind, dtype=chargeinfo.qtype)
        sort = np.argsort(qind[:, 0])  # sort by slice start
        qind = qind[sort, :]
        res = cls(chargeinfo, qind, qconj)
        res.sorted = True
        res.bunched = res.is_bunched()
        return res

    def test_sanity(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
        qind = self.qind
        if qind.shape[1] != 2 + self.chinfo.qnumber:
            raise ValueError("shape of qind incompatible with qnumber")
        if np.any(qind[:, 0] >= qind[:, 1]):
            raise ValueError("Invalid slice in qind: beg >= end:\n" + str(self))
        if np.any(qind[:-1, 1] != qind[1:, 0]):
            raise ValueError("The slices of qind are not contiguous.\n" + str(self))
        if not self.chinfo.check_valid(qind[:, 2:]):
            raise ValueError("qind charges invalid for " + str(self.chinfo) + "\n" + str(self))
        if self.qconj not in [-1, 1]:
            raise ValueError("qconj has invalid value != +-1 :" + repr(self.qconj))

    @property
    def ind_len(self):
        """the number of indices for this leg"""
        return self.qind[-1, 1]

    @property
    def block_number(self):
        """the number of blocks, i.e., a qindex is in ``range(block_number)``."""
        return self.qind.shape[0]

    def conj(self):
        """return a shallow copy with opposite ``self.qconj``"""
        res = copy.copy(self)  # shallow
        res.qconj *= -1
        return res

    def to_qflat(self):
        """return `self.qind` in `qdict` form"""
        qflat = np.empty((self.ind_len, self.chinfo.qnumber), dtype=self.chinfo.qtype)
        for qsec in self.qind:
            qflat[slice(qsec[0], qsec[1])] = qsec[2:]
        return qflat

    def to_qind(self):
        """return `self.qind`"""
        return self.qind

    def to_qdict(self):
        """return `self.qind` in `qdict` form. Raises ValueError, if not blocked."""
        res = dict([(tuple(qsec[2:]), slice(qsec[0], qsec[1])) for qsec in self.qind])
        if len(res) < self.block_number:  # ensures self is blocked
            raise ValueError("can't convert qflat to qdict for non-blocked LegCharge")
        return res

    def is_blocked(self):
        """returns whether self is blocked, i.e. qindex map 1:1 to charge values."""
        if self.sorted and self.bunched:
            return True
        s = {tuple(c) for c in self.qind[:, 2:]}  # a set has unique elements
        return (len(s) == self.block_number)

    def is_sorted(self):
        """returns whether the charge values in qind are sorted lexiographically"""
        if self.chinfo.qnumber == 0:
            return True
        res = np.lexsort(self.qind[:, 2:].T)
        return np.all(res == np.arange(len(res)))

    def is_bunched(self):
        """returns whether there are contiguous blocks"""
        return len(_find_row_differences(self.qind[:, 2:])) == self.block_number + 1

    def test_contractible(self, other):
        """Raises a ValueError if charges are incompatible for contraction with other.

        Parameters
        ----------
        other : :class:`LegCharge`
            The LegCharge of the other leg condsidered for contraction.

        Raises
        ------
        ValueError
            If the charges are incompatible for direct contraction.

        Notes
        -----
        This function checks that two legs are `ready` for contraction.
        This is the case, if all of the following conditions are met:

        - the ``ChargeInfo`` is equal
        - the charge blocks are equal, i.e., ``qind[:, :2]`` are equal
        - the charges are the same up to opposite signs ``qconj``::

                self.qind[:, 2:] * self.qconj = -other.qind[:, 2:] * other.qconj[:, 2:]

        In general, there could also be a change of the total charge, see :doc:`../IntroNpc`
        This special case is not considered here - instead use
        :meth:~tenpy.linalg.np_conserved.gauge_total_charge`, if a change of the charge is desired.

        If you are sure that the legs should be contractable,
        check whether it is necessary to use :meth:`ChargeInfo.make_valid`,
        or whether ``self`` and ``other`` are blocked or should be sorted.

        See also
        --------
        test_equal :
            ``self.test_contractible(other)`` is equivalent to ``self.test_equal(other.conj())``.

        """
        self.test_equal(other.conj())

    def test_equal(self, other):
        """test if charges are *equal* including `qconj`.

        Check that all of the following conditions are met:

        - the ``ChargeInfo`` is equal
        - the charge blocks are equal, i.e., ``qind[:, :2]`` are equal
        - the charges are the same up to the signs ``qconj``::

                self.qind[:, 2:] * self.qconj = other.qind[:, 2:] * other.qconj[:, 2:]

        See also
        --------
        test_contractible :
            ``self.test_equal(other)`` is equivalent to ``self.test_contractible(other.conj())``.
        """

        if self.chinfo != other.chinfo:
            raise ValueError(''.join(["incompatible ChargeInfo\n", str(self.chinfo), str(
                other.chinfo)]))
        if self.qind is other.qind and self.qconj == other.qconj:
            return  # optimize: don't need to check all charges explicitly
        if not np.array_equal(self.qind[:, :2], other.qind[:, :2]):
            raise ValueError("incomatible charge blocks. self.qind=\n{0!s}\nother.qind={1!s}"
                             .format(self, other))
        if not np.array_equal(self.qind[:, 2:] * self.qconj, other.qind[:, 2:] * other.qconj):
            raise ValueError("incompatible charges. qconj={0:+d}, {1:+d}, qind:\n{2!s}\n{3!s}"
                             .format(self.qconj, other.qconj, self, other))

    def get_slice(self, qindex):
        """return slice selecting the block for a given `qindex`"""
        return slice(*self.qind[qindex, :2])

    def get_qindex(self, flat_index):
        """find qindex containing a flat index.

        Given a flat index, to find the corresponding entry in an Array, we need to determine the
        block it is saved in. For example, if ``qind[:, 2] = [[0, 3], [3, 7], [7, 12]]``,
        the flat index ``5`` corresponds to the second entry, ``qindex = 1`` (since 5 is in [3:7]),
        and the index within the block would be ``5-3 =2``.

        Parameters
        ----------
        flat_index : int
            a flat index of the leg. Negative index counts from behind.

        Returns
        -------
        qindex : int
            the qindex, i.e. the index of the block containing `flat_index`
        index_within_block : int
            the index of `flat_index` within the block given by `qindex`.
        """
        if flat_index < 0:
            flat_index += self.ind_len
            if flat_index < 0:
                raise IndexError("flat index {0:d} too negative for leg with ind_len {1:d}"
                                 .format(flat_index - self.ind_len, self.ind_len))
        elif flat_index > self.ind_len:
            raise IndexError("flat index {0:d} too large for leg with ind_len {1:d}"
                             .format(flat_index, self.ind_len))
        block_begin = self.qind[:, 0]
        qind = bisect.bisect(block_begin, flat_index) - 1
        return qind, flat_index - block_begin[qind]

    def get_charge(self, qindex):
        """Return charge ``self.qind[qindex, 2:] * self.qconj`` of a given `qindex`."""
        return self.qind[qindex, 2:] * self.qconj

    def sort(self, bunch=True):
        """Return a copy of `self` sorted by charges (but maybe not bunched).

        If bunch=True, the returned copy is completely blocked by charge.

        Parameters
        ----------
        bunch : bool
            whether `self.bunch` is called after sorting.
            If True, the leg is guaranteed to be fully blocked by charge.

        Returns
        -------
        perm_qind : array (self.block_len,)
            the permutation of the qind (before bunching) used for the sorting.
            To obtain the flat permuation such that
            ``sorted_array[..., :] = unsorted_array[..., perm_flat]``, use
            ``perm_flat = unsorted_array.perm_flat_from_perm_qind(perm_qind)``
        sorted_copy : :class:`LegCharge`
            a shallow copy of self, with new qind sorted (and thus blocked if bunch) by charges.

        See also
        --------
        bunch : enlarge blocks for contiguous qind of the same charges.
        np.take : can apply `perm_flat` to a given axis
        reverse_sort_perm : returns inverse of a permutation
        """
        if self.sorted and ((not bunch) or self.bunched):  # nothing to do
            return np.arange(self.block_number, dtype=np.intp), self
        perm_qind = np.lexsort(self.qind[:, 2:].T)
        cp = copy.copy(self)
        cp.qind = self.qind[perm_qind, :]  # apply permutation. (advanced indexing -> copy)
        block_sizes = cp._get_block_sizes()  # uses ``qind[:, 1]-qind[:, 0]``,
        # which gives the *permuted* block sizes
        cp._set_qind_block_sizes(block_sizes)
        cp.sorted = True
        # finally bunch: re-ordering can have brought together equal charges
        if bunch:
            _, cp = cp.bunch()
        else:
            cp.bunched = False
        return perm_qind, cp

    def bunch(self):
        """Return a copy with bunched self.qind: form blocks for contiguous equal charges.

        Returns
        -------
        idx : 1D array
            the indices of the old qind which are kept
        cp : :class:`LegCharge`
            a new LegCharge with the same charges at given indices of the leg,
            but (possibly) shorter ``self.qind``.

        See also
        --------
        sort : sorts by charges, thus enforcing complete blocking in combination with bunch"""
        if self.bunched:  # nothing to do
            return np.arange(self.block_number, dtype=np.intp), self
        cp = copy.copy(self)
        idx = _find_row_differences(self.qind[:, 2:])[:-1]
        cp.qind = cp.qind[idx]  # avanced indexing -> copy
        cp.qind[:-1, 1] = cp.qind[1:, 0]
        cp.qind[-1, 1] = self.ind_len
        cp.bunched = True
        return idx, cp

    def project(self, mask):
        """Return copy keeping only the indices specified by `mask`.

        Parameters
        ----------
        mask : 1D array(bool)
            whether to keep of the indices

        Returns
        -------
        map_qind : 1D array
            map of qindices, such that ``qind_new = map_qind[qind_old]``,
            and ``map_qind[qind_old] = -1`` for qindices projected out.
        block_masks : 1D array
            the bool mask for each of the *remaining* blocks
        projected_copy : :class:`LegCharge`
            copy of self with the qind projected by `mask`
        """
        mask = np.asarray(mask, dtype=np.bool_)
        cp = copy.copy(self)
        cp.qind = cp.qind.copy()
        block_masks = [mask[qi[0]:qi[1]] for qi in cp.qind]
        new_block_lens = [np.sum(bm) for bm in block_masks]
        keep = np.nonzero(new_block_lens)[0]
        block_masks = [block_masks[i] for i in keep]
        cp.qind = cp.qind[keep]
        map_qind = -np.ones(self.block_number, np.int_)
        map_qind[keep] = np.arange(len(keep))
        cp._set_qind_block_sizes(np.array(new_block_lens)[keep])
        cp.bunched = self.is_blocked()  # no, it's not `is_bunched`
        return map_qind, block_masks, cp

    def __str__(self):
        """return a string of qind"""
        return str(self.qind)

    def __repr__(self):
        """full string representation"""
        return "LegCharge({0!r},\n{1!r}, {2:d})".format(self.chinfo, self.qind, self.qconj)

    def _set_qind_block_sizes(self, block_sizes):
        """Set self.qind[:, :2] from an list of the blocksizes."""
        block_sizes = np.asarray(block_sizes, dtype=self.chinfo.qtype)
        self.qind[:, 1] = np.cumsum(block_sizes)
        self.qind[0, 0] = 0
        self.qind[1:, 0] = self.qind[:-1, 1]

    def _get_block_sizes(self):
        """return block sizes"""
        return (self.qind[:, 1] - self.qind[:, 0])

    def perm_flat_from_perm_qind(self, perm_qind):
        """Convert a permutation of qind (acting on self) into a flat permutation."""
        return np.concatenate([np.arange(b, e) for (b, e) in self.qind[perm_qind, :2]])

    def perm_qind_from_perm_flat(self, perm_flat):
        """Convert flat permutation into qind permutation.

        Parameters
        ----------
        perm_flat : 1D array
            a permutation acting on self, which doesn't mix the blocks of qind.

        Returns
        -------
        perm_qind : 1D array
            the permutation of self.qind described by perm_flat.

        Raises
        ------
        ValueError
            If perm_flat mixes blocks of different qindex
        """
        perm_flat = np.asarray(perm_flat)
        perm_qind = perm_flat[self.qind[:, 0]]
        # check if perm_qind indeed resembles the permutation
        if np.any(perm_flat != self.perm_flat_from_perm_qind(perm_qind)):
            raise ValueError("Permutation mixes qind")
        return perm_qind


class LegPipe(LegCharge):
    r"""A `LegPipe` combines multiple legs of a tensor to one.

    Often, it is necessary to "combine" multiple legs into one:
    for example to perfom a SVD, the tensor needs to be viewed as a matrix.

    This class does exactly this job: it combines multiple LegCharges ('incoming legs')
    into one 'pipe' (*the* 'outgoing leg').
    The pipe itself is a :class:`LegCharge`, with indices running from 0 to the product of the
    individual legs' `ind_len`, corresponding to all possible combinations of input leg indices.

    Parameters
    ----------
    legs : list of :class:`LegCharge`
        the legs which are to be combined.
    qconj : {+1, -1}
        A flag telling whether the charge of the *resulting* pipe points inwards
        (+1, default) or outwards (-1).
    block : bool
        Wheter `self.sort` should be called at the end of initializition in order
        to ensure complete blocking.
    sort : bool
        Whether the outgoing pipe should be sorted. Defaults ``True``; recommended.
        Note: calling :meth:`sort` after initialization converts to a LegCharge.
    bunch : bool
        Whether the outgoing pipe should be bunched. Default ``True``; recommended.
        Note: calling :meth:`bunch` after initialization converts to a LegCharge.

    Attributes
    ----------
    nlegs
    legs : tuple of :class:`LegCharge`
        the original legs, which were combined in the pipe.
    subshape : tuple of int
        ind_len for each of the incoming legs
    subqshape : tuple of int
        block_number for each of the incoming legs
    q_map:  2D array
        shape (`block_number`, 2+`nlegs`+1). rows: ``[ m_j, m_{j+1}, i_1, ..., i_{nlegs}, I_s]``,
        see Notes below for details. lex-sorted by (I_s, i's), i.e. by colums [2:].
    q_map_slices : list of views onto q_map
        defined such that ``q_map_slices[I_s] == q_map[(q_map[:, -1] == I_s)]``
    _perm : 1D array
        a permutation such that ``q_map[_perm, :]`` is sorted by `i_l` (ignoring the `I_s`).
    _strides : 1D array
        strides for mapping incoming qindices `i_l` to the index of of ``q_map[_perm, :]``

    Methods
    -------
    :meth:`to_LegCharge`
        converts to a :class:`LegCharge`
    :meth:`test_sanity`
    :meth:`conj`
        flip ``qconj`` for all incoming legs and the outgoing leg.
    :meth:`outer_conj`
        flip the outgoing `qconj` and outgoing charges
    modify blocks
        (these convert to LegCharge)
        :meth:`sort`, :meth:`bunch`, :meth:`project`

    Notes
    -----
    For np.reshape, taking, for example,  :math:`i,j,... \rightarrow k` amounted to
    :math:`k = s_1*i + s_2*j + ...` for appropriate strides :math:`s_1,s_2`.

    In the charged case, however, we want to block :math:`k` by charge, so we must
    implicitly permute as well.  This reordering is encoded in `q_map`.

    Each qindex combination of the `nlegs` input legs :math:`(i_1, ..., i_{nlegs})`,
    will end up getting placed in some slice :math:`a_j:a_{j+1}` of the outgoing pipe.
    Within this slice, the data is simply reshaped in usual row-major fashion ('C'-order),
    i.e., with strides :math:`s_1 > s_2 > ...`.

    It will be a subslice of a new total block labeled by qindex :math:`I_s`.
    Because many charge combinations fuse to the same total charge,
    in general there will be many tuples :math:`(i_1, ..., i_{nlegs})` belonging to the same
    :math:`I_s`.  The rows of `q_map` are precisely the collections of
    ``[b_j, b_{j+1}, i_1, . . . , i_{nlegs}, I_s ]``,

    Here, :math:`b_j:b_{j+1}` denotes the slice of this qindex combination *within*
    the total block `I_s`, i.e., ``b_j = a_j - self.qind[I_s, 0]``.

    The rows of map_qind are lex-sorted first by ``I_s``, then the ``i``s.
    Each ``I_s`` will have multiple rows,
    and the order in which they are stored in `q_map` is the order the data is stored
    in the actual tensor, i.e., it might look like ::

        [ ...,
         [ b_j,     b_{j+1},  i_1,    ..., i_{nlegs},     I_s   ],
         [ b_{j+1}, b_{j+2},  i'_1,   ..., i'_{nlegs},    I_s   ],
         [ 0,       b_{j+3},  i''_1,  ..., i''_{nlegs},   I_s+1 ],
         [ b_{j+3}, b_{j+4},  i'''_1, ..., i''''_{nlegs}, I_s+1
         ...]


    The charge fusion rule is::

        self.qind[Qi]*self.qconj == sum([l.qind[qi_l] * l.qconj  for l in self.legs])  mod qmod

    Here the qindex ``Qi`` of the pipe corresponds to qindices ``qi_l`` on the individual legs.
    """

    def __init__(self, legs, qconj=1, sort=True, bunch=True):
        """see help(self)"""
        chinfo = legs[0].chinfo
        # initialize LegCharge with trivial qind, which gets overwritten in _init_from_legs
        super(LegPipe, self).__init__(chinfo, [[0, 1] + [0] * chinfo.qnumber], qconj)
        # additional attributes
        self.legs = legs = tuple(legs)
        self.subshape = tuple([l.ind_len for l in self.legs])
        self.subqshape = tuple([l.block_number for l in self.legs])
        # the diffuclt part: calculate self.qind, self.q_map and self.q_map_slices
        self._init_from_legs(sort, bunch)
        self.test_sanity()

    @property
    def nlegs(self):
        """the number of legs"""
        return len(self.subshape)

    def test_sanity(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
        super(LegPipe, self).test_sanity()
        if not hasattr(self, "subshape"):
            return  # omit further check during ``super(LegPipe, self).__init__``
        assert (all([l.chinfo == self.chinfo for l in self.legs]))
        assert (self.subshape == tuple([l.ind_len for l in self.legs]))
        assert (self.subqshape == tuple([l.block_number for l in self.legs]))

    def to_LegCharge(self):
        """convert self to a LegCharge, discarding the information how to split the legs.
        Usually not needed, but called by functions, which are not implemented for a LegPipe."""
        warnings.warn("Converting LegPipe to LegCharge")
        return LegCharge(self.chinfo, self.qind, self.qconj)

    def conj(self):
        """return a shallow copy with opposite ``self.qconj``.

        Also conjugates each of the incoming legs."""
        res = super(LegPipe, self).conj()  # invert self.conj
        res.legs = tuple([l.conj() for l in self.legs])
        return res

    def outer_conj(self):
        """like :meth:`conj`, but don't change ``qconj`` for incoming legs."""
        res = copy.copy(self)  # shallow
        res.qconj = -1
        res.qind = res.qind.copy()
        res.qind[:, 2:] = self.chinfo.make_valid(-self.qind[:, 2:])
        return res

    def sort(self, *args, **kwargs):
        """convert to LegCharge and call :meth:`LegCharge.sort`"""
        # could be implemented for a LegPipe, but who needs it?
        res = self.to_LegCharge()
        return res.sort(*args, **kwargs)

    def bunch(self, *args, **kwargs):
        """convert to LegCharge and call :meth:`LegCharge.bunch`"""
        # could be implemented for a LegPipe, but who needs it?
        res = self.to_LegCharge()
        return res.bunch(*args, **kwargs)

    def project(self, *args, **kwargs):
        """convert self to LegCharge and call :meth:`LegCharge.project`"""
        # could be implemented for a LegPipe, but who needs it?
        res = self.to_LegCharge()
        return res.project(*args, **kwargs)

    def __str__(self):
        """fairly short debug output"""
        res_lines = ["LegPipe(shape {0!s}->{1:d}, ".format(self.subshape, self.ind_len),
                     "qconj {0}->{1:+1};".format(
                         '(' + ', '.join(['%+d' % l.qconj for l in self.legs]) + ')', self.qconj),
                     "block numbers {0!s}->{1:d})".format(self.subqshape, self.block_number)]
        return '\n'.join(res_lines)

    def __repr__(self):
        """full string representation"""
        return "LegPipe({legs},\nqconj={qconj:+d}, sort={s!r}, bunch={b!r})".format(
            legs='[' + ',\n'.join([repr(l) for l in self.legs]) + ']',
            qconj=self.qconj,
            s=self.sorted,
            b=self.bunched)

    def _init_from_legs(self, sort=True, bunch=True):
        """calculate ``self.qind``, ``self.q_map`` and ``self.q_map_slices`` from ``self.legs``.

        `qind` is constructed to fullfill the charge fusion rule stated in the class doc-string.
        """
        # this function heavily uses numpys advanced indexing, for details see
        # `http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html`_
        # and the documentation of np.mgrid
        nlegs = self.nlegs
        qnumber = self.chinfo.qnumber
        qshape = self.subqshape

        # create a grid to select the multi-index sector
        grid = np.mgrid[[slice(0, l) for l in qshape]]
        # grid is an array with shape ``(nlegs,) + qshape``,
        # with grid[li, ...] = {np.arange(qshape[li]) increasing in the li-th direcion}
        # save the strides of grid, which is needed for :meth:`_map_incoming_qind`
        self._strides = np.array(grid.strides, np.intp)[1:] // grid.itemsize
        # collapse the different directions into one.
        grid = grid.reshape(nlegs, -1)  # *this* is the actual `reshaping`
        # *columns* of grid are now all possible cominations of qindices.

        nblocks = grid.shape[1]  # number of blocks in the pipe = np.product(qshape)
        # determine q_map -- it's essentially the grid.
        q_map = np.empty((nblocks, 2 + nlegs + 1), dtype=self.chinfo.qtype)
        q_map[:, 2:-1] = grid.T  # transpose -> rows are possible combinations.
        # the block size for given (i1, i2, ...) is the product of ``legs._get_block_sizes()[il]``
        legbs = [l._get_block_sizes() for l in self.legs]
        # andvanced indexing:
        # ``grid[li]`` is a 1D array containing the qindex `q_li` of leg ``li`` for all blocks
        blocksizes = np.prod([lbs[gr] for lbs, gr in itertools.izip(legbs, grid)], axis=0)
        # q_map[:, :2] and q_map[:, -1] are initialized after sort/bunch.

        # calculate total charges
        qind = np.zeros((nblocks, 2 + qnumber), dtype=self.chinfo.qtype)
        if qnumber > 0:
            # similar scheme as for the block sizes above, but now for 1D arrays of charges
            legcharges = [(self.qconj * l.qconj) * l.qind[:, 2:] for l in self.legs]
            # ``legcharges[li]`` is a 2D array mapping `q_li` to the charges.
            # thus ``(legcharges[li])[grid[li], :]`` gives a 2D array of shape (nblocks, qnumber)
            charges = np.sum([lq[gr] for lq, gr in itertools.izip(legcharges, grid)], axis=0)
            # now, we have what we need according to the charge **fusion rule**
            # namely for qi=`leg qindices` and li=`legs`:
            # charges[(q1, q2,...)] == self.qconj * (l1.qind[q1]*l1.qconj +
            #                                        l2.qind[q2]*l2.qconj + ...)
            qind[:, 2:] = self.chinfo.make_valid(charges)  # modulo qmod
        # qind[:, :2] is initialized after sorting

        if sort:
            # sort by charge. Similar code as in :meth:`LegCharge.sort`,
            # but don't want to create a copy, nor is qind[:, 0] initialized yet.
            perm_qind = np.lexsort(qind[:, 2:].T)
            q_map = q_map[perm_qind]
            qind = qind[perm_qind]
            blocksizes = blocksizes[perm_qind]
            self._perm = reverse_sort_perm(perm_qind)
        else:
            self._perm = None
        self.qind = qind
        self.sorted = sort
        self._set_qind_block_sizes(blocksizes)  # sets qind[:, :2]
        q_map[:, :2] = qind[:, :2]

        if bunch:
            # call LegCharge.bunch(), which also calculates new blocksizes
            idx, bunched = super(LegPipe, self).bunch()
            self.qind = bunched.qind  # copy qind back to self
            # calculate q_map[:, -1], the qindices corresponding to the rows of q_map
            q_map_Qi = np.zeros(len(q_map), dtype=q_map.dtype)
            q_map_Qi[idx[1:]] = 1  # not for the first entry => np.cumsum starts with 0
            q_map[:, -1] = q_map_Qi = np.cumsum(q_map_Qi)
        else:
            q_map[:, -1] = q_map_Qi = np.arange(len(q_map), dtype=q_map.dtype)
        # calculate the slices within blocks: subtract the start of each block
        q_map[:, :2] -= (self.qind[q_map_Qi, 0])[:, np.newaxis]
        self.q_map = q_map  # finished

        # finally calculate q_map_slices
        diffs = _find_row_differences(q_map[:, -1:])
        self.q_map_slices = [q_map[i:j] for i, j in itertools.izip(diffs[:-1], diffs[1:])]
        # q_map_slices contains only views!

    def _map_incoming_qind(self, qind_incoming):
        """map incoming qindices to indices of q_map.

        Needed for :meth:`~tenpy.linalg.np_conserved.Array.combine_legs`.

        Parameters
        ----------
        qind_incoming : 2D array
            rows are qindices :math:`(i_1, i_2, ... i_{nlegs})` for incoming legs

        Returns
        -------
        q_map_indices : 1D array
            for each row of `qind_incoming` an index `j` such that
            ``self.q_map[j, 2:-1] == qind_incoming[j]``.
        """
        assert (qind_incoming.shape[1] == self.nlegs)
        # calculate indices of q_map[_perm], which is sorted by :math:`i_1, i_2, ...`,
        # by using the appropriate strides
        inds_before_perm = np.sum(qind_incoming * self._strides[np.newaxis, :], axis=1)
        # permute them to indices in q_map
        if self._perm is None:
            return inds_before_perm  # no permutation necessary
        return self._perm[inds_before_perm]

# ===== functions =====


def reverse_sort_perm(perm):
    """reverse sorting indices.

    Sort functions (as :meth:`LegCharge.sort`) return a (1D) permutation `perm` array,
    such that ``sorted_array = old_array[perm]``.
    This function reverses the permutation `perm`,
    such that ``old_array = sorted_array[reverse_sort_perm(perm)]``.

    .. todo ::
        should we move this to another file? maybe tools/math (also move the test!)
        At least rename this to `reverse_perm`
    """
    return np.argsort(perm)


def _find_row_differences(qflat):
    """Return indices where the rows of the 2D array `qflat` change.

    Parameters
    ----------
    qflat : 2D array
        the rows of this array are compared.

    Returns
    -------
    diffs: 1D array
        The indices where rows change, including the first and last. Equivalent to:
        ``[0]+[i for i in range(1, len(qflat)) if np.any(qflat[i-1] != qflat[i])] + [len(qflat)]``
    """
    if qflat.shape[1] == 0:
        return np.array([0, qflat.shape[0]], dtype=np.intp)
    diff = np.ones(qflat.shape[0] + 1, dtype=np.bool_)
    diff[1:-1] = np.any(qflat[1:] != qflat[:-1], axis=1)
    return np.nonzero(diff)[0]  # get the indices of True-values
