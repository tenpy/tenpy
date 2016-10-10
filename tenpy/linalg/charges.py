r"""Basic definitions of a charge: classes :class:`ChargeInfo` and :class:`LegCharge`.

.. note ::
    The contents of this module are imported in :mod:`~tenpy.linalg.np_conserved`,
    so you usually don't need to import this module in your application.

A detailed introduction to np_conserved can be found in :doc:`np_conserved`.
"""

from __future__ import division

import numpy as np
import copy
import itertools
import bisect

"""the dtype of a single charge"""
QDTYPE = np.int_


class ChargeInfo(object):
    r"""Meta-data about the charge of a tensor.

    Saves info about the nature of the charge of a tensor.
    Provides :meth:`make_valid` for taking modulo `m`.

    Parameters
    ----------
    qmod : iterable of `QDTYPE`
        The len gives the number of charges, `qnumber`.
        For each charge one entry `m`: the charge is conserved modulo `m`.
        Defaults to trivial, i.e., no charge.
    names : list of str
        Descriptive names for the charges.  Defaults to ``['']*qnumber``.

    Attributes
    ----------
    qnumber
    mod : 1D array_like of ints
        The periodicity of the charges. One entry for each charge.
    names : list of strings
        A descriptive name for each of the charges.  May have '' entries.

    Notes
    -----
    Instances of this class can (and should be) shared between different `LegCharge`s and even
    `npc.Array`s
    """

    def __init__(self, mod=[], names=None):
        """see help(self)"""
        self.mod = np.array(mod, dtype=QDTYPE)
        self._mod_1 = np.equal(self.mod, 1)  # pre-convert for faster make_valid
        if names is None:
            names = [''] * self.qnumber
        self.names = [str(n) for n in names]
        self.test_sanity()  # checks for invalid arguments

    def test_sanity(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
        if self.mod.ndim != 1:
            raise ValueError("mod has wrong shape")
        assert np.all(self._mod_1 == np.equal(self.mod, 1))
        if np.any(self.mod <= 0):
            raise ValueError("mod should be > 0")
        if len(self.names) != self.qnumber:
            raise ValueError("names has incompatible length with mod")

    @property
    def qnumber(self):
        """the number of charges, also refered to as qnumber"""
        return len(self.mod)

    def make_valid(self, charges=None):
        r"""Take charges modulo self.mod.

        Parameters
        ----------
        charges : array_like or None
            1D or 2D array of charges, last dimension `self.qnumber`
            None defaults to np.zeros(qnumber).

        Returns
        -------
        charges :
            `charges` taken modulo self.mod, but with x % 1 := x
        """
        if charges is None:
            return np.zeros((self.qnumber,), dtype=QDTYPE)
        charges = np.asarray(charges, dtype=QDTYPE)
        return np.where(self._mod_1, charges, np.mod(charges, self.mod))

    def check_valid(self, charges):
        r"""Check, if `charges` has all entries as expected from self.mod.

        Returns
        -------
        Bool
            True, if all 0 <= charges <= self.mod (whenever self.mod != 1)
        """
        charges = np.asarray(charges, dtype=QDTYPE)
        return np.all(np.logical_or(self._mod_1, np.logical_and(0 <= charges, charges < self.mod)))

    def __repr__(self):
        """full string representation"""
        return "ChargeInfo({0!s}, {1!s})".format(list(self.mod), self.names)

    def __eq__(self, other):
        r"""compare self.mod and self.names for equality, ignoring missin names."""
        if self is other:
            return True
        if not self.mod == other.mod:
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

    In the end, this is a wrapper around a 2D numpy array qind.

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
    qind : np.array, shape(blocknumber, 2+qnumber)
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
        self.qind = np.array(qind, dtype=QDTYPE)
        self.qconj = qconj
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

        Parameters
        ----------
        chargeinfo : :class:`ChargeInfo`
            the nature of the charge
        qflat : list(charges)
            `qnumber` charges for each index of the leg on entry
        qconj : {-1, 1}
            A flag telling whether the charge points inwards (+1) or outwards (-1).

        See also
        --------
        sort : sorts by charges
        block : blocks by charges
        """
        qflat = np.asarray(qflat, dtype=QDTYPE)
        indices = _find_row_differences(qflat).reshape(-1, 1)
        qind = np.hstack((indices[:-1], indices[1:], qflat[indices[:-1, 0]]))
        res = cls(chargeinfo, qind, qconj)
        res.sorted = res.is_sorted()
        res.bunched = res.is_bunched()
        return res

    @classmethod
    def from_qind(cls, chargeinfo, qflat, qconj=1):
        """just a wrapper around self.__init__(), see class doc-string for parameters.

        See also
        --------
        sort : sorts by charges
        block : blocks by charges
        """
        res = cls(chargeinfo, qflat, qconj)
        res.sorted = res.is_sorted()
        res.bunched = res.is_bunched()
        return res

    @classmethod
    def from_qdict(cls, chargeinfo, qdict, qconj=1):
        """create a LegCharge from qdict form.

        """
        qind = [[sl.start, sl.stop] + list(ch) for (ch, sl) in qdict.iteritems()]
        qind = np.array(qind, dtype=QDTYPE)
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
            raise ValueError("qconj has invalid value != +-1 :" + str(self.qconj))

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
        qflat = np.empty((self.ind_len, self.chinfo.qnumber), dtype=QDTYPE)
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
        res = np.lexsort(self.qind[:, 2:].T)
        return np.all(res == np.arange(len(res)))

    def is_bunched(self):
        """returns whether there are contiguous blocks"""
        return len(_find_row_differences(self.qind[:, 2:])) == self.block_number + 1

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
                                 .format(flat_index-self.ind_len, self.ind_len))
        elif flat_index > self.ind_len:
            raise IndexError("flat index {0:d} too large for leg with ind_len {1:d}"
                             .format(flat_index, self.ind_len))
        block_begin = self.qind[:, 0]
        qind = bisect.bisect(block_begin, flat_index) - 1
        return qind, flat_index - block_begin[qind]

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
        perm_flat : array (ind_len,)
            the permutation of the indices in flat form.
            For a flat ndarray, ``sorted_array[..., :] = unsorted_array[..., perm]``.
        perm_qind : array (self.block_len,)
            the permutation of the qind (before bunching) used for the sorting.
        sorted_copy : :class:`LegCharge`
            a shallow copy of self, with new qind sorted (and thus blocked if bunch) by charges.

        See also
        --------
        bunch : enlarge blocks for contiguous qind of the same charges.
        np.take : can apply `perm_flat` to a given axis
        reverse_sort_perm : returns inverse of a permutation
        """
        perm_qind = np.lexsort(self.qind[:, 2:].T)
        cp = copy.copy(self)
        cp.qind = np.empty_like(self.qind)
        cp.qind[:, 2:] = self.qind[perm_qind, 2:]
        # figure out the re-ordered slice boundaries
        block_sizes = self._get_block_sizes()
        cp._set_qind_block_sizes(block_sizes[perm_qind])
        # finally bunch: re-ordering can have brought together equal charges
        if bunch:
            _, cp = cp.bunch()
        return _perm_flat_from_qind(perm_qind, self.qind), perm_qind, cp

    def bunch(self):
        """Return a copy with bunched self.qind: form blocks for contiguous equal charges.

        Returns
        -------
        idx : 1D array
            the indices of the old qind which are kept
        cp : :class:`LegCharge`
            a copy of self, which is bunched

        See also
        --------
        sort : sorts by charges, thus enforcing complete blocking in combination with bunch"""
        cp = copy.copy(self)
        idx = _find_row_differences(self.qind[:, 2:])[:-1]
        cp.qind = np.copy(cp.qind[idx])
        cp.qind[:-1, 1] = cp.qind[1:, 0]
        cp.qind[-1, 1] = self.ind_len
        return idx, cp

    def check_contractible(self, other):
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

        - the ChargeInfo is equal
        - the charge blocks are equal, i.e., ``qind[:, :2]`` are equal
        - the charges are the same up to the signs ``qconj``::

                self.qind[:, 2:] * self.qconj = other.qind[:, 2:] * other.qconj[:, 2:].

        In general, there could also be a change of the total charge, see :doc:`../IntroNpc`
        This special case is not considered here - instead use
        :meth:~tenpy.linalg.np_conserved.gauge_total_charge`, if a change of the charge is desired.

        If you are sure that the legs should be contractable,
        check whether it is necessary to use :meth:`ChargeInfo.make_valid`,
        or whether self and other are blocked or should be sorted.

        .. todo ::

            should we allow a `bunch` before?
        """
        if self.chinfo != other.chinfo:
            raise ValueError(''.join(["incompatible ChargeInfo\n", str(self.chinfo), str(
                other.chinfo)]))
        if self.qind is other.qind and self.qconj == -other.qconj:
            return  # optimize: don't need to check all charges explicitly
        if not np.array_equal(self.qind[:, :2], other.qind[:, :2]):
            raise ValueError(''.join(["incomatible charge blocks. qind self, other=\n", str(self),
                                      "\n", str(other)]))
        if not np.array_equal(self.qind[:, 2:] * self.qconj, other.qind[:, 2:] * (-other.qconj)):
            raise ValueError(''.join(["incompatible charges. qind:\n", str(self), "\n", str(other)
                                      ]))

    def project(self, mask):
        """Return copy keeping only the indices specified by `mask`.

        Parameters
        ----------
        mask : 1D array(bool)
            the indices which you want to keep

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
        map_qind = - np.ones(self.block_number, np.int_)
        map_qind[keep] = np.arange(len(keep))
        cp._set_qind_block_sizes(np.array(new_block_lens)[keep])
        return map_qind, block_masks, cp

    def __str__(self):
        """return a string of qind"""
        return str(self.qind)

    def __repr__(self):
        """full string representation"""
        return "LegCharge({0!r}, {1!r}, {2:d})".format(self.chinfo, self.qind, self.qconj)

    def _set_qind_block_sizes(self, block_sizes):
        """Set self.qind[:, :2] from an list of the blocksizes."""
        block_sizes = np.asarray(block_sizes, np.int_)
        self.qind[:, 1] = np.add.accumulate(block_sizes)
        self.qind[0, 0] = 0
        self.qind[1:, 0] = self.qind[:-1, 1]

    def _get_block_sizes(self):
        """return block sizes"""
        return (self.qind[:, 1] - self.qind[:, 0])


class LegPipe(object):
    """A LegPipe combines multiple legs of a tensor to one.

    .. todo ::
        implement. Doesn't it make sense to derive this from LegCharge?!?"""
    def __init__(self):
        raise NotImplementedError()


# ===== functions =====

def reverse_sort_perm(perm):
    """reverse sorting indices.

    Sort functions (as :meth:`LegCharge.sort`) return a (1D) permutation `perm` array,
    such that ``sorted_array = old_array[perm]``.
    This function reverses the permutation `perm`,
    such that ``old_array = sorted_array[reverse_sort_perm(perm)]``.

    .. todo ::
        should we move this to another file? maybe tools/math (also move the test!)
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
    if len(qflat) == 1:
        return
    diff = np.ones(qflat.shape[0] + 1, dtype=np.bool_)
    diff[1:-1] = np.any(qflat[1:] != qflat[:-1], axis=1)
    return np.nonzero(diff)[0]  # get the indices of True-values


def _perm_flat_from_qind(perm_qind, qind):
    """translate a permutation of qind into a flat permutation"""
    return np.concatenate([np.arange(b, e) for (b, e) in qind[perm_qind, :2]])


def _perm_qind_from_perm_flat(perm_flat, qind):
    """translate flat permutaiton into qind permutation.

    Parameters
    ----------
    perm_flat : 1D array
        a permutation, which doesn't mix the blocks of qind
    qind : 2D array
        a LegCharge.qind for which the permutation should be obtained

    Returns
    -------
    perm_qind : 1D array
        the permutation of qind described by perm_flat.

    Raises
    ------
    ValueError
        If perm_flat mixes blocks of different qind
    """
    perm_flat = np.asarray(perm_flat)
    perm_qind = perm_flat[qind[:, 0]]
    # check if perm_qind indeed resembles the permutation
    if np.any(perm_flat != _perm_flat_from_qind(perm_qind, qind)):
        raise ValueError("Permutation mixes qind")
    return perm_qind
