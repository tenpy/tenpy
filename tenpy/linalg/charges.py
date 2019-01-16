r"""Basic definitions of a charge.

Contains implementation of classes
:class:`ChargeInfo`,
:class:`LegCharge` and
:class:`LegPipe`.

.. note ::
    The contents of this module are imported in :mod:`~tenpy.linalg.np_conserved`,
    so you usually don't need to import this module in your application.

A detailed introduction to `np_conserved` can be found in :doc:`/intro_npc`.
"""
# Copyright 2018 TeNPy Developers

import numpy as np
import copy
import bisect
import warnings

from ..tools.misc import lexsort, inverse_permutation
from ..tools.string import vert_join
from ..tools.optimization import optimize, OptimizationFlag

__all__ = ['ChargeInfo', 'LegCharge', 'LegPipe', 'QTYPE']

QTYPE = np.int_  # numpy dtype for the charges
"""Numpy data type for the charges."""


class ChargeInfo(object):
    """Meta-data about the charge of a tensor.

    Saves info about the nature of the charge of a tensor.
    Provides :meth:`make_valid` for taking modulo `m`.

    (This class is implemented in :mod:`tenpy.linalg.charges` but also imported in
    :mod:`tenpy.linalg.np_conserved` for convenience.)

    Parameters
    ----------
    mod : iterable of QTYPE
        The len gives the number of charges, `qnumber`.
        For each charge one entry `m`: the charge is conserved modulo `m`.
        Defaults to trivial, i.e., no charge.
    names : list of str
        Descriptive names for the charges.  Defaults to ``['']*qnumber``.

    Attributes
    ----------
    qnumber
    mod
    names : list of strings
        A descriptive name for each of the charges.  May have '' entries.
    _mask_mod1 : 1D array bool
        mask ``(mod == 1)``, to speed up `make_valid`
    _mod_masked : 1D array QTYPE
        Equivalent to ``self.mod[self._maks_mod1]``

    Notes
    -----
    Instances of this class can (should) be shared between different `LegCharge` and `Array`'s.
    """

    def __init__(self, mod=[], names=None):
        mod = np.array(mod, dtype=QTYPE)
        self._mask = np.not_equal(mod, 1)  # where we need to take modulo in :meth:`make_valid`
        self._mod_masked = mod[self._mask].copy()  # only where mod != 1
        if names is None:
            names = [''] * self.qnumber
        self.names = [str(n) for n in names]
        self.test_sanity()  # checks for invalid arguments

    @classmethod
    def add(cls, chinfos):
        """Create a :class:`ChargeInfo` combining multiple charges.

        Parameters
        ----------
        chinfos : iterable of :class:`ChargeInfo`
            ChargeInfo instances to be combined into a single one (in the given order).

        Returns
        -------
        chinfo : :class:`ChargeInfo`
            ChargeInfo combining all the given charges.
        """
        charges = [ci.mod for ci in chinfos]
        names = sum([ci.names for ci in chinfos], [])
        return cls(np.concatenate(charges), names)

    @classmethod
    def drop(cls, chinfo, charge=None):
        """Remove a charge from a :class:`ChargeInfo`.

        Parameters
        ----------
        chinfo : :class:`ChargeInfo`
            The ChargeInfo from where to drop/remove a charge.
        charge : int | str
            Number or `name` of the charge (within `chinfo`) which is to be dropped.
            ``None`` means dropping all charges.

        Returns
        -------
        chinfo : :class:`ChargeInfo`
            ChargeInfo where the specified charge is dropped.
        """
        if charge is None:
            return cls()  # trivial charge
        if isinstance(charge, str):
            charge = chinfo.names.index(charge)
        names = list(chinfo.names)
        names.pop(charge)
        return cls(np.delete(chinfo.mod, charge), names)

    @classmethod
    def change(cls, chinfo, charge, new_qmod, new_name=''):
        """Change the `qmod` of a given charge.

        Parameters
        ----------
        chinfo : :class:`ChargeInfo`
            The ChargeInfo for which `qmod` of `charge` should be changed.
        new_qmod : int
            The new `qmod` to be set.
        new_name : str
            The new name of the charge.

        Returns
        -------
        chinfo : :class:`ChargeInfo`
            ChargeInfo where `qmod` of the specified charge was changed.
        """
        if isinstance(charge, str):
            charge = chinfo.names.index(charge)
        names = list(chinfo.names)
        names[charge] = new_name
        mod = chinfo.mod.copy()
        mod[charge] = new_qmod
        return cls(mod, names)

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
        """The number of charges."""
        return len(self._mask)

    @property
    def mod(self):
        """Modulo how much each of the charges is taken.
        1 for a U(1) charge, i.e., mod 1 -> mod infinity.
        """
        res = np.ones(self.qnumber, dtype=QTYPE)
        res[self._mask] = self._mod_masked
        return res

    def make_valid(self, charges=None):
        """Take charges modulo self.mod.

        Parameters
        ----------
        charges : array_like or None
            1D or 2D array of charges, last dimension `self.qnumber`
            None defaults to np.zeros(qnumber).

        Returns
        -------
        charges :
            A copy of `charges` taken modulo `mod`, but with ``x % 1 := x``
        """
        if charges is None:
            return np.zeros((self.qnumber, ), dtype=QTYPE)
        charges = np.array(charges, dtype=QTYPE)
        charges[..., self._mask] = np.mod(charges[..., self._mask], self._mod_masked)
        return charges

    def check_valid(self, charges):
        r"""Check, if `charges` has all entries as expected from self.mod.

        Parameters
        ----------
        charges : 2D ndarray QTYPE_t
            Charge values to be checked.

        Returns
        -------
        res : bool
            True, if all 0 <= charges <= self.mod (wherever self.mod != 1)
        """
        charges = np.asarray(charges, dtype=QTYPE)[..., self._mask]
        return np.all(np.logical_and(0 <= charges, charges < self._mod_masked))

    def __repr__(self):
        """Full string representation."""
        return "ChargeInfo({0!s}, {1!s})".format(list(self.mod), self.names)

    def __eq__(self, other):
        """Compare self.mod and self.names for equality, ignore missing names."""
        if self is other:
            return True
        if not np.all(self.mod == other.mod):
            return False
        for l, r in zip(self.names, other.names):
            if r != l and l != '' and r != '':
                return False
        return True

    def __ne__(self, other):
        r"""Define `self != other` as `not (self == other)`"""
        return not self.__eq__(other)

    def __getstate__(self):
        """Allow to pickle and copy."""
        return (self.qnumber, self.mod, self.names)

    def __setstate__(self, state):
        """Allow to pickle and copy."""
        qnumber, mod, names = state
        self._mask = np.not_equal(mod, 1)  # where we need to take modulo in :meth:`make_valid`
        self._mod_masked = mod[self._mask].copy()  # only where mod != 1
        self.names = names


class LegCharge(object):
    r"""Save the charge data associated to a leg of a tensor.

    This class is more or less a wrapper around a 2D numpy array `charges` and a 1D array `slices`.
    See :doc:`/intro_npc` for more details.

    (This class is implemented in :mod:`tenpy.linalg.charges` but also imported in
    :mod:`tenpy.linalg.np_conserved` for convenience.)

    Parameters
    ----------
    chargeinfo : :class:`ChargeInfo`
        The nature of the charge.
    slices: 1D array_like, len(block_number+1)
        A block with 'qindex' ``qi`` correspondes to the leg indices in
        ``slice(slices[qi], slices[qi+1])``.
    charges : 2D array_like, shape(block_number, chargeinfo.qnumber)
        ``charges[qi]`` gives the charges for a block with 'qindex' ``qi``.
    qconj : {+1, -1}
        A flag telling whether the charge points inwards (+1, default) or outwards (-1).

    Attributes
    ----------
    ind_len
    block_number
    chinfo : :class:`ChargeInfo` instance
        The nature of the charge. Can be shared between LegCharges.
    slices : ndarray (block_number+1)
        A block with 'qindex' ``qi`` correspondes to the leg indices in
        ``slice(self.slices[qi], self.slices[qi+1])``. See :meth:`get_slice`.
    charges : ndarray (block_number, chinfo.qnumber)
        ``charges[qi]`` gives the charges for a block with 'qindex' ``qi``.
        Note: the sign might be changed by `qconj`. See also :meth:`get_charge`.
    qconj : {-1, 1}
        A flag telling whether the charge points inwards (+1) or outwards (-1).
        Whenever charges are added, they should be multiplied with their `qconj` value.
    sorted : bool
        Whether the charges are guaranteed to be sorted.
    bunched : bool
        Whether the charges are guaranteed to be bunched.

    Notes
    -----
    Instances of this class can be shared between different `npc.Array`.
    Thus, functions changing ``self.slices`` or ``self.charges`` *must* always make copies.
    Further they *must* set `sorted` and `bunched` to ``False`` (if they might not preserve them).
    """

    def __init__(self, chargeinfo, slices, charges, qconj=1):
        self.chinfo = chargeinfo
        self.slices = np.array(slices, dtype=np.intp)
        self.charges = np.array(charges, dtype=QTYPE)
        self.qconj = int(qconj)
        self.sorted = False
        self.bunched = False
        self.test_sanity()

    @classmethod
    def from_trivial(cls, ind_len, chargeinfo=None, qconj=1):
        """Create trivial (qnumber=0) LegCharge for given len of indices `ind_len`."""
        if chargeinfo is None:
            chargeinfo = ChargeInfo()
            charges = [[]]
        else:
            charges = [[0] * chargeinfo.qnumber]
        return cls(chargeinfo, [0, ind_len], charges, qconj)

    @classmethod
    def from_qflat(cls, chargeinfo, qflat, qconj=1):
        """Create a LegCharge from qflat form.

        Does *neither* bunch *nor* sort. We recommend to sort (and bunch) afterwards,
        if you expect that tensors using the LegCharge have entries at all positions compatible
        with the charges.

        Parameters
        ----------
        chargeinfo : :class:`ChargeInfo`
            The nature of the charge.
        qflat : array_like (ind_len, `qnumber`)
            `qnumber` charges for each index of the leg on entry.
        qconj : {-1, 1}
            A flag telling whether the charge points inwards (+1) or outwards (-1).

        See also
        --------
        :meth:`sort` : sorts by charges
        :meth:`bunch` : bunches contiguous blocks of the same charge.
        """
        qflat = np.asarray(qflat, dtype=QTYPE)
        if qflat.ndim == 1 and chargeinfo.qnumber == 1:
            # accept also 1D arrays, if the qnumber is 1
            qflat = qflat.reshape(-1, 1)
        ind_len, qnum = qflat.shape
        if qnum != chargeinfo.qnumber:
            raise ValueError("qflat with second dimension != qnumber")
        res = cls(chargeinfo, np.arange(ind_len + 1), qflat, qconj)
        res.sorted = res.is_sorted()
        res.bunched = res.is_bunched()
        return res

    @classmethod
    def from_qind(cls, chargeinfo, slices, charges, qconj=1):
        """Just a wrapper around self.__init__(), see class doc-string for parameters.

        See also
        --------
        sort : sorts by charges
        block : blocks by charges
        """
        res = cls(chargeinfo, slices, charges, qconj)
        res.sorted = res.is_sorted()
        res.bunched = res.is_bunched()
        return res

    @classmethod
    def from_qdict(cls, chargeinfo, qdict, qconj=1):
        """Create a LegCharge from qdict form.

        Parameters
        ----------
        chargeinfo : :class:`ChargeInfo`
            The nature of the charge.
        qdict : dict
            A dictionary mapping a tuple of charges to slices.
        """
        slices = np.array([(sl.start, sl.stop) for sl in qdict.values()], np.intp)
        charges = np.array(list(qdict.keys()), dtype=QTYPE).reshape((-1, chargeinfo.qnumber))
        sort = np.argsort(slices[:, 0])  # sort by slice start
        slices = slices[sort, :]
        charges = charges[sort, :]
        if np.any(slices[:-1, 1] != slices[1:, 0]):
            raise ValueError("The slices are not contiguous.\n" + str(slices))
        slices = np.append(slices[:, 0], [slices[-1, 1]])
        res = cls(chargeinfo, slices, charges, qconj)
        res.sorted = True
        res.bunched = res.is_bunched()
        return res

    @classmethod
    def from_add_charge(cls, legs, chargeinfo=None):
        """Add the (independent) charges of two or more legs to get larger `qnumber`.

        Parameters
        ----------
        legs : iterable of :class:`LegCharge`
            The legs for which the charges are to be combined/added.
        chargeinfo : :class:`ChargeInfo`
            The ChargeInfo for all charges; create new if ``None``.

        Returns
        -------
        combined : :class:`LegCharge`
            A LegCharge with the charges of both legs. Is neither sorted nor bunched!
        """
        legs = list(legs)
        chinfo = ChargeInfo.add([leg.chinfo for leg in legs])
        if chargeinfo is not None:
            assert chinfo == chargeinfo
            chinfo = chargeinfo
        ind_len = legs[0].ind_len
        qconj = legs[0].qconj
        if any([ind_len != leg.ind_len for leg in legs]):
            raise ValueError("different length")
        if any([qconj != leg.qconj for leg in legs]):
            raise ValueError("different qconj")
        qflat = np.empty([ind_len, chinfo.qnumber], dtype=QTYPE)
        i0 = 0
        for leg in legs:
            i1 = i0 + leg.chinfo.qnumber
            qflat[:, i0:i1] = leg.to_qflat()
            i0 = i1
        return cls.from_qflat(chinfo, qflat, qconj)

    @classmethod
    def from_drop_charge(cls, leg, charge=None, chargeinfo=None):
        """Remove a charge from a LegCharge.

        Parameters
        ----------
        leg : :class:`LegCharge`
            The leg from which to drop/remove a charge.
        charge : int | str
            Number or `name` of the charge (within `chinfo`) which is to be dropped.
            ``None`` means dropping all charges.
        chargeinfo : :class:`ChargeInfo`
            The ChargeInfo with `charge` dropped; create new if ``None``.

        Returns
        -------
        dropped : :class:`LegCharge`
            A LegCharge with the specified charge dropped. Is neither sorted nor bunched!
        """
        if charge is None:
            return cls.from_trivial(leg.ind_len, chargeinfo, leg.qconj)
        chinfo = ChargeInfo.drop(leg.chinfo, charge)
        if chargeinfo is not None:
            assert chinfo == chargeinfo
            chinfo = chargeinfo
        if isinstance(charge, str):
            charge = chinfo.names.index(charge)
        return cls.from_qflat(chinfo, np.delete(leg.to_qflat(), charge, 1), leg.qconj)

    @classmethod
    def from_change_charge(cls, leg, charge, new_qmod, new_name='', chargeinfo=None):
        """Remove a charge from a LegCharge.

        Parameters
        ----------
        leg : :class:`LegCharge`
            The leg from which to drop/remove a charge.
        charge : int | str
            Number or `name` of the charge (within `chinfo`) for which `mod` is to be changed.
        new_qmod : int
            The new `mod` to be set for `charge` in the :class:`ChargeInfo`.
        new_name : str
            The new name for `charge`.
        chargeinfo : :class:`ChargeInfo`
            The ChargeInfo with `charge` changed; create new if ``None``.

        Returns
        -------
        leg : :class:`LegCharge`
            A LegCharge with the specified charge changed. Is neither sorted nor bunched!
        """
        chinfo = ChargeInfo.change(leg.chinfo, charge, new_qmod, new_name)
        if chargeinfo is not None:
            assert chinfo == chargeinfo
            chinfo = chargeinfo
        charges = chinfo.make_valid(leg.charges)
        return cls.from_qind(chinfo, leg.slices, charges, leg.qconj)

    def test_sanity(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
        if optimize(OptimizationFlag.skip_arg_checks):
            return
        sl = self.slices
        ch = self.charges
        if sl.shape != (self.block_number + 1, ):
            raise ValueError("wrong len of `slices`")
        if sl[0] != 0:
            raise ValueError("slices does not start with 0")
        if ch.shape[1] != self.chinfo.qnumber:
            raise ValueError("shape of `charges` incompatible with qnumber")
        if not self.chinfo.check_valid(ch):
            raise ValueError("charges invalid for " + str(self.chinfo) + "\n" + str(self))
        if self.qconj not in [-1, 1]:
            raise ValueError("qconj has invalid value != +-1 :" + repr(self.qconj))

    @property
    def ind_len(self):
        """The number of indices for this leg."""
        return self.slices[-1]

    @property
    def block_number(self):
        """The number of blocks, i.e., a 'qindex' for this leg is in ``range(block_number)``."""
        return self.charges.shape[0]

    def conj(self):
        """Return a (shallow) copy with opposite ``self.qconj``."""
        res = copy.copy(self)  # shallow
        res.qconj *= -1
        return res

    def to_qflat(self):
        """Return charges in `qflat` form."""
        qflat = np.empty((self.ind_len, self.chinfo.qnumber), dtype=QTYPE)
        for start, stop, ch in zip(self.slices[:-1], self.slices[1:], self.charges):
            qflat[slice(start, stop)] = ch
        return qflat

    def to_qdict(self):
        """Return charges in `qdict` form. Raises ValueError, if not blocked."""
        res = dict()
        for start, stop, ch in zip(self.slices[:-1], self.slices[1:], self.charges):
            res[tuple(ch)] = slice(start, stop)
        if len(res) < self.block_number:  # ensures self is blocked
            raise ValueError("can't convert qflat to qdict for non-blocked LegCharge")
        return res

    def is_blocked(self):
        """Returns whether self is blocked, i.e. qindex map 1:1 to charge values."""
        if self.sorted and self.bunched:
            return True
        s = {tuple(c) for c in self.charges}  # a set has unique elements
        return (len(s) == self.block_number)

    def is_sorted(self):
        """Returns whether `self.charges` is sorted lexiographically."""
        if self.chinfo.qnumber == 0:
            return True
        res = lexsort(self.charges.T)
        return np.all(res == np.arange(len(res)))

    def is_bunched(self):
        """Checks whether :meth:`bunch` would change something."""
        return len(_find_row_differences(self.charges)) == self.block_number + 1

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
        - the `slices` are equal
        - the `charges` are the same up to *opposite* signs ``qconj``::

                self.charges * self.qconj = - other.charges * other.qconj

        In general, there could also be a change of the total charge, see :doc:`/intro_npc`
        This special case is not considered here - instead use
        :meth:`~tenpy.linalg.np_conserved.gauge_total_charge`,
        if a change of the charge is desired.

        If you are sure that the legs should be contractable,
        check whether the charges are actually valid
        or whether ``self`` and ``other`` are blocked or should be sorted.

        See also
        --------
        test_equal :
            ``self.test_contractible(other)`` just performs ``self.test_equal(other.conj())``.

        """
        if optimize(OptimizationFlag.skip_arg_checks):
            return
        self.test_equal(other.conj())

    def test_equal(self, other):
        """Test if charges are *equal* including `qconj`.

        Check that all of the following conditions are met:

        - the ``ChargeInfo`` is equal
        - the `slices` are equal
        - the `charges` are the same up to the signs ``qconj``::

                self.charges * self.qconj = other.charges * other.qconj

        See also
        --------
        test_contractible :
            ``self.test_equal(other)`` is equivalent to ``self.test_contractible(other.conj())``.
        """
        if optimize(OptimizationFlag.skip_arg_checks):
            return
        if self.chinfo != other.chinfo:
            raise ValueError(''.join(
                ["incompatible ChargeInfo\n",
                 str(self.chinfo),
                 str(other.chinfo)]))
        if self.charges is other.charges and self.qconj == other.qconj and \
                (self.slices is other.slices or np.all(self.slices == other.slices)):
            return  # optimize: don't need to check all charges explicitly
        if not np.array_equal(self.slices, other.slices) or \
                not np.array_equal(self.charges * self.qconj, other.charges * other.qconj):
            raise ValueError("incompatible LegCharge\n" + vert_join(
                ["self\n" + str(self), "other\n" + str(other)], delim=' | '))

    def get_slice(self, qindex):
        """Return slice selecting the block for a given `qindex`."""
        return slice(self.slices[qindex], self.slices[qindex + 1])

    def get_qindex(self, flat_index):
        """Find qindex containing a flat index.

        Given a flat index, to find the corresponding entry in an Array, we need to determine the
        block it is saved in. For example, if ``qind[:, 2] = [[0, 3], [3, 7], [7, 12]]``,
        the flat index ``5`` corresponds to the second entry, ``qindex = 1`` (since 5 is in [3:7]),
        and the index within the block would be ``5-3 =2``.

        Parameters
        ----------
        flat_index : int
            A flat index of the leg. Negative index counts from behind.

        Returns
        -------
        qindex : int
            The qindex, i.e. the index of the block containing `flat_index`.
        index_within_block : int
            The index of `flat_index` within the block given by `qindex`.
        """
        if flat_index < 0:
            flat_index += self.ind_len
            if flat_index < 0:
                raise IndexError("flat index {0:d} too negative for leg with ind_len {1:d}".format(
                    flat_index - self.ind_len, self.ind_len))
        elif flat_index > self.ind_len:
            raise IndexError("flat index {0:d} too large for leg with ind_len {1:d}".format(
                flat_index, self.ind_len))
        qind = bisect.bisect(self.slices, flat_index) - 1
        return qind, flat_index - self.slices[qind]

    def get_charge(self, qindex):
        """Return charge ``self.charges[qindex] * self.qconj`` for a given `qindex`."""
        return self.charges[qindex] * self.qconj

    def sort(self, bunch=True):
        """Return a copy of `self` sorted by charges (but maybe not bunched).

        If bunch=True, the returned copy is completely blocked by charge.

        Parameters
        ----------
        bunch : bool
            Whether `self.bunch` is called after sorting.
            If True, the leg is guaranteed to be fully blocked by charge.

        Returns
        -------
        perm_qind : array (self.block_len,)
            The permutation of the qindices (before bunching) used for the sorting.
            To obtain the flat permuation such that
            ``sorted_array[..., :] = unsorted_array[..., perm_flat]``, use
            ``perm_flat = unsorted_leg.perm_flat_from_perm_qind(perm_qind)``
        sorted_copy : :class:`LegCharge`
            A shallow copy of self, with new qind sorted (and thus blocked if bunch) by charges.

        See also
        --------
        bunch : enlarge blocks for contiguous qind of the same charges.
        np.take : can apply `perm_flat` to a given axis
        inverse_permutation : returns inverse of a permutation
        """
        if self.sorted and ((not bunch) or self.bunched):  # nothing to do
            return np.arange(self.block_number, dtype=np.intp), self
        perm_qind = lexsort(self.charges.T)
        cp = copy.copy(self)
        cp.charges = self.charges[perm_qind, :]
        block_sizes = self._get_block_sizes()
        cp._set_block_sizes(block_sizes[perm_qind])
        cp.sorted = True
        # finally bunch: re-ordering can have brought together equal charges
        if bunch:
            _, cp = cp.bunch()
        else:
            cp.bunched = False
        return perm_qind, cp

    def bunch(self):
        """Return a copy with bunched self.charges: form blocks for contiguous equal charges.

        Returns
        -------
        idx : 1D array
            ``idx[:-1]`` are the indices of the old qind which are kept,
            ``idx[-1] = old_block_number``.
        cp : :class:`LegCharge`
            A new LegCharge with the same charges at given indices of the leg,
            but (possibly) shorter ``self.charges`` and ``self.slices``.

        See also
        --------
        sort : sorts by charges, thus enforcing complete blocking in combination with bunch"""
        if self.bunched:  # nothing to do
            return np.arange(self.block_number + 1, dtype=np.intp), self
        cp = copy.copy(self)
        idx = _find_row_differences(self.charges)
        cp.charges = cp.charges[idx[:-1]]  # avanced indexing -> copy
        cp.slices = cp.slices[idx]
        cp.bunched = True
        return idx, cp

    def project(self, mask):
        """Return copy keeping only the indices specified by `mask`.

        Parameters
        ----------
        mask : 1D array(bool)
            Whether to keep of the indices.

        Returns
        -------
        map_qind : 1D array
            Map of qindices, such that ``qind_new = map_qind[qind_old]``,
            and ``map_qind[qind_old] = -1`` for qindices projected out.
        block_masks : 1D array
            The bool mask for each of the *remaining* blocks.
        projected_copy : :class:`LegCharge`
            Copy of self with the qind projected by `mask`.
        """
        mask = np.asarray(mask, dtype=np.bool_)
        cp = copy.copy(self)
        block_masks = [mask[b:e] for b, e in self._slice_start_stop()]
        new_block_lens = [np.sum(bm) for bm in block_masks]
        keep = np.nonzero(new_block_lens)[0]
        block_masks = [block_masks[i] for i in keep]
        cp.charges = cp.charges[keep]
        map_qind = -np.ones(self.block_number, np.int_)
        map_qind[keep] = np.arange(len(keep))
        cp._set_block_sizes(np.array(new_block_lens)[keep])
        cp.bunched = self.is_blocked()  # no, it's not `is_bunched`
        return map_qind, block_masks, cp

    def extend(self, extra):
        """Return a new :class:`LegCharge`, which extends self with futher charges.

        This is needed to formally increase the dimension of an Array.

        Parameters
        ----------
        extra : :class:`LegCharge` | int
            By what to extend, i.e. the charges to be appended to `self`.
            An int stands for extending the length of the array by a single new block of that size
            and zero charges.

        Returns
        -------
        extended_leg : :class:`LegCharge`
            Copy of `self` extended by the charge blocks of the `extra` leg.
        """
        if not isinstance(extra, LegCharge):
            extra = LegCharge.from_trivial(extra, self.chinfo, self.qconj)
        bn = self.block_number
        new_slices = np.zeros(bn + extra.block_number + 1, np.intp)
        new_slices[:bn + 1] = self.slices
        new_slices[bn:] = extra.slices + self.ind_len
        new_charges = np.zeros((bn + extra.block_number, self.chinfo.qnumber), dtype=QTYPE)
        new_charges[:bn] = self.charges
        if self.qconj == extra.qconj:
            new_charges[bn:] = extra.charges
        else:
            new_charges[bn:] = - extra.charges
        return LegCharge(self.chinfo, new_slices, new_charges, qconj=self.qconj)

    def charge_sectors(self):
        """Return unique rows of self.charges.

        Returns
        -------
        charges : 2D array
            Rows are the rows of self.charges lexsorted and without duplicates.
        """
        charges = self.charges.copy()
        if not self.sorted:
            charges = charges[np.lexsort(self.charges.T), :]
        charges = charges[_find_row_differences(charges)[:-1], :]
        return charges

    def __str__(self):
        """Return a string of nicely formatted slices & charges."""
        qconj = " {0:+d}\n".format(self.qconj)
        slices = '\n'.join([str(s) for s in self.slices])
        return qconj + vert_join([slices, str(self.charges)], delim=' ')

    def __repr__(self):
        """Full string representation."""
        return "LegCharge({0!r}, qconj={1:+d},\n{2!r}, {3!r})".format(
            self.chinfo, self.qconj, self.slices, self.charges)

    def _set_block_sizes(self, block_sizes):
        """Set self.slices from an list of the block-sizes."""
        self.slices = np.append([0], np.cumsum(block_sizes)).astype(np.intp, copy=False)

    def _get_block_sizes(self):
        """Return block sizes."""
        return self.slices[1:] - self.slices[:-1]

    def _slice_start_stop(self):
        """Yield (start, stop) for each qindex."""
        return zip(self.slices[:-1], self.slices[1:])

    def perm_flat_from_perm_qind(self, perm_qind):
        """Convert a permutation of qind (acting on self) into a flat permutation."""
        begend = np.stack([self.slices[:-1], self.slices[1:]], axis=0).T
        res = [np.arange(b, e) for b, e in begend[perm_qind]]
        return np.concatenate(res)

    def perm_qind_from_perm_flat(self, perm_flat):
        """Convert flat permutation into qind permutation.

        Parameters
        ----------
        perm_flat : 1D array
            A permutation acting on self, which doesn't mix the blocks of qind.

        Returns
        -------
        perm_qind : 1D array
            The permutation of self.qind described by perm_flat.

        Raises
        ------
        ValueError
            If perm_flat mixes blocks of different qindex.
        """
        perm_flat = np.asarray(perm_flat)
        perm_qind = perm_flat[self.slices[:-1]]
        # check if perm_qind indeed resembles the permutation
        if np.any(perm_flat != self.perm_flat_from_perm_qind(perm_qind)):
            raise ValueError("Permutation mixes qind")
        return perm_qind

    def __getstate__(self):
        """Allow to pickle and copy."""
        return (self.ind_len, self.block_number, self.chinfo, self.slices, self.charges,
                self.qconj, self.sorted, self.bunched)

    def __setstate__(self, state):
        """Allow to pickle and copy."""
        ind_len, block_number, chinfo, slices, charges, qconj, sorted, bunched = state
        self.chinfo = chinfo
        self.slices = slices
        self.charges = charges
        self.qconj = qconj
        self.sorted = sorted
        self.bunched = bunched


class LegPipe(LegCharge):
    r"""A `LegPipe` combines multiple legs of a tensor to one.

    Often, it is necessary to "combine" multiple legs into one:
    for example to perfom a SVD, the tensor needs to be viewed as a matrix.

    This class does exactly this job: it combines multiple LegCharges ('incoming legs')
    into one 'pipe' (*the* 'outgoing leg').
    The pipe itself is a :class:`LegCharge`, with indices running from 0 to the product of the
    individual legs' `ind_len`, corresponding to all possible combinations of input leg indices.

    (This class is implemented in :mod:`tenpy.linalg.charges` but also imported in
    :mod:`tenpy.linalg.np_conserved` for convenience.)

    Parameters
    ----------
    legs : list of :class:`LegCharge`
        The legs which are to be combined.
    qconj : {+1, -1}
        A flag telling whether the charge of the *resulting* pipe points inwards
        (+1, default) or outwards (-1).
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
        The original legs, which were combined in the pipe.
    subshape : tuple of int
        `ind_len` for each of the incoming legs.
    subqshape : tuple of int
        `block_number` for each of the incoming legs.
    q_map:  2D array
        Shape (`block_number`, 3 + `nlegs`). Rows: ``[ b_j, b_{j+1}, I_s, i_1, ..., i_{nlegs}]``,
        See Notes below for details.
    q_map_slices : list of views onto q_map
        Defined such that ``q_map_slices[I_s] == q_map[(q_map[:, 2] == I_s)]``.
    _perm : 1D array
        A permutation such that ``q_map[_perm, 3:]`` is sorted by `i_l`.
    _strides : 1D array
        Strides for mapping incoming qindices `i_l` to the index of of ``q_map[_perm, :]``.

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
    ``[b_j, b_{j+1}, I_s, i_1, . . . , i_{nlegs}]``.
    Here, :math:`b_j:b_{j+1}` denotes the slice of this qindex combination *within*
    the total block `I_s`, i.e., ``b_j = a_j - self.slices[I_s]``.

    The rows of `q_map` are lex-sorted first by ``I_s``, then the ``i``.
    Each ``I_s`` will have multiple rows,
    and the order in which they are stored in `q_map` is the order the data is stored
    in the actual tensor, i.e., it might look like ::

        [ ...,
         [ b_j,     b_{j+1},  I_s,     i_1,    ..., i_{nlegs}   ],
         [ b_{j+1}, b_{j+2},  I_s,     i'_1,   ..., i'_{nlegs}  ],
         [ 0,       b_{j+3},  I_s + 1, i''_1,  ..., i''_{nlegs} ],
         [ b_{j+3}, b_{j+4},  I_s + 1, i'''_1, ..., i'''_{nlegs}],
         ...]


    The charge fusion rule is::

        self.charges[Qi]*self.qconj == sum([l.charges[qi_l]*l.qconj for l in self.legs])  mod qmod

    Here the qindex ``Qi`` of the pipe corresponds to qindices ``qi_l`` on the individual legs.
    """

    def __init__(self, legs, qconj=1, sort=True, bunch=True):
        chinfo = legs[0].chinfo
        # initialize LegCharge with trivial charges/slices; gets overwritten in _init_from_legs
        LegCharge.__init__(self, chinfo, [0, 1], [[0] * chinfo.qnumber], qconj)
        # additional attributes
        self.legs = legs = tuple(legs)
        self.subshape = tuple([l.ind_len for l in self.legs])
        self.subqshape = tuple([l.block_number for l in self.legs])
        self.q_map = None  # overwritten in _init_from_legs, but necessary for copies
        self.q_map_slices = None  # overwritten in _init_from_legs, but necessary for copies
        # the diffuclt part: calculate self.slices, self.charges, self.q_map and self.q_map_slices
        self._init_from_legs(sort, bunch)
        self.test_sanity()

    @property
    def nlegs(self):
        """The number of legs."""
        return len(self.subshape)

    def test_sanity(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
        if optimize(OptimizationFlag.skip_arg_checks):
            return
        LegCharge.test_sanity(self)
        if not hasattr(self, "subshape"):
            return  # omit further check during ``super(LegPipe, self).__init__``
        assert (all([l.chinfo == self.chinfo for l in self.legs]))
        assert (self.subshape == tuple([l.ind_len for l in self.legs]))
        assert (self.subqshape == tuple([l.block_number for l in self.legs]))

    def to_LegCharge(self):
        """Convert self to a LegCharge, discarding the information how to split the legs.
        Usually not needed, but called by functions, which are not implemented for a LegPipe."""
        return LegCharge(self.chinfo, self.slices, self.charges, self.qconj)

    def conj(self):
        """Return a shallow copy with opposite ``self.qconj``.

        Also conjugates each of the incoming legs."""
        res = LegCharge.conj(self)  # invert self.qconj
        res.legs = tuple([l.conj() for l in self.legs])
        return res

    def outer_conj(self):
        """Like :meth:`conj`, but don't change ``qconj`` for incoming legs."""
        res = copy.copy(self)  # shallow
        res.qconj = -1
        res.charges = self.chinfo.make_valid(-self.charges)
        return res

    def sort(self, *args, **kwargs):
        """Convert to LegCharge and call :meth:`LegCharge.sort`."""
        # could be implemented for a LegPipe, but who needs it?
        warnings.warn("Converting LegPipe to LegCharge for `sort`")
        res = self.to_LegCharge()
        return res.sort(*args, **kwargs)

    def bunch(self, *args, **kwargs):
        """Convert to LegCharge and call :meth:`LegCharge.bunch`."""
        # could be implemented for a LegPipe, but who needs it?
        warnings.warn("Converting LegPipe to LegCharge for `bunch`")
        res = self.to_LegCharge()
        return res.bunch(*args, **kwargs)

    def project(self, *args, **kwargs):
        """Convert self to LegCharge and call :meth:`LegCharge.project`.

        In general, this could be implemented for a LegPipe, but would make
        :meth:`~tenpy.linalg.np_conserved.Array.split_legs` more complicated, thus we keep it
        simple.  If you really want to project and split afterwards, use the following work-around,
        which is for example used in :class:`~tenpy.algorithms.exact_diagonalization`:

        1) Create the full pipe and save it separetely.
        2) Convert the Pipe to a Leg & project the array with it.
        3) [... do calculations ...]
        4) To split the 'projected pipe' of `A`, create and empty array `B` with the legs of A,
           but replace the projected leg by the full pipe. Set `A` as a slice of `B`.
           Finally split the pipe.
        """
        warnings.warn("Converting LegPipe to LegCharge for `project`")
        res = self.to_LegCharge()
        return res.project(*args, **kwargs)

    def map_incoming_flat(self, incoming_indices):
        """Map (flat) incoming indices to an index in the outgoing pipe.

        Parameters
        ----------
        incoming_indices : iterable of int
            One (flat) index on each of the incoming legs.

        Returns
        -------
        outgoing_index : int
            The index in the outgoing leg.
        """
        # need to calculate the `a_j` in the Notes of the doc-string of self.
        if len(incoming_indices) != self.nlegs:
            raise ValueError("wrong len of flat_ind_incoming")
        qind_in = np.empty((1, self.nlegs), dtype=np.intp)
        within_block_out = 0
        stride = 1
        for ax in range(self.nlegs - 1, -1, -1):  # reversed: C order within the block
            leg = self.legs[ax]
            qind, within_block = leg.get_qindex(incoming_indices[ax])
            qind_in[0, ax] = qind
            within_block_out += stride * within_block
            stride *= (leg.slices[qind + 1] - leg.slices[qind])
        j = self._map_incoming_qind(qind_in)[0]
        q_map = self.q_map[j, :]
        assert (q_map[1] - q_map[0] == stride)
        qind_out = q_map[2]  # I_s
        return self.slices[qind_out] + q_map[0] + within_block_out

    def __str__(self):
        """Fairly short debug output."""
        res_lines = [
            "LegPipe(shape {0!s}->{1:d}, ".format(
                self.subshape, self.ind_len), "    qconj {0}->{1:+1};".format(
                    '(' + ', '.join(['%+d' % l.qconj for l in self.legs]) + ')', self.qconj),
            "    block numbers {0!s}->{1:d})".format(self.subqshape, self.block_number),
            vert_join([str(l) for l in self.legs], delim=' | '), ')'
        ]
        return '\n'.join(res_lines)

    def __repr__(self):
        """Full string representation."""
        return "LegPipe({legs},\nqconj={qconj:+d}, sort={s!r}, bunch={b!r})".format(
            legs='[' + ',\n'.join([repr(l) for l in self.legs]) + ']',
            qconj=self.qconj,
            s=self.sorted,
            b=self.bunched)

    def _init_from_legs(self, sort=True, bunch=True):
        """Calculate ``self.qind``, ``self.q_map`` and ``self.q_map_slices`` from ``self.legs``.

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
        q_map = np.empty((nblocks, 3 + nlegs), dtype=np.intp)
        q_map[:, 3:] = grid.T  # transpose -> rows are possible combinations.
        # the block size for given (i1, i2, ...) is the product of ``legs._get_block_sizes()[il]``
        legbs = [l._get_block_sizes() for l in self.legs]
        # andvanced indexing:
        # ``grid[li]`` is a 1D array containing the qindex `q_li` of leg ``li`` for all blocks
        blocksizes = np.prod([lbs[gr] for lbs, gr in zip(legbs, grid)], axis=0)
        # q_map[:, :3] is initialized after sort/bunch.

        # calculate total charges
        charges = np.zeros((nblocks, qnumber), dtype=QTYPE)
        if qnumber > 0:
            # similar scheme as for the block sizes above, but now for 1D arrays of charges
            legcharges = [(self.qconj * l.qconj) * l.charges for l in self.legs]
            # ``legcharges[li]`` is a 2D array mapping `q_li` to the charges.
            # thus ``(legcharges[li])[grid[li], :]`` gives a 2D array of shape (nblocks, qnumber)
            charges = np.sum([lq[gr] for lq, gr in zip(legcharges, grid)], axis=0)
            # now, we have what we need according to the charge **fusion rule**
            # namely for qi=`leg qindices` and li=`legs`:
            # charges[(q1, q2,...)] == self.qconj * (l1.qind[q1]*l1.qconj +
            #                                        l2.qind[q2]*l2.qconj + ...)
            charges = self.chinfo.make_valid(charges)  # modulo qmod

        if sort:
            # sort by charge. Similar code as in :meth:`LegCharge.sort`,
            # but don't want to create a copy, nor is qind[:, 0] initialized yet.
            perm_qind = lexsort(charges.T)
            q_map = q_map[perm_qind]
            charges = charges[perm_qind]
            blocksizes = blocksizes[perm_qind]
            self._perm = inverse_permutation(perm_qind)
        else:
            self._perm = None
        self.charges = charges
        self.sorted = sort
        self._set_block_sizes(blocksizes)  # sets self.slices
        q_map[:, 0] = self.slices[:-1]
        q_map[:, 1] = self.slices[1:]

        if bunch:
            # call LegCharge.bunch(), which also calculates new blocksizes
            idx, bunched = LegCharge.bunch(self)
            self.charges = bunched.charges  # copy information back to self
            self.slices = bunched.slices
            # calculate q_map[:, 2], the qindices corresponding to the rows of q_map
            q_map_Qi = np.zeros(len(q_map), dtype=np.intp)
            q_map_Qi[idx[1:-1]] = 1  # not for the first entry => np.cumsum starts with 0
            q_map[:, 2] = q_map_Qi = np.cumsum(q_map_Qi)
        else:
            q_map[:, 2] = q_map_Qi = np.arange(len(q_map), dtype=np.intp)
            idx = np.arange(len(q_map) + 1, dtype=np.intp)
        # calculate the slices within blocks: subtract the start of each block
        q_map[:, :2] -= (self.slices[q_map_Qi])[:, np.newaxis]
        self.q_map = q_map  # finished

        # finally calculate q_map_slices
        self.q_map_slices = [q_map[i:j] for i, j in zip(idx[:-1], idx[1:])]
        # q_map_slices contains only views!

    def _map_incoming_qind(self, qind_incoming):
        """Map incoming qindices to indices of q_map.

        Needed for :meth:`~tenpy.linalg.np_conserved.Array.combine_legs`.

        Parameters
        ----------
        qind_incoming : 2D array
            Rows are qindices :math:`(i_1, i_2, ... i_{nlegs})` for incoming legs.

        Returns
        -------
        q_map_indices : 1D array
            For each row of `qind_incoming` an index `j` such that
            ``self.q_map[j, 3:] == qind_incoming[j]``.
        """
        assert (qind_incoming.shape[1] == self.nlegs)
        # calculate indices of q_map[_perm], which is sorted by :math:`i_1, i_2, ...`,
        # by using the appropriate strides
        inds_before_perm = np.sum(qind_incoming * self._strides[np.newaxis, :], axis=1)
        # permute them to indices in q_map
        if self._perm is None:
            return inds_before_perm  # no permutation necessary
        return self._perm[inds_before_perm]

    def __getstate__(self):
        """Allow to pickle and copy."""
        super_state = LegCharge.__getstate__(self)
        return (super_state, self.nlegs, self.legs, self.subshape, self.subqshape, self.q_map,
                self.q_map_slices, self._perm, self._strides)

    def __setstate__(self, state):
        """Allow to pickle and copy."""
        super_state, nlegs, legs, subshape, subqshape, q_map, q_map_slices, _perm, _strides = state
        self.legs = legs
        self.subshape = subshape
        self.subqshape = subqshape
        self.q_map = q_map
        self.q_map_slices = q_map_slices
        self._perm = _perm
        self._strides = _strides
        LegCharge.__setstate__(self, super_state)


def _find_row_differences(qflat):
    """Return indices where the rows of the 2D array `qflat` change.

    Parameters
    ----------
    qflat : 2D array
        The rows of this array are compared.

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
