"""Optimization of charges.py and np_conserved.py

This module is written in Cython, such that it can be compiled.
It implements some functions and classes with the same interface as np_conserved.py/charges.py.


``tenpy.linalg.__init__.py`` tries to import the compiled module and uses the
functions/classes defined here to overwrite those written in pure Python.
If the file could not be compiled, it just uses the non-compiled Cython version.
"""
# Copyright 2018 TeNPy Developers

import numpy as np
cimport numpy as np
cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, postincrement as inc  # TODO

import copy
import bisect
import warnings
import time  # TODO
import itertools


import scipy.linalg
from scipy.linalg import blas as BLAS  # python interface to BLAS
from scipy.linalg.cython_blas cimport dgemm, zgemm

from ..tools.misc import lexsort, inverse_permutation  # TODO: get rid of this?
from ..tools.string import vert_join  # TODO get rid of this?
from ..tools import optimization
from . import np_conserved

cdef int optimization_compare = optimization.OptimizationFlag.skip_arg_checks

__all__ = ['ChargeInfo', 'LegCharge', 'LegPipe', 'QTYPE', '_tensordot_worker']

np.import_array()

cdef inline np.ndarray _np_empty(np.PyArray_Dims dims, int type):
    return <np.ndarray>np.PyArray_EMPTY(dims.len, dims.ptr, type, 0 )

cdef inline np.ndarray _np_zeros(np.PyArray_Dims dims, int type):
    return <np.ndarray>np.PyArray_ZEROS(dims.len, dims.ptr, type, 0 )


QTYPE = np.int64             # numpy dtype for the charges
# QTYPE_t define in npc_helper.pxd
# intp_t defined in npc_helper.pxd

# ################################# #
# replacements for charges.py       #
# ################################# #

cdef class ChargeInfo(object):
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
    qnumber :
        The number of charges.
    mod :  ndarray[QTYPE,ndim=1]
        Modulo how much each of the charges is taken.
        1 for a U(1) charge, i.e., mod 1 -> mod infinity.
    names : list of strings
        A descriptive name for each of the charges.  May have '' entries.

    Notes
    -----
    Instances of this class can (should) be shared between different `LegCharge` and `Array`'s.
    """

    def __init__(ChargeInfo self, mod=[], names=None):
        mod = np.asarray(mod, dtype=QTYPE)
        self.qnumber = len(mod)
        self.mod = mod
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

    cpdef void test_sanity(ChargeInfo self) except *:
        """Sanity check. Raises ValueErrors, if something is wrong."""
        cdef int opt_level = optimization._level
        if opt_level >= optimization_compare:
            return
        if len(self.names) != self.qnumber:
            raise ValueError("names has incompatible length with mod")
        if np.any(self.mod < 0):
            raise ValueError("mod with negative entries???")

    cpdef np.ndarray make_valid(ChargeInfo self, charges=None):
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
        charges = np.asarray(charges, dtype=QTYPE)
        if charges.ndim == 1:
            assert(charges.shape[0] == self.qnumber)
            return self._make_valid_1D(charges)
        elif charges.ndim == 2:
            assert(charges.shape[1] == self.qnumber)
            return self._make_valid_2D(charges)
        raise ValueError("wrong dimension of charges " + str(charges))

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef np.ndarray _make_valid_1D(ChargeInfo self, np.ndarray charges):
        cdef np.ndarray res = np.empty_like(charges)
        cdef int j
        cdef QTYPE_t q
        for j in range(self.qnumber):
            q = self.mod[j]
            if q == 1:
                res[j] = charges[j]
            else:
                res[j] = charges[j]  % q
                if res[j] < 0:  # correct for C-modulo opposed to python modulo
                    res[j] += q
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef np.ndarray _make_valid_2D(ChargeInfo self, np.ndarray charges):
        cdef np.ndarray res = np.empty_like(charges)
        cdef np.ndarray mod = self.mod
        cdef int L = charges.shape[0]
        cdef int i, j
        cdef QTYPE_t q
        for j in range(self.qnumber):
            q = mod[j]
            if q == 1:
                for i in range(L):
                    res[i, j] = charges[i, j]
            else:
                for i in range(L):
                    res[i, j] = charges[i, j] % q
                    if res[i, j] < 0:  # correct for C-modulo opposed to python modulo
                        res[i, j] += q
                    continue
        return res

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    def check_valid(ChargeInfo self, np.ndarray charges):
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
        assert (charges.shape[1] == self.qnumber)
        cdef np.ndarray mod = self.mod
        cdef int i, j
        cdef QTYPE_t q, x
        cdef int L = charges.shape[0]
        for j in range(self.qnumber):
            q = mod[j]
            if q == 1:
                continue
            for i in range(L):
                x = charges[i, j]
                if x < 0 or x >= q:
                    return False
        return True

    def __repr__(self):
        """Full string representation."""
        return "ChargeInfo({0!s}, {1!s})".format(list(self.mod), self.names)

    def __richcmp__(self, other, int operator):
        if operator == 2: # equal:
            return self._equal(other)
        elif operator == 3:
            return not self._equal(other)
        else:
            raise NotImplementedError("No ordering of `ChargeInfo` possible")

    def _equal(ChargeInfo self, ChargeInfo other):
        """Compare self.mod and self.names for equality, ignore missing names."""
        if self is other:
            return True
        if not np.all(self.mod == other.mod):
            return False
        for l, r in zip(self.names, other.names):
            if r != l and l != '' and r != '':
                return False
        return True

    cpdef tuple __getstate__(ChargeInfo self):
        """Allow to pickle and copy."""
        return (self.qnumber, self.mod, self.names)

    cpdef void __setstate__(ChargeInfo self, tuple state):
        """Allow to pickle and copy."""
        qnumber, mod, names = state
        self.qnumber = qnumber
        self.mod = mod
        self.names = names


cdef class LegCharge(object):
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
    slices : ndarray[np.intp_t,ndim=1] (block_number+1)
        A block with 'qindex' ``qi`` correspondes to the leg indices in
        ``slice(self.slices[qi], self.slices[qi+1])``. See :meth:`get_slice`.
    charges : ndarray[QTYPE_t,ndim=1] (block_number, chinfo.qnumber)
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

    def __init__(LegCharge self, chargeinfo, slices, charges, qconj=1):
        self.chinfo = chargeinfo
        self.slices = np.array(slices, dtype=np.intp)
        self.charges = np.array(charges, dtype=QTYPE)
        self.qconj = qconj
        self.sorted = False
        self.bunched = False
        self.ind_len = self.slices[-1]
        self.block_number = self.charges.shape[0]
        LegCharge.test_sanity(self)

    cdef LegCharge copy(LegCharge self):
        """Return a (shallow) copy of self."""
        cdef LegCharge res = LegCharge.__new__(LegCharge)
        res.__setstate__(self.__getstate__())
        return res

    @classmethod
    def from_trivial(cls, ind_len, chargeinfo=None, qconj=1):
        """Create trivial (qnumber=0) LegCharge for given len of indices `ind_len`."""
        if chargeinfo is None:
            chargeinfo = ChargeInfo()
            charges = [[]]
        else:
            charges = [[0] * chargeinfo.qnumber]
        res = cls(chargeinfo, [0, ind_len], charges, qconj)
        return res

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

    cpdef void test_sanity(LegCharge self) except *:
        """Sanity check. Raises ValueErrors, if something is wrong."""
        cdef int opt_level = optimization._level
        if opt_level >= optimization_compare:
            return
        cdef np.ndarray sl = self.slices
        cdef np.ndarray ch = self.charges
        if sl.ndim != 1 or sl.shape[0] != self.block_number + 1:
            raise ValueError("wrong len of `slices`")
        if sl[0] != 0:
            raise ValueError("slices does not start with 0")
        if ch.shape[1] != self.chinfo.qnumber:
            raise ValueError("shape of `charges` incompatible with qnumber")
        if not self.chinfo.check_valid(ch):
            raise ValueError("charges invalid for " + str(self.chinfo) + "\n" + str(self))
        if self.qconj != -1 and self.qconj != 1:
            raise ValueError("qconj has invalid value != +-1 :" + repr(self.qconj))

    cpdef LegCharge conj(LegCharge self):
        """Return a (shallow) copy with opposite ``self.qconj``."""
        res = self.copy()
        res.qconj = -self.qconj
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

    cpdef bint is_blocked(self):
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
        return len(_c_find_row_differences(self.charges)) == self.block_number + 1

    cpdef void test_contractible(LegCharge self, LegCharge other) except *:
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
        cdef int opt_level = optimization._level
        if opt_level >= optimization_compare:
            return
        self.test_equal(other.conj())

    cpdef void test_equal(LegCharge self, LegCharge other) except *:
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
        cdef int opt_level = optimization._level
        if opt_level >= optimization_compare:
            return
        if self.chinfo != other.chinfo:
            raise ValueError(
                ''.join(["incompatible ChargeInfo\n", str(self.chinfo), str(other.chinfo)]))
        if self.charges is other.charges and self.qconj == other.qconj and \
                (self.slices is other.slices or np.all(self.slices == other.slices)):
            return  # optimize: don't need to check all charges explicitly
        if not np.array_equal(self.slices, other.slices) or \
                not np.array_equal(self.charges * self.qconj, other.charges * other.qconj):
            raise ValueError("incompatible LegCharge\n" + vert_join(
                ["self\n" + str(self), "other\n" + str(other)], delim=' | '))

    cpdef slice get_slice(LegCharge self, int qindex):
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

    def sort(LegCharge self, bint bunch=True):
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
        cp = self.copy()
        cp._set_charges(self.charges[perm_qind, :])
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
        cp = self.copy()
        idx = _c_find_row_differences(self.charges)
        cp._set_charges(cp.charges[idx[:-1]])  # avanced indexing -> copy
        cp._set_slices(cp.slices[idx])
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
        cp = self.copy()
        block_masks = [mask[b:e] for b, e in self._slice_start_stop()]
        new_block_lens = [np.sum(bm) for bm in block_masks]
        keep = np.nonzero(new_block_lens)[0]
        block_masks = [block_masks[i] for i in keep]
        cp._set_charges(cp.charges[keep])
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
        charges = charges[_c_find_row_differences(charges)[:-1], :]
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

    cpdef void _set_charges(LegCharge self, np.ndarray charges):
        """Provide hook to set 'private' charges."""
        self.charges = charges
        self.block_number = charges.shape[0]

    cpdef void _set_slices(LegCharge self, np.ndarray slices):
        self.slices = slices
        self.ind_len = slices[-1]

    cpdef _set_block_sizes(self, block_sizes):
        """Set self.slices from an list of the block-sizes."""
        self._set_slices(np.append([0], np.cumsum(block_sizes)).astype(np.intp, copy=False))

    cpdef _get_block_sizes(self):
        """Return block sizes."""
        cdef np.ndarray sl = self.slices
        return sl[1:] - sl[:-1]

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

    cpdef tuple __getstate__(LegCharge self):
        """Allow to pickle and copy."""
        return (self.ind_len,
                self.block_number,
                self.chinfo,
                self.slices,
                self.charges,
                self.qconj,
                self.sorted,
                self.bunched)

    cpdef void __setstate__(LegCharge self, tuple state):
        """Allow to pickle and copy."""
        ind_len, block_number, chinfo, slices, charges, qconj, sorted, bunched = state
        self.ind_len = ind_len
        self.block_number = block_number
        self.chinfo = chinfo
        self.slices = slices
        self.charges = charges
        self.qconj = qconj
        self.sorted = sorted
        self.bunched = bunched


cdef class LegPipe(LegCharge):
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
    nlegs : int
        The number of legs.
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
        self.nlegs = len(legs)
        self.subshape = tuple([l.ind_len for l in legs])
        self.subqshape = tuple([l.block_number for l in legs])
        # the diffuclt part: calculate self.slices, self.charges, self.q_map and self.q_map_slices
        self._init_from_legs(sort, bunch)
        self.test_sanity()

    cdef LegPipe copy(LegPipe self):
        """Return a (shallow) copy of self."""
        cdef LegPipe res = LegPipe.__new__(LegPipe)
        res.__setstate__(self.__getstate__())
        return res

    cpdef void test_sanity(LegPipe self) except *:
        """Sanity check. Raises ValueErrors, if something is wrong."""
        cdef int opt_level = optimization._level
        if opt_level >= optimization_compare:
            return
        LegCharge.test_sanity(self)
        assert (all([l.chinfo == self.chinfo for l in self.legs]))
        assert (self.subshape == tuple([l.ind_len for l in self.legs]))
        assert (self.subqshape == tuple([l.block_number for l in self.legs]))

    def to_LegCharge(self):
        """Convert self to a LegCharge, discarding the information how to split the legs.
        Usually not needed, but called by functions, which are not implemented for a LegPipe."""
        return LegCharge(self.chinfo, self.slices, self.charges, self.qconj)

    cpdef LegPipe conj(LegPipe self):
        """Return a shallow copy with opposite ``self.qconj``.

        Also conjugates each of the incoming legs."""
        cdef LegPipe res = LegCharge.conj(self)  # invert self.qconj
        res.legs = tuple([l.conj() for l in self.legs])
        return res

    def outer_conj(self):
        """Like :meth:`conj`, but don't change ``qconj`` for incoming legs."""
        res = self.copy()  # shallow
        res.qconj = -1
        res._set_charges(self.chinfo.make_valid(-self.charges))
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

    def __str__(self):
        """Fairly short debug output."""
        res_lines = [
            "LegPipe(shape {0!s}->{1:d}, ".format(self.subshape, self.ind_len),
            "    qconj {0}->{1:+1};".format(
                '(' + ', '.join(['%+d' % l.qconj for l in self.legs]) + ')', self.qconj),
            "    block numbers {0!s}->{1:d})".format(self.subqshape, self.block_number), vert_join(
                [str(l) for l in self.legs], delim=' | '), ')'
        ]
        return '\n'.join(res_lines)

    def __repr__(self):
        """Full string representation."""
        return "LegPipe({legs},\nqconj={qconj:+d}, sort={s!r}, bunch={b!r})".format(
            legs='[' + ',\n'.join([repr(l) for l in self.legs]) + ']',
            qconj=self.qconj,
            s=self.sorted,
            b=self.bunched)

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
        for ax in range(self.nlegs -1, -1, -1):   # reversed: C order within the block
            leg = self.legs[ax]
            qind, within_block = leg.get_qindex(incoming_indices[ax])
            qind_in[0, ax] = qind
            within_block_out += stride * within_block
            stride *= (leg.slices[qind+1] - leg.slices[qind])
        j = self._map_incoming_qind(qind_in)[0]
        q_map = self.q_map[j, :]
        assert(q_map[1] - q_map[0] == stride)
        qind_out = q_map[2]  # I_s
        return self.slices[qind_out] + q_map[0] + within_block_out

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void _init_from_legs(LegPipe self, bint sort=True, bint bunch=True) except *:
        """Calculate ``self.qind``, ``self.q_map`` and ``self.q_map_slices`` from ``self.legs``.

        `qind` is constructed to fullfill the charge fusion rule stated in the class doc-string.
        """
        # this function heavily uses numpys advanced indexing, for details see
        # `http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html`_
        # and the documentation of np.mgrid
        cdef int nlegs = self.nlegs
        cdef int qnumber = self.chinfo.qnumber
        cdef np.ndarray qshape = np.array(self.subqshape, dtype=np.intp)
        cdef int i, j, sign
        cdef np.intp_t a

        # create a grid to select the multi-index sector
        grid = np.mgrid[[slice(0, l) for l in qshape]]
        # grid is an array with shape ``(nlegs,) + qshape``,
        # with grid[li, ...] = {np.arange(qshape[li]) increasing in the li-th direcion}
        cdef np.ndarray strides =  \
            np.array(grid.strides, np.intp)[1:] // grid.itemsize
        self._strides = strides  # save for :meth:`_map_incoming_qind`
        # collapse the different directions into one.
        cdef np.ndarray grid2 = grid.reshape(nlegs, -1)
            # *this* is the actual `reshaping`
        # *columns* of grid are now all possible cominations of qindices.

        cdef int nblocks = grid2.shape[1]  # number of blocks in the pipe = np.product(qshape)
        cdef np.ndarray q_map = np.empty((nblocks, 3 + nlegs), dtype=np.intp)
        # determine q_map -- it's essentially the grid.
        q_map[:, 3:] = grid2.T  # transpose -> rows are possible combinations.
        # q_map[:, :3] is initialized after sort/bunch.

        # determine block sizes
        cdef np.ndarray blocksizes = np.ones((nblocks,), dtype=np.intp)
        cdef np.ndarray leg_bs
        for i in range(nlegs):
            leg_bs = self.legs[i]._get_block_sizes()
            for j in range(nblocks):
                blocksizes[j] *= leg_bs[grid2[i, j]]

        # calculate total charges
        cdef np.ndarray charges = np.zeros((nblocks, qnumber), dtype=QTYPE)
        cdef np.ndarray legcharges
        if qnumber > 0:
            for i in range(nlegs):
                legcharges = self.legs[i].charges
                sign = self.qconj * self.legs[i].qconj
                for j in range(nblocks):
                    for k in range(qnumber):
                        charges[j, k] += sign * legcharges[grid2[i, j], k]
            charges = self.chinfo.make_valid(charges)
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
        self._set_charges(charges)
        self.sorted = sort
        self._set_block_sizes(blocksizes)  # sets self.slices
        cdef np.ndarray slices = self.slices
        for j in range(nblocks):
            q_map[j, 0] = slices[j]
            q_map[j, 1] = slices[j+1]

        cdef np.ndarray idx
        if bunch:
            # call LegCharge.bunch(), which also calculates new blocksizes
            idx, bunched = LegCharge.bunch(self)
            self._set_charges(bunched.charges)  # copy information back to self
            self._set_slices(bunched.slices)
            a = 0
            for i in range(idx.shape[0]-1):
                for j in range(idx[i], idx[i+1]):
                    q_map[j, 2] = a
                a += 1
            for j in range(idx[idx.shape[0]-1], nblocks):
                q_map[j, 2] = a
        else:
            # trivial mapping for q_map[:, 2]
            for j in range(nblocks):
                q_map[j, 2] = j
            idx = np.arange(len(q_map)+1, dtype=np.intp)

        # calculate the slices within blocks: subtract the start of each block
        slices = self.slices
        for j in range(nblocks):
            a = slices[q_map[j, 2]]
            q_map[j, 0] -= a
            q_map[j, 1] -= a

        self.q_map = q_map  # finished
        # finally calculate q_map_slices
        self.q_map_slices = [q_map[idx[i]:idx[i+1]] for i in range(len(idx)-1)]
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

    cpdef tuple __getstate__(LegPipe self):
        """Allow to pickle and copy."""
        super_state = LegCharge.__getstate__(self)
        return (super_state,
                self.nlegs,
                self.legs,
                self.subshape,
                self.subqshape,
                self.q_map,
                self.q_map_slices,
                self._perm,
                self._strides)

    cpdef void __setstate__(LegPipe self, tuple state):
        """Allow to pickle and copy."""
        super_state, nlegs, legs, subshape, subqshape, q_map, q_map_slices, _perm, _strides = state
        self.nlegs = nlegs
        self.legs = legs
        self.subshape = subshape
        self.subqshape = subqshape
        self.q_map = q_map
        self.q_map_slices = q_map_slices
        self._perm = _perm
        self._strides = _strides
        LegCharge.__setstate__(self, super_state)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef np.ndarray _c_find_row_differences(np.ndarray qflat):
    """C-version of :func:`_find_row_differences` (which uses numpy for optimization)"""
    if qflat.shape[1] == 0:
        return np.array([0, qflat.shape[0]], dtype=np.intp)
    cdef int i, j, n=1, L = qflat.shape[0], M = qflat.shape[1]
    cdef bint rows_equal = False
    cdef np.ndarray res = np.empty(max(L + 1, 2), dtype=np.intp)
    res[0] = 0
    for i in range(1, L):
        rows_equal = True
        for j in range(M):
            if qflat[i-1, j] != qflat[i, j]:
                rows_equal = False
                break
        if not rows_equal:
            res[n] = i
            n += 1
    res[n] = L
    return res[:n+1]

cdef np.ndarray _partial_qtotal(ChargeInfo chinfo,
                                                list legs,
                                                np.ndarray qdata):
    """Calculate qtotal of a part of the legs of a npc.Array.

    Parameters
    ----------
    chinfo : ChargeInfo
        Common ChargeInfo of the legs.
    legs : list of LegCharge
        The legs over which wich we sum
    qdata : 2D array
        Rows of qindices for the specified legs.

    Returns
    -------
    partial_qtotal : 2D ndarray
        Valid 2D version of
        ``chinfo.make_valid(np.sum([l.get_charge(qi) for l, qi in zip(legs, qdata.T)], axis=0))``.
    """
    cdef np.ndarray res = np.zeros([qdata.shape[0], chinfo.qnumber], QTYPE)
    cdef int a, k, qi
    cdef LegCharge leg
    cdef np.ndarray charges
    for a in range(qdata.shape[1]):
        leg = legs[a]
        charges  = leg.charges
        for i in range(qdata.shape[0]):
            qi = qdata[i, a]
            for k in range(chinfo.qnumber):
                res[i, k] += charges[qi, k] * leg.qconj
    return chinfo._make_valid_2D(res)


# ############################################### #
# replacements for np_conserved.Array methods     #
# ############################################### #


def _combine_legs_worker(self, list combine_legs, list new_axes, list pipes):
    """The main work of combine_legs: create a copy and reshape the data blocks.

    Assumes standard form of parameters.

    Parameters
    ----------
    combine_legs : list(1D np.array)
        Axes of self which are collected into pipes.
    new_axes : 1D array
        The axes of the pipes in the new array. Ascending.
    pipes : list of :class:`LegPipe`
        All the correct output pipes, already generated.

    Returns
    -------
    res : :class:`Array`
        Copy of self with combined legs.
    """
    all_combine_legs = np.concatenate(combine_legs)
    # non_combined_legs: axes of self which are not in combine_legs
    cdef np.ndarray[np.intp_t, ndim=1] non_combined_legs = np.array(
        [a for a in range(self.rank) if a not in all_combine_legs], dtype=np.intp)
    legs = [self.legs[i] for i in non_combined_legs]
    for na, p in zip(new_axes, pipes):  # not reversed
        legs.insert(na, p)
    res = self.copy(deep=False)
    res.legs = legs
    res._set_shape()
    if self.stored_blocks == 0:
        res._data = []
        res._qdata = np.empty((0, res.rank), dtype=np.intp)
        return res
    non_new_axes_ = [i for i in range(res.rank) if i not in new_axes]
    cdef np.ndarray[np.intp_t, ndim=1] non_new_axes = np.array(non_new_axes_, dtype=np.intp)

    # map `self._qdata[:, combine_leg]` to `pipe.q_map` indices for each new pipe
    qmap_inds = [
        p._map_incoming_qind(self._qdata[:, cl]) for p, cl in zip(pipes, combine_legs)
    ]

    # get new qdata
    cdef np.ndarray[np.intp_t, ndim=2] qdata = np.empty((self.stored_blocks, res.rank),
                                                        dtype=np.intp)
    qdata[:, non_new_axes] = self._qdata[:, non_combined_legs]
    for na, p, qmap_ind in zip(new_axes, pipes, qmap_inds):
        np.take(
            p.q_map[:, 2],  # column 2 of q_map maps to qindex of the pipe
            qmap_ind,
            out=qdata[:, na])  # write the result directly into qdata
    # now we have probably many duplicate rows in qdata,
    # since for the pipes many `qmap_ind` map to the same `qindex`
    # find unique entries by sorting qdata
    sort = np.lexsort(qdata.T)
    qdata_s = qdata[sort]
    old_data = [self._data[s] for s in sort]
    qmap_inds = [qm[sort] for qm in qmap_inds]
    # divide into parts, which give a single new block
    cdef np.ndarray[np.intp_t, ndim=1] diffs = _c_find_row_differences(qdata_s)
    # including the first and last row

    # now the hard part: map data
    cdef list data = []
    cdef list slices = [slice(None)] * res.rank  # for selecting the slices in the new blocks
    # iterate over ranges of equal qindices in qdata_s
    cdef np.ndarray new_block, old_block #, new_block_view # TODO: shape doesn't work...
    cdef np.ndarray[np.intp_t, ndim=1] qindices
    cdef int beg, end, bi, j, old_data_idx, qi
    cdef np.ndarray[np.intp_t, ndim=2] q_map
    cdef tuple sl
    cdef int npipes = len(combine_legs)
    for bi in range(diffs.shape[0]-1):
        beg = diffs[bi]
        end = diffs[bi+1]
        qindices = qdata_s[beg]
        new_block = np.zeros(res._get_block_shape(qindices), dtype=res.dtype)
        data.append(new_block)
        # copy blocks
        for old_data_idx in range(beg, end):
            for j in range(npipes):
                q_map = pipes[j].q_map
                qi = qmap_inds[j][old_data_idx]
                slices[new_axes[j]] = slice(q_map[qi, 0], q_map[qi, 1])
            sl = tuple(slices)
            # reshape block while copying
            new_block_view = new_block[sl]
            old_block = old_data[old_data_idx].reshape(new_block_view.shape)
            np.copyto(new_block_view, old_block, casting='no')
    res._qdata = qdata_s[diffs[:-1]]  # (keeps the dimensions)
    res._qdata_sorted = True
    res._data = data
    return res


def _split_legs_worker(self, list split_axes_, float cutoff):
    """The main work of split_legs: create a copy and reshape the data blocks.

    Called by :meth:`split_legs`. Assumes that the corresponding legs are LegPipes.
    """
    # calculate mappings of axes
    # in self
    cdef np.ndarray[np.intp_t, ndim=1] split_axes = np.sort(split_axes_)
    cdef int a, i, j, nsplit=split_axes.shape[0]
    pipes = [self.legs[a] for a in split_axes]
    cdef np.ndarray[np.intp_t, ndim=1] nonsplit_axes = np.array(
        [i for i in range(self.rank) if i not in split_axes], dtype=np.intp)
    # in result
    cdef np.ndarray[np.intp_t, ndim=1] new_nonsplit_axes = np.arange(self.rank, dtype=np.intp)
    for a in split_axes:
        new_nonsplit_axes[a + 1:] += self.legs[a].nlegs - 1
    cdef np.ndarray[np.intp_t, ndim=1] new_split_axes_first = new_nonsplit_axes[split_axes]
    #    = the first leg for splitted pipes
    cdef list new_split_slices = [slice(a, a + p.nlegs) for a, p in zip(new_split_axes_first, pipes)]
    new_nonsplit_axes = new_nonsplit_axes[nonsplit_axes]

    res = self.copy(deep=False)
    for a in reversed(split_axes):
        res.legs[a:a + 1] = res.legs[a].legs  # replace pipes with saved original legs
    res._set_shape()

    # get new qdata by stacking columns
    tmp_qdata = np.empty((self.stored_blocks, res.rank), dtype=np.intp)
    tmp_qdata[:, new_nonsplit_axes] = self._qdata[:, nonsplit_axes]
    tmp_qdata[:, new_split_axes_first] = self._qdata[:, split_axes]

    # now split the blocks
    cdef list data = []
    cdef list qdata = []  # rows of the new qdata
    cdef np.ndarray[np.intp_t, ndim=1] new_block_shape = np.empty(res.rank, dtype=np.intp)
    cdef np.ndarray[np.intp_t, ndim=1] qdata_row, qm
    cdef list block_slice = [slice(None)] * self.rank
    cdef list qmap_slices = [None] * nsplit
    cdef slice sl
    cdef LegPipe pipe
    cdef LegCharge leg
    cdef np.ndarray old_block, new_block

    for old_block, qdata_row in zip(self._data, tmp_qdata):
        for j in range(nonsplit_axes.shape[0]):
            new_block_shape[new_nonsplit_axes[j]] = old_block.shape[nonsplit_axes[j]]
        for j in range(nsplit):
            pipe = pipes[j]
            qmap_slices[j] = pipe.q_map_slices[qdata_row[new_split_axes_first[j]]]
        for qmap_rows in itertools.product(*qmap_slices):
            for i in range(nsplit):
                qm = qmap_rows[i]
                a = new_split_axes_first[i]
                pipe = pipes[i]
                for j in range(pipe.nlegs):
                    qi = qm[3+j]
                    qdata_row[a+j] = qi
                    leg = pipe.legs[j]
                    new_block_shape[a+j] = leg.slices[qi+1] - leg.slices[qi]
                block_slice[split_axes[i]] = slice(qm[0], qm[1])
            new_block = old_block[tuple(block_slice)].reshape(new_block_shape)
            # all charges are compatible by construction, but some might be zero
            if np.any(np.abs(new_block) > cutoff):
                data.append(new_block.copy())  # copy, not view
                qdata.append(qdata_row.copy())  # copy! qdata_row is changed afterwards...
    if len(data) > 0:
        res._qdata = np.array(qdata, dtype=np.intp)
        res._qdata_sorted = False
    else:
        res._qdata = np.empty((0, res.rank), dtype=np.intp)
        res._qdata_sorted = True
    res._data = data
    return res


# ##################################################### #
# replacements for global functions in np_conserved.py  #
# ##################################################### #

@cython.wraparound(False)
@cython.boundscheck(False)
cdef Py_ssize_t _iter_common_sorted(
        np.ndarray[np.intp_t, ndim=1] a, Py_ssize_t i_start, Py_ssize_t i_stop,
        np.ndarray[np.intp_t, ndim=1] b, Py_ssize_t j_start, Py_ssize_t j_stop,
        np.ndarray[np.intp_t, ndim=2] out):
    """Find indices ``i, j`` for which ``a[i] == b[j]``.

    *Assumes* that ``a[i_start:i_stop]`` and ``b[j_start:j_stop]`` are strictly ascending.
    Given that, it is equivalent to (but faster than)::

        count = 0
        for j, i in itertools.product(range(j_start, j_stop), range(i_start, i_stop)):
            if a[i] == b[j]:
                out[count] = [i, j]
                count += 1
        return count
    """
    cdef Py_ssize_t i=i_start, j=j_start, count=0
    while i < i_stop and j < j_stop:
        if a[i] < b[j]:
            i += 1
        elif b[j] < a[i]:
            j += 1
        else:
            #  yield i, j
            out[count, 0] = i
            out[count, 1] = j
            count += 1
            i += 1
            j += 1
    return count


cdef _tensordot_pre_sort(a, b, int cut_a, int cut_b):
    """Pre-calculations before the actual matrix product.

    Called by :func:`_tensordot_worker`.
    See doc-string of :func:`tensordot` for details on the implementation.

    Parameters
    ----------
    a, b : :class:`Array`
        the arrays to be contracted with tensordot. Should have non-empty ``a._data``
    cut_a, cut_b : int
        contract `a.legs[cut_a:]` with `b.legs[:cut_b]`

    Returns
    -------
    a_data, a_qdata_keep, a_qdata_contr, b_data, b_qdata_keep, b_qdata_contr
    """
    cdef list a_data, b_data
    cdef np.ndarray[np.intp_t, ndim=2] a_qdata_keep, b_qdata_keep
    cdef np.ndarray[np.intp_t, ndim=1] a_qdata_contr, b_qdata_contr
    # convert qindices over which we sum to a 1D array for faster lookup/iteration
    stride = np.cumprod([1] + [l.block_number for l in a.legs[cut_a:-1]])
    a_qdata_contr = np.sum(a._qdata[:, cut_a:] * stride, axis=1)
    # lex-sort a_qdata, dominated by the axes kept, then the axes summed over.
    a_sort = np.lexsort(np.append(a_qdata_contr[:, np.newaxis], a._qdata[:, :cut_a], axis=1).T)
    a_qdata_keep = a._qdata[a_sort, :cut_a]
    a_qdata_contr = a_qdata_contr[a_sort]
    a_data = a._data
    a_data = [a_data[i] for i in a_sort]
    # combine all b_qdata[axes_b] into one column (with the same stride as before)
    b_qdata_contr = np.sum(b._qdata[:, :cut_b] * stride, axis=1)
    # lex-sort b_qdata, dominated by the axes summed over, then the axes kept.
    b_data = b._data
    if not b._qdata_sorted:
        b_sort = np.lexsort(np.append(b_qdata_contr[:, np.newaxis], b._qdata[:, cut_b:], axis=1).T)
        b_qdata_keep = b._qdata[b_sort, cut_b:]
        b_qdata_contr = b_qdata_contr[b_sort]
        b_data = [b_data[i] for i in b_sort]
    else:
        b_data = list(b_data)  # make a copy: we write into it
        b_qdata_keep = b._qdata[:, cut_b:]
    return a_data, a_qdata_keep, a_qdata_contr, b_data, b_qdata_keep, b_qdata_contr


@cython.boundscheck(False)
cdef _tensordot_match_charges(int n_rows_a,
                              int n_cols_b,
                              int qnumber,
                              np.ndarray a_charges_keep,
                              np.ndarray b_charges_match):
    """Estimate number of blocks in res and get order for iteration over row_a and col_b

    Parameters
    ----------
    a_charges_keep, b_charges_match: 2D ndarray
        (Unsorted) charges of dimensions (n_rows_a, qnumber) and (n_cols_b, qnumber)

    Returns
    -------
    max_n_blocks_res : int
        Maximum number of block in the result a.dot(b)
    row_a_sort: np.ndarray[np.intp_t, ndim=1]
    match_rows: np.ndarray[np.intp_t, ndim=2]
        For given `col_b`, rows given by `row_a` in
        ``row_a_sort[match_rows[col_b, 0]:match_rows[col_b,1]]`` fulfill
        ``a_charges_keep[col_a, :] == b_charges_match[col_b, :]``
    """
    # This is effectively a more complicated version of _iter_common_sorted....
    cdef np.ndarray[np.intp_t, ndim=2] match_rows = np.empty((n_cols_b, 2), np.intp)
    cdef np.ndarray[QTYPE_t, ndim=2] a_charges_keep_C = a_charges_keep
    cdef np.ndarray[QTYPE_t, ndim=2] b_charges_match_C = b_charges_match
    if qnumber == 0:  # special case no restrictions due to charge
        match_rows[:, 0] = 0
        match_rows[:, 1] = n_rows_a
        return n_rows_a * n_cols_b, np.arange(n_rows_a), match_rows
    # general case
    cdef np.ndarray[np.intp_t, ndim=1] row_a_sort = np.lexsort(a_charges_keep.T)
    cdef np.ndarray[np.intp_t, ndim=1] col_b_sort = np.lexsort(b_charges_match.T)
    cdef int res_max_n_blocks = 0
    cdef int i=0, j=0, i0, j0, ax, j1
    cdef int i_s, j_s, i0_s, j0_s  # corresponding entries in row_a_sort/col_b_sort
    cdef int lexcomp
    while i < n_rows_a and j < n_cols_b: # go through sort_a and sort_b at the same time
        i_s = row_a_sort[i]
        j_s = col_b_sort[j]
        # lexcompare a_charges_keep[i_s, :] and b_charges_match[j_s, :]
        lexcomp = 0
        for ax in range(qnumber-1, -1, -1):
            if a_charges_keep_C[i_s, ax] > b_charges_match_C[j_s, ax]:
                lexcomp = 1
                break
            elif a_charges_keep_C[i_s, ax] < b_charges_match_C[j_s, ax]:
                lexcomp = -1
                break
        if lexcomp > 0:  # a_charges_keep is larger: advance j
            match_rows[j_s, 0] = 0  # nothing to iterate for this col_b = j_s
            match_rows[j_s, 1] = 0
            j += 1
            continue
        elif lexcomp < 0: # b_charges_match is larger
            i += 1
            continue
        # else: charges for i_s and j_s and match
        # which/how many rows_a have the same charge? Increase i until the charges change.
        i0 = i
        i0_s = i_s
        i += 1
        while i < n_rows_a:
            i_s = row_a_sort[i]
            lexcomp = 0
            for ax in range(qnumber-1, -1, -1):
                if a_charges_keep_C[i_s, ax] != a_charges_keep_C[i0_s, ax]:
                    lexcomp = 1
                    break
            if lexcomp > 0:  # (sorted -> can only increase)
                break
            i += 1
        # => the rows in row_a_sort[i0:i] have the current charge
        j0 = j
        j0_s = j_s
        j += 1
        while j < n_cols_b:
            j_s = col_b_sort[j]
            lexcomp = 0
            for ax in range(qnumber-1, -1, -1):
                if b_charges_match_C[j_s, ax] != b_charges_match_C[j0_s, ax]:
                    lexcomp = 1
                    break
            if lexcomp > 0:  # (sorted -> can only increase)
                break
            j += 1
        # => the colums in col_b_sort[j0:j] have the current charge
        # save rows for iteration for the given j_s in col_b
        for j1 in range(j0, j):
            j_s = col_b_sort[j1]
            match_rows[j_s, 0] = i0
            match_rows[j_s, 1] = i
        res_max_n_blocks += (j-j0) * (i-i0)
    for j1 in range(j, n_cols_b):
        j_s = col_b_sort[j1]
        match_rows[j_s, 0] = 0
        match_rows[j_s, 1] = 0
    return res_max_n_blocks, row_a_sort, match_rows


@cython.cdivision(True)
def _tensordot_worker(a, b, int axes):
    """Main work of tensordot, called by :func:`tensordot`.

    Assumes standard form of parameters: axes is integer,
    sum over the last `axes` legs of `a` and first `axes` legs of `b`.

    Notes
    -----
    Looking at the source of numpy's tensordot (which is just 62 lines of python code),
    you will find that it has the following strategy:

    1. Transpose `a` and `b` such that the axes to sum over are in the end of `a` and front of `b`.
    2. Combine the legs `axes`-legs and other legs with a `np.reshape`,
       such that `a` and `b` are matrices.
    3. Perform a matrix product with `np.dot`.
    4. Split the remaining axes with another `np.reshape` to obtain the correct shape.

    The main work is done by `np.dot`, which calls LAPACK to perform the simple matrix product.
    [This matrix multiplication of a ``NxK`` times ``KxM`` matrix is actually faster
    than the O(N*K*M) needed by a naive implementation looping over the indices.]

    We follow the same overall strategy, viewing the :class:`Array` as a tensor with
    data block entries.
    Step 1) is performed directly in :func:`tensordot`.

    The steps 2) and 4) could be implemented with :meth:`Array.combine_legs`
    and :meth:`Array.split_legs`.
    However, that would actually be an overkill: we're not interested
    in the full charge data of the combined legs (which would be generated in the LegPipes).
    Instead, we just need to track the qindices of the `a._qdata` and `b._qdata` carefully.

    Our step 2) is implemented in :func:`_tensordot_pre_worker`:
    We split `a._qdata` in `a_qdata_keep` and `a_qdata_sum`, and similar for `b`.
    Then, view `a` is a matrix :math:`A_{i,k1}` and `b` as :math:`B_{k2,j}`, where
    `i` can be any row of `a_qdata_keep`, `j` can be any row of `b_qdata_keep`.
    The `k1` and `k2` are rows of `a_qdata_sum` and `b_qdata_sum`, which stem from the same legs
    (up to a :meth:`LegCharge.conj()`).
    In our storage scheme, `a._data[s]` then contains the block :math:`A_{i,k1}` for
    ``j = a_qdata_keep[s]`` and ``k1 = a_qdata_sum[s]``.
    To identify the different indices `i` and `j`, it is easiest to lexsort in the `s`.
    Note that we give priority to the `#_qdata_keep` over the `#_qdata_sum`, such that
    equal rows of `i` are contiguous in `#_qdata_keep`.
    Then, they are identified with :func:`Charges._find_row_differences`.

    Now, the goal is to calculate the sums :math:`C_{i,j} = sum_k A_{i,k} B_{k,j}`,
    analogous to step 3) above. This is implemented in :func:`_tensordot_worker`.
    It is done 'naively' by explicit loops over ``i``, ``j`` and ``k``.
    However, this is not as bad as it sounds:
    First, we loop only over existent ``i`` and ``j``
    (in the sense that there is at least some non-zero block with these ``i`` and ``j``).
    Second, if the ``i`` and ``j`` are not compatible with the new total charge,
    we know that ``C_{i,j}`` will be zero.
    Third, given ``i`` and ``j``, the sum over ``k`` runs only over
    ``k1`` with nonzero :math:`A_{i,k1}`, and ``k2` with nonzero :math:`B_{k2,j}`.

    How many multiplications :math:`A_{i,k} B_{k,j}` we actually have to perform
    depends on the sparseness. In the ideal case, if ``k`` (i.e. a LegPipe of the legs summed over)
    is completely blocked by charge, the 'sum' over ``k`` will contain at most one term!
    """
    cdef ChargeInfo chinfo = a.chinfo
    cdef Py_ssize_t cut_a = a.rank - axes
    cdef Py_ssize_t cut_b = axes
    cdef Py_ssize_t b_rank = b.rank
    cdef Py_ssize_t res_rank = cut_a + b_rank - cut_b
    cdef bint DEBUG_PRINT = 0  # TODO XXX
    if DEBUG_PRINT:
        t0 = time.time()
    # determine calculation type and result type
    dtype = np.find_common_type([a.dtype, b.dtype], [])
    prefix, _, _ = BLAS.find_best_blas_type(dtype=dtype)
    # we always use 64-bit float calculations....
    res_dtype = np.dtype({'s': np.float64,
                          'd': np.float64,
                          'c': np.complex128,
                          'z': np.complex128}[prefix])
    # TODO: handle special dtypes?
    cdef int CALC_DTYPE_NUM = res_dtype.num  # can be compared to np.NPY_DOUBLE/NPY_CDOUBLE
    cdef np.ndarray[QTYPE_t, ndim=1] qtotal = chinfo._make_valid_1D(a.qtotal + b.qtotal)
    res = np_conserved.Array(a.legs[:cut_a] + b.legs[cut_b:], res_dtype, qtotal)

    cdef np.ndarray[np.intp_t, ndim=2] a_qdata = a._qdata
    cdef np.ndarray[np.intp_t, ndim=2] b_qdata = b._qdata
    cdef list a_data = a._data, b_data = b._data
    cdef Py_ssize_t len_a_data = len(a_data)
    cdef Py_ssize_t len_b_data = len(b_data)
    cdef bint equal = 1
    cdef int i, j
    # special case: a or b is zero
    if len_a_data == 0 or len_b_data == 0:  # special case: `a` or `b` is 0
        return np_conserved.zeros(a.legs[:-axes] + b.legs[axes:],
                     np.find_common_type([a.dtype, b.dtype], []), a.qtotal + b.qtotal)
    # special case: only one stored block
    if len_a_data == 1 and len_b_data == 1:
        # optimize for special case that a and b have only 1 entry
        # this is (usually) the case if we have trivial charges
        for i in range(axes):
            if a_qdata[0, cut_a + i] != b_qdata[0, i]:
                equal = 0
                break
        if equal:
            # contract innner
            res._data = [np.tensordot(a._data[0], b._data[0], axes=axes)]
            c_qdata = np.zeros([1, res_rank], np.intp)
            for i in range(cut_a):
                c_qdata[0, i] = a_qdata[0, i]
            for i in range(b_rank - cut_b):
                c_qdata[0, cut_a + i] = b_qdata[0, cut_b + i]
            res._qdata = c_qdata
            return res
        #  else: return zero
        return res

    cdef np.ndarray[np.intp_t, ndim=2] a_qdata_keep, b_qdata_keep
    cdef np.ndarray[np.intp_t, ndim=1] a_qdata_contr, b_qdata_contr
    # pre_worker
    if DEBUG_PRINT:
        t1 = time.time()
        print("types", t1-t0)
        t0 = time.time()
    a_data, a_qdata_keep, a_qdata_contr, b_data, b_qdata_keep, b_qdata_contr = _tensordot_pre_sort(a, b, cut_a, cut_b)
    if DEBUG_PRINT:
        t1 = time.time()
        print("tensordot_pre_sort", t1-t0)
        t0 = time.time()


    # find blocks where a_qdata_keep and b_qdata_keep change; use that they are sorted.
    cdef np.ndarray[np.intp_t, ndim=1] a_slices = _c_find_row_differences(a_qdata_keep)
    cdef np.ndarray[np.intp_t, ndim=1] b_slices = _c_find_row_differences(b_qdata_keep)
    # the slices divide a_data and b_data into rows and columns
    cdef int n_rows_a = a_slices.shape[0] - 1
    cdef int n_cols_b = b_slices.shape[0] - 1
    a_qdata_keep = a_qdata_keep[a_slices[:n_rows_a]]  # TODO: might get optimized
    b_qdata_keep = b_qdata_keep[b_slices[:n_cols_b]]

    if DEBUG_PRINT:
        t1 = time.time()
        print("find_row_differences", t1-t0)
        t0 = time.time()

    cdef np.ndarray block
    cdef vector[void*] a_data_ptr, b_data_ptr
    a_data_ptr.resize(len_a_data)
    b_data_ptr.resize(len_b_data)
    cdef Py_ssize_t row_a, col_b, k_contr, ax  # indices
    cdef Py_ssize_t m, n, k   # reshaped dimensions: a_block.shape = (m, k), b_block.shape = (k,n)
    cdef np.ndarray[np.intp_t, ndim=2] a_shape_keep = np.empty((n_rows_a, cut_a), np.intp)
    cdef np.ndarray[np.intp_t, ndim=1] block_dim_a_keep = np.empty(n_rows_a, np.intp)
    cdef np.ndarray[np.intp_t, ndim=1] block_dim_a_contr = np.empty(len_a_data, np.intp)
    # inline what's  _tensordot_pre_reshape in the python version
    for row_a in range(n_rows_a):
        i = a_slices[row_a]
        block = <np.ndarray> a_data[i]
        n = 1
        for ax in range(cut_a):
            a_shape_keep[row_a, ax] = block.shape[ax]
            n *= block.shape[ax]
        block_dim_a_keep[row_a] = n
        for j in range(a_slices[row_a], a_slices[row_a+1]):
            block = np.PyArray_GETCONTIGUOUS(a_data[j].astype(res_dtype))
            m = np.PyArray_SIZE(block) / n
            assert m*n == block.size  # TODO XXX  DEBUG
            block_dim_a_contr[j] = m  # needed for dgemm
            a_data_ptr[j] = np.PyArray_DATA(block)
            a_data[j] = block  # important to keep the arrays of the pointers alive
    cdef np.ndarray[np.intp_t, ndim=2] b_shape_keep = np.empty((n_cols_b, b_rank-cut_b), np.intp)
    cdef np.ndarray[np.intp_t, ndim=1] block_dim_b_keep = np.empty(n_cols_b, np.intp)
    for col_b in range(n_cols_b):
        i = b_slices[col_b]
        block = <np.ndarray> b_data[i]
        n = 1
        for ax in range(b_rank-cut_b):
            b_shape_keep[col_b, ax] = block.shape[ax+cut_b]
            n *= block.shape[ax+cut_b]
        block_dim_b_keep[col_b] = n
        for j in range(b_slices[col_b], b_slices[col_b+1]):
            block = np.PyArray_GETCONTIGUOUS((b_data[j].astype(res_dtype)))
            b_data_ptr[j] = np.PyArray_DATA(block)
            b_data[j] = block  # important to keep the arrays of the pointers alive
    if DEBUG_PRINT:
        t1 = time.time()
        print("_tensordot_pre_reshape", t1-t0)
        t0 = time.time()

    # Step 3) loop over column/row of the result
    # (rows_a changes faster than cols_b, such that the resulting array is qdata lex-sorted)

    # first find output colum/row indices of the result, which are compatible with the charges
    cdef np.ndarray[QTYPE_t, ndim=2] a_charges_keep = _partial_qtotal(
        chinfo, a.legs[:cut_a], a_qdata_keep)
    cdef np.ndarray[QTYPE_t, ndim=2] b_charges_keep = _partial_qtotal(
        chinfo, b.legs[cut_b:], b_qdata_keep)
    # a_charges_match: for each row in a, which charge in b is compatible?
    cdef np.ndarray[QTYPE_t, ndim=2] b_charges_match = chinfo._make_valid_2D(qtotal - b_charges_keep)
    cdef np.ndarray[np.intp_t, ndim=1] row_a_sort
    cdef np.ndarray[np.intp_t, ndim=2] match_rows
    cdef int res_max_n_blocks, res_n_blocks = 0
    # the main work for that is in _tensordot_match_charges
    res_max_n_blocks, row_a_sort, match_rows = _tensordot_match_charges(
        n_rows_a, n_cols_b, chinfo.qnumber, a_charges_keep, b_charges_match)
    if DEBUG_PRINT:
        t1 = time.time()
        print("_match_charges", t1-t0)
        t0 = time.time()

    cdef list res_data = []
    cdef np.ndarray[np.intp_t, ndim=2] res_qdata = np.empty((res_max_n_blocks, res_rank), np.intp)
    cdef np.ndarray[np.intp_t, ndim=2, mode='c'] inds_contr  # takes the inner indices
    inds_contr = np.empty((max(len_a_data, len_b_data), 2), np.intp, 'C')
    #  (for the size just estimate the maximal number of blocks to be contracted at once)
    cdef Py_ssize_t inds_contr_count
    cdef intp_t match0, match1
    cdef Py_ssize_t row_a_sort_idx

    #  cdef vector[int] M, N, K, G
    cdef np.ndarray c_block
    cdef np.PyArray_Dims c_block_shape
    c_block_shape.len = res_rank
    c_block_shape.ptr = <np.npy_intp*>PyMem_Malloc(res_rank * sizeof(np.npy_intp))
    if not c_block_shape.ptr:
        raise MemoryError

    # #### the actual loop executing the summation over blocks
    for col_b in range(n_cols_b):  # columns of b
        match0 = match_rows[col_b, 0]
        match1 = match_rows[col_b, 1]
        if match1 == match0:
            continue
        for ax in range(b_rank - cut_b):
            c_block_shape.ptr[cut_a + ax] = b_shape_keep[col_b, ax]
        n = block_dim_b_keep[col_b]
        for row_a_sort_idx in range(match0, match1):  # rows of a
            row_a = row_a_sort[row_a_sort_idx]
            # find common inner indices
            inds_contr_count = _iter_common_sorted(a_qdata_contr, a_slices[row_a], a_slices[row_a+1],
                                             b_qdata_contr, b_slices[col_b], b_slices[col_b+1],
                                             inds_contr)
            if inds_contr_count == 0:
                continue  # no compatible blocks for given row_a, col_b

            # sum over inner indices
            for ax in range(cut_a):
                c_block_shape.ptr[ax] = a_shape_keep[row_a, ax]
            m = block_dim_a_keep[row_a]

            i = inds_contr[0, 0]
            j = inds_contr[0, 1]
            k = block_dim_a_contr[i]
            c_block = _np_empty(c_block_shape, CALC_DTYPE_NUM)
            if CALC_DTYPE_NUM == np.NPY_DOUBLE:
                _blas_dgemm(m, n, k, a_data_ptr[i], b_data_ptr[j],
                            0., np.PyArray_DATA(c_block))
            else: # if CALC_DTYPE_NUM == np.NPY_CDOUBLE:
                _blas_zgemm(m, n, k, a_data_ptr[i], b_data_ptr[j],
                            0., np.PyArray_DATA(c_block))
            for k_contr in range(1, inds_contr_count):
                i = inds_contr[k_contr, 0]
                j = inds_contr[k_contr, 1]
                k = block_dim_a_contr[i]
                if CALC_DTYPE_NUM == np.NPY_DOUBLE:
                    _blas_dgemm(m, n, k, a_data_ptr[i], b_data_ptr[j],
                                1., np.PyArray_DATA(c_block))
                else: # if CALC_DTYPE_NUM == np.NPY_CDOUBLE:
                    _blas_zgemm(m, n, k, a_data_ptr[i], b_data_ptr[j],
                                1., np.PyArray_DATA(c_block))
            # Step 4) reshape back to tensors
            # c_block is already created in the correct shape, which is ignored by BLAS.
            for ax in range(cut_a):
                res_qdata[res_n_blocks, ax] = a_qdata_keep[row_a, ax]
            for ax in range(b_rank - cut_b):
                res_qdata[res_n_blocks, cut_a + ax] = b_qdata_keep[col_b, ax]
            res_data.append(c_block)
            res_n_blocks += 1
    # TODO: type of C???

    if DEBUG_PRINT:
        t1 = time.time()
        print("_inner loop", t1-t0)
        t0 = time.time()

    PyMem_Free(c_block_shape.ptr)
    if res_n_blocks != 0:
        # (at least one entry is non-empty, so res_qdata[keep] is also not empty)
        if res_n_blocks != res_max_n_blocks:
            res_qdata = res_qdata[:res_n_blocks, :]
        res._qdata = res_qdata
        res._qdata_sorted = True
        res._data = res_data
    if DEBUG_PRINT:
        t1 = time.time()
        print("finalize", t1-t0)
        t0 = time.time()
    return res





@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _blas_dgemm(int M, int N, int K, void* A, void* B, double beta, void* C) nogil:
    """use blas to calculate ``C = A.dot(B) + beta * C``, overwriting to C.

    Assumes (!) that A, B, C are contiguous C-style matrices of dimensions MxK, KxN , MxN.
    """
    # HACK: We want ``C = A.dot(B)``, but this is equivalent to ``C.T = B.T.dot(A.T)``.
    # reading a C-style matrix A of dimensions MxK as F-style Matrix with LD= K yields A.T
    # Thus we can use C-style A, B, C without transposing.
    cdef char * tr = 'n'
    cdef double alpha = 1.
    # fortran call of dgemm(transa, transb, M, N, K, alpha, A, LDB, B, LDB, beta, C LDC)
    # but switch A <-> B and M <-> N to transpose everything
    dgemm(tr, tr, &N, &M, &K, &alpha, <double*> B, &N, <double*> A, &K, &beta, <double*> C, &N)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _blas_zgemm(int M, int N, int K, void* A, void* B, double complex beta, void* C) nogil:
    """use blas to calculate C = A.B + beta C, overwriting to C

    Assumes (!) that A, B, C are contiguous F-style matrices of dimensions MxK, KxN , MxN."""
    cdef char * tr = 'n'
    cdef double complex alpha = 1.
    # switch A <-> B and M <-> N to transpose everything: c.f. _blas_dgemm
    zgemm(tr, tr, &N, &M, &K, &alpha, <double complex*> B, &N, <double complex*> A, &K, &beta,
          <double complex*> C, &N)


# TODO: _inner_worker !?!
