"""Optimization of charges.py and np_conserved.py

This module is written in Cython, such that it can be compiled.
It implements some functions and classes with the same interface as np_conserved.py/charges.py.


``tenpy.linalg.__init__.py`` tries to import the compiled module and uses the
functions/classes defined here to overwrite those written in pure Python.
If the file could not be compiled, it just uses the non-compiled Cython version.

"""
# Copyright 2018 TeNPy Developers

DEF DEBUG_PRINT = 0  # TODO XXX

# TODO memory leak if using the `np.ndarray[type, ndim=2]` variables with zero second dimension!!!
# the same memory leak appears for memory views `type[:, :]`
# see https://github.com/cython/cython/issues/2828
# TODO: check if this code has memory leaks.

import numpy as np
cimport numpy as np
cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, postincrement as inc  # TODO
from libc.string cimport memcpy

import bisect
import warnings
import time  # TODO
import itertools


import scipy.linalg
from scipy.linalg import blas as BLAS  # python interface to BLAS
from scipy.linalg.cython_blas cimport dgemm, zgemm

from ..tools.misc import lexsort, inverse_permutation  # TODO: get rid of this?
from ..tools import optimization

np.import_array()

QTYPE = np.int_             # numpy dtype for the charges
ctypedef np.int_t QTYPE_t   # compile time type for QTYPE
cdef int QTYPE_num = np.NPY_LONG # == np.dtype(QTYPE).num

ctypedef np.intp_t intp_t   # compile time type for np.intp
cdef int intp_num = np.NPY_INTP

assert QTYPE_num == np.dtype(QTYPE).num
assert intp_num == np.dtype(np.intp).num

# We can not ``from . import np_conserved`` because
# importing np_conserved requires this cython module to be imported.
# These modules require python anyways, so it doesn't hurt to import them later on.
# Therefore, the following variables are set to the correpsonding modules in
# tenpy/linalg/__init__.py once the modules have been imported in the correct order.
_np_conserved = None  # tenpy.linalg.np_conserved
_charges = None       # tenpy.linalg.charges

# ################################# #
# helper functions                  #
# ################################# #


cdef inline np.ndarray _np_empty(np.PyArray_Dims dims, int type_):
    return <np.ndarray>np.PyArray_EMPTY(dims.len, dims.ptr, type_, 0 )

cdef inline np.ndarray _np_empty_1D(intp_t dim, int type_):
    return <np.ndarray>np.PyArray_SimpleNew(1, [dim], type_)

cdef inline np.ndarray _np_empty_2D(intp_t dim1, intp_t dim2, int type_):
    return <np.ndarray>np.PyArray_SimpleNew(2, [dim1, dim2], type_)

cdef inline np.ndarray _np_zeros_1D(intp_t dim, int type_):
    return <np.ndarray>np.PyArray_ZEROS(1, [dim], type_, 0)

cdef inline np.ndarray _np_zeros_2D(intp_t dim1, intp_t dim2, int type_):
    return <np.ndarray>np.PyArray_ZEROS(2, [dim1, dim2], type_, 0)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef np.ndarray make_stride(tuple shape, bint cstyle=1):
    """Create the strides for C-style arrays with a given shape.

    Equivalent to ``x = np.zeros(shape); return np.array(x.strides, np.intp) // x.itemsize``.
    """
    cdef intp_t a, d, L = len(shape), stride = 1
    cdef np.ndarray[intp_t, ndim=1] res = _np_empty_1D(L, intp_num)
    if cstyle:
        res[L-1] = 1
        for a in range(L-1, 0, -1):
            d = shape[a]
            stride *= d
            res[a-1] = stride
    else:
        res[0] = 1
        for a in range(0, L-1):
            d = shape[a]
            stride *= d
            res[a+1] = stride
    return res


cdef void _batch_accumulate_gemm(vector[intp_t] batch_slices,
                            vector[idx_tuple] batch_m_n,
                            vector[idx_tuple] inds_contr,
                            vector[intp_t] block_dim_a_contr,
                            vector[void*] a_data_ptr,
                            vector[void*] b_data_ptr,
                            vector[void*] c_data_ptr,
                            int calc_dtype_num
                            ) nogil:
    cdef size_t b, batch_count = batch_m_n.size()
    cdef intp_t x, batch_beg, batch_end,
    cdef intp_t i, j, m, n, k
    cdef idx_tuple i_j, m_n

    for b in range(batch_count): # TODO parallelize !?
        m_n = batch_m_n[b]
        m = m_n.first
        n = m_n.second
        batch_beg = batch_slices[b]
        batch_end = batch_slices[b+1]
        i_j = inds_contr[batch_beg]
        i = i_j.first
        j = i_j.second
        k = block_dim_a_contr[i]
        if calc_dtype_num == np.NPY_DOUBLE:
            _blas_dgemm(m, n, k, a_data_ptr[i], b_data_ptr[j],
                        0., c_data_ptr[b])
        else: # if calc_dtype_num == np.NPY_CDOUBLE:
            _blas_zgemm(m, n, k, a_data_ptr[i], b_data_ptr[j],
                        0., c_data_ptr[b])
        for x in range(batch_beg + 1, batch_end):
            i_j = inds_contr[x]
            i = i_j.first
            j = i_j.second
            k = block_dim_a_contr[i]
            if calc_dtype_num == np.NPY_DOUBLE:
                _blas_dgemm(m, n, k, a_data_ptr[i], b_data_ptr[j],
                            1., c_data_ptr[b])
            else: # if calc_dtype_num == np.NPY_CDOUBLE:
                _blas_zgemm(m, n, k, a_data_ptr[i], b_data_ptr[j],
                            1., c_data_ptr[b])


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


cdef void _blas_zgemm(int M, int N, int K, void* A, void* B, double complex beta, void* C) nogil:
    """use blas to calculate C = A.B + beta C, overwriting to C

    Assumes (!) that A, B, C are contiguous C-style matrices of dimensions MxK, KxN , MxN."""
    cdef char * tr = 'n'
    cdef double complex alpha = 1.
    # switch A <-> B and M <-> N to transpose everything: c.f. _blas_dgemm
    zgemm(tr, tr, &N, &M, &K, &alpha, <double complex*> B, &N, <double complex*> A, &K, &beta,
          <double complex*> C, &N)



cdef void _sliced_strided_copy(char* dest_data, intp_t* dest_strides,
                          char* src_data, intp_t* src_strides,
                          intp_t* slice_shape, intp_t ndim) nogil:
    """Implementation of :func:`sliced_copy`.

    `src_beg` and `dest_beg` are [0, 0, ...] and the arrays are given by pointers & strides"""
    cdef intp_t width, i, j, k, d0, d1, d2, s0, s1, s2, l0, l1, l2
    if ndim < 1:
        return
    # explicitly unravel for up to 3 dimensions
    d0 = dest_strides[0]
    s0 = src_strides[0]
    l0 = slice_shape[0]
    if ndim == 1:
        width = l0 * s0
        memcpy(dest_data, src_data, width)
        return
    d1 = dest_strides[1]
    s1 = src_strides[1]
    l1 = slice_shape[1]
    if ndim == 2:
        width = l1 * s1
        for i in range(l0):
            memcpy(&dest_data[i*d0] , &src_data[i*s0], width)
        return
    d2 = dest_strides[2]
    s2 = src_strides[2]
    l2 = slice_shape[2]
    if ndim == 3:
        width = l2 * s2
        for i in range(l0):
            for j in range(l1):
                memcpy(&dest_data[i*d0 + j*d1] , &src_data[i*s0 + j*s1], width)
        return
    # ndim >= 4: from here on recursively
    # go down by 3 dimensions at once
    for i in range(l0):
        for j in range(l1):
            for k in range(l2):
                _sliced_strided_copy(&dest_data[i*d0 + j*d1 + k*d2], &dest_strides[3],
                                     &src_data[i*s0 + j*s1 + k*s2], &src_strides[3],
                                     &slice_shape[3], ndim-3)




# ################################# #
# replacements for charges.py       #
# ################################# #

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _make_valid_charges_1D(QTYPE_t[::1] chinfo_mod, QTYPE_t[::1] charges) nogil:
    """same as ChargeInfo.make_valid for 1D charges, but works in place"""
    cdef intp_t qnumber = chinfo_mod.shape[0]
    cdef int j
    cdef QTYPE_t qm, q
    for j in range(qnumber):
        qm = chinfo_mod[j]
        if qm != 1:
            q = charges[j] % qm
            if q < 0:  # correct for C-modulo opposed to python modulo
                q += qm
            charges[j] = q
    # done

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _make_valid_charges_2D(QTYPE_t[::1] chinfo_mod, QTYPE_t[:, ::1] charges) nogil:
    """same as ChargeInfo.make_valid for 2D charges, but works in place"""
    cdef intp_t qnumber = chinfo_mod.shape[0]
    cdef intp_t L = charges.shape[0]
    cdef intp_t i, j
    cdef QTYPE_t qm, q
    for j in range(qnumber):
        qm = chinfo_mod[j]
        if qm != 1:
            for i in range(L):
                q = charges[i, j] % qm
                if q < 0:
                    q += qm
                charges[i, j] = q
    # done


@cython.binding(True)
def ChargeInfo_make_valid(self, charges=None):
    """Take charges modulo self.mod.

    Parameters
    ----------
    charges : array_like or None
        1D or 2D array of charges, last dimension `self.qnumber`
        None defaults to trivial charges ``np.zeros(qnumber, dtype=QTYPE)``.

    Returns
    -------
    charges :
        A copy of `charges` taken modulo `mod`, but with ``x % 1 := x``
    """
    cdef intp_t qnumber = self._qnumber
    if charges is None:
        return _np_zeros_1D(qnumber, QTYPE_num)
    cdef np.ndarray charges_ = np.array(charges, dtype=QTYPE, copy=True, order="C")
    if charges_.ndim == 1:
        assert (charges_.shape[0] == qnumber)
        if qnumber == 0:
            return _np_zeros_1D(qnumber, QTYPE_num)
        _make_valid_charges_1D(self._mod, charges_)
        return charges_
    elif charges_.ndim == 2:
        assert (charges_.shape[1] == qnumber)
        if qnumber == 0:
            return _np_zeros_2D(charges_.shape[0], qnumber, QTYPE_num)
        _make_valid_charges_2D(self._mod, charges_)
        return charges_
    raise ValueError("wrong dimension of charges " + str(charges))


@cython.binding(True)
def ChargeInfo_check_valid(self, charges):
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
    cdef QTYPE_t[::1] chinfo_mod = self._mod
    cdef intp_t i, j, L = charges.shape[0], qnumber = self._qnumber
    if qnumber == 0:
        return True
    cdef QTYPE_t[:, :] charges_ = charges
    cdef QTYPE_t q, x
    for j in range(qnumber):
        q = chinfo_mod[j]
        if q == 1:
            continue
        for i in range(L):
            x = charges_[i, j]
            if x < 0 or x >= q:
                return False
    return True


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.binding(True)
def LegPipe__init_from_legs(self, bint sort=True, bint bunch=True):
    """Calculate ``self.qind``, ``self.q_map`` and ``self.q_map_slices`` from ``self.legs``.

    `qind` is constructed to fullfill the charge fusion rule stated in the class doc-string.
    """
    # this function heavily uses numpys advanced indexing, for details see
    # `http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html`_
    cdef intp_t nlegs = self.nlegs
    cdef QTYPE_t[::1] chinfo_mod = self.chinfo._mod
    cdef intp_t qnumber = chinfo_mod.shape[0]
    cdef intp_t i, j
    cdef QTYPE_t sign
    cdef intp_t a
    self._strides = make_stride(self.subqshape, 1)  # save for :meth:`_map_incoming_qind`

    # create a grid to select the multi-index sector
    grid = np.indices(self.subqshape, np.intp)
    # grid is an array with shape ``(nlegs,) + qshape``,
    # with grid[li, ...] = {np.arange(qshape[li]) increasing in the li-th direcion}
    # collapse the different directions into one.
    cdef intp_t[:, ::1] grid2 = grid.reshape(nlegs, -1)
        # *this* is the actual `reshaping`
    # *columns* of grid are now all possible cominations of qindices.
    cdef intp_t nblocks = grid2.shape[1]  # number of blocks in the pipe = np.product(qshape)
    cdef np.ndarray[intp_t, ndim=2, mode="c"] q_map = _np_empty_2D(nblocks, 3 + nlegs, intp_num)
    # determine q_map -- it's essentially the grid.
    q_map[:, 3:] = grid2.T  # transpose -> rows are possible combinations.
    # q_map[:, :3] is initialized after sort/bunch.

    # determine block sizes
    cdef np.ndarray[intp_t, ndim=1] blocksizes = np.ones((nblocks,), dtype=np.intp)
    cdef intp_t[::1] leg_bs
    for i in range(nlegs):
        leg_bs = self.legs[i]._get_block_sizes()
        for j in range(nblocks):
            blocksizes[j] *= leg_bs[grid2[i, j]]

    # calculate total charges
    cdef np.ndarray charges = _partial_qtotal(chinfo_mod, self.legs, grid2.T, self.qconj)
    if sort and qnumber > 0:
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
    self.sorted = sort or (qnumber == 0)
    self._set_block_sizes(blocksizes)  # sets self.slices
    cdef intp_t[::1] slices = self.slices
    for j in range(nblocks):
        q_map[j, 0] = slices[j]
        q_map[j, 1] = slices[j+1]

    cdef intp_t[::1] idx
    if bunch:
        # call LegCharge.bunch(), which also calculates new blocksizes
        idx, bunched = _charges.LegCharge.bunch(self)
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
    self.q_map_slices = np.asarray(idx, dtype=np.intp)


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef np.ndarray _find_row_differences(np.ndarray qflat):
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
    cdef int i, j, n=1, L = qflat.shape[0], M = qflat.shape[1]
    cdef bint rows_equal = False
    cdef np.ndarray[QTYPE_t, ndim=2] qflat_c = qflat
    cdef np.ndarray[np.intp_t, ndim=1] res = _np_empty_1D(max(L + 1, 2), intp_num)
    res[0] = 0
    for i in range(1, L):
        rows_equal = True
        for j in range(M):
            if qflat_c[i-1, j] != qflat_c[i, j]:
                rows_equal = False
                break
        if not rows_equal:
            res[n] = i
            n += 1
    res[n] = L
    return res[:n+1]


@cython.wraparound(False)
@cython.boundscheck(False)
cdef np.ndarray _partial_qtotal(QTYPE_t[::1] chinfo_mod, legs, intp_t[:, :] qdata, QTYPE_t qconj,
                                QTYPE_t[::1] add_qtotal=None):
    """Calculate qtotal of a part of the legs of a npc.Array.

    Equivalent to:
        charges = np.sum([l.get_charge(qi) for l, qi in zip(legs, qdata.T)], axis=0)
        return chinfo.make_valid(charges * qconj + add_qtotal)
    """
    cdef intp_t nlegs = qdata.shape[1]
    cdef intp_t qnumber = chinfo_mod.shape[0]
    if qnumber == 0:
        return _np_zeros_2D(qdata.shape[0], qnumber, QTYPE_num)
    cdef np.ndarray[QTYPE_t, ndim=2] res = _np_zeros_2D(qdata.shape[0], qnumber, QTYPE_num)
    cdef intp_t a, k, qi
    #  cdef LegCharge leg
    cdef np.ndarray[QTYPE_t, ndim=2] charges
    cdef QTYPE_t sign, q
    for a in range(nlegs):
        leg = legs[a]
        sign = leg.qconj
        sign = sign * qconj
        charges = leg.charges
        for i in range(qdata.shape[0]):
            qi = qdata[i, a]
            for k in range(qnumber):
                res[i, k] += charges[qi, k] * sign
    if add_qtotal is not None:
        for k in range(qnumber):
            q = add_qtotal[k]
            for i in range(res.shape[0]):
                res[i, k] += q
    _make_valid_charges_2D(chinfo_mod, res)
    return res


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef void _sliced_copy(np.ndarray dest, intp_t[::1] dest_beg, np.ndarray src, intp_t[::1] src_beg,
                       intp_t[::1] slice_shape) except *:
    """Copy slices from `src` into slices of `dest`.


    *Assumes* that `src` and `dest` are C-contiguous (strided) Arrays.

    Equivalent to ::

        assert dest.ndim == src.ndim == len(dest_beg) == len(src_beg) == len(slice_shape)
        dst_sl = tuple([slice(i, i+d) for (i, d) in zip(dest_beg, slice_shape)])
        src_sl = tuple([slice(i, i+d) for (i, d) in zip(src_beg, slice_shape)])
        dest[dst_sl] = src[src_sl]

    For example ``dest[0:4, 2:5] = src[1:5, 0:3]`` is equivalent to
    ``sliced_copy(dest, [0, 2], src, [1, 0], [4, 3])``

    Parameters
    ----------
    dest : array
        The array to copy into.
        Assumed to be C-contiguous.
    dest_beg : intp[ndim]
        Entries are start of the slices used for `dest`
    src : array
        The array to copy from.
        Assumed to be C-contiguous and of same dtype and dimension as `dest`.
    src_beg : intp[ndim]
        Entries are start of the slices used for `src`
    slice_shape : intp[ndim]
        The lenght of the slices.
    """
    cdef char *dest_data = np.PyArray_BYTES(dest)
    cdef char *src_data = np.PyArray_BYTES(src)
    cdef intp_t *dest_strides = np.PyArray_STRIDES(dest),
    cdef intp_t *src_strides = np.PyArray_STRIDES(src)
    cdef intp_t ndim = np.PyArray_NDIM(dest)
    if not (dest_beg.shape[0] == ndim == np.PyArray_NDIM(src)
            == src_beg.shape[0] == slice_shape.shape[0]):
        raise ValueError("wrong dimensions")
    if not (dest_strides[ndim-1] == np.PyArray_ITEMSIZE(dest)
            == np.PyArray_ITEMSIZE(src) == src_strides[ndim-1]):
        raise ValueError("different data types or not C-contiguous")
    #  add offset of *_beg to *_data.
    cdef intp_t i, j = 0
    for i in range(ndim):
        j += dest_beg[i] * dest_strides[i]
    dest_data = &dest_data[j]
    j = 0
    for i in range(ndim):
        j += src_beg[i] * src_strides[i]
    src_data = &src_data[j]
    _sliced_strided_copy(dest_data, dest_strides, src_data, src_strides, &slice_shape[0], ndim)

# ############################################### #
# replacements for np_conserved.Array methods     #
# ############################################### #


def _combine_legs_worker(self, list combine_legs, list new_axes, list pipes):
    """The main work of :meth:`Array.combine_legs`: create a copy and reshape the data blocks.

    Assumes standard form of parameters.

    Parameters
    ----------
    self : Array
        The array where legs are being combined.
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
    if DEBUG_PRINT:
        print("_combine_legs_worker")
        t0 = time.time()
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
    cdef intp_t res_rank = res.rank
    cdef intp_t self_stored_blocks = self.stored_blocks
    if self_stored_blocks == 0:
        res._data = []
        res._qdata = _np_empty_2D(0, res.rank, intp_num)
        return res
    # TODO: handle self_stored_blocks == 1 separately for optimization
    non_new_axes_ = [i for i in range(res.rank) if i not in new_axes]
    cdef np.ndarray[np.intp_t, ndim=1] non_new_axes = np.array(non_new_axes_, dtype=np.intp)
    if DEBUG_PRINT:
        t1 = time.time()
        print("init", t1-t0)
        t0 = time.time()

    # map `self._qdata[:, combine_leg]` to `pipe.q_map` indices for each new pipe
    qmap_inds = [
        p._map_incoming_qind(self._qdata[:, cl]) for p, cl in zip(pipes, combine_legs)
    ]

    # get new qdata
    cdef np.ndarray[np.intp_t, ndim=2, mode="c"] qdata = _np_empty_2D(self_stored_blocks, res_rank, intp_num)
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
    qdata = qdata[sort]
    old_data = [self._data[s] for s in sort]
    qmap_inds = [qm[sort] for qm in qmap_inds]
    # divide into parts, which give a single new block
    cdef np.ndarray[np.intp_t, ndim=1] diffs = _find_row_differences(qdata)
    # including the first and last row
    if DEBUG_PRINT:
        t1 = time.time()
        print("get new qdata", t1-t0)
        t0 = time.time()

    # now the hard part: map data
    cdef list data = []
    cdef list slices = [slice(None)] * res.rank  # for selecting the slices in the new blocks
    # iterate over ranges of equal qindices in qdata
    cdef np.ndarray new_block, old_block
    cdef np.ndarray[np.intp_t, ndim=1] qindices
    cdef intp_t beg, end, bi, j, old_data_idx, qi
    cdef np.ndarray[np.intp_t, ndim=2] q_map
    cdef tuple sl
    cdef int npipes = len(combine_legs)
    for bi in range(diffs.shape[0]-1):
        beg = diffs[bi]
        end = diffs[bi+1]
        qindices = qdata[beg]
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
    res._qdata = qdata[diffs[:-1]]  # (keeps the dimensions)
    res._qdata_sorted = True
    res._data = data
    if DEBUG_PRINT:
        t1 = time.time()
        print("reshape loop", t1-t0)
        t0 = time.time()
    return res


def _split_legs_worker(self, list split_axes_, float cutoff):
    """The main work of split_legs: create a copy and reshape the data blocks.

    Called by :meth:`split_legs`. Assumes that the corresponding legs are LegPipes.
    """
    # calculate mappings of axes
    # in self
    cdef np.ndarray[np.intp_t, ndim=1] split_axes = np.sort(split_axes_)
    cdef int a, i, j, nsplit = split_axes.shape[0]
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
    cdef intp_t r
    cdef slice sl
    # TODO
    #  cdef LegPipe pipe
    #  cdef LegCharge leg
    cdef np.ndarray old_block, new_block

    for old_block, qdata_row in zip(self._data, tmp_qdata):
        for j in range(nonsplit_axes.shape[0]):
            new_block_shape[new_nonsplit_axes[j]] = old_block.shape[nonsplit_axes[j]]
        for j in range(nsplit):
            pipe = pipes[j]
            r = qdata_row[new_split_axes_first[j]]
            qmap_slices[j] = pipe.q_map[pipe.q_map_slices[r]:pipe.q_map_slices[r+1]]
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

ctypedef struct idx_tuple:
    intp_t first
    intp_t second


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline intp_t _iter_common_sorted_push(
        intp_t[::1] a, intp_t i_start, intp_t i_stop,
        intp_t[::1] b, intp_t j_start, intp_t j_stop,
        vector[idx_tuple]* out) nogil:
    """Find indices ``i, j`` for which ``a[i] == b[j]`` and pushes these (i,j) into `out`.

    *Assumes* that ``a[i_start:i_stop]`` and ``b[j_start:j_stop]`` are strictly ascending.
    Given that, it is equivalent to (but faster than)::

        count = 0
        for j, i in itertools.product(range(j_start, j_stop), range(i_start, i_stop)):
            if a[i] == b[j]:
                out.push_back([i, j])
                count += 1
        return count
    """
    cdef intp_t i=i_start, j=j_start, count=0
    cdef idx_tuple i_j
    while i < i_stop and j < j_stop:
        if a[i] < b[j]:
            i += 1
        elif b[j] < a[i]:
            j += 1
        else:
            #  yield i, j
            i_j.first = i
            i_j.second = j
            out.push_back(i_j) # (equivalent to C's out->push_back)
            count += 1
            i += 1
            j += 1
    return count


@cython.boundscheck(False)
@cython.wraparound(False)
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
    stride = make_stride(tuple([l.block_number for l in a.legs[cut_a:a.rank]]), 0)  # TODO: can use c-style!?!
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
@cython.wraparound(False)
cdef _tensordot_match_charges(QTYPE_t[::1] chinfo_mod,
                              a,
                              b,
                              intp_t cut_a,
                              intp_t cut_b,
                              np.ndarray a_qdata_keep,
                              np.ndarray b_qdata_keep,
                              intp_t n_rows_a,
                              intp_t n_cols_b,
                              np.ndarray qtotal
                              ):
    """Estimate number of blocks in res and get order for iteration over row_a and col_b

    Parameters
    ----------
    a_charges_keep, b_charges_match: 2D ndarray
        (Unsorted) charges of dimensions (n_rows_a, qnumber) and (n_cols_b, qnumber)

    Returns
    -------
    max_n_blocks_res : int
        Maximum number of block in the result a.dot(b)
    row_a_sort: intp_t[:]
    match_rows: intp_t[:, ::1]
        For given `col_b`, rows given by `row_a` in
        ``row_a_sort[match_rows[col_b, 0]:match_rows[col_b,1]]`` fulfill
        the charge rule, i.e.
        ``a_charges_keep[col_a, :] == b_charges_match[col_b, :]``
    """
    cdef intp_t qnumber = chinfo_mod.shape[0]
    cdef np.ndarray[np.intp_t, ndim=2] match_rows = np.empty((n_cols_b, 2), np.intp)
    # This is effectively a more complicated version of _iter_common_sorted....
    if qnumber == 0:  # special case no restrictions due to charge
        match_rows[:, 0] = 0
        match_rows[:, 1] = n_rows_a
        return n_rows_a * n_cols_b, np.arange(n_rows_a), match_rows
    # general case
    cdef np.ndarray[QTYPE_t, ndim=2] a_charges_keep = _partial_qtotal(
        chinfo_mod, a.legs[:cut_a], a_qdata_keep, 1)
    cdef np.ndarray[QTYPE_t, ndim=2] b_charges_match = _partial_qtotal(
        chinfo_mod, b.legs[cut_b:], b_qdata_keep, -1, add_qtotal=qtotal)
    _make_valid_charges_2D(chinfo_mod, b_charges_match)
    cdef intp_t[::1] row_a_sort = np.lexsort(a_charges_keep.T)
    cdef intp_t[::1] col_b_sort = np.lexsort(b_charges_match.T)
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
            if a_charges_keep[i_s, ax] > b_charges_match[j_s, ax]:
                lexcomp = 1
                break
            elif a_charges_keep[i_s, ax] < b_charges_match[j_s, ax]:
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
                if a_charges_keep[i_s, ax] != a_charges_keep[i0_s, ax]:
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
                if b_charges_match[j_s, ax] != b_charges_match[j0_s, ax]:
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
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
    cdef QTYPE_t[::1] chinfo_mod = a.chinfo._mod
    cdef intp_t qnumber = chinfo_mod.shape[0]
    chinfo = a.chinfo
    cdef intp_t cut_a = a.rank - axes
    cdef intp_t cut_b = axes
    cdef intp_t b_rank = b.rank
    cdef intp_t res_rank = cut_a + b_rank - cut_b
    if DEBUG_PRINT:
        print("a.stored_blocks", a.stored_blocks, "b.stored_blocks", b.stored_blocks)
        t0 = time.time()
    # determine calculation type and result type
    res_dtype = np.find_common_type([a.dtype, b.dtype], [])
    prefix, _, _ = BLAS.find_best_blas_type(dtype=res_dtype)
    # we always use 64-bit float calculations....
    calc_dtype = np.dtype({'s': np.float64, 'd': np.float64,
                           'c': np.complex128, 'z': np.complex128}[prefix])
    # TODO: handle special dtypes?
    cdef int CALC_DTYPE_NUM = calc_dtype.num  # can be compared to np.NPY_DOUBLE/NPY_CDOUBLE
    cdef np.ndarray[QTYPE_t, ndim=1] qtotal = a.qtotal + b.qtotal
    _make_valid_charges_1D(chinfo_mod, qtotal)
    res = _np_conserved.Array(a.legs[:cut_a] + b.legs[cut_b:], res_dtype, qtotal)
    if a.dtype.num != CALC_DTYPE_NUM:
        a = a.astype(calc_dtype)
    if b.dtype.num != CALC_DTYPE_NUM:
        b = b.astype(calc_dtype)

    cdef np.ndarray[np.intp_t, ndim=2] a_qdata = a._qdata
    cdef np.ndarray[np.intp_t, ndim=2] b_qdata = b._qdata
    cdef list a_data = a._data, b_data = b._data
    cdef intp_t len_a_data = len(a_data)
    cdef intp_t len_b_data = len(b_data)
    cdef bint equal = 1
    cdef intp_t i, j
    # special case: a or b is zero
    if len_a_data == 0 or len_b_data == 0:  # special case: `a` or `b` is 0
        return res
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
            res._data = [np.asarray(np.tensordot(a._data[0], b._data[0], axes=axes),
                                    dtype=res_dtype)]
            c_qdata = np.zeros([1, res_rank], np.intp)
            for i in range(cut_a):
                c_qdata[0, i] = a_qdata[0, i]
            for i in range(b_rank - cut_b):
                c_qdata[0, cut_a + i] = b_qdata[0, cut_b + i]
            res._qdata = c_qdata
        #  else: zero
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
    cdef np.ndarray[np.intp_t, ndim=1] a_slices = _find_row_differences(a_qdata_keep)
    cdef np.ndarray[np.intp_t, ndim=1] b_slices = _find_row_differences(b_qdata_keep)
    # the slices divide a_data and b_data into rows and columns
    cdef intp_t n_rows_a = a_slices.shape[0] - 1
    cdef intp_t n_cols_b = b_slices.shape[0] - 1
    a_qdata_keep = a_qdata_keep[a_slices[:n_rows_a]]  # TODO: might get optimized
    b_qdata_keep = b_qdata_keep[b_slices[:n_cols_b]]

    if DEBUG_PRINT:
        t1 = time.time()
        print("find_row_differences", t1-t0)
        t0 = time.time()

    cdef np.ndarray block
    cdef vector[void*] a_data_ptr, b_data_ptr, c_data_ptr
    cdef vector[intp_t] block_dim_a_contr
    a_data_ptr.resize(len_a_data)
    b_data_ptr.resize(len_b_data)
    block_dim_a_contr.resize(len_a_data)
    cdef intp_t row_a, col_b, k_contr, ax  # indices
    cdef intp_t m, n, k   # reshaped dimensions: a_block.shape = (m, k), b_block.shape = (k,n)
    cdef intp_t[:, ::1] a_shape_keep = _np_empty_2D(n_rows_a, cut_a, intp_num)
    cdef intp_t[::1] block_dim_a_keep = _np_empty_1D(n_rows_a, intp_num)
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
            block = np.PyArray_GETCONTIGUOUS(a_data[j])
            m = np.PyArray_SIZE(block) / n
            block_dim_a_contr[j] = m  # needed for dgemm
            a_data_ptr[j] = np.PyArray_DATA(block)
            a_data[j] = block  # important to keep the arrays of the pointers alive
    cdef intp_t[:, ::1] b_shape_keep = _np_empty_2D(n_cols_b, b_rank-cut_b, intp_num)
    cdef intp_t[::1] block_dim_b_keep = _np_empty_1D(n_cols_b, intp_num)
    for col_b in range(n_cols_b):
        i = b_slices[col_b]
        block = <np.ndarray> b_data[i]
        n = 1
        for ax in range(b_rank-cut_b):
            b_shape_keep[col_b, ax] = block.shape[ax+cut_b]
            n *= block.shape[ax+cut_b]
        block_dim_b_keep[col_b] = n
        for j in range(b_slices[col_b], b_slices[col_b+1]):
            block = np.PyArray_GETCONTIGUOUS(b_data[j])
            b_data_ptr[j] = np.PyArray_DATA(block)
            b_data[j] = block  # important to keep the arrays of the pointers alive
    if DEBUG_PRINT:
        t1 = time.time()
        print("_tensordot_pre_reshape", t1-t0)
        t0 = time.time()

    # Step 3) loop over column/row of the result
    # (rows_a changes faster than cols_b, such that the resulting array is qdata lex-sorted)

    # first find output colum/row indices of the result, which are compatible with the charges
    cdef intp_t[::1] row_a_sort
    cdef intp_t[:, ::1] match_rows
    cdef intp_t res_max_n_blocks, res_n_blocks = 0
    # the main work for that is in _tensordot_match_charges
    res_max_n_blocks, row_a_sort, match_rows = _tensordot_match_charges(
        chinfo_mod, a, b, cut_a, cut_b, a_qdata_keep, b_qdata_keep, n_rows_a, n_cols_b, qtotal)
    if DEBUG_PRINT:
        t1 = time.time()
        print("_match_charges", t1-t0)
        t0 = time.time()

    cdef list res_data = []
    cdef np.ndarray[np.intp_t, ndim=2] res_qdata = np.empty((res_max_n_blocks, res_rank), np.intp)
    cdef vector[idx_tuple] inds_contr
    cdef vector[intp_t] batch_slices
    cdef vector[idx_tuple] batch_m_n
    cdef idx_tuple m_n
    cdef intp_t contr_count_batch=0, contr_count
    #  (for the size just estimate the maximal number of blocks to be contracted at once)
    batch_slices.push_back(contr_count_batch)
    cdef intp_t match0, match1
    cdef intp_t row_a_sort_idx
    cdef np.ndarray c_block
    cdef np.PyArray_Dims c_block_shape
    c_block_shape.len = res_rank
    c_block_shape.ptr = <np.npy_intp*>PyMem_Malloc(res_rank * sizeof(np.npy_intp))
    if not c_block_shape.ptr:
        raise MemoryError

    # the inner loop finding the blocks to be contracted
    for col_b in range(n_cols_b):  # columns of b
        match0 = match_rows[col_b, 0]
        match1 = match_rows[col_b, 1]
        if match1 == match0:
            continue
        for ax in range(b_rank - cut_b):
            c_block_shape.ptr[cut_a + ax] = b_shape_keep[col_b, ax]
        m_n.second = block_dim_b_keep[col_b]
        for row_a_sort_idx in range(match0, match1):  # rows of a
            row_a = row_a_sort[row_a_sort_idx]
            # find common inner indices
            contr_count = _iter_common_sorted_push(a_qdata_contr, a_slices[row_a], a_slices[row_a+1],
                                                   b_qdata_contr, b_slices[col_b], b_slices[col_b+1],
                                                   &inds_contr)
            if contr_count == 0:
                continue  # no compatible blocks for given row_a, col_b
            contr_count_batch += contr_count
            batch_slices.push_back(contr_count_batch)
            # we need to sum over inner indices
            # create output block
            for ax in range(cut_a):
                c_block_shape.ptr[ax] = a_shape_keep[row_a, ax]
            m_n.first = block_dim_a_keep[row_a]

            c_block = _np_empty(c_block_shape, CALC_DTYPE_NUM)
            c_data_ptr.push_back(np.PyArray_DATA(c_block))
            batch_m_n.push_back(m_n)

            # the actual contraction is done below in _batch_accumulate_gemm

            # Step 4) reshape back to tensors
            # c_block is already created in the correct shape, which is ignored by BLAS.
            for ax in range(cut_a):
                res_qdata[res_n_blocks, ax] = a_qdata_keep[row_a, ax]
            for ax in range(b_rank - cut_b):
                res_qdata[res_n_blocks, cut_a + ax] = b_qdata_keep[col_b, ax]
            res_data.append(c_block)
            res_n_blocks += 1

    if DEBUG_PRINT:
        t1 = time.time()
        print("pack inner loop gemm", t1-t0)
        t0 = time.time()

    # Step 3.2) the actual matrix-matrix multiplications
    with nogil:
        _batch_accumulate_gemm(batch_slices,
                               batch_m_n,
                               inds_contr,
                               block_dim_a_contr,
                               a_data_ptr,
                               b_data_ptr,
                               c_data_ptr,
                               CALC_DTYPE_NUM)

    if DEBUG_PRINT:
        t1 = time.time()
        print("inner loop gemm", t1-t0)
        print("had ", inds_contr.size(), "contractions into ", res_n_blocks, "new blocks")
        t0 = time.time()

    PyMem_Free(c_block_shape.ptr)
    if res_n_blocks != 0:
        # (at least one entry is non-empty, so res_qdata[keep] is also not empty)
        if res_n_blocks != res_max_n_blocks:
            res_qdata = res_qdata[:res_n_blocks, :]
        res._qdata = res_qdata
        res._qdata_sorted = True
        res._data = res_data
        if res_dtype.num != CALC_DTYPE_NUM:
            res.dtype = calc_dtype
            res = res.astype(res_dtype)
    if DEBUG_PRINT:
        t1 = time.time()
        print("finalize", t1-t0)
        print("res.stored_blocks", res_n_blocks)
        t0 = time.time()
    return res
# TODO: _inner_worker !?!
# TODO: _svd_workder !?!
