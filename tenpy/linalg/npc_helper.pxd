# Copyright 2018 TeNPy Developers
cimport numpy as np
ctypedef np.int_t QTYPE_t   # compile time type for QTYPE

cdef class ChargeInfo(object):
    cdef readonly int qnumber
    cdef readonly np.ndarray mod
    cdef readonly list names
    cpdef np.ndarray make_valid(ChargeInfo self, charges=?)
    cdef np.ndarray[QTYPE_t,ndim=1] _make_valid_1D(ChargeInfo self,
                                                   np.ndarray charges)
    cdef np.ndarray[QTYPE_t,ndim=2] _make_valid_2D(ChargeInfo self,
                                                   np.ndarray charges)
    cpdef void test_sanity(ChargeInfo self) except *
    cpdef tuple __getstate__(ChargeInfo self)
    cpdef void __setstate__(ChargeInfo self, tuple state)

cdef class LegCharge(object):
    cdef readonly int ind_len
    cdef readonly int block_number
    cdef readonly ChargeInfo chinfo
    cdef readonly np.ndarray slices
    cdef readonly np.ndarray charges
    cdef readonly int qconj
    cdef public bint sorted
    cdef public bint bunched
    cdef LegCharge copy(LegCharge self)
    cpdef void test_sanity(LegCharge self) except *
    cpdef LegCharge conj(LegCharge self)
    cpdef bint is_blocked(self)
    cpdef void test_contractible(LegCharge self, LegCharge other) except *
    cpdef void test_equal(LegCharge self, LegCharge other) except *
    cpdef slice get_slice(LegCharge self, int qindex)
    cpdef void _set_charges(LegCharge self, np.ndarray charges)
    cpdef void _set_slices(LegCharge self, np.ndarray slices)
    cpdef _set_block_sizes(self, block_sizes)
    cpdef _get_block_sizes(self)
    cpdef tuple __getstate__(LegCharge self)
    cpdef void __setstate__(LegCharge self, tuple state)


cdef class LegPipe(LegCharge):
    cdef readonly int nlegs
    cdef readonly tuple legs
    cdef readonly tuple subshape
    cdef readonly tuple subqshape
    cdef readonly np.ndarray q_map
    cdef readonly list q_map_slices
    cdef readonly np.ndarray _perm
    cdef readonly np.ndarray _strides
    cdef LegPipe copy(LegPipe self)
    cpdef void test_sanity(LegPipe self) except *
    cpdef LegPipe conj(self)
    cdef void _init_from_legs(LegPipe self, bint sort=?, bint bunch=?) except *
    cpdef tuple __getstate__(LegPipe self)
    cpdef void __setstate__(LegPipe self, tuple state)

cdef np.ndarray _c_find_row_differences(np.ndarray qflat)
cdef np.ndarray _partial_qtotal(ChargeInfo chinfo,
                                                list legs,
                                                np.ndarray qdata)
