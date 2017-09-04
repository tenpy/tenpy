
cimport numpy as np
ctypedef np.int_t QTYPE_t   # compile time type for QTYPE

cdef class ChargeInfo(object):
    cdef readonly int qnumber
    cdef readonly np.ndarray mod
    cdef readonly list names
    cpdef np.ndarray make_valid(ChargeInfo self, charges=?)
    cdef np.ndarray[QTYPE_t,ndim=1] _make_valid_1D(ChargeInfo self,
                                                   np.ndarray[QTYPE_t,ndim=1] charges)
    cdef np.ndarray[QTYPE_t,ndim=2] _make_valid_2D(ChargeInfo self,
                                                   np.ndarray[QTYPE_t,ndim=2] charges)

cdef class LegCharge(object):
    cdef readonly int ind_len
    cdef readonly int block_number
    cdef readonly ChargeInfo chinfo
    cdef readonly np.ndarray slices
    cdef readonly np.ndarray charges
    cdef readonly int qconj
    cdef public bint sorted
    cdef public bint bunched
    cpdef LegCharge conj(LegCharge self)
    cpdef bint is_blocked(self)
    cpdef void test_contractible(LegCharge self, LegCharge other) except *
    cpdef void test_equal(LegCharge self, LegCharge other) except *
    cpdef slice get_slice(LegCharge self, int qindex)
    cpdef void _set_charges(LegCharge self, np.ndarray[QTYPE_t,ndim=2] charges)
    cpdef void _set_slices(LegCharge self, np.ndarray[np.intp_t,ndim=1] slices)
    cpdef _set_block_sizes(self, block_sizes)
    cpdef _get_block_sizes(self)
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
    cpdef LegPipe conj(self)
    cdef void _init_from_legs(LegPipe self, bint sort=?, bint bunch=?) except *
    cpdef void __setstate__(LegPipe self, tuple state)

cdef np.ndarray _c_find_row_differences(np.ndarray[QTYPE_t,ndim=2] qflat)
cdef np.ndarray[QTYPE_t,ndim=2] _partial_qtotal(ChargeInfo chinfo,
                                                list legs,
                                                np.ndarray[np.intp_t, ndim=2] qdata)
