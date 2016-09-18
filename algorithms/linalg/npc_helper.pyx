#cython: linetrace=False
#cython: profile=False
#distutils: define_macros=CYTHON_TRACE=0
#cython: initializedcheck=False

DEF TENSORDOT_TIMING = 0
DEF TENSORDOT_VERBOSE = 0
DEF COLLECT_MKN_STATS = 0 #Collect stats and timing on gemm calls. Appends (M*K*N, took) for each call in mkn_stats
DEF USE_DRESDEN = 1 #Direct calls to BLAS

import numpy as np
import np_conserved as npc
import time
import itertools
import sys
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, postincrement as inc

cimport numpy as np
cimport cython
np.import_array()

from algorithms.linalg.cblas cimport *

from cython.parallel import parallel, prange, threadid
IF PARALLEL_TDOT:
	from cython.parallel cimport *
	cimport openmp

dot = np.dot
np_inner = np.inner
np_vdot = np.vdot

cdef int tensordot_threads = 1

def set_tensordot_threads(n):
	global tensordot_threads
	tensordot_threads = n

def get_tensordot_threads():
	global tensordot_threads
	return tensordot_threads


cdef list mkn_stats
cdef long long num_dot
cdef long long mkn_grain

global mkn_stats
mkn_stats = []
global num_dot
num_dot = 0
global mkn_grain
mkn_grain = 10

def get_mkn_stats():
	global mkn_stats
	return mkn_stats

def init_mkn_stats():
	global mkn_stats
	mkn_stats = []
	global num_dot
	num_dot = 0

def set_mkn_grain(n):
	global mkn_grain
	mkn_grain = n




ctypedef np.float_t c_float
ctypedef np.int_t c_int
ctypedef np.uint_t c_uint
ctypedef np.intp_t c_intp

cdef extern from "stdlib.h" nogil:
	void *memcpy(void *dst, void *src, long n)

cdef inline np.ndarray empty(np.PyArray_Dims dims, int type):
	return <np.ndarray>np.PyArray_EMPTY(dims.len, dims.ptr, type, 0 )

cdef inline np.ndarray zeros(np.PyArray_Dims dims, int type):
	return <np.ndarray>np.PyArray_ZEROS(dims.len, dims.ptr, type, 0 )
			
################################################################################
################################################################################

"""
def test():
	print "Testing type!"
	cdef np.PyArray_Dims dims1
	dims1.len = 1
	dims1.ptr = [3]
	cdef np.ndarray[c_int, ndim=1, mode='c'] C = empty(dims1, np.NPY_LONG)
	cdef np.ndarray[c_uint, ndim=1, mode="c"] A = empty(dims1, np.NPY_ULONG)

test()
"""


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def tensordot(a, b, axes = 2):
	""" Tensor contraction of npc arrays a, b. Identical usage as numpy.

	"""

	if TENSORDOT_TIMING:
		t0 = time.time()
	
	cdef c_intp na, nb
	cdef list axes_a, axes_b

	# Bring all axes specification to standard form
	try:
		iter(axes)
	except:
		axes_a = range(-axes,0)
		axes_b = range(0,axes)
		na = nb = axes
	else:
		if type(axes[0])!=list and type(axes[0])!=tuple:
			axes_a = [ a.get_index(axes[0]) ]
			na = 1
		else:
			na = len(axes[0])
			axes_a = [ a.get_index(m) for m in axes[0] ]

		if type(axes[1])!=list and type(axes[1])!=tuple:
			axes_b = [ b.get_index(axes[1]) ]
			nb = 1
		else:
			nb = len(axes[1])
			axes_b = [ b.get_index(m) for m in axes[1] ]

	# axes_a,b are now each list (same len) of indices to contract (na,b are the number of indices)


	###
	#  as_, bs			: shape of a, b
	#  nda, ndb, ncd	: rank of a, b, c = a.b
	#  na = nb = nin	: number of legs being contracted
	#  nao = nda - nin	: number of uncontracted outer legs
	#  nbo = ndb - nin

	cdef np.ndarray[c_intp, ndim=1] as_ = a.shape #SHAPE
	cdef c_intp nda = a.rank
	cdef np.ndarray[c_intp, ndim=1] bs = b.shape #SHAPE
	cdef c_intp ndb = b.rank
	
	cdef int equal = 1
	cdef c_uint k
	if (na != nb):
		equal = 0
	else:
		for k in range(na):
			axes_a[k] = axes_a[k]%nda
			axes_b[k] = axes_b[k]%ndb			
			if as_[axes_a[k]] != bs[axes_b[k]]:
				equal = 0
				break
	if not equal:
		raise ValueError, "shape-mismatch for sum"
		# %s %s %s %s"%(a.shape, axes_a, b.shape, axes_b)

	cdef c_intp i, r, p
	cdef c_intp ndc, nao, nbo, nin
	cdef np.ndarray ta, tb, tc
	nin = na

	# Move the axes to sum over to the end of "a"
	# and to the front of "b"
	cdef list notin_a = [k for k in range(nda) if k not in axes_a]
	cdef list newaxes_a = notin_a + axes_a
	cdef list olda = [as_[axis] for axis in notin_a]
	nao = nda - nin
	
	cdef list notin_b = [k for k in range(ndb) if k not in axes_b]
	cdef list newaxes_b = axes_b + notin_b
	cdef list oldb = [bs[axis] for axis in notin_b]
	nbo = ndb - nin	
	ndc = nao + nbo
	
	if TENSORDOT_TIMING:
		print "Axes", time.time() - t0
		t0 = time.time()
		
	if nao == 0 and nbo == 0:
		return npc.inner(a, b, [ axes_a, axes_b])
		
#	print a.q_dat, "original a.q_dat"
#	print b.q_dat, "original b.q_dat"

	if newaxes_a!=range(nda):
		#a = a.transpose(newaxes_a)
		a = transpose_fast(a, np.array(newaxes_a))
	else:
		a.dat = [ np.PyArray_GETCONTIGUOUS(ta) for ta in a.dat]

	if newaxes_b!=range(ndb):
		#b = b.transpose(newaxes_b)
		b = transpose_fast(b, np.array(newaxes_b))
	else:
		b.dat = [ np.PyArray_GETCONTIGUOUS(tb) for tb in b.dat]

	#if TENSORDOT_TIMING:
	#	print "Transpose", time.time() - t0
	#	t0 = time.time()


	# C_lr = A_lc B_cr

	cdef np.PyArray_Dims dims1
	dims1.len = 1
	dims1.ptr = [ndc]
		
	cdef np.PyArray_Dims dims2
	dims2.len = 2
	dims2.ptr = [0, 0]
	
	cdef list dat_a = a.dat
	cdef list dat_b = b.dat
	cdef list q_ind_a = a.q_ind
	cdef list q_ind_b = b.q_ind
	cdef np.ndarray[c_uint, ndim=2] q_dat_a = a.q_dat
	cdef np.ndarray[c_uint, ndim=2] q_dat_b = b.q_dat
	cdef np.ndarray[c_int, ndim=1, mode = "c"] mod_q = a.mod_q
	cdef np.ndarray[c_int, ndim=1, mode = "c"] q_conj_a = a.q_conj
	cdef np.ndarray[c_int, ndim=1, mode = "c"] q_conj_b = b.q_conj
	cdef np.ndarray[c_int, ndim=1, mode = "c"] q_conj_c = empty(dims1, np.NPY_LONG)

	cdef c_intp num_q = a.num_q
	
	cdef c_intp l_dat_a = q_dat_a.shape[0]
	cdef c_intp l_dat_b = q_dat_b.shape[0]

	cdef object c = npc.array(ndc, dtype = None) 	# dtype set later
	c.shape = np.array(olda + oldb)#SHAPE
	c.num_q = num_q
	c.mod_q = mod_q
	c.charge = mod_onetoinf1D(a.charge + b.charge, mod_q)		##
	c.q_ind = a.q_ind[:nao] + b.q_ind[nin:]


	"""Make new labels for c	
		No labels made if either a or b doesnt have labels
		
		Any labels that would be duplicates are dropped
	"""
	cdef dict alab = a.labels
	cdef dict blab = b.labels
	cdef dict clab
	cdef list keys
	cdef c_uint v

	if alab is not None and blab is not None:
		clab = {}
		for key in alab:
			v = alab[key]
			if v < nao:
				clab[key] = v
		for key in blab:
			v = blab[key]
			if v>=nin:
				if key in clab: #Drop duplicate key
					del clab[key]
				else:
					clab[key] = v + nao - nin
		c.labels = clab
	else:
		c.labels = None

	for i in range(nao):
		q_conj_c[i] = q_conj_a[i]
	for i in range(nbo):
		q_conj_c[i + nao] = q_conj_b[i + nin]
	c.q_conj = q_conj_c
	c.sorted = False
	
	if TENSORDOT_VERBOSE > 0:
		print "c.shape", c.shape

	### ------------  Special Cases -----------------

	#Zero
	if l_dat_a == 0 or l_dat_b == 0:
		c.q_dat = np.zeros((0, ndc), np.uint)
		c.dat = []
		c.dtype = np.dtype(float)
		return c
	
	cdef np.ndarray[c_uint, ndim=2, mode="c"] q_dat_c
	cdef np.PyArray_Dims shpA, shpB, shpC
	shpA.len = shpB.len = 2
	shpA.ptr = [-1, -1]
	shpB.ptr = [-1, -1]

	# Only one block in each. Usual case for num_q = 0 (unless fractured!)
	if l_dat_a == 1 and l_dat_b == 1:
		equal = 1
		
		for k in range(nin): #check if entries match on inner
			if q_dat_a[0, nao + k]!=q_dat_b[0, k]:
				equal = 0
				break
		if equal: #So contract inner				
			ta = dat_a[0]
			tb = dat_b[0]
			
			shpC.ptr = <np.npy_intp*>PyMem_Malloc(ndc * sizeof(np.npy_intp))
			if not shpC.ptr:
				raise MemoryError
			shpC.len = ndc
			dims2.ptr = [1, ndc]
			c.q_dat = q_dat_c = empty(dims2, np.NPY_ULONG) #make q_dat entry

			for k in range(nao):			
				q_dat_c[0, k] = q_dat_a[0, k]
				shpC.ptr[k] = ta.shape[k]
			for k in range(nbo):
				q_dat_c[0, nao + k] = q_dat_b[0, k + nin]
				shpC.ptr[nao + k] = tb.shape[k + nin]
						
			r = 1
			for k in range(nin): #inner dimension
				r *= tb.shape[k]  
			
			shpA.ptr[1] = r
			shpB.ptr[0] = r

			ta = np.PyArray_Newshape(ta, &shpA, np.NPY_CORDER)
			tb = np.PyArray_Newshape(tb, &shpB, np.NPY_CORDER)
			tc = dot(ta, tb)
			
			c.dat = [np.PyArray_Newshape(tc, &shpC, np.NPY_CORDER)] 
			c.dtype = c.dat[0].dtype			
			PyMem_Free(shpC.ptr)
			return c
		else:
			c.q_dat = np.zeros((0,ndc), np.uint) #Blocks didn't overlap, so return 0
			c.dat = []
			c.dtype = np.dtype(float)
			return c
	
	if TENSORDOT_TIMING:
		print "C+SC", time.time() - t0
		t0 = time.time()
					

	#print "q_dat [leg, block]:\n", q_dat_a, "= q_dat_a\n", q_dat_b, "= q_dat_b"
	
	"""
	Pre-compute conventient data for a & b: the "Keys"

	Key structure:
	       width=  | nin     | n(a/b)o  | num_q | 1 
	     encodes=  | in inds | out inds | Q_tot | perm
		
		sorted by (lex order)
		1. total charge of the inner indices, 
		2. the outer indices
		3. the inner indices
	"""
	
	cdef c_int wa = nda + num_q + 1 #width of keys
	cdef c_int wb = ndb + num_q + 1
	
	#First I create a tranposed key, *_key_t, ommitting the perm column 		
	
	#NOTE: memview style is slower
	dims2.ptr[0] = wa-1
	dims2.ptr[1] = l_dat_a
	cdef np.ndarray[c_int, ndim=2, mode="c"] a_key_t = empty(dims2, np.NPY_LONG)
	#cdef c_int[:,::1] a_key_t = empty(dims2, np.NPY_LONG)
	
	dims2.ptr[0] = wb - 1
	dims2.ptr[1] = l_dat_b
	cdef np.ndarray[c_int, ndim=2, mode="c"] b_key_t = empty(dims2, np.NPY_LONG)
	#cdef c_int[:, ::1] b_key_t = empty(dims2, np.NPY_LONG)
	
	# Now fill in total charge over contracted legs
	cdef np.ndarray[c_int, ndim = 2, mode = "c"] q_sum_res
	#cdef c_int[:, ::1] q_sum_res
	if nin > 0:
		q_sum_res = q_sum(q_dat_a, q_ind_a, q_conj_a, nao, nda, mod_q)
		for r in range(l_dat_a): #copy
			for i in range(num_q):
				a_key_t[nda + i, r] = q_sum_res[r, i]
		
		q_sum_res = q_sum(q_dat_b, q_ind_b, -1*q_conj_b, 0, nin, mod_q)
		for r in range(l_dat_b): #copy
			for i in range(num_q):
				b_key_t[ndb+i, r] = q_sum_res[r, i]
	else:
		for r in range(l_dat_a):
			for i in range(num_q):
				a_key_t[nda + i, r] = 0
		
		for r in range(l_dat_b):
			for i in range(num_q):
				b_key_t[ndb + i, r] = 0

	#Now fill in index info (need to reverse order in w.r.t. q_dat for a)
			
	for r in range(l_dat_a):
		for i in range(nin):
			a_key_t[i, r] = q_dat_a[r, nao + i]
		for i in range(nao):
			a_key_t[nin + i, r] = q_dat_a[r, i]

	for r in range(l_dat_b):
		for i in range(ndb):
			b_key_t[i, r] = q_dat_b[r, i]

	if TENSORDOT_VERBOSE > 0:
		print a_key_t.T[:, :nda+num_q], "= unsorted a_key"
		print b_key_t.T[:, :ndb+num_q], "= unsorted b_key"
	
	#Sort by Qt > out ind > in ind 
	cdef np.ndarray[c_intp, ndim=1, mode="c"] perma = np.PyArray_LexSort( a_key_t, 0 )
	cdef np.ndarray[c_intp, ndim=1, mode="c"] permb = np.PyArray_LexSort( b_key_t, 0 )
	
	dims2.ptr[0] = l_dat_a
	dims2.ptr[1] = wa
	cdef np.ndarray[c_int, ndim=2, mode="c"] a_key = empty(dims2, np.NPY_LONG)
	#cdef c_int[:, ::1] a_key = empty(dims2, np.NPY_LONG)
	dims2.ptr[0] = l_dat_b
	dims2.ptr[1] = wb
	cdef np.ndarray[c_int, ndim=2, mode="c"] b_key = empty(dims2, np.NPY_LONG)
	#cdef c_int[:, ::1] b_key = empty(dims2, np.NPY_LONG)

	#a_key = np.take(a_key, perma, axis = 0)
	#b_key = np.take(b_key, permb, axis = 0)
	#plus append permutation so we now which item in .dat the row corresponds to
	for r in range(l_dat_a):
		for i in range(wa - 1):
			a_key[r, i] = a_key_t[i, perma[r]]
		a_key[r, wa-1] = perma[r]
	
	for r in range(l_dat_b):
		for i in range(wb - 1):
			b_key[r, i] = b_key_t[i, permb[r]]
		b_key[r, wb-1] = permb[r]
	
	# now a_key is sorted and is shaped (nda + num_q + 1, num_block a), ditto for b_key
	
	#print "perma", perma, "permb", permb

	if TENSORDOT_VERBOSE > 0:
		print a_key, "= sorted a_key"
		print b_key, "= sorted b_key"
	
	if TENSORDOT_TIMING:
		print "Keys", time.time() - t0
		t0 = time.time()
	# tensordot_guts will process the keys and dats, calling np.dot to generate the output.  The code is in npc_helper.pyx
	
	#tensordot_colinitis(c, nao, nbo, nin, a_key, b_key, q_ind_a, q_ind_b, dat_a, dat_b)
	tensordot_guts(c, nao, nbo, nin, a_key, b_key, q_ind_a, q_ind_b, dat_a, dat_b)

	if len(c.dat) > 0:
		c.dtype = c.dat[0].dtype
	else:
		c.q_dat = np.zeros((0,ndc),np.uint)
		c.dtype = np.dtype(float)

	if TENSORDOT_TIMING:
		print "Gut", time.time() - t0
		t0 = time.time()
		

	return c
	
	
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def  tensordot_guts(object c, c_intp nao, c_intp nbo, c_intp nin, \
		c_int[:, ::1] a_key, \
		c_int[:, ::1] b_key, \
		list q_ind_a, list q_ind_b, list dat_a, list dat_b):
	"""Given 'keys' in current format, the q_ind, and the dat, construct q_ind and dat for c

		The columns of a_key are 0:nin (inner), nin:nda (outer), nda:-1 (total inner charge), -1 perma
		The columns of b_key are 0:nin (inner), nin:ndb (outer), ndb:-1 (total inner charge), -1 permb
		both a_key and b_key must be lex-sorted and C-contiguous

		The order of q_ind_a are outer, inner
		The order of q_ind_b are inner, outer
		Note it's different from teh keys -- Yeah, it's weird
		"""


	cdef c_int type = dat_a[0].dtype.num
	
	#For double and cdouble types, we will call BLAS directly.
	if dat_b[0].dtype.num!=type:
		type = -1
	if USE_DRESDEN and (type==np.NPY_DOUBLE or type==np.NPY_CDOUBLE):
		pass
	else:
		type = -1
	cdef double complex cone = 1. + 0j
	cdef double complex czero = 0. + 0j
	if TENSORDOT_TIMING:
		t0 = time.time()
	# TODO check if nao + nbo is zero
	cdef c_uint num_q = q_ind_a[0].shape[1]
	num_q = num_q - 2
	cdef c_uint nda = nao + nin
	cdef c_uint ndb = nbo + nin
	cdef c_uint ndc = nao + nbo
	# TODO check if nin is zero

	#First bounds of total charge sectors

	cdef c_int[::1] qta_bnds = find_differences_contiguous2( a_key, nda, nda + num_q )
	cdef c_int[::1] qtb_bnds = find_differences_contiguous2( b_key, ndb, ndb + num_q )
	
	
#	print qta_bnds, "qta"
#	print qtb_bnds, "qtb"

	#Second bounds of outer charge sectors
	cdef c_int[::1] qoa_bnds = find_differences_contiguous2( a_key, nin, nda )
	cdef c_int[::1] qob_bnds = find_differences_contiguous2( b_key, nin, ndb )
#	print qoa_bnds, "qoa"
#	print qob_bnds, "qob"
	
	cdef c_uint Nqta = qta_bnds.shape[0] - 1	# number total charge sector in a
	cdef c_uint Nqtb = qtb_bnds.shape[0] - 1	# number total charge sector in b
	cdef c_uint i1, i2, i3	# 1: total q sectors, 2: outer sectors, 3: actual rows
	cdef c_uint j1, j2, j3
	cdef c_int lexcomp = 0
	cdef c_int lex_i	# for lex compare
	cdef c_uint curQtA, nxtQtA, curQtB, nxtQtB
	i1 = i2 = i3 = 0
	j1 = j2 = j3 = 0
	
	#First bound the maximum number of possible entries (so we can abort if 0)
	#This is allows us to preallocate q_dat, as well as abort if no valid contractions.
	cdef c_uint max_size = 0
	cdef c_uint i20, j20

	while i1 < Nqta and j1 < Nqtb:
		
		curQtA = qta_bnds[i1]
		nxtQtA = qta_bnds[i1 + 1]
		curQtB = qtb_bnds[j1]
		nxtQtB = qtb_bnds[j1 + 1]
		lexcomp = 0		# check if a[i1] > b[j1]
		for lex_i in range(num_q-1, -1, -1):
			if a_key[curQtA, lex_i + nda] > b_key[curQtB, lex_i + ndb]:
				lexcomp = 1
				break
			elif a_key[curQtA, lex_i + nda] < b_key[curQtB, lex_i + ndb]:
				lexcomp = -1
				break
		if lexcomp > 0:		# a_key[i1] is bigger
			j1 += 1
			continue
		elif lexcomp < 0:		# b_key[j1] is bigger
			i1 += 1	
			continue
	
		while qoa_bnds[i2] < curQtA:
			i2 += 1
		while qob_bnds[j2] < curQtB:
			j2 += 1
	
		i20 = i2
		j20 = j2
		while qoa_bnds[i2] < nxtQtA:
			i2 += 1
		while qob_bnds[j2] < nxtQtB:
			j2 += 1

		max_size+= (i2 - i20)*(j2 - j20)
		i1 += 1	
		j1 += 1

	#Preallocate dat_c and q_dat_c
	cdef np.PyArray_Dims dims2
	dims2.len = 2
	dims2.ptr = [ max_size, nao + nbo]
	#cdef c_uint[:, ::1] q_dat_c = empty(dims2, np.NPY_ULONG)
	cdef np.ndarray[c_uint, ndim=2, mode="c"] q_dat_c = empty(dims2, np.NPY_ULONG)
	cdef list dat = [None]*max_size

	#Hurrah! Nothing to do.
	if max_size==0:
		c.q_dat = q_dat_c
		c.dat = dat
		return c

	#Cache outer leg shapes in 2-arrays (corresponding to qo*_bnds)	
	dims2.ptr = [qoa_bnds.shape[0]-1, nao]	
	cdef c_intp[:, ::1] qoa_shapes = empty(dims2, np.NPY_INTP)
	#cdef np.ndarray[c_intp, ndim=2, mode="c"] qoa_shapes = empty(dims2, np.NPY_INTP)

	dims2.ptr[0] = qob_bnds.shape[0]-1
	dims2.ptr[1] = nbo
	cdef c_intp[:, ::1] qob_shapes = empty(dims2, np.NPY_INTP)
	#cdef np.ndarray[c_intp, ndim=2, mode="c"] qob_shapes = empty(dims2, np.NPY_INTP)

	#Cache dimension of the inner index to be contracted
	cdef c_int[::1] m_size = np.ones( a_key.shape[0], dtype = np.int)
	#cdef np.ndarray[c_int, ndim=1, mode="c"] m_size = np.ones( a_key.shape[0], dtype = np.int)

	cdef c_intp l
	cdef c_intp r
	cdef c_int p
	# fill in qoa_shapes
	cdef np.ndarray[c_int, ndim=2] qi	
	for l in range(nao):
		qi = q_ind_a[l]
		i1 = nin+l
		for r in range(qoa_shapes.shape[0]):
			p =  a_key[qoa_bnds[r], i1]
			qoa_shapes[r, l] = qi[p, 1] - qi[p, 0]
				
	#print qoa_shapes, "qoa shapes"

	# fill in qob_shapes	
	for l in range(nbo):
		i1 = nin+l
		qi = q_ind_b[i1]
		for r in range(qob_shapes.shape[0]):
			p = b_key[qob_bnds[r], i1]
			qob_shapes[r, l] = qi[p, 1] - qi[p, 0]
	#print qob_shapes, "qob shapes"
	
	# calculate 'SM', the inner charge dimension. For simplicty, we make one for every row of a_key
	for l in range(nin):
		qi = q_ind_a[l + nao]
		for r in range(a_key.shape[0]):
			p = a_key[r, l]
			m_size[r]*= ( qi[p, 1] - qi[p, 0] )


	################# Outer Loop over Total Charge ##############

	if TENSORDOT_TIMING:
		print "A", time.time() - t0
		t0 = time.time()
	
	#Placeholders for submatrices ta, tb, tc and their shapes
	cdef np.ndarray ta, tb, tc

	tc = q_dat_c #spuriously bind tc so cython knows it WILL be bound

	cdef np.PyArray_Dims shpA, shpB, shpC, shptC #the dims of submats for dot
	shptC.len = shpA.len = shpB.len = 2
	shpA.ptr = [-1, 1]
	shpB.ptr = [1, -1]
	shptC.ptr = [1, 1]

	shpC.ptr = <np.npy_intp*>PyMem_Malloc(ndc * sizeof(np.npy_intp))
	if not shpC.ptr:
		raise MemoryError	
	shpC.len = ndc
		
	cdef np.npy_intp oa_size, ob_size, in_size  #size of combined outer legs / inner leg
	cdef c_uint used_aobo = 0 #Is there any entry for this outer combination?
	cdef c_uint start_b = 0
	cdef c_uint dat_len = 0 # Number of entries in dat_c so far
	i1 = i2 = i3 = 0
	j1 = j2 = j3 = 0
	cdef c_int num_dot = 0 
	cdef c_uint a_key_shift = nda+num_q
	cdef c_uint b_key_shift = ndb+num_q
	while i1 < Nqta and j1 < Nqtb: #For all charge sectors
		curQtA = qta_bnds[i1]
		nxtQtA = qta_bnds[i1 + 1]
		curQtB = qtb_bnds[j1]
		nxtQtB = qtb_bnds[j1 + 1]
		
		lexcomp = 0		# check if a[i1] > b[j1]
		for lex_i in range(num_q-1, -1, -1):
			if a_key[curQtA, lex_i + nda] > b_key[curQtB, lex_i + ndb]:
				lexcomp = 1
				break
			elif a_key[curQtA, lex_i + nda] < b_key[curQtB, lex_i + ndb]:
				lexcomp = -1
				break

		if lexcomp > 0:		# a_key[i1] is bigger
			j1 += 1
			continue
		elif lexcomp < 0:		# b_key[j1] is bigger
			i1 += 1	
			continue
		
		# product Loop over Outer Charge Sectors
		while qoa_bnds[i2] < curQtA:
			i2 += 1
		while qob_bnds[j2] < curQtB:
			j2 += 1
			
		start_b = j2	# mark the start of this total charge sector

		# double loop over aouter and bouter, bouter will go through multiple passes, while aouter only goes through one
		while  qoa_bnds[i2] < nxtQtA: 	# For each entry in A sector of current Qtot
		
			oa_size = 1
			for l in range(nao):
				shpC.ptr[l] = qoa_shapes[i2,l] #Store oa size
				oa_size*=qoa_shapes[i2,l]
				
			j2 = start_b #Put B counter at start of current QT, start_b
			while qob_bnds[j2] < nxtQtB: 	# Loop through entries in B sector
				
				ob_size = 1
				for r in range(nbo):
					shpC.ptr[nao+r] = qob_shapes[j2,r] #Store ob size
					ob_size*=qob_shapes[j2,r]
					
				i3 = qoa_bnds[i2] #i3 denotes rows in a_key,
				j3 = qob_bnds[j2] #j3 denotes rows in b_key, 
				
				used_aobo = 0 # mark if this combination of aobo has any entries in it
				# print "aobo:", a_key[i3,nin:nda], b_key[j3,nin:ndb], aobo
				
				# inner loop, match by inner indices
				while i3 < qoa_bnds[i2 + 1] and j3 < qob_bnds[j2 + 1]: #Stop when you hit next outer sector
					# compare inner
					lexcomp = 0	
					for lex_i in range(nin-1, -1, -1):
						if a_key[i3, lex_i] > b_key[j3, lex_i]:
							lexcomp = 1
							break
						elif a_key[i3, lex_i] < b_key[j3, lex_i]:
							lexcomp = -1
							break
							
					if lexcomp > 0:
						j3 += 1
					elif lexcomp < 0:
						i3 += 1
					else:
						if COLLECT_MKN_STATS:
							t0 = time.clock()
						in_size = m_size[i3]
						#Make sure data is C-contiguous - this is now done before guts
						ta = dat_a[a_key[i3, a_key_shift]]
						#ta = np.PyArray_GETCONTIGUOUS(ta)
						tb = dat_b[b_key[j3, b_key_shift]]
						#tb = np.PyArray_GETCONTIGUOUS(tb)
						
						if type == -1: #If using np.dot, actually have to reshape matrices
							shpA.ptr = [oa_size, <np.npy_intp>in_size]
							ta = np.PyArray_Newshape(dat_a[a_key[i3, a_key_shift]], &shpA, np.NPY_CORDER)
							shpB.ptr = [<np.npy_intp>in_size, ob_size]
							tb = np.PyArray_Newshape(dat_b[b_key[j3, b_key_shift]], &shpB, np.NPY_CORDER)



						if used_aobo: #We've already initialized this sector, so accumulate	
							#GEMM: c -> a A.B + b C , so let a = b = 1
							
							if type==np.NPY_DOUBLE:
								"""
								if oa_size==1 and in_size==1:
									lib_daxpy(ob_size, (<double*>np.PyArray_DATA(ta))[0], <double*>np.PyArray_DATA(tb), 1, <double*>np.PyArray_DATA(tc), 1 )
								elif ob_size==1 and in_size==1:
									lib_daxpy(oa_size, (<double*>np.PyArray_DATA(tb))[0], <double*>np.PyArray_DATA(ta), 1, <double*>np.PyArray_DATA(tc), 1 )
								else:
								"""
								lib_dgemm( CblasRowMajor, CblasNoTrans,CblasNoTrans, oa_size, ob_size, in_size, 1., <double*>np.PyArray_DATA(ta), in_size, <double*>np.PyArray_DATA(tb), ob_size, 1., <double*>np.PyArray_DATA(tc), ob_size )
									
									
							elif type==np.NPY_CDOUBLE:
								"""
								if oa_size==1 and in_size==1:
									lib_zaxpy(ob_size,<void*>np.PyArray_DATA(ta), <void*>np.PyArray_DATA(tb), 1, <void*>np.PyArray_DATA(tc), 1 )
								elif ob_size==1 and in_size==1:
									lib_zaxpy(oa_size, <void*>np.PyArray_DATA(tb), <void*>np.PyArray_DATA(ta), 1, <void*>np.PyArray_DATA(tc), 1 )
								else:
								"""
								lib_zgemm( CblasRowMajor, CblasNoTrans,CblasNoTrans, oa_size, ob_size, in_size, &cone, <void*>np.PyArray_DATA(ta), in_size, <void*>np.PyArray_DATA(tb), ob_size, &cone, <void*>np.PyArray_DATA(tc), ob_size )
							else:
								tc+=dot(ta, tb)
						else:
							for l in range(nao): q_dat_c[dat_len, l] = a_key[i3, nin+l]
							for r in range(nbo): q_dat_c[dat_len, nao+r] = b_key[j3,nin+r]
							#GEMM: c -> a A.B + b C , so let a = 1, b = 0

							if type==np.NPY_DOUBLE:
								tc = empty(shpC, type)
								#with nogil:
								lib_dgemm( CblasRowMajor, CblasNoTrans,CblasNoTrans, oa_size, ob_size, in_size, 1., <double*>np.PyArray_DATA(ta), in_size, <double*>np.PyArray_DATA(tb), ob_size, 0., <double*>np.PyArray_DATA(tc), ob_size )
									#lib_dgemm( CblasRowMajor, CblasNoTrans,CblasNoTrans, oa_size, ob_size, in_size, 1., <double*>ta.data, in_size, <double*>tb.data, ob_size, 0., <double*>tc.data, ob_size )
							elif type==np.NPY_CDOUBLE:
								tc = empty(shpC, type)
								lib_zgemm( CblasRowMajor, CblasNoTrans,CblasNoTrans, oa_size, ob_size, in_size, &cone, <void*>np.PyArray_DATA(ta), in_size, <void*>np.PyArray_DATA(tb), ob_size, &czero, <void*>np.PyArray_DATA(tc), ob_size )
							else:
								tc=dot(ta, tb)
									
							dat_len += 1
							used_aobo = 1

						if COLLECT_MKN_STATS:
							t0 = time.clock() - t0
							global mkn_stats
							global num_dot
							if num_dot%mkn_grain == 0:
								mkn_stats.append([oa_size*in_size*ob_size, t0])

							num_dot+=1

						j3 += 1		# increment both rows
						i3 += 1
				
				if used_aobo:
					if type==-1:
						dat[dat_len-1] = np.PyArray_Newshape(tc, &shpC, np.NPY_CORDER)
					else:
						dat[dat_len-1] = tc
					
				j2+=1 # advances to next B outer sector
				
			i2+=1 # advance to next A outer sector
		# advance to the next charge sector
		i1 += 1	
		j1 += 1
	
	if TENSORDOT_TIMING:
		print "B", time.time() - t0
		t0 = time.time()
				
	PyMem_Free(shpC.ptr)

	if max_size != dat_len: #Usually not the case
		c.q_dat = q_dat_c[0:dat_len, :]
		c.dat = dat[0:dat_len]
	else:
		c.q_dat = q_dat_c
		c.dat = dat



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cpdef tensordot2(a, b, axes = 2):
	""" Tensor contraction of npc arrays a, b. Identical usage as numpy.

	"""
	
	cdef list dat_a = a.dat
	cdef list dat_b = b.dat

	if TENSORDOT_VERBOSE:
		print "Welcome to tensordot2", a.shape, b.shape, len(dat_a), len(dat_b),a.dtype.num, b.dtype.num

	cdef c_int d_type
	d_type = a.dtype.num

	#For double and cdouble types, we will call BLAS directly.
	if b.dtype.num!=d_type:
		d_type = -1
	if (d_type==np.NPY_DOUBLE or d_type==np.NPY_CDOUBLE):
		pass
	else:
		d_type = -1

	if d_type == -1:
		if TENSORDOT_VERBOSE:
			print "Nonstandard type"
		return tensordot(a, b, axes)



	if TENSORDOT_TIMING:
		t0 = time.time()
	
	cdef c_intp na, nb
	cdef list axes_a, axes_b

	# Bring all axes specification to standard form
	try:
		iter(axes)
	except:
		axes_a = range(-axes,0)
		axes_b = range(0,axes)
		na = nb = axes
	else:
		if type(axes[0])!=list and type(axes[0])!=tuple:
			axes_a = [ a.get_index(axes[0]) ]
			na = 1
		else:
			na = len(axes[0])
			axes_a = [ a.get_index(m) for m in axes[0] ]

		if type(axes[1])!=list and type(axes[1])!=tuple:
			axes_b = [ b.get_index(axes[1]) ]
			nb = 1
		else:
			nb = len(axes[1])
			axes_b = [ b.get_index(m) for m in axes[1] ]

	# axes_a,b are now each list (same len) of indices to contract (na,b are the number of indices)


	###
	#  as_, bs			: shape of a, b
	#  nda, ndb, ncd	: rank of a, b, c = a.b
	#  na = nb = nin	: number of legs being contracted
	#  nao = nda - nin	: number of uncontracted outer legs
	#  nbo = ndb - nin

	cdef np.ndarray[c_intp, ndim=1] as_ = a.shape #SHAPE
	cdef c_intp nda = a.rank
	cdef np.ndarray[c_intp, ndim=1] bs = b.shape #SHAPE
	cdef c_intp ndb = b.rank
	
	cdef int equal = 1
	cdef c_uint k
	if (na != nb):
		equal = 0
	else:
		for k in range(na):
			axes_a[k] = axes_a[k]%nda
			axes_b[k] = axes_b[k]%ndb			
			if as_[axes_a[k]] != bs[axes_b[k]]:
				equal = 0
				break
	if not equal:
		raise ValueError, "shape-mismatch for sum"
		# %s %s %s %s"%(a.shape, axes_a, b.shape, axes_b)

	cdef c_int i, r, l, p
	cdef c_int ndc, nao, nbo, nin
	cdef np.ndarray ta, tb, tc
	nin = na

	# Move the axes to sum over to the end of "a"
	# and to the front of "b"
	cdef list notin_a = [k for k in range(nda) if k not in axes_a]
	cdef list newaxes_a = notin_a + axes_a
	cdef list olda = [as_[axis] for axis in notin_a]
	nao = nda - nin
	
	cdef list notin_b = [k for k in range(ndb) if k not in axes_b]
	cdef list newaxes_b = axes_b + notin_b
	cdef list oldb = [bs[axis] for axis in notin_b]
	nbo = ndb - nin	
	ndc = nao + nbo
	
	if TENSORDOT_TIMING:
		print "Axes", time.time() - t0
		t0 = time.time()
		
	if nao == 0 and nbo == 0:
		return npc.inner(a, b, [ axes_a, axes_b])

	if newaxes_a!=range(nda):
		a = transpose_fast(a, np.array(newaxes_a))
		dat_a = a.dat
	else:
		dat_a = [ np.PyArray_GETCONTIGUOUS(ta) for ta in a.dat]

	if newaxes_b!=range(ndb):
		b = transpose_fast(b, np.array(newaxes_b))
		dat_b = b.dat
	else:
		dat_b = [ np.PyArray_GETCONTIGUOUS(tb) for tb in b.dat]

	if TENSORDOT_TIMING:
		print "Transpose", time.time() - t0
		t0 = time.time()


	# C_lr = A_lc B_cr

	cdef np.PyArray_Dims dims1
	dims1.len = 1
	dims1.ptr = [ndc]
		
	cdef np.PyArray_Dims dims2
	dims2.len = 2
	dims2.ptr = [0, 0]

	cdef list q_ind_a = a.q_ind
	cdef list q_ind_b = b.q_ind
	cdef np.ndarray[c_uint, ndim=2] q_dat_a = a.q_dat
	cdef np.ndarray[c_uint, ndim=2] q_dat_b = b.q_dat
	cdef np.ndarray[c_int, ndim=1, mode = "c"] mod_q = a.mod_q
	cdef np.ndarray[c_int, ndim=1, mode = "c"] q_conj_a = a.q_conj
	cdef np.ndarray[c_int, ndim=1, mode = "c"] q_conj_b = b.q_conj
	cdef np.ndarray[c_int, ndim=1, mode = "c"] q_conj_c = empty(dims1, np.NPY_LONG)

	cdef c_intp num_q = a.num_q
	
	cdef c_intp l_dat_a = q_dat_a.shape[0]
	cdef c_intp l_dat_b = q_dat_b.shape[0]

	cdef object c = npc.array(ndc, dtype = None) 	# dtype set later
	c.shape = np.array(olda + oldb)#SHAPE
	c.num_q = num_q
	c.mod_q = mod_q
	c.charge = mod_onetoinf1D(a.charge + b.charge, mod_q)		##
	c.q_ind = a.q_ind[:nao] + b.q_ind[nin:]


	"""Make new labels for c	
		No labels made if either a or b doesnt have labels
		
		Any labels that would be duplicates are dropped
	"""
	cdef dict alab = a.labels
	cdef dict blab = b.labels
	cdef dict clab
	cdef list keys
	cdef c_int v

	if alab is not None and blab is not None:
		clab = {}
		for key in alab:
			v = alab[key]
			if v < nao:
				clab[key] = v
		for key in blab:
			v = blab[key]
			if v>=nin:
				if key in clab: #Drop duplicate key
					del clab[key]
				else:
					clab[key] = v + nao - nin
		c.labels = clab
	else:
		c.labels = None

	for i in range(nao):
		q_conj_c[i] = q_conj_a[i]
	for i in range(nbo):
		q_conj_c[i + nao] = q_conj_b[i + nin]
	c.q_conj = q_conj_c
	c.sorted = False

	if TENSORDOT_TIMING:
		print "Labels", time.time() - t0
		t0 = time.time()

	if TENSORDOT_VERBOSE > 0:
		print "c.shape", c.shape

	### ------------  Special Cases -----------------

	#Zero
	if l_dat_a == 0 or l_dat_b == 0:
		if TENSORDOT_VERBOSE > 0:
			print "Zero"
		c.q_dat = np.zeros((0, ndc), np.uint)
		c.dat = []
		c.dtype = np.dtype(np.float)
		return c
	
	cdef np.ndarray[c_uint, ndim=2, mode="c"] q_dat_c
	cdef np.PyArray_Dims shpA, shpB, shpC
	shpA.len = shpB.len = 2
	shpA.ptr = [-1, -1]
	shpB.ptr = [-1, -1]
	tc = mod_q #spuriously bind
	# Only one block in each. Usual case for num_q = 0 (unless fractured!)
	if l_dat_a == 1 and l_dat_b == 1:
		if TENSORDOT_VERBOSE > 0:
			print "One block"
		equal = 1
		
		for k in range(nin): #check if entries match on inner
			if q_dat_a[0, nao + k]!=q_dat_b[0, k]:
				equal = 0
				break
		if equal: #So contract inner				
			ta = dat_a[0]
			tb = dat_b[0]
			
			shpC.ptr = <np.npy_intp*>PyMem_Malloc(ndc * sizeof(np.npy_intp))
			if not shpC.ptr:
				raise MemoryError
			shpC.len = ndc
			dims2.ptr = [1, ndc]
			c.q_dat = q_dat_c = empty(dims2, np.NPY_ULONG) #make q_dat entry

			for k in range(nao):			
				q_dat_c[0, k] = q_dat_a[0, k]
				shpC.ptr[k] = ta.shape[k]
			for k in range(nbo):
				q_dat_c[0, nao + k] = q_dat_b[0, k + nin]
				shpC.ptr[nao + k] = tb.shape[k + nin]
						
			r = 1
			for k in range(nin): #inner dimension
				r *= tb.shape[k]  
			
			shpA.ptr[1] = r
			shpB.ptr[0] = r

			ta = np.PyArray_Newshape(ta, &shpA, np.NPY_CORDER)
			tb = np.PyArray_Newshape(tb, &shpB, np.NPY_CORDER)
			tc = dot(ta, tb)
			
			c.dat = [np.PyArray_Newshape(tc, &shpC, np.NPY_CORDER)] 
			c.dtype = c.dat[0].dtype			
			PyMem_Free(shpC.ptr)
			return c
		else:
			c.q_dat = np.zeros((0,ndc), np.uint) #Blocks didn't overlap, so return 0
			c.dat = []
			c.dtype = np.dtype(float)
			return c

	"""
	Pre-compute conventient data for a & b: the "Keys"

	Key structure:
	       width=  | nin     | n(a/b)o  | num_q | 1 
	     encodes=  | in inds | out inds | Q_tot | perm
		
		sorted by (lex order)
		1. total charge of the inner indices, 
		2. the outer indices
		3. the inner indices
	"""
	
	cdef c_int wa = nda + num_q + 1 #width of keys
	cdef c_int wb = ndb + num_q + 1
	
	#First I create a tranposed key, *_key_t, ommitting the perm column 		
	
	#NOTE: memview style is slower
	dims2.ptr[0] = wa-1
	dims2.ptr[1] = l_dat_a
	#cdef np.ndarray[c_int, ndim=2, mode="c"] a_key_t = empty(dims2, np.NPY_LONG)
	cdef c_int[:,::1] a_key_t = empty(dims2, np.NPY_LONG)
	
	dims2.ptr[0] = wb - 1
	dims2.ptr[1] = l_dat_b
	#cdef np.ndarray[c_int, ndim=2, mode="c"] b_key_t = empty(dims2, np.NPY_LONG)
	cdef c_int[:, ::1] b_key_t = empty(dims2, np.NPY_LONG)
	
	# Now fill in total charge over contracted legs
	cdef np.ndarray[c_int, ndim = 2, mode = "c"] q_sum_res
	#cdef c_int[:, ::1] q_sum_res
	if nin > 0:
		q_sum_res = q_sum(q_dat_a, q_ind_a, q_conj_a, nao, nda, mod_q)
		for r in range(l_dat_a): #copy
			for i in range(num_q):
				a_key_t[nda + i, r] = q_sum_res[r, i]
		
		q_sum_res = q_sum(q_dat_b, q_ind_b, -1*q_conj_b, 0, nin, mod_q)
		for r in range(l_dat_b): #copy
			for i in range(num_q):
				b_key_t[ndb+i, r] = q_sum_res[r, i]
	else:
		for r in range(l_dat_a):
			for i in range(num_q):
				a_key_t[nda + i, r] = 0
		
		for r in range(l_dat_b):
			for i in range(num_q):
				b_key_t[ndb + i, r] = 0

	#Now fill in index info (need to reverse order in w.r.t. q_dat for a)
			
	for r in range(l_dat_a):
		for i in range(nin):
			a_key_t[i, r] = q_dat_a[r, nao + i]
		for i in range(nao):
			a_key_t[nin + i, r] = q_dat_a[r, i]

	for r in range(l_dat_b):
		for i in range(ndb):
			b_key_t[i, r] = q_dat_b[r, i]

	if TENSORDOT_VERBOSE > 1:
		print a_key_t.T[:, :nda+num_q], "= unsorted a_key"
		print b_key_t.T[:, :ndb+num_q], "= unsorted b_key"
	
	#Sort by Qt > out ind > in ind 
	#cdef np.ndarray[c_intp, ndim=1, mode="c"] perma = np.PyArray_LexSort( a_key_t, 0 )
	#cdef np.ndarray[c_intp, ndim=1, mode="c"] permb = np.PyArray_LexSort( b_key_t, 0 )
	cdef c_intp[::1] perma = np.PyArray_LexSort( a_key_t, 0 )
	cdef c_intp[::1] permb = np.PyArray_LexSort( b_key_t, 0 )
	
	dims2.ptr[0] = l_dat_a
	dims2.ptr[1] = wa
	#cdef np.ndarray[c_int, ndim=2, mode="c"] a_key = empty(dims2, np.NPY_LONG)
	cdef c_int[:, ::1] a_key = empty(dims2, np.NPY_LONG)
	dims2.ptr[0] = l_dat_b
	dims2.ptr[1] = wb
	#cdef np.ndarray[c_int, ndim=2, mode="c"] b_key = empty(dims2, np.NPY_LONG)
	cdef c_int[:, ::1] b_key = empty(dims2, np.NPY_LONG)

	#a_key = np.take(a_key, perma, axis = 0)
	#b_key = np.take(b_key, permb, axis = 0)
	#plus append permutation so we now which item in .dat the row corresponds to
	for r in range(l_dat_a):
		p = perma[r]
		for i in range(wa - 1):
			a_key[r, i] = a_key_t[i, p]
		a_key[r, wa-1] = p
	
	for r in range(l_dat_b):
		p = permb[r]
		for i in range(wb - 1):
			b_key[r, i] = b_key_t[i, p]
		b_key[r, wb-1] = p
	
	# now a_key is sorted and is shaped (nda + num_q + 1, num_block a), ditto for b_key
	

	if TENSORDOT_VERBOSE > 1:
		print a_key, "= sorted a_key"
		print b_key, "= sorted b_key"
	
	if TENSORDOT_TIMING:
		print "Keys", time.time() - t0
		t0 = time.time()


	"""Given 'keys' in current format, the q_ind, and the dat, construct q_ind and dat for c

		The columns of a_key are 0:nin (inner), nin:nda (outer), nda:-1 (total inner charge), -1 perma
		The columns of b_key are 0:nin (inner), nin:ndb (outer), ndb:-1 (total inner charge), -1 permb
		both a_key and b_key must be lex-sorted and C-contiguous

		The order of q_ind_a are outer, inner
		The order of q_ind_b are inner, outer
		Note it's different from teh keys -- Yeah, it's weird
	"""


	# TODO check if nao + nbo is zero
	# TODO check if nin is zero

	#First bounds of total charge sectors

	cdef c_int[::1] qta_bnds = find_differences_contiguous2( a_key, nda, nda + num_q )
	cdef c_int[::1] qtb_bnds = find_differences_contiguous2( b_key, ndb, ndb + num_q )


	#Second bounds of outer charge sectors
	cdef c_int[::1] qoa_bnds = find_differences_contiguous2( a_key, nin, nda )
	cdef c_int[::1] qob_bnds = find_differences_contiguous2( b_key, nin, ndb )

	
	cdef c_int Nqta = qta_bnds.shape[0] - 1	# number total charge sector in a
	cdef c_int Nqtb = qtb_bnds.shape[0] - 1	# number total charge sector in b
	cdef c_int i1, i2, i3	# 1: total q sectors, 2: outer sectors, 3: actual rows
	cdef c_int j1, j2, j3
	cdef c_int lexcomp = 0
	cdef c_int lex_i	# for lex compare
	cdef c_int curQtA, nxtQtA, curQtB, nxtQtB
	i1 = i2 = i3 = 0
	j1 = j2 = j3 = 0
	
	#First bound the maximum number of possible entries (so we can abort if 0)
	#This is allows us to preallocate q_dat, as well as abort if no valid contractions.
	cdef c_int max_size = 0
	cdef c_int i20, j20

	while i1 < Nqta and j1 < Nqtb:
		
		curQtA = qta_bnds[i1]
		nxtQtA = qta_bnds[i1 + 1]
		curQtB = qtb_bnds[j1]
		nxtQtB = qtb_bnds[j1 + 1]
		lexcomp = 0		# check if a[i1] > b[j1]
		for lex_i in range(num_q-1, -1, -1):
			if a_key[curQtA, lex_i + nda] > b_key[curQtB, lex_i + ndb]:
				lexcomp = 1
				break
			elif a_key[curQtA, lex_i + nda] < b_key[curQtB, lex_i + ndb]:
				lexcomp = -1
				break
		if lexcomp > 0:		# a_key[i1] is bigger
			j1 += 1
			continue
		elif lexcomp < 0:		# b_key[j1] is bigger
			i1 += 1	
			continue

		while qoa_bnds[i2] < curQtA:
			i2 += 1
		while qob_bnds[j2] < curQtB:
			j2 += 1

		i20 = i2
		j20 = j2
		while qoa_bnds[i2] < nxtQtA:
			i2 += 1
		while qob_bnds[j2] < nxtQtB:
			j2 += 1

		max_size+= (i2 - i20)*(j2 - j20)
		i1 += 1	
		j1 += 1

	#Preallocate dat_c and q_dat_c
	dims2.ptr[0] = max_size
	dims2.ptr[1] = nao + nbo
	#cdef c_uint[:, ::1] q_dat_c = empty(dims2, np.NPY_ULONG)
	q_dat_c = empty(dims2, np.NPY_ULONG)
	cdef list dat = [None]*max_size

	#Hurrah! Nothing to do.
	if max_size==0:
		c.q_dat = q_dat_c
		c.dat = dat
		return c

	#Cache outer leg shapes in 2-arrays (corresponding to qo*_bnds)	
	dims2.ptr[0] = qoa_bnds.shape[0]-1
	dims2.ptr[1] =  nao
	cdef c_intp[:, ::1] qoa_shapes = empty(dims2, np.NPY_INTP)

	dims2.ptr[0] = qob_bnds.shape[0]-1
	dims2.ptr[1] = nbo
	cdef c_intp[:, ::1] qob_shapes = empty(dims2, np.NPY_INTP)

	#Cache dimension of the inner index to be contracted

	dims1.ptr[0] = a_key.shape[0]
	cdef c_int[::1] m_size = empty( dims1, np.NPY_LONG)
	#cdef c_int[::1] m_size = np.ones( a_key.shape[0], dtype = np.int)

	# fill in qoa_shapes
	cdef np.ndarray[c_int, ndim=2] qi	
	for l in range(nao):
		qi = q_ind_a[l]
		i1 = nin+l
		for r in range(qoa_shapes.shape[0]):
			p =  a_key[qoa_bnds[r], i1]
			qoa_shapes[r, l] = qi[p, 1] - qi[p, 0]
				

	# fill in qob_shapes	
	for l in range(nbo):
		i1 = nin+l
		qi = q_ind_b[i1]
		for r in range(qob_shapes.shape[0]):
			p = b_key[qob_bnds[r], i1]
			qob_shapes[r, l] = qi[p, 1] - qi[p, 0]

	# calculate 'SM', the inner charge dimension. For simplicty, we make one for every row of a_key


	if nin==0:
		for r in range(a_key.shape[0]):
			m_size[r] = 1
	else:
		qi = q_ind_a[nao]
		for r in range(a_key.shape[0]):
			p = a_key[r, 0]
			m_size[r] = ( qi[p, 1] - qi[p, 0] )

	for l in range(1, nin):
		qi = q_ind_a[l + nao]
		for r in range(a_key.shape[0]):
			p = a_key[r, l]
			m_size[r]*= ( qi[p, 1] - qi[p, 0] )


	if TENSORDOT_TIMING:
		print "Shape & Bounds", time.time() - t0
		t0 = time.time()



	################# Outer Loop over Total Charge ##############
	
	#Placeholders for submatrices ta, tb, tc and their shapes
	cdef np.PyArray_Dims shptC #the dims of submats for dot
	shptC.len = 2
	shpA.ptr = [-1, 1]
	shpB.ptr = [1, -1]
	shptC.ptr = [1, 1]

	shpC.ptr = <np.npy_intp*>PyMem_Malloc(ndc * sizeof(np.npy_intp))
	if not shpC.ptr:
		raise MemoryError	
	shpC.len = ndc
		
	cdef c_intp oa_size, ob_size, in_size  #size of combined outer legs / inner leg
	cdef c_int used_aobo = 0 #Is there any entry for this outer combination?
	cdef c_int start_b = 0
	cdef c_int dat_len = 0 # Number of entries in dat_c so far
	i1 = i2 = i3 = 0
	j1 = j2 = j3 = 0
	cdef c_int num_dot = 0 
	cdef c_uint a_key_shift = nda+num_q
	cdef c_uint b_key_shift = ndb+num_q




	cdef vector[int] M, N, K, G
	cdef vector[void*] A, B, C


	while i1 < Nqta and j1 < Nqtb: #For all charge sectors
		curQtA = qta_bnds[i1]
		nxtQtA = qta_bnds[i1 + 1]
		curQtB = qtb_bnds[j1]
		nxtQtB = qtb_bnds[j1 + 1]
		
		lexcomp = 0		# check if a[i1] > b[j1]
		for lex_i in range(num_q-1, -1, -1):
			if a_key[curQtA, lex_i + nda] > b_key[curQtB, lex_i + ndb]:
				lexcomp = 1
				break
			elif a_key[curQtA, lex_i + nda] < b_key[curQtB, lex_i + ndb]:
				lexcomp = -1
				break

		if lexcomp > 0:		# a_key[i1] is bigger
			j1 += 1
			continue
		elif lexcomp < 0:		# b_key[j1] is bigger
			i1 += 1	
			continue
		
		# product Loop over Outer Charge Sectors
		while qoa_bnds[i2] < curQtA:
			i2 += 1
		while qob_bnds[j2] < curQtB:
			j2 += 1
			
		start_b = j2	# mark the start of this total charge sector

		# double loop over aouter and bouter, bouter will go through multiple passes, while aouter only goes through one
		while  qoa_bnds[i2] < nxtQtA: 	# For each entry in A sector of current Qtot
		
			oa_size = 1
			for l in range(nao):
				shpC.ptr[l] = qoa_shapes[i2,l] #Store oa size
				oa_size*=qoa_shapes[i2,l]
				
			j2 = start_b #Put B counter at start of current QT, start_b
			while qob_bnds[j2] < nxtQtB: 	# Loop through entries in B sector
				
				ob_size = 1
				for r in range(nbo):
					shpC.ptr[nao+r] = qob_shapes[j2,r] #Store ob size
					ob_size*=qob_shapes[j2,r]
					
				i3 = qoa_bnds[i2] #i3 denotes rows in a_key,
				j3 = qob_bnds[j2] #j3 denotes rows in b_key, 
				
				used_aobo = 0 # mark if this combination of aobo has any entries in it

				# inner loop, match by inner indices
				while i3 < qoa_bnds[i2 + 1] and j3 < qob_bnds[j2 + 1]: #Stop when you hit next outer sector
					# compare inner
					lexcomp = 0	
					for lex_i in range(nin-1, -1, -1):
						if a_key[i3, lex_i] > b_key[j3, lex_i]:
							lexcomp = 1
							break
						elif a_key[i3, lex_i] < b_key[j3, lex_i]:
							lexcomp = -1
							break
							
					if lexcomp > 0:
						j3 += 1
					elif lexcomp < 0:
						i3 += 1
					else:
						in_size = m_size[i3]
						#ta = dat_a[a_key[i3, a_key_shift]]
						#tb = dat_b[b_key[j3, b_key_shift]]

						K.push_back(in_size)
						A.push_back(np.PyArray_DATA(<np.ndarray>dat_a[a_key[i3, a_key_shift]]))
						B.push_back(np.PyArray_DATA(<np.ndarray>dat_b[b_key[j3, b_key_shift]]))

						if used_aobo ==0: #We've haven't 'initialized this sector

							for l in range(nao): q_dat_c[dat_len, l] = a_key[i3, nin+l]
							for r in range(nbo): q_dat_c[dat_len, nao+r] = b_key[j3,nin+r]

							tc = empty(shpC, d_type)
							M.push_back(oa_size)
							N.push_back(ob_size)
							C.push_back(np.PyArray_DATA(tc))
							dat[dat_len] = tc
							dat_len+=1

						used_aobo+=1


						j3 += 1		# increment both rows
						i3 += 1
				
				if used_aobo:
					G.push_back(used_aobo)
					
				j2+=1 # advances to next B outer sector
				
			i2+=1 # advance to next A outer sector
		# advance to the next charge sector
		i1 += 1	
		j1 += 1

	PyMem_Free(shpC.ptr)

	if TENSORDOT_TIMING:
		print "Pack gemm", time.time() - t0
		t0 = time.time()

	with nogil:
		global tensordot_threads
		if PARALLEL_TDOT and tensordot_threads > 1:
			if d_type==np.NPY_DOUBLE:
				batch_dgemm_accumulate_parallel( M, K, N, A, B, C, G, tensordot_threads)
			if d_type==np.NPY_CDOUBLE:
				batch_zgemm_accumulate_parallel( M, K, N, A, B, C, G, tensordot_threads)
		else:
			if d_type==np.NPY_DOUBLE:
				batch_dgemm_accumulate( M, K, N, A, B, C, G)
			if d_type==np.NPY_CDOUBLE:
				batch_zgemm_accumulate( M, K, N, A, B, C, G)

	if TENSORDOT_TIMING:
		print "gemm", time.time() - t0
		t0 = time.time()


	if max_size != dat_len: #Usually not the case
		c.q_dat = q_dat_c[0:dat_len, :]
		c.dat = dat[0:dat_len]
	else:
		c.q_dat = q_dat_c
		c.dat = dat

	if len(c.dat) > 0:
		c.dtype = c.dat[0].dtype
	else:
		c.dtype = np.dtype(float)



	if TENSORDOT_TIMING:
		print "Trim", time.time() - t0
		t0 = time.time()

	return c



################################################################################
################################################################################

cdef void batch_dgemm_accumulate( vector[int] M, vector[int] K, vector[int] N, vector[void*] A, vector[void*] B, vector[void*] C, vector[int] G) nogil:



	cdef vector[void*].iterator a = A.begin()
	cdef vector[void*].iterator b = B.begin()
	cdef vector[void*].iterator c = C.begin()

	cdef vector[int].iterator m = M.begin()
	cdef vector[int].iterator n = N.begin()
	cdef vector[int].iterator k = K.begin()

	cdef vector[int].iterator g = G.begin()
	cdef int i
	
	while g != G.end():
		lib_dgemm( CblasRowMajor, CblasNoTrans,CblasNoTrans, deref(m), deref(n), deref(k), 1., <double*>deref(a), deref(k), <double*>deref(b), deref(n), 0., <double*>deref(c), deref(n))
		inc(a)
		inc(b)
		inc(k)

		for i in range(deref(g) - 1):
			lib_dgemm( CblasRowMajor, CblasNoTrans,CblasNoTrans, deref(m), deref(n), deref(k), 1., <double*>deref(a), deref(k), <double*>deref(b), deref(n), 1., <double*>deref(c), deref(n))
			inc(a)
			inc(b)
			inc(k)

		inc(g)
		inc(c)
		inc(m)
		inc(n)


cdef void batch_dgemm_accumulate_parallel( vector[int] M, vector[int] K, vector[int] N, vector[void*] A, vector[void*] B, vector[void*] C, vector[int] G, c_int num_threads) nogil:


	cdef int i,j, x, size = G.size()
	cdef vector[int] starts
	x = 0
	for i in range(size):
		starts.push_back(x)
		x+=G[i]

	cdef int NT = num_threads
	for i in prange(size, num_threads = NT, schedule='guided'):
		x = starts[i]

		lib_dgemm( CblasRowMajor, CblasNoTrans,CblasNoTrans, M[i], N[i], K[x], 1., <double*>(A[x]), K[x], <double*>(B[x]), N[i], 0., <double*>(C[i]), N[i])
		x = x + 1

		for j in range(G[i] - 1):
			lib_dgemm( CblasRowMajor, CblasNoTrans,CblasNoTrans, M[i], N[i], K[x], 1., <double*>(A[x]), K[x], <double*>(B[x]), N[i], 1., <double*>(C[i]), N[i])
			x = x+1

cdef void batch_zgemm_accumulate( vector[int] M, vector[int] K, vector[int] N, vector[void*] A, vector[void*] B, vector[void*] C, vector[int] G) nogil:


	cdef double complex cone = 1. + 0j
	cdef double complex czero = 0. + 0j

	cdef vector[void*].iterator a = A.begin()
	cdef vector[void*].iterator b = B.begin()
	cdef vector[void*].iterator c = C.begin()

	cdef vector[int].iterator m = M.begin()
	cdef vector[int].iterator n = N.begin()
	cdef vector[int].iterator k = K.begin()

	cdef vector[int].iterator g = G.begin()

	cdef int i
	while g != G.end():
		lib_zgemm( CblasRowMajor, CblasNoTrans,CblasNoTrans, deref(m), deref(n), deref(k), &cone, deref(a), deref(k), deref(b), deref(n), &czero, deref(c), deref(n))
		inc(a)
		inc(b)
		inc(k)

		for i in range(deref(g) - 1):
			lib_zgemm( CblasRowMajor, CblasNoTrans,CblasNoTrans, deref(m), deref(n), deref(k), &cone, deref(a), deref(k), deref(b), deref(n), &cone, deref(c), deref(n))
			inc(a)
			inc(b)
			inc(k)

		inc(g)
		inc(c)
		inc(m)
		inc(n)


cdef void batch_zgemm_accumulate_parallel( vector[int] M, vector[int] K, vector[int] N, vector[void*] A, vector[void*] B, vector[void*] C, vector[int] G, c_int num_threads) nogil:

	cdef double complex cone = 1. + 0j
	cdef double complex czero = 0. + 0j
	
	cdef int i,j, x, size = G.size()
	cdef vector[int] starts
	x = 0
	for i in range(size):
		starts.push_back(x)
		x+=G[i]

	cdef int NT = num_threads

	for i in prange(size, num_threads = NT, schedule='guided'):
		x = starts[i]

		lib_zgemm( CblasRowMajor, CblasNoTrans,CblasNoTrans, M[i], N[i], K[x], &cone, A[x], K[x], B[x], N[i], &czero, C[i], N[i])
		x = x + 1

		for j in range(G[i] - 1):
			lib_zgemm( CblasRowMajor, CblasNoTrans,CblasNoTrans, M[i], N[i], K[x], &cone, A[x], K[x], B[x], N[i], &cone, C[i], N[i])
			x = x+1




################################################################################
################################################################################

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef combine_legs(self, list axes, pipes_i, qt_conj_i, bint block_single_legs, bint inplace = False):
	"""	Combine legs together. 
		
		For now, assumes axes = [ [0, 1, . . ., j_1 ], [i_2, . . . , j_2] , [. . . , d]] , a python list of lists enumerating indices to be grouped, 	
			So T_ijkl ---> T_i(jk)l would have axes =  [[0],[1, 2], [3]]
			The length of axes is the rank of the combined tensor.
			
		If pipes = None, the code will make the pipes (and if you haven't stored them elsewhere, you won't be able to split back.)
		Otherwise, pipes = [ pipe0, None, . .. ], with pipes being calculated for positions None. The length of pipes is the rank of the combined tensor.
		
		Optionally, one may provide qt_conj = [1, -1, . . .]. The length of qt_conj it the rank of the combined tensor. However, the entries are ignored except where pipes[j] is None. If pipes[j] is not None, then when constructing the pipe for the new leg, it is constructed such that qt_conj[j]  = 1 / -1 on the new leg.
		
		TODO - would be nice to allow [ [2], [0, 3], [1]], with combine legs carrying out the required transposition first. Tricky what convention to use if leg pipes are involved though.
		
		Example:
		>>>	pipeL = oldarray.make_pipe([0, 1])
		>>>	pipeR = oldarray.make_pipe([3, 4], qt_conj = -1)
		>>>	newarray = oldarray.combine_legs([[0, 1], [2], [3, 4]], pipes = [pipeL, None, pipeR])
		"""


	axes = [ np.array(a, np.intp) for a in axes] #useful for index tricks

	cdef int c_ndim = len(axes)
	
	##	set defaults
	if qt_conj_i is None:
		qt_conj_i = np.ones(self.rank, dtype = np.int)
	else:
		qt_conj_i = np.array(qt_conj_i)
		
	if pipes_i is None:
		pipes_i = [None]*c_ndim
	
	cdef list pipes = pipes_i
	cdef np.ndarray[c_int, ndim=1, mode="c"] qt_conj = qt_conj_i
	cdef np.ndarray[c_int, ndim=1, mode="c"] cq_conj = np.empty(c_ndim, dtype = np.int)	
	cdef np.ndarray[c_int, ndim=1, mode="c"] q_conj = self.q_conj
	cdef np.ndarray[c_int, ndim=1, mode="c"] qc1
	cdef np.ndarray[c_int, ndim=1, mode="c"] qc2
	
	cdef c_int match
	cdef c_int t_conj
	cdef Py_ssize_t i, j, n
	
	cdef dict rec
	cdef str s
	cdef dict c_labels
	cdef c_uint v
	#New labels formed by concatenating old - only defined if all legs labeled
	if self.labels is not None:
		#reverse lookup for labels
		wildcard_label = 0
		rev = { v:k for k, v in self.labels.iteritems()}
		c_labels = {}
		for i in range(c_ndim): #Each new leg

			v = axes[i][0]
			if v in rev:
				s = rev[v]
			else:
				s = '?' + str(wildcard_label)
				wildcard_label+=1

			for v in axes[i][1:]:
				if v in rev:
					s = s + '.' + rev[v]
				else:
					s = s + '.?' + str(wildcard_label)
					wildcard_label+=1


			c_labels[s] = i

	else:
		c_labels = None

	#Make pipes as needed, and check the validity of those provided
	for i in range(c_ndim):	#for each new leg
		
		if pipes[i] is None:
			pipes[i] = self.make_pipe( axes[i], qt_conj=qt_conj[i] , block_single_legs=block_single_legs )
				
		pipe = pipes[i]
				
		qc1 = q_conj[axes[i]]
		qc2 = pipe.q_conj
		n = qc1.shape[0]
		
		#First we check if the pipe has same q_conj as legs, or if we need to flip (or neither - then we have error)
		
		t_conj = pipe.qt_conj	#This is pipe's convention for total leg
		match = 1		
		for j in range(n):
			if qc1[j] != (qc2[j]*t_conj): 
				match = 0
				break				
		if match:	
			cq_conj[i] = 1			
		else:
			t_conj*=-1		
			match = 1		
			for j in range(n):
				if qc1[j] != (qc2[j]*t_conj): 
					match = 0
					break
			if match:
				cq_conj[i] = -1			
			else:
				print "Pipe", i, " conj data:", qc1, qc2*t_conj
				raise ValueError, "Conj tags don't match pipe!"
				
	if inplace:
		c = self
		c.rank = c_ndim

	else:
		c = npc.array(c_ndim, dtype = self.dtype)
		c.num_q = self.num_q
		c.mod_q = self.mod_q
		c.charge = self.charge.copy()

	c.labels = c_labels
	c.q_conj = cq_conj
	c.q_ind = [ pipe.qt_ind for pipe in pipes]	
	c.shape = np.array([pipe.t_shape for pipe in pipes])#SHAPE
	if len(self.dat) == 0:
		c.q_dat = np.zeros( (0, c_ndim), dtype = np.uint)
		c.sorted = True
		return c
	
	cdef np.ndarray[c_uint, ndim=2, mode="c"] q_dat = self.q_dat
	cdef list dat = self.dat
	cdef list q_maps = [ p.get_q_map() for p in pipes ]
	
	if inplace:
		self.dat = []
		self.q_dat = None
	 
	#npc_helper.combine_legs_colinitis(c, q_dat, dat, axes,  pipes,  q_maps)
	combine_legs_guts(c, q_dat, dat, axes,  pipes,  q_maps, inplace)

	return c
		
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void combine_legs_guts(c, np.ndarray[c_uint, ndim=2, mode="c"] q_dat, list dat, list axes, list pipes, list q_maps, bint inplace):
	#Perform the actual shuffling; c is the combined tensor.
	
	cdef c_intp numrow = q_dat.shape[0]
	cdef c_intp a_ndim = q_dat.shape[1]
	cdef c_intp c_ndim = len(pipes)
	
	cdef np.ndarray[c_uint, ndim=2] qm
	cdef np.ndarray[c_int, ndim=2] qt_ind #TODO should be "c" but isn't with MKL?!?
	cdef np.ndarray[c_intp, ndim=1, mode="c"] stride
	cdef np.ndarray[c_intp, ndim=1, mode="c"] axesi
	
	cdef c_uint i, row, iT
	cdef c_uint ind, numa, a	

	#temps for reshaping info
	cdef np.PyArray_Dims dims2
	dims2.len = 2
	dims2.ptr = [numrow, c_ndim]
	
	# For each incoming dat (the 'rows'), we store the Q_ind of the destination, and the reshaped dimensions:	
	# Qt = [c_ndim x rows], the total Q_ind
	
	# In slicing language we have c_out[m0:m0+l0, m1:m1+l1, . . . ] = c_in
	# dest_0 =  [rows x c_ndim], the top left location [m0, m1 ...] of the incoming block in the target
	# lengths = [rows x c_ndim], the lengths of the block, [l0,l1, . . . ]

	cdef np.ndarray[c_int, ndim=2, mode="c"] Qt = empty(dims2, np.NPY_LONG)
	cdef np.ndarray[c_intp, ndim=2, mode="c"] dest_0 = empty(dims2, np.NPY_INTP)
	cdef np.ndarray[c_intp, ndim=2, mode="c"] lengths = empty(dims2, np.NPY_INTP)
	

	for i in range(c_ndim): #For each outgoing axis
		qm = q_maps[i]
		stride = pipes[i].strides
		axesi = axes[i]
		numa = axesi.shape[0]
				
		for row in range(numrow):
			ind = q_dat[row, axesi[0]]*stride[0]
			for a in range(1, numa): #convert multi-index to flat index
				ind+=q_dat[row, axesi[a]]*stride[a]
				
			lengths[row, i] = qm[ind, 1] - qm[ind, 0]
			dest_0[row, i] = qm[ind, 0]
			Qt[row, i] = qm[ind, 2]		

	cdef np.ndarray[c_intp, ndim=1, mode="c"] sort
	if numrow > 0:
		sort = np.PyArray_LexSort(Qt.T, 0 )
			
		Qt = np.take(Qt, sort, axis = 0, mode = 'wrap')
		lengths = np.take(lengths, sort, axis = 0, mode = 'wrap')
		dest_0 = np.take(dest_0, sort, axis = 0, mode = 'wrap')
		
		#Qt = np.PyArray_Take(Qt, sort, 0)
		#lengths = np.PyArray_Take(lengths, sort, 0)
		#dest_0 = np.PyArray_Take(dest_0, sort, 0)
	
	cdef c_int[::1] bounds = find_differences_contiguous2(Qt, 0, c_ndim) #Boundaries of each QT-ind

	cdef c_intp num_dat = bounds.shape[0] - 1 #number of outgoing data blocks
	
	dims2.ptr = [num_dat, c_ndim]	
	
	cdef np.ndarray[c_uint, ndim=2, mode="c"] c_q_dat = empty(dims2, np.NPY_ULONG)
	cdef np.ndarray[c_intp, ndim=2, mode="c"] c_sh = empty(dims2, np.NPY_INTP)
	
	for i in range(c_ndim): #compute the q_dat entry and shape of each out going block
		qt_ind = pipes[i].qt_ind
		
		for iT in range(num_dat):
			row = Qt[bounds[iT], i]
			c_q_dat[iT, i] = row
			c_sh[iT, i] = qt_ind[row, 1] - qt_ind[row, 0]
		
	cdef list c_dat = [None]*num_dat
	
	cdef list sl
	cdef np.ndarray t, ct
	
	cdef np.PyArray_Dims shpC
	shpC.len = c_ndim
	
	cdef int ctype = c.dtype.num
		
	cdef np.ndarray src
	cdef np.npy_intp* src_strides = <np.npy_intp*>PyMem_Malloc(c_ndim * sizeof(np.npy_intp))
	cdef np.npy_intp* ct_strides
	
	cdef c_uint sort_row
	for iT in range(num_dat):
		shpC.ptr = &c_sh[iT, 0]	
		ct = zeros(shpC, ctype)
		c_dat[iT] = ct
		ct_strides = np.PyArray_STRIDES(ct)
		ct_strides[c_ndim - 1] = np.PyArray_ITEMSIZE(ct) #for RELAXED_STRIDES this entry is arbitrary! FUCKERS

		for row in range(bounds[iT], bounds[iT+1]):
			sort_row = sort[row]
			src = np.PyArray_GETCONTIGUOUS(dat[sort_row])
				
			#src_strides[c_ndim-1] = src.strides[a_ndim-1]
			src_strides[c_ndim-1] = np.PyArray_ITEMSIZE(src) #src.strides[a_ndim-1]
			for i in range(c_ndim - 1, 0, -1):
				src_strides[i - 1]=lengths[row, i]*src_strides[i]
			copy_strided(<char*>ct.data, &dest_0[row, 0], ct.strides, <char*>src.data, NULL, src_strides, &lengths[row, 0], c_ndim)
			
			if inplace:
				dat[sort_row] = 0
	
	PyMem_Free(src_strides)						
	c.q_dat = c_q_dat
	c.dat = c_dat
	c.sorted = True
	return 
		
#########################
##########################
@cython.wraparound(False)
@cython.boundscheck(False)
def itranspose_fast(self, np.ndarray[np.npy_intp, ndim=1, mode="c"] axes):
	"""	IN PLACE. self = np.transpose(self), np-style tranposition. 
		
		Same as itranspose, but assumes axes is an np.npy_intp ndarray	
	"""

	cdef np.PyArray_Dims perm
	perm.len = axes.shape[0]
	perm.ptr = <np.npy_intp*>axes.data
	
	cdef list dat = self.dat
	cdef np.ndarray m
	self.dat = [np.PyArray_Transpose(m, &perm) for m in dat]

	self.shape = np.take(self.shape, axes)#SHAPE
	self.q_conj = np.take(self.q_conj, axes)
	self.q_ind = [ self.q_ind[i] for i in axes]
	self.q_dat = np.take(self.q_dat, axes, axis = 1)
	self.sorted = False
	
	cdef dict labels = self.labels
	cdef np.ndarray[np.npy_intp, ndim=1, mode="c"] order
	if labels is not None:
		order = np.argsort(axes)
		self.labels = { k : order[ labels[k] ] for k in labels.keys() }
		
	return self

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef transpose_fast(self, np.ndarray[np.npy_intp, ndim=1, mode="c"] axes):
	"""	COPY. c = np.transpose(self), np-style tranposition.
		axes = (i0, i1, . . . ) is a permutation of the axes; by default reverses order
			axes[j]=i means a's i-th axis becomes a.transpose()'s j-th axis.
		Transposes data blocks and charge lists, but does not sort them according to new lex order """

	#TODO: don't copy, then transpose.
	cdef np.PyArray_Dims perm
	perm.len = axes.shape[0]
	perm.ptr = <np.npy_intp*>axes.data
	
	a = npc.array(self.rank, dtype = self.dtype)
	#a.shape = np.take(self.shape, axes)
	a.shape = self.shape[axes] #SHAPE
	a.mod_q = self.mod_q
	a.num_q = self.num_q
	a.q_conj = np.take( self.q_conj, axes)
	a.q_ind = [ self.q_ind[i] for i in axes] #DOESN'T copy
	a.charge = self.charge.copy()
	self.sorted = False
	
	cdef list dat = self.dat
	cdef np.ndarray m
	a.dat = [ np.PyArray_NewCopy(np.PyArray_Transpose(m, &perm), np.NPY_CORDER) for m in dat]
	a.q_dat = np.take(self.q_dat, axes, axis = 1)
	
	cdef dict labels = self.labels
	cdef np.ndarray[np.npy_intp, ndim=1, mode="c"] order
	if labels is not None:
		order = np.argsort(axes)
		a.labels = { k : order[ labels[k] ] for k in labels.keys()}

	return a


@cython.nonecheck(False)
def idaxpy(double alpha, np.ndarray x, np.ndarray y):		
	""" y -> y + alpha*x"""
	#cdef double alphaD = alpha
	cdef int nd

	if (np.PyArray_ISCONTIGUOUS(x) and np.PyArray_ISCONTIGUOUS(y)) or (np.PyArray_ISFORTRAN(x) and np.PyArray_ISFORTRAN(y)):
		lib_daxpy(x.size, alpha, <double*>x.data, 1, <double*>y.data, 1 )
		return y
	else:
		y+=alpha*x
		return y

@cython.nonecheck(False)
def izaxpy(double complex alpha, np.ndarray x, np.ndarray y):			
	#cdef double complex alphaDC = alpha
	cdef int nd
	if (np.PyArray_ISCONTIGUOUS(x) and np.PyArray_ISCONTIGUOUS(y)) or (np.PyArray_ISFORTRAN(x) and np.PyArray_ISFORTRAN(y)):
		lib_zaxpy(x.size, &alpha, <void*>x.data, 1, <void*>y.data, 1 )
		return y
	else:
		y+=alpha*x
		return y

cdef extern from "math.h":
	double sqrt(double x)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cpdef two_norm(a):
	"""Computes 2nrm (Frobenius) of tensor
		Calls BLAS directly for doubles, cdoubles
	
	"""
	cdef double c = 0.
	cdef double n = 0.
	cdef c_int i = 0
	cdef np.ndarray t
	cdef list dat = a.dat
	cdef c_int num = len(dat)
	if len(dat)==0:
		return 0.
	cdef c_int type = dat[0].dtype.num
	
	if type==np.NPY_DOUBLE:
		for i in range(num):
			t = dat[i]
			c=lib_dnrm2(t.size, <double*>t.data, 1)
			n+=c*c
	elif type==np.NPY_CDOUBLE:
		for i in range(num):
			t = dat[i]
			c=lib_dznrm2(t.size, <void*>t.data, 1)
			n+=c*c
	else:
		for i in range(num):
			t = dat[i]
			c=np.linalg.norm(t)
			n+=c*c

	return sqrt(n)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def iscal(double alpha, a):
	"""Scales a inplace by real double 'alpha'
		Calls BLAS directly for doubles, cdoubles
	
	"""
	if alpha == 0.:
		a.dat = []
		a.q_dat = np.empty( (0, a.rank), np.uint)
		a.sorted = True
		
	cdef c_int i = 0
	cdef np.ndarray t	
	cdef c_int type = a.dat[0].dtype.num
	
	if type==np.NPY_DOUBLE:
		for i in range(len(a.dat)):
			t = a.dat[i]
			lib_dscal(t.size, alpha, <double*>t.data, 1)
	elif type==np.NPY_CDOUBLE:
		for i in range(len(a.dat)):
			t = a.dat[i]
			lib_zdscal(t.size, alpha, <void*>t.data, 1)

	else:
		for i in range(len(a.dat)):
			a.dat[i]*=alpha

	return a


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def iaxpy(alpha, a, b):
	"""	Optimized inplace axpy operation on npc arrays:
		b -> alpha*a + b

		Calls BLAS directly for a & b double or a & b complex double
	"""
	alpha = np.array(alpha)
	a.sort_q_dat()
	b.sort_q_dat()
		
	cdef list adat = a.dat
	cdef list bdat = b.dat
	cdef np.ndarray[c_uint, ndim=2, mode = "c"] aq = a.q_dat
	cdef np.ndarray[c_uint, ndim=2, mode = "c"] bq = b.q_dat
	cdef c_uint Na = aq.shape[0]
	cdef c_uint Nb = bq.shape[0]
	mul = np.multiply

	#Special cases - a or b == 0
	if Na==0:
		return b
	if Nb==0:
		if alpha==1.:
			b.dat = [ t.copy() for t in adat]
			b.q_dat = aq.copy()
			return b
		if alpha==0.:
			return b
		b.dat = [ mul(t, alpha) for t in adat]
		b.q_dat = aq.copy()
		return b

	cdef c_uint ndim = aq.shape[1]	
	#pre allocate q_dat - probably too big
	cdef np.PyArray_Dims shp
	shp.len = 2
	shp.ptr = [Na+Nb, ndim]
	cdef np.ndarray[c_uint, ndim=2, mode = "c"] cq = empty( shp, np.NPY_ULONG)
	cdef c_uint i = 0
	cdef c_uint j = 0
	cdef c_uint num_dat = 0
	#cdef list q_dat = []
	cdef list	dat = []
	cdef bint lexcomp
	cdef c_int k
	cdef bint is_zero
	cdef np.ndarray ta
	cdef np.ndarray tb

	cdef c_int type = adat[0].dtype.num
	#For double and cdouble types, we will call BLAS directly.
	if type==np.NPY_DOUBLE and bdat[0].dtype.num==type and alpha.dtype.num == type:
		pass
	elif type==np.NPY_CDOUBLE and bdat[0].dtype.num==type:
		pass
	else:
		type = -1

	cdef double complex zalpha
	zalpha.real = alpha.real; zalpha.imag =  alpha.imag #alpha.real+1j*alpha.imag
	cdef double dalpha = alpha.real

	while i < Na or j < Nb:
		is_zero=False
		if j >= Nb:
			is_zero=True
		elif i < Na:
			is_zero = False
			for k in range(ndim-1, -1, -1):
				if aq[i, k] < bq[j, k]:
					is_zero = True
					break
				if aq[i, k] > bq[j, k]:
					break
		if is_zero: #b is 0
			dat.append( mul(adat[i], alpha))

			for k in range(ndim):
				cq[num_dat, k] = aq[i, k]
			num_dat+=1
			#q_dat.append(aq[i])
			i+=1
		else:
			is_zero=False
			if i >= Na:
				is_zero=True
			else:
				is_zero=False
				for k in range(ndim-1, -1, -1):
					if aq[i, k] > bq[j, k]:
						is_zero = True
						break
					if aq[i, k] < bq[j, k]:
						break
					
			if is_zero: #a is 0
				dat.append(bdat[j])
				for k in range(ndim):
					cq[num_dat, k] = bq[j, k]
				num_dat+=1
				#q_dat.append(bq[j])
				j+=1
			else: #both are non zero
				ta = np.PyArray_GETCONTIGUOUS(adat[i])
				tb = np.PyArray_GETCONTIGUOUS(bdat[j])
				if type==np.NPY_DOUBLE:
					lib_daxpy(ta.size, dalpha, <double*>(ta.data), 1, <double*>(tb.data), 1)
				elif type==np.NPY_CDOUBLE:
					lib_zaxpy(ta.size, &zalpha, <void*>(ta.data), 1, <void*>(tb.data), 1)
				else:
					tb = tb+mul(ta, alpha)
				dat.append(tb)
				
				for k in range(ndim):
					cq[num_dat, k] = bq[j, k]
				num_dat+=1
				#q_dat.append(bq[j])
				
				i+=1
				j+=1
	

	b.dat = dat
	cdef np.ndarray[c_uint, ndim=2] q_dat
	if num_dat < Na + Nb:
		shp.ptr = [num_dat, ndim]
		q_dat = empty(shp, np.NPY_ULONG)
		for i in range(num_dat):
			for j in range(ndim):
				q_dat[i, j] = cq[i, j]
		b.q_dat = q_dat
	else:
		b.q_dat = cq

		#b.q_dat = np.array(q_dat, np.uint)

	if len(b.dat) > 0:
		b.dtype = b.dat[0].dtype
	
	return b

@cython.wraparound(False)
@cython.boundscheck(False)
def inner(a, b, axes = None, bint do_conj = False):
	"""	Contracts all the legs in a and b and return a scalar.

		If axes is None, contracts all axes in matching order
		Otherwise, axes = [ [i1, i2 , . . . ] , [j1, j2, . . . ] ],
		 	contracting i1j1, i2j2, etc.
		
		If do_conj = True, a is conjugated before contraction (giving hermitian inner product)
		"""
	
	if TENSORDOT_TIMING:
		t0 = time.time()
	
	if id(a) == id(b):
		#raise NotImplementedError, "Use npc.norm for identical arrays"
		print "making copy for inner"
		b = b.copy()
	
	cdef list axes_a
	cdef list axes_b
	if axes is None:
		axes_a = range(a.rank)
		axes_b = range(b.rank)
	else:
		axes_a = list(axes[0])
		axes_b = list(axes[1])
	
	cdef c_uint nda = len(axes_a)
	cdef c_uint ndb = len(axes_b)
	if nda!=a.rank or ndb!=b.rank:
		raise ValueError, "Incomplete contraction requested."
	if nda!=ndb:
		raise ValueError, "Tensors to be contracted have different dimensions."
		
	cdef c_uint ndim = nda
	# axes_a,b are now each list (same len) of indices to contract (na,b are the number of indices)

	cdef np.ndarray[c_intp, ndim=1, mode = "c"] as_ = a.shape
	cdef np.ndarray[c_intp, ndim=1, mode = "c"] bs = b.shape

	cdef bint equal = 1
	cdef c_int k
	for k in range(ndim):
		if as_[axes_a[k]] != bs[axes_b[k]]:
			equal = 0
			break
		if axes_a[k] < 0: axes_a[k] += ndim
		if axes_b[k] < 0: axes_b[k] += ndim
	if not equal:
		print "Shapes:", [as_[axes_a[k]] for k in range(ndim)], [bs[axes_b[k]] for k in range(ndim)]
		raise ValueError, "Shape-mismatch for inner."

	if (do_conj and array_equiv_mod_q(a.charge, b.charge, a.mod_q) == False) or (do_conj==False and array_equiv_mod_q(-1*a.charge, b.charge, a.mod_q) == False):
		return 0.

	if len(a.q_dat) == 0 or len(b.q_dat) == 0:
		return 0

	#TODO - no reason to transpose both, just bring b to a
	for k in range(ndim):
		if axes_a[k]!=k:
			a = a.transpose_fast(np.array(axes_a, dtype = np.intp))
			break

	for k in range(ndim):
		if axes_b[k]!=k:
			b.transpose_fast(np.array(axes_b, dtype = np.intp))
			break



	cdef c_intp num_q = a.num_q

	a.sort_q_dat()
	b.sort_q_dat()

	cdef np.ndarray[c_uint, ndim=2, mode = "c"] aq = a.q_dat
	cdef np.ndarray[c_uint, ndim=2, mode = "c"] bq = b.q_dat

	cdef c_intp Na = aq.shape[0]
	cdef c_intp Nb = bq.shape[0]
	cdef  double complex zc = 0.
	cdef  double complex zc_temp = 0.
	cdef  double dc = 0.
	c = 0.

	cdef list adat = a.dat
	cdef list bdat = b.dat


	cdef tuple triv = (-1,)

	cdef np.ndarray ta
	cdef np.ndarray tb

	cdef c_uint i = 0
	cdef c_uint j = 0
	cdef c_int lex_i
	

	cdef c_int comp

	if BLAS_COMPLEX_INNER and a.dtype.num==np.NPY_CDOUBLE and b.dtype.num==np.NPY_CDOUBLE:
			while i < Na and j < Nb:
				comp = 0
				for lex_i in range(ndim-1, -1, -1):
					if bq[j, lex_i] > aq[i, lex_i]:
						comp = 1
						break
					elif bq[j, lex_i] < aq[i, lex_i]:
						comp = -1
						break

				if comp==1:
					i+=1
				elif comp==-1:
					j+=1
				else:
					ta = adat[i]
					tb = bdat[j]
					ta = np.PyArray_GETCONTIGUOUS(ta)
					tb = np.PyArray_GETCONTIGUOUS(tb)
					if do_conj:
						lib_zdotc_sub(ta.size, <void*>(ta.data), 1, <void*>(tb.data), 1, &zc_temp )
					else:
						lib_zdotu_sub(ta.size, <void*>(ta.data), 1, <void*>(tb.data), 1, &zc_temp )
					zc+=zc_temp
					i+=1
					j+=1
			c = zc

	elif a.dtype.num==np.NPY_DOUBLE and b.dtype.num==np.NPY_DOUBLE:
		while i < Na and j < Nb:
			comp = 0
			for lex_i in range(ndim-1, -1, -1):
				if bq[j, lex_i] > aq[i, lex_i]:
					comp = 1
					break
				elif bq[j, lex_i] < aq[i, lex_i]:
					comp = -1
					break

			if comp==1:
				i+=1
			elif comp==-1:
				j+=1
			else:
				ta = adat[i]
				tb = bdat[j]
				ta = np.PyArray_GETCONTIGUOUS(ta)
				tb = np.PyArray_GETCONTIGUOUS(tb)

				dc+=lib_ddot(ta.size, <double*>(ta.data), 1, <double*>(tb.data), 1 )
				i+=1
				j+=1
			c = dc
	else:
		c = 0
		while i < Na and j < Nb:
			comp = 0
			for lex_i in range(ndim-1, -1, -1):
				if bq[j, lex_i] > aq[i, lex_i]:
					comp = 1
					break
				elif bq[j, lex_i] < aq[i, lex_i]:
					comp = -1
					break

			if comp==1:
				i+=1
			elif comp==-1:
				j+=1
			else:
				ta = adat[i]
				tb = bdat[j]
				if do_conj:
					c+=np_vdot( ta.reshape(triv), tb.reshape(triv) )
				else:
					c+=np_inner( ta.reshape(triv), tb.reshape(triv) )
				i+=1
				j+=1
	
		
	return c

################################################################################
################################################################################
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cpdef bint is_blocked_by_charge(np.ndarray[c_int, ndim=2] q_ind):
	"""Given a q_ind 2 - array, determines if the leg is fully blocked by charges.
		I.e., do the charges uniquely determine the q-index?
	"""

	cdef c_intp num_q = q_ind.shape[1] - 2
	cdef c_intp length = q_ind.shape[0]

	if length < 2:
		return 1
		
	if num_q == 0: #Then should have length 1 if blocked
		return 0
	
	#So num_q > 0 and more that 1 entry
	cdef np.ndarray[c_intp, ndim=1, mode='c'] sort = np.lexsort( q_ind[:, 2:].T)
	cdef c_uint i
	cdef c_uint j
	cdef bint blocked = True
	cdef bint same

	for i in range(0, length - 1): #For each pair of (sorted) entries
		same = ( q_ind[sort[i], 2] == q_ind[sort[i + 1], 2] ) #I
		for j in range(3, num_q + 2):
			same = same and ( q_ind[sort[i], j] == q_ind[sort[i + 1], j])
			
		if same:
			return 0
	
	return 1


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef bint array_equiv_mod_q(np.ndarray[c_int, ndim=1] a1, np.ndarray[c_int, ndim=1] a2, np.ndarray[c_int, ndim=1] mod_q):
	"""	Check if a1 == a2 (mod mod_q). """

	cdef c_uint len = a2.shape[0]
	
	cdef c_uint i
	
	for i in range(len):
		if (a1[i] - a2[i]) %  mod_q[i] + (mod_q[i] == 1)*(a1[i] - a2[i]):
			return 0
		
	return 1

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef np.ndarray[c_int, ndim=1] mod_onetoinf1D(np.ndarray[c_int, ndim=1] a, np.ndarray[c_int, ndim=1] mod_q):
	"""	Return a (mod mod_q). """

	cdef c_intp num_q = a.shape[0]
	cdef np.ndarray[c_int, ndim=1] b = np.empty_like(a)
	
	cdef c_uint i
	
	#Annoying - must correct for C-style modulo (versus python)
	for i in range(num_q):
		if mod_q[i]!=1:
			b[i] = a[i]%mod_q[i]
			if b[i] < 0:
				b[i]+=mod_q[i]
		else:
			b[i] = a[i]


	return b


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef np.ndarray[c_int, ndim=2] mod_onetoinf2D(np.ndarray[c_int, ndim=2] a, np.ndarray[c_int, ndim=1] mod_q):
	"""	Return a (mod mod_q). """

	cdef c_intp lenR = a.shape[0]
	cdef c_intp num_q = a.shape[1]
	cdef np.ndarray[c_int, ndim=2] b = np.empty_like(a)
	
	cdef c_uint i, j
	for j in range(num_q):
		if mod_q[j] == 1:
			for i in range(lenR):
				b[i, j] = a[i, j]
		else:
			for i in range(lenR):
				b[i, j] = a[i, j]%mod_q[j]
				if b[i, j] < 0:
					b[i, j]+=mod_q[j]
						
	return b

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef np.ndarray[c_int, ndim=1] block_charge(self, np.ndarray[c_uint, ndim=1] ind):
	"""	Returns charge of a block (as ndarray)"""
	
	cdef list q_ind = self.q_ind
	cdef np.ndarray[c_int, ndim=1] q_conj = self.q_conj
	cdef np.ndarray[c_int, ndim=1] mod_q = self.mod_q
	cdef np.ndarray[c_int, ndim=1] charge = np.empty_like(mod_q)
		
	cdef c_int rank = ind.shape[0]
	cdef np.ndarray[c_int, ndim=2] q_i
	cdef c_int r, i, q_c, c

	q_i = q_ind[0]
	q_c = q_conj[0]
	r = ind[0]
	
	cdef c_intp num_q = q_i.shape[1]-2
	
	for c in range(num_q):
		charge[c] = q_i[r, 2+c] * q_c
	
	for i in range(1, rank):
		q_i = q_ind[i]
		r = ind[i]
		q_c = q_conj[i]
		for c in range(num_q):
			charge[c] += q_i[r, 2+c] * q_c
	
	for i in range(num_q):
		charge[i] = charge[i]%mod_q[i] + (mod_q[i] == 1)*charge[i]
		
	return charge		## mod q
	
################################################################################
################################################################################
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cpdef c_int find_row_uint(np.ndarray[c_uint, ndim=2] t, np.ndarray[c_uint, ndim=1] r):
	"""	Find the first row of t which matches r.
		
		Returns:
			The first index i such that t[i, :] = r,
			Return -1 if none of them match.
		"""
	cdef c_uint i, j
	cdef c_uint numrow = t.shape[0]
	cdef c_uint numcol = t.shape[1]
	cdef int match

	for i in range(numrow):
		match = 1
		for j in range(numcol):
			if t[i, j] != r[j]:
				match = 0
				break
		if match == 1:
			return i
	
	return -1



################################################################################
################################################################################
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef np.ndarray[c_int, ndim=1, mode='c'] find_differences_contiguous(np.ndarray[c_int, ndim=2, mode='c'] a, c_uint startc, c_uint stopc):
	"""Given 2 - array, returns 1-array of bounds where rows differ.
		startc, stopc delineated which columns should be searched.
		assumes that startc and stopc are in range, otherwise funny things happen

		if startc == stopc, then it returns two-elements which gives the entire range
		"""
	cdef np.PyArray_Dims dims1
	dims1.len = 1
	dims1.ptr = [a.shape[0] + 1]
	cdef np.ndarray[c_int, ndim=1, mode='c'] indices = empty(dims1, np.NPY_LONG)
	
	indices[0] = 0
	cdef c_uint num = 1	# pointer to current row of indices
	cdef c_uint i	# row loop index
	cdef c_uint j	# column loop index
	
	for i in range(1, a.shape[0]):
		for j in range(startc, stopc):
			if a[i, j]!= a[i-1, j]:
				indices[num] = i
				num = num + 1
				break
	indices[num] = a.shape[0]
	
	return indices[0:num+1]

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef c_int[::1] find_differences_contiguous2(c_int[:, ::1] a, c_uint startc, c_uint stopc):
	"""Given 2 - array, returns 1-array of bounds where rows differ.
		startc, stopc delineated which columns should be searched.
		assumes that startc and stopc are in range, otherwise funny things happen

		if startc == stopc, then it returns two-elements which gives the entire range
		"""
	cdef np.PyArray_Dims dims1
	dims1.len = 1
	dims1.ptr = [a.shape[0] + 1]
	cdef c_int[::1] indices = empty(dims1, np.NPY_LONG)
	
	indices[0] = 0
	cdef c_intp num = 1	# pointer to current row of indices
	cdef c_intp i	# row loop index
	cdef c_intp j	# column loop index
	
	for i in range(1, a.shape[0]):
		for j in range(startc, stopc):
			if a[i, j]!= a[i-1, j]:
				indices[num] = i
				num = num + 1
				break
	indices[num] = a.shape[0]
	
	return indices[0:num+1]

################################################################################
################################################################################
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cpdef np.ndarray[c_int, ndim=1, mode='c'] find_differences(np.ndarray[c_int, ndim=2] a):
	"""Given 2 - array, returns 1-array of bounds where rows differ.
	"""

	cdef np.PyArray_Dims dims1
	dims1.len = 1
	dims1.ptr = [a.shape[0] + 1]
	cdef np.ndarray[c_int, ndim=1, mode='c'] indices = empty(dims1, np.NPY_LONG)
	indices[0] = 0
	cdef c_uint num = 1	# pointer to current row of indices
	cdef c_uint i	# row loop index
	cdef c_uint j	# column loop index
	
	cdef c_uint stopc = a.shape[1]
	for i in range(1, a.shape[0]):
		for j in range(stopc):
			if a[i, j]!= a[i-1, j]:
				indices[num] = i
				num = num + 1
				break
	indices[num] = a.shape[0]
	
	return indices[0:num+1]



################################################################################
################################################################################
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef np.ndarray[c_int, ndim=2, mode='c'] q_sum( \
			np.ndarray[c_uint, ndim=2] q_dat, \
			list q_ind, \
			np.ndarray[c_int, ndim=1, mode='c'] q_conj, \
			int startc, int stopc, \
			np.ndarray[c_int, ndim=1, mode='c'] mod_q ):
	"""	Accumulates the total charge in each entry of q_dat for a subset of legs specified by startc, stopc.
	
		charge[r] = sum_{start <= l < stop} q_ind[l][ q_dat[r, l], -num_q:] * q_conj[l]
		mod_q is an nparray shaped (num_q,), must be all positive int
		
		Returns
			A 2D nparray shaped (len(q_dat),num_q), list of total charges
	"""
	
	cdef c_uint r, c, l
	cdef c_int length = q_dat.shape[0]
	
	cdef np.ndarray[c_int, ndim=2] qi = q_ind[startc] 
	cdef c_intp num_q = qi.shape[1] - 2
	
	cdef np.PyArray_Dims dims2
	dims2.len = 2
	dims2.ptr = [length, num_q]	
	cdef np.ndarray[c_int, ndim=2, mode='c'] charges = empty(dims2, np.NPY_LONG)
	
	#for first leg, we directly initialize
	cdef c_int qconj = q_conj[startc]
	for r in range(length):
		for c in range(num_q):
			charges[r, c] = qconj*qi[ q_dat[r, startc], 2 + c ]
	
	#for remaining legs, we accumulate
	for l in range(startc + 1, stopc):
		qi = q_ind[l]		# deref leg variables
		qconj = q_conj[l]
		for r in range(length):		#i for each entry in q_dat
			for c in range(num_q):		#i vectorize across charge
				charges[r, c] += qconj*qi[ q_dat[r, l], 2 + c ]
	
	cdef c_int mod_by
	cdef c_int q
	for c in range(num_q):
		mod_by = <c_int>mod_q[c]
		if mod_by != 1:
			for r in range(length):
				q = charges[r, c]%mod_by
				while q < 0:
					q+= mod_by
				charges[r, c] = q
	
	return charges
	
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void copy_strided(char* dest, np.npy_intp* dest_0, np.npy_intp* dest_strides, char* src, np.npy_intp* src_0, np.npy_intp* src_strides, np.npy_intp* len, int nd ) nogil:
	""" This is a strided copy function in order to accomplish
	
			a[sl1, sl2 . . . ] = b[sl1', sl2', . . . ]
		
		for slices sl*.
		
		CAUTION: it is assumed that both a and b are C-contiguous.
		
		As an example, consider a[1:3, 0:3] = b[4:6, 1:4]
		
		dest: a.data (pointers to the underlying data)
		src: b.data
		
		dest_0: an intp array designating the top left of the block to be copied, ie, [1, 0]
		src_0: an intp array designating the top left of the block to be copied, ie, [4, 1]		If either of *_0 == NULL, it is assumed to be [0, 0]
		
		dest_strides: the strides of a, ie, a.strides
		src_strices: the strides of b, ie, b.strides
		
		len: an intp array designating the lenght of each slice, ie, [2, 3]
		nd : the number of dimensions.
		
		NOTE: *_0 can also be accomplished by letting dest, src point to the top of the block, ie,
		
		copy_strided(a.data, [1, 1], . . . ) = copy_strided(<char*>&a.data[1,1], NULL, . . . )
		
	"""
	
	cdef np.npy_intp i, j, d0, s0, d1, s1, width
	
	if nd<1:
		return
	
	#Offset data to top left entry
	if dest_0!=NULL:
		j = 0
		for i in range(nd):
			j+=dest_0[i]*dest_strides[i]
		dest = &dest[j]
	if src_0!=NULL:
		j = 0
		for i in range(nd):
			j+=src_0[i]*src_strides[i]
		src = &src[j]
	
	#explicitly unroll first 3 levels
	if nd==1:
		width = len[0]*dest_strides[0]
		memcpy(dest, src, width)
		return

	if nd==2:
		width = len[1]*dest_strides[1]
		d0 = dest_strides[0]
		s0 = src_strides[0]
				
		for i in range(len[0]):
			memcpy(&dest[i*d0] , &src[i*s0], width)
		return
			
	if nd==3:
		width = len[2]*dest_strides[2]
		d1 = dest_strides[1]
		d0 = dest_strides[0]
		s1 = src_strides[1]
		s0 = src_strides[0]
		
		for i in range(len[0]):
			for j in range(len[1]):
				memcpy(&dest[i*d0 + j*d1] , &src[i*s0 + j*s1], width)
		return
		
	else: #From here out, define recursively.
		d0 = dest_strides[0]
		s0 = src_strides[0]
		
		for j in range(len[0]):
			copy_strided(&dest[j*d0], NULL, &dest_strides[1], &src[j*s0], NULL, &src_strides[1], &len[1], nd-1)



################################################################################
################################################################################
# this is here just for show, tensordot_guts already inlines this
@cython.wraparound(False)
@cython.boundscheck(False)
cpdef int lex_comp_gt(np.ndarray[c_uint, ndim=1] a, np.ndarray[c_uint, ndim=1] b, c_int startc, c_int stopc):
	"""lex - comparison of 1d numpy array

		returns 1 if a > b, 0 if a == b, -1 if a < b
	"""
	cdef c_int i
	for i in range(stopc - 1, startc - 1, -1):
		if a[i] > b[i]:
			return 1
		elif a[i] < b[i]:
			return -1
		
	return 0



################## multilayer QH ##############


@cython.wraparound(False)
@cython.boundscheck(False)
def packVmk(list Vmk_list, np.ndarray[c_float, ndim=6] vmk, np.ndarray[np.uint8_t, ndim=6] mask,  c_int maxM, c_int maxK, spec1, spec2, c_float tol):
	""" This code if actually for multilayerQH"""
### COMMENTS COMMENTS COMMENTS!!
### COMMENTS COMMENTS COMMENTS!!
### COMMENTS COMMENTS COMMENTS!!
### COMMENTS COMMENTS COMMENTS!!
	
	cdef c_int a, b, c, d, m, k, sa, sb, sc, sd
	cdef c_float v
	cdef c_int n_mu = vmk.shape[0]
	cdef c_int n_nu = vmk.shape[2]
	
	for a in range(n_mu):
		sa = spec1[a]
		for b in range(n_mu):
			sb = spec1[b]
			for c in range(n_nu):
				sc = spec2[c]
				for d in range(n_nu):
					sd = spec2[d]
					for m in range(0, 2*maxM+1):
						m_loc = m-maxM
						for k in range(0, 2*maxK+1):		
							v = vmk[a, b, c, d, m, k]
							if abs(v) > tol:
								Vmk_list.append([ m_loc, k-maxK, sa, sc, sd, sb, v, mask[a, b, c, d, m, k] ])


