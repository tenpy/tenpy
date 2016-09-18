
"""	conserved-charge tensor package (np_conserved or npc)

	it is designed to have similar interfaces as numpy.ndarray, so that one can almost reuse old code
	but as with all code, only Mike really understands the mess created below.
	"""
################################################################################
"""	Some notations
	
	For tensor T_{i1, i2, ... in}

	Each "leg" of a tensor corresponds the vector space spanned by some 'im'.
	The "rank" of the tensor is the number of legs, ie, n. 
	An "index" of leg m is a particular value of im, and specifies a particular basis element. 
	
	A "block" of a leg is a contiguous slice of the leg containing indices of the same charge. However, it does not have to contain every index with that charge.
	A "q-index" labels a block of a leg. Sometimes a 'block' and 'q-index' are used interchangeably.
	A "charge sector" of a leg is a charge and the set of all indices of that charge. The indices may be a single block, or may be split into several blocks.	
"""

"""					Introduction to the conserved-charge data structures.

	Each npc.array object (counterpart to np.ndarray) encodes a tensor in addition to information about the charges of the tensor. The charges are associated independently to each 'leg'. 

	The nature of the charge is specified by num_q (how many abelian charges) and an array mod_q (whether the charge 'n' is conserved modulo mod_q[n]). We set mod_q[n] = 1 if it is a U(1) charge.

	Information concerning the charge structure on each leg is contained in q_ind, q_conj and q_dat.
	
	The actual entries of the tensor are stored in .dat, a list of np.arrays storing the non zero blocks of the tensor.

									----- Charge Data ------

	For any particular leg of an npc.array object, each index has an associated charge, which is what tells npc which arrays to store / multiply.
	
	
	This can be represented in "flat form" (q_flat) where the charges are simply listed.
	This can also be stored in table form (q_ind); each row of the table corresponds to a slice of a leg of fixed charge (a block); each block is labeled by the 'q-index', which we take to be it's row in the table q_ind.
	Finally, it is also sometimes convenient to use a dictionary (q_dict) which takes the charge sector of a leg to the slice of the leg with that charge.
	
	In npc.array, we store the information in q_ind form.

	Example: 
	(For simplicity, we'll assume only a single conserved charge (num_q=1))

		In q_flat form, we have [[-2], [-1], [-1], [0], [0], [0], [0], [3], [3]], it tells you that the leg has size 9, and the charges for each entry are -2, -1 etc.
			Notice that there are four charge sectors (charges [-2], [-1], [0], [3]), and the q-index (0 to 3) labels these sectors
		q_ind is a map from the q-indices (row) to  slice/charges (col) on the leg.
			The corresponding q_ind form is a 2-array
			[[0 1 -2]
			 [1 3 -1]
			 [3 7  0]
			 [7 9  3]]
			the first two columns are the range of indices for each q-index. The remaining columns are the charge
			the table above tells you that the leg is divided in to four charge sectors, labeled by the q-index (0 to 3)
				q-index 0 has charge -2, and takes up the slice [0:1] in flat form,
				q-index 1 has charge -1, occupies slice [1:3] ... etc.
		The q_dict form is a dictionary from charges to slice objects
			{ (-2,): [0:1], (-1,): [1:3], (0,):[3:7], (3,):[7:9] }



	NOTE: For most algorithms (such as tdot), the charges do NOT actually have to be completely blocked by charge. For example, we could have
	
		q_flat = [ [-2], [0], [0], [-2] ] 
		q_ind = [ [0 1 -2]
				  [1 3  0]
				  [3 4 -2] ]	
		q_dict = (only makes sense if charge is completely blocked)
	
		So charge sector -2 has been split into two blocks. Sometimes we refer to this as "fractured", as opposed to "completely blocked by charge."
		
		At the moment, the only algorithms that DO require complete blocking are those that must view the tensor as a matrix, such as eigh and SVD. This is because they assume the matrices are block diagonal.
	
		RULE - any alg that requires completely blocking should state this in its doc - string!
	
		If you expect the tensor to be dense subject to charge constraint, it will be most efficient to fully block by charge, so that work is done on large chunks.
		
		However, if you expect the tensor to be sparser than required by charge (as for an MPO), it may be convenient not to completely block, which forces smaller matrices to be stored, and hence many zeroes to be dropped. This allows npc to in effect implement sparse storage algorithms. Nevertheless, the algorithms were not designed with this in mind, so it is not recommended in general.
	
	
								-------	The pesky q_conj -------
	
	By convention, we take all charge to point "inward". But consider two tensors we intend to contract,

		A_{a b} B_{b c}
	
	With the inward pointing convention, the charges on 'b' must be equal and opposite for A, B. Algorithmically, it is very convenient to allow a SINGLE q_ind to be shared by both, with a flag denoting that the charge flows "in" to one, and "out" of the other. This flag is the q_conj flag.
	
	q_conj = [1, 1, -1, . . . ] is a 1-array which denotes that the charge stored in q_ind flows in (1) or out (-1) on the nth leg.
	
	Hence we could make one q_ind_b for the bond, and set
	
	A.q_ind[1] = q_ind_b
	B.q_ind[0] = q_ind_b
	
	A.q_conj[1] = 1
	B.q_conj[0] = -1
	
	So when the algorithm tallies charges, it must multiply by q_conj where appropriate.
	
							-------	q_ind storage -------
							
	In some sense, q_ind really lives on bonds rather than tensors. If we SVD C_{i j} = A_{i k} B_{k j}, it is most natural to make a single q_ind for k, and assign A and B the SAME copy, with q_conj = 1 for A, q_conj = -1 for B. Likewise, if we use tensordot to construct C = A.B, we can use the uncontracted q_ind of A and B for the q_ind of C.
	
	This leads to the following convention:
	
	* When an npc algorithm makes tensors which share a bond (either with the input tensors, as for tensordot, or amongst the output tensors, as for SVD) the algorithm is free, but not required, to use the SAME q_ind for the tensors sharing the bond, without making a copy *
	
	This requires the following rule:
	
	RULE - If a tensor ever modifies its q_ind, it MUST make a copy first, because other tensors may be pointing to the same q_ind.
		
	
	
	With q_ind and q_conj set, we have fully specified the charge data of each leg.
	
							------- q_dat and dat --------
							
	The actual data of the tensor is stored in 'dat'. Rather than keeping a single np.array (which would have many zeros in it), we store only the non-zero sub blocks. So dat is a python list of np.arrays. The order in which they are stored in the list is not physically meaningful, and so not guaranteed (more on this later). So to figure out where the sub block sits in the tensor, we need the 'q_dat' structure.
	
	Consider a rank 3 tensor, with q_ind something like
	
	q_ind[0] = [ [0 1 -2]			(and something else for q_ind[1], q_ind[2] )
				 [1 4  1]
					... ]
					
	Each row of q_ind[i] is a "block" of leg[i], labeled by its q-index (which is just its row in q_ind). Picking a block from each leg, we have a subblock of the tensor.
	
	For each non-zero subblock of the tensor, we put a np.array entry in the .dat list. Since each subblock of the tensor is specified by 'rank' q-indices, we put a corresponding entry in q_dat, which is a [#blocks x rank] 2 - array. Each row corresponds to a non-zero subblock, and there are rank columns giving the corresponding q-indices.
		
	Example; for a rank 3 tensor we might have
	
	dat = [t1, t2, t3, t4, ...]
	q_dat = [ 	[3 2 1]
				[1 1 1]
				[4 2 2]
				[3 1 2]
				  ....	]
	The 'third' subblock has an nd.array t3, and q-indices [4 2 2]. Recall that each row of q_ind looks like [start, stop, charge]. So:
		
		To find  t3s position in the actual tensor, we would look at the data
	
				q_ind[0][4, 0:2], q_ind[1][2, 0:2], q_ind[2][2, 0:2]		  
	
		To find the charge of t3, we would look at
				
				q_ind[0][4, 2:], q_ind[1][2, 2:], q_ind[2][2, 2:]
	
				----------  More on q_dat ordering ----------
	
	Recall that the order in which the blocks stored in dat/q_dat is arbitrary (though of course dat and q_dat must be in correspondence). However, for many purposes it is useful to sort them according to some convention. So we include a flag ".sorted". If .sorted = True, then q_dat is gauranteed to be lex-sorted by q-index. So, if sorted, the q_dat example above goes to
		
		q_dat = [ 	[1 1 1]
					[3 2 1]
					[3 1 2]
					[4 2 2]	]			  		  
	
		Note that np.lexsort choose the right-most column to be the dominant key, a convention we follow throughout.
		
		RULE: If .sorted = True, q_dat and dat are lexsorted. If .sorted = False, there is no gaurantee. If an algorithm modifies q_dat, it MUST set .sorted = False (unless it gaurantees it is still sorted).

		The routine sort_q_dat brings the data to sorted form.
		
"""

"""	Array Creation

	Making an npc.array requires both the tensor entries (data) and charge data.
	
	The data can be provided either as a dense np.array (from_ndarray, from_ndarray_flat) or by providing a numpy function such as np.random, np.ones etc. (from_npfunc).
	
	For the from_ndarray methods, the charge data is provided either by specifying q_flats (via from_ndarray_flat), or q_inds (from_ndarray)
		Note about q_ind: unlike q_conj, it is referenced and not copied - so it can be shared across many npc.arrays. Consequently, one must be careful not to alter q_ind after their creation, or something bad will happen! To be safe, you can always copy q_ind before passing to from_ndarray.
"""


"""	Implementation notes:
	q_ind is NOT copied when creating via from_ndarray (or similar functions).
		Hence they must not be altered after creation, this is done such that multiple npc.array can share q_ind's
		If somehow one wants to alter the array after the array is built, pass in a copy to from_ndarray()
	Similarly, mod_q is also not copied.

	On the other hand, q_conj, charge, are copied

	While the code was designed in such a way that each charge sector has a different charge, most of the code
		will still run correctly if multiple charge sectors (q-indices) correspond to the same charge.  In this sense
		npc.array acts like a sparse array class and can selectively store subblocks.  This way of using the class is
		not recommended - unless the user really knows the ins and outs of this class.
"""

"""
				Introduction to combine_legs, split_legs, and pipes.

Unlike an np.array, the only sensible "reshape" operation on an npc.array is to combine multiple legs into one (combine_legs),  or the reverse (split_legs). 

Each leg has a Hilbert space, and a representation of the symmetry on that Hilbert space. Combining legs  corresponds to the tensor product operation, and for abelian groups, the corresponding "fusion" of the representation is the simple addition of charge. 

Fusion is not a lossless process, so if we ever want to split the combined leg, we need some additional data to tell us how to reverse the tensor product. This data is called a "leg_pipe", which we implemented as a class. A detail of the information contained in the leg_pipe is given in the class doc string.

Rough usage idea:

a: If you want to combine legs, and do NOT intend to  split any of the newly formed legs back, you can call combine_legs without supplying any leg_pipes. (combine_legs will then make them for you).

	Nevertheless, if you plan to perform the combination over and over again on sets of legs you know to be identical (ie, same charges etc.) you might make a leg_pipe anyway to save on the overhead of computing it each time.

b: If you want to combine legs, and for some subset of the new legs you will want to split back (either on the tensor in question, or progeny formed by SVD, tensordot, etc), you DO need to compute a leg_pipe for the legs in questions before combining them. split_legs will then use the pipes to split the leg.


(more details in body)


"""

"""	Leg Labeling

		It's convenient to name the legs of a tensor: for instance, we can name legs 0, 1, 2   to be 'a', 'b', 'c':
		
			T_{i_a i_b i_c}

		That way we don't have to remember the ordering!
		
		Under tensordot, we can then call
		
			U = tensordot(S, T, axes = [ [...],  ['b'] ] )
		
		without having to remember where exactly 'b' is. Obviously U should then inherit the name of its legs from the uncontracted legs of S, T.
		
		So here is how it works:
		
		- Labels can ONLY be strings. The labels should not include the character '.' or '?' .  Internally, the labels are stored as dict
			
				a.labels = { label: leg_position, . . . }
		
			Not all legs need a label.
		
		- To set the labels, call
		
				A.set_labels( ['a', 'b', None, 'c' . . . ] )
			
			which will set up the labeling
					
				['a', 'b', 'c' . . . ]  -> [0, 1, 3, . . . ]
					
		- Where implemented (list to follow), the specification of axes can use either the labels OR the index positions. For instance, the call
		
			tensordot(A, B, [ ['a', 2, 'c'], [...]]

			will interpret 'a', 'c' as labels (and find their positions using the dict) and 2 as 'the 2nd leg'

			That's why we require labels to be strings!
		
		- Labels will be intelligently inherited through the various operations.Implemented in:
		
				tensordot, transpose, conj, svd, pinv, combine_legs, split_legs
		
		- Under transpose, labels are permuted.
		
		- Under conj, iconj: takes 	'a' -> 'a*'
									'a*'-> 'a'
							
		- Under tensordot, labels are inherited from uncontracted legs. If there is a collision, both labels are dropped.
			
		- Under combine_legs, labels get concatenated with a '.' delimiter. 
		
			Example: let a.labels = ['a', 'b', 'c'] -> [0, 1, 2]

			Then if b = a.combine_legs([[0, 1], [2] ])
			
				b.labels = [ 'a.b', 'c'] -> [0, 1]
							
			If some sub-leg of a combined leg isn't named, then a '?x' label is inserted: 'a.?0.c' etc.
			
									
		- Under split_legs, the labels are split using the delimiters (and the ?x are dropped)
		
		- SVD : the outer labels are inherited, and inner labels can be optionally passed
		
		- pinv : transpose
"""

################################################################################
################################################################################
import numpy as np
import scipy as sp
import scipy.linalg as linalg
import svd_dgesvd, svd_zgesvd
import itertools
import functools
import warnings
import time
import cPickle
from numbers import Number
from numpy import random
from algorithms.linalg import npc_helper
from tools.math import toiterable
from tools.math import tonparray
from tools.math import anynan
from tools.string import joinstr
from tools.math import speigs as sp_speigs
#Uncomment this to find errant prints
"""
import sys
import traceback

class TracePrints(object):
  def __init__(self):    
    self.stdout = sys.stdout
  def write(self, s):
    self.stdout.write("Writing %r\n" % s)
    traceback.print_stack(file=self.stdout)

sys.stdout = TracePrints()
"""


float64 = np.zeros(0).dtype



mod_onetoinf1D = npc_helper.mod_onetoinf1D
mod_onetoinf2D = npc_helper.mod_onetoinf2D
def mod_onetoinf(p, mod_q):
	"""	Return np.mod(p, mod_q) when mod_q != 1
		For mod_q = 1, take mod_q -> inf, i.e. return p

		mod_q should be a 1D (num_q,) np.array
		p can be a 1D (num_q,) or 2D (#, num_q) np.array
		"""
	#return np.mod(p, mod_q) + np.equal(np.abs(mod_q), 1)*p*mod_q
	return np.mod(p, mod_q) + np.equal(mod_q, 1) * p


def check_invalid_charge(q, mod_q):
	"""	Checks that q is in the range 0 to mod_q-1
		(unless mod_q = 1) """
	return np.any(np.logical_and(mod_q > 1, np.logical_or(q < 0, q >= mod_q)))


array_equiv_mod_q = npc_helper.array_equiv_mod_q
#def array_equiv_mod_q(a1, a2, mod_q):
#	return npc_helper.array_equiv_mod_q(a1, a2, mod_q)

#Definition:
#bint array_equiv_mod_q(np.ndarray[c_int, ndim=1] a1, np.ndarray[c_int, ndim=1] a2, np.ndarray[c_int, ndim=1] mod_q)

################################################################################
################################################################################
class array:

	def __init__(self, rank, dtype=float64):
		if rank <= 0: raise ValueError, "d not positive: " + str(rank)
		# TODO call d rank
		self.rank = rank		# dimension (number of legs), in np.ndarray this is ndim
		self.shape = None		# np.array( (rank,), dtype = np.intp) total shape tuple or ndarray = (n_0, ..., n_{d-1})
		self.num_q = None		# np.int number of conserved charges (applies globally), shoulde be >= 0.
		self.mod_q = None		# np.array( (num_q, ) dtype = np.int) 1-array of mod-q
		self.charge = None		# np.array( (num_q,), dtype = np.int ) total charge of tensor, inward pointing convention
		self.q_ind = None		# [ np.ndarray( (num_block, 2 + num_q), dtype = np.int) ] List[d] of 2-arrays. Each array, qi_{m, n}, is such that m labels charges sectors, qi_{m, 0} is the starting index of mth charge sector, qi_{m, 1} is ending index, and q_{m, 2:} is charge of mth sector
		self.q_conj = None		# 1-array shaped (d,), which notes if the corresponding q_ind is charge-conjugated (values +-1)
		self.q_dat = None		# np.ndarray( (num_dat, d), dtype = np.uint) 2-array, of the q-index (refer to q_ind) associated with dat[]. Format: block#, leg; shape = (num_blocks, d)
		self.dat = []			# data that holds the tensors (list of d-tensors)
		self.dtype = np.dtype(dtype)
		self.sorted = False		# Is current .dat lex sorted? Responsibility of all funcs modifying q_dat to flag appropriately (but not necessarily to sort)
		self.labels = None #{ x:x for x in range(rank)} # Dict takes labels -> index numbers


	def check_sanity(self, suppress_warning=False):
		"""	Checks if the internal structure of the array is consistent.

			Raise an exception if anything is wrong, do nothing if everything is right.
		"""
	##	rank
		rank = self.rank
		if not (isinstance(rank, np.int) or isinstance(rank, np.long)): raise RuntimeError, "rank is not an int (type=" + str(type(rank)) + ")"
		if rank < 0: raise RuntimeError, "rank is not positive: rank = " + str(rank)
		if rank == 0: raise NotImplementedError, "rank is zero"
	##	shape
		shape = self.shape
		if not (isinstance(shape, np.ndarray)): raise RuntimeError, "shape not an array: " + str(shape)
		if len(shape) != rank: raise RuntimeError, "shape,rank mismatch: rank = " + str(rank) + ", shape = " + str(shape)
		if np.any(np.array(shape) < 0): raise RuntimeError, "shape has negative dimensions: " + str(shape)
		if np.any(np.array(shape) == 0):
			warningMsg = "npc: shape %s has a zero-length leg, this feature has not been fully tested, use at your own risk." % (shape,)
			if not suppress_warning: warnings.warn(warningMsg, RuntimeWarning)
	##	num_q
		num_q = self.num_q
		if not isinstance(num_q, int): raise RuntimeError, "num_q is not an int (type=" + str(type(num_q)) + ")"
		if num_q < 0: raise RuntimeError, "negative num_q: " + str(num_q)
	##	mod_q
		mod_q = self.mod_q
		if not isinstance(mod_q, np.ndarray): raise RuntimeError, "mod_q not a numpy.ndarray"
		if mod_q.shape != (num_q,): raise RuntimeError, "incorrect mod_q shape: " + str(mod_q.shape)
		if mod_q.dtype != int: raise RuntimeError, "mod_q.dtype not int (dtype=" + str(mod_q.dtype) + ")"
		if np.any(mod_q <= 0): raise RuntimeError, "mod_q has non-positive values: mod_q = %s" % (mod_q,)
	##	charge
		charge = self.charge
		if not isinstance(charge, np.ndarray): raise RuntimeError, "charge not a numpy.ndarray"
		if charge.shape != (num_q,): raise RuntimeError, "charge has incorrect shape:" + str(charge.shape)
		if charge.dtype != int: raise RuntimeError, "charge.dtype not int (dtype=" + str(charge.dtype) + ")"
		if check_invalid_charge(charge, mod_q):
			warningMsg = "some charges out or range: mod_q = %s, charge = %s" % (mod_q, charge)
			if not suppress_warning: warnings.warn(warningMsg, RuntimeWarning)
			raise RuntimeError, warningMsg
	##	q_ind & q_conj
		q_ind = self.q_ind
		q_conj = self.q_conj
		if not isinstance(q_ind, list): raise RuntimeError, "q_ind type not list: " + str(type(q_ind))
		if len(q_ind) != rank: raise RuntimeError, "q_ind has incorrect size: rank = " + str(rank) + ", q_ind = " + str(q_ind)
		if not isinstance(q_conj, np.ndarray): raise RuntimeError, "q_conj not an numpy.ndarray: " + str(type(q_conj))
		if q_conj.shape != (rank,): raise RuntimeError, "q_conj has incorrect size: (rank,) = " + str((rank,)) + ", q_conj.shape = " + str(q_conj.shape)
		if q_conj.dtype != int: raise RuntimeError, "q_conj not type int: " + str(q_conj.dtype)
		for l in range(rank):
			tbl = q_ind[l]
			if not isinstance(tbl, np.ndarray): raise RuntimeError, "q_ind["+str(l)+"] not a numpy.ndarray"
			if tbl.dtype != int: raise RuntimeError, "q_ind[%s].dtype not int (type %s)" % (l, type(tbl.dtype))
			if tbl.ndim != 2 or tbl.shape[1] != num_q + 2: raise RuntimeError, "q_ind["+str(l)+"] has incorrect shape: " + str(tbl)
		##	Check shape - q_ind compatibility
			if shape[l] > 0:
				if tbl.shape[0] == 0: raise RuntimeError, "shape[%s] is nonzero, but q_ind[%s] has zero rows" % (l, l)
				if tbl[0,0] != 0 or tbl[-1, 1] != shape[l]: raise RuntimeError, "q_ind["+str(l)+"] has inconsistent values: " + str(tbl)
			else:
				if tbl.shape[0] != 0: raise RuntimeError, "shape[%s] is zero, but q_ind[%s] has %s rows" % (l, l, tbl.shape[0])
		##	Check the entries of q_ind
			for r in range(tbl.shape[0] - 1):
				if tbl[r, 1] != tbl[r+1, 0]: raise RuntimeError, "q_ind[%s] has inconsistent values between rows %s and %s:\n%s" % (l, r, r+1, tbl)
			for r in range(tbl.shape[0]):
				if tbl[r, 0] >= tbl[r, 1]: raise RuntimeError, "q_ind[%s] has invalid start/stop indices: %s,%s " % (l, tbl[r, 0], tbl[r, 1])
				#for i in range(num_q):
				#	if check_invalid_charge(tbl[r, i+2], mod_q[i]): raise RuntimeError, "invalid charge at q_ind[%s][%s,%s], q_ind =\n%s" % (l, r, i, tbl)
			if q_conj[l] != 1 and q_conj[l] != -1: raise RuntimeError, "q_conj contains values that are neither 1 or -1: " + str(q_conj)
	##	q_dat & dat
		q_dat = self.q_dat
		dat = self.dat
		dtype = self.dtype
		if not isinstance(q_dat, np.ndarray): raise RuntimeError, "q_dat not a numpy.ndarray"
		if q_dat.dtype != np.uint: raise RuntimeError, "q_dat.dtype not np.uint (%s)" % q_dat.dtype
		if q_dat.ndim != 2: raise RuntimeError, "q_dat not a 2d array: q_dat.shape = " + str(q_dat.shape)
		if q_dat.shape[1] != rank: raise RuntimeError, "q_dat have " + str(q_dat.shape[1]) + " columns, where d = " + str(rank)
		if not isinstance(dat, list): raise RuntimeError, "dat type not a list: " + str(type(dat))
		if len(q_dat) != len(dat): raise RuntimeError, "q_dat and dat length mismatch: " + str(len(q_dat)) + " != " + str(len(dat))
		q_dat_total_q = npc_helper.q_sum( q_dat, q_ind, q_conj, 0, rank, mod_q )
		for i in range(len(dat)):
			q_row = q_dat[i]
			T_row = dat[i]
			if not isinstance(T_row, np.ndarray): raise RuntimeError, "dat["+str(i)+"] not a numpy.ndarray: " + str(type(T_row))
			if T_row.dtype != dtype:
				warningMsg = "dtype mismatch: dat[%s].dtype = %s, self.dtype = %s" % (i, T_row.dtype, dtype)
				raise RuntimeError, warningMsg
			if anynan(T_row):
				raise RuntimeError, "Nans in dat[%s]" % (i, )
				
			sh = []		# constructing the shape of T_row
			for l in range(rank):
				ql = q_row[l]
				if ql < 0 or ql >= len(q_ind[l]): raise RuntimeError, "q_dat["+str(i)+","+str(l)+"] not in range [0," + str(len(q_ind[l])) + ")."
				sh.append(q_ind[l][ql, 1] - q_ind[l][ql, 0])
			if np.any(q_dat_total_q[i] != charge): raise RuntimeError, "total charge mismatch: charge = %s, q_dat[%s] charge = %s" % (charge, i, q_dat_total_q[i])
			if T_row.shape != tuple(sh): raise RuntimeError, "dat["+str(i)+"].shape different from q_ind spectification: " + str(T_row.shape) + " != " + str(sh)
	##	sorted
		if len(q_dat) > 0:
			perm = np.lexsort(q_dat.transpose())
			for r in range(len(q_dat) - 1):
				# look for duplicate entries
				if np.array_equal(q_dat[perm[r]], q_dat[perm[r+1]]): raise RuntimeError, "q_dat rows "+str(perm[r])+" and "+str(perm[r+1])+" are identical.\n\t" + str(q_dat[perm[r]]) + "\n\t" + str(q_dat[perm[r+1]])
			if self.sorted == True and np.array_equal(perm, np.arange(len(perm))) == False:
				raise RuntimeError, "sorted = True, but q_dat is not sorted"
				

	def fix_types(self):
		"""Converts auxilliary data (shape, num_q, q_ind, mod_q, charge, q_conj, q_dat) to correct types.
	Modifies self (done in-place)
		"""
		self.shape = self.shape.astype(np.intp)
		self.q_dat = self.q_dat.astype(np.uint)
		self.q_ind = [qi.astype(np.int) for qi in self.q_ind]
		self.mod_q = self.mod_q.astype(np.int)
		self.charge = self.charge.astype(np.int)
		self.q_conj = self.q_conj.astype(np.int)
		self.num_q = int(self.num_q)



	
	################################################################################
	##	Making npc.array's / converting them to np.array's
	
	def zeros_like(self):
		return zeros(self.q_ind, self.dtype, self.q_conj, self.charge, self.mod_q)
		
	
	def empty_like(self, dup_q_dat=True, dup_q_ind=True):
		"""	Returns a copy of self, but does NOT initialize the data blocks.  Will not be a valid instance unless .dat is set elsewhere.
			Note: will NOT make a copy of q_ind (even with dup_q_ind is True), the new tensor will still reference the old one's q_ind.
			"""
		a = array(self.rank, self.dtype)
		a.shape = self.shape.copy()
		a.charge = self.charge.copy()
		a.mod_q = self.mod_q
		a.num_q = self.num_q
		if self.labels is not None:
			a.labels = self.labels.copy()
		
		if dup_q_ind:
			a.q_conj = self.q_conj.copy()
			a.q_ind = [ qi for qi in self.q_ind ]
		if dup_q_dat:
			a.q_dat = self.q_dat.copy()
			a.sorted = self.sorted
		a.dat = []
		
		return a


	def copy(self):
		"""	returns a deep copy
			Note: will NOT make a copy of q_ind.
		"""
		a = self.empty_like()
		a.dat = [ t.copy() for t in self.dat ]
		return a
	
	def astype(self, type):
		self.dtype = np.dtype(type)
		self.dat = [t.astype(type) for t in self.dat]
		return self
		
	def shallow_copy(self):
		"""	returns a shallow copy
		"""
		a = self.empty_like(dup_q_dat=False, dup_q_ind=False)
		a.q_conj = self.q_conj
		a.q_ind = self.q_ind
		a.q_dat = self.q_dat
		a.sorted = self.sorted
		a.dat = self.dat
		
		return a
	
	def save(self, name):
		"""
			Save the array to disk at location name = str
			
			Storage format:
				Prepends a cPickled dat-gutted version of the instance to sequence of np.save of the .dat
				
		"""
		with open(name, 'w') as f:
	
			dat = self.dat
			self.dat = len(dat) #dat-gutted
			
			cPickle.dump(self, f, -1)
			for t in dat:
				np.save(f, t)

			self.dat = dat

	@classmethod
	def from_npfunc(cls, func, q_ind, q_conj = None, charge = None, mod_q = None, dtype = None):
		"""	Generate npc.array using 'func' to initialize valid blocks. For each block of charge 'charge' (which will default to zero), calls 'func' to initialize the np array.
		
			Sample usage:
			
			To generate an array of ones wherever allowed by charge conservation:
				>>>	a = array.from_npfunc( np.ones, q_ind, dtype = np.float )
			
			To generate a random array of floats with total charge 3:	
				>>>	a = array.from_npfunc( np.random.standard_normal, q_ind, charge = [3], dtype = np.float )
		
			Details on the form of 'func':
			
			At a minimum, the func must accept a shape object, and return a corresponding array. To deal with the case in which func may have further options, we allow two cases: func is either 1) a functools.partial object, or 2) a function object. Note the conventions for handling dtype!
			
			Case 1) The shape of each sub block is prepended to func.args, and func.keywords is not modified. If dtype is None and func.keywords has dtype key, set dtype = keywords['dtype']; if dtype is None, but there is no key, set dtype = np.float
			Case 2) func.args is taken to be the shape of each sub block. If 'dtype'is not None, then 'dtype' is passed as a keyword to func. If 'dtype' is None, then set dtype = np.float, but do NOT pass as keyword to func.
			"""
		try: #functools.partial
			args = func.args
			if args is None:
				args = ()
				
			keywords = func.keywords
			if keywords is None:
				keywords = {}
			
			if keywords.has_key('dtype'):
				dtype = keywords['dtype']
			else:
				dtype = np.float
			func = func.func
		except: #function object
			args = ()
			keywords = {}
			if dtype is None:
				dtype = np.float
			else:
				keywords['dtype'] = dtype

		ac = zeros(q_ind, dtype = dtype, q_conj = q_conj, charge = charge, mod_q = mod_q)
		
		d = ac.rank
		num_q = ac.num_q
		q_dat = []
		charge = ac.charge
		q_conj = ac.q_conj
		mod_q = ac.mod_q
		
		#I invert the natural product order in accord with lexsort
		for inds in itertools.product( *[ xrange(len(q_ind[i]))  for i in reversed(xrange(d))]) :
			inds = inds[::-1]
			sector = [ q[j, :] for q, j in itertools.izip(q_ind, inds)]
			q_tot = np.empty( num_q, dtype = np.int)
			q_tot[:] = sector[0][2:]*q_conj[0]
			
			for i, s in enumerate(sector[1:]):
				q_tot += s[2:]*q_conj[i+1]
			
			if array_equiv_mod_q(q_tot, charge, mod_q):
				sh = tuple( [ s[1] - s[0]  for s in sector] )
				ac.dat.append( func( *( (sh,) + args), **keywords) )
				q_dat.append( np.array(inds, dtype = np.uint) )
				
		if len(q_dat) > 0:
			ac.q_dat = np.array(q_dat, dtype = np.uint)
		else:
			ac.q_dat = np.zeros((0,d), dtype = np.uint)
			
		ac.sorted = True
		return ac
	

	@classmethod
	def detect_ndarray_charge(cls, a, q_ind, q_conj = None, mod_q = None):
		"""	Returns the total charge of first non-zero sector found in a
		"""
		d = len(q_ind)
		num_q = q_ind[0].shape[1] - 2
		if mod_q is None:
			mod_q = np.ones(num_q, dtype = int)
		if q_conj is None:
			q_conj = np.ones(num_q, dtype = int)
			
		# TODO set q_conj
		q_tot = np.empty( num_q, dtype = np.int )
		for qs in itertools.product( *[ xrange(len(q_ind[i])) for i in reversed(xrange(d)) ] ):
			# loop over all charge sectors in lex order (last leg most significant)
			qs = qs[::-1]		# qs is now back in forward order
			sector = [ q[j, :] for q, j in itertools.izip(q_ind, qs) ]		# pick out rows of q_inds corresponding to qs
			
			sl = tuple( [ slice(s[0], s[1]) for s in sector] )		# slices corresponding to qs
			if np.any(a[sl] != 0):	# check if non-zero
				q_tot[:] = sector[0][2:] * q_conj[0]
				for i, s in enumerate(sector[1:]):
					q_tot += s[2:] * q_conj[i+1]
				return mod_onetoinf1D(q_tot, mod_q)

				
	@classmethod
	def detect_ndarray_charge_flat(cls, a, q_flat, q_conj = None, mod_q = None):
		"""	Returns the total charge of first non-zero sector found in a
			"""
		q_ind = [ q_ind_from_q_flat(q) for q in q_flat ]
		return cls.detect_ndarray_charge(a, q_ind, q_conj, mod_q)

#	@classmethod
#	def test_array(cls, q_ind, q_conj = None, charge = None, mod_q = None, rand_seed = 0):
#	
#		if rand_seed=='ones':
#			return cls.from_npfunc(np.ones, q_ind, q_conj, mod_q )
#		else:
#			random.seed(rand_seed)
#			np.random.seed(rand_seed)
#		
#			return cls.from_npfunc(np.random.random_sample, q_ind, q_conj, mod_q )
		
	@classmethod
	def from_ndarray(cls, a, q_ind, q_conj = None, charge = None, mod_q = None, cutoff = 0.):
		"""	Make an array from values taken from an ndarray.
			a is the array, q_ind are the charges on each leg.
			charge is an option to project onto certain total charge sectors; defaults to zero
			"""
	
		if q_ind is None:
			return array.from_ndarray_trivial(a, q_conj, name = name)
			
		ac = zeros(q_ind, dtype = a.dtype, q_conj = q_conj, charge = charge, mod_q = mod_q)

		d = ac.rank
		num_q = ac.num_q
		q_dat = []
		charge = ac.charge
		q_conj = ac.q_conj
		mod_q = ac.mod_q

		#reverse product order to keep q_dat sorted
		for inds in itertools.product( *[ xrange(len(q_ind[i])) for i in reversed(xrange(d)) ] ):
			inds = np.array(inds[::-1], dtype = np.uint)
			
			#q_tot = ac.block_charge(inds)
			q_tot = npc_helper.block_charge(ac, inds)
			if npc_helper.array_equiv_mod_q(q_tot, charge, mod_q):
				#sector = [ q[j, :] for q, j in itertools.izip(q_ind, inds) ]
				sl = tuple( [ slice(q[j, 0], q[j, 1]) for q, j in itertools.izip(q_ind, inds)] )

				if np.linalg.norm(a[sl]) > cutoff:		# check if non-zero
					ac.dat.append( a[sl].copy() )
					q_dat.append( inds )


		if len(q_dat) > 0:
			ac.q_dat = np.array(q_dat, dtype = np.uint)
		else:
			ac.q_dat = np.zeros((0,d), dtype = np.uint)

		ac.sorted = True
		
		return ac

#Sort / Dont Sort / Permute
#Bunch / Not bunch

	@classmethod
	def from_ndarray_flat(cls, a, q_flat, q_conj = None, charge = None, mod_q = None, sort = True, bunch = True, cutoff = 0.):
		"""	Contruct array with charge info given in 'flat' form.
		
			Two options can be passed to specify how q_ind is formed from q_flat.
			
			First, for each leg we may want to sort indices by charge ('S'), don't sort ('DS'), or permute according to some given permutation. Both a and q_flat are permuted accordingly. You can optionally pass 
			
				sort = ['S', 'DS', a_permutation, 'S', . . .]
				
				to specify what to do on each leg individually, or you can pass sort = True / False to S or DS on all legs. Note that the sorting order is lex-order of q_flat, NOT q_flat as conjugated by q_conj
			
			Second, we may want to form blocks of the largest possible contiguous charge (bunch = True), or make blocks of length 1 (bunch = False). Again this can be passed either as a list to specify legs seperately
			
				bunch = [True, False , . . .]
				
				or as a single boolean value.
			
			Returns perm, thearray
			
			perm - A list of permuations specifying the permutation applied to each leg
			thearray - the resulting npc.array
			"""
		d = len(q_flat)
		num_q = q_flat[0].shape[1]
		
		if sort == True:
			sort = ['S']*d
		elif sort == False:
			sort = ['DS']*d
			
		if bunch == True:
			bunch = [True]*d
		elif bunch == False:
			bunch = [False]*d
		
		if q_flat[0].shape[1] == 0: #num_q = 0, so better not sort
			for i in range(d):
				if sort[i] == 'S':
					sort[i] = 'DS'
		elif mod_q is not None:
			q_flat = [ mod_onetoinf2D(q, mod_q) for q in q_flat]
			
		perm = [None]*d
		q_ind = [None]*d
		for i in range(d):
			if sort[i] == 'S':
				perm[i] = np.lexsort(q_flat[i].T)
				q_flat[i] = q_flat[i][perm[i]]
				a = a.take(perm[i], axis = i)
			elif sort[i] == 'DS':
				perm[i] = np.arange(len(q_flat[i]))
			else:
				perm[i] = sort[i]
				q_flat[i] = q_flat[i][perm[i]]
				a = a.take(perm[i], axis = i)
			
			if bunch[i]==True:
				q_ind[i] = q_ind_from_q_flat(q_flat[i])
			else:
				l = q_flat[i].shape[0]
				q = np.empty((l,2+num_q), np.int)
				q[:, 0] = np.arange(l)
				q[:, 1] = np.arange(1,l+1)
				q[:, 2:] = q_flat[i]
				q_ind[i] = q
					
		return perm, array.from_ndarray(a, q_ind,  q_conj, charge, mod_q, cutoff)



	@classmethod
	def from_ndarray_trivial(cls, a, q_conj = None):
		"""	The point of this function is to wrap ndarrays for the case num_q = 0. """
		rank = a.ndim
		if rank == 0: raise ValueError, "len(a.shape) = 0"
		if np.any(a.shape == 0): raise ValueError, "a.shape has a 0: " + str(a.shape)
		ac = cls(rank, a.dtype)
		#ac.shape = a.shape #SHAPE
		ac.shape = np.array(a.shape, dtype = np.intp) #SHAPE
		ac.num_q = 0
		ac.charge = np.empty( (0,), np.int )
		ac.mod_q = np.empty( (0,), np.int )
		
		if q_conj is None:
			ac.q_conj = np.ones( (rank), np.int )
		else:
			ac.q_conj = q_conj.copy()	
		
		ac.sorted = True
		ac.q_dat = np.zeros( (1, rank), np.uint )
		ac.dat = [a.copy()]
		
		ac.q_ind = [ np.empty( (1, 2), np.int ) for i in range(rank) ]
		for i in range(rank):
			ac.q_ind[i][0, 0] = 0
			ac.q_ind[i][0, 1] = a.shape[i]
		
		return ac
		


	def to_ndarray(self, perm = None):
		"""	return an ndarray without the conserved charge baggage
			"""
		# TODO, should have an option to take the perm
		a = np.zeros(self.shape, self.dtype)

		qi = self.q_ind
		d = self.rank
		for qs,T in itertools.izip(self.q_dat, self.dat):
			s = [ slice( qi[l][ qs[l], 0], qi[l][ qs[l], 1] )  for l in xrange(d)] 
			a[s] = T
		return a


	def set_labels(self, labels):
		""" set labels = [l0, l1, ...]
			where each label is either valid str, or None
		"""
		self.labels = {}
		for x in xrange(self.rank):
			if isinstance(labels[x], str):
				self.labels[ labels[x] ] = x
			elif labels[x] is not None:
				raise ValueError, labels[x]

	def get_index(self, label):
		""" Given the label for leg, which can either be a # or a str, return the corresponding position of the lef. For #, the # is simply returned; for str, the labels dict is used.
		"""
		if isinstance(label, Number):
			return label
		elif isinstance(label, str):
			return self.labels[label]
		else:
			raise ValueError, type(label)

	def get_indices(self, labels):
		"""	Same as get_index, but takes in an iterable and returns an array."""
		return np.fromiter([self.get_index(m) for m in labels], dtype=np.uint)

	def get_labels_list(self):
		"""	Return a list of labels. """
		lb = [None] * self.rank
		if self.labels is None: return lb
		for k,v in self.labels.iteritems():
			lb[v] = k
		return lb
		

################################################################################
	##	Printing stuff & info

	def __repr__(self):
		return "<npc.array shape:{0:s}>".format(self.shape.tolist())
		
	def __str__(self):
		return self.to_ndarray().__str__()


	def print_sparse_stats(self):
		print joinstr(['\t', self.sparse_stats()])

	def stored(self):
		stored = 0
		for t in self.dat:
			stored += t.size
		return stored
		
	def sparse_stats(self):
		"""	returns a string detailing the sparse statistics of an npc.array """
		size = np.prod(self.shape)
		num_blocks = len(self.q_dat)
		
		nonzero = 0
		stored = 0
		s1 = 0.
		s2 = 0.
		s3 = 0.
		for t in self.dat:
			nonzero += np.count_nonzero(t)
			s = t.size
			stored += s
			s1 += s**(0.5)
			s2 += s**(1.)
			s3 += s**(1.5)
		
		if num_blocks > 0:
			s1 = (s1/num_blocks)
			s2 = (s2/num_blocks)**(1./2.)
			s3 = (s3/num_blocks)**(1./3.)
			captsparse = float(nonzero)/stored
		else:
			captsparse = 1.
			
		sparse_str = str(nonzero) + " of " + str(size) + " entries (" + str(float(nonzero)/size) + ") nonzero, " + \
			"stored in " + str(num_blocks) + " blocks for a total storage of " + str(stored) + " entries." + \
			"\nCaptured sparsity: " + str(captsparse) + \
			"\nEffective block size: " + str([s1, s2, s3]) 
		return sparse_str


	def q_ind_str(self, legs = None):
		"""	Returns a string of the q_ind's printed side-by-side.
			
			legs is a list which specifies the q_ind's to display,
			if legs is None, then all the q_ind's are shown.

			The string has no newline at the end.
			"""
		if legs is None:
			legs = range(self.rank)
		if type(legs) == int:
			legs = [legs]
		return joinstr([ " (q_conj = {0:+d})\n".format(self.q_conj[i]) + str(self.q_ind[i]) for i in legs ], delim=' ')
	

	def stride_order(self):
		"""	returns an array (len d) indicating the order of the tensor's strides
			[0 1 ... d-1] means it's stored in C-order (first leg most significant / largest stride)
			"""
		if len(self.dat) > 0:
			# sum of stides for each dat
			st = np.sum(np.array([ T.strides for T in self.dat ]), axis = 0)
			# st may have duplicate strides (if dim = 1 for a particular leg), so vstacking np.arange makes lexsort stable
			return np.lexsort( np.vstack((np.arange(self.rank, 0, -1), st)) ) [::-1]
		else:
			return None
		
	
	def print_q_dat(self, sort=False, print_norm=None, print_tensors=False, print_trace=False):
		"""	Print out each line of q_dat
			
			print_norm will also print the norms in addition, and it sets the ord of the norm.
			"""
		print "\tq_conj " + str([ self.q_conj[l] for l in range(self.rank) ]) + ", mod_q " + str(self.mod_q),
		print "\tfull shape:", self.shape
		if len(self.q_dat) == 0: return
		if print_norm == True: print_norm = 2
		if sort and self.sorted == False:
			# print sorted, does not alter the original
			perm = np.lexsort(self.q_dat.reshape((-1,self.rank)).transpose())
			q_dat = self.q_dat[perm, :]
			dat = [ self.dat[p] for p in perm ]
		else:
			q_dat = self.q_dat
			dat = self.dat
		
		#calculate widths of columns in order to delimit
		line = ['qind', 'charge', 'shape']
		if print_norm is not None:
			line.append('norm')
		if print_trace:
			line.append('trace')
		
		table = [line]
		
		max_widths = map(len, line)
		
		for i in range(len(q_dat)):
			line = [ str(q_dat[i]), str([ self.q_ind[l][q_dat[i,l],2:].tolist() for l in range(self.rank)]), str(dat[i].shape)]
			if print_norm is not None:
				line.append(str(np.linalg.norm(1. * dat[i].reshape((-1)), ord=print_norm)))
			if print_trace:
				line.append(str(np.trace(dat[i])))
			table.append(line)
			max_widths = map(max, map(len, line), max_widths)
		
		max_widths = [w + 4 for w in max_widths]
		
	
		table = [ joinstr(map(str.ljust, line, max_widths)) for line in table ]
		print joinstr(table[0])
		for i in range(len(q_dat)):
			print joinstr(table[i+1])
			if print_tensors:
				print joinstr(["    ", str(dat[i])])
		
		if print_norm is not None:
			print "\ttotal norm:", self.norm(ord=print_norm)
	

	def is_blocked_by_charge(self):
		"""	Return a boolean array shaped (ndim,) of whether each leg is blocked by charge """
 		return np.array([ npc_helper.is_blocked_by_charge(ind) for ind in self.q_ind ])


	def is_completely_blocked_by_charge(self):
		"""	Return a boolean of whether *all* legs are blocked by charge """
 		return np.all(np.array([ npc_helper.is_blocked_by_charge(ind) for ind in self.q_ind ]))

	
	################################################################################
	##	Accessing blocks and indices
	def get_q_index(self, leg, i):
		"""	Given an index on a leg, find which q-index it belongs in.
				Return (qi, subindex)
					where qi is the q-index, subindex is the where i sits inside that q-block.
				
				Example:
					with a.q_ind[4] = [[0 3 0], [3 7 1]]
					>>>	a.get_q_index(4, 5)
						(1, 2)
					which means "5" is in the second row of q_ind: [3 7 1], and it's the second element of that block (5-3=2)
			"""
		# TODO, cython this
		q_ind = self.q_ind[leg]
		if i < 0 or i >= self.shape[leg]: raise ValueError, "Out of range: shape = %s, leg %s, i = %s" % (self.shape, leg, i)
		for qi in range(len(q_ind)):
			if i < q_ind[qi, 1]: break
		return qi, i - q_ind[qi, 0]

	def index_charge(self, leg, i):
		"""Returns the charge of leg, index i"""
		qi, subindex = self.get_q_index(leg, i)
		return self.q_ind[leg][qi, 2:]*self.q_conj[leg]

	def block_charge(self, ind):
		"""	Returns charge of a block (as ndarray)
		
		"""
		charge = self.q_ind[0][ind[0], 2:].copy() * self.q_conj[0]
		for i in xrange(1, self.rank):
			charge += self.q_ind[i][ind[i], 2:] * self.q_conj[i]
	
		#charge = npc.
		return charge		## mod q
	

	def block_shape(self, ind):
		"""	Returns shape of a block (as tuple) given the q-indices """
		return tuple([ self.q_ind[i][ind[i], 1] - self.q_ind[i][ind[i], 0] for i in xrange(self.rank) ])
		

	def block_i(self, qi):
		"""	Returns location of a block in q_dat, or -1 if not present.
			qi is an np.array shaped (rank,) of q-indices.
		"""
		## Slow code in python
		#for j in xrange(len(self.q_dat)): #See if entry is there
		#	if np.array_equiv(self.q_dat[j], qi):
		#		return j
		#return -1
		return npc_helper.find_row_uint(self.q_dat, qi)


	def block_dat(self, qi):
		"""	Returns data of a block, or None if not present.
			qi is an np.array shaped (rank,) of q-indices.
			"""
		r = npc_helper.find_row_uint(self.q_dat, qi)
		if r < 0:
			return None
		else:
			return self.dat[r]
		

	def get_block(self, ind, write = True):
		"""	Returns a the block labeled by ind (an np.array shape (rank,) of q-indices) 
				
				If write = True, returns a view.
					If the block does not exist, a zeroed array is added (IF it respects charge cons) and returned.
				If write = False, returns a copy.
					If block does not exist, a zeroed array is returned without adding it. 
			"""
		
		ind = np.array(ind, dtype = np.uint)
		if len(ind) != self.rank: raise ValueError
		
		r = npc_helper.find_row_uint(self.q_dat, ind)
		
		#if dat is None: #Not there - make zeros
		if r < 0: #Not there - make zeros
			shape = self.block_shape(ind)
			z = np.zeros(shape, dtype = self.dtype)
			if write and npc_helper.array_equiv_mod_q(self.block_charge(ind), self.charge, self.mod_q):
				self.dat.append(z)
				self.q_dat = np.vstack( [self.q_dat, ind])
				self.sorted = False
			return z
			
		else:
			dat = self.dat[r]
			if write:
				return dat
			else:
				return dat.copy()

	def __getitem__(self, inds):
		""" a[i1, i2, . . . ] returns value of corresponding entry """
		
		inds = np.array(inds )
		pos = np.array([ self.get_q_index(i, inds[i]) for i in range(self.rank)], dtype = np.uint)
		b = self.block_dat(pos[:, 0])
		if b is None:
			return 0.
		else:
			return b[tuple(pos[:, 1])]
			
	def __setitem__(self, inds, val):
		""" a[i1, i2, . . . ] = val sets value of corresponding entry """
		
		inds = np.array(inds )
		pos = np.array([ self.get_q_index(i, inds[i]) for i in range(self.rank)], dtype = np.uint)
		b = self.get_block(pos[:, 0], write = True)
		b[tuple(pos[:, 1])] = val

	def ipurge_zeros(self, cutoff = 0.):
		"""	Removes dat blocks of norm <= cutoff. (Done in place) """
		## TODO, move this to different section
		len0 = self.q_dat.shape[0]
		if len0 > 0:
			norms = np.array([np.linalg.norm(t) for t in self.dat])
			keep = (norms > cutoff) #True where we will keep them
			self.q_dat = self.q_dat[keep]
			keep = np.nonzero(keep)[0] #locations of non-zero
			self.dat = [self.dat[i] for i in keep]
		return self

#	def take_blocks(self, qi, axis=0):
#		"""
#			given qi is an q-index (or a list of), return an npc.array (same rank)
#			"""
#		# TODO
#		pass


	def take_slice(self, i, axes=0):
		"""
			given i an index on leg axes, return an npc.array with one less dimension
			"""
		if not isinstance(axes, int): raise ValueError
		qind = self.q_ind[axes]
		for r in len(qind):
			if i < qind[r, 1]:
				qi = r
				break
		ri = i - qind[qi, 0]		# the relative i within the qi block
		not_axes = [ a for a in range(self.rank) if a != axis ]
		slcharge = mod_onetoinf1D(self.charge - qind[qi, 2:] * self.q_conj[axis], self.mod_q)
		sl = zeros( [ self.q_ind[a] for a in not_axes ], self.q_conj[not_axes], sl_charge, self.mod_q )
		q_dat = self.q_dat
		sl_q_dat = []
		for r in range(len(q_dat)):
			if q_dat[r, axes] == qi:
				sl_q_dat.append(q_dat[r][not_axes])
				#sl.dat.append(self.dat[r])
				## TODO, write code to take shit
		sl.q_dat = np.array(sl_q_dat, np.uint)
		#sl.check_sanity()
		return sl



	################################################################################
	##	Manipulating arrays -- playing with axes

	#@profile
	def itranspose(self, axes = None):
		"""	IN PLACE. self = np.transpose(self), np-style tranposition. 
			axes = (i0, i1, . . . ) is a permutation of the axes; by default reverses order
				axes[j]=i means a's i-th axis becomes a.transpose()'s j-th axis.
			Transposes data blocks and charge lists, but does not sort them according to new lex order """
		if axes is None:
			axes = np.arange(self.rank - 1, -1, -1, dtype = np.intp)
		elif self.labels is not None:
			axes = np.fromiter([self.get_index(m) for m in axes], dtype=np.intp)
		else:
			axes = np.fromiter(axes, dtype=np.intp)
			
		npc_helper.itranspose_fast(self, axes)
		return self

		
		"""
		tran = np.transpose
		self.dat = [ tran(m, axes) for m in self.dat]
		self.shape = np.take(self.shape, axes) #SHAPE
		self.q_conj = self.q_conj[axes]	## TODO, this does not work if axes is a tuple
		self.q_ind = [ self.q_ind[i] for i in axes]
		self.q_dat = np.take(self.q_dat, axes, axis = 1)
		self.sorted = False
		if not hasattr(self, 'labels'): self.labels = None #REMOVE ME at some point
		if self.labels is not None:
			perm = np.argsort(axes)
			self.labels = { k : perm[ self.labels[k] ] for k in self.labels.keys() }

		return self
		"""
		
		

	def transpose(self, axes = None):
		"""	COPY. c = np.transpose(self), np-style tranposition.
			axes = (i0, i1, . . . ) is a permutation of the axes; by default reverses order
				axes[j]=i means a's i-th axis becomes a.transpose()'s j-th axis.
			Transposes data blocks and charge lists, but does not sort them according to new lex order """
		
		#TODO: don't copy, then transpose.
		if axes is None:
			axes = np.arange(self.rank - 1, -1, -1, dtype = np.intp)
		elif self.labels is not None:
			axes = np.fromiter([self.get_index(m) for m in axes], dtype=np.intp)
		else:
			axes = np.fromiter(axes, dtype=np.intp)

		return npc_helper.transpose_fast(self, axes)
		"""
		a = array(self.rank, dtype = self.dtype)
		a.shape = self.shape[axes] #SHAPE
		a.mod_q = self.mod_q
		a.num_q = self.num_q
		a.q_conj = self.q_conj[axes]
		a.q_ind = [ self.q_ind[i] for i in axes] #DOESN'T copy
		a.charge = self.charge.copy()
		self.sorted = False
		
		a.dat = [ np.transpose(m, axes).copy(order = 'C') for m in self.dat]
		a.q_dat = np.take(self.q_dat, axes, axis = 1)
		
		if self.labels is not None:
			perm = np.argsort(axes)
			a.labels = { k : perm[ self.labels[k] ] for k in self.labels.keys()}
		return a
		"""
		

	

	def make_array_contiguous(self):
		self.dat = [np.ascontiguousarray(t) for t in self.dat]


	def iexpand_dims(self, axis = 0):
		"""	IN PLACE. Equivalent to np.expand_dims. The new leg is assumed to have trivial charge.
		"""
	
		self.dat = [np.expand_dims(t, axis) for t in self.dat]
		self.shape = np.insert(self.shape, axis, 1)
		self.q_conj = np.insert(self.q_conj, axis, 1)
		self.rank+=1

		#make a trivial q_ind
		q_ind = np.empty((1, self.num_q + 2), dtype = np.int)
		q_ind[0, 0] = 0
		q_ind[0, 1] = 1
		q_ind[0, 2:] = 0
		self.q_ind.insert(axis, q_ind)
	
		#insert column of 1s
		self.q_dat = np.insert( self.q_dat, axis, np.zeros(self.q_dat.shape[0], dtype = np.uint),  axis = 1 )
		
		return self
	
	def isqueeze(self, axis = None):
		"""	IN PLACE. Equivalent to np.squeeze. If squeezed leg has non-zero charge, this charge is added to self.charge
		
			axis is either a list, or a single index. 
			
			If axis is None, squeezes all length 1 indices
		"""
		
		if axis is None:
			axis = tuple([a for a in range(self.rank) if self.shape[a] == 1])
		else:
			axis = tuple(toiterable(axis))
		
		keep = range(self.rank)

		for a in axis:
			if self.shape[a]!=1:
				raise ValueError, "Tried to squeeze non-unit leg"
			
			self.charge -= self.q_ind[a][0, 2:]*self.q_conj[a]
			keep.remove(a)
		
							
		if self.labels is not None:
			new_pos = { keep[i]:i for i in range(len(keep))}
			self.labels = {k:new_pos[v] for k, v in self.labels.iteritems() if v in keep}

		self.charge = mod_onetoinf1D(self.charge, self.mod_q)
		keep = np.array(keep)
		self.dat = [ np.squeeze(t, axis=axis) for t in self.dat]
		
		self.shape = self.shape[keep]#SHAPE
		
		self.q_conj = self.q_conj[keep]
		self.rank = len(keep)
		self.q_ind = [ self.q_ind[a] for a in keep]
		self.q_dat = np.take(self.q_dat, keep, axis = 1)
		
		return self
			


			
	"""	----- pipes for "identical" or "conjugate" legs -----
		This is relevant both to combine_legs and split_legs
		
	 	Suppose we have tensors A, B naturally linked  via
		
		A_{... i k ...} B_{ ... i k ...}
		
		so that the charge data on the contracted legs is conjugate between A and B (ie, equal and opposite)
		
		If we want to combine legs i k ----> j, it would be nice to make  a single leg pipe made from the data of one of A or B, but be able to use it for either. The code does just that. 
		
		RULE: A pipe made from the q_ind, q_conj data from one set of legs will  work for a second set of legs whose charge data is either identical or conjugate.
		
			However! - they must be identical/conjugate in the sense that they have IDENTICAL q_ind, and identical/opposite q_conj (even though physically some combination would be sufficient).
		
		See "make_pipe" for some implementation issues.
		"""

	def combine_legs(self, axes, pipes = None, qt_conj = None, block_single_legs = False, timing = False):
		"""	Combine legs together. 
			
			Assumes axes = [ [0, 1, . . ., j_1 ], [i_2, . . . , j_2] , [. . . , d]] , a python list of lists enumerating indices to be grouped, 	
				So T_ijkl ---> T_i(jk)l would have axes =  [[0],[1, 2], [3]]
				The length of axes is the rank of the combined tensor.
				
			If pipes = None, combine_legs will make the pipes itself (and if you haven't stored them elsewhere, you won't be able to split back.)
			Otherwise, pipes = [ pipe0, None, ... ]; combine_legs will calculate pipes for entries None, and otherwise use pipe_i. Note: must be an entry (either None or pipe_i) for EACH combined leg, so len(pipes) = rank of the combined tensor.
			
			Optionally, one may provide qt_conj = [1, -1, . . .]. The length of qt_conj is the rank of the combined tensor. However, the entries are ignored except where pipes[j] is None. If pipes[j] = None, then when constructing the pipe for the new leg it is constructed such that qt_conj[j]  = 1 / -1 on the new leg. If pipes[j] is not None, qt_conj is constructed automatically to be compatible with charge structure of the provided pipe.
			
			
			-Labelling: Concatenates leg labels using a '.' delimiter.
		
				Example: Let 
					a.labels = ['a', 'b', 'c'] -> [0, 1, 2]
				Then if 
					b = a.combine_legs([[0, 1], [2] ]) 
				gives
					b.labels = [ 'a.b', 'c'] -> [0, 1]
							
			If some sub-leg of a combined leg isn't named, then a '?x' label is inserted: 'a.?0.c' etc.
			The x value of '?x' is incremented as needed
			
			
			
			TODO - would be nice to allow [ [2], [0, 3], [1]], with combine legs carrying out the required transposition first. Tricky what convention to use if leg pipes are involved though.
			
			Example:
			>>>	pipeL = oldarray.make_pipe([0, 1])
			>>>	pipeR = oldarray.make_pipe([3, 4], qt_conj = -1)
			>>>	newarray = oldarray.combine_legs([[0, 1], [2], [3, 4]], pipes = [pipeL, None, pipeR])
			"""
		
		return npc_helper.combine_legs(self, axes, pipes, qt_conj, block_single_legs, inplace = False)

	def icombine_legs(self, axes, pipes = None, qt_conj = None, block_single_legs = False, timing = False):
		"""	Combine legs together. IN PLACE. See combine_legs.
		"""
		
		return npc_helper.combine_legs(self, axes, pipes, qt_conj, block_single_legs, inplace = True)
		
	def split_legs(self, legs, leg_pipes, verbose = 0, timing = False):
		"""	legs is number or a list of
			
			Example:
			>>>	A.split_legs( [1], [pipe] )	# split off leg 1
			"""
		# a bunch of pre-processing and boring checks
		if legs is None: return self.copy()
		legs = tonparray(legs)
		if len(legs) == 0: return self.copy()
		leg_pipes = toiterable(leg_pipes)
		if len(legs) != len(leg_pipes): raise ValueError
		nsplit = len(legs)
		legs = np.array(legs) % self.rank
		for l in range(nsplit):
			if self.shape[legs[l]] != (leg_pipes[l].t_shape,): raise ValueError
		# make sures that the leg are sorted in increasing order
		perm = np.argsort(legs)
		legs = np.array(legs)[perm]
		leg_pipes = [ leg_pipes[i] for i in perm ]
		if verbose > 0:
			print "split_legs, legs: ", legs
			if verbose > 1:
				for i in range(nsplit):
					print "leg pipe", i
					print leg_pipes[i].qt_map, "= qt_map"
					print leg_pipes[i].qt_map_ind, "= qt_map_ind"

		# create u_map and s_map
		u_map = []		# list of legs that are unsplit [(src slice, dest slice), ...]
		s_map = []		# list of legs that are split [dest slice, ...]
		last_unspleg = 0		# the previous unsplit leg in old shape
		last_array_ptr = 0	# in the new shape
		for i in range(nsplit):		# scan through sections that will be split
			cur_array_ptr = last_array_ptr + legs[i] - last_unspleg
			if legs[i] > last_unspleg:		# save the section before the split (in to u_map)
				u_map.append( (slice(last_unspleg, legs[i]), slice(last_array_ptr, cur_array_ptr)) )
			# save the current section (in to s_map)
			s_map.append( slice(cur_array_ptr, cur_array_ptr + leg_pipes[i].nlegs) )
			cur_array_ptr += leg_pipes[i].nlegs
			last_unspleg = legs[i] + 1
			last_array_ptr = cur_array_ptr
		if last_unspleg < self.rank:		# take care of the trailing section (in to u_map)
			u_map.append( (slice(last_unspleg, self.rank), slice(last_array_ptr, None)) )
		
		a = self.empty_like(dup_q_dat=False, dup_q_ind=False)
		a.rank = self.rank + last_array_ptr - last_unspleg

		# fill in the easy stuff
		newshape = [0]*a.rank
		a.q_ind = [None]*a.rank
		a.q_conj = np.zeros((a.rank), int)
		for m in u_map:
			newshape[m[1]] = self.shape[m[0]]
			a.q_ind[m[1]] = self.q_ind[m[0]]
			a.q_conj[m[1]] = self.q_conj[m[0]]
		for l in range(nsplit):
			newshape[s_map[l]] = leg_pipes[l].shape
			a.q_ind[s_map[l]] = leg_pipes[l].q_ind
			a.q_conj[s_map[l]] = leg_pipes[l].q_conj * leg_pipes[l].qt_conj * self.q_conj[legs[l]]
		a.shape = np.array(newshape, dtype = np.intp) #SHAPE
		
		#print u_map, s_map, self.shape, a.shape
		#Split labels
		if self.labels is not None:
			rev = {v:k for k, v in self.labels.iteritems()}
			a.labels = {}
			for s, d in u_map: #Un split segments: slices src->dest
				bump = d.start - s.start #How far indices are shifted
				for v in xrange(s.start, s.stop):
						if v in rev:# and rev[v][0]!='?': #If it has label, add and translate index
							a.labels[rev[v]] = v + bump

			for l in range(nsplit): #Split segments
				if legs[l] in rev: #Was it labeled?
					bump = s_map[l].start
					keys = rev[legs[l]].rsplit('.', 1) #Split into sub-labels
					for i in xrange(len(keys)):
						if keys[i][0]!='?':
							a.labels[keys[i]] = bump + i
						
		if verbose > 0:
			print "u", u_map, "\ns", s_map
			print "old shape:", self.shape, " -->  new shape:", a.shape
			print "old q.conj:", self.q_conj, " -->  new q.conj:", a.q_conj

		q_dat = []
		a.dat = []
		row = np.zeros(a.rank, np.uint)
		newshape = [0] * a.rank
		Tslice = [slice(None)] * self.rank
		for r in xrange(len(self.q_dat)):		# scan q_dat's rows
			oldrow = self.q_dat[r]
			oldT = self.dat[r]
			for m in u_map:		# copy the unsplit parts
				row[m[1]] = oldrow[m[0]]
				newshape[m[1]] = oldT.shape[m[0]]
			qt_map_startstop = [ leg_pipes[l].qt_map_ind[oldrow[legs[l]], :] for l in range(nsplit) ]		# list of pair start/stop (in qt_map)
			entry_xrange = [ xrange(s[0], s[1]) for s in qt_map_startstop ]		# list of ranges to iterate over (in qt_map)
			if verbose > 0:
				print "\told q_dat row:", oldrow, oldT.shape
				print "\t", qt_map_startstop, entry_xrange
			for qs in itertools.product(*entry_xrange):		# scan every charge sub sector that is contained in oldrow
				for l in range(nsplit):
					pipe = leg_pipes[l]
					qt_map_row = pipe.qt_map[ qs[l], : ]		# new q-sub-indices
					row[s_map[l]] = qt_map_row[2:-1]			# write in the new q_dat row
					newshape[s_map[l]] = [ pipe.q_ind[i][qt_map_row[2+i],1] - pipe.q_ind[i][qt_map_row[2+i],0] for i in range(pipe.nlegs) ]
					Tslice[legs[l]] = slice(qt_map_row[0], qt_map_row[1])
					#print qs, l, qt_map_row
				if verbose > 0: print row, newshape, Tslice
				q_dat.append(row.copy())
				a.dat.append(oldT[Tslice].reshape(newshape))
		
		a.q_dat = np.array(q_dat, dtype = np.uint)
		a.sorted = False
		return a
				
			

	def make_pipe(self, legs, qt_conj = 1, block_single_legs = True, verbose = 0):
		"""
			legs is a list of integers
		"""
		legs = toiterable(legs)
		
		
		return leg_pipe.make_pipe([self.q_ind[l] for l in legs], self.q_conj[legs], qt_conj=qt_conj, block_single_legs=block_single_legs, mod_q=self.mod_q, verbose=verbose)




	################################################################################
	##	Manipulating arrays -- playing with charges

	def imap_Q(self, func, r):
		"""	Shift charges by 'r' using function 'func'.
			
			func operates on np.arrays with shape (num_q,) or (#, num_q)
			"""
		if func is None: return self
					
		self.q_ind = [ q.copy() for q in self.q_ind ]
		for l in range(self.rank):
			self.q_ind[l][:, 2:] = func(self.q_ind[l][:, 2:], r)
		self.charge = func(self.charge, r)
					
		return self
		
		
	def shallow_map_Q(self, func, r):
		"""	Shift charges by 'r' using function 'func', return a shallow copy of self. 
		
			Func is assumed to have the following properties:
				1. func operates on np.arrays of shape (num_q,) or (#, num_q).
				2. 'func' MUST return copy: should not operate in place
		
			The shallow_map has the following guaranteed properties:
				1. Does NOT modify self.
				2. a.dat points to self.dat (so changes to the matrices will register).
				3. a.q_ind may or may not point to self.q_ind, depending on whether func is trivial or not.
				
				WTF does 3. mean???
								
		"""
		if func is None: return self
			
		a = self.empty_like(dup_q_dat = False, dup_q_ind=False)
		a.q_conj = self.q_conj
		a.q_dat = self.q_dat
		a.dat = self.dat
		a.sorted = self.sorted

		a.q_ind = [ q.copy() for q in self.q_ind ]
		for l in range(a.rank):
			a.q_ind[l][:, 2:] = func(a.q_ind[l][:, 2:], r)
		a.charge = func(a.charge, r) 	# charge is already copied in empty_like

		return a
		

	def add_Q(self):
		# TODO
		pass


	def remove_Q(self, Q, fracture=True):
		#TODO comment this
		"""
			Q is a number (0 to num_q - 1) indicating charge to be removed,
				or a list of such numbers
			"""
		Q = tonparray(Q)
		if np.any(Q < 0) or np.any(Q >= self.num_q): raise ValueError, "Q not between 0 and "+str(self.num_q)+": " + str(Q)
		notQ = [ q for q in range(self.num_q) if q not in Q.tolist() ]
		notQplus2 = [0, 1] + [ q+2 for q in notQ ]
		
		a = self.copy()
		a.num_q = len(notQ)
		a.charge = a.charge[notQ]
		a.mod_q = a.mod_q[notQ]
		a.q_ind = [ ind[:, notQplus2].copy() for ind in a.q_ind ]

		if not fracture:
			a = a.poormans_unfracture()
			#raise NotImplementedError, "Just wait for it."

		#print joinstr(map(str,self.q_ind) + ["\t\t"] + map(str, a.q_ind))
		#a.check_sanity()
		return a

	def permute_Q(self, Qorder):
		"""	Qorder has the same structure used in numpy.transpose """
		Qorder = np.array(Qorder, int)
		if Qorder.ndim > 1: raise ValueError
		if len(Qorder) != self.num_q: raise ValueError
		if np.any(np.argsort(np.argsort(Qorder)) != Qorder): raise ValueError, (Qorder, np.argsort(Qorder), np.argsort(np.argsort(Qorder)))
		a = self.copy()
		a.charge = a.charge[Qorder]
		a.mod_q = a.mod_q[Qorder]
		Qind_order = np.array([0, 1] + [q+2 for q in Qorder])		# keep the first two, and swap the last num_q charges
		a.q_ind = [ ind[:, Qind_order].copy() for ind in a.q_ind ]
		#a.check_sanity()
		return a


	def poormans_unfracture(self, legs=None):
		"""	For each leg, group contiguous blocks of charges together.  This is done the dumb way.
				For example, if q_ind = [[0, 7, 1], [7, 43, 0], [43 60 0], [60, 80, 1]],
					then after running this code you should get q_ind = [[0, 7, 1], [7, 60 0], [60, 80, 1]].
			"""
		if legs is None:
			legs = np.arange(self.rank)
		legs = toiterable(legs)
		a = self
		for l in legs:
			q_flat = q_flat_from_q_ind(self.q_ind[l])
			new_q_ind = q_ind_from_q_flat(q_flat)
			#print joinstr([l, self.q_ind[l], '-->', new_q_ind, self.shape[l]], delim=' ')
			I = array.from_ndarray(np.eye(self.shape[l]), [self.q_ind[l], new_q_ind], q_conj=np.array([-1,1])*self.q_conj[l], charge=None, mod_q=self.mod_q)
			tensordot_compat(a, I, axes=([l],[0]))
			a = npc_helper.tensordot(a, I, axes=([l],[0])).itranspose(range(l) + [self.rank-1] + range(l,self.rank-1))
		#a.check_sanity()
		return a

	def unfracture(self, legs=None):
		raise NotImplementedError, "Use poormans_unfracture in the mean time."

		
	################################################################################
	##	Manipulating arrays -- data changes

	def iunary_blockwise(self, func):
		"""	IN PLACE. self = f(self). Applies a unary function 'block - wise' to sub blocks.
			func is a function (or partial function) which take an array as the first argument and returns an array of the same kind. Assumes func(0) = 0
			
			Example:
				a.iunaray_blockwise(np.real)
				
				will take the real part the tensor.
			"""
		try:	#assume a functools.partial (or similar)	
			args = func.args
			if args is None:
				args = ()
				
			keywords = func.keywords
			if keywords is None:
				keywords = {}
				
			func = func.func
		except:
			args = ()
			keywords = {}
 		
		self.dat = [func(*( (t,) + args ), **keywords ) for t in self.dat]
		if len(self.dat) > 0:
			self.dtype = self.dat[0].dtype
	
		return self
		

	def unary_blockwise(self, func):
		"""	COPY. c =  f(self).
			Applies a function 'block - wise' to sub blocks.
			func is a function (or partial function) which take an array as the first argument and returns an array of the same kind. Assumes func(0) = 0 
			
			Example:
			>>>	b = a.unaray_blockwise(np.abs)
			"""
		try:			
			args = func.args
			if args is None:
				args = ()
				
			keywords = func.keywords
			if keywords is None:
				keywords = {}
				
			func = func.func
		except:
			args = ()
			keywords = {}
			
 		a = self.empty_like()
		a.dat = [ func(*( (t,) + args ), **keywords ) for t in self.dat]
		if len(a.dat) > 0:
			a.dtype = a.dat[0].dtype
		
		return a
		

	def ibinary_blockwise(self, func, b):
		"""	IN PLACE. self = f(self, b), assuming self, b, are of same shape and charge, applies a binary function 'block - wise' to sub blocks, storing result in place.
		
			func is a function (or partial function) which takes two arrays as the first argument and returns an array of the same kind. Assumes func(0, 0) = 0
			
			I REPEAT: Assumes func(0, 0) = 0
			
			Example:
			>>>	a.ibinary_blockwise(np.max, b)	
				# will overwrite entries of a with the max of those in (a, b). 
		
		"""
			
		self.sort_q_dat()
		b.sort_q_dat()
		
		try: #extract the args and keywords			
			args = func.args
			if args is None:
				args = ()
					
			keywords = func.keywords
			if keywords is None:
				keywords = {}
			
			func = func.func
		except:
			args = ()
			keywords = {}
		
		adat = self.dat
		bdat = b.dat
		aq = self.q_dat
		bq = b.q_dat
		Na = len(aq)
		Nb = len(bq)

			
		#If the q_dat structure is identical, we can immediately run through the data.
		if Na == Nb and np.array_equiv(aq, bq):
			self.dat = [ func(*( (at, bt) + args ), **keywords ) for at, bt in itertools.izip(adat, bdat)]
			
		else: #Damn, have to step through comparing left and right

			i, j = 0, 0
			q_dat = []
			dat = []
			while i < Na or j < Nb:
				if j >= Nb or ( (i < Na) and (tuple(aq[i, ::-1]) < tuple(bq[j, ::-1]) ) ): #b is 0
					dat.append(func(*( (adat[i], np.zeros_like(adat[i])) + args ), **keywords ) )
					q_dat.append(aq[i])
					i+=1
				elif i >= Na or tuple(aq[i, ::-1]) > tuple(bq[j, ::-1]): #a is 0
					dat.append(func(*( (np.zeros_like(bdat[j]), bdat[j]) + args ), **keywords ) )
					q_dat.append(bq[j])
					j+=1
				else: #both are non zero
					dat.append(func(*( (adat[i], bdat[j]) + args ), **keywords ) )
					q_dat.append(aq[i])
					i+=1
					j+=1
				#if both are zero, we assume f(0, 0) = 0
				
			self.dat = dat
			self.q_dat = np.array(q_dat, np.uint)

		if len(self.dat) > 0:
			self.dtype = self.dat[0].dtype
		
		return self
	
	def iscale(self, alpha):
		""" Scales the tensor in place by real double 'alpha':
		
				self -> alpha self
				
			Calls BLAS directly when self.dtype is double, cdouble
		"""
	
		npc_helper.iscal(alpha, a)
				
	def iscale_axis(self, s, axis = -1):
		"""	IN PLACE. Scale an axis according a vector of values s.
			a_{i1, ..., i_axis, ...} = s_{i_axis} a_{i1, ..., i_axis, ...}
			"""
	
		if axis == self.rank - 1:
			axis = -1
			
		if s.ndim!=1 or s.shape[0] != self.shape[axis]:
			print "s != dimension"
			print "Axis:", axis
			print "Shape", self.shape
			print "s.shape", s.shape
			raise ValueError
		
		self.dtype = np.find_common_type([self.dtype], [s.dtype])
		
		q_i = self.q_ind[axis]

		if axis != -1: 
			self.dat = [ np.swapaxes(np.swapaxes(t, axis, -1)*s[ q_i[q, 0]:q_i[q, 1] ], axis, -1) for q, t in itertools.izip(self.q_dat[:, axis], self.dat ) ]
		else: #not sure if this is faster
			
			self.dat = [ t*s[ q_i[q, 0]:q_i[q, 1] ] for q, t in itertools.izip(self.q_dat[:, axis], self.dat ) ]
		
		return self


	def scale_axis(self, s, axis = -1):
		"""	COPY. Scale an axis according a vector of values s.
			b_{i1, ..., i_axis, ...} = s_{i_axis} a_{i1, ..., i_axis, ...}
			"""
	
		if axis == self.rank - 1:
			axis = -1
		if len(s) != self.shape[axis]: raise ValueError, "incompatible lengths, len(s) = %i, self.shape[%i] = %i" % (len(s), axis, self.shape[axis])
		b = self.empty_like()
		b.dtype = np.find_common_type([self.dtype], [s.dtype])
		
		q_i = self.q_ind[axis]

		if axis != -1: 
			b.dat = [ np.swapaxes(np.swapaxes(t, axis, -1)*s[ q_i[q, 0]:q_i[q, 1] ], axis, -1) for q, t in itertools.izip(self.q_dat[:, axis], self.dat ) ]
		else: #not sure if this is faster
			#try:
			b.dat = [ t*s[ q_i[q, 0]:q_i[q, 1] ] for q, t in itertools.izip(self.q_dat[:, axis], self.dat ) ]
			#except:
			#	self.check_sanity()
			#	print self.dat
			#	print self.q_dat[:, axis]
			#	for q, t in itertools.izip(self.q_dat[:, axis], self.dat ):
			#		print q, q_i.shape, q_i[q, 0], q_i[q, 1]
				
		return b
		

	def ipermute_qi(self, axis, perm):
		"""	Permute the q-indicies on a particular leg (axis).
			Parameters:
				axis: specifies which leg
				perm: a list, of length len(q_ind[axis])
				
			Example:
			>>>	B.ipermute_qi(0, [1,0])
			permutes the 0th leg (assumed to have two q-blocks), e.g. takes the q_flat [1,1,1,3] to [3,1,1,1]
			"""
		if axis < 0: axis = axis + self.rank
		qind = self.q_ind[axis].copy()
		if len(perm) != len(qind): raise ValueError
		qind[:, 0] = qind[:, 1] - qind[:, 0]		# temporarily store the size of each q-block in the 0th column
		qind = qind[perm, :]
		qind[:, 1] = np.cumsum(qind[:, 0])		# figure out the q_ind from the sizes
		qind[1:, 0] = qind[:-1, 1]
		qind[0, 0] = 0
		self.q_ind[axis] = qind
		for r in self.q_dat:
			r[axis] = perm[r[axis]]
		return self


	def iproject(self, mask, axes):
		"""	Similar function to "np.take" when using boolean index arrays. IN PLACE.
		
			If mask/axes are lists, projections are performed on multiple axes.
		
			'mask' is a boolean index array (or list of them) specifing locations of the indices to be kept.
			'axes' specifies either the axes (or a list of axes) on which the projection occurs.
		
			"""
		axes = tonparray(axes) % self.rank
		if type(mask) != list: mask = [mask]
#		print "iproject: shape, axes, mask:", self.shape, axes, mask
#		self.print_q_dat()
		if len(axes) != len(mask): raise ValueError
		naxes = len(axes)
		list_q_mask = []		# list of q_mask's, each one being a list of slices/Nones
		list_q_map = []
		for a,m in itertools.product(axes,mask):
#			print self.q_ind[a]
			if len(m) != self.shape[a]: raise ValueError, "mismatched size len(mask[axes "+str(a)+"]) (=" + str(len(m)) + ") != shape["+str(a)+"] (=" + str(self.shape[a]) + ")"
			if m.dtype != bool: raise ValueError, "mask[axes "+str(a)+"].dtype not bool"
			if not np.any(m): raise NotImplementedError, "mask at axes " + str(a) + " have all False.  It'll get fixed eventually."
			q_mask = []		# list of mask/Nones that qs maps to, tells you how to update dat
			q_map = []		# list of where the new charge sector is (or None), tells you how to update q_dat
			new_qs = 0		# qs will map to new_qs, new_qs <= qs as some of the old charge sectors might disappear
			for qs in range(self.q_ind[a].shape[0]):
				qs_mask = m[ self.q_ind[a][qs,0] : self.q_ind[a][qs,1] ]
				if np.any(qs_mask):
					q_mask.append(tuple( [slice(None)]*a + [qs_mask] + [slice(None)]*(self.rank-a-1) ))
					q_map.append( new_qs )
					new_qs += 1
				else:
					q_mask.append(None)
					q_map.append(None)
			list_q_mask.append(q_mask)
			list_q_map.append(q_map)
			self.q_ind[a] = q_ind_from_q_flat( q_flat_from_q_ind(self.q_ind[a]) [m] )
#			print self.q_ind[a]
		self.shape = shape_from_q_ind(self.q_ind)
#		print list_q_mask
#		print list_q_map

		if naxes > 1: raise NotImplemented		#############
		targetr = 0			# q_dat[r] will be copied to q_dat[targetr]
		for r in range(len(self.q_dat)):
			row_dat = self.q_dat[r]		# row_dat is a view

#			print self.dat[r].shape, "= T.shape"
			a = axes[0]			# would change if there are multiple axes
			qs_mask = list_q_mask[0][row_dat[a]]
			qs_map = list_q_map[0][row_dat[a]]
			if qs_map is not None:
				self.dat[targetr] = self.dat[r][qs_mask]
				row_dat[a] = qs_map
				self.q_dat[targetr, :] = row_dat			# this is a copy
#				print self.dat[targetr].shape, "= target.shape (" + str(targetr) + ")"
				targetr += 1

		# eliminate trailing rows in q_dat (and dat)
		if targetr == 0:
			self.dtype = np.dtype(int)
			self.q_dat = np.zeros((0, self.rank), np.uint)
			self.dat = []
#			print "No q_dat left!"
		elif targetr < len(self.q_dat):
			self.q_dat = self.q_dat[:targetr, :]
			self.dat = self.dat[:targetr]
		# self.sorted does not need to be changed
#		self.print_q_dat()
			
		return self


	def iconj(self, K = True):
		"""	IN PLACE. self = self.conj. Element-wise conjugation + charge conjugation
		"""
		if self.dtype == np.complex and K:
			self.iunary_blockwise(np.conj)
				
		self.charge*=-1		##
		self.q_conj*=-1
		if self.labels is not None:
			new = {}
			for k, v in self.labels.iteritems():
				
				if k[-1] == '*':
					new[k[:-1]] = v
				else:
					new[k + '*'] = v

			self.labels = new

		return self


	def conj(self, K = True):
		"""	COPY. c = self.conj. Takes the complex conjugate of entries, and charge conjugates.
		
			If K = False, then the matrices aren't complex conjugated, but a copy is made.
				
		-Labeling: takes 	'a' -> 'a*'
							'a*'-> 'a'
		"""
		if self.dtype == np.complex and K:
			c = self.unary_blockwise(np.conj)
		else:
			c = self.copy()
		c.charge*=-1		##
		c.q_conj*=-1
		if c.labels is not None:
			new = {}
			for k, v in c.labels.iteritems():
				if k[-1] == '*':
					new[k[:-1]] = v
				else:
					new[k + '*'] = v
			c.labels = new

		return c


	def matvec(self, y):
		#TODO comment this			
		return tensordot(self, y, axes = 1)
		
	
	def iaxpy(self, alpha, b):
		""" In place a -> a + alpha*b for scalar alpha, a = self
		
			The point of this function is to implement frequent BLAS axpy without intermediate conversion to npc.
		"""
		npc_helper.iaxpy(alpha, b, self)
		return self


	def __add__(self, other):
		if isinstance(other, array):
			return binary_blockwise( np.add, self, other)
		elif sp.isscalar( other ):
			addf = functools.partial(np.add, other)
			return self.unary_blockwise(addf)
		else:
			return self.to_ndarray().__add__(other)
		

	def __radd__(self, other):
		return self + other
	

	def __iadd__(self, other):
		if isinstance(other, array):
			return self.ibinary_blockwise( np.add, other )
		elif sp.isscalar( other ):
			for t in self.dat:
				t[:] = t + other
			return self
		else:
			raise ValueError, "Unknown type: " + str(type(other))
			

	def __sub__(self, other):
		if isinstance(other, array):
			return binary_blockwise( np.subtract, self, other)
		elif sp.isscalar( other ):
			subf = functools.partial(np.subtract, other)
			return self.unary_blockwise(subf)
		else:
			return self.to_ndarray().__sub__(other)
	

	def __isub__(self, other):
		if isinstance(other, array):
			return self.ibinary_blockwise( np.subtract, other)		
		elif sp.isscalar( other ):
			for t in self.dat:
				t[:]= t - other
			return self
		else:
			raise NotImplemented
				

	def __mul__(self, other):
		if sp.isscalar( other ):
			if other == 0:
				return self.zeros_like()
			mulf = functools.partial(np.multiply, other)
			return self.unary_blockwise(mulf)
		else:
			raise NotImplemented
	
	def __div__(self, other):
		if sp.isscalar( other ):
			if other == 0:
				raise ZeroDivisonError
			else:
				other = 1./other
			mulf = functools.partial(np.multiply, other)
			return self.unary_blockwise(mulf)
		else:
			raise NotImplemented


	def __rmul__(self, other):
		return self*other
	

	def __imul__(self, other):
		if sp.isscalar( other ):
			if other == 0:
				self.dat = []
				self.q_dat = np.empty( (0, self.rank), np.uint)
				self.sorted = True
				return self
			for t in self.dat:
				t[:] = t*other
			return self
		else:
			raise NotImplemented	


	def __idiv__(self, other):
		if sp.isscalar( other ):
			for t in self.dat:
				t[:] = t / other
			return self
		else:
			raise NotImplemented



	################################################################################
	##	Misc

	def sort_q_dat(self):
		"""	sort q_dat by lex order """
		
		if self.sorted:
			return
			
		if len(self.q_dat) <2:
			self.sorted = True
			return
			
		perm = np.lexsort(self.q_dat.transpose())
		self.q_dat = self.q_dat[perm, :]
		self.dat = [ self.dat[p] for p in perm ]
		self.sorted = True


	def norm(self, ord=None, convert_to_float=True):
		"""	Calculate the norm.
			ord may be one of
				None/'fro'	Frobenius norm (same as 2-norm)
				inf		max(abs(x))
				-inf		min(abs(x))
				0			sum(x != 0)  (this is not the same as the 0-norm, which is ill-defined)
				1			max(sum(abs(x), axis=0)) for matrices  (this is not the 1-norm)
				-1 		min(sum(abs(x), axis=0)) for matrices  (this is not the (-1)-norm)
				2			largest sing. value for matrices, 2-norm otherwise
				-2 		smallest singular value for matrices, (-2)-norm otherwise
				other		[ sum_i |a_i|**ord ] **(1./ord), the usual (ord)-norm
			"""
		# TODO, doesn't work for ord = +-1, +-2 for 2-arrays (according to numpy.linalg.norm)
		# TODO, might overflow if dtype = int
		# works for ord = 'fro', +-inf, 0, and other
		dat = self.dat
		if convert_to_float:
			if self.dtype == complex:
				list_norm = np.empty((len(dat)), complex)
				for i in range(len(dat)):
					list_norm[i] = np.linalg.norm(np.array(dat[i].reshape((-1,)), complex), ord=ord)
			else:
				list_norm = np.empty((len(dat)), float)
				for i in range(len(dat)):
					list_norm[i] = np.linalg.norm(np.array(dat[i].reshape((-1,)), float), ord=ord)
		else:
			list_norm = np.empty((len(dat)), self.dtype)
			for i in range(len(dat)):
				list_norm[i] = np.linalg.norm(dat[i].reshape((-1,)), ord=ord)
		return np.linalg.norm(list_norm, ord=ord)
		


	

################################################################################
################################################################################

class leg_pipe:
	"""	This class holds the data required to combine or split a set of legs into a total ('t') leg.  I will speak of the multiple legs as "incoming" and the combined leg as "outgoing".
	
		For np.reshape, taking, for example,  ij --> k amounted to k = s1*i + s2*j for appropriate strides s1, s2. In the charged case, however, we want to block k by charge, so we must implicitly permute as well. This reordering is encoded in qt_map.
		
			Each q-index combination of the "d" input legs, (i_1, . . ., i_d) , will end up getting placed in some slice m_i:m_{i+1} of the outgoing leg. Within this slice, the data is simply reshaped in usual row-major fashion. It will be a subslice of a new total block labeled by q-index I_s. Because many charge combinations fuse to the same total charge, in general there will be many tuples (i_1, . . . , i_d) belonging to the same I_s.
			
			The rows of qt_map are precisely the collection of
			[ m_i, m_{i+1},  i_1, . . . , i_d, I_s ]
			
			They are lex sorted by I_s, then i. Each I_s will have multiple rows, and the order in which they are stored in qt_map is the order the data is stored in the actual tensor, ie, it might look like
	
			[ ...
			  [ m_i, m_{i+1},  i_1, . . . , i_d, I_s ]
			  [ m_{i+1}, m_{i+2},  i'_1, . . . , i'_d, I_s ]
			  ... 											 ]
		"""
	
	
	def __init__(self, legs):
		self.nlegs = legs		# number of legs to be contracted
		self.mod_q = None
		
		self.qshape = None		# Number of blocks in each individual leg, represented as tuple (l1, l2, ...).  Equivalent to len(q_ind[i])
		self.t_qshape = None	# Number of blocks in new combined leg, tuple  (l,).  Equivalent to len(qt_ind)
		
		self.shape = None		# Size in each individual leg, represented as 1-array (l1, l2, ...).  Equivalent to q_ind[i][-1,1].
		self.t_shape = None		# Size of the combined leg, np.int64.  Equivalent to qt_ind[-1,1] = product of shape.
		
		self.q_ind = []			# list of q_ind for the constituents legs (each np.array)
		self.qt_ind = None		# q_ind for the total leg (np.array)
		
		self.q_map = None		# As below, but ordered by leg q-indices 'i_1, ..., i_d'
		self.qt_map = None		# 2-array (ordered by incoming q-indices) shape (# block, 2 + d + 1)
			# Entries:  [ m_i, m_{i+1},  i_1, . . . , i_d, I_s ] (lex) sorted by I_s, then i
		
		self.perm = None		# permutation array from qt_map to q_map (size # block)
		self.qt_map_ind = None	# a 2-array that takes total charge sectors to slices in qt_map, shape (t_qshape, 2)
		
		self.q_conj = None		# np.array shape (nlegs,) of +-1
		self.qt_conj = None		# single number +-1




	def check_sanity(self):
		"""	Checks if the internal structure of the array is consistent.

			Raise an exception if anything is wrong, do nothing if everything is right.
			"""
	#TODO do sanity check
	##	rank
		nlegs = self.nlegs
		if not isinstance(nlegs, int): raise ValueError, "nlegs is not an int (type=%s)" % str(type(rank))
		if nlegs <= 0: raise ValueError, "nlegs is negative: " + str(nlegs)
	##	mod_q
		mod_q = self.mod_q
		if not isinstance(mod_q, np.ndarray): raise ValueError, "mod_q not a numpy.ndarray (type=%s)" % str(type(mod_q))
		if mod_q.ndim != 1: raise ValueError, "mod_q is not 1D: shape = " + str(mod_q.shape)
		if mod_q.dtype != int: raise ValueError, "mod_q.dtype not int (dtype=" + str(mod_q.dtype) + ")"
		if np.any(mod_q <= 0): raise ValueError, "mod_q has non-positive values: mod_q = %s" % (mod_q,)
		num_q = mod_q.shape[0]
	##	shapes
		qshape = self.qshape
		t_qshape = self.t_qshape
		shape = self.shape
		t_shape = self.t_shape
		#print type(qshape), type(t_qshape), type(shape), type(t_shape)
	##	q_conj
	##	q_ind's
	##	q_map's
	##	More....


	def __str__(self):
		s = "leg_pipe (d=%s) with shape %s->%s\n" % (self.nlegs, self.shape, self.t_shape)
		s += joinstr( ["\tq_ind: "] + map(str, self.q_ind) + ["  ->  qt_ind: ", str(self.qt_ind)]) + "\n"
		s += "\tq_conj: %s, qt_conj: %s\n" % (self.q_conj, self.qt_conj)
		s += joinstr( ["\tqt_map: ", str(self.qt_map)] )
		return s

	@classmethod
	def trivial_pipe(cls, q_ind, q_conj, mod_q = None):
		"""	Make a pipe for a single leg that does nothing.

			Special constructor for a single q_ind and converts to pipe form.
			Should be equivalent to make_pipe (d=1, block_single_legs=False),
			except it always sets qt_conj = q_conj[0].
			
			Parameters:
				q_ind, an np.array, shaped (#, 2 + num_q)
				q_conj, an integer +-1
				mod_q, an np.array shaped (num_q,)
			"""
	#	if len(q_ind) > 1:
	#		raise ValueError, "Trivial pipe called with multiple legs?"
	#	q = q_ind[0]
		pipe = cls(1)
		pipe.q_ind = [ q_ind ]

		length = len(q_ind)			# number of charge sectors
		pipe.qshape = (length,)
		if length > 0: pipe.shape = (shape_from_q_ind(q_ind),)
		else: pipe.shape = (0,)
		pipe.t_shape = pipe.shape[0]
		pipe.t_qshape = pipe.qshape
		pipe.strides = np.array([1], dtype = np.intp)
		pipe.qt_map = np.empty((length, 4), dtype = np.uint)
		pipe.qt_map[:, 0] = 0
		pipe.qt_map[:, 1] = q_ind[:, 1] - q_ind[:, 0]
		pipe.qt_map[:, 2] = np.arange(length)
		pipe.qt_map[:, 3] = np.arange(length)
		pipe.qt_map_ind = np.empty((length,2), int)
		pipe.qt_map_ind[:, 0] = np.arange(length)
		pipe.qt_map_ind[:, 1] = np.arange(1, length+1)
		pipe.qmul_map = pipe.qt_map
		pipe.qt_ind = q_ind
		
		pipe.q_conj = np.array([q_conj], int)
		pipe.qt_conj = np.array([q_conj], int)
		
		if mod_q is None:
			warningMsg = "npc.leg_pipe.make_pipe: mod_q = None"
			print warningMsg
			warnings.warn(warningMsg, RuntimeWarning)
			pipe.mod_q = np.ones(q_ind.shape[0] - 2, dtype = np.int)
		else:
			pipe.mod_q = mod_q

		#pipe.check_sanity()
		return pipe
		
	
	@classmethod
	#@profile
	def make_pipe(cls, q_ind, q_conj, qt_conj = 1, block_single_legs = True, mod_q = None, verbose = 0):
		# TODO more description, write a sample
		"""	Takes a list of q_inds and q_conj, and builds appropriate leg_pipe's.
		
			When forming the new qt_ind (the q_ind for the combined leg), we can choose the sign such that either qt_conj = 1 or -1, so we allow for the optional specification of qt_conj.
			
			 q_conj example. Suppose I fuse 2 legs ---> 1 leg. To focus on the sign convention, suppose all legs are of dimension 1, and the physical, inward pointing charges of the  2 legs are Q0 and Q1. Fusion of course gives 
			 
			 	QT = Q0 + Q1 (fusion of physical charges)
				
				However, the q_ind for the input legs may be conjugated, and we might want to produce a conjugated qt_ind. Let q0, q1, qt be the charges ***as stored in q_ind***. Then
				
				qt*qt_conj = q0*q_conj[0] + q1*q_conj[1]  (implementation of fusion with q_ind, q_conj)
			 
			 So the qt stored in qt_ind is constructed in accord with this convention.
		"""
		
		d = len(q_ind)
		q_conj = tonparray(q_conj)
		
		# Single legs get processed (combining blocks with the same charges), unless block_single_legs = False
		if not block_single_legs and d == 1:
			return cls.trivial_pipe(q_ind[0], q_conj[0], mod_q = mod_q)
		t0 = time.time()
		pipe = cls(d)
		pipe.q_ind = q_ind		# No need to copy?
		pipe.q_conj = q_conj.copy()
		pipe.qt_conj = qt_conj
		qshape = pipe.qshape = tuple([ len(q) for q in q_ind ])		# number of charge sectors in each leg
		pipe.shape = shape_from_q_ind(q_ind)
		pipe.t_shape = np.prod(pipe.shape)
		num_q = q_ind[0].shape[1] - 2
		if mod_q is None:
			warningMsg = "npc.leg_pipe.make_pipe: mod_q = None"
			print warningMsg
			warnings.warn(warningMsg, RuntimeWarning)
			pipe.mod_q = np.ones(num_q, dtype = np.int)
		else:
			pipe.mod_q = mod_q

		"""	We enumerate over the outer product of each constituent leg's charge sectors, 'q_i', forming a temporaray table
			qt_map = [ [ s_q0=0 , s_q1=(n_i1 * n_i2 ...) , i1 , i2 , ... , QT ], . . . ]
			where n_i1 is the dimension of the charge sector q_i1 on leg 1, etc. QT = q_i1 + q_i2 + . . .
			The outer product is taken over all i1, i2 . . .

			this is temporary - the charge will be removed and replaced by q-index later
			"""
		
		##	First make grid to set the multi-index sector
		
		size = 2 + d + max(num_q, 1)

		qt_map = np.empty( (size,) + qshape, dtype = int )		# created transpose	
		grid = np.mgrid[ [slice(0, l) for l in qshape] ]
		qt_map[2:2+d, ...] = grid # mgrid gives a (d+1)-array shaped: (d,)+qshape
		pipe.strides = np.array(grid.strides, np.intp)[1:]/grid.itemsize
		qt_map = qt_map.reshape((size, -1)).transpose()				# now qt_map[:, 2:2+d] has a list of all possible sub q-indices
		
		if num_q > 0:
			##	Second initialize total charge
			qt_map[:, 2+d:] = npc_helper.q_sum( qt_map[:, 2:2+d].astype(np.uint), q_ind, q_conj*qt_conj, 0, d, pipe.mod_q )
		##	Next, temporarily keep size of charge sector
		qt_map[:, 1] = q_ind[0][qt_map[:, 2], 1] - q_ind[0][qt_map[:, 2], 0]
		for k in range(1, d):
			qt_map[:, 1] *= (q_ind[k][qt_map[:, 2+k], 1] - q_ind[k][qt_map[:, 2+k], 0])	# length
		
		##	Now sort by final charge
		revperm = np.lexsort(qt_map[:, 2:].transpose())
		qt_map = qt_map[revperm, :]
		pipe.perm = np.argsort(revperm)	# Save order needed to bring back mult_ind order
		
		##	Partition into sectors of final charge
		indices = npc_helper.find_differences( qt_map[:, 2+d:] )
		num_sec = len(indices) - 1
		pipe.t_qshape = (num_sec,)
		qt_ind = np.empty( (num_sec, num_q + 2), dtype = int )
		
		##	compute the ranges (first two columns of qt_map)
		qt_map_ind = np.empty((num_sec,2), int)
		for j in xrange(num_sec):
			i1 = indices[j]
			i2 = indices[j+1]
			qt_map[i1:i2, 1] = np.cumsum( qt_map[i1:i2, 1] )
			qt_map[i1 + 1:i2, 0] = qt_map[i1:i2 - 1, 1]
			qt_map[i1, 0] = 0
			qt_ind[j, 2:] = qt_map[i1, 2+d:]
			qt_map[i1:i2, 2+d] = j
			qt_map_ind[j, 0] = i1
			qt_map_ind[j, 1] = i2
	
		qt_map = qt_map[:, 0:3 + d]		# chop qt_map back down to the specification

		qt_ind[:, 1] = np.cumsum(qt_map[indices[1:]-1, 1])
		qt_ind[0, 0] = 0
		qt_ind[1:, 0] = qt_ind[:-1, 1]

		pipe.qt_map = qt_map.astype(np.uint)
		pipe.qt_ind = qt_ind
		pipe.qt_map_ind = qt_map_ind

		#pipe.check_sanity()
		return pipe


		
	def get_q_map(self):
		"""	returns the q_map """
		if self.q_map is None:
			q_map = np.take(self.qt_map, [0, 1, -1], axis = 1)
			if self.perm is not None:				
				self.q_map = np.take(q_map, self.perm, axis = 0)
			else:
				self.q_map = q_map
					
			return self.q_map
		else:
			return self.q_map


	def npc_to_flatform(self, a, qindex=None):
		"""	Return a 1D np.ndarray filled with elements of a.

				Schematically:
					a -> a_combine (combine_legs with self)
					a_combine only has one non-zero element (because a conserves charge),
					which is the output.
			"""
		if qindex is None:
			qt_ind_dict = { tuple(q[2:].tolist()): i for i,q in enumerate(self.qt_ind) }
			#TODO, look up the qindex from a
			raise NotImplementedError
		else:
			qt_map_slice = self.qt_map[self.qt_map_ind[qindex][0]: self.qt_map_ind[qindex][1]]
			if len(qt_map_slice) == 0: return np.zeros(0, dtype=a.dtype)
			a.sort_q_dat()
			t = np.zeros(qt_map_slice[-1, 1], dtype=a.dtype)
			a_qdat_i = 0
			for qrow in qt_map_slice:
				if a_qdat_i >= len(a.q_dat): break
				if np.array_equiv(qrow[2:5], a.q_dat[a_qdat_i]):
					t[qrow[0]: qrow[1]] = a.dat[a_qdat_i].reshape(-1)
					a_qdat_i += 1
			return t


	def flatform_to_npc(self, t, qindex=None, cutoff=1e-16):
		"""	Return a npc.array with elements from t.
			
				This function is the opposite from npc_to_flatform()
			"""
		if qindex is None: raise ValueError
		qt_map_slice = self.qt_map[self.qt_map_ind[qindex][0]: self.qt_map_ind[qindex][1]]
		a = array(self.nlegs, dtype=t.dtype)
		a.shape = self.shape
		a.mod_q = self.mod_q
		a.num_q = len(a.mod_q)
		a.q_ind = self.q_ind
		a.q_conj = self.q_conj
		a.charge = self.qt_ind[qindex, 2:] * self.qt_conj

		qdat = np.zeros((len(qt_map_slice), self.nlegs), dtype=np.uint)
		dat = [None] * len(qt_map_slice)
		qdim = [ q[:,1]-q[:,0] for q in self.q_ind ]
		mat_shapes = qdat.copy()		# store list of shapes for each piece in qt_map_slice
		for l in range(self.nlegs):
			mat_shapes[:, l] = qdim[l][qt_map_slice[:, 2+l]]
		num_qdat = 0
		for r,qrow in enumerate(qt_map_slice):
			v = t[qrow[0]:qrow[1]]
			if np.any(np.abs(v) >= cutoff):
				qdat[num_qdat] = qrow[2:2+a.rank]
				dat[num_qdat] = v.reshape(mat_shapes[r])
			#	print qrow, np.linalg.norm(v), mat_shapes[r]
				num_qdat += 1
		a.q_dat = qdat[:num_qdat]
		a.dat = dat[:num_qdat]
		#a.check_sanity()
		return a


	def t_to_mul_ndarray(self, t, qindex=None, a=None):
		"""	Returns a d-imensional ndarray (partially) filled with elements of t. (d = nlegs)
			This function takes the flattened form of an npc.array (with q_ind's that of self), and unflatten it.
			-	This function is similar to flatform_to_npc(), but splits out an np.ndarray
			
			Parameters:
				t is an np.ndarray
				if qindex is not set, takes in an array length t_shape
				if qindex is set, takes in an array length specified in qt_ind[qindex, :]
				if a is set, write to a (instead of creating new np.ndarray)
			"""
		t = tonparray(t)
		if qindex is None:
			if t.shape != (self.t_shape,): raise ValueError
			a = np.zeros(self.shape, t.dtype)
			for r,qt_row in enumerate(self.qt_ind):
				self.t_to_mul_ndarray(t[qt_row[0]:qt_row[1]], qindex=r, a=a)
			return a
		else:
			qt_map_slice = self.qt_map_ind[qindex, :]
			qt_map = self.qt_map[qt_map_slice[0]:qt_map_slice[1], :]
			if t.shape != (shape_from_q_ind(qt_map),): raise ValueError, "incorrect length"
			if a is None:
				a = np.zeros(self.shape, t.dtype)
			else:
				if np.any(a.shape != self.shape): raise ValueError
			q_ind = self.q_ind
			# fill in the subblocks of a
			for r in qt_map:
				slices = [ slice(q_ind[l][r[2+l],0], q_ind[l][r[2+l],1]) for l in range(self.nlegs) ]
				shape = [ q_ind[l][r[2+l],1] - q_ind[l][r[2+l],0] for l in range(self.nlegs) ]
				a[slices] = t[r[0]:r[1]].reshape(shape)
			return a


	def imap_Q(self, func, r):
		"""Maps charge data, in place"""
		if func is None: return self
		
		self.q_ind = [ func(qi, r) for qi in self.q_ind]
		self.qt_ind = func(self.qt_ind, r)
		
		return self


	def shallow_map_Q(self, func, r):
		# TODO
		raise NotImplementedError
			


################################################################################
################################################################################
##	The rest of this are not class functions


def load(name):
	"""Load an npc.array from a .save() call
	
	"""
	with open(name, 'r') as f:
		a = cPickle.load(f)
		dat = []
		for i in xrange(a.dat):
			dat.append(np.load(f))
		a.dat = dat
			
	return a

		
def q_ind_from_q_flat(q_flats):
	"""	Given a list of lex-sorted charges q_flat, generate the corresponding q_ind format.
		
		Assumes q_flat is sorted!
		If q_flat is not sorted, it will group by sectors of contiguous charge, i.e,,
		[2, 2, 0, 1, 1] -> [2, 0, 1]
		[0, 1, 0, 2, 2] -> [0, 1, 0, 2]
		with correct multiplicity.
		"""
	if type(q_flats) != list:
		q_flats = [q_flats]
		strip = True
	else:
		strip = False
	
	q_inds = []
	for q_flat in q_flats:
		length, num_q = q_flat.shape
		#indices = find_differences_inline(q_flat)
		
		indices = npc_helper.find_differences(q_flat)
		indices = indices.reshape( (-1, 1))
		q_inds.append( np.hstack((indices[0:-1, :], indices[1:, :], q_flat[ indices[0:-1, 0], : ])) )
	
	if strip:
		return q_inds[0]
	else:
		return q_inds


def q_flat_from_q_ind(q_inds):
	"""	Take q_inf to q_flat form"""
	
	if type(q_inds) != list:
		q_inds = [q_inds]
		strip = True
	else:
		strip = False
		
	q_flats = []
	
	for q_ind in q_inds:
		na, nb = q_ind.shape
		num_q = nb - 2
		length = shape_from_q_ind(q_ind)
		
		q_flat = np.empty( (length, num_q) , dtype = np.int)
		
		for sec in q_ind:
			q_flat[sec[0]:sec[1], :] = sec[2:]
		
		q_flats.append(q_flat)
		
	if strip:
		return q_flats[0]
	else:
		return q_flat
	

def q_dict_from_q_ind(q_ind):
	"""	Convert q_ind form to q_dict form, ie, returns list of dicts, each with charge -> slice key/values """
	# make a list of dictionaries (one for each leg) that takes charges to a slice
	if type(q_ind) == list:
		return [ dict( [ (  tuple(qsec[2:]),  slice(qsec[0], qsec[1]) ) for qsec in q_i] ) for q_i in q_ind ]
	else:
		return dict( [ (  tuple(qsec[2:]),  slice(qsec[0], qsec[1]) ) for qsec in q_ind] )

#@profile
def q_dict_from_q_flat(q_flats):
	
	if type(q_flats) != list:
		q_flats = [q_flats]
		strip = True
	
	q_dicts = []

	for q_flat in q_flats:
		length, num_q = q_flat.shape
		#indices = find_differences_inline(q_flat)
		indices = npc_helper.find_differences(q_flat)
		q_dicts.append( dict( [ ( tuple(q_flat[indices[i], :]), slice(indices[i], indices[i+1]) ) for i in range(len(indices)-1) ] ))
	if strip:
		return q_dicts[0]
	else:
		return q_dicts


def shape_from_q_ind(q_inds):
	"""	Given a list of q_ind's, return the shape.
		In the case of a single q_ind, returns an int (its length)
		"""
	if type(q_inds) != list:
		if q_inds.shape[0] > 0: return q_inds[-1,1]
		return 0
	else:
		return np.array([ (0 if q_ind.shape[0] == 0 else q_ind[-1,1]) for q_ind in q_inds ], dtype = np.intp) #SHAPE


def q_ind_match_raw(qind1, qind2, qconj1, qconj2, mod_q):
	"""	Return true if (qind1, qconj1) is the same as (qind2, qconj2).  I.e. interchangable. """
	if not np.array_equal(qind1[:, 0:1], qind2[:, 0:1]): return False
	q_diff = mod_onetoinf2D( qind1[:, 2:] * qconj1 - qind2[:, 2:] * qconj2, mod_q )
	if np.any(q_diff != 0): return False
	return True


def q_ind_match(a1, leg1, a2, leg2):
	"""	Return true if (a1.q_ind[leg1], a1.q_conj[leg1]) is the same as (a2.q_ind[leg2], a2.q_conj[leg2]).  I.e. interchangable. """
	if not np.array_equal(a1.mod_q, a2.mod_q): return False
	return q_ind_match_raw(a1.q_ind[leg1], a2.q_ind[leg2], a1.q_conj[leg1], a2.q_conj[leg2], a1.mod_q)


def q_ind_contractible(a1, leg1, a2, leg2, uncontractible_error_msg = None):
	"""Return true if (a1.q_ind[leg1], a1.q_conj[leg1]) is the opposite of (a2.q_ind[leg2], a2.q_conj[leg2]).  I.e. contractible.
	If the legs are not contractible and uncontractible_error_msg is not None, than raise a ValueError.
		"""
	if not np.array_equal(a1.mod_q, a2.mod_q):
		return False
	contractible = q_ind_match_raw(a1.q_ind[leg1], a2.q_ind[leg2], a1.q_conj[leg1], -a2.q_conj[leg2], a1.mod_q)
	if contractible: return True
	if uncontractible_error_msg is not None:
		print "npc.q_ind_contractible(): uncontractible legs."
		print joinstr(["    ", "q_ind[0] = \n ({})".format(a1.q_conj[leg1]), a1.q_ind[leg1], \
				",  ", "q_ind[1] = \n ({})".format(a2.q_conj[leg2]), a2.q_ind[leg2]])
		raise ValueError, uncontractible_error_msg
	return False

	


########################################################################################

#@profile
def eye_like(a, axis = 0):
	"""An identity matrix whose row is equivalent to axis of a"""
	return diag(1., a.q_ind[axis], a.q_conj[axis], a.mod_q)

def diag(s, q_ind, q_conj = 1, mod_q = None, dtype = None):
	"""	Returns a square, diagonal matrix of entries 's', with 		
		q_ind[0] = q_ind = q_ind[1] 
		q_conj[0] = q_conj = - q_conj[1]
	
		The resulting charge is 0, and dtype is determined by s.
		
		's' can be of three types: a python scalar, a length 1 1D-ndarray, or a length L 1D-ndarray (with L size of matrix). For the scalar cases, the diagonal is taken constant.
		
		the dtype is inferred from s unless dtype is not None
	"""
	
	s = np.asarray(s, dtype)
	if s.ndim == 0:
		scalar = True
	elif len(s) != shape_from_q_ind(q_ind):
		raise ValueError, "Length of 's' not equal to length of q_ind"
	else:
		scalar = False
		
	a = zeros( [q_ind, q_ind], s.dtype, q_conj = [q_conj, -q_conj], charge = None, mod_q = mod_q )
	
	q_dat = np.arange(len(a.q_ind[0]), dtype = np.uint).reshape((-1, 1))
	a.q_dat = np.tile(q_dat, (1, 2) ) #q_dat = [ [0, 0], [1, 1], ... ]
	dat = [None]*len(a.q_ind[0])
	at = 0
	for i, q in enumerate(a.q_ind[0]): #For each q_index 
		size = q[1] - q[0]		
		if scalar:
			dat[i] = np.diag(s*np.ones(size, dtype = s.dtype))       #make the diagonal block
		else:
			dat[i] = np.diag(s[at:at+size])
		at = at + size
	
	a.dat = dat
	a.sorted = True

	return a


def zeros( q_ind, dtype = float64, q_conj = None, charge = None, mod_q = None ):
	"""	Returns zero array with charge data specified by q_ind.
		(of course these zeros are not stored -- .dat and .q_dat are both empty
		"""
			
	if dtype == float:
		dtype = float64
	d = len(q_ind)
	if d <= 0: 
		raise ValueError
	num_q = q_ind[0].shape[1] - 2
	shape = shape_from_q_ind(q_ind)
	

	ac = array(d, dtype)
	ac.shape = shape
	ac.num_q = num_q

	if mod_q is None:
		mod_q = ac.mod_q = np.ones(num_q, dtype = np.int)
	else:
		ac.mod_q = mod_q
	if ac.mod_q.shape != (num_q,): raise ValueError, "mod_q is wrong length (mod_q.shape = %s)" % (ac.mod_q.shape,)
	if np.any(ac.mod_q == 0): raise ValueError, "mod_q = %s has a 0 in it" % (ac.mod_q,)
		
	if charge is None:
		ac.charge = np.zeros(num_q, dtype = np.int)
	elif type(charge) == list:
		ac.charge = np.array(charge, dtype = np.int)
	elif sp.isscalar(charge):
		ac.charge = np.array([charge])
	else:
		ac.charge = charge.copy()
	if ac.charge.shape != (num_q,):
		raise ValueError, "Charge is wrong length"
		
		
	if q_conj is None:
		q_conj = ac.q_conj = np.ones(d, dtype = np.int)
	elif type(q_conj) == list:
		q_conj = ac.q_conj = np.array(q_conj)
	else:
		ac.q_conj = q_conj.copy()
	
	if ac.q_conj.shape != (d,):
		raise ValueError, "qconj is wrong length"
				
		
	ac.q_ind = [ qi for qi in q_ind ] # not copied!
	ac.q_dat = np.empty([0, ac.rank], dtype = np.uint)
	ac.dat = []
		
	ac.sorted = True
	return ac


def concatenate(bs, axis = 0, copy = True):
	""" Concatenates npc arrays like np.concatenate.
		Does NOT block by charge on the modified leg (just stacks q_ind data).
		
		If copy == False, does NOT duplicate data and the input bs are destructively modified.
		
		If there are duplicate entries in bs (the same array twice) it is important that copy = False
		
	"""
	#print "Concatenate", axis, [b.shape for b in bs]
	#for b in bs:
	#	b.check_sanity()
	num = len(bs)
	num_q = bs[0].num_q
	mod_q = bs[0].mod_q
	
	if copy:
		bs = [b.copy() for b in bs]
	
	for b in bs:
		b.q_ind[axis] = b.q_ind[axis].copy()

	#Light check for compatability (doesn't check charges)
	for r in range(num-1):
		if bs[r].num_q!=num_q or not np.array_equiv(bs[r].mod_q, mod_q):
			raise ValueError, [b.num_q for b in bs], [b.mod_q for b in bs]
		diffs = (bs[0].shape!=bs[r].shape)
		diffs[axis] = False
		if np.any(diffs):
			raise ValueError, [ b.shape for b in bs]
		if not array_equiv_mod_q(bs[0].charge, bs[r].charge, bs[0].mod_q):
			raise ValueError, [ b.charge for b in bs]

	#First. Shift indices of q_inds
	n_inds = [b.shape[axis] for b in bs]
	cum_inds = np.cumsum(n_inds)

	for r in range(1, num):
		bs[r].q_ind[axis][:, 0]+=cum_inds[r-1]
		bs[r].q_ind[axis][:, 1]+=cum_inds[r-1]
	#Second. Shift block indices of q_dat
	n_blocks = [b.q_ind[axis].shape[0] for b in bs]
	cum_blocks = np.cumsum(n_blocks)

	for r in range(1, num):
		bs[r].q_dat[:, axis]+=cum_blocks[r-1]

	#Concatenate data
	dat = list(itertools.chain.from_iterable( [ b.dat for b in bs ] ))
	q_dat = np.concatenate([b.q_dat for b in bs], axis = 0)
	q_ind = np.concatenate([b.q_ind[axis] for b in bs], axis = 0)

	#Plop into bs
	b = bs[0]
	b.dat = dat
	b.q_dat = q_dat
	b.q_ind[axis] = q_ind
	b.shape[axis] = cum_inds[-1]
	b.sorted=False

	b.check_sanity()
	return b

def grid(bs, axes, copy=True):
	""" Given bs = np.array[dtype=object] of npc-arrays, performs a multi-dimensional concatentation along 'axes'.
		Does NOT block by charge on the modified leg (just stacks q_ind data).
		
		For example,
			bs = [[A, B], [C, D] ], axes = [0, 1]
			with each entry rank-2, will form a rank-2 np.array with a block structure
			[[A, B], [C, D] ]
			
			If A... were rank-3, then the resulting array would also have rank-3, but with enlarged dimensions over axes.
	"""

	if type(bs)==list:
		bs = np.array(bs, dtype = np.object)
	if bs.ndim<1:
		raise ValueError

	#Simple recursion on ndim. Copy only required on first go.
	if bs.ndim > 1:
		bs = [ grid(b, axes = axes[1:], copy = copy) for b in bs ]
		copy = False
	

	bs = concatenate(bs, axes[0], copy = copy)
	

	return bs


def norm(a, ord = None):
	"""	Returns the norm of an np.array or npc.array
		"""
	if isinstance(a, array):
		return a.norm(ord)
	elif isinstance(a, np.ndarray):
		return np.linalg.norm(a.reshape((-1)), ord=ord)
	else:
		# differs from the standard 2-array implementation
		return np.linalg.norm(a, ord=ord)


def unary_blockwise(func, a):
	return a.unary_blockwise(func)
	

def binary_blockwise(func, a, b):
	"""	COPY. c = a + b.

		Assuming a, b, are of same shape and charge, applies a binary function 'block - wise' to sub blocks, generating third tensor.
		func is a function (or partial function) which takes two arrays as the first argument and returns an array of the same kind.
		"""
		
	a.sort_q_dat()
	b.sort_q_dat()
	
	try:			
		args = func.args
		if args is None:
			args = ()
		keywords = func.keywords
		if keywords is None:
			keywords = {}
		func = func.func
	except:
		args = ()
		keywords = {}
	
	Na = len(a.q_dat)
	Nb = len(b.q_dat)
		
	if Na == Nb and np.array_equiv(a.q_dat, b.q_dat):
			c = a.empty_like()
			c.dat = [ func(*( (at, bt) + args ), **keywords ) for at, bt in itertools.izip(a.dat, b.dat)]
	else:
		c = a.empty_like(dup_q_dat = False)
		aq = a.q_dat
		bq = b.q_dat
		adat = a.dat
		bdat = b.dat
		i, j = 0, 0
		q_dat = []
		dat = []
		while i < Na or j < Nb:
			if j >= Nb or ( (i < Na) and (tuple(aq[i, ::-1]) < tuple(bq[j, ::-1]))):
				dat.append(func(*( (adat[i], np.zeros_like(adat[i])) + args ), **keywords ) )
				q_dat.append(aq[i])
				i+=1
			elif i >= Na or tuple(aq[i, ::-1]) > tuple(bq[j, ::-1]):
				dat.append(func(*( (np.zeros_like(bdat[j]), bdat[j]) + args ), **keywords ) )
				q_dat.append(bq[j])
				j+=1
			else:
				try:
					dat.append(func(*( (adat[i], bdat[j]) + args ), **keywords ) )
				except ValueError:
					print aq
					print bq
					raise
				q_dat.append(aq[i])
				i+=1
				j+=1
		
		c.dat = dat
		c.q_dat = np.array(q_dat, np.uint)
	if len(c.dat) > 0:
		c.dtype = c.dat[0].dtype
		
	return c
	

def swapaxes():
	"""	TODO """
	raise NotImplementedError


def tensordot_compat(a, b, axes = 2, suppress_warning = False, verbose = 0):
	"""	checks if a, b have compatible axes to do this product, raise a ValueError if they can't be tensordotted together """
	#a.check_sanity(suppress_warning = suppress_warning)
	#b.check_sanity(suppress_warning = suppress_warning)

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


	as_ = a.shape; nda = len(a.shape)
	bs = b.shape; ndb = len(b.shape)
	equal = True
	if (na != nb):
		equal = False
	else:
		for k in xrange(na):
			if as_[axes_a[k]] != bs[axes_b[k]]:
				equal = False
				break
			if axes_a[k] < 0: axes_a[k] += nda
			if axes_b[k] < 0: axes_b[k] += ndb
	if not equal:
		print as_, bs, axes_a, axes_b

		raise ValueError, "shape-mismatch for sum,"
	if a.num_q != b.num_q: raise ValueError, "num_q mismatch"
	if np.any(a.mod_q != b.mod_q): raise ValueError, "mod_q mismatch"
	mod_q = a.mod_q

	for i in range(na):		# na = nb
		ai = a.q_ind[axes_a[i]]
		bi = b.q_ind[axes_b[i]]
		if not np.array_equal(ai[:, 0:1], bi[:, 0:1]):
			print "npc.tensordot_compat error:"
			print joinstr(["a.qind[%s] = " % axes_a[i], ai])
			print joinstr(["b.qind[%s] = " % axes_b[i], bi])
			raise ValueError, "q_ind size mismatch"
		q_diff = mod_onetoinf2D( ai[:, 2:] * a.q_conj[axes_a[i]] + bi[:, 2:] * b.q_conj[axes_b[i]], mod_q )
		if np.any(q_diff != 0):
			print "npc.tensordot_compat error:"
			print "  mod_q = %s" % (mod_q,)
			print joinstr([ "  a.qind[%s] = " % axes_a[i], ai, ", q_conj = %i" % a.q_conj[axes_a[i]] ])
			print joinstr([ "  b.qind[%s] = " % axes_b[i], bi, ", q_conj = %i" % b.q_conj[axes_b[i]] ])
			raise ValueError, "q_ind charges mismatch"



#@profile
def tensordot(a, b, axes = 2, verbose = 0, timing = False):
	"""	Should behave identically to np.tensordot
	
		TODO - for full contraction equivalent to inner, should it return scalar or 0-d array? 
	"""
	# TODO, add description
	
	#tensordot_compat(a, b, axes)
	return npc_helper.tensordot2(a, b,axes)
	

def mixed_tensordot(a, b, axes = 2, verbose = 0, timing = False):
	"""	a is npc.array, b is numpy.array, returns a numpy.array
	
		TODO does this dumbo way (ie converts a to np.array)
	"""
	try:
		iter(axes)
	except:
		axes_a = range(-axes,0)
		axes_b = range(0,axes)
	else:
		axes_a, axes_b = axes
	try:
		na = len(axes_a)
		axes_a = list(axes_a)
	except TypeError:
		axes_a = [axes_a]
		na = 1
	try:
		nb = len(axes_b)
		axes_b = list(axes_b)
	except TypeError:
		axes_b = [axes_b]
		nb = 1
	# axes_a,b are now each list (same len) of indices to contract (na,b are the number of indices)

	if isinstance(b, array) and isinstance(a, np.ndarray):
		# swap a and b
		swap_ab = True
		# TODO, do something!!!!!

	as_ = a.shape; nda = len(a.shape)
	bs = b.shape; ndb = len(b.shape)
	equal = True
	in_dim = 1
	if (na != nb):
		equal = False
	else:
		for k in xrange(na):
			in_dim *= as_[axes_a[k]]
			if as_[axes_a[k]] != bs[axes_b[k]]:
				equal = False
				break
			if axes_a[k] < 0: axes_a[k] += nda
			if axes_b[k] < 0: axes_b[k] += ndb
	if not equal:
		raise ValueError, "shape-mismatch for sum"


	# TODO, we need some real guts here
	if not swap_ab:
		return np.tensordot(a.to_ndarray(), b, axes)
	else:
		return np.tensordot(a, b.to_ndarray(), axes)


def trace(a, axis1=0, axis2=1):
	"""	Trace of a along axis1/2.
		Currently requires that the contracted legs have the same q_ind and opposite charge.
		If a has only two legs, returns a scalar, otherwise the resultant array.
		"""
	if a.labels is not None:
		axis1 = a.get_index(axis1)
		axis2 = a.get_index(axis2)
	
	if a.shape[axis1]!= a.shape[axis2]:
		raise ValueError, "shape mismatch for traced legs"
	if (np.array_equiv(a.q_ind[axis1], a.q_ind[axis2]) == False) or a.q_conj[axis1] != -a.q_conj[axis2]:
		raise NotImplementedError, "Trace only supports charge-conjugate index structure at the moment"
		
	#mask = np.logical_and( np.not_equal( np.arange(a.rank), axis1), np.not_equal( np.arange(a.rank), axis2) ) #True where legs survive trace
	mask = np.ones(a.rank, np.bool)
	mask[axis1] = False
	mask[axis2] = False
	
	if a.rank > 2:
		c = array(a.rank - 2, dtype = a.dtype)
		c.num_q = a.num_q
		c.mod_q = a.mod_q
		c.charge = a.charge.copy() #No need to copy?
		c.q_conj = a.q_conj[mask]
		c.q_ind = [a.q_ind[i] for i in range(a.rank) if mask[i] ]
		c.sorted = a.sorted
		c.shape = np.array( [a.shape[i] for i in range(a.rank) if mask[i] ], dtype = np.intp )
		#print a.q_dat
		if len(a.dat) > 0: 	
			cdat = {}
			for q, t in itertools.izip(a.q_dat, a.dat):
				if np.array_equiv(q[axis1], q[axis2]): #Are they on the "diagonal" ?
					if tuple(q[mask]) in cdat:
						cdat[tuple(q[mask])] += np.trace(t, axis1 = axis1, axis2 = axis2)
					else:
						cdat[tuple(q[mask])] = np.trace(t, axis1 = axis1, axis2 = axis2)
			
			c.dat = cdat.values()
			if len(c.dat) > 0:
				#np.array(k) for 
				c.q_dat = np.array(cdat.keys(), np.uint)
			else:
				c.q_dat = np.empty( (0, c.rank), np.uint)
		else:
			c.dat = []
			c.q_dat = np.empty( (0, c.rank), np.uint)

		#inherit labels
		if a.labels is not None:
			c.labels = {k:v for k, v in a.labels.iteritems() if (v!=axis1 and v!=axis2)}
			for k, v in c.labels.iteritems():
				if v > axis1:
					c.labels[k]-=1
				if v > axis2:
					c.labels[k]-=1
		else:
			c.labels = None


	else:
		c = np.zeros(1, a.dtype)[0]
		for q, t in itertools.izip(a.q_dat, a.dat):
			if q[axis1] == q[axis2]:
				c += np.trace(t, axis1 = axis1, axis2 = axis2)

	return c
		

		
def inner(a, b, axes = None, do_conj = False, verbose = 0, timing = False): 
	"""	Contracts all the legs in a and b and return a scalar.

		If axes is None, contracts all axes in matching order
		Otherwise, axes = [ [i1, i2 , . . . ] , [j1, j2, . . . ] ],
		 	contracting i1j1, i2j2, etc.
		
		If do_conj = True, a is conjugated before contraction (giving hermitian inner product)
		"""
	
	if timing:
		t0 = time.time()
	
	if id(a) == id(b):
		#raise NotImplementedError, "Use npc.norm for identical arrays"
		print "making copy for inner"
		b = a.copy()
		
	if axes is None:
		axes_a = range(a.rank)
		axes_b = range(b.rank)
	else:
		axes_a = list(axes[0])
		axes_b = list(axes[1])
	
	nda = len(axes_a)
	ndb = len(axes_b)
	if nda!=a.rank or ndb!=b.rank:
		raise ValueError, "Incomplete contraction requested."
	if nda!=ndb:
		raise ValueError, "Tensors to be contracted have different dimensions."
		
	nd = nda
	# axes_a,b are now each list (same len) of indices to contract (na,b are the number of indices)

	as_ = a.shape
	bs = b.shape

	equal = True
	for k in xrange(nd):
		if as_[axes_a[k]] != bs[axes_b[k]]:
			equal = False
			break
		if axes_a[k] < 0: axes_a[k] += nd
		if axes_b[k] < 0: axes_b[k] += nd
	if not equal:
		print "Shapes:", [as_[axes_a[k]] for k in range(nd)], [bs[axes_b[k]] for k in range(nd)]
		raise ValueError, "Shape-mismatch for inner."


	if len(a.q_dat) == 0 or len(b.q_dat) == 0:
		return 0
		
	if axes_a != range(nd):
		a.itranspose(axes_a)
	if axes_b != range(nd):
		b.itranspose(axes_b)

	if (do_conj and npc_helper.array_equiv_mod_q(a.charge, b.charge, a.mod_q) == False) or (do_conj==0 and npc_helper.array_equiv_mod_q(-a.charge, b.charge, a.mod_q) == False):
		return 0.

	num_q = a.num_q

	a.sort_q_dat()
	b.sort_q_dat()
	
	i, j = 0, 0
	
	Na = len(a.q_dat)
	Nb = len(b.q_dat)
	c = 0 
	aq = a.q_dat
	bq = b.q_dat
	adat = a.dat
	bdat = b.dat
	ndim = bq.shape[1]
	if verbose > 2:
		print a.q_dat
		print b.q_dat
	
	if do_conj and a.dtype==np.complex:
		while i < Na and j < Nb:
			comp = npc_helper.lex_comp_gt(bq[j], aq[i], 0, ndim)
			if comp==1:
				i+=1
			elif comp==-1:
				j+=1
			else:
				c+= np.vdot( adat[i], bdat[j])
				i+=1
				j+=1
	else:
		while i < Na and j < Nb:
			comp = npc_helper.lex_comp_gt(bq[j], aq[i], 0, ndim)
			if comp==1:
				i+=1
			elif comp==-1:
				j+=1
			else:
				c+=np.inner( adat[i].reshape((-1,)), bdat[j].reshape((-1,)) )
				i+=1
				j+=1
	
		# undo the terrible mess created
	if axes_a != range(nd):
		a.itranspose(np.argsort(axes_a))
	if axes_b != range(nd):
		b.itranspose(np.argsort(axes_b))
		
	return c


def pinv(a, rcond=1e-15):
	""" 
	Compute the (Moore-Penrose) pseudo-inverse of a matrix.

	Calculate the generalized inverse of a matrix using its singular-value decomposition (SVD) and including all large singular values.
	
	For convenience, if a is an ndim = 1 np.ndarray, takes pseudo-inverse of vector
	
	Parameters :	
		a : (M, N) npc.array, or (M, ) np.ndarray		
			Matrix to be pseudo-inverted.
		rcond : float
			Cutoff for small singular values. Singular values smaller (in modulus) than rcond * largest_singular_value (again, in modulus) are set to zero.
	Returns :	
		B : (N, M) npc.array or (M, ) np.ndarray
			The pseudo-inverse of a.
		Labeling: for a_xy, b_gh obeys g = y, h = x
		
	Raises :	
		LinAlgError :
			If the SVD computation does not converge.

	"""
	
	if isinstance(a, np.ndarray):
		if a.ndim == 1:
			if rcond!=0.:
				s_max = np.max(np.abs(a))
				rcond*=s_max #This is convention form numpy!
				return np.choose(np.abs(a) < rcond, [a**(-1), np.zeros_like(a)])	# take inverse
			else:
				return a**(-1)				
		else:
			raise ValueError, a.shape

			
	if a.rank!=2:
		raise ValueError
		
	is_blocked = a.is_blocked_by_charge()
	
	pipes = [None, None]
	for i in range(2):
		if not is_blocked[i]:
			pipes[i] = a.make_pipe([i], block_single_legs=True)

	is_blocked = is_blocked[0] and is_blocked[1]

	if is_blocked == False:
		a = a.combine_legs([[0], [1]], pipes) #executes permutation

	#v (the result) has charge structure of a.transpose.conj
	#So temporarily gut a
	dat = a.dat #cache
	a.dat = [] #gut it
	v = a.transpose() #copy charge structure
	v.charge = mod_onetoinf1D(-v.charge, v.mod_q)
	v.q_conj*=-1
	a.dat = dat #repair damage

	for t in a.dat: #For each non-zero entry
		v.dat.append( np.linalg.pinv(t, rcond))
				
	if is_blocked == False:
		v = v.split_legs([0, 1], pipes[::-1]) #The 'outer' facing legs are permuted back. Reverse order of pipes!


	return v


def svd(a, full_matrices=0, compute_uv=1, overwrite_a = False, cutoff = None, chargeR = None, inner_labels = [None, None]):
	"""	U, S, V = svd(a) ----> a = U diag(S) V
		U and V are npc.arrays, S is a normal numpy.array
		
		By default, assigns Q = 0 to V, otherwise chargeR (with U set by charge conservation)
		 
		*Assumes full charge blocking*	
		
		Labeling: for U_rx, V_yc  r, c inherited and x, y passed as inner_labels
	"""
	if full_matrices:
		raise NotImplementedError, "Full_matrix SVD not implemented."

	if a.rank != 2:
		raise ValueError, "SVD requires 2-dim current view"
	
	is_blocked = a.is_blocked_by_charge()
	
	pipes = [None, None]
	for i in range(2):
		if not is_blocked[i]:
			pipes[i] = a.make_pipe([i], block_single_legs=True)

	is_blocked = is_blocked[0] and is_blocked[1]
	
	if is_blocked == False:
		a = a.combine_legs([[0], [1]], pipes) #executes permutation
	
	#if is_blocked == False:
	#	raise NotImplementedError, "Must be blocked by charge"
		
	num_q = a.num_q	
	M, N = a.shape
	K = min(M, N)


	U = array(2, dtype = a.dtype)
	S = np.zeros(K, dtype = np.float)
	V = array(2, dtype = a.dtype)
	U.num_q = V.num_q = num_q
	U.mod_q = V.mod_q = a.mod_q


	#Find new labels
	U.labels = U_labels = {}
	V.labels = V_labels = {}

	if a.labels is not None:
		rev = { v:k for k, v in a.labels.iteritems()}
		if 0 in rev:
			U_labels[rev[0]] = 0
		if 1 in rev:
			V_labels[rev[1]] = 1

	if inner_labels[0] is not None:
		U_labels[inner_labels[0]] = 1
	if inner_labels[1] is not None:
		V_labels[inner_labels[1]] = 0
	
	if len(U_labels)>0:
		U.labels = U_labels
	if len(V_labels)>0:
		V.labels = V_labels
	
	if chargeR is None:
		chargeR = np.zeros(num_q, dtype = np.int)
	elif type(chargeR) == int:
		if num_q ==1:
			chargeR = np.array([chargeR], dtype = np.int)
		else:
			raise ValueError
	U.charge = mod_onetoinf1D(a.charge - chargeR, U.mod_q)
	V.charge = chargeR
	
	U.q_conj = np.array([a.q_conj[0], 0])
	V.q_conj = np.array([0, a.q_conj[1]])
	
	at = 0

	ql = np.empty(2 + num_q, dtype = np.int)
	q_il= []

	i = 0
	U.q_dat = []
	V.q_dat = []
	for q, t in itertools.izip(a.q_dat, a.dat): #TODO The order here determines q_ind, pick convention?

		try:
			if compute_uv:
				U_block, S_block, V_block = linalg.svd(t, full_matrices = 0, compute_uv = 1, overwrite_a = overwrite_a )
				
				#eps = np.dot(np.dot(U_block.T, t), V_block.T)
				#eps = eps*(S_block**(-0.5))
				#eps  = ((eps.T)*(S_block**(-0.5))).T - np.eye(len(S_block))
				#print np.linalg.norm(eps),
				if anynan(U_block):
					print "U_block nansum(s):", np.nansum(U_block),
					raise sp.linalg.LinAlgError
				if anynan(V_block):
					print "V_block nansum(s):", np.nansum(V_block),
					raise sp.linalg.LinAlgError
				
			else:
				S_block = linalg.svd(t, full_matrices = 0, compute_uv = 0, overwrite_a = overwrite_a)
		##	Check for nan's
			if anynan(S_block):
				print "S_block nansum(s):", np.nansum(S_block),
				raise sp.linalg.LinAlgError

		except ValueError:
			if anynan(t):
				raise ValueError, "Nan in block"
			else:
				raise
		except sp.linalg.LinAlgError:
			
			if t.dtype != np.complex128 :

				print "[Appears SVD did not converge! Trying backup dGESVD.]"
				if compute_uv:
					U_block, S_block, V_block = svd_dgesvd.svd_dgesvd(t, full_matrices = 0, compute_uv = 1)
				else:
					S_block = svd_dgesvd.svd_dgesvd(t, full_matrices = 0, compute_uv = 0)

			else:
				print "[Appears SVD did not converge! Trying backup zGESVD for complex numbers.]"
		
				if compute_uv:
					U_block, S_block, V_block = svd_zgesvd.svd_zgesvd(t, full_matrices = 0, compute_uv = 1)
				else:
					S_block = svd_dgesvd.svd_zgesvd(t, full_matrices = 0, compute_uv = 0)

			
		if cutoff is not None: #in case we want to discard tiny values
			cut = S_block > cutoff		# a boolean array
			
			S_block = S_block[cut]
			
			if compute_uv:
				U_block = U_block[:, cut]
				V_block = V_block[cut, :]

		num = len(S_block)
		if num > 0:
			S[at:at + num ] = S_block
			if compute_uv:
				U.dat.append(U_block)
				V.dat.append(V_block)
				

				#make new q_ind entry using RIGHT q_ind
				ql[0] = at
				ql[1]  = at + len(S_block)
				ql[2:] = a.q_ind[1][ q[1], 2: ] - chargeR*a.q_conj[1]
				q_il.append(ql.copy())
				
				#q_dats entries of U, V identical to a (except for dropped sectors)
				U.q_dat.append( np.array([q[0], i]))
				V.q_dat.append( np.array([i, q[1]]))
				i+=1

			at += num
			
	
	if at == 0:
		raise RuntimeError, (0, "SVD found no singular eigenvalues", a)
		#TODO implement default behavior?

	if compute_uv:
		q_il = np.array(q_il)
		U.q_ind = [ a.q_ind[0], q_il ]
		U.q_conj[1] = a.q_conj[1]
		
		q_ir = q_il.copy()
		V.q_ind = [ q_ir,  a.q_ind[1] ]
		V.q_conj[0] = -a.q_conj[1]
		 
		U.q_dat = np.array(U.q_dat, dtype = np.uint)
		V.q_dat = np.array(V.q_dat, dtype = np.uint)
		
		U.shape = np.array([M, at], dtype = np.intp) #SHAPE
		V.shape = np.array([at, N], dtype = np.intp) #SHAPE
		
		U.sorted = False #TODO technically could be sorted if a is
		V.sorted = False

		if is_blocked == False:
			U = U.split_legs([0],pipes[0:1]) #The 'outer' facing leg is permuted back.
			V = V.split_legs([1],pipes[1:2])
		return U, S[0:at], V
		
	else:
		return S[0:at]

def kron(a, b):
	"""	Kronecker product, the charges are added.
		"""
	if a.rank != b.rank: raise ValueError
	if a.num_q != b.num_q: raise ValueError
	if np.any(a.mod_q != b.mod_q): raise ValueError

	raise NotImplemented
	

#@profile
def eigh(a, UPLO='L', sort = None, debug = False):
	"""
		w, v = eigh(a) ----> a = v diag(w) v^d
		
		Sort options. In each charge sector, eigenvalues can be sorted by:
		'm>': Largest magnitude first
		'm<': Smallest magnitude first
		'>' : Largest algebraic first
		'<' : Smallest algebraic first 	
		'None' : whatever np defaults to
		
		Requires that q_ind[0] is compatible with q_ind[1]
		
		If a is not blocked by charge, a blocked copy is made via a permutation P,
		
			a' = P a P = v' w (v')^d
		
		The "outer" legs of the resulting unitarity are then permuted back,	
			
			v = P^(-1) v'
			a = v w v^d
		
		The resulting 'v' is of course unitary, but q_ind[0]!=q_ind[1], as they are related by permutation.
	"""
	if len(a.shape)!=2 or a.shape[0]!=a.shape[1]: raise ValueError, "npc.eigh():  Non-square matrix with shape = {}.".format(a.shape)
	assert q_ind_contractible(a, 0, a, 1, uncontractible_error_msg="npc.eigh():  Matrix legs mismatch.")
		
	is_blocked = npc_helper.is_blocked_by_charge(a.q_ind[0])
	
	if is_blocked == False: #Make copy of a which is blocked by charge, and save permutation in 'pipe'
		pipe = a.make_pipe([0], block_single_legs=True)
		a = a.combine_legs([[0], [1]], pipes = [pipe, pipe]) #executes permutation
	
	num_q = a.num_q	
	resw = np.zeros(a.shape[0], dtype = np.float)
	resv = diag(np.array(1., dtype = np.promote_types(a.dtype, np.float)), a.q_ind[0], q_conj = a.q_conj[0], mod_q = a.mod_q)
	#w, v now default to 0 and the Identity
	
	for q, t in itertools.izip(a.q_dat, a.dat): #For each non-zero entry
		
		try:
			rw, rv = np.linalg.eigh( t, UPLO )
		except np.linalg.linalg.LinAlgError as err:
			print err.args[0]
			print "Shape, norm:", t.shape, np.linalg.norm(t)
			print t
			raise np.linalg.linalg.LinAlgError, err.args
		rw = rw.real
		if sort is not None: #Apply sorting options
			if sort == 'm>':
				piv = np.argsort( -np.abs(rw) )
			elif sort =='m<':
				piv = np.argsort( np.abs(rw) )
			elif sort == '>':
				piv = np.argsort( -rw )
			elif sort == '<':
				piv = np.argsort( rw )
			else:
				raise ValueError, "Unrecognized eigh sorting option"
			rw = np.take( rw, piv)
			rv = np.take( rv, piv, axis = 1)

		resv.dat[q[0]] = rv #By construction, resv is sorted and has all entries, 
		resw[a.q_ind[0][q[0], 0]: a.q_ind[0][q[0], 1]] = rw
	
	if is_blocked == False:
		resv = resv.split_legs([0],[pipe]) #The 'outer' facing leg is permuted back.
		
	return resw, resv


def eig(a, sort = None, debug = False):
	"""
		w, v = eig(a) ----> a.v[:, i] = w[i] v 
		
		Sort options. In each charge sector, eigenvalues can be sorted by:
		'm>': Largest magnitude first
		'm<': Smallest magnitude first
		'None' : whatever np defaults to
		
		Requires that q_ind[0] is compatible with q_ind[1]
		
		If a is not blocked by charge, a blocked copy is made via a permutation P,
		
			a' = P a P = v' w (v')^(-1)
		
		The "outer" legs of the resulting unitarity are then permuted back,	
			
			v = P^(-1) v'
			a = v w v^(-1)
		
		The resulting 'v' is of course invertible, but q_ind[0]!=q_ind[1], as they are related by permutation.
	"""

	if len(a.shape)!=2 or a.shape[0]!=a.shape[1]: raise ValueError, "npc.eig():  Non-square matrix with shape = {}.".format(a.shape)
	assert q_ind_contractible(a, 0, a, 1, uncontractible_error_msg="npc.eig():  Matrix legs mismatch.")
		
	is_blocked = npc_helper.is_blocked_by_charge(a.q_ind[0])
	
	if is_blocked == False: #Make copy of a which is blocked by charge, and save permutation in 'pipe'
		pipe = a.make_pipe([0], block_single_legs=True)
		a = a.combine_legs([[0], [1]], pipes = [pipe, pipe]) #executes permutation
	
	num_q = a.num_q	
	resw = np.zeros(a.shape[0], dtype = np.complex)
	resv = diag(np.array(1., dtype = np.promote_types(a.dtype, np.complex)), a.q_ind[0], q_conj = a.q_conj[0], mod_q = a.mod_q)
	#w, v now default to 0 and the Identity
	
	for q, t in itertools.izip(a.q_dat, a.dat): #For each non-zero entry
		
		try:
			rw, rv = np.linalg.eig( t )
		except np.linalg.linalg.LinAlgError as err:
			print err.args[0]
			print "Shape, norm:", t.shape, np.linalg.norm(t)
			print t
			raise np.linalg.linalg.LinAlgError, err.args
		if sort is not None: #Apply sorting options
			if sort == 'm>':
				piv = np.argsort( -np.abs(rw) )
			elif sort =='m<':
				piv = np.argsort( np.abs(rw) )
			else:
				raise ValueError, "Unrecognized eig sorting option"
			rw = np.take( rw, piv)
			rv = np.take( rv, piv, axis = 1)

		resv.dat[q[0]] = rv #By construction, resv is sorted and has all entries, 
		resw[a.q_ind[0][q[0], 0]: a.q_ind[0][q[0], 1]] = rw
	
	if is_blocked == False:
		resv = resv.split_legs([0],[pipe]) #The 'outer' facing leg is permuted back.
		
	return resw, resv



def eigvalsh(a, UPLO='L', sort = None, debug = False):
	"""
		w = eigvalsh(a) ----> eigenspectrum of a (Hermitian)
		
		Sort options. In each charge sector, eigenvalues can be sorted by:
		'm>': Largest magnitude first
		'm<': Smallest magnitude first
		'>' : Largest algebraic first
		'<' : Smallest algebraic first 	
		'None' : whatever np defaults to
		
		Requires that q_ind[0] is compatible with q_ind[1]
	"""
	if len(a.shape)!=2 or a.shape[0]!=a.shape[1]: raise ValueError, "npc.eigvalsh():  Non-square matrix with shape = {}.".format(a.shape)
	assert q_ind_contractible(a, 0, a, 1, uncontractible_error_msg="npc.eigvalsh():  Matrix legs mismatch.")
		
	is_blocked = npc_helper.is_blocked_by_charge(a.q_ind[0])
	
	if is_blocked == False: #Make copy of a which is blocked by charge, and save permutation in 'pipe'
		pipe = a.make_pipe([0], block_single_legs=True)
		a = a.combine_legs([[0], [1]], pipes = [pipe, pipe]) #executes permutation
	
	num_q = a.num_q	
	resw = np.zeros(a.shape[0], dtype = np.float)

	#w now default to 0 
	
	for q, t in itertools.izip(a.q_dat, a.dat): #For each non-zero entry
		
		try:
			rw = np.linalg.eigvalsh( t, UPLO )
		except np.linalg.linalg.LinAlgError as err:
			print err.args[0]
			print "Shape, norm:", t.shape, np.linalg.norm(t)
			print t
			raise np.linalg.linalg.LinAlgError, err.args
		rw = rw.real
		if sort is not None: #Apply sorting options
			if sort == 'm>':
				piv = np.argsort( -np.abs(rw) )
			elif sort =='m<':
				piv = np.argsort( np.abs(rw) )
			elif sort == '>':
				piv = np.argsort( -rw )
			elif sort == '<':
				piv = np.argsort( rw )
			else:
				raise ValueError, "Unrecognized eigh sorting option"
			rw = np.take( rw, piv)

		resw[a.q_ind[0][q[0], 0]: a.q_ind[0][q[0], 1]] = rw
	
	
	return resw, a.q_ind[0]


def eigvals(a, sort = None, debug = False):
	"""
		w = eigvals(a) ----> eigenspectrum of a 
		
		Sort options. In each charge sector, eigenvalues can be sorted by:
		'm>': Largest magnitude first
		'm<': Smallest magnitude first
		'None' : whatever np defaults to
		
		Requires that q_ind[0] is compatible with q_ind[1]
	"""
	if len(a.shape)!=2 or a.shape[0]!=a.shape[1]: raise ValueError, "npc.eigvals():  Non-square matrix with shape = {}.".format(a.shape)
	assert q_ind_contractible(a, 0, a, 1, uncontractible_error_msg="npc.eigvals():  Matrix legs mismatch.")
		
	is_blocked = npc_helper.is_blocked_by_charge(a.q_ind[0])
	
	if is_blocked == False: #Make copy of a which is blocked by charge, and save permutation in 'pipe'
		pipe = a.make_pipe([0], block_single_legs=True)
		a = a.combine_legs([[0], [1]], pipes = [pipe, pipe]) #executes permutation
	
	num_q = a.num_q	
	resw = np.zeros(a.shape[0], dtype = np.complex)

	#w now default to 0 
	
	for q, t in itertools.izip(a.q_dat, a.dat): #For each non-zero entry
		
		try:
			rw = np.linalg.eigvals( t )
		except np.linalg.linalg.LinAlgError as err:
			print err.args[0]
			print "Shape, norm:", t.shape, np.linalg.norm(t)
			print t
			raise np.linalg.linalg.LinAlgError, err.args

		if sort is not None: #Apply sorting options
			if sort == 'm>':
				piv = np.argsort( -np.abs(rw) )
			elif sort =='m<':
				piv = np.argsort( np.abs(rw) )
			else:
				raise ValueError, "Unrecognized eigh sorting option"
			rw = np.take( rw, piv)

		resw[a.q_ind[0][q[0], 0]: a.q_ind[0][q[0], 1]] = rw
	
	
	return resw, a.q_ind[0]


def speigs(a, charge_sector, k=6, M=None, sigma=None, which='LM', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=True, Minv=None, OPinv=None, OPpart=None):
	"""Sparse eigenvalue decomposition w, v of square npc.array A:
		A.v[i] = w[i] v[i]
		
		Finds k right eigenvectors (chosen by 'which') in charge block defined by charge_sector;
		charge_sector is the total charge of the returned vector.
		
		Returns w, v. 
			w is a np.array
			v is a list of npc.array (note that when interpreted as a matrix, this is the transpose of what np.eigs normally gives.)
		"""
	charge_sector = np.array(charge_sector, dtype = np.int)
	if len(charge_sector)!=a.num_q:
		raise ValueError
	if a.rank != 2 or a.shape[0] != a.shape[1]:
		raise ValueError, "Non-square matrix for eigh, aborting"
	assert q_ind_contractible(a, 0, a, 1, uncontractible_error_msg="npc.speigs():  Matrix legs mismatch.")
		
	is_blocked = npc_helper.is_blocked_by_charge(a.q_ind[0])
	
	if is_blocked == False: #Make copy of a which is blocked by charge, and save permutation in 'pipe'
		pipe = a.make_pipe([0], block_single_legs=True)
		a = a.combine_legs([[0], [1]], pipes = [pipe, pipe]) #executes permutation

	found = False
	for q, t in itertools.izip(a.q_dat, a.dat): #Find relevant entry
		charge = a.q_ind[0][q[0], 2:] * a.q_conj[0]
		want = array_equiv_mod_q(charge, charge_sector, a.mod_q)
		if not want: continue

		if k > t.shape[0]:
			print "npc.speigs():  k > size, trimming k."
			k = t.shape[0]
		w, np_v = sp_speigs(t, k, M, sigma, which, v0, ncv, maxiter, tol, return_eigenvectors, Minv, OPinv, OPpart)

		##	turn into npc
		v = []
		for j in range(np_v.shape[1]):
			u = zeros( a.q_ind[0:1], np_v.dtype, a.q_conj[0:1], charge_sector, a.mod_q )
			u.dat = [np_v[:, j]]
			u.q_dat = np.array([[q[0]]], dtype = np.uint)
			v.append(u)
		found = True
		break

	#TODO - should actually check if charge sector exists, and return appropriate null eigensystem if present or raise error if not.
	if not found:
		print "Warning, requested sector is absent, returning None. (TODO)"
		return None

	return w, v

