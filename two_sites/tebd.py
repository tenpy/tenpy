import numpy as np
import scipy as sp
from scipy import linalg

def group_legs(a, axes):
	""" Given list of lists like axes = [ [l1, l2], [l3], [l4 . . . ]]
	
		does a transposition of "a" according to l1 l2 l3... followed by a reshape according to parantheses.
		
		Return the reformed tensor along with a "pipe" which can be used to undo the move
	"""

	nums = [len(k) for k in axes]


	flat = []
	for ax in axes:
		flat.extend(ax)

	a = np.transpose(a, flat)
	perm = np.argsort(flat)

	oldshape = a.shape

	shape = []
	oldshape = []
	m = 0
	for n in nums:
		shape.append(np.prod(a.shape[m:m+n]))
		oldshape.append(a.shape[m:m+n])
		m+=n
	
	a = np.reshape(a, shape)
	
	pipe = (oldshape, perm)

	return a, pipe

def ungroup_legs(a, pipe):
	"""
		Given the output of group_legs,  recovers the original tensor (inverse operation)
		
		For any singleton grouping [l],  allows the dimension to have changed (the new dim is inferred from 'a').
	"""
	if a.ndim!=len(pipe[0]):
		raise ValueError
	shape = []
	for j in range(a.ndim):
		if len(pipe[0][j])==1:
			shape.append(a.shape[j])
		else:
			shape.extend(pipe[0][j])

	a = a.reshape(shape)
	a = a.transpose(pipe[1])
	return a

def transpose_mpo(Psi):
	"""Transpose row / column of an MPO"""
	return [ b.transpose([1, 0, 2, 3]) for b in Psi]

def mps_group_legs(Psi, axes = 'all'):
	""" Given an 'MPS' with a higher number of physical legs (say, 2 or 3), with B tensors

			physical leg_1 x physical leg_2 x . . . x virtual_left x virtual_right
			
		groups the physical legs according to axes = [ [l1, l2], [l3], . .. ] etc, 
		
		Example:
		
		
			Psi-rank 2,	axes = [[0, 1]]  will take MPO--> MPS
			Psi-rank 2, axes = [[1], [0]] will transpose MPO
			Psi-rank 3, axes = [[0], [1, 2]] will take to MPO
		
		If axes = 'all', groups all of them together.
		
		Returns:
			Psi
			pipes: list which will undo operation
	"""
	
	if axes == 'all':
		axes = [ range(Psi[0].ndim-2) ]
	
	psi = []
	pipes = []
	for j in range(len(Psi)):
		ndim = Psi[j].ndim
		b, pipe = group_legs( Psi[j], axes + [[ndim-2], [ndim-1]])
		
		psi.append(b)
		pipes.append(pipe)

	return psi, pipes

def mps_ungroup_legs(Psi, pipes):
	"""Inverts mps_group_legs given its output"""
	psi = []
	for j in range(len(Psi)):
		psi.append(ungroup_legs( Psi[j], pipes[j]))
	
	return psi

def mps_invert(Psi):
	np = Psi[0].ndim - 2
	return [ b.transpose(range(np) + [-1, -2]) for b in Psi[::-1] ]

def mpo_invert(Psi):
	np = Psi[0].ndim
	return [ b.transpose([1,0] + range(2,np)) for b in Psi[::-1] ]

def mps_2form(Psi, form = 'A'):
	"""Puts an mps with an arbitrary # of legs into A or B-canonical form
		
		hahaha so clever!!!
	"""
	Psi, pipes = mps_group_legs(Psi, axes='all')

	if form=='B':
		Psi = [ b.transpose([0, 2, 1]) for b in Psi[::-1] ]
	
	L = len(Psi)
	T = Psi[0]
	for j in range(L-1):
		T, pipe = group_legs(T, [[0, 1], [2]]) #view as matrix
		A, s = np.linalg.qr(T) #T = A s can be given from QR
		Psi[j] = ungroup_legs(A, pipe)
		T = np.tensordot( s, Psi[j+1], axes = [[1], [1]]).transpose([1, 0, 2]) #Absorb s into next tensor

	Psi[L-1] = T

	if form=='B':
		Psi = [ b.transpose([0, 2, 1]) for b in Psi[::-1] ]
	
	Psi =  mps_ungroup_legs(Psi, pipes)

	return Psi

def peel(Psi, p):
	""" Put Psi into B-canonical form, and reshape the physical legs to transfer p-dof from right to left
	"""

	D = [ b.shape[:2] for b in Psi]
	
	psi = mps_2form(Psi, 'B')
	psi = [b.reshape((d[0]*p, d[1]/p, b.shape[2], b.shape[3])) for b,d  in izip(psi, D)]

	return psi
def mps_entanglement_spectrum(Psi):
	
	Psi, pipes = mps_group_legs(Psi, axes='all')
	#First bring to A-form
	L = len(Psi)
	T = Psi[0]
	for j in range(L-1):
		T, pipe = group_legs(T, [[0, 1], [2]]) #view as matrix
		A, s = np.linalg.qr(T) #T = A s can be given from QR
		Psi[j] = ungroup_legs(A, pipe)
		T = np.tensordot( s, Psi[j+1], axes = [[1], [1]]).transpose([1, 0, 2]) #Absorb s into next tensor
	Psi[L-1] = T

	#Flip the MPS around
	Psi = [ b.transpose([0, 2, 1]) for b in Psi[::-1] ]

	T = Psi[0]
	Ss = []
	for j in range(L-1):
		T, pipe = group_legs(T, [[0, 1], [2]]) #view as matrix
		U, s, V = np.linalg.svd(T, full_matrices=False) 
		Ss.append(s)
		Psi[j] = ungroup_legs(U, pipe)
		s = ((V.T)*s).T
		T = np.tensordot( s, Psi[j+1], axes = [[1], [1]]).transpose([1, 0, 2]) #Absorb sV into next tensor

	return Ss[::-1]

def mpo_on_mpo(X, Y, form = None):
	""" Multiplies two two-sided MPS, XY = X*Y and optionally puts in a canonical form
	"""
	if X[0].ndim!=4 or Y[0].ndim!=4:
		raise ValueError
	
	XY = [  group_legs( np.tensordot(x, y, axes = [[1], [0]]), [[0], [3], [1, 4], [2, 5]] )[0] for x, y in izip(X, Y)]
	
	if form is not None:
		XY = mps_2form(XY, form)
	
	return XY

def svd_theta(theta, truncation_par,return_XYZ=None):
	""" SVD and truncate a matrix based on truncation_par """
	
	U, s, V = np.linalg.svd( theta, compute_uv=True, full_matrices=False)
	s[np.abs(s)<10e-14] = 0.
	nrm = np.linalg.norm(s)
	eta = np.min([ np.count_nonzero((1 - np.cumsum(s**2)/nrm**2) > truncation_par['p_trunc'])+1, truncation_par['chi_max']])
	nrm_t = np.linalg.norm(s[:eta])

	if return_XYZ:
		Y = s[:eta]/nrm_t; 
		X = U[:,:eta] 
		Z = V[:eta,:]
	
		info = {'p_trunc': 1 - (nrm_t/nrm)**2, 's': s[:eta], 'nrm':nrm}
		return X,Y,Z,info
		
	A = U[:, :eta]
	SB = ((V[:eta, :].T)*s[:eta]/nrm_t).T
	
	info = {'p_trunc': 1 - (nrm_t/nrm)**2, 's': s[:eta], 'nrm':nrm}
	return A, SB, info

def mpo_to_full(H_mpo_list):
	D = H_mpo_list[0].shape[0]
	vL = np.zeros(D)
	vL[0] = 1.
	vR = np.zeros(D)
	vR[D-1] = 1.
	
	L = len(H_mpo_list)
	d =  H_mpo_list[0].shape[2]

	H_full = np.tensordot(vL,H_mpo_list[0],axes=(0,0))
	for i in range(0,L-1):
		H_full = np.tensordot(H_full,H_mpo_list[i+1],axes=(2*i,0))

	H_full = np.tensordot(H_full,vR,axes=(L*2-2,0))
	H_full=np.transpose(H_full,np.hstack([np.arange(0,L*2,2),np.arange(0,L*2,2)+1]))
	H_full=np.reshape(H_full,[d**(L),d**(L)])
	
	return H_full	

def site_expectation_value(Psi,s,O):
	L=len(Psi)
	expectation_O = []
	for j in range(L):

		theta = np.tensordot(np.diag(s[j]),Psi[j], axes = [1,1]) # a,i,b
		theta = theta.transpose(1,0,2) # i,a,b
		expectation_O.append(np.tensordot(np.conj(theta), np.tensordot(O[j], theta, axes = (1,0)), axes = ([0,1,2],[0,1,2])))
	
	return expectation_O

def mpo_expectation_value(Psi,W):
	L = len(Psi)
	if not hasattr(W, "__getitem__"):
		W = L*[W]
	
	D = W[0].shape[0]
	Rp = np.zeros([1,1,D],dtype=float)
	Rp[0,0,D-1] = 1.

	D = W[-1].shape[1]	
	Lp = np.zeros([1,1,D],dtype=float)
	Lp[0,0,0] = 1.
	
	for i in np.arange(L-1,-1,-1):
		Rp = np.tensordot(Psi[i], Rp, axes=(2,0))
		Rp = np.tensordot(W[i], Rp, axes=([1,2],[3,0]))            
		Rp = np.tensordot(np.conj(Psi[i]), Rp, axes=([0,2],[1,3]))
		Rp = np.transpose(Rp,(2,0,1))
		
	N = np.ones([1,1])
	for i in np.arange(L):
		N = np.tensordot(N,np.conj(Psi[i]), axes=(1,1))
		N = np.tensordot(N,Psi[i], axes=([0,1],[1,0]))
	N = np.trace(N)
	return np.tensordot(Lp,Rp,axes = ([0,1,2],[0,1,2]))/N
	
def entropy(s):
	return -2*np.vdot(s**2, np.log(s))

def initial_state(L,d,chi,dtype=float, mpstype = 'random_mps',chi0=1):
	B = []
	for i in range(L):
		chi1 = np.min([d**np.min([i,L-i]),chi])
		chi2 = np.min([d**np.min([i+1,L-i-1]),chi])
		if mpstype == 'fm':
			B.append(np.zeros((d,chi1,chi2)));B[-1][1,0,0] = 1.
		elif mpstype == 'afm':
			B.append(np.zeros((d,chi1,chi2)));B[-1][(i+1)%2,0,0] = 1.
		elif mpstype == 'random_product':
			B.append(np.zeros((d,chi1,chi2))); B[-1][:,0,0] = 0.5-np.random.rand(d) + 1j*(0.5-np.random.rand(d))
		elif mpstype == 'random_mps':
			B.append(0.5-np.random.rand(d,chi1,chi2) + 1j*(0.5-np.random.rand(d,chi1,chi2)))
		elif mpstype == 'random_mps_chi0':
			B.append(np.zeros((d,chi1,chi2),dtype=complex))
			chi1 = np.min([chi1,chi0])
			chi2 = np.min([chi2,chi0])
			B[-1][:,:chi1,:chi2] = 0.5-np.random.rand(d,chi1,chi2) + 1j*(0.5-np.random.rand(d,chi1,chi2))
		else:
			print("Not implemented!")
			exit()
	return B

def compress_state(Psi,L,d,chi_max):
	A,s,V = np.linalg.svd(np.reshape(psi,[d,d**(L-1)]),full_matrices=0)

	A_list = []
	chi = np.min([np.sum(s>10.**(-12)), chi_max])
	A_list.append(np.reshape(A[:,:chi],(d,1,chi)))
	for i in range(1,L-1):
		psi = np.tensordot(np.diag(s),V,axes=(1,0))[:chi,:]

		A,s,V = np.linalg.svd(np.reshape(psi,[chi*d,d**(L-i-1)]),full_matrices=0)
		
		A = np.reshape(A,[chi,d,-1])
		chi = np.min([np.sum(s>10.**(-12)), chi_max])
		A_list.append(np.transpose(A[:,:,:chi],(1,0,2)))

	A_list.append(np.reshape(np.transpose(np.tensordot(np.diag(s),V,axes=(1,0))[:chi,:],(1,0)),(d,chi,1)))

	return A_list
