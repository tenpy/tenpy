""" module with all routines to perform specific linear algebra operations"""
from __future__ import division
from . import np_conserved as npc
from . import npc_helper as npc_helper
import scipy as sp
import numpy as np
import time
import math
from ...tools.math import anynan
display_estats = False

if display_estats:
	import matplotlib.pyplot as plt
	import matplotlib.cm as cm

class lin_op:
	""" constructs a generic npc matrix-vector linear operator """
	def __init__(self, M):
		self.M = M

	def matvec(self, v):
		return npc.tensordot(self.M, v, axes = [[1], [0]])

def gram_schmidt(q, rcond = 10**(-14), verbose = 0):
	""" In place gram_schmidt orthogonalization

		Discards vectors of magnitude < rcond after projecting out preceding linear space
	"""
	k = len(q)
	es = []

	r = np.zeros((k, k), dtype = q[0].dtype)

	for j in range(k):
		r[j, j] = e = npc_helper.two_norm(q[j])
		if e > rcond:
			q[j]*=(1./e)
			for i in range(j+1, k):
				r[j, i] = npc_helper.inner(q[j], q[i], do_conj=True)
				npc_helper.iaxpy(-r[j, i], q[j], q[i] )
		else:
			if verbose == 1 : print "Rank defficient", e
			q[j] = None


	#Drop null vectors
	k = 0
	while k < len(q):
		if q[k]==None:
			del q[k]
		else:
			k = k+1
	if k==0:
		return [], r

	if verbose:
		G = np.empty((k, k), dtype = q[0].dtype)
		for i in range(k):
			for j in range(k):
				if i!=j:
					G[i, j] = npc_helper.inner(q[i], q[j], do_conj=True)
				else:
					G[i, j] = npc_helper.two_norm(q[i])**2
		print k, np.diag(r),  np.linalg.norm(G - np.eye(k))

	return q, r

def lanczos(A, psi, LANCZOS_PAR = {'N_min':2, 'N_max':20, 'e_tol':10**(-15), 'p_tol': 0., 'cache_v': 6}, orthogonal_to = [], verbose=0):
	"""
	Frank's Lanczos Algorithm for Lowest Eigenvector

	A - the hermitian operator. Must implement matvec on npc.array
	psi - the starting vector (npc.array)
	LANCZOS_PAR = { N_min, N_max, e_tol }
		Stops if N_max reached, or if energy difference per step < e_tol.

	orthogonal_to = [] A list of vectors (same tensor structure as psi) Lanczos will orthogonalize against, ensuring result is perpendicular to them.

	Returns:
		(E0, psi0, N) - ground state energy, ground state vector, number iterations used.

	I have computed the Ritz residual (RitzRes) according to

	http://web.eecs.utk.edu/~dongarra/etemplates/node103.html#estimate_residual

	Given the gap, the Ritz residual gives a bound on the error in the wavefunction, err <  (RitzRes/gap)^2

	I estimate the gap from the full Lanczos spectrum

	"""

	if len(orthogonal_to) > 0:
		orthogonal_to, r = gram_schmidt(orthogonal_to)

	#Cache stores the last cache_v Lanczos vectors
	cache_v = LANCZOS_PAR.setdefault('cache_v', 6)
	if cache_v<2:
		cache_v = 2
	v = []

	def append_v(psi):
		if len(v) < cache_v:
			v.append(psi)
		else:
			v.pop(0)
			v.append(psi)

	N_min = LANCZOS_PAR['N_min']
	N_max = LANCZOS_PAR['N_max']
	max_Delta_e0 = LANCZOS_PAR.setdefault('e_tol', 1.)
	max_p_err = LANCZOS_PAR.setdefault('p_tol', 1.)
	Delta_e0 = 2.
	p_err=2.
	es = []
	ps = []

	# Lanczos I. Form tridiagonal form of A in the Krylov subspace, stored in T

	psinorm = npc_helper.two_norm(psi)

	#if debug:
	#	assert not np.isnan(psinorm)

	npc_helper.iscal(1./psinorm, psi)
	append_v(psi)

	T = np.zeros([N_max+1,N_max+1],dtype=np.float)

	ULP = 5*10**(-15) #Cutoff for beta

	#Reminder: iaxpy
	# v -= alpha*q1  ---->
	# v.iaxpy(-alpha, q1)

	beta = 1.
	above_ulp = True
	k=0


	while (( (p_err > max_p_err or Delta_e0 > max_Delta_e0) and k < N_max) or (k < N_min)) and above_ulp:

		if k > 0:
			npc_helper.iscal(1./beta, w)
			append_v(w)

		w = v[-1].copy()
		for o in orthogonal_to: #Project out
			overlap = npc_helper.inner(o, w, do_conj=False)
			npc_helper.iaxpy(-overlap, o.conj(), w )
		w = A.matvec(w)
		for o in orthogonal_to[::-1]:
			overlap = npc_helper.inner(o, w, do_conj=False)
			npc_helper.iaxpy(-overlap, o.conj(), w )

		#if debug:
		#	assert not np.isnan(npc_helper.two_norm(w))
		#	assert not np.isnan(npc_helper.two_norm(v[-1]))

		#alpha = np.real(npc.inner(w, v[-1], do_conj = True))
		alpha = np.real(npc_helper.inner(w, v[-1], do_conj = True))
		#if debug:
		#	assert not np.isnan(alpha)
		if k > 0:
			#w.iaxpy(-beta, v[-2])
			npc_helper.iaxpy(-beta, v[-2], w)

		#w.iaxpy(-alpha, v[-1])
		npc_helper.iaxpy(-alpha, v[-1], w)
		#beta = w.norm()
		beta = npc_helper.two_norm(w)
		#if debug:
		#	assert not np.isnan(beta)
		T[k,k]=alpha

		above_ulp = abs(beta) > ULP
		if above_ulp:
			T[k,k+1]=beta
			T[k+1,k]=beta

		k+=1

		# Diag T
		if k==1:
			e_T = [alpha]
		elif k>1:
			e_T,v_T= np.linalg.eigh(T[0:k,0:k])
			piv = np.argsort(e_T)
			e_T = e_T[piv]
			v_T = v_T[:,piv]
			RitzRes =  np.abs(v_T[k-1, 0]*T[k-1, k])
			Delta_e0=(e0_T_old - e_T[0])
			gap = max(e_T[1] - e_T[0], 10**-10)
			p_err = (RitzRes/gap)**2

			#ps.append(p_err)
			#if verbose > 2:
			#	print "dE, Res, gap, p_err", Delta_e0, RitzRes, gap, (RitzRes/gap)**2
			#print 1. - np.abs(np.inner(np.conj(v_T[0:k-1, 0]), v0_T_old))

		es.append(e_T)
		e0_T_old = e_T[0]
		#v0_T_old = v_T[:, 0]

	N=k

	if display_estats and k > 3:
		ks = []
		e_all = []
		for k in range(len(es)):
			ks.extend( [k]*len(es[k]))
			e_all.extend( es[k])
		plt.scatter(ks, e_all)
		plt.show()
	#if k > 1:
	#	print gap

	if verbose:
		if k > 1:
			print N, gap, "|", Delta_e0, max_Delta_e0, "|", p_err, max_p_err
		else:
			print N, alpha, beta

	#Lanczos II. Now that we know the eigenvector's coefficients in the Krylov subspace, construct the actual vector.
	#The last len(v) vectors have been cached, so N-len(v) must be explicitly computed

	if N>1:
		q1 = psi
		psi0 = psi*v_T[0, 0]

		#These vectors are not cached
		for k in range(0, N-len(v)-1):

			w = q1.copy()
			for o in orthogonal_to:
				overlap = npc_helper.inner(o, w, do_conj=False)
				npc_helper.iaxpy(-overlap, o.conj(), w )
			w = A.matvec(w)
			for o in orthogonal_to[::-1]:
				overlap = npc_helper.inner(o, w, do_conj=False)
				npc_helper.iaxpy(-overlap, o.conj(), w )

			if k > 0:
				npc_helper.iaxpy(-beta, q0, w)

			alpha = T[k, k]
			npc_helper.iaxpy(-alpha, q1, w)

			beta = T[k,k+1]
			npc_helper.iscal(1./beta, w)

			q0 = q1
			q1 = w

			npc_helper.iaxpy(v_T[k+1,0], q1, psi0 )

		for k in range(N-len(v)+(N<=len(v)), N):

			npc_helper.iaxpy(v_T[k,0], v[k-N], psi0)

		psinorm = npc_helper.two_norm(psi0)
		if abs(1.-psinorm)>10**(-3):
			print "|psi_0| = ", psinorm, ", poorly conditioned Lanczos?"

		npc_helper.iscal(1./psinorm, psi0)
		#print "Ortho:",
		#for o in orthogonal_to:
		#	print np.abs(npc.inner(o, psi0, do_conj=False)),
		#print
		return e_T[0], psi0, N

	else:
		return e_T[0], psi, N



def svd_theta(theta, truncation_par, tol = 0., chargeR = None):
	"""Performs SVD of wavefunction Theta (a matrix), and truncates according truncation and tol criteria.

		truncation_par = {'chi_max', 'svd_max', 'trunc_cut'}
			chi_max:	maximum chi allowed for truncation
			svd_max:	exp(-svd_max) is the minimum Schmidtt value allowed for truncation.
			trunc_cut:  the desired truncation bound, 1 - |PsiTrunc|^2 < trunc_cut
			tol: minimum allowed splitting between last sv kept and first dropped

		chargeR - see notes for npc.svd. Sets charge of Z.

		Returns X, Y, Z and trunc_err after truncation (theta = XYZ)
			Y is normalized (Y.Y) to 1
	"""

	if 'tol' in truncation_par: tol = truncation_par['tol']		# override tol if it's in the dictionary
	X, Y, Z = npc.svd(theta, full_matrices = False, compute_uv = True, cutoff = 1e-16, chargeR = chargeR, inner_labels = ['b*', 'b'])

	nrm_theta = np.linalg.norm(Y)
	Y = Y / nrm_theta

	##	Obtain the new values for s, and appropriately sort
	piv, svd_trunc = findCut( Y, truncation_par, tol = tol )
	if len(piv)*100 < len(Y):
		print "Catastrophic reduction in chi:", len(Y), "->", len(piv)
		print "Nans?"
		print "Y:", anynan(Y)
		print "X:",
		for x in X.dat:
		  print anynan(x),
		print "Z:",
		for x in Z.dat:
		  print anynan(x),

		print "|X^d X - 1|",
		xxd = npc.tensordot(X.conj(), X, axes = [[0], [0]])
		print npc.norm( xxd - npc.eye_like(xxd) )

		print "|Z^d Z - 1|",
		xxd = npc.tensordot(Z, Z.conj(), axes = [[1], [1]])
		print npc.norm( xxd - npc.eye_like(xxd) )
	t = np.zeros(len(Y), np.bool)

	t[piv] = True
	piv = t

	Y = Y[piv]
	X.iproject(piv, axes = 1) #Boolean selection method for npc tensors, X = X[:, piv]

	Z.iproject(piv, axes = 0) # Z = [piv, :]
	nrm = np.linalg.norm(Y)
	s_trunc = 1 - nrm**2
	Y = Y/nrm

	return X, Y, Z, s_trunc


#@profile
def mixed_svd_theta(theta, rho_L, rho_R, truncation_par, tol = 0., chargeR = None):
	""" theta is MxN and rho_L/R are square hermitian matrices of dim MxM and NxN respectively.

	Note: for both rho_L and rho_R, they should take form

		|psi><psi| as |row><col|.

	"""

	#Left.
	#print rho_L.shape, rho_R.shape
	l, X = npc.eigh(rho_L)
	l = l*(l>0.) #For stability reasons
	l = l / np.sum(l)
	pivL, svd_trunc = findCut( np.sqrt(l), truncation_par, tol = tol )
	keep = np.zeros(len(l), dtype=np.bool)
	keep[pivL] = True
	X = X.iproject(keep, axes = 1)
	s_trunc = (1. - np.sum(l[pivL]))

	l, Z = npc.eigh(rho_R)
	l = l*(l>0.)
	l = l/np.sum(l)
	pivR, svd_trunc = findCut( np.sqrt(l), truncation_par, tol = tol )
	keep = np.zeros(len(l), dtype=np.bool)
	keep[pivR] = True
	Z.itranspose()
	Z = Z.iproject(keep, axes = 0)
	#print "R", len(l), l[pivR]

	theta = npc.tensordot(X.conj(), theta, axes = [[0], [0]])
	theta = npc.tensordot(theta, Z.conj(), axes = [[1], [1]])


	#x, Y, z = npc.svd(theta, full_matrices = True, compute_uv = True, cutoff = None,  chargeR = chargeR)
	#Y = Y / np.linalg.norm(Y)

	s_trunc += (1. - np.sum(l[pivR]))

	return X, theta, Z, s_trunc*0.5


def findCut( Y, truncation_par, tol = 0. ):
	"""Given a schmidt spectrum Y, the goal is to determine a truncation point based on:

	  	1) a maximum allowed chi (chi_max)
		2) a bound on singular values, -log(s_i) < svd_max.
		3) a desired truncation error, trunc_cut
		4) a desire not to split near degenerate s. values., tol. Due to various symmetries there may be degenerate (or near degenerate) s_i, so we don't split any s_i s.t. log(s_i/s_j)  < tol .

		truncation_par = {'chi_min','chi_max', 'svd_max', 'trunc_cut'}
	   	chi_max:	maximum chi allowed for truncation
		svd_max:	exp(-svd_max) is the minimum Schmidt value allowed for truncation.
		trunc_cut:  the desired truncation bound, 1 - |PsiTrunc|^2 < trunc_cut
	   	chi_min:	minimum chi allowed for truncation. used for preventing TEBD lowering the value of chi after a local perturbation.

		tol: minimum allowed splitting between last sv kept and first dropped

		If chi_max or svd_max is None, the corresponding criteria is not checked.
		If trunc_cut = None, it is considered 0.

		Returns
		a) The locations of the Y to keep
		b) the first s dropped (or np.inf if all were kept)
	"""

	chi_max = truncation_par.get('chi_max', None)
	svd_max = truncation_par.get('svd_max', None)
	chi_min = truncation_par.get('chi_min', 0)
	trunc_cut = truncation_par.get('trunc_cut', 0.)

	if trunc_cut >= 1.:
		raise ValueError, "trunc_cut >=1."

	if np.any( Y > 10**(-10)) == False:
		print "Caution, no s above 10**(-10)"

	Y[ Y<=0. ] = 10**(-16)
	eS = np.log(Y)
	piv = np.argsort(eS)
	eS = eS[piv]

	#I check the five criteria - with the following priority.
	# chi < chi_max
	# levels are split by tol
	# satisfies SV bound
	# satisfied truncation goal
	# chi > chi_min

	if chi_max is not None:
		good = np.greater( chi_max, np.arange(len(eS) -1 , -1, -1) ) #With allowed chi
	else:
		good = np.ones(len(eS), dtype = np.bool)

	if tol:
		goodSplit = np.empty(len(eS), np.bool)
		goodSplit[0] = True
		goodSplit[1:] = np.greater_equal(eS[1:] - eS[0:-1], tol )
		good2 = np.logical_and(good, goodSplit)
	else:
		good2 = good

	if np.any(good2):
		good = good2

		if svd_max is not None:
			goodS = np.greater_equal(eS, -svd_max) #Satisfies cut criterion
			good2 = np.logical_and(good, goodS)
		else:
			good2 = good

		if np.any(good2):
			good = good2
		else:
			print "No good candidate cuts - ignoring svd_max request"
	else:
		print "No good candidate cuts - might split spectrum"
		print eS

	if trunc_cut > 0.:
		running = np.cumsum(Y[piv]**2) #cumulative sum of highest ith levels
		goodTrunc = (running <= trunc_cut)
		good2 = np.logical_and(good, goodTrunc)
		if np.any(good2):
			cut = np.nonzero(good2)[0][-1]
		else:
			cut = np.nonzero(good)[0][0]
	else:
		cut = np.nonzero(good)[0][0]

	if cut == 0:
		dropped = np.inf
	else:
		dropped = eS[cut - 1]

	if chi_min != 0 : # increase piv length to have at least chi_min levels

		if len(Y)-cut < chi_min:
			cut = len(Y) - chi_min
		if cut == 0:
			dropped = np.inf
		else:
			dropped = eS[cut - 1]

	return piv[cut:], dropped

def timeFT(Ct,t_list, nw=4):
	"""
	full complex data as input.
	data only for positive time.
	"""
	n = len(t_list)
	Wfunlist = [np.cos(np.pi*t_list[t]/(2*t_list[-1]))**nw  for t in range(n)]
	input_list = Wfunlist[:]*(Ct[:])

	FTresult = np.fft.fft( input_list )
	freq = 2*np.pi*np.fft.fftfreq(n,t_list[1]-t_list[0])
	freq = np.fft.fftshift(freq)
	FTresult = np.fft.fftshift(FTresult)
	return freq,FTresult

def spaceFT(Cx):
	"""
	does the FT in space
	available momenta are  defined by the box
	"""
	FTresult = np.fft.fft(Cx)
	dx = 1
	momenta	 =	np.fft.fftfreq(len(Cx),dx)
	momenta = np.fft.fftshift(momenta)*2*np.pi
	Ck = np.fft.fftshift(  FTresult[:]	)
	momenta = np.append(momenta, -momenta[0])
	Ck = np.append(Ck,Ck[0])
	return momenta,Ck



