r"""Compression of a MPS.


"""

# Copyright 2019 TeNPy Developers, GNU GPLv3

import numpy as np

from ..linalg import np_conserved as npc
from .truncation  import svd_theta
from ..networks import mps, mpo

__all__ = [
    'make_U', 'make_UI', 'make_UII', 'make_WII', 'mps_compress', 'svd_two_site', 'apply_mpo'
]

def make_U(H, dt, which='II'):
    r"""Creates the UI or UII propagator for a given Hamiltonian.

    Parameters
    ----------
    H : :class:`~tenpy.networks.mpo.MPO`
    dt : float|complex

    Returns
    -------
    U : :class:`~tepy.networks.mpo.MPO`
        The propagator.
    """
    if which=='II':
        return make_UII(H, dt)
    if which=='I':
        return make_UI(H, dt)
    raise ValueError("'%s' not implemented"%which)

def make_UI(H, dt):
    r"""Creates the UI propagator for a given Hamiltonian.

    Parameters
    ----------
    H : :class:`~tenpy.networks.mpo.MPO`
        The Hamiltonian to use
    dt : float|complex
        the time step per application of the propagator

    Returns
    -------
    UI : :class:`~tenpy.networks.mpo.MPO`
        The propagator.

    """

    U = [H.get_W(i).astype(np.result_type(dt,H.dtype)).transpose(['wL', 'wR', 'p', 'p*']) for i in range(H.L)]

    IdLR = []
    for i in range(0, H.L):  # correct?
        U1 = U[i]
        U2 = U[(i+1) % H.L]
        IdL = H.IdL[i+1]
        IdR = H.IdR[i+1]
        assert IdL is not None and IdR is not None
        U1[:, IdL, :, :] = U1[:, IdL, :, :] + dt * U1[:, IdR, :, :]
        keep=np.ones(U1.shape[1], dtype=bool)
        keep[IdR] = False
        U1.iproject(keep, 1)
        if H.finite and i + 1 == H.L:
            keep = np.ones(U2.shape[0], dtype=bool)
            assert H.IdR[0] is not None
            keep[H.IdR[0]] = False
        U2.iproject(keep, 0)

        if IdL > IdR:
            IdLR.append(IdL - 1)
        else:
            IdLR.append(IdL)

    IdL = H.IdL[0]
    IdR = H.IdR[0]
    assert IdL is not None and IdR is not None
    if IdL > IdR:
        IdLR_0 = IdL - 1
    else:
        IdLR_0 = IdL
    IdLR = [IdLR_0] + IdLR

    return mpo.MPO(H.sites, U, H.bc, IdLR, IdLR, np.inf)


def make_UII(H, dt):
    r"""Creates the UII propagator for a given Hamiltonian.

    Parameters
    ----------
    H : :class:`~tenpy.networks.mpo.MPO`
        The Hamiltonian to use
    dt : float|complex
        the time step per application of the propagator

    Returns
    -------
    UII : :class:`~tenpy.networks.mpo.MPO`
        The propagator.

    """
    dtype = np.result_type(dt, H.dtype)
    IdL=H.IdL
    IdR=H.IdR
    v = np.empty_like(IdL)

    c0 = np.zeros((1, H[0].num_q), dtype=np.int) #TODO 
    U = []
    for i in xrange(0, self.L):
            W = H.get_W(i).itranspose(['wL','wR', 'p', 'p*'])
            W = W.to_ndarray()
            aL = range(W.shape[0]); aL.remove(IdL[i]); aL.remove(IdR[i]);
            aR = range(W.shape[1]); aR.remove(IdL[i+1]); aR.remove(IdR[i+1])
            
            #Extract (A, B, C, D)
            D = W[IdL[i] , IdR[i+1], :, : ]
            C = W[IdL[i], aR, :, :]
            B = W[aL, IdR[i+1], :, :]
            A = W[aL, aR, :, :]
    
            WII = make_WII(dt, A, B, C, D)
            #TODO from here on
            qL_flat = npc.q_flat_from_q_ind(H[i].q_ind[0])[aL, :]
            qL_flat = np.vstack( (c0, qL_flat))
            qR_flat = npc.q_flat_from_q_ind(H[i].q_ind[1])[aR, :]
            qR_flat = np.vstack( (c0, qR_flat))
            qp_flat = npc.q_flat_from_q_ind(H[i].q_ind[2])
            
            
            perm, WII = npc.Array.from_ndarray(WII, [qL_flat, qR_flat, qp_flat, qp_flat], q_conj = H[i].q_conj, mod_q = H[i].mod_q, cutoff=1e-16)

            v[i-1] = np.argsort(perm[0])[0]
            v[i] = np.argsort(perm[1])[0]
            U.append(WII)
            
    return mpo.MPO(H.sites, U, H.bc, v, v.copy()) #  translate_Q1_data = self.translate_Q1_data ??




def make_WII(t, A, B, C, D):
	r""" WII approx to exp(t H) from sys (A, B, C, D)

        Parameters
        ----------
        t : float
            time intervall for the propagator
        A : ndarray
        B : ndarray
        C : ndarray
        D : ndarray
		
	"""
	### Algorithm
	#
	# In the paper, we have two formal parameter "phi_{r/c}" which satisfies phi_r^2 = phi_c^2 = 0
	# To implement this, we temporarily extend the virtual Hilbert space with two hard-core bosons "br, bl"
	# The components of Eqn (11) can be computed for each index of the virtual row / column independently
	# The matrix exponential is done in the hard-core extended Hilbert space
	
	tB = t/np.sqrt(np.abs(t)) #spread time step across B, C
	tC = np.sqrt(np.abs(t))
	d = D.shape[0]
	
	#The virtual size of W is  (1+Nr, 1+Nc)
	Nr = A.shape[0]
	Nc = A.shape[1]
	W = np.zeros( (1 + Nr, 1 + Nc, d, d), dtype = np.result_type(D, t))

	Id_ = np.array([[1, 0], [0, 1]]) #2x2 operators in a hard-core boson space
	b = np.array([[0, 0], [1, 0]])

	Id = np.kron(Id_, Id_) #4x4 operators in the 2x hard core boson space
	Br = np.kron(b, Id_)
	Bc = np.kron(Id_, b)
	Brc = np.kron(b, b)
	for r in xrange(Nr):    #double loop over row / column of A
		for c in xrange(Nc):
			#Select relevent part of virtual space and extend by hardcore bosons
			h = np.kron(Brc, A[r, c, :, :]) + np.kron(Br, tB*B[r, :, :]) + np.kron(Bc, tC*C[c, :, :]) + t*np.kron(Id, D)
			w = sp.linalg.expm(h) #Exponentiate in the extended Hilbert space
			w = w.reshape((2, 2, d, 2, 2, d))
			w = w[:, :, :, 0, 0, :]
			W[1+r, 1+c, :, :] = w[1, 1]  #This part now extracts relevant parts according to Eqn 11
			if c==0:
				W[1+r, 0] = w[1, 0]
			if r==0:
				W[0, 1+c] = w[0, 1]
				if c==0:
					W[0, 0] = w[0, 0]
		if Nc==0: #technically only need one boson
			h = np.kron(Br, tB*B[r, :, :]) + t*np.kron(Id, D)
			w = sp.linalg.expm(h)
			w = w.reshape((2, 2, d, 2, 2, d))
			w = w[:, :, :, 0, 0, :]
			W[1+r, 0] = w[1, 0]
			if r==0:
				W[0, 0] = w[0, 0]

	if Nr == 0:
		for c in xrange(Nc):
			h = np.kron(Bc, tC*C[c, :, :]) + t*np.kron(Id, D)
			w = sp.linalg.expm(h)
			w = w.reshape((2, 2, d, 2, 2, d))
			w = w[:, :, :, 0, 0, :]
			W[0, 1+c] = w[0, 1]
			if c==0:
				W[0, 0] = w[0, 0]
		if Nc==0:
			W =sp.linalg.expm(t*D)

	return W


def mps_compress(psi, trunc_par):
    r"""Takes an MPS and compresses it. In Place.

    Parameters
    ----------
    psi : :class:`tenpy.networks.mps.MPS`
        MPS to be compressed.
        The singular values are usually ignored and need not be set to sensible values, except for bc='infinite', where psi.S[0] is needed as an initial guess.
    trunc_par : dict
        See :func:`~tenpy.algorithms.truncation.truncate`
    """
    bc=psi.bc
    L=psi.L
    if bc=='finite':
        # Do QR starting from the left
        B=psi.get_B(0,form=None)
        for i in range(psi.L - 1):
            B=B.combine_legs(['vL', 'p'])
            q,r =npc.qr(B, inner_labels=['vR', 'vL'])
            B=q.split_legs()
            psi.set_B(i,B,form=None)
            B=psi.get_B((i+1)%L,form=None)
            B=npc.tensordot(r,B, axes=('vR', 'vL'))
        # Do SVD from right to left, truncate the singular values according to trunc_par
        for i in range(psi.L-1,0,-1):
            B=B.combine_legs(['p', 'vR'])
            u, s, vh, err, norm_new = svd_theta(B, trunc_par)
            psi.norm*=norm_new
            vh=vh.split_legs()
            psi.set_B(i%L, vh, form='B')
            B=psi.get_B(i-1, form=None)
            B=npc.tensordot(B, u, axes=('vR', 'vL'))
            B.iscale_axis(s, 'vR')
            psi.set_SL(i%L, s)
        psi.set_B(0, B, form='Th')
    if bc=='infinite':
        for i in range(psi.L):
            svd_two_site(i, psi)
        for i in range(psi.L-1,-1, -1):
            svd_two_site(i, psi, trunc_par)


def svd_two_site(i, mps, trunc_par=None):
    r"""Builds a theta and splits it using svd for an MPS.

    Parameters
    ----------
    i : int
        First site.
    mps : :class:`tenpy.networks.mps.MPS`
        MPS to use on.
    trunc_par : None|dict
       If None no truncation is done. Else dict as in :func:`~tenpy.algorithms.truncation.truncate`.
    """
    theta=mps.get_theta(i)
    theta=theta.combine_legs([['vL','p0'],['p1','vR']], qconj=[+1,-1])
    if trunc_par==None:
        u, s, vh, err, renorm = svd_theta(theta, {'chi_max':10000, 'svd_min':1.e-15, 'trunc_cut':1.e-15})
        mps.norm*=renorm
    else:
        u, s, vh, err, renorm = svd_theta(theta, trunc_par)
        mps.norm*=renorm
    u=u.split_legs()
    vh=vh.split_legs()
    u.ireplace_label('p0', 'p')
    vh.ireplace_label('p1', 'p')
    mps.set_B(i, u, form='A')
    mps.set_B((i+1)%mps.L, vh, form='B')
    mps.set_SR(i, s)


def apply_mpo(psi, mpo, trunc_par):
    """Applies an mpo and truncates the resulting MPS using SVD.

    Parameters
    ----------
    mps : :class:`~tenpy.networks.mps.MPS`
        MPS to apply operator on
    mpo : :class:`~tenpy.networks.mpo.MPO`
        MPO to apply. Usually one of make_UI() or make_UII(). The approximation being made are uncontrolled for other mpos and infinite bc.
    trunc_par : dict
        Truncation parameters. See :func:`truncate`

    Returns
    -------
    mps : :class:`~tenpy.networks.mps.MPS` 
        Resulting new MPS
    """
    bc=psi.bc
    if bc!=mpo.bc:
        raise ValueError("Boundary conditions of MPS and MPO are not the same")
    if psi.L != mpo.L:
        raise ValueError("Length of MPS and MPO not the same")
    Bs=[npc.tensordot(psi.get_B(i, form='B'), mpo.get_W(i), axes=('p', 'p*')) for i in range(psi.L)]
    Bs[0]=npc.tensordot(psi.get_theta(0,1), mpo.get_W(0), axes=('p0', 'p*'))
    for i in range(psi.L):
        if i==0 and bc=='finite':
            Bs[i]=Bs[i].take_slice(mpo.get_IdL(i) ,'wL')
            Bs[i]=Bs[i].combine_legs(['wR','vR'], qconj=[-1])
            Bs[i].ireplace_labels(['(wR.vR)'], ['vR'])
            Bs[i].get_leg('vR').to_LegCharge
        elif i==psi.L-1 and bc=='finite':
            Bs[i]=Bs[i].take_slice(mpo.get_IdR(i) ,'wR')
            Bs[i]=Bs[i].combine_legs(['wL','vL'], qconj=[1])
            Bs[i].ireplace_labels(['(wL.vL)'], ['vL'])
            Bs[i].get_leg('vL').to_LegCharge()
        else:
            Bs[i]=Bs[i].combine_legs([['wL','vL'],['wR','vR']], qconj=[+1, -1])
            Bs[i].ireplace_labels(['(wL.vL)', '(wR.vR)'], ['vL','vR'])
            Bs[i].get_leg('vL').to_LegCharge()
            Bs[i].get_leg('vR').to_LegCharge()

    if bc=='infinite':
        #calculate good (rather arbitrary) guess for S[0] (no we don't like it either)
        weight=np.ones(mpo.get_W(0).shape[mpo.get_W(0).get_leg_index('wL')])*0.05
        weight[mpo.get_IdL(0)]=1
        weight=weight/np.linalg.norm(weight)
        S=[np.kron(psi.get_SL(0), weight)]
    else:
        S=[np.ones(Bs[0].shape[Bs[0].get_leg_index('vL')])]
    #Wrong S values but will be calculated in mps_compress
    for i in range(psi.L):
        S.append(np.ones(Bs[i].shape[Bs[i].get_leg_index('vR')]))
        
    new_mps = mps.MPS(psi.sites, Bs, S, form='B', bc=psi.bc) # form='B' is not true but works
    mps_compress(new_mps, trunc_par)
    return new_mps
