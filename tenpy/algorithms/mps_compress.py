r"""Compression of a MPS.


"""
import numpy as np

from ..linalg import np_conserved as npc
from .truncation  import truncate
from ..networks import mps, mpo


def make_UI(H, dt):
    r"""Creates the UI propagator for a given Hamiltonian.

    Parameters
    ----------
    H: MPO
        The Hamiltonian to use
    dt:
        the time step per application of the propagator

    Returns
    -------
    UI: MPO
        The propagator.

    """
    if H.bc!='finite':
        raise ValueError("Only finite bc implemented")
    

    U = [H.get_W(i).astype(np.result_type(dt*H.get_W(i))) for i in range(H.L)]
    for i in range(0, H.L):
        if i!= H.L-1:
            keep=np.ones(U[i].shape[1], dtype=bool)
            keep[H.get_IdL(i+1)]=False
            U[i].iproject(keep,1)

            ind=H.get_IdR(i)
            if ind>=H.get_IdL(i+1): ind-=1
            perm=[i for i in range(U[i].shape[1])]
            del perm[ind]
            perm.insert(0, ind)
            U[i].permute(perm, 1)
        if i!=0:
            keep=np.ones(U[i].shape[0], dtype=bool)
            keep[H.get_IdR(i-1)]=False
            U[i].iproject(keep,0)

            ind=H.get_IdL(i)
            if ind>=H.get_IdR(i-1): ind-=1
            perm=[i for i in range(U[i].shape[1])]
            del perm[ind]
            perm.insert(0, ind)
            U[i].permute(perm, 1)


        U[i][:,0,:,:]*=dt
        U[i][0,0,:,:]+=H.sites[i].Id

            
    return mpo.MPO(H.sites, U, bc=H.bc)



def mps_compress(psi, trunc_par, renormalize=True):
    r"""Takes an MPS and compresses it. In Place.

    Parameters
    ----------
    psi: MPS
        MPS to be compressed
    trunc_par: dict
        See :func:`truncate`
    renormalize: bool
        If True the norm is set to 1.
    """
    if psi.bc != 'finite':
        raise NotImplemented

    # Do QR starting from the left
    B=psi.get_B(0,form=None)
    for i in range(psi.L-1):
        B=B.combine_legs(['vL', 'p'])
        q,r =npc.qr(B, inner_labels=['vR', 'vL'])    
        B=q.split_legs()
        psi.set_B(i,B,form=None)
        B=psi.get_B(i+1,form=None)
        B=npc.tensordot(r,B, axes=('vR', 'vL'))
    # Do SVD from right to left, truncate the singular values according to trunc_par
    for i in range(psi.L-1,0,-1):
        B=B.combine_legs(['p', 'vR'])
        u, s, vh = npc.svd(B, inner_labels=['vR', 'vL'])
        mask, norm_new, err = truncate(s, trunc_par)
        vh.iproject(mask, 'vL')
        vh=vh.split_legs()
        s=s[mask]/norm_new
        u.iproject(mask, 'vR')
        psi.set_B(i, vh, form='B')
        B=psi.get_B(i-1, form=None)
        B=npc.tensordot(B, u, axes=('vR', 'vL'))
        B.iscale_axis(s, 'vR')
        psi.set_SL(i, s)
    psi.set_B(0, B, form='B')
    if renormalize:
        psi.norm=1.


def apply_mpo(psi, mpo, trunc_par):
    """Applies an mpo and truncates the resulting MPS using SVD.
    
    Parameters
    ----------
    mps: MPS
        MPS to apply operator on
    mpo: MPO
        MPO to apply
    trunc_par: dict
        Truncation parameters. See :func:`truncate`
    
    Returns
    -------
    mps: MPS
        Resulting new MPS
    """
    if psi.bc!='finite' or mpo.bc != 'finite':
        raise ValueError("Only implemented for finite bc so far")
    if psi.L != mpo.L:
        raise ValueError("Length of MPS and MPO not the same")
    Bs=[npc.tensordot(psi.get_B(i, form=None), mpo.get_W(i), axes=('p', 'p*')) for i in range(psi.L)]
    for i in range(psi.L):
        if i==0:
            if mpo.get_IdL(0)!=None: # TODO remove and make orderly (IdL in make_UI)
                Bs[i]=Bs[i].take_slice(mpo.get_IdL(0) ,'wL')
            else:
                Bs[i]=Bs[i].take_slice(0 ,'wL')
            Bs[i]=Bs[i].combine_legs(['wR','vR'], qconj=[-1])
            Bs[i].ireplace_labels(['(wR.vR)'], ['vR'])
            Bs[i].get_leg('vR').to_LegCharge()
        elif i==psi.L-1:  
            if mpo.get_IdR(i)!=None:
                Bs[i]=Bs[i].take_slice(mpo.get_IdR(i) ,'wR')
            else:
                Bs[i]=Bs[i].take_slice(Bs[i].shape[Bs[i].get_leg_index('wR')]-1 ,'wR')
            Bs[i]=Bs[i].combine_legs(['wL','vL'], qconj=[1])
            Bs[i].ireplace_labels(['(wL.vL)'], ['vL'])
            Bs[i].get_leg('vL').to_LegCharge()
        else:
            Bs[i]=Bs[i].combine_legs([['wL','vL'],['wR','vR']], qconj=[+1, -1])
            Bs[i].ireplace_labels(['(wL.vL)', '(wR.vR)'], ['vL','vR'])
            Bs[i].get_leg('vL').to_LegCharge()
            Bs[i].get_leg('vR').to_LegCharge()
    
    #Wrong S values but will be calculated in mps_compress
    S=[np.ones(1)]
    for i in range(psi.L-1):
        S.append(np.ones(Bs[i].shape[Bs[i].get_leg_index('vR')]))
    S.append(np.ones(1))
    new_mps = mps.MPS(psi.sites, Bs, S, form=None)
    mps_compress(new_mps, trunc_par)
    return new_mps
    
