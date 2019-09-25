r"""Compression of a MPS.


"""
import numpy as np

from ..linalg import np_conserved as npc
from .truncation  import svd_theta
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

        #  if i!= H.L-1:
        #      keep=np.ones(U[i].shape[1], dtype=bool)
        #      keep[H.get_IdL(i+1)]=False
        #      U[i].iproject(keep,1)

        #      ind=H.get_IdR(i)
        #      if ind>=H.get_IdL(i+1): ind-=1
        #      perm=[i for i in range(U[i].shape[1])]
        #      del perm[ind]
        #      perm.insert(0, ind)
        #      U[i].permute(perm, 1)
        #  if i!=0:
        #      keep=np.ones(U[i].shape[0], dtype=bool)
        #      keep[H.get_IdR(i-1)]=False
        #      U[i].iproject(keep,0)

        #      ind=H.get_IdL(i)
        #      if ind>=H.get_IdR(i-1): ind-=1
        #      perm=[i for i in range(U[i].shape[0])]
        #      del perm[ind]
        #      perm.insert(0, ind)
        #      U[i].permute(perm, 0)

        #  U[i][:,0,:,:]*=dt
        #  U[i][0,0,:,:]+=H.sites[i].Id

    return mpo.MPO(H.sites, U, H.bc, IdLR, IdLR, np.inf)



def mps_compress(psi, trunc_par, renormalize=True):
    r"""Takes an MPS and compresses it. In Place.

    Parameters
    ----------
    psi: MPS
        MPS to be compressed. It is taken in form=None so it has to be in a form such that no singular values have to be inserted.
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
        u, s, vh, err, norm_new = svd_theta(B, trunc_par)
        #u, s, vh = npc.svd(B, inner_labels=['vR', 'vL'])
        #mask, norm_new, err = truncate(s, trunc_par)
        #vh.iproject(mask, 'vL')
        vh=vh.split_legs()
        #s=s[mask]/norm_new
        #u.iproject(mask, 'vR')
        psi.set_B(i, vh, form='B')
        B=psi.get_B(i-1, form=None)
        B=npc.tensordot(B, u, axes=('vR', 'vL'))
        B.iscale_axis(s, 'vR')
        psi.set_SL(i, s)
    psi.set_B(0, B, form='Th')
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
    Bs=[npc.tensordot(psi.get_B(i, form='B'), mpo.get_W(i), axes=('p', 'p*')) for i in range(psi.L)]
    Bs[0]=npc.tensordot(psi.get_theta(0,1), mpo.get_W(0), axes=('p0', 'p*'))
    for i in range(psi.L):
        if i==0:
            Bs[i]=Bs[i].take_slice(mpo.get_IdL(i) ,'wL')
            Bs[i]=Bs[i].combine_legs(['wR','vR'], qconj=[-1])
            Bs[i].ireplace_labels(['(wR.vR)'], ['vR'])
            Bs[i].get_leg('vR').to_LegCharge()
        elif i==psi.L-1:
            Bs[i]=Bs[i].take_slice(mpo.get_IdR(i) ,'wR')
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
