# Copyright 2021 TeNPy Developers, GNU GPLv3

from ..linalg import np_conserved as npc
import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    pass  # error/warning in mpi_parallel.py


DONE = None  # sentinel to say that the worker should finishj


def run(action, node_local, meta, on_main=None):
    """Special action to call other actions"""
    node_local.comm.bcast((action, meta))
    return action(node_local, on_main, *meta)


def replica_main(node_local):
    while True:
        action, meta = node_local.comm.bcast(None)
        # TODO: make all of those functions to limit the scope of local variables!
        if action is DONE:  # allow to gracefully terminate
            print("MPI rank %d signing off" % node_local.comm.rank)
            return
        action(node_local, None, *meta)


def distribute_H(node_local, on_main, H):
    node_local.add_H(H)


def matvec(node_local, on_main, theta, LH_key, RH_key):
    LHeff = node_local.distributed[LH_key]
    RHeff = node_local.distributed[RH_key]
    worker = node_local.worker
    if node_local.H.explicit_plus_hc:
        if worker is None:
            theta_hc = matvec_hc(LHeff, theta, RHeff)
        else:
            res = {}
            worker.put_task(matvec_hc, LHeff, theta, RHeff, return_dict=res, return_key="theta_hc")
    theta = matvec_plain(LHeff, theta, RHeff)
    if node_local.H.explicit_plus_hc:
        if worker is not None:
            worker.join_tasks()
            theta_hc = res['theta_hc']
        theta = theta + theta_hc
    return node_local.comm.reduce(theta, op=MPI.SUM)


def matvec_plain(LHeff, theta, RHeff):
    theta = npc.tensordot(LHeff, theta, axes=['(vR.p0*)', '(vL.p0)'])
    theta = npc.tensordot(theta, RHeff, axes=[['wR', '(p1.vR)'], ['wL', '(p1*.vL)']])
    theta.ireplace_labels(['(vR*.p0)', '(p1.vL*)'], ['(vL.p0)', '(p1.vR)'])
    return theta


def matvec_hc(LHeff, theta, RHeff):
    theta = theta.conj()  # copy!
    theta = npc.tensordot(theta, LHeff, axes=['(vL*.p0*)', '(vR*.p0)'])
    theta = npc.tensordot(RHeff, theta,
                                axes=[['(p1.vL*)', 'wL'], ['(p1*.vR*)', 'wR']])
    theta.iconj().itranspose()
    theta.ireplace_labels(['(vR*.p0)', '(p1.vL*)'], ['(vL.p0)', '(p1.vR)'])
    return theta


def effh_to_matrix(node_local, on_main, LH_key, RH_key):
    LHeff = node_local.distributed[LH_key]
    RHeff = node_local.distributed[RH_key]
    contr = npc.tensordot(LHeff, RHeff, axes=['wR', 'wL'])
    contr = contr.combine_legs([['(vR*.p0)', '(p1.vL*)'], ['(vR.p0*)', '(p1*.vL)']],
                                qconj=[+1, -1])
    return node_local.comm.reduce(contr, op=MPI.SUM)


def scatter_distr_array(node_local, on_main, key, in_cache):
    local_part = node_local.comm.scatter(on_main)
    if in_cache:
        node_local.cache[key] = local_part
    else:
        node_local.distributed[key] = local_part


def gather_distr_array(node_local, on_main, key, in_cache):
    local_part = node_local.cache[key] if in_cache else node_local.distributed[key]
    return node_local.comm.gather(local_part)


def attach_B(node_local, on_main, old_key, new_key, B):
    local_part = node_local.distributed[old_key]
    #B = B.combine_legs(['p1', 'vR'], pipes=local_part.get_leg('(p1.vL*)'))
    local_part = npc.tensordot(B, local_part, axes=['(p1.vR)', '(p1*.vL)'])
    local_part = npc.tensordot(B.conj(), local_part, axes=['(p1*.vR*)', '(p1.vL*)'])
    node_local.cache[new_key] = local_part


def attach_A(node_local, on_main, old_key, new_key, A):
    local_part = node_local.distributed[old_key]
    #A = A.combine_legs(['vL', 'p0'], pipes=local_part.get_leg('(vR*.p0)'))
    local_part = npc.tensordot(A, local_part, axes=['(vL.p0)', '(vR.p0*)'])
    local_part = npc.tensordot(A.conj(), local_part, axes=['(vL*.p0*)', '(vR*.p0)'])
    node_local.cache[new_key] = local_part


def full_contraction(node_local, on_main, case, LP_key, LP_ic, RP_key, RP_ic, theta):
    LP = node_local.cache[LP_key] if LP_ic else node_local.distributed[LP_key]
    RP = node_local.cache[RP_key] if RP_ic else node_local.distributed[RP_key]
    if case == 0b11:
        if isinstance(theta, npc.Array):
            LP = npc.tensordot(LP, theta, axes=['vR', 'vL'])
            LP = npc.tensordot(theta.conj(), LP, axes=['vL*', 'vR*'])
        else:
            S = theta  # S is real, so no conj() needed
            LP = LP.scale_axis(S, 'vR').scale_axis(S, 'vR*')
    elif case == 0b10:
        RP = npc.tensordot(theta, RP, axes=['(p1.vR)', '(p1*.vL)'])
        RP = npc.tensordot(RP, theta.conj(), axes=['(p1.vL*)', '(p1*.vR*)'])
    elif case == 0b01:
        LP = npc.tensordot(LP, theta, axes=['(vR.p0*)', '(vL.p0)'])
        LP = npc.tensordot(theta.conj(), LP, axes=['(vL*.p0*)', '(vR*.p0)'])
    else:
        assert False
    full_contr = npc.inner(LP, RP, [['vR*', 'wR', 'vR'], ['vL*', 'wL', 'vL']], do_conj=False)
    if node_local.H.explicit_plus_hc:
        full_contr = full_contr + full_contr.conj()
    return node_local.comm.reduce(full_contr, op=MPI.SUM)


def mix_rho(node_local, on_main, theta, i0, amplitude, update_LP, update_RP, LH_key, RH_key):
    comm = node_local.comm
    # b_IdL = block with IdL; IdL_b = index of IdL inside block
    i1 = (i0 + 1) % node_local.H.L
    (b_IdL, IdL_b), (b_IdR, IdR_b) = node_local.IdLR_blocks[i1]
    D = node_local.local_MPO_chi[i1]
    mix_L = np.full((D, ), amplitude)
    mix_R = np.full((D, ), amplitude)
    if b_IdL == comm.rank:
        mix_L[IdL_b] = 1.
        mix_R[IdL_b] = 0.
    if b_IdR == comm.rank:
        mix_L[IdR_b] = 0.
        mix_R[IdR_b] = 1.
    # TODO: optimize to minimize transpose
    if node_local.H.explicit_plus_hc:
        raise NotImplementedError("TODO respect the explicit_plus_hc flag!")
    if update_LP:
        LHeff = node_local.distributed[LH_key]
        rho_L = npc.tensordot(LHeff, theta, axes=['(vR.p0*)', '(vL.p0)'])
        rho_L.ireplace_label('(vR*.p0)', '(vL.p0)')
        rho_L_c = rho_L.conj()
        rho_L.iscale_axis(mix_L, 'wR')
        rho_L = npc.tensordot(rho_L, rho_L_c, axes=[['wR', '(p1.vR)'], ['wR*', '(p1*.vR*)']])
        rho_L = comm.reduce(rho_L, op=MPI.SUM)
    if update_RP:
        RHeff = node_local.distributed[RH_key]
        rho_R = npc.tensordot(theta, RHeff, axes=['(p1.vR)', '(p1*.vL)'])
        rho_R.ireplace_label('(p1.vL*)', '(p1.vR)')
        rho_R_c = rho_R.conj()
        rho_R.iscale_axis(mix_R, 'wL')
        rho_R = npc.tensordot(rho_R, rho_R_c, axes=[['(vL.p0)', 'wL'], ['(vL*.p0*)', 'wL*']])
        rho_R = comm.reduce(rho_R, op=MPI.SUM)
    if comm.rank == 0 and (b_IdL == -1 or b_IdR == -1):  # IdL/IdR is None
        raise NotImplementedError("TODO: add explicit identities to rho_L/rho_R")
    if update_LP and update_RP:
        return rho_L, rho_R
    elif update_LP:
        return rho_L
    else:
        return rho_R


def cache_optimize(node_local, on_main, short_term_keys, preload):
    node_local.cache.set_short_term_keys(*short_term_keys, *preload)
    node_local.cache.preload(*preload)


def cache_del(node_local, on_main, *keys):
    for key in keys:
        del node_local.cache[key]
