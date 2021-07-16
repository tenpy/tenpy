# Copyright 2021 TeNPy Developers, GNU GPLv3

from ..linalg import np_conserved as npc
import numpy as np
import copy

try:
    from mpi4py import MPI
except ImportError:
    pass  # error/warning in mpi_parallel.py


DONE = None  # sentinel to say that the worker should finish

def sum_none(A, B, C):
    assert C is None    # TODO, why do we need a third argument?
    if A is not None and B is not None:
        return A + B
    elif A is not None:
        return A
    elif B is not None:
        return B
    else:
        return None

MPI_SUM_NONE = MPI.Op.Create(sum_none, commute=False)

def run(action, node_local, meta, on_main=None):
    """Special action to call other actions"""
    node_local.comm.bcast((action, meta))
    return action(node_local, on_main, *meta)


def replica_main(node_local):
    while True:
        action, meta = node_local.comm.bcast(None)
        if action is DONE:  # allow to gracefully terminate
            print(f"MPI rank {node_local.comm.rank:d} signing off")
            return
        action(node_local, None, *meta)


def distribute_H(node_local, on_main, H):
    node_local.add_H(H)


def matvec(node_local, on_main, theta, LH_key, RH_key):
    LHeff = node_local.distributed[LH_key]
    RHeff = node_local.distributed[RH_key]
    if LHeff is None or RHeff is None:
        assert LHeff is RHeff
        theta = None
    else:
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
    #return node_local.comm.reduce(theta, op=MPI.SUM)
    return node_local.comm.reduce(theta, op=MPI_SUM_NONE)


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
    if LHeff is not None and RHeff is not None:
        contr = npc.tensordot(LHeff, RHeff, axes=['wR', 'wL'])
        contr = contr.combine_legs([['(vR*.p0)', '(p1.vL*)'], ['(vR.p0*)', '(p1*.vL)']],
                                qconj=[+1, -1])
    else:
        contr = None
    #return node_local.comm.reduce(contr, op=MPI.SUM)
    return node_local.comm.reduce(contr, op=MPI_SUM_NONE)


def scatter_distr_array(node_local, on_main, key, in_cache):
    local_part = node_local.comm.scatter(on_main)
    if in_cache:
        node_local.cache[key] = local_part
    else:
        node_local.distributed[key] = local_part


def gather_distr_array(node_local, on_main, key, in_cache):
    local_part = node_local.cache[key] if in_cache else node_local.distributed[key]
    return node_local.comm.gather(local_part)

def attach_LP_to_W(node_local, on_main, i, old_key, new_key):
    assert on_main is None
    comm = node_local.comm
    my_env = node_local.cache[old_key]
    received_env = None
    LHeff = None
    rank = comm.rank
    L = node_local.H.L

    interactions = reverse_interactions(node_local.jobs[i % L])
    commands = generate_instructions(interactions, comm.size)
    for cycle in commands:
        my_commands = cycle[rank]
        for order in my_commands:
            if order[0] == 'self_contract':
                Wb = npc.tensordot(my_env, node_local.W_blocks[i % L][comm.rank][comm.rank], ['wR', 'wL']).replace_labels(['p', 'p*'], ['p0', 'p0*'])
                if LHeff is None:
                    LHeff = Wb
                else:
                    LHeff += Wb
            elif order[0] == 'recv':
                src, tag = order[1], order[2]
                received_env = comm.recv(source=src, tag=tag)
                Wb = npc.tensordot(received_env, node_local.W_blocks[i % L][src][comm.rank], ['wR', 'wL']).replace_labels(['p', 'p*'], ['p0', 'p0*'])
                if LHeff is None:
                    LHeff = Wb
                else:
                    LHeff += Wb
            elif order[0] == 'send':
                dest, tag = order[1], order[2]
                comm.send(my_env, dest=dest, tag=tag)
            else:
                raiseValueError('Unexpected order for LHeff contraction.')
        comm.Barrier()  # Synchronize nodes after each cycle. # TODO is barrier necessary? Probs not since calls are blocking.

    if LHeff is not None:
        pipeL = LHeff.make_pipe(['vR*', 'p0'], qconj=+1)
        LHeff = LHeff.combine_legs([['vR*', 'p0'], ['vR', 'p0*']], pipes=[pipeL, pipeL.conj()], new_axes=[0, 2]) # vR*.p, wR, vR.p*
    node_local.distributed[new_key] = LHeff

def attach_W_to_RP(node_local, on_main, i, old_key, new_key):
    assert on_main is None
    comm = node_local.comm
    my_env = node_local.cache[old_key]
    received_env = None
    RHeff = None
    rank = comm.rank
    L = node_local.H.L

    interactions = node_local.jobs[i % L]
    commands = generate_instructions(interactions, comm.size)
    for cycle in commands:
        my_commands = cycle[rank]
        for order in my_commands:
            if order[0] == 'self_contract':
                Wb = npc.tensordot(node_local.W_blocks[i % L][comm.rank][comm.rank], my_env, ['wR', 'wL']).replace_labels(['p', 'p*'], ['p1', 'p1*'])
                if RHeff is None:
                    RHeff = Wb
                else:
                    RHeff += Wb
            elif order[0] == 'recv':
                src, tag = order[1], order[2]
                received_env = comm.recv(source=src, tag=tag)
                Wb = npc.tensordot(node_local.W_blocks[i % L][comm.rank][src], received_env, ['wR', 'wL']).replace_labels(['p', 'p*'], ['p1', 'p1*'])
                if RHeff is None:
                    RHeff = Wb
                else:
                    RHeff += Wb
            elif order[0] == 'send':
                dest, tag = order[1], order[2]
                comm.send(my_env, dest=dest, tag=tag)
            else:
                raiseValueError('Unexpected order for RHeff contraction.')
        comm.Barrier()  # Synchronize nodes after each cycle.
    if RHeff is not None:
        pipeR = RHeff.make_pipe(['p1', 'vL*'], qconj=-1)
        RHeff = RHeff.combine_legs([['p1', 'vL*'], ['p1*', 'vL']], pipes=[pipeR, pipeR.conj()], new_axes=[2, 1])
    node_local.distributed[new_key] = RHeff

def attach_B(node_local, on_main, old_key, new_key, B):
    local_part = node_local.distributed[old_key]
    if local_part is not None:
    #B = B.combine_legs(['p1', 'vR'], pipes=local_part.get_leg('(p1.vL*)'))
        local_part = npc.tensordot(B, local_part, axes=['(p1.vR)', '(p1*.vL)'])
        local_part = npc.tensordot(B.conj(), local_part, axes=['(p1*.vR*)', '(p1.vL*)'])
    node_local.cache[new_key] = local_part


def attach_A(node_local, on_main, old_key, new_key, A):
    local_part = node_local.distributed[old_key]
    if local_part is not None:
    #A = A.combine_legs(['vL', 'p0'], pipes=local_part.get_leg('(vR*.p0)'))
        local_part = npc.tensordot(A, local_part, axes=['(vL.p0)', '(vR.p0*)'])
        local_part = npc.tensordot(A.conj(), local_part, axes=['(vL*.p0*)', '(vR*.p0)'])
    node_local.cache[new_key] = local_part


def full_contraction(node_local, on_main, case, LP_key, LP_ic, RP_key, RP_ic, theta):
    LP = node_local.cache[LP_key] if LP_ic else node_local.distributed[LP_key]
    RP = node_local.cache[RP_key] if RP_ic else node_local.distributed[RP_key]
    full_contr = None
    if LP is None or RP is None:
        assert LP is RP
    else:
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
    #return node_local.comm.reduce(full_contr, op=MPI.SUM)
    return node_local.comm.reduce(full_contr, op=MPI_SUM_NONE)


def mix_rho(node_local, on_main, theta, i0, amplitude, update_LP, update_RP, LH_key, RH_key):
    comm = node_local.comm
    # b_IdL = block with IdL; IdL_b = index of IdL inside block
    i1 = (i0 + 1) % node_local.H.L
    (b_IdL, IdL_b), (b_IdR, IdR_b) = node_local.IdLR_blocks[i1]
    D = node_local.local_MPO_chi[i1]
    mix_L = np.full((D, ), amplitude)
    mix_R = np.full((D, ), amplitude)
    if b_IdL == comm.rank:
        mix_L[IdL_b] = 1. if not node_local.H.explicit_plus_hc else 0.5
        mix_R[IdL_b] = 0.
    if b_IdR == comm.rank:
        mix_L[IdR_b] = 0.
        mix_R[IdR_b] = 1. if not node_local.H.explicit_plus_hc else 0.5
    # TODO: optimize to minimize transpose
    if update_LP:
        LHeff = node_local.distributed[LH_key]
        rho_L = npc.tensordot(LHeff, theta, axes=['(vR.p0*)', '(vL.p0)'])
        rho_L.ireplace_label('(vR*.p0)', '(vL.p0)')
        rho_L_c = rho_L.conj()
        rho_L.iscale_axis(mix_L, 'wR')
        rho_L = npc.tensordot(rho_L, rho_L_c, axes=[['wR', '(p1.vR)'], ['wR*', '(p1*.vR*)']])
        rho_L = comm.reduce(rho_L, op=MPI.SUM)
        if comm.rank == 0:
            if node_local.H.explicit_plus_hc:
                rho_L = rho_L + rho_L.conj().itranspose()
            if b_IdL == -1:
                rho_L = rho_L + npc.tensordot(theta, theta.conj(), axes=['(p1.vR)', '(p1*.vR*)'])
    elif comm.rank == 0:
        rho_L = npc.tensordot(theta, theta.conj(), axes=['(p1.vR)', '(p1*.vR*)'])
    else:
        rho_L = None

    if update_RP:
        RHeff = node_local.distributed[RH_key]
        rho_R = npc.tensordot(theta, RHeff, axes=['(p1.vR)', '(p1*.vL)'])
        rho_R.ireplace_label('(p1.vL*)', '(p1.vR)')
        rho_R_c = rho_R.conj()
        rho_R.iscale_axis(mix_R, 'wL')
        rho_R = npc.tensordot(rho_R, rho_R_c, axes=[['(vL.p0)', 'wL'], ['(vL*.p0*)', 'wL*']])
        rho_R = comm.reduce(rho_R, op=MPI.SUM)
        if comm.rank == 0:
            if node_local.H.explicit_plus_hc:
                rho_R = rho_R + rho_R.conj().itranspose()
            if b_IdR == -1:
                rho_R = rho_R + npc.tensordot(theta, theta.conj(), axes=['(p1.vR)', '(p1*.vR*)'])
    elif comm.rank == 0:
        rho_R = npc.tensordot(theta, theta.conj(), axes=[['(vL.p0)'], ['(vL*.p0*)']])
    else:
        rho_R = None
    return rho_L, rho_R


def cache_optimize(node_local, on_main, short_term_keys, preload):
    node_local.cache.set_short_term_keys(*short_term_keys, *preload)
    node_local.cache.preload(*preload)


def cache_del(node_local, on_main, *keys):
    for key in keys:
        del node_local.cache[key]

def reverse_interactions(interactions):
    """
    interactions will be a list of lists, with the sublists containing tuples (i,j) such that node i
    needs the environment of node j. These tuples were made for attaching W - R, so we want to reverse
    them for attaching L - W.
    """
    # TODO Fix this; need to reorder list so that sublists are grouped by destination node.
    new_list = []
    for row in interactions:
        new_list.extend([(y,x) for (x,y) in row])
    reversed_interactions = []
    for i in range(len(interactions)):
        reversed_interactions.append([])
        for entry in new_list:
            if entry[0] == i:
                reversed_interactions[-1].append(entry)
    return reversed_interactions

def generate_instructions(interactions, Nnodes):
    """
    Given interactions, we want to generate instructions of which node sends its environment to which
    other nodes. We need to make sure that we do not get stuck in any deadlocks as sending and receiving
    calls may be blocking.
    """
    instructions = []
    interactions = copy.deepcopy(interactions)
    def len_list_list(list_list):
        return np.sum([len(x) for x in list_list])

    while len_list_list(interactions):
        send = []
        receive = []
        round_instructions = []

        for i in range(Nnodes):
            assert i not in receive

            row_interactions = interactions[i]
            for j, inter in enumerate(row_interactions):
                assert inter[0] == i    # Ensure that node i is receiving environment in this interaction.
                if inter[1] not in send:
                    round_instructions.append(inter)
                    send.append(inter[1])
                    receive.append(inter[0])
                    row_interactions.pop(j)
                    break

            if i not in receive and len(row_interactions):
                inter = row_interactions[0]
                assert inter[0] == i    # Ensure that node i is receiving environment in this interaction.
                round_instructions.append(inter)
                send.append(inter[1])
                receive.append(inter[0])
                row_interactions.pop(0)
        instructions.append(round_instructions)     # Each node receives at most one environment.
    # We now have commands separated into rounds. We need to get specific orders for each node.
    commands = []
    for sync in instructions:
        orders = [[] for i in range(Nnodes)]    # One set of orders per node, per round.
        for job in sync:
            if job[0] == job[1]:    # No need to send data
                orders[job[0]].append(("self_contract", None))
            else:
                orders[job[0]].append(('recv', job[1], int(str(job[0]) + str(job[1]))))
                orders[job[1]].append(('send', job[0], int(str(job[0]) + str(job[1]))))
        commands.append(orders)

    return commands
