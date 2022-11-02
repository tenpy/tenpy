"""Functions that are called on all MPI nodes in parallel.

The replica nodes are set up to call :func:`replica_main`, which is an infinite loop waiting for
the main node to call :func:`run`, specifiying which of the functions in this module they should
execute. All of the nodes (including main) will execute this function, followed by the replica
nodes waiting for the next instruction and the main node continuing with whatever it did before.

Only functions in this module are valid "actions", but you can add new functions to this module
if necessary; from outside tenpy just
``tenpy.mpi_parallel.actions.my_new_action = my_new_function``.
"""
# Copyright 2021 TeNPy Developers, GNU GPLv3

import numpy as np
import copy
import warnings


from ..linalg import np_conserved as npc
from ..tools.hdf5_io import save_to_hdf5, load_from_hdf5
from ..tools.misc import transpose_list_list
from . import helpers

DONE = None  # sentinel to say that the worker should finish


try:
    from mpi4py import MPI
except ImportError:
    warnings.warn("mpi4py not installed; MPI parallelization will not work.")
else:
    MPI_SUM_NONE = MPI.Op.Create(helpers.sum_none, commute=True)


def run(action, node_local, meta, on_main=None):
    """Special action to call other actions.

    Calling ``action.run(action.some_function, node_local, ...)`` will execute `some_function`
    (which needs to be a function in this module) on all nodes.
    """
    if action is DONE:
        action_name = "DONE"
    else:
        action_name = action.__name__
    assert globals()[action_name] is action
    node_local.comm.bcast((action_name, meta))
    return action(node_local, on_main, *meta)


def replica_main(node_local):
    """The main loop for replica nodes.

    They wait for the main node to call ``action.run(action.some_function, ...)``
    and then execute that function.
    The loop terminates when the main node broadcasts `DONE` as action.
    """
    while True:
        action_name, meta = node_local.comm.bcast(None)
        if action_name is DONE or action_name == "DONE":
            return  # gracefully terminate
        action = globals()[action_name]
        action(node_local, None, *meta)


def distr_array_scatter(node_local, on_main, key, in_cache):
    # on_main should be list of npc_arrays
    local_part = node_local.comm.scatter(on_main)
    if in_cache:
        node_local.cache[key] = local_part
    else:
        node_local.distributed[key] = local_part


def distr_array_gather(node_local, on_main, key, in_cache):
    local_part = node_local.cache[key] if in_cache else node_local.distributed[key]
    return node_local.comm.gather(local_part)


def distr_array_save_hdf5(node_local, on_main, key, in_cache, filename_template, hdf5_key):
    fn = filename_template.format(mpirank=node_local.comm.rank)
    f = getattr(node_local, 'hdf5_export_file', None)
    if f is None:
        import h5py
        f = h5py.File(fn, 'w')
        node_local.hdf5_export_file = f
    else:
        assert f.filename == fn

    local_part = node_local.cache[key] if in_cache else node_local.distributed[key]
    save_to_hdf5(f, local_part, hdf5_key)


def distr_array_keep_alive(node_local, on_main, key, in_cache):
    """Get reference to the `DistributedArray.local_part` that survives deleting the cache."""
    # generate *global* class attribute
    if hasattr(node_local.__class__, '_keep_alive'):
        keep_alive = node_local.__class__._keep_alive
    else:
        node_local.__class__._keep_alive = keep_alive = {}
    keep_alive[key] = node_local.cache[key] if in_cache else node_local.distributed[key]

    
def split_redistr_array_load_hdf5(node_local, on_main, key, in_cache, filename_template, hdf5_key, projs, N_old, N_new, label, boundary_leg):
    #print('Node', node_local.comm.rank, ' is ready to split env.', flush=True)
    if node_local.comm.rank  == 0:
        # Load the old environments all on the root node
        local_parts = []
        for o_r in range(N_old):
            if filename_template is not None:
                fn = filename_template.format(mpirank=o_r)
                f = getattr(node_local, 'hdf5_import_file', None) # Should always be none since we open multiple files.
                assert f is None
                import h5py
                f = h5py.File(fn, 'r')
                local_parts.append(load_from_hdf5(f, hdf5_key))
                f.close()

                print('Root node loaded partial env indexed', o_r, 'with shape', local_parts[-1].shape, flush=True)

            else:
                # in sequential simulations after `DistributedArray._keep_alive_beyond_cache()`
                raise NotImplementedError("Should not occur with sequential simulations; node number should be constant.")
        
        print('MPO dimensions of original envs:', [lp.get_leg(label).ind_len for lp in local_parts])
        
        whole_env = npc.concatenate(local_parts, axis=label).sort_legcharge(True)[1]
        env_leg = whole_env.get_leg(label)
        env_leg.test_equal(boundary_leg)
        
        new_local_parts = []
        for pj in projs:
            new_lp = whole_env.copy(deep=False)
            new_lp.iproject(mask=pj, axes=label)
            new_local_parts.append(new_lp)
        print('MPO dimensions of redistributed envs:', [lp.get_leg(label).ind_len for lp in new_local_parts])

        local_part = new_local_parts[0]
        
        for i in range(1, N_new):
            dest = i
            source = 0
            # Can't use scatter since the envs may be larger than 2.1GB and thus cannot be pickled
            npc_send(node_local.comm, new_local_parts[i], dest, tag = dest*10000 + source)
        
    else:
        # Get redistributed env from root node
        source = 0
        #print('Node', node_local.comm.rank, 'is receiving from node', source, flush=True)
        local_part = npc_recv(node_local.comm, source, tag = node_local.comm.rank*10000 + source)
        
    print('Node', node_local.comm.rank, 'has local env with shape', local_part.shape, flush=True)
    if in_cache:
        node_local.cache[key] = local_part
    else:
        node_local.distributed[key] = local_part
    

def split_distr_array_load_hdf5(node_local, on_main, key, in_cache, filename_template, hdf5_key, projs, N_old, N_new, label):
    #print('Node', node_local.comm.rank, ' is ready to split env.', flush=True)
    if node_local.comm.rank % (N_new // N_old) == 0:
        # Load the old environments
        old_rank = node_local.comm.rank //  (N_new // N_old)
        if filename_template is not None:
            fn = filename_template.format(mpirank=old_rank)
            f = getattr(node_local, 'hdf5_import_file', None)
            if f is None:
                import h5py
                f = h5py.File(fn, 'r')
                node_local.hdf5_import_file = f
            else:
                assert f.filename == fn, 'Conflict between ' + f.filename + ' and ' + fn + '.'
                
            whole_local_part = load_from_hdf5(f, hdf5_key)
            print('Node', node_local.comm.rank, 'loaded env with shape ', whole_local_part.shape, flush=True)
            
            for i in range(N_new // N_old):
                dest = node_local.comm.rank + i
                source = node_local.comm.rank
                
                split_local_part = whole_local_part.copy(deep=True)
                #print(source, dest, len(projs[dest]), np.sum(projs[dest]), flush=True)
                split_local_part.iproject(mask=projs[dest], axes=label)
                if dest != source:
                    #print('Node', node_local.comm.rank, ' is sending to node', dest, flush=True)
                    npc_send(node_local.comm, split_local_part, dest, tag = dest*10000 + source)
                else:
                    local_part = split_local_part
        else:
            # in sequential simulations after `DistributedArray._keep_alive_beyond_cache()`
            raise NotImplementedError("Should not occur with sequential simulations; node number should be constant.")
    else:
        # Get split env from node which loaded and split it
        source = node_local.comm.rank - node_local.comm.rank % (N_new // N_old)
        #print('Node', node_local.comm.rank, 'is receiving from node', source, flush=True)
        local_part = npc_recv(node_local.comm, source, tag = node_local.comm.rank*10000 + source)
        
    print('Node', node_local.comm.rank, 'has local env with shape', local_part.shape, flush=True)
    if in_cache:
        node_local.cache[key] = local_part
    else:
        node_local.distributed[key] = local_part

    
def distr_array_load_hdf5(node_local, on_main, key, in_cache, filename_template, hdf5_key):
    if filename_template is not None:
        fn = filename_template.format(mpirank=node_local.comm.rank)
        f = getattr(node_local, 'hdf5_import_file', None)
        if f is None:
            import h5py
            f = h5py.File(fn, 'r')
            node_local.hdf5_import_file = f
        else:
            assert f.filename == fn, 'Conflict between ' + f.filename + ' and ' + fn + '.'

        local_part = load_from_hdf5(f, hdf5_key)
    else:
        # in sequential simulations after `DistributedArray._keep_alive_beyond_cache()`
        local_part = node_local.__class__._keep_alive.pop(key)
        if len(node_local.__class__._keep_alive) == 0:
            del node_local.__class__._keep_alive
    if in_cache:
        node_local.cache[key] = local_part
    else:
        node_local.distributed[key] = local_part


def node_local_close_hdf5_file(node_local, on_main, attr_name):
    f = getattr(node_local, attr_name, None)
    if f is not None:
        f.close()
        delattr(node_local, attr_name)


def distribute_H(node_local, on_main, H, mpi_split_params):
    """Distribute and split H and prepare communication schedules."""
    node_local.H = H
    comm = node_local.comm
    N = comm.size
    node_local.W_blocks = []
    node_local.projs_L = []
    node_local.IdLR_blocks = []
    node_local.local_MPO_chi = []
    node_local.sparse_comm_schedule = []
    for bond in range(H.L if not H.finite else H.L + 1):
        if bond != H.L:
            leg = H.get_W(bond).get_leg('wL')
        else:
            leg = H.get_W(H.L - 1).get_leg('wR')
        projs = helpers.split_MPO_leg(leg, N, mpi_split_params)
        node_local.projs_L.append(projs)
        node_local.local_MPO_chi.append(np.sum(projs[comm.rank]))
        IdL, IdR = H.IdL[bond], H.IdR[bond]
        IdLR = (helpers.index_in_blocks(projs, IdL), helpers.index_in_blocks(projs, IdR))
        node_local.IdLR_blocks.append(IdLR)
    for i in range(H.L):
        W_blocks = helpers.split_W_into_blocks(H.get_W(i),
                                               projs_L=node_local.projs_L[i],
                                               projs_R=node_local.projs_L[(i+1) % H.L])
        node_local.W_blocks.append(W_blocks)
        schedule_RP = helpers.build_sparse_comm_schedule(W_blocks, comm.rank)
        schedule_LP = helpers.build_sparse_comm_schedule(transpose_list_list(W_blocks), comm.rank)
        node_local.sparse_comm_schedule.append((schedule_LP, schedule_RP))


def matvec(node_local, on_main, theta, LH_key, RH_key):
    LHeff = node_local.distributed[LH_key]
    RHeff = node_local.distributed[RH_key]
    if LHeff is None or RHeff is None:
        # (can be that only one of them is None...)
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
    return node_local.comm.reduce(theta, op=MPI_SUM_NONE)


def matvec_plain(LHeff, theta, RHeff):
    """Not an action, but a helper function called from `matvec`."""
    theta = npc.tensordot(LHeff, theta, axes=['(vR.p0*)', '(vL.p0)'])
    theta = npc.tensordot(theta, RHeff, axes=[['wR', '(p1.vR)'], ['wL', '(p1*.vL)']])
    theta.ireplace_labels(['(vR*.p0)', '(p1.vL*)'], ['(vL.p0)', '(p1.vR)'])
    return theta


def matvec_hc(LHeff, theta, RHeff):
    """Not an action, but a helper function called from `matvec`."""
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
    return node_local.comm.reduce(contr, op=MPI_SUM_NONE)

def npc_send(comm, array, dest, tag):
    if array is None or array.stored_blocks == 0:
        comm.isend(array, dest=dest, tag=tag).wait()
        return
    
    try:
        block_shapes = [d.shape for d in array._data]
    except AttributeError:
        # array._data is somehow a list of lists?
        # Sometimes array._data is [blocK_shapes] + [array.dtype]; I think this happened when the the pickled object 
        # was sent using isend and not waited upon before changing array._data
        print(array._data)
        print(len(array._data))
        print([len(d) for d in array._data])
        raise ValueError

    if array.dtype == 'complex128':
        dtype_size = 16
    elif array.dtype == 'float64':
        dtype_size = 8
    else:
        raise ValueError('dtype %s of environment not recognized' % array.dtype)
    
    send_array = array.copy() # Make shallow copy of npc array and ONLY change copy.
    send_array._data = [block_shapes] + [array.dtype]
    request = comm.isend(send_array, dest=dest, tag=tag)

    requests = [MPI.REQUEST_NULL] * array.stored_blocks
    for i, d in enumerate(array._data):
        requests[i] = comm.Isend(np.ascontiguousarray(d), dest=dest, tag=tag+i)

    MPI.Request.Waitall(requests + [request])
    return array

def npc_recv(comm, source, tag):
    request = comm.irecv(bytearray(1<<22), source=source, tag=tag) # Assume shell npc array is less than 4 MB in size.
    array = request.wait()
    if array is None or array.stored_blocks == 0:
        return array
    
    block_shapes = array._data[0]
    dtype = array._data[1]
    array._data = []

    requests = [MPI.REQUEST_NULL] * len(block_shapes)

    for i, b_s in enumerate(block_shapes):
        array._data.append(np.empty(b_s, dtype=dtype))
        requests[i] = comm.Irecv(array._data[-1], source=source, tag=tag+i)

    MPI.Request.Waitall(requests)
    return array


def contract_LP_W_sparse(node_local, on_main, i, old_key, new_key):
    assert on_main is None
    comm = node_local.comm
    my_LP = node_local.cache[old_key]
    LHeff = None
    rank = comm.rank
    i = i % node_local.H.L
    W = node_local.W_blocks[i]
    for cycle in node_local.sparse_comm_schedule[i][0]:
        received_LP = None
        # first communicate everything
        for dest, source in cycle:
            tag = dest * 10000 + source
            if dest == rank:
                assert received_LP is None  # shouldn't receive 2 envs per cycle
                if source == rank:
                    received_LP = my_LP  # no communication necessary
                else:
                    #received_LP = comm.recv(source=source, tag=tag)
                    received_LP = npc_recv(comm, source, tag)
                Wb = W[source][dest].replace_labels(['p', 'p*'], ['p0', 'p0*'])
            elif source == rank:
                #comm.send(my_LP, dest=dest, tag=tag)
                my_LP = npc_send(comm, my_LP, dest, tag)
            else:
                assert False, f"Invalid cycle on node {rank:d}: {cycle!r}"
        # now every node (which has work left) received one part
        if received_LP is not None:
            try:
                LHeff = helpers.sum_none(LHeff, npc.tensordot(received_LP, Wb, ['wR', 'wL']))
            except TypeError:
                print(rank, dest, source, flush=True)
                raise ValueError
        #comm.Barrier()  # TODO: probably not needed?
    if LHeff is not None:
        pipeL = LHeff.make_pipe(['vR*', 'p0'], qconj=+1)
        LHeff = LHeff.combine_legs([['vR*', 'p0'], ['vR', 'p0*']], pipes=[pipeL, pipeL.conj()], new_axes=[0, 2]) # vR*.p, wR, vR.p*
    node_local.distributed[new_key] = LHeff


def contract_W_RP_sparse(node_local, on_main, i, old_key, new_key):
    assert on_main is None
    comm = node_local.comm
    my_RP = node_local.cache[old_key]
    RHeff = None
    rank = comm.rank
    i = i % node_local.H.L
    W = node_local.W_blocks[i]
    for cycle in node_local.sparse_comm_schedule[i][1]:
        received_RP = None
        # first communicate everything
        for dest, source in cycle:
            tag = dest * 10000 + source
            if dest == rank:
                assert received_RP is None  # shouldn't receive 2 envs per cycle
                if source == rank:
                    received_RP = my_RP  # no communication necessary
                else:
                    #received_RP = comm.recv(source=source, tag=tag)
                    received_RP = npc_recv(comm, source, tag)
                Wb = W[dest][source].replace_labels(['p', 'p*'], ['p1', 'p1*'])
            elif source == rank:
                #comm.send(my_RP, dest=dest, tag=tag)
                my_RP = npc_send(comm, my_RP, dest, tag)
            else:
                assert False, f"Invalid cycle on node {rank:d}: {cycle!r}"
        # now every node (which has work left) received one part
        if received_RP is not None:
            try:
                RHeff = helpers.sum_none(RHeff, npc.tensordot(Wb, received_RP, ['wR', 'wL']))
            except TypeError:
                print(rank, dest, source, flush=True)
                raise ValueError
        #comm.Barrier()  # TODO: probably not needed?
    if RHeff is not None:
        # TODO: is it faster to combine legs before addition?
        pipeR = RHeff.make_pipe(['p1', 'vL*'], qconj=-1)
        RHeff = RHeff.combine_legs([['p1', 'vL*'], ['p1*', 'vL']], pipes=[pipeR, pipeR.conj()], new_axes=[2, 1])
    node_local.distributed[new_key] = RHeff


def attach_B(node_local, on_main, old_key, new_key, B):
    local_part = node_local.distributed[old_key]
    if local_part is not None:
        local_part = npc.tensordot(B, local_part, axes=['(p.vR)', '(p1*.vL)'])
        local_part = npc.tensordot(local_part, B.conj(), axes=['(p1.vL*)', '(p*.vR*)'])
    node_local.cache[new_key] = local_part


def attach_A(node_local, on_main, old_key, new_key, A):
    local_part = node_local.distributed[old_key]
    if local_part is not None:
        local_part = npc.tensordot(local_part, A, axes=['(vR.p0*)', '(vL.p)'])
        local_part = npc.tensordot(A.conj(), local_part, axes=['(vL*.p*)', '(vR*.p0)'])
    node_local.cache[new_key] = local_part


def full_contraction(node_local, on_main, case, LP_key, LP_ic, RP_key, RP_ic, theta):
    LP = node_local.cache[LP_key] if LP_ic else node_local.distributed[LP_key]
    RP = node_local.cache[RP_key] if RP_ic else node_local.distributed[RP_key]
    if LP is None or RP is None:
        full_contr = None
        # note: can happen that only one is None
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
        if LHeff is None:
            rho_L = None
        else:
            rho_L = npc.tensordot(LHeff, theta, axes=['(vR.p0*)', '(vL.p0)'])
            rho_L.ireplace_label('(vR*.p0)', '(vL.p0)')
            rho_L_c = rho_L.conj()
            rho_L.iscale_axis(mix_L, 'wR')
            rho_L = npc.tensordot(rho_L, rho_L_c, axes=[['wR', '(p1.vR)'], ['wR*', '(p1*.vR*)']])
        rho_L = comm.reduce(rho_L, op=MPI_SUM_NONE)
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
        if RHeff is None:
            rho_R = None
        else:
            rho_R = npc.tensordot(theta, RHeff, axes=['(p1.vR)', '(p1*.vL)'])
            rho_R.ireplace_label('(p1.vL*)', '(p1.vR)')
            rho_R_c = rho_R.conj()
            rho_R.iscale_axis(mix_R, 'wL')
            rho_R = npc.tensordot(rho_R, rho_R_c, axes=[['(vL.p0)', 'wL'], ['(vL*.p0*)', 'wL*']])
        rho_R = comm.reduce(rho_R, op=MPI_SUM_NONE)
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
