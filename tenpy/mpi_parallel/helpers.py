"""Helper functions and classes for MPI parallelziation."""
# Copyright 2021 TeNPy Developers, GNU GPLv3


import numpy as np
import copy

"""
try:
    from mpi4py import MPI
except ImportError:
    warnings.warn("mpi4py not installed; MPI parallelization will not work.")
"""

#: blocks/arrays with norm smaller than EPSILON are dropped.
EPSILON = 1.e-14

def sum_none(A, B, _=None):
    assert _ is None  # only needed for compatiblity with mpi4py
    if A is None:
        return B
    if B is None:
        return A
    return A + B


def split_MPO_leg(leg, N_nodes, mpi_split_params):
    """Split MPO leg indices as evenly as possible amongst N_nodes nodes.

    Options
    -------
    .. cfg:config: mpi_split_params

        method : str
            How to split. One of the following options.

            uniform:
                Split the leg into equal-sized slices, ignoring any charge structure.
                This is the only method that works without charge conservation!
            block_slices:
                Divide leg into roughly equal-sized slices, but cut only between charge blocks
                already existing in the `leg`.
            Z_charge_values:
                Split by the charge values of a Z_{N_nodes} charge.
                You need to specify the `Z_charge` as well.
            block_size:
                Distribute by sizes of the charge blocks from large to small giving to
                whichever node has least total so far.
            two-step:
                Use a string ``'first-second'``, e.g. ``'Z_charge_values-block_size'``,
                to split the leg into `N_nodes_first` subsectors with the `first` method,
                and then split each subsector again with the `second` method.
        Z_charge: str | int
            Name or index of the charge for `Z_charge_values` in the
            :class:`~tenpy.linalg.np_conserved.ChargeInfo`. Needs to have `mod` equal to `N_nodes`.
        N_nodes_first: int
            `N_nodes` for the first method of `two-step`.
            We require `N_nodes` to be a multiple of `N_nodes_first`.

    Parameters
    ----------
    leg : :class:`~tenpy.linalg.charges.LegCharge`
        The leg to be split.
    N_nodes : int
        Into how many parts we need to split it.
    mpi_split_params : dict
        Options to determine how to split, see above.
    method :
        One of the following methods for splitting the leg.

    **kwargs :
        Possibly extra keyword arguments depending on `method`.
    """
    D = leg.ind_len
    method = mpi_split_params.get('method', 'uniform')
    if method == 'uniform':
        remaining = D
        running = 0
        res = []
        for i in range(N_nodes):
            proj = np.zeros(D, dtype=bool)
            assigned = int(np.ceil(remaining / (N_nodes - i)))
            proj[running: running + assigned] = True
            res.append(proj)
            remaining -= assigned
            running += assigned
        assert running == D
        return res
    elif method == 'block_slices':
        slices = leg.slices
        running = 0
        res = []
        for i in range(N_nodes):
            proj = np.zeros(D, dtype=bool)
            next_cut = running + int(np.ceil((D - running) / (N_nodes - i)))
            next_cut = slices[np.argmin(np.abs(slices - next_cut))]
            proj[running:next_cut] = True
            res.append(proj)
            running = next_cut
        assert running == D
        return res
    elif method == 'Z_charge_values':
        charge = find_Z_charge(leg.chinfo, mpi_split_params.get('Z_charge', None))
        if leg.chinfo.mod[charge] != N_nodes :
            raise ValueError("Can't split Z_{leg.chinfo.mod[charge]:d} onto {N_nodes} nodes")
        qflat = leg.to_qflat()[:, charge]
        res = []
        for i in range(N_nodes):
            proj = np.zeros(D, dtype=bool)
            proj[qflat == i] = True
            res.append(proj)
        return res
    elif method == 'block_size':
        sizes = leg.get_block_sizes()
        slices = leg.slices
        has_size = [0] * N_nodes
        res = [np.zeros(D, dtype=bool) for _ in range(N_nodes)]
        for block in np.argsort(sizes)[::-1]:
            to_node = np.argmin(has_size)
            res[to_node][leg.slices[block]: leg.slices[block + 1]] = True
            has_size[to_node] += sizes[block]
        return res
    elif '-' in method:
        first_method, second_method = method.split('-')
        N_nodes_first = mpi_split_params['N_nodes_first']
        assert N_nodes % N_nodes_first == 0
        N_nodes_second = N_nodes // N_nodes_first
        mpi_split_params['method'] = first_method
        projs_first = split_MPO_leg(leg, N_nodes_first, mpi_split_params)
        mpi_split_params['method'] = second_method
        res = []
        for p_first in projs_first:
            _, _, p_leg = leg.project(p_first)
            projs_second = split_MPO_leg(p_leg, N_nodes_second, mpi_split_params)
            for p_second in projs_second:
                proj_full = np.zeros(D, dtype=bool)
                proj_full[p_first] = p_second
                res.append(proj_full)
        mpi_split_params['method'] = method
        return res
    else:
        raise ValueError(f"Unknown method={method!r}")

def find_Z_charge(chinfo, charge=None):
    if charge is None:
        if np.sum(chinfo.mod > 1) == 1:
            charge = np.nonzero(chinfo.mod > 1)[0][0]
        else:
            raise ValueError("You need to specify the name of the `Z_charge`")
    if isinstance(charge, str):
        charge = chinfo.names.index(charge)
    assert chinfo.mod[charge] > 1
    return charge


def index_in_blocks(block_projs, index):
    if index is None:
        return (-1, None)
    for j, proj in enumerate(block_projs):
        if proj[index]:
            return (j, np.sum(proj[:index]))  # (block index,  index within block)
    assert False, "None of block_projs has `index` True"


def split_W_into_blocks(W, projs_L, projs_R, labels=['wL', 'wR']):
    assert len(projs_L) == len(projs_R)
    blocks = []
    for p_L in projs_L:
        row = []
        for p_R in projs_R:
            Wblock = W.copy()
            Wblock.iproject([p_L, p_R], labels)
            if Wblock.norm() < EPSILON:
                Wblock = None
            row.append(Wblock)
        # note: for (MPO bond dim) < N (which is legit for finite at the boundaries),
        # some rows/columns of Wblock can be all None.
        blocks.append(row)
    return blocks


def build_sparse_comm_schedule(W_blocks, on_node=None):
    """Find schedule for multiplying W_block with RP.

    Returns
    -------
    schedule : list of list of (int, int)
    """
    N = len(W_blocks)  # number of nodes
    nonzero_inds = [[c for c, W in enumerate(row) if W is not None] for row in W_blocks]
    schedule = []
    while any(len(inds) > 0 for inds in nonzero_inds):
        comms = [] # (dest, source)
        # which node needs to communicate with whom this round?
        # try to balance sends/recieves on each node with greedy minimization
        comms_on_node = np.zeros(N, int)
        for dest, needs_from in enumerate(nonzero_inds):
            if len(needs_from) == 0:
                continue
            i = np.argmin([comms_on_node[i] for i in needs_from])
            source = needs_from.pop(i)
            if dest != source:
                comms_on_node[dest] += 1
                comms_on_node[source] += 1
            comms.append((dest, source))
        # now sort those communications such that they can happen in parallel as much as possible
        # for example [(0, 1), (1, 2), (2, 3), (3, 0)] => [(0, 1), (2, 3), (1, 2), (3, 0)]
        comms_sorted = []
        delay_on_node = np.zeros(N, int)
        while comms:
            delay = [max(delay_on_node[t], delay_on_node[s]) for (t, s) in comms]
            i = np.argmin(delay)
            t, s = comms.pop(i)
            if t != s:
                delay_on_node[t] = delay_on_node[s] = delay[i] + 1
            comms_sorted.append((t, s))
        if on_node is not None:
            comms_sorted = [(t, s) for (t, s) in comms_sorted if t == on_node or s == on_node]
        schedule.append(comms_sorted)
    return schedule


########### NPC - MPI4PY Communication ###########

"""
def npc_send(comm, array, dest, tag):
    if array is None or array.stored_blocks == 0:
        comm.isend(array, dest=dest, tag=tag).wait()
        return
    
    #array_data_orig = array._data #[:] #Slice list to make a shallow copy? Seems to be faster than list.copy()
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
    #print('Here')
    #print(send_array._data, array._data, array_data_orig)
    request = comm.isend(send_array, dest=dest, tag=tag)
    #array._data = array_data_orig

    requests = [MPI.REQUEST_NULL] * array.stored_blocks
    #requests = [MPI.REQUEST_NULL] * len(array_data_orig)

    #print(array.stored_blocks, len(array_data_orig))
    #for i in range(array.stored_blocks):
    for i, d in enumerate(array._data):
        #requests[i] = comm.Isend(np.ascontiguousarray(array._data[i]), dest=dest, tag=tag+i)
        requests[i] = comm.Isend(np.ascontiguousarray(d), dest=dest, tag=tag+i)

    MPI.Request.Waitall(requests + [request])
    #array._data = array_data_orig
    
    #for d in array._data:
    #    assert type(d) is np.ndarray
    #print(array._data)
    return array
"""

"""
def npc_send(comm, array, dest, tag):
    if array is None or array.size == 0:
        comm.send(array, dest=dest, tag=tag)
        return
    
    array_data_orig = array._data
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

    array._data = [block_shapes] + [array.dtype]
    comm.send(array, dest=dest, tag=tag)
    #buf = MPI.Alloc_mem(1<<20)
    #MPI.Attach_buffer(buf)
    #comm.bsend(array, dest=dest, tag=tag)
    #MPI.Detach_buffer()
    #MPI.Free_mem(buf)
    array._data = array_data_orig

    
    if array.stored_blocks:
        requests = [MPI.REQUEST_NULL] * array.stored_blocks
        #buf = MPI.Alloc_mem(array.size*16+MPI.BSEND_OVERHEAD)
        print(comm.Get_rank(), array.size, array.stored_blocks, array.size*dtype_size+MPI.BSEND_OVERHEAD*array.stored_blocks)
        MPI.Attach_buffer(MPI.Alloc_mem(array.size*dtype_size+MPI.BSEND_OVERHEAD))
        for i in range(array.stored_blocks):
            requests[i] = comm.Ibsend(np.ascontiguousarray(array._data[i]), dest=dest, tag=tag+i)

        MPI.Request.Waitall(requests)
        MPI.Detach_buffer()
        #MPI.Free_mem(buf)
    
    
    requests = [MPI.REQUEST_NULL] * array.stored_blocks
    for i in range(array.stored_blocks):
        requests[i] = comm.Isend(np.ascontiguousarray(array._data[i]), dest=dest, tag=tag+i)

    MPI.Request.Waitall(requests)
"""    

"""
def npc_recv(comm, source, tag):
    request = comm.irecv(bytearray(1<<20), source=source, tag=tag) # Assume shell npc array is less than 1 MB in size.
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
    #for d in array._data:
    #    assert type(d) is np.ndarray

    return array
"""

"""
def npc_recv(comm, source, tag):
    array = comm.recv(bytearray(1<<20), source=source, tag=tag) # Assume shell npc array is less than 1 MB in size.
    if array is None or array.size == 0:
        return array
    
    block_shapes = array._data[0]
    dtype = array._data[1]
    array._data = []

    if len(block_shapes):
        requests = [MPI.REQUEST_NULL] * len(block_shapes)

        for i, b_s in enumerate(block_shapes):
            array._data.append(np.empty(b_s, dtype=dtype))
            requests[i] = comm.Irecv(array._data[-1], source=source, tag=tag+i)

        MPI.Request.Waitall(requests)
    return array
"""