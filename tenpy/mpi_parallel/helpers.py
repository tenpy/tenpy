"""Helper functions and classes for MPI parallelziation."""
# Copyright 2021 TeNPy Developers, GNU GPLv3


import numpy as np
import copy


#: blocks/arrays with norm smaller than EPSILON are dropped.
EPSILON = 1.e-14

def sum_none(A, B, _=None):
    assert _ is None  # only needed for compatiblity with mpi4py
    if A is None:
        return B
    if B is None:
        return A
    return A + B


def split_MPO_leg(leg, N_nodes):
    """Split MPO leg indices as evenly as possible amongst N_nodes nodes.

    Ensure that lower numbered legs have legs.
    """
    # TODO: make this more clever
    D = leg.ind_len
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
