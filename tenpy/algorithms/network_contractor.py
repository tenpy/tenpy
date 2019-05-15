"""
Network Contractor

A tool to contract a network of multiple tensors.

This is an implementation of 'NCON: A tensor network contractor for MATLAB'
by Robert N. C. Pfeifer, Glen Evenbly, Sukhwinder Singh, Guifre Vidal, see :arxiv:`1402.0939`

.. todo ::
    - implement or wrap netcon.m, a function to find optimal contractionn sequences
        (:arxiv:`1304.6112`)
    - improve helpfulness of Warnings
    - _do_trace: trace over all pairs of legs at once. need the corresponding npc function first.

"""
# Copyright 2018 TeNPy Developers

import numpy as np
import warnings
import collections
from ..linalg import np_conserved as npc

__all__ = ['outer_product', 'contract']

outer_product = -66666666  # a constant that represents an outer product in the sequence of ncon


def contract(tensor_list, tensor_names=None, leg_contractions=None, open_legs=None, sequence=None):
    """

    Contract a network of tensors.
    Based on the MatLab function ncon.m as described in :arxiv:`1402.0939`

    Parameters
    ----------
    tensor_list : list of :class:`~tenpy.linalg.np_conserved.Array`
        The tensors to be contracted.
    leg_contractions : list of ``[n1, l1, n2, l2]``
        A list of contraction instructions. An entry of leg_contractions has the form
        ``[n1, l1, n2, l2]``, where ``n1, n2`` are entries of `tensor_names` and each identify an
        :class:`~tenpy.linalg.np_conserved.Array` in `tensor_list`.
        ``l1, l2`` are leg labels of the corresponding :class:`~tenpy.linalg.np_conserved.Array`.
        The instruction implies to contract leg ``l1`` of tensor ``n1``
        with leg ``l2`` of tensor ``n2``.
    open_legs : list of ``[n1, l1, l]``
        A list of instructions for "open" (uncontracted) legs.
        ``[n1, l1, l]`` implies that leg ``l1`` of tensor ``n1`` is not contracted
        and is labelled ``l`` in the result.
    tensor_names : list of str
        A list of names for each tensor, to be used in `leg_contractions` and `open_legs`.
        The default value is list(range(len(tensor_list))), so that the tensor "names" are
        ``0, 1, 2, ...``.
    sequence : list of int
        The order in which the leg_contractions are to be performed.
        An entry of network_contractor.outer_product indicates performing an outer product.
        This corresponds to the zero-in-sequence convention of :arxiv:`1304.6112`

    Returns
    -------
    result : :class:`Array` | complex
        The number or tensor resulting from the contraction.

    """

    if leg_contractions is None:
        leg_contractions = []
    if open_legs is None:
        open_legs = []

    # translate tensor names to the numbers used for indexing tensor_list
    for n in range(len(leg_contractions)):
        leg_contractions[n][0] = tensor_names.index(leg_contractions[n][0])
        leg_contractions[n][2] = tensor_names.index(leg_contractions[n][2])
    for n in range(len(open_legs)):
        open_legs[n][0] = tensor_names.index(open_legs[n][0])

    # default sequence
    if sequence is None:
        sequence = list(range(len(leg_contractions)))

    # translate leg_contractions and open_legs to a leg_links list as used by _ncon
    # initialise leg_links
    leg_links = []
    for tensor in tensor_list:
        leg_links.append([None] * len(tensor.legs))

    # fill in the contractions
    contraction_counter = 0
    new_sequence = []
    for n in sequence:
        if n == outer_product:
            new_sequence.append(outer_product)
            continue

        con = leg_contractions[n]
        leg_idx1 = tensor_list[con[0]].get_leg_index(con[1])
        leg_idx2 = tensor_list[con[2]].get_leg_index(con[3])

        if leg_links[con[0]][leg_idx1] is not None:
            raise RuntimeError('Multiple contradictory contractions for the leg ' + str(con[1]) +
                               ' of tensor ' + str(tensor_names[con[0]]) + 'were supplied')
        if leg_links[con[2]][leg_idx2] is not None:
            raise RuntimeError('Multiple contradictory contractions for the leg ' + str(con[3]) +
                               ' of tensor ' + str(tensor_names[con[2]]) + 'were supplied')

        leg_links[con[0]][leg_idx1] = contraction_counter
        leg_links[con[2]][leg_idx2] = contraction_counter
        new_sequence.append(contraction_counter)
        contraction_counter = contraction_counter + 1

    # fill in open legs
    final_labels = []
    open_leg_counter = -1
    for entry in open_legs:
        leg_idx = tensor_list[entry[0]].get_leg_index(entry[1])
        leg_links[entry[0]][leg_idx] = open_leg_counter
        open_leg_counter = open_leg_counter - 1
        final_labels.append(entry[2])

    # call _ncon and relabel the results' legs
    res = _ncon(tensor_list, leg_links, new_sequence)
    if len(final_labels) > 0:
        res.iset_leg_labels(final_labels)

    return res


def _ncon(tensor_list, leg_links, sequence):
    """Helper function for contract.

    _ncon is a python implementation of ncon.m (:arxiv:`1304.6112`) for tenpy :class:'Array's
    ncon is a wrapper that translates from a more python/tenpy input style

    Parameters
    ----------
    tensor_list : list of :class:'Array'
        Tensors to be contracted.
    leg_links : list of list of int
        Each entry of leg_links describes the connectivity of the corresponding tensor in
        `tensor_list`.
        Each entry is a list that has an entry for each leg of the corresponding tensor.
        Values ``0,1,2,...`` are labels of contracted legs and should appear
        exactly twice in `leg_links`.
        Values ``-1,-2,-3,...`` are labels of uncontracted legs and indicate the final ordering
        (``-1`` is first axis).
    sequence : list of int
        The order in which the contractions are to be performed.
        An entry of network_contractor.outer_product indicates performing an outer product.
        This corresponds to the zero-in-sequence convention of :arxiv:`1304.6112`

    Returns
    -------
    result : :class:`Array` | complex
        The number or tensor resulting from the contraction.

    """

    tensor_list = list(tensor_list)

    # check contractibility of legs?

    def flatten(l):
        return [item for sublist in l for item in sublist]

    while len(tensor_list) > 1 or any(i >= 0 for i in flatten(leg_links)):
        if len(sequence) == 0:
            sequence = [outer_product] * (len(tensor_list) - 1)
        if sequence[0] == outer_product:
            # outer product
            tensor_list, leg_links, sequence = _outer_product(tensor_list, leg_links, sequence)
        else:
            # identify and perform tensor contraction
            # find tensors that the index sequence[0] corresponds to
            tensors = []
            for a in range(len(leg_links)):
                if sequence[0] in leg_links[a]:
                    tensors.append(a)

            if len(tensors) == 1:
                # its a trace

                # find all traced indices on that tensor
                traced_indices = np.sort(leg_links[tensors[0]])
                traced_indices = [
                    item for item, count in collections.Counter(traced_indices).items()
                    if count == 2
                ]

                # check if this is in accordance with sequence and update sequence
                doing_traces, sequence = _find_in_sequence(traced_indices, sequence)
                if not np.array_equal(np.sort(traced_indices), np.sort(doing_traces)):
                    warnings.warn(
                        'Suboptimal contraction sequence. When tracing legs ' + str(doing_traces) +
                        ' the legs ' +
                        str(list(filter(lambda n: n not in doing_traces, traced_indices))) +
                        'should also be traced')
                # TODO translate this back to human readable leg label?

                # perform traces
                tensor_list[tensors[0]], leg_links[tensors[0]] = \
                    _do_trace(tensor_list[tensors[0]], leg_links[tensors[0]], doing_traces)

                # update leg links
                for idx in doing_traces:
                    leg_links[tensors[0]] = list(filter(lambda b: b != idx, leg_links[tensors[0]]))

            else:
                # its a contraction

                # find all other contracted legs between the two tensors
                common_indices = []
                for idx in leg_links[tensors[0]]:
                    if idx in leg_links[tensors[1]]:
                        common_indices.append(idx)

                # check if this is in accordance with sequence
                contraction_indices, sequence = _find_in_sequence(common_indices, sequence)
                if not np.array_equal(np.sort(contraction_indices), np.sort(common_indices)):
                    warnings.warn(
                        'Suboptimal contraction sequence. When contracting legs ' +
                        str(contraction_indices) + ' the legs ' +
                        str(list(filter(lambda n: n not in contraction_indices, common_indices))) +
                        'should also be traced')
                    # TODO translate this back to human readable leg names

                # are there traced indices on either of these tensors?
                # noinspection PyArgumentList
                traces0 = [
                    item for item, count in collections.Counter(leg_links[tensors[0]]).items()
                    if count == 2
                ]
                traces1 = [
                    item for item, count in collections.Counter(leg_links[tensors[1]]).items()
                    if count == 2
                ]
                if len(traces0) + len(traces1) != 0:
                    warnings.warn(
                        'Suboptimal contraction sequence. When processing ' + str(None) +
                        ' one of the involved tensors have legs that could be traced over first')
                    # TODO human readable identifier
                # contract all contraction_indices and update leg_links
                tensor_list[tensors[0]], leg_links[tensors[0]] = \
                    _tcontract(tensor_list[tensors[0]], tensor_list[tensors[1]], leg_links[tensors[0]],
                               leg_links[tensors[1]], contraction_indices)
                tensor_list.pop(tensors[1])
                leg_links.pop(tensors[1])
    assert len(tensor_list) == 1
    return tensor_list[0]


def _find_in_sequence(indices, sequence):
    """Helper function for _ncon

    check if the supplied indices appear at the beggining of sequence

    Parameters
    ----------
    indices : list of int
    sequence : list of int

    Returns
    -------
    idcs : list of int
        All the given indices that appear consecutively at the front of sequence

    """

    ptr = 0
    while ptr < len(sequence) and sequence[ptr] in indices:
        ptr = ptr + 1
    rtn_indices = sequence[:ptr]
    sequence = sequence[ptr:]
    return rtn_indices, sequence


def _do_trace(a, leg_link, traced_indices):
    """
    Helper function for _ncon

    Trace over pair(s) of legs on a given tensor
    Update the leg_link entry
    .. todo :
        perform all traces simultaneously

    Parameters
    ----------
    a : :class:'Array'
        the tensor to perform traces on
    leg_link : list of int
        the leg_links entry of a
    traced_indices list of int
        the labels of the legs to be traced. each should appear twice in leg_link

    Returns
    -------
    a : :class:'Array
        the traced tensor
    leg_link : list of int
        the updated entry for leg_links

    """

    untraced_indices = list(filter(lambda n: n not in traced_indices, leg_link))

    # sort traced legs into two blocks blocks to trace over
    block_a = []
    block_b = []
    block_untraced = []
    for idx in traced_indices:
        # position of first and last appearance of idx in leg_labels
        block_a.append(leg_link.index(idx))
        block_b.append(len(leg_link) - 1 - leg_link[::-1].index(idx))
    for idx in untraced_indices:
        block_untraced.append(leg_link.index(idx))

    new_order = block_untraced + [l[n] for n in range(len(block_a)) for l in [block_a, block_b]]
    a.itranspose(new_order)
    for n in range(len(new_order) - 1, len(block_untraced), -2):
        a = npc.trace(a, n, n - 1)
    leg_link = untraced_indices

    return a, leg_link


def _tcontract(t1, t2, links1, links2, contract_legs):
    """Helper function for _ncon.

    Contract two tensors along one or multiple axis

    Parameters
    ----------
    t1 : :class:'Array'
        The first tensor.
    t2 : :class:'Array'
        The second tensor.
    links1 : list of int
        The leg_links entry of the first tensor
    links2 : list of int
        The leg_links entry of the second tensor
    contract_legs : list of int
        The labels of the legs to be contracted. Each should appear exactly once in links1 and exactly once in links2.

    Returns
    -------
    res : :class:'Array'
        The result of the pairwise contraction
    links : list of int
        a leg_links entry for the res tensor

    """

    # if either tensor is not an :class:'Array' try converting
    # this may occur if a closed disconnected diagram is part of the contraction.
    # ncon will then try to process the resulting number with _tcontract of _outer_product
    if type(t1) is not npc.Array:
        t1 = npc.Array.from_ndarray_trivial(t1)
    if type(t2) is not npc.Array:
        t2 = npc.Array.from_ndarray_trivial(t2)

    # find uncontracted legs
    free_legs_1 = list(filter(lambda n: n not in contract_legs, links1))
    free_legs_2 = list(filter(lambda n: n not in contract_legs, links2))

    # find positions of legs
    pos_cont_legs_1 = [links1.index(leg) for leg in contract_legs]
    pos_cont_legs_2 = [links2.index(leg) for leg in contract_legs]

    # contract
    res = npc.tensordot(t1, t2, axes=(pos_cont_legs_1, pos_cont_legs_2))
    # tensordot keeps order of uncontracted legs intact. first those of T1 then those of T2
    links = free_legs_1 + free_legs_2

    return res, links


def _outer_product(tensor_list, leg_links, sequence):
    """
    Helper function for _ncon

    Perform an outer product of multiple tensors and optionally contract all their legs with one single tensor.
    This can be caused a value OP in the sequence or if there are more than one tensors
    left but no legs to be contracted

    Details see :arxiv:`1304.6112`

    Parameters
    ----------
    tensor_list : list of :class:'Array'
        the whole list of tensors currently processed by _ncon
    leg_links : list of list of int
        the whole leg_links currently processed by _ncon
    sequence :
     the remaining sequence currently processe by _ncon

    Returns
    -------
    tensor_list : list of :class:'Array'
        an updated list of tensors containing the result of the outer product and all untouched tensors
    leg_links : list of list of int
        the corresponding updated leg_links
    sequence : list of int
        the remaining sequence

    """

    if all(n == outer_product for n in sequence):
        # final outer product of all remaining tensors
        # ensure there are enough entries
        if len(sequence) < len(tensor_list) - 1:
            sequence = [outer_product] * len(tensor_list - 1)
            warnings.warn('Not enough OP entries in sequence')

    # determine number of pending outer products
    num_op = len(
        sequence
    )  # default value: if no entry in sequence is find that is not outer_product, all of them are
    for n in range(len(sequence)):
        if sequence[n] != outer_product:
            num_op = n
            break

    # determine tensors on which OPs are to be performed
    # for num_op outer products we need num_op+1 tensors that are all contracted with one extra tensor
    # see :arxiv:`1304.6112`
    # find the next num_op+2 tensors coming up in sequence
    # failure to find this many implies an invalid sequence.

    if num_op == len(leg_links) - 1:
        # OP of all remaining tensors
        op_list = list(range(len(leg_links)))
    else:
        # flag relevant num_op + 2 tensors
        flags = [False] * len(leg_links)
        ptr = num_op  # sequence[num_op] is the first entry that is not _outer
        while sum(flags) < num_op + 2:
            if ptr >= len(sequence):
                raise ValueError(
                    'sequence contains outer products but ended before finding enough '
                    'tensors for the outer product.')
            if sequence[ptr] == outer_product:
                raise ValueError(
                    'sequence contains outer products but ncon encountered another OP before '
                    'identifying all tensors involved in the first.')
            count = 0
            for a in range(len(leg_links)):
                if sequence[ptr] in leg_links[a]:
                    flags[a] = True
                    count = count + 1
            if count != 2:
                raise ValueError(
                    'sequence contains outer products. An index on one of the legs is not appearing twice.'
                )
            ptr = ptr + 1

        # identify which of these tensors is *not* participating in the OP
        # but is instead contracted with the result of the OP
        # - identify the two tensors involved in the first contraction
        # - examine following contractions until one of them is with a third tensor
        #   (thus only occurs on one of the initial candidates)
        first_tensors = [None, None]
        ptr = num_op
        for a in range(len(leg_links)):
            if sequence[ptr] in leg_links[a]:
                if first_tensors[0] is None:
                    first_tensors[0] = a
                else:
                    first_tensors[1] = a
                    break
        done = False
        next_tensors = [None, None]
        while not done:
            next_tensors = [None, None]
            ptr = ptr + 1
            for a in range(len(leg_links)):
                if sequence[ptr] in leg_links[a]:
                    if next_tensors[0] is None:
                        next_tensors[0] = a
                    else:
                        next_tensors[1] = a
                        break
            if not first_tensors == next_tensors:
                done = True
        if next_tensors[0] in first_tensors:
            post_op_tensor = next_tensors[0]
        else:
            post_op_tensor = next_tensors[1]
        flags[post_op_tensor] = False
        op_list = list(filter(lambda n: flags[n], range(len(leg_links))))

        # check that all indices of op_list are contracted with post_op_tensor
        op_indices = [leg_links[n] for n in op_list]
        op_indices = [item for sublist in op_indices for item in sublist]
        for a in range(len(op_indices)):
            if not op_indices[a] in leg_links[post_op_tensor]:
                raise ValueError(
                    'Outer product failure. OP tensor has contraction with multiple other tensors.')

    # if either tensor is not an :class:'Array' try converting
    # this may occur if a closed disconnected diagram is part of the contraction.
    # ncon will then try to process the resulting number with _tcontract of _outer_product
    for n in op_list:
        if type(tensor_list[n]) is not npc.Array:
            tensor_list[n] = npc.Array.from_ndarray_trivial(tensor_list[n])

    # perform OPs, starting with the smallest tensors
    op_sizes = [tensor_list[n].size for n in op_list]
    while len(op_sizes) > 1:
        order = np.argsort(op_sizes)

        # construct outer product of the two smallest tensors
        tensor_list[op_list[order[0]]] = npc.outer(tensor_list[op_list[order[0]]],
                                                   tensor_list[op_list[order[1]]])
        tensor_list.pop(op_list[order[1]])
        leg_links[op_list[order[0]]] = leg_links[op_list[order[0]]] + leg_links[op_list[order[1]]]
        leg_links.pop(op_list[order[1]])

        # re adjust op_sizes
        op_sizes[order[0]] = op_sizes[order[0]] + op_sizes[order[1]]
        op_sizes.pop(order[1])
        op_list = [n - 1 if n > op_list[order[1]] else n for n in op_list]
        op_list.pop(order[1])

    return tensor_list, leg_links, sequence
