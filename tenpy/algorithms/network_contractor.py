"""Network Contractor.

A tool to contract a network of multiple tensors.

This is an implementation of 'NCON: A tensor network contractor for MATLAB'
by Robert N. C. Pfeifer, Glen Evenbly, Sukhwinder Singh, Guifre Vidal, see :arxiv:`1402.0939`

.. todo ::
    - implement or wrap netcon.m, a function to find optimal contraction sequences
        (:arxiv:`1304.6112`)
"""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np
from ..linalg import np_conserved as npc

__all__ = ['contract', 'ncon']


def ncon(tensor_list, leg_links, sequence=None):
    """Implementation of ``ncon.m`` for TeNPy Arrays.

    This function is a python implementation of ``ncon.m`` (:arxiv:`1304.6112`) for tenpy
    :class:`~tenpy.linalg.np_conserved.Array`.
    :func:`contract` is a wrapper that translates from a more python/tenpy input style

    Parameters
    ----------
    tensor_list : iterable of :class:'Array'
        Tensors to be contracted.
    leg_links : iterable of iterable of int
        Each entry of leg_links describes the connectivity of the corresponding tensor in
        `tensor_list`.
        Each entry is a sequence (e.g. a list) that has an integer value for each leg of the corresponding tensor.
        Positive values ``1,2,...`` are labels of contracted legs and should appear
        exactly twice in `leg_links`.
        Negative values ``-1,-2,-3,...`` are labels of uncontracted legs and indicate the final ordering
        (``-1`` is the first axis).
    sequence : list of int, optional
        The order in which the contractions (indicated by positive values in `leg_links`) are to be performed.
        Ascending order is used by default.

    Returns
    -------
    result : :class:`Array` | complex
        The number or tensor resulting from the contraction.
    """
    tensor_list, leg_links, sequence = _ncon_input_checks(tensor_list, leg_links, sequence)
    tensor_list, leg_links, sequence = _ncon_do_traces(tensor_list, leg_links, sequence)
    tensor_list, leg_links, sequence = _ncon_do_binary_contractions(tensor_list, leg_links, sequence)
    tensor_list, leg_links = _ncon_do_outer_products(tensor_list, leg_links)
    result, = tensor_list
    if len(leg_links[0]) > 0:
        result.itranspose(np.argsort(-leg_links[0]))
    return result


def contract(tensor_list, tensor_names=None, leg_contractions=None, open_legs=None, sequence=None):
    """Contract a network of tensors.

    Based on the MatLab function ``ncon.m`` as described in :arxiv:`1402.0939`.

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
        sequence = list(range(1, len(leg_contractions) + 1))

    # translate leg_contractions and open_legs to a leg_links list as used by ncon
    # initialise leg_links
    leg_links = []
    for tensor in tensor_list:
        leg_links.append([None] * len(tensor.legs))

    # fill in the contractions
    contraction_counter = 1
    new_sequence = []
    for n in sequence:

        con = leg_contractions[n - 1]
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

    # call ncon and relabel the results' legs
    res = ncon(tensor_list, leg_links, new_sequence)
    if len(final_labels) > 0:
        res.iset_leg_labels(final_labels)

    return res
    

def _ncon_input_checks(tensor_list, leg_links, sequence):
    """Check inputs for consistency and convert to the following format
    leg_links: list of np.ndarray
    sequence: np.ndarray
    """
    tensor_list = list(tensor_list)
    
    if len(leg_links) != len(tensor_list):
        msg = f'Mismatching lengths: Got {len(tensor_list)} Arrays and {len(leg_links)} leg_links'
        raise ValueError(msg)
    leg_links = [np.asarray(tensor_links) for tensor_links in leg_links]
    leg_links_flat = [value for tensor_links in leg_links for value in tensor_links]
    num_contractions = max((value for value in leg_links_flat if value > 0), default=0)
    missing_positive_values = [n for n in range(1, num_contractions + 1) if n not in leg_links_flat]
    if missing_positive_values:
        which = ", ".join(map(str, missing_positive_values))
        msg = f'The following positive values are missing in leg_links: {which}'
        raise ValueError(msg)
    
    if sequence is None:
        sequence = np.arange(1, num_contractions + 1)
    else:
        sequence = np.asarray(sequence)
        if len(sequence) != num_contractions or set(sequence) != set(range(1, num_contractions + 1)):
            msg = f'Invalid sequence. Expected a permutation of [1, ..., {num_contractions}]. Got {sequence}.'
            raise ValueError(msg)
        
    return tensor_list, leg_links, sequence


def _ncon_do_traces(tensor_list, leg_links, sequence):
    for n in range(len(leg_links)):
        if len(leg_links[n]) > len(np.unique(leg_links[n])):
            tensor_list[n], leg_links[n], used_values = _partial_trace(tensor_list[n], leg_links[n], loc=n)
            sequence = np.delete(sequence, np.intersect1d(sequence, used_values, return_indices=True)[1])
    return tensor_list, leg_links, sequence


def _partial_trace(tensor, tensor_links, loc):
    """Perform all partial traces on a given tensor.
    
    Parameters
    ----------
    tensor: :class:'Array'
    tensor_links: np.ndarray
        the corresponding entry of `tensor_links`
    loc: int
        the index of `tensor` in `tensor_list`
        
    Returns
    -------
    result: :class:'Array'
        the traced tensor
    tensor_links: np.ndarray
        the proper replacement entry for `tensor_links`
    used_values: np.ndarray
        the entries of `tensor_links` that indicated the trace and are missing in the returned `tensor_links`
    """
    num_occurrences = np.sum(tensor_links[:, None] == tensor_links[None, :], axis=1)
    trace_links = np.unique(tensor_links[np.where(num_occurrences > 1)[0]])
    res_links = tensor_links[np.where(num_occurrences == 1)[0]]
    num_traces = len(trace_links)
    assert num_traces > 0

    trace_axes = np.zeros((num_traces, 2), dtype=np.int_)
    for n, value in enumerate(trace_links):
        trace_axes[n, :] = np.where(tensor_links == value)[0]
    
    try:
        tensor = tensor.combine_legs(combine_legs=(trace_axes[:, 0], trace_axes[:, 1]), new_axes=(-2, -1))
        tensor = npc.trace(tensor, leg1=-2, leg2=-1)
    except Exception as e:
        msg = (f'An error ocurred while performing the partial trace on tensor_list[{loc}]. '
               f'Original stacktrace below.')
        raise type(e)(msg) from e
    
    return tensor, res_links, trace_links
    

def _ncon_do_binary_contractions(tensor_list, leg_links, sequence):
    while len(sequence) > 0:
        to_contract = sequence[0]
        which_tensors = [n for n in range(len(tensor_list)) if sum(leg_links[n] == to_contract) > 0]
        try:
            loc_a, loc_b = which_tensors
        except ValueError:
            msg = f'Invalid leg_links. Value {to_contract} appeared on {len(which_tensors)} different tensors!'
            raise ValueError(msg) from None

        # pop b first since lob_b > loc_a
        tensor_b = tensor_list.pop(loc_b)
        links_b = leg_links.pop(loc_b)
        tensor_a = tensor_list.pop(loc_a)
        links_a = leg_links.pop(loc_a)
        
        common_values, a_axes, b_axes = np.intersect1d(links_a, links_b, assume_unique=True, return_indices=True)
        
        try:
            res = npc.tensordot(tensor_a, tensor_b, (a_axes, b_axes))
        except Exception as e:
            msg = (f'An error occurred while performing the pairwise contraction indicated by '
                   f'values {", ".join(common_values)} in leg_links. '
                   f'Original stacktrace below.')
            raise type(e)(msg) from e

        res_links = np.append(np.delete(links_a, a_axes), np.delete(links_b, b_axes))
        
        tensor_list.append(res)
        leg_links.append(res_links)
        used_sequence_values = np.intersect1d(sequence, common_values, assume_unique=True, return_indices=True)[1]
        sequence = np.delete(sequence, used_sequence_values)
        
    return tensor_list, leg_links, sequence
    
    
def _ncon_do_outer_products(tensor_list, leg_links):
    while len(tensor_list) > 1:
        tensor_b = tensor_list.pop(-1)
        links_b = leg_links.pop(-1)
        try:
            tensor_list[-1] = npc.outer(tensor_list[-1], tensor_b)
        except Exception as e:
            msg = (f'An error occurred while performing a final outer product between the last two '
                   f'of {len(tensor_list) + 1} remaining tensors. '
                   f'Original stacktrace below.')
            raise type(e)(msg) from e
        leg_links[-1] = np.append(leg_links[-1], links_b)
    
    return tensor_list, leg_links
    
        