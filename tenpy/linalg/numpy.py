from __future__ import annotations

from tenpy.linalg.tensors import Tensor


def tdot(t1: Tensor, t2: Tensor, legs1: int | str | list[int | str], legs2: int | str | list[int | str],
         relabel1: dict[str, str] = None, relabel2: dict[str, str] = None) -> Tensor:
    """
    TODO: decide name, eg from tensordot, tdot, contract

    Contraction of two tensors

    Parameters
    ----------
    t1 : Tensor
    t2 : Tensor
    legs1 : int or str or list of int or list of str
        the leg(s) on t1 to be contracted, referenced either by index or by label
    legs2 : int or str of list of int or list of str
        the leg(s) on t2 to be contracted, referenced either by index or by label
    relabel1 : dict
        labels of the result are determined as if t1 had been relabelled by this mapping before contraction
    relabel2
        labels of the result are determined as if t2 had been relabelled by this mapping before contraction

    Returns
    -------

    """

    if isinstance(legs1, int):
        ax1 = [legs1]
    elif isinstance(legs1, str):
        ax1 = [t1.get_leg_idx(legs1)]
    else:
        ax1 = t1.get_leg_idcs(legs1)

    if isinstance(legs2, int):
        ax2 = [legs2]
    elif isinstance(legs2, str):
        ax2 = [t2.get_leg_idx(legs2)]
    else:
        ax2 = t2.get_leg_idcs(legs2)

    assert len(ax1) == len(ax2)
    assert all(t1.legs[n1].can_contract_with(t2.legs[n2]) for n1, n2 in zip(ax1, ax2))

    open_legs1 = [l for n, l in enumerate(t1.legs) if n not in ax1]
    open_legs2 = [l for n, l in enumerate(t2.legs) if n not in ax2]
    open_labels1 = [l for n, l in enumerate(t1._leg_labels) if n not in ax1]
    open_labels2 = [l for n, l in enumerate(t2._leg_labels) if n not in ax2]
    new_labels = result_leg_labels(open_labels1, open_labels2, relabel1, relabel2)

    res_data = t1.backend.tdot(t1.data, t2.data, ax1, ax2)
    res_dtype = t1.backend.infer_dtype(res_data)
    return Tensor(res_data, backend=t1.backend, legs=open_legs1 + open_legs2, leg_labels=new_labels, dtype=res_dtype)


def result_leg_labels(labels1: list[str | None], labels2: list[str | None],
                      relabel1: dict[str, str] | None, relabel2: dict[str, str] | None
                      ) -> list[str | None]:
    """basically just list concatenation, i.e. labels1 + labels2,
    but with duplicate checks and (optional) relabelling."""

    if relabel1 is not None:
        relabel1 = relabel1.copy()  # this may be inefficient, but should be rewritten in C anyway
        labels1 = [relabel1.pop(l, l) for l in labels1]
        if len(relabel1) > 0:
            raise ValueError(f'relabel1 has superfluous entries: {list(relabel1.keys())}')

    if relabel2 is not None:
        relabel2 = relabel2.copy()
        labels2 = [relabel2.pop(l, l) for l in labels1]
        if len(relabel2) > 0:
            raise ValueError(f'relabel2 has superfluous entries: {list(relabel1.keys())}')

    res_labels = labels1 + labels2
    duplicates = []
    for n, l in enumerate(res_labels):
        if l is None:
            continue

        if l in duplicates:
            res_labels[n] = None
        elif l in res_labels[n + 1:]:
            duplicates.append(l)
            res_labels[n] = None
    # TODO warn if there were duplicates?

    return res_labels
