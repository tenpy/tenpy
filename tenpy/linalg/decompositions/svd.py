from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from tenpy.linalg.decompositions.leg_bipartition import leg_bipartition
from tenpy.linalg.tensors import Tensor


@dataclass
class TruncationParams:
    max_singular_values: int = np.inf  # maximum number of singular values kept
    singular_value_threshold: float = 0  # singular values below this threshold are discarded
    max_err: float = 0  # upper bound on truncation error (see 4-th return of svd)


no_truncation = TruncationParams(max_singular_values=np.inf, singular_value_threshold=0, max_err=0)


def svd(a: Tensor, u_legs: list[int | str] = None, vh_legs: list[int | str] = None,
        new_labels: str | tuple[str, ...] = None,  truncation: TruncationParams = no_truncation,
        algorithm: str = None,
        ) -> tuple[Tensor, Tensor, Tensor, float]:
    """

    Parameters
    ----------
    a
    u_legs
        Bipartition of the legs (these legs will be legs of the resulting U). See `leg_bipartition`.
    vh_legs
        Bipartition of the legs (these legs will be legs of the resulting Vh). See `leg_bipartition`.
    new_labels
        Specification for the labels of new legs, i.e. those legs that would need to be contracted to recover `a`.
        If a single str, its is used for the leg on U, its partner on S and on Vh, the second label on S will have an
          appended prime.
        If a tuple of two strs, the first is for the leg on U, the second for Vh, legs on S have the same label
          as their partner.
        If a tuple of fours strs, all the above are specified, in the order U, S (U side), S (Vh side), Vh
        If None, the legs will not be labelled
    truncation
    algorithm
        can select an algorithm that performs the matrix-SVDs in the backend.
        possible values are backend specific.
        meaning of the default, `None`, is backend specific.
        TODO: this is intended to accommodate eg 'robust' as done in linalg.old

    Returns
    -------
    fourth return is relative truncation error norm(S_discarded) / norm(S_all)
    """
    u_idcs, u_labels, u_spaces, vh_idcs, vh_labels, vh_spaces = leg_bipartition(a, u_legs, vh_legs)
    try:
        if new_labels is None or isinstance(new_labels, str):
            l_u = l_su = l_sv = l_vh = new_labels
        elif len(new_labels) == 2:
            l_u, l_vh = new_labels
            l_su, l_sv = new_labels
        elif len(new_labels) == 4:
            l_u, l_su, l_sv, l_vh = new_labels
        else:
            raise ValueError
    except AttributeError:  # len failed
        raise TypeError
    u_data, s_data, vh_data, trunc_err, new_space = a.backend.svd(
        a.data, u_idcs, vh_idcs, max_singular_values=truncation.max_singular_values,
        threshold=truncation.singular_value_threshold, max_err=truncation.max_err,
        algorithm=algorithm
    )

    # TODO revisit dtypes
    U = Tensor(u_data, backend=a.backend, legs=u_spaces + [new_space], leg_labels=u_labels + [l_u],
               dtype=a.dtype.as_complex())
    S = Tensor(s_data, backend=a.backend, legs=[new_space.dual, new_space], leg_labels=[l_su, l_sv],
               dtype=a.dtype.as_real())
    Vh = Tensor(vh_data, backend=a.backend, legs=[new_space.dual] + vh_spaces, leg_labels=[l_vh] + vh_labels,
                dtype=a.dtype.as_complex())

    return U, S, Vh, trunc_err


def svd_split(a: Tensor, legs1: list[int | str], legs2: list[int | str], new_labels: str | tuple[str, str],
              truncation: TruncationParams = no_truncation, s_exponent: float = .5, algorithm: str = None):
    """Convenience wrapper around svd, where the singular values S are absorbed into either U (s_exponent==1),
    Vh (s_exponent==0) or both."""
    raise NotImplementedError  # TODO
