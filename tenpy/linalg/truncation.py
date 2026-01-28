r"""Truncation of Schmidt values.

Often, it is necessary to truncate the number of states on a virtual bond of an MPS,
keeping only the state with the largest Schmidt values.
The function :func:`truncate` picks exactly those from a given Schmidt spectrum
:math:`\lambda_a`, depending on some parameters explained in the doc-string of the function.

Further, we provide :class:`TruncationError` for a simple way to keep track of the
total truncation error.

The SVD on a virtual bond of an MPS actually gives a Schmidt decomposition
:math:`|\psi\rangle = \sum_{a} \lambda_a |L_a\rangle |R_a\rangle`
where :math:`|L_a\rangle` and :math:`|R_a\rangle` form orthonormal bases of the parts
left and right of the virtual bond.
Let us assume that the state is properly normalized,
:math:`\langle\psi | \psi\rangle = \sum_{a} \lambda^2_a = 1`.
Assume that the singular values are ordered descending, and that we keep the first :math:`\chi_c`
of the initially :math:`\chi` Schmidt values.

Then we decompose the untruncated state as
:math:`|\psi\rangle = \sqrt{1-\epsilon}|\psi_{tr}\rangle + \sqrt{\epsilon}|\psi_{tr}^\perp\rangle`
where
:math:`|\psi_{tr}\rangle =
\frac{1}{\sqrt{1-\epsilon}} \sum_{a < \chi_c} \lambda_a|L_a\rangle|R_a\rangle`
is the truncated state kept (normalized to 1),
:math:`|\psi_{tr}^\perp\rangle =
\frac{1}{\sqrt{\epsilon}} \sum_{a >= \chi_c} \lambda_a |L_a\rangle|R_a\rangle`
is the discarded part (orthogonal to the kept part) and the
*truncation error of a single truncation* is defined as
:math:`\epsilon = 1 - |\langle \psi | \psi_{tr}\rangle |^2 = \sum_{a >= \chi_c} \lambda_a^2`.

.. warning ::
    For imaginary time evolution (e.g. with TEBD), you try to project out the ground state.
    Then, looking at the truncation error defined in this module does *not* give you any
    information how good the found state coincides with the actual ground state!
    (Instead, the returned truncation error depends on the overlap with the initial state,
    which is arbitrary > 0)

.. warning ::
    This module takes only track of the errors coming from the truncation of Schmidt values.
    There might be other sources of error as well, for example TEBD has also an discretization
    error depending on the chosen time step.
"""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np

from ..tools.hdf5_io import Hdf5Exportable


class TruncationError(Hdf5Exportable):
    r"""Class representing a truncation error.

    The default initialization represents "no truncation".

    .. warning ::
        For groundstate search with imaginary time evolution, this quantifies the error
        in approximating the evolution. It is *not* related to the quality of the resulting
        groundstate approximation.

    Parameters
    ----------
    eps, ov : float
        See below.


    Attributes
    ----------
    eps : float
        The total sum of all discarded Schmidt values squared.
        Note that if you keep singular values up to 1.e-14 (= a bit more than machine precision
        for 64bit floats), `eps` is on the order of 1.e-28 (due to the square)!
    ov : float
        A lower bound for the overlap :math:`|\langle \psi_{trunc} | \psi_{correct} \rangle|^2`
        (assuming normalization of both states).
        This is probably the quantity you are actually interested in.
        Takes into account the factor 2 explained in the section on Errors in the
        `TEBD Wikipedia article <https://en.wikipedia.org/wiki/Time-evolving_block_decimation>`.

    """

    def __init__(self, eps=0.0, ov=1.0):
        self.eps = eps
        self.ov = ov

    def copy(self):
        """Return a copy of self."""
        return TruncationError(self.eps, self.ov)

    @classmethod
    def from_norm(cls, norm_new, norm_old=1.0):
        r"""Construct TruncationError from norm after and before the truncation.

        Parameters
        ----------
        norm_new : float
            Norm of Schmidt values kept, :math:`\sqrt{\sum_{a kept} \lambda_a^2}`
            (before re-normalization).
        norm_old : float
            Norm of all Schmidt values before truncation, :math:`\sqrt{\sum_{a} \lambda_a^2}`.

        """
        eps = 1.0 - norm_new**2 / norm_old**2  # = (norm_old**2 - norm_new**2)/norm_old**2
        return cls(eps, 1.0 - 2.0 * eps)

    @classmethod
    def from_S(cls, S_discarded, norm_old=None):
        r"""Construct TruncationError from discarded singular values.

        Parameters
        ----------
        S_discarded : 1D numpy array
            The singular values discarded.
        norm_old : float
            Norm of all Schmidt values before truncation, :math:`\sqrt{\sum_{a} \lambda_a^2}`.
            Default (``None``) is 1.

        """
        eps = np.sum(np.square(S_discarded))
        if norm_old:
            eps /= norm_old * norm_old
        return cls(eps, 1.0 - 2.0 * eps)

    def __add__(self, other):
        res = TruncationError()
        res.eps = self.eps + other.eps  # whatever that actually means...
        res.ov = self.ov * other.ov
        return res

    @property
    def ov_err(self):
        """Error ``1.-ov`` of the overlap with the correct state."""
        return 1.0 - self.ov

    def __repr__(self):
        if self.eps != 0 or self.ov != 1.0:
            return f'TruncationError(eps={self.eps:.4e}, ov={self.ov:.10f})'
        else:
            return 'TruncationError()'
