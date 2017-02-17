r"""This module contains an MPS class representing an density matrix by purification.

Usually, an MPS represents a pure state, i.e. the density matrix is :math:`\rho = |\psi><\psi|`,
describing observables as :math:`<O> = Tr(O|\psi><\psi|) = <\psi|O|\psi>`.
Clearly, if :math:`|\psi>` is the ground state of a Hamiltonian, this is the density matrix at
`T=0`.

At finite temperatures :math:`T > 0`, we want to describe a non-pure density matrix
:math:`\rho = \exp(-H/T)`. This can be accieved by the so-called purification: in addition to
the physical space `P`, we introduce a second 'auxiliar' space `Q` and define the density matrix
of the physical system as :math:`\rho = Tr_Q(|\phi><\phi|)`, where :math:`|\phi>` is a pure state
in the combined phyisical and auxiliar system.

For :math:`T=\infty`, the density matrix :math:`\rho_\infty` is the identity matrix.
In other words, expectation values are sums over all possible states
:math:`<O> = Tr_P(\rho_\infty O) = Tr_P(O)`.
Saying that each ``:`` on top is to be connected with the corresponding ``:`` on the bottom,
the trace is simply a contraction::

    |         :   :   :   :   :   :
    |         |   |   |   |   |   |
    |         |-------------------|
    |         |        O          |
    |         |-------------------|
    |         |   |   |   |   |   |
    |         :   :   :   :   :   :

Clearly, we get the same result, if we insert an identiy operators, written as MPO, on the top
and bottom::

    |         :   :   :   :   :   :
    |         |   |   |   |   |   |
    |         B---B---B---B---B---B
    |         |   |   |   |   |   |
    |         |-------------------|
    |         |        O          |
    |         |-------------------|
    |         |   |   |   |   |   |
    |         B*--B*--B*--B*--B*--B*
    |         |   |   |   |   |   |
    |         :   :   :   :   :   :

We  use the following label convention::

    |         q
    |         ^
    |         |
    |  vL ->- B ->- vR
    |         |
    |         ^
    |         p

You can view the `MPO` as an MPS by combining the `p` and `q` leg and defining every physical
operator to act trivial on the `q` leg. In expecation values, you would then sum over
over the `q` legs, which is exactly what we need.
In other words, the choice :math:`B = \delta_{p,q}` with trivial (length-1) virtual bonds yields
infinite temperature expectation values for operators action only on the `p` legs!

Now, you go a step further and also apply imaginary time evolution (acting only on `p` legs)
to the initial infinite temperature state.
For example, the state :math:`\exp(-\beta/2 H)|\phi>` yields expecation values
:math:`<O> = <\phi|\exp(-\beta/2 H) O \exp(-\beta/2 H)|\phi> = Tr(\exp(-\beta H) O)`.


An additional real-time evolution allows to calculate time correlation functions,
e.g. :math:`<A(t)B(0)> = <\phi|\exp(-\beta H/2) \exp(+i H t) A \exp(-i H t) B \exp(-\beta H/2) |\phi>`.


See also [2]_ for additional tricks! On of their crucial observations is, that
one can apply arbitrary unitaries on the auxiliar space (i.e. the `q`) without changing the result.
This can actually be used to reduce the necessary virtual bond dimensions:
From the definition, it is easy to see that if we apply :math:`exp(-i H t)` to the `p` legs of
:math:`|\phi>`, and :math:`\exp(+iHt)` to the `q` legs, they just cancel out!
(They commute with :math:`\exp(-\beta H/2)`...)
If the state is modified (e.g. by applying `A` or `B` to calculate correlation functions),
this is not true any more. However, we still can find unitaries, which are 'optimal' in the sense
of reducing the entanglement of the MPS/MPO to the minimal value.


.. Note :
    The literature (e.g. section 7.2 of [1]_ or [2]_) suggests to use a `singlet` as a maximally
    entangled state. Here, we use instead the identity :math:`\delta_{p,q}`, since it is easier to
    generalize for `p` running over more than two indices, and allows a simple use of charge
    conservation with the above `qconj` convention.
    Moreover, we don't split the physical and auxiliar space into separate sites, which makes
    TEBD as costly as :math:`O(d^6 \chi^3)`.

Of course, we are not only intereseted at infinite-temperature expecation values, but primarily on
finite temperature expectation values.


.. [1] U. Schollwoeck, Annals of Physics 326, 96 (2011), arXiv:1008.3477
.. [2] C. Karrasch, J. H. Bardarson, J. E. Moore, New J. Phys. 15, 083031 (2013), arXiv:1303.3942
"""

from __future__ import division
import numpy as np

from .mps import MPS
from ..linalg import np_conserved as npc


class PurificationMPS(MPS):
    r"""An MPS representing a finite-temperature ensemble using purification.

    Similar as an MPS, but each `B` has now the four legs ``'vL', 'vR', 'p', 'q'``.
    From the point of algorithms, it is to be considered as a ususal MPS by combining the legs
    `p` and `q`, but all physical operators act only on the `p` part.
    For example, the right-canonical form is defined as if the legs 'p' and 'q' would be combined,
    e.g. a right-canonical `B` full-fills::

        npc.tensordot(B, B.conj(),axes=[['vR', 'p', 'q'], ['vR*', 'p*', 'q*']]) == \
            npc.eye_like(B, axes='vL')  # up to round-off errors

    For expectation values / correlation functions, all operators are to understood to act on
    `p` only, i.e. they act trivial on `q`, so we just trace over `q`,`q*`.

    See also the docstring of the module for details.

    .. todo :
        Formally, the order of the algorithms is better if we split the `p` and `q` legs onto
        different 'sites' (i.e. making a ladder-structure). For TEBD, this requires a `swap` of
        the sites. Also, I'm not sure, how much faster this actually would be....
    """
    # `MPS.get_B` & co work, thanks to using labels. `B` just have the additional `q` labels.
    # `get_theta` works thanks to `_replace_p_label`
    # correlation_function works as it should, if we adjust _corr_up_diag

    def test_sanity(self):
        """Sanity check. Raises Errors if something is wrong."""
        for B in self._B:
            if not set(['vL', 'vR', 'p', 'q']) <= set(B.get_leg_labels()):
                raise ValueError("B has wrong labels " + repr(B.get_leg_labels()))
        super(PurificationMPS, self).test_sanity()

    @classmethod
    def from_infinteT(cls, sites, bc='finite', form='B'):
        """Initial state corresponding to infinite-Temperature ensemble.

        Parameters
        ----------

        Returns
        -------
        infiniteT_MPS : :class:`PurificationMPS`
            Describes the infinite-temperature (grand canonical) ensemble,
            i.e. expectation values give a trave over all basis states.
        """
        sites = list(sites)
        L = len(sites)
        S = [[1.]]*(L+1)  # trivial S: product state
        Bs = [None]*L
        for i in range(L):
            p_leg = sites[i].leg
            B = npc.diag(1., p_leg, np.float)
            B.set_leg_labels(['p', 'q'])  # `q` has the physical leg with opposite `qconj`
            B = B.add_trivial_leg(0, label='vL', qconj=+1).add_trivial_leg(1, label='vR', qconj=-1)
            Bs[i] = B
        res = cls(sites, Bs, S, bc, form)
        return res

    def overlap(self, other):
        raise NotImplementedError("TODO: does this make sense? Need separate MPSEnvironment")

    def expectation_value(self, ops, sites=None, axes=None):
        """Expectation value ``<psi|ops|psi>`` of (n-site) operator(s).

        Given the MPS in canonical form, it calculates n-site expectation values.
        For example the contraction for a two-site (n=2) operator on site `i` would look like
        the following picture (where the ``:`` on top are connected with the ones on the bottom)::

            |                :     :
            |                |     |
            |          .--S--B[i]--B[i+1]--.
            |          |     |     |       |
            |          |     |-----|       |
            |          |     | op  |       |
            |          |     |-----|       |
            |          |     |     |       |
            |          .--S--B*[i]-B*[i+1]-.
            |                |     |
            |                :     :

        Parameters
        ----------
        ops : (list of) { :class:`~tenpy.linalg.np_conserved.Array` | str }
            The operators, for wich the expectation value should be taken,
            All operators should all have the same number of legs (namely `2 n`).
            If less than ``len(sites)`` operators are given, we repeat them periodically.
            Strings (like ``'Id', 'Sz'``) are translated into single-site operators defined by
            `self.sites`.
        sites : None | list of int
            List of site indices. Expectation values are evaluated there.
            If ``None`` (default), the entire chain is taken (clipping for finite b.c.)
        axes : None | (list of str, list of str)
            Two lists of each `n` leg labels giving the physical legs of the operator used for
            contraction. The first `n` legs are contracted with conjugated B`s,
            the second `n` legs with the non-conjugated `B`.
            ``None`` defaults to ``(['p'], ['p*'])`` for single site operators (n=1), or
            ``(['p0', 'p1', ... 'p{n-1}'], ['p0*', 'p1*', .... 'p{n-1}*'])`` for n > 1.

        Returns
        -------
        exp_vals : 1D ndarray
            Expectation values, ``exp_vals[i] = <psi|ops[i]|psi>``, where ``ops[i]`` acts on
            site(s) ``j, j+1, ..., j+{n-1}`` with ``j=sites[i]``.

        Examples
        --------
        One site examples (n=1):

        >>> psi.expectation_value('Sz')
        [Sz0, Sz1, ..., Sz{L-1}]
        >>> psi.expectation_value(['Sz', 'Sx'])
        [Sz0, Sx1, Sz2, Sx3, ... ]
        >>> psi.expectation_value('Sz', sites=[0, 3, 4])
        [Sz0, Sz3, Sz4]

        Two site example (n=2), assuming homogeneous sites:

        >>> SzSx = npc.outer(psi.sites[0].Sz.replace_labels(['p', 'p*'], ['p0', 'p0*']),
                             psi.sites[1].Sx.replace_labels(['p', 'p*'], ['p1', 'p1*']))
        >>> psi.expectation_value(SzSx)
        [Sz0Sx1, Sz1Sx2, Sz2Sx3, ... ]   # with len ``L-1`` for finite bc, or ``L`` for infinite

        Example measuring <psi|SzSx|psi2> on each second site, for inhomogeneous sites:

        >>> SzSx_list = [npc.outer(psi.sites[i].Sz.replace_labels(['p', 'p*'], ['p0', 'p0*']),
                                   psi.sites[i+1].Sx.replace_labels(['p', 'p*'], ['p1', 'p1*']))
                         for i in range(0, psi.L-1, 2)]
        >>> psi.expectation_value(SzSx_list, range(0, psi.L-1, 2))
        [Sz0Sx1, Sz2Sx3, Sz4Sx5, ...]

        """
        ops, sites, n, th_labels, (axes_p, axes_pstar) = self._expectation_value_args(
            ops, sites, axes)
        th_labels = th_labels + ['q'+str(j) for j in range(n)]  # additional q0, q1, ...
        vLvR_axes_p_q = ('vL', 'vR') + tuple(axes_p) + tuple(['q'+str(j) for j in range(n)])
        E = []
        for i in sites:
            op = self.get_op(ops, i)
            theta = self.get_theta(i, n)  # vL, vR, p0, q0, p1, q1
            C = npc.tensordot(op, theta, axes=[axes_pstar, th_labels[2:2+n]])  # ignore 'q'
            E.append(npc.inner(theta, C, axes=[th_labels, vLvR_axes_p_q], do_conj=True))
        return np.array(E)

    def _replace_p_label(self, A, k):
        """Return npc Array `A` with replaced label, ``'p' -> 'p'+str(k)``.

        Instead of re-implementing `get_theta`, the derived `PurificationMPS` needs only to
        implement this function."""
        return A.replace_labels(['p', 'q'], ['p'+str(k), 'q'+str(k)])

    def _corr_up_diag(self, ops1, ops2, i, j_gtr, opstr, str_on_first, apply_opstr_first):
        """correlation function above the diagonal: for fixed i and all j in j_gtr, j > i."""
        # compared to MPS._corr_up_diag just perform additional contractions of the 'q'
        op1 = self.get_op(ops1, i)
        opstr1 = self.get_op(opstr, i)
        if opstr1 is not None:
            axes = ['p*', 'p'] if apply_opstr_first else ['p', 'p*']
            op1 = npc.tensordot(op1, opstr1, axes=axes)
        theta = self.get_theta(i, n=1)
        C = npc.tensordot(op1, theta, axes=['p*', 'p0'])
        C = npc.tensordot(theta.conj(), C, axes=[['p0*', 'vL*', 'q0*'], ['p', 'vL', 'q0']])
        # C has legs 'vR*', 'vR'
        js = list(j_gtr[::-1])  # stack of j, sorted *descending*
        res = []
        for r in range(i+1, js[0]+1):  # js[0] is the maximum
            B = self.get_B(r, form='B')
            C = npc.tensordot(C, B, axes=['vR', 'vL'])
            if r == js[-1]:
                Cij = npc.tensordot(self.get_op(ops2, r), C, axes=['p*', 'p'])
                Cij = npc.inner(B.conj(), Cij, axes=[['vL*', 'p*', 'q*', 'vR*'],
                                                     ['vR*', 'p', 'q', 'vR']])
                res.append(Cij)
                js.pop()
            if len(js) > 0:
                op = self.get_op(opstr, r)
                if op is not None:
                    C = npc.tensordot(op, C, axes=['p*', 'p'])
                C = npc.tensordot(B.conj(), C, axes=[['vL*', 'p*', 'q*'], ['vR*', 'p', 'q']])
        return res
