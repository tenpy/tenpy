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

Clearly, we get the same result, if we insert an identity operator, written as MPO, on the top
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
For example, the normalized state :math:`|\psi> \propto \exp(-\beta/2 H)|\phi>`
yields expecation values

.. math ::
    <O>  = Tr(\exp(-\beta H) O) / Tr(\exp(-\beta H))
    \propto <\phi|\exp(-\beta/2 H) O \exp(-\beta/2 H)|\phi>.

An additional real-time evolution allows to calculate time correlation functions:

.. math ::
    <A(t)B(0)> \propto <\phi|\exp(-\beta H/2) \exp(+i H t) A \exp(-i H t) B \exp(-\beta H/2) |\phi>


See also [Karrasch2013]_ for additional tricks! On of their crucial observations is, that
one can apply arbitrary unitaries on the auxiliar space (i.e. the `q`) without changing the result.
This can actually be used to reduce the necessary virtual bond dimensions:
From the definition, it is easy to see that if we apply :math:`exp(-i H t)` to the `p` legs of
:math:`|\phi>`, and :math:`\exp(+iHt)` to the `q` legs, they just cancel out!
(They commute with :math:`\exp(-\beta H/2)`...)
If the state is modified (e.g. by applying `A` or `B` to calculate correlation functions),
this is not true any more. However, we still can find unitaries, which are 'optimal' in the sense
of reducing the entanglement of the MPS/MPO to the minimal value.
For a discussion of `Disentanglers` (implemented in :mod:`~tenpy.algorithms.purification_tebd`),
see [Hauschild2018]_.

.. note ::
    The classes :class:`~tenpy.linalg.networks.mps.MPSEnvironment` and
    :class:`~tenpy.linalg.networks.mps.TransferMatrix` should also work for the
    :class:`PurificationMPS` defined here.
    For example, you can use :meth:`~tenpy.networks.mps.MPSEnvironment.expectation_value`
    for the expectation value of operators between different PurificationMPS.
    However, this makes only sense if the *same* disentangler was applied
    to the `bra` and `ket` PurificationMPS.

.. note ::
    The literature (e.g. section 7.2 of [Schollwoeck2011]_ or [Karrasch2013]_) suggests to use
    a `singlet` as a maximally entangled state.
    Here, we use instead the identity :math:`\delta_{p,q}`, since it is easier to
    generalize for `p` running over more than two indices, and allows a simple use of charge
    conservation with the above `qconj` convention.
    Moreover, we don't split the physical and auxiliar space into separate sites, which makes
    TEBD as costly as :math:`O(d^6 \chi^3)`.

.. todo ::
    One can also look at the canonical ensembles by defining the conserved quantities
    differently, see Barthel (2016), :arxiv:`1607.01696` for details.
    Idea: usual charges on `p`, trivial charges on `q`; fix total charge to desired value.
    I think it should suffice to implement another `from_infiniteT`.

"""
# Copyright 2018 TeNPy Developers

import numpy as np

from .mps import MPS
from ..linalg import np_conserved as npc
from ..tools.math import entropy

__all__ = ['PurificationMPS']


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
    `p` only, i.e. they act trivial on `q`, so we just trace over ``'q', 'q*'``.

    See also the docstring of the module for details.
    """

    # `MPS.get_B` & co work, thanks to using labels. `B` just have the additional `q` labels.
    _p_label = ['p', 'q']  # this adjustment makes `get_theta` & friends work
    _B_labels = ['vL', 'p', 'q', 'vR']

    # Thanks to using `self._replace_p_label`,
    # correlation_function works as it should, if we adjust _corr_up_diag

    def test_sanity(self):
        """Sanity check. Raises Errors if something is wrong."""
        for B in self._B:
            if not set(['vL', 'vR', 'p', 'q']) <= set(B.get_leg_labels()):
                raise ValueError("B has wrong labels " + repr(B.get_leg_labels()))
        super().test_sanity()

    def copy(self):
        """Returns a copy of `self`.

        The copy still shares the sites, chinfo, and LegCharges of the _B,
        but the values of B and S are deeply copied.
        """
        # __init__ makes deep copies of B, S
        return PurificationMPS(self.sites, self._B, self._S, self.bc, self.form)

    @classmethod
    def from_infiniteT(cls, sites, bc='finite', form='B'):
        """Initial state corresponding to infinite-Temperature ensemble.

        Parameters
        ----------
        sites : list of :class:`~tenpy.networks.site.Site`
            The sites defining the local Hilbert space.
        bc : {'finite', 'segment', 'infinite'}
            MPS boundary conditions as described in :class:`~tenpy.networks.mps.MPS`.
        form : (list of) {``'B' | 'A' | 'C' | 'G' | None`` | tuple(float, float)}
            The canonical form of the stored 'matrices', see table in :mod:`~tenpy.networks.mps`.
            A single choice holds for all of the entries.

        Returns
        -------
        infiniteT_MPS : :class:`PurificationMPS`
            Describes the infinite-temperature (grand canonical) ensemble,
            i.e. expectation values give a trace over all basis states.
        """
        sites = list(sites)
        L = len(sites)
        S = [[1.]] * (L + 1)  # trivial S: product state
        Bs = [None] * L
        for i in range(L):
            p_leg = sites[i].leg
            B = npc.diag(1., p_leg, np.float) / sites[i].dim**0.5
            B.iset_leg_labels(['p', 'q'])  # `q` has the physical leg with opposite `qconj`
            B = B.add_trivial_leg(0, label='vL', qconj=+1).add_trivial_leg(1, label='vR', qconj=-1)
            Bs[i] = B
        res = cls(sites, Bs, S, bc, form)
        return res

    def entanglement_entropy_segment(self, segment=[0], first_site=None, n=1, legs='p'):
        r"""Calculate entanglement entropy for general geometry of the bipartition.

        This function is similar as :meth:`entanglement_entropy`,
        but for more general geometry of the region `A` to be a segment of a *few* sites.

        This is acchieved by explicitly calculating the reduced density matrix of `A`
        and thus works only for small segments.

        Parameters
        ----------
        segment : list of int
            Given a first site `i`, the region ``A_i`` is defined to be ``[i+j for j in segment]``.
        first_site : ``None`` | (iterable of) int
            Calculate the entropy for segments starting at these sites.
            ``None`` defaults to ``range(L-segment[-1])`` for finite
            or `range(L)` for infinite boundary conditions.
        n : int | float
            Selects which entropy to calculate;
            `n=1` (default) is the ususal von-Neumann entanglement entropy,
            otherwise the `n`-th Renyi entropy.
        leg : 'p', 'q', 'pq'
            Whether we look at the entanglement entropy in both (`pq`) or
            only one of auxiliar (`q`) and physical (`p`) space.

        Returns
        -------
        entropies : 1D ndarray
            ``entropies[i]`` contains the entropy for the the region ``A_i`` defined above.
        """
        segment = np.sort(segment)
        if first_site is None:
            if self.finite:
                first_site = range(0, self.L - segment[-1])
            else:
                first_site = range(self.L)
        N = len(segment)

        def labels(choice):
            res1 = [c + str(k) for k in range(N) for c in choice]
            res2 = [c + str(k) + '*' for k in range(N) for c in choice]
            return res1, res2

        if legs == 'pq':
            tr_legs = ([], [])
            comb_legs = labels(['p', 'q'])
        elif legs == 'p':
            tr_legs = labels(['q'])
            comb_legs = labels(['p'])
        elif legs == 'q':
            tr_legs = labels(['p'])
            comb_legs = labels(['q'])
        res = []
        for i0 in first_site:
            rho = self.get_rho_segment(segment + i0)  # p0, q0, p0*, q0*, ...
            # extra contraction
            for a, b in zip(*tr_legs):
                rho = npc.trace(rho, a, b)
            rho = rho.combine_legs(comb_legs, qconj=[+1, -1])
            p = npc.eigvalsh(rho)
            res.append(entropy(p, n))
        return np.array(res)

    def mutinf_two_site(self, max_range=None, n=1, legs='p'):
        """Calculate the two-site mutual information :math:`I(i:j)`.

        Calculates :math:`I(i:j) = S(i) + S(j) - S(i,j)`,
        where :math:`S(i)` is the single site entropy on site :math:`i`
        and :math:`S(i,j)` the two-site entropy on sites :math:`i,j`.

        Parameters
        ----------
        max_range : int
            Maximal distance ``|i-j|`` for which the mutual information should be calculated.
            ``None`` defaults to `L-1`.
        n : float
            Selects the entropy to use, see :func:`~tenpy.tools.math.entropy`.
        leg : 'p', 'q', 'pq'
            Whether we look at the entanglement entropy in both (`pq`) or
            only one of auxiliar (`q`) and physical (`p`) space.

        Returns
        -------
        coords : 2D array
            Coordinates for the mutinf array.
        mutinf : 1D array
            ``mutinf[k]`` is the mutual information :math:`I(i:j)` between the
            sites ``i, j = coords[k]``.
        """
        # Now same as MPS.mutinf_two_site(), but contract additionally over leg.
        if max_range is None:
            max_range = self.L
        S_i = self.entanglement_entropy_segment(n=n, legs=legs)  # single-site entropy

        def labels(choice):
            res1 = [c + str(k) for k in range(2) for c in choice]
            res2 = [c + str(k) + '*' for k in range(2) for c in choice]
            return res1, res2

        if legs == 'pq':
            tr_legs = ([], [])
            comb_legs = labels(['p', 'q'])
        elif legs == 'p':
            tr_legs = labels(['q'])
            comb_legs = labels(['p'])
        elif legs == 'q':
            tr_legs = labels(['p'])
            comb_legs = labels(['q'])
        contr_rho = (
            ['vR*'] + self._get_p_label('1'),  # 'vL', 'p1'
            ['vL*'] + self._get_p_label('1*'))  # 'vL*', 'p1*'
        mutinf = []
        coord = []
        for i in range(self.L):
            rho = self.get_theta(i, 1)
            rho = npc.tensordot(rho, rho.conj(), axes=('vL', 'vL*'))
            jmax = i + max_range + 1
            if self.finite:
                jmax = min(jmax, self.L)
            for j in range(i + 1, jmax):
                B = self.get_B(j, form='B', label_p='1')  # 'vL', 'vR', 'p1'
                rho = npc.tensordot(rho, B, axes=['vR', 'vL'])
                rho_ij = npc.tensordot(rho, B.conj(), axes=(['vR*', 'vR'], ['vL*', 'vR*']))
                for a, b in zip(*tr_legs):
                    rho_ij = npc.trace(rho_ij, a, b)
                rho_ij = rho_ij.combine_legs(comb_legs, qconj=[+1, -1])
                S_ij = entropy(npc.eigvalsh(rho_ij), n)
                mutinf.append(S_i[i] + S_i[j % self.L] - S_ij)
                coord.append((i, j))
                if j + 1 < jmax:
                    rho = npc.tensordot(rho, B.conj(), axes=contr_rho)
        return np.array(coord), np.array(mutinf)

    def swap_sites(self, i, swapOP='auto', trunc_par={}):
        raise NotImplementedError()

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
        for r in range(i + 1, js[0] + 1):  # js[0] is the maximum
            B = self.get_B(r, form='B')
            C = npc.tensordot(C, B, axes=['vR', 'vL'])
            if r == js[-1]:
                Cij = npc.tensordot(self.get_op(ops2, r), C, axes=['p*', 'p'])
                Cij = npc.inner(B.conj(),
                                Cij,
                                axes=[['vL*', 'p*', 'q*', 'vR*'], ['vR*', 'p', 'q', 'vR']])
                res.append(Cij)
                js.pop()
            if len(js) > 0:
                op = self.get_op(opstr, r)
                if op is not None:
                    C = npc.tensordot(op, C, axes=['p*', 'p'])
                C = npc.tensordot(B.conj(), C, axes=[['vL*', 'p*', 'q*'], ['vR*', 'p', 'q']])
        return res

    def _replace_p_label(self, A, s):
        """Return npc Array `A` with replaced label, ``'p' -> 'p'+s, 'q' -> 'q'+s``."""
        return A.replace_labels(self._p_label, self._get_p_label(s))

    def _get_p_label(self, s, star=False):
        """return  self._p_label with additional string `s`."""
        return ['p' + s, 'q' + s]

    def _get_p_labels(self, ks, star=False):
        """join ``self._get_p_label(str(k) {+'*'} ) for k in range(ks)`` to a single list."""
        if star:
            return [lbl + str(k) + '*' for k in range(ks) for lbl in self._p_label]
        else:
            return [lbl + str(k) for k in range(ks) for lbl in self._p_label]
