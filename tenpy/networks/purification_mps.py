r"""This module contains an MPS class representing an density matrix by purification.

Usually, an MPS represents a pure state, i.e. the density matrix is :math:`\rho = |\psi><\psi|`,
describing observables as :math:`<O> = Tr(O|\psi><\psi|) = <\psi|O|\psi>`.
Clearly, if :math:`|\psi>` is the ground state of a Hamiltonian, this is the density matrix at
`T=0`.

At finite temperatures :math:`T > 0`, we want to describe a mixed density matrix
:math:`\rho = \exp(-H/T)`. The following approaches have been used to lift the power of tensor
network ansätze (representing pure states= to finite temperatures (and mixed states in general).

1. Naively represent the density matrix as an MPO. This has the disadvantage that truncation can
   quickly lead to non-positive (and hence unphysical) density matrices.
2. Minimally entangled typical thermal states (METTS) as introduced in :cite:`white2009`.
3. Use Purification to represent the mixed density matrix by pure states in the doubled Hilbert
   space.
   In the literature, this is also referred to as matrix product density operators (MPDO) or
   locally purified density operator (LPDO).


Here, we follow the third approach.
In addition to the physical space `P`, we introduce a second 'auxiliary' space `Q`
and define the density matrix
of the physical system as :math:`\rho = Tr_Q(|\phi><\phi|)`, where :math:`|\phi>` is a pure state
in the combined physical and auxiliary system.

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
operator to act trivial on the `q` leg. In expectation values, you would then sum over
over the `q` legs, which is exactly what we need.
In other words, the choice :math:`B = \delta_{p,q}` with trivial (length-1) virtual bonds yields
infinite temperature expectation values for operators action only on the `p` legs!

Now, you go a step further and also apply imaginary time evolution (acting only on `p` legs)
to the initial infinite temperature state.
For example, the normalized state :math:`|\psi> \propto \exp(-\beta/2 H)|\phi>`
yields expectation values

.. math ::
    <O>  = Tr(\exp(-\beta H) O) / Tr(\exp(-\beta H))
    \propto <\phi|\exp(-\beta/2 H) O \exp(-\beta/2 H)|\phi>.

An additional real-time evolution allows to calculate time correlation functions:

.. math ::
    <A(t)B(0)> \propto <\phi|\exp(-\beta H/2) \exp(+i H t) A \exp(-i H t) B \exp(-\beta H/2) |\phi>

Time evolution algorithms (TEBD and MPO application) are adjusted in the module
:mod:`~tenpy.algorithms.purification`.

See also :cite:`karrasch2013` for additional tricks! One of their crucial observations is, that
one can apply arbitrary unitaries on the auxiliary space (i.e. the `q`) without changing the result.
This can actually be used to reduce the necessary virtual bond dimensions:
From the definition, it is easy to see that if we apply :math:`exp(-i H t)` to the `p` legs of
:math:`|\phi>`, and :math:`\exp(+iHt)` to the `q` legs, they just cancel out!
(They commute with :math:`\exp(-\beta H/2)`...)
If the state is modified (e.g. by applying `A` or `B` to calculate correlation functions),
this is not true any more. However, we still can find unitaries, which are 'optimal' in the sense
of reducing the entanglement of the MPS/MPO to the minimal value.
For a discussion of `Disentanglers` (implemented in :mod:`~tenpy.algorithms.disentanglers`),
see :cite:`hauschild2018`.

.. note ::
    The classes :class:`~tenpy.linalg.networks.mps.MPSEnvironment` and
    :class:`~tenpy.linalg.networks.mps.TransferMatrix` should also work for the
    :class:`PurificationMPS` defined here.
    For example, you can use :meth:`~tenpy.networks.mps.MPSEnvironment.expectation_value`
    for the expectation value of operators between different PurificationMPS.
    However, this makes only sense if the *same* disentangler was applied
    to the `bra` and `ket` PurificationMPS.

.. note ::
    The literature (e.g. section 7.2 of :cite:`schollwoeck2011` or :cite:`karrasch2013`) suggests
    to use a `singlet` as a maximally entangled state.
    Here, we use instead the identity :math:`\delta_{p,q}`, since it is easier to
    generalize for `p` running over more than two indices, and allows a simple use of charge
    conservation with the above `qconj` convention.
    Moreover, we don't split the physical and auxiliary space into separate sites, which makes
    TEBD as costly as :math:`O(d^6 \chi^3)`.
"""
# Copyright (C) TeNPy Developers, Apache license

import copy
import numpy as np

from .mps import MPS
from ..linalg import np_conserved as npc
from ..tools.math import entropy

__all__ = ['PurificationMPS', 'convert_model_purification_canonical_conserve_ancilla_charge']


class PurificationMPS(MPS):
    r"""An MPS representing a finite-temperature ensemble using purification.

    Similar as an MPS, but each `B` has now the four legs ``'vL', 'vR', 'p', 'q'``.
    From the point of algorithms, it is to be considered as a usual MPS by combining the legs
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
        """Sanity check, raises ValueErrors, if something is wrong."""
        for B in self._B:
            if not set(['vL', 'vR', 'p', 'q']) <= set(B.get_leg_labels()):
                raise ValueError("B has wrong labels " + repr(B.get_leg_labels()))
        super().test_sanity()

    @classmethod
    def from_infiniteT(cls, sites, bc='finite', form='B', dtype=np.float64):
        """Initial state corresponding to grand-canonical infinite-temperature ensemble.

        Parameters
        ----------
        sites : list of :class:`~tenpy.networks.site.Site`
            The sites defining the local Hilbert space.
            For usual :class:`tenpy.models.model.Model` given by `model.lat.mps_sites()`.
        bc : {'finite', 'segment', 'infinite'}
            MPS boundary conditions as described in :class:`~tenpy.networks.mps.MPS`.
        form : (list of) {``'B' | 'A' | 'C' | 'G' | None`` | tuple(float, float)}
            The canonical form of the stored 'matrices', see table in :mod:`~tenpy.networks.mps`.
            A single choice holds for all of the entries.
        dtype : type or string
            The data type of the array entries.

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
            B = npc.diag(1., p_leg, dtype, ['p', 'q']) / sites[i].dim**0.5
            # leg `q` has the physical leg with opposite `qconj`
            B = B.add_trivial_leg(0, label='vL', qconj=+1).add_trivial_leg(1, label='vR', qconj=-1)
            Bs[i] = B
        res = cls(sites, Bs, S, bc, form)
        return res

    @classmethod
    def from_infiniteT_canonical(cls, sites, charge_sector, dtype=np.float64,
                                 conserve_ancilla_charge=False):
        """Initial state corresponding to *canonical* infinite-temperature ensemble.

        Works only for finite boundary conditions, following the idea outlined in
        :cite:`barthel2016`.

        Parameters
        ----------
        sites : list of :class:`~tenpy.networks.site.Site`
            The sites defining the local Hilbert space (on the physical legs).
            For usual :class:`tenpy.models.model.Model` given by `model.lat.mps_sites()`.
        charge_sector : tuple of int
            The desired charge sector to be taken for the canonical ensemble.
        dtype : type or string
            The data type of the array entries.
        conserve_ancilla_charge : bool
            Whether to conserve the charges on the ancilla leg.
            If False, do *not* double the number of conserved charges to get a separate charge
            for the ancilla degrees of freedom.
            If True, separately conserve charges on physical and ancilla spaces.
            In that case, use the function
            :func:`convert_model_purification_canonical_conserve_ancilla_charge`
            to get a converted model before using algorithms like the `PurificationTEBD`.

        Returns
        -------
        infiniteT_MPS : :class:`PurificationMPS`
            Describes the infinite-temperature (grand canonical) ensemble,
            i.e. expectation values give a trace over all basis states.

        """
        sites = list(sites)
        L = len(sites)
        assert L > 0
        chinfo = sites[0].leg.chinfo
        assert all(s.leg.chinfo == chinfo for s in sites), "Charge Info for all sites must be identical"
        # get a 'charge_tree', (list of sets with possible charges for each site, including 0 to the left)
        charge_tree = cls.get_charge_tree_for_given_charge_sector(sites, charge_sector)
        # get charges from charge_tree in correct array form
        Q_L_arrays = []
        for possible_charges_L in charge_tree:
            Q_L_arrays.append(np.array(list(possible_charges_L)))

        # now we can define the tensors following section VI.C) of [barthel2016]_:
        # B[vL, vR, p, q] = delta_{p,q} delta_{Q(p) + Q(vL), Q(vR)}
        # the normalization will be ensured by a call to `canonical_form_finite()` in the end.
        Bs = []
        Ss = [np.ones(1, dtype=np.float64)]
        Q_R = Q_L_arrays[0]
        if not conserve_ancilla_charge:  # cac := conserve ancilla charges
            leg_R = npc.LegCharge.from_qflat(chinfo, Q_R, qconj=-1)
        else:
            chinfo_cac = npc.ChargeInfo(list(chinfo.mod) * 2,
                                        chinfo.names + [n + ' ancilla' for n in chinfo.names])
            Q_R_cac = chinfo_cac.make_valid(np.hstack([Q_R, -Q_R]))
            leg_R = npc.LegCharge.from_qflat(chinfo_cac, Q_R_cac, qconj=-1)
            sites_cac = []
        for i in range(L):
            leg_p = sites[i].leg
            Q_p = leg_p.to_qflat()
            Q_L = Q_L_arrays[i]
            Q_R = Q_L_arrays[i+1]
            Q_R_map = dict((tuple(q),i) for i, q in enumerate(Q_R))

            leg_L = leg_R.conj()
            if not conserve_ancilla_charge:
                leg_q = npc.LegCharge.from_trivial(leg_p.ind_len, chinfo, -leg_p.qconj)
                leg_R = npc.LegCharge.from_qflat(chinfo, Q_R, qconj=-1)
            else:
                Q_p_cac = np.hstack([Q_p, np.zeros_like(Q_p)])
                Q_q_cac = np.hstack([np.zeros_like(Q_p), Q_p])
                Q_R_cac = chinfo_cac.make_valid(np.hstack([Q_R, -Q_R]))
                leg_p = npc.LegCharge.from_qflat(chinfo_cac, Q_p_cac, qconj=+1)
                leg_q = npc.LegCharge.from_qflat(chinfo_cac, Q_q_cac, qconj=-1)
                leg_R = npc.LegCharge.from_qflat(chinfo_cac, Q_R_cac, qconj=-1)
                s_cac = copy.copy(sites[i])
                s_cac.change_charge(leg_p)  # note: if Q_p is sorted, so are Q_{p,q}_cac
                sites_cac.append(s_cac)
            B = npc.zeros([leg_L, leg_R, leg_p, leg_q],
                          dtype=dtype,
                          labels=['vL', 'vR', 'p', 'q'])
            for j in range(leg_p.ind_len):
                Q_p_j = Q_p[j]
                for vL, Q_L_vL in enumerate(Q_L):
                    Q_R_vR = tuple(chinfo.make_valid(Q_L_vL + Q_p_j))
                    vR = Q_R_map.get(Q_R_vR, None)
                    if vR is not None:
                        B[vL, vR, j, j] = 1.  # add an entry in the tensor
                    # else: dropped Q_R_vR since it can't reach charge_sector on the right any more
            Bs.append(B)
            Ss.append(np.ones(B.shape[1], np.float64))

        if conserve_ancilla_charge:
            sites = sites_cac
        res = cls(sites, Bs, Ss, 'finite', form='B')
        res.canonical_form_finite()  # calculate S values and normalize
        return res

    def entanglement_entropy_segment(self, segment=[0], first_site=None, n=1, legs='p'):
        r"""Calculate entanglement entropy for general geometry of the bipartition.

        This function is similar as :meth:`entanglement_entropy`,
        but for more general geometry of the region `A` to be a segment of a *few* sites.

        This is achieved by explicitly calculating the reduced density matrix of `A`
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
            `n=1` (default) is the usual von-Neumann entanglement entropy,
            otherwise the `n`-th Renyi entropy.
        leg : 'p', 'q', 'pq'
            Whether we look at the entanglement entropy in both (`pq`) or
            only one of auxiliary (`q`) and physical (`p`) space.

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
            only one of auxiliary (`q`) and physical (`p`) space.

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

    def sample_measurements(self,
                            sample_q,
                            first_site=0,
                            last_site=None,
                            ops=None,
                            rng=None,
                            norm_tol=1.e-12,
                            complex_amplitude=True):
        """Sample measurement results in the computational basis.

        See :meth:`tenpy.networks.mps.MPS.sample_measurements` for documentation of the function.
        The only functional difference between these two functions is that we must now deal with
        the ancilla leg on each site. There are two options, specified by `sample_q`:

        1. Sample both the p and q leg on each site; at the end, forget about the outcomes for the
        q leg to sample from the distribution just on the physical legs. The probability we return
        is the joint probability of both p and q outcomes. We don't care about in which basis we
        sample the q legs, and additionally we do not return the q outcomes to the user.

        2. Leave the ancilla leg behind on each site. Then we sample directly from the distribution
        on p legs, but this is more expensive. The total cost of sampling is now O(chi^3) rather
        than O(chi^2). The returned probability is just that of the p outcomes, so this is physical.

        Below we list the differences in the parameters and return values from the MPS function.

        Parameters
        ----------
        sample_q : bool
             Do we sample the q leg (True) or leave it behind (False)?
        first_site, last_site, ops, rng, norm_tol :
            Same as in :meth:`tenpy.networks.mps.MPS.sample_measurements`.
        complex_amplitude : bool
            Do we return the complex amplitude (``True``) of the sampled bit string or the
            probability (``False``), which is the ``abs(amplitude) ** 2``.
            Only ``False`` is supported for :class:`PurificationMPS`.

        Returns
        -------
        sigmas : list of int | list of float
            On each site the index of the local basis that was measured for the PHYSICAL site only,
            as specified in the corresponding :class:`~tenpy.networks.site.Site` in :attr:`sites`.
            Note that this can change depending on whether/what charges you conserve!
            Explicitly specifying the measurement operator will avoid that issue.
            We DO NOT return any measurement outcomes/indices for the ancilla leg, even if they
            are explicitly sampled. This is because to get expectation values of the density matrix,
            one should trace over (i.e. forget the outcome of) the ancilla legs.
        probability : float
            If `sample_q` == False, the probability ``trace(|psi><psi|sigmas...><sigmas...|)``,
            i.e. the probability of measuring ``|sigmas...>`` on the physical legs.
            If `sample_q` == True, we return the probability of measuring a particular configuration
            on both physical and ancilla legs, even though we don't return the ancilla configuration.
            Hence, the returned probability isn't really meaningful.
        """
        if complex_amplitude:
            raise ValueError("Sampling a purification MPS only retuns the probability of the sampled string; rerun with 'complex_amplitude=False'.")

        if last_site is None:
            last_site = self.L - 1
        if rng is None:
            rng = np.random.default_rng()
        sigmas = []
        total_probability = 1.
        theta = self.get_theta(first_site, n=1).replace_labels(['p0', 'q0'], ['p', 'q'])
        for i in range(first_site, last_site + 1):
            # theta = wave function in basis vL [sigmas...] p q vR
            # where the `sigmas` are already fixed to the measurement results
            i0 = self._to_valid_index(i)
            site = self.sites[i0]
            if ops is not None:
                op_name = ops[(i - first_site) % len(ops)]
                op = site.get_op(op_name).transpose(['p', 'p*'])
                if npc.norm(op - op.conj().transpose()) > 1.e-13:
                    raise ValueError(f"measurement operator {op_name!r} not hermitian")
                W, V = npc.eigh(op)
                theta = npc.tensordot(V.conj(), theta, axes=['p*', 'p']).replace_label('eig*', 'p')
            else:
                W = np.arange(site.dim)
            # perform a projective measurement:
            # trace out rest except site `i`
            if not sample_q:
                rho = npc.tensordot(theta.conj(), theta, [['vL*', 'vR*', 'q*'], ['vL', 'vR', 'q']]) # physical RDM on site i
                # probabilities p(sigma) = <sigma|rho|sigma>
                rho_diag = np.abs(np.diag(rho.to_ndarray()))  # abs: real dtype & roundoff err
                if abs(np.sum(rho_diag) - 1.) > norm_tol:
                    raise ValueError("not normalized to `norm_tol`")
                rho_diag /= np.sum(rho_diag)
                sigma = rng.choice(site.dim, p=rho_diag)  # randomly select index from probabilities
                sigmas.append(W[sigma]) # return eigenvalue if an op was specified
                theta = theta.take_slice(sigma, 'p')  # project to sigma in theta; now has legs vL (trivial), q, vR
                probability = rho_diag[sigma] # this is probability of seeing sigma conditioned on previous results.
                # rho_diag[sigma] which should be the same as the norm of theta squared
                # assert np.isclose(probability, npc.tensordot(theta.conj(), theta, axes=(['vL*', 'vR*', 'q*'], ['vL', 'vR', 'q'])))
                total_probability *= probability    # probability of p outcome
            else:
                W2 = np.arange(site.dim)    # outcomes for q leg
                # Sample p
                rho = npc.tensordot(theta.conj(), theta, [['vL*', 'vR*', 'q*'], ['vL', 'vR', 'q']]) # physical RDM on site i
                # probabilities p(sigma) = <sigma|rho|sigma>
                rho_diag = np.abs(np.diag(rho.to_ndarray()))  # abs: real dtype & roundoff err
                if abs(np.sum(rho_diag) - 1.) > norm_tol:
                    raise ValueError("not normalized to `norm_tol`")
                rho_diag /= np.sum(rho_diag)
                sigma_1 = rng.choice(site.dim, p=rho_diag)  # randomly select index from probabilities
                probability = rho_diag[sigma_1] # rho_diag[sigma_1] is probability of p outcome, conditioned on all previous outcomes
                # So by Bayes' rule, we now have the joint probability of all sampled outcomes.
                theta = theta.take_slice([sigma_1], ['p'])  # project to sigma in theta; now has legs vL (trivial), vR

                # Sample q
                rho = npc.tensordot(theta.conj(), theta, [['vL*', 'vR*'], ['vL', 'vR']]) # physical RDM on site i
                # probabilities p(sigma) = <sigma|rho|sigma>
                rho_diag = np.abs(np.diag(rho.to_ndarray()))  # abs: real dtype & roundoff err
                # rho_diag will nothave trace = 1 since we didn't normalize theta after slicing.
                rho_diag /= np.sum(rho_diag)
                sigma_2 = rng.choice(site.dim, p=rho_diag)  # randomly select index from probabilities
                probability *= rho_diag[sigma_2] # probabilty of all outcomes seen so far.
                theta = theta.take_slice([sigma_2], ['q'])  # project to sigma in theta; now has legs vL (trivial), vR

                sigmas.append(W[sigma_1]) # For ancilla, we do not return the sampled index W[sigma_2] since the outcome
                # is in an arbitrary basis.
                # rho_diag[sigma] which should be the same as the norm of theta squared
                # assert np.isclose(probability, npc.tensordot(theta.conj(), theta, axes=(['vL*', 'vR*'], ['vL', 'vR'])))
                total_probability *= probability    # probability of q outcome

            if i != last_site:
                # Move orthogonality center to the next site
                theta = theta / npc.norm(theta) # renormalize
                B = self.get_B(i + 1)
                if sample_q:
                    theta = npc.tensordot(theta, B, axes=['vR', 'vL'])
                else:
                    Q, R = npc.qr(theta.combine_legs(['vL', 'q']),
                                  inner_labels=['vR', 'vL'],
                                  pos_diag_R=True,
                                  )
                    theta = npc.tensordot(R, B, axes=['vR', 'vL'])
                # B is right-canonical -> theta still normalized
            elif self.bc == 'finite' and first_site == 0 and last_site == self.L - 1 and sample_q:
                assert theta.shape == (1,1) # This contains the phase; but we don't want this since
                # we are returning the probability, not the weight.
        return sigmas, total_probability

    def _corr_up_diag(self, ops1, ops2, i, j_gtr, opstr, str_on_first, apply_opstr_first):
        """correlation function above the diagonal: for fixed i and all j in j_gtr, j > i."""
        # compared to MPS._corr_up_diag just perform additional contractions of the 'q'
        op1, _ = self.get_op(ops1, i)
        opstr1, _ = self.get_op(opstr, i)
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
                op2, _ = self.get_op(ops2, r)
                Cij = npc.tensordot(op2, C, axes=['p*', 'p'])
                Cij = npc.inner(B.conj(),
                                Cij,
                                axes=[['vL*', 'p*', 'q*', 'vR*'], ['vR*', 'p', 'q', 'vR']])
                res.append(Cij)
                js.pop()
            if len(js) > 0:
                op, _ = self.get_op(opstr, r)
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


def convert_model_purification_canonical_conserve_ancilla_charge(model):
    """Extend charges of model for :meth:`PurificationMPS.from_infiniteT_canonical`.

    Parameters
    ----------
    model : :class:`tenpy.models.model.Model`
        Model to be converted.

    Returns
    -------
    model_with_extra_charges : :class:`tenpy.models.model.Model`
        Shallow copy of the `model` with charges of sites, `H_MPO` and `H_bond` adjusted
        to fit the doubled (with 0 extended) charges of the canonical ensemble of the
        :class:`PurificationMPS`. The number of
    """
    # cac := conserve_ancilla_charge
    model = model.copy()
    chinfo = model.lat.unit_cell[0].leg.chinfo
    chinfo_cac = npc.ChargeInfo(list(chinfo.mod) * 2,
                                chinfo.names + [n + ' ancilla' for n in chinfo.names])

    converted_sites_cache = {}
    def _convert_site(site):
        s = converted_sites_cache.get(site, None)
        if s is not None:
            return s
        s_new = copy.copy(site)
        leg_p = s_new.leg
        Q_p = leg_p.charges
        Q_p_cac = np.hstack([Q_p, np.zeros_like(Q_p)])  # still sorted
        leg_p_cac = npc.LegCharge.from_qind(chinfo_cac, leg_p.slices, Q_p_cac, leg_p.qconj)
        s_new.change_charge(leg_p_cac)
        converted_sites_cache[site] = s_new
        return s_new

    model.lat = model.lat.copy()
    model.lat.unit_cell = [_convert_site(s) for s in model.lat.unit_cell]

    if hasattr(model, 'H_MPO'):
        model.H_MPO = H_MPO = model.H_MPO.copy()
        H_MPO.sites = [_convert_site(s) for s in H_MPO.sites]
        H_MPO.chinfo = chinfo_cac
        new_W = []
        for W in H_MPO._W:
            W = W.copy()
            W.itranspose(['wL', 'wR', 'p', 'p*'])
            W.legs = W.legs[:]
            for i in range(3):
                leg = W.legs[i]
                if i < 2:
                    Q = np.hstack([leg.charges, -leg.charges])  # wL, wR
                else:
                    Q = np.hstack([leg.charges, np.zeros_like(leg.charges)])  # p
                W.legs[i] = npc.LegCharge.from_qind(chinfo_cac,
                                                    leg.slices,
                                                    chinfo_cac.make_valid(Q),
                                                    leg.qconj)
            W.qtotal = np.hstack([W.qtotal, np.zeros_like(W.qtotal)])
            W.legs[3] = W.legs[2].conj()
            new_W.append(W)
        H_MPO._W = new_W

    if hasattr(model, 'H_bond'):
        sites = model.lat.mps_sites()  # already updated!
        model.H_bond = H_bond = model.H_bond[:]
        L = len(sites)
        assert len(sites) == len(H_bond)
        for i, H in enumerate(H_bond):
            if H is None:
                continue
            leg_p0 = sites[(i-1) % L].leg
            leg_p1 = sites[i].leg
            H = H.transpose(['p0', 'p1', 'p0*', 'p1*'])  # copy!
            H.chinfo = chinfo_cac
            H.legs = [leg_p0, leg_p1, leg_p0.conj(), leg_p1.conj()]
            H.qtotal = np.hstack([H.qtotal, np.zeros_like(H.qtotal)])
            H.test_sanity()
            H_bond[i] = H
    model.test_sanity()
    return model
