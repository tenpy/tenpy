"""Algorithms for using Purification.
"""

# Copyright 2019-2021 TeNPy Developers, GNU GPLv3

import numpy as np
import tenpy.linalg.np_conserved as npc
from . import tebd
from ..tools.params import asConfig
from .mps_common import VariationalApplyMPO, TwoSiteH
from .truncation import svd_theta, TruncationError
from .disentangler import get_disentangler

__all__ = ['PurificationTwoSiteU', 'PurificationApplyMPO', 'PurificationTEBD', 'PurificationTEBD2']


class PurificationTwoSiteU(TwoSiteH):
    """Variant of `TwoSiteH` suitable for purification.

    The MPO gets only applied to the physical legs `p0`, `p1`, the ancialla legs `q0`, `q1` of
    `theta` are ignored.
    """
    length = 2
    acts_on = ['vL', 'p0', 'q0', 'p1', 'q1', 'vR']

    # initialization, matvec, combine_theta and adjoint derived from `TwoSiteH` work.
    # to_matrix() should in general multiply with identity on q0/q1; but it isn't used anyways.

    def combine_Heff(self):
        super().combine_Heff()  # almost correct
        self.acts_on = ['(vL.p0)', 'q0', 'q1', '(p1.vR)']  # overwrites class attribute!


class PurificationApplyMPO(VariationalApplyMPO):
    """Variant of `VariationalApplyMPO` suitable for purification."""
    EffectiveH = PurificationTwoSiteU

    def update_local(self, _, optimize=True):
        """Perform local update.

        This simply contracts the environments and `theta` from the `ket` to get an updated
        `theta` for the bra `self.psi` (to be changed in place).
        """
        i0 = self.i0
        self.make_eff_H()
        th = self.env.ket.get_theta(i0, n=2)  # ket is old psi
        th = self.eff_H.combine_theta(th)
        th = self.eff_H.matvec(th)
        if self.eff_H.combine:
            th = th.split_legs()
        th = th.combine_legs([['vL', 'p0', 'q0'], ['p1', 'q1', 'vR']], qconj=[+1, -1])
        return self.update_new_psi(th)

    def update_new_psi(self, theta):
        i0 = self.i0
        new_psi = self.psi
        qtotal_i0 = new_psi.get_B(i0, form=None).qtotal
        U, S, VH, err, renormalize = svd_theta(theta,
                                               self.trunc_params,
                                               qtotal_LR=[qtotal_i0, None],
                                               inner_labels=['vR', 'vL'])
        self.renormalize.append(renormalize)
        # TODO: up to the `renormalize`, we could use `new_psi.set_svd_theta`.
        B0 = U.split_legs(['(vL.p0.q0)']).replace_labels(['p0', 'q0'], ['p', 'q'])
        B1 = VH.split_legs(['(p1.q1.vR)']).replace_labels(['p1', 'q1'], ['p', 'q'])
        new_psi.set_B(i0, B0, form='A')  # left-canonical
        new_psi.set_B(i0 + 1, B1, form='B')  # right-canonical
        new_psi.set_SR(i0, S)
        # the old stored environments are now invalid
        # => delete them to ensure that they get calculated again in :meth:`update_LP` / RP
        for o_env in self.ortho_to_envs:
            o_env.del_LP(i0 + 1)
            o_env.del_RP(i0)
        self.env.del_LP(i0 + 1)
        self.env.del_RP(i0)
        return {'U': U, 'VH': VH, 'err': err}


class PurificationTEBD(tebd.TEBDEngine):
    r"""Time evolving block decimation (TEBD) for purification MPS.

    .. deprecated :: 0.6.0
        Renamed parameter/attribute `TEBD_params` to :attr:`options`.

    Parameters
    ----------
    psi : :class:`~tenpy.networs.purification_mps.PurificationMPS`
        Initial state to be time evolved. Modified in place.
    model : :class:`~tenpy.models.NearestNeighborModel`
        The model representing the Hamiltonian for which we want to find the ground state.
    options : dict
        Further optional parameters as described in the following table.
        Use ``verbose=1`` to print the used parameters during runtime.
        See :func:`run` and :func:`run_GS` for more details.

    Attributes
    ----------
    used_disentangler : :class:`Disentangler`
        The disentangler to be used on the auxiliar indices.
        Chosen by :func:`get_disentangler`, called with the TEBD parameter ``'disentangle'``.
        Defaults to the trivial disentangler for ``options['disentangle']=None``.
    _disent_iterations : 1D ndarray
        Number of iterations performed on all bonds, including trivial bonds; lenght `L`.
    _guess_U_disent : list of list of npc.Array
        Same index strucuture as `self._U`: for each two-site U of the physical time evolution
        the disentangler from the last application. Initialized to identities.
    """
    def __init__(self, psi, model, options):
        super().__init__(psi, model, asConfig(options, 'PurificationTEBD'))
        self._disent_iterations = np.zeros(psi.L)
        self._guess_U_disent = None  # will be set in calc_U
        method = self.options.get('disentangle', None)
        self.used_disentangler = get_disentangler(str(method), self)

    def run_imaginary(self, beta):
        """Run imaginary time evolution to cool down to the given `beta`.

        Note that we don't change the `norm` attribute of the MPS, i.e. normalization is preserved.

        Parameters
        ----------
        beta : float
            The inverse temperature `beta` = 1/T, by which we should cool down.
            We evolve to the closest multiple of ``options['dt']``,
            see also :attr:`evolved_time`.
        """
        delta_t = self.options.get('dt', 0.1)
        TrotterOrder = 2  # currently, imaginary time evolution works only for second order.
        self.calc_U(TrotterOrder, delta_t, type_evo='imag')
        self.update_imag(N_steps=int(beta / delta_t + 0.5))
        if self.verbose >= 1:
            E = np.average(self.model.bond_energies(self.psi))
            S = np.average(self.psi.entanglement_entropy())
            print("--> time={t:.6f}, E_bond={E:.10f}, S={S:.10f}".format(t=self.evolved_time,
                                                                         E=E.real,
                                                                         S=S.real))

    @property
    def disent_iterations(self):
        """For each bond the total number of iterations performed in any :class:`Disentangler`."""
        return self._disent_iterations[self.psi.nontrivial_bonds]

    def calc_U(self, order, delta_t, type_evo='real', E_offset=None):
        """see :meth:`~tenpy.algorithms.tebd.eng.calc_U`"""
        super().calc_U(order, delta_t, type_evo, E_offset)
        self._guess_U_disent = [[None] * len(Us) for Us in self._U]

    def update_bond(self, i, U_bond):
        """Updates the B matrices on a given bond.

        Function that updates the B matrices, the bond matrix s between and the
        bond dimension chi for bond i. This would look something like::

        |           |             |
        |     ... - B1  -  s  -  B2 - ...
        |           |             |
        |           |-------------|
        |           |      U      |
        |           |-------------|
        |           |             |


        Parameters
        ----------
        i : int
            Bond index; we update the matrices at sites ``i-1, i``.
        U_bond : :class:`~tenpy.linalg.np_conserved.Array`
            The bond operator which we apply to the wave function.
            We expect labels ``'p0', 'p1', 'p0*', 'p1*'`` for `U_bond`.

        Returns
        -------
        trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
            The error of the represented state which is introduced by the truncation
            during this update step.
        """
        i0, i1 = i - 1, i
        if self.verbose >= 30:
            print("Update sites ({0:d}, {1:d})".format(i0, i1))
        # Construct the theta matrix
        theta = self.psi.get_theta(i0, n=2)  # 'vL', 'vR', 'p0', 'p1', 'q0', 'q1'
        theta = npc.tensordot(U_bond, theta, axes=(['p0*', 'p1*'], ['p0', 'p1']))
        # ##### new hook compared to tebd.TEBDEngine.calc_U
        theta, U_disent = self.disentangle(theta)
        # ####
        theta = theta.combine_legs([('vL', 'p0', 'q0'), ('vR', 'p1', 'q1')], qconj=[+1, -1])

        # Perform the SVD and truncate the wavefunction
        U, S, V, trunc_err, renormalize = svd_theta(theta,
                                                    self.trunc_params,
                                                    inner_labels=['vR', 'vL'])

        # bring back to right-canonical 'B' form and update matrices
        B_R = V.split_legs(1).ireplace_labels(['p1', 'q1'], ['p', 'q'])
        #  In general, we want to do the following:
        #      B_L = U.iscale_axis(S, 'vR')
        #      B_L = B_L.split_legs(0).iscale_axis(self.psi.get_SL(i0)**(-1), 'vL')
        #      B_L = B_L.ireplace_labels(['p0', 'q0'], ['p', 'q'])
        # i.e. with SL = self.psi.get_SL(i0), we have ``B_L = SL**(-1) U S``
        # However, the inverse of SL is problematic, as it might contain very small singular
        # values.  Instead, we calculate ``C == SL**-1 theta == SL**-1 U S V``,
        # such that we obtain ``B_L = SL**-1 U S = SL**-1 U S V V^dagger = C V^dagger``
        C = self.psi.get_theta(i0, n=2, formL=0.)
        # here, C is the same as theta, but without the `S` on the very left
        # (Note: this requires no inverse if the MPS is initially in 'B' canonical form)
        C = npc.tensordot(U_bond, C, axes=(['p0*', 'p1*'], ['p0', 'p1']))  # apply U as for theta
        if U_disent is not None:
            C = npc.tensordot(U_disent, C, axes=[['q0*', 'q1*'], ['q0', 'q1']])
        B_L = npc.tensordot(C.combine_legs(('vR', 'p1', 'q1'), pipes=theta.legs[1]),
                            V.conj(),
                            axes=['(vR.p1.q1)', '(vR*.p1*.q1*)'])
        B_L.ireplace_labels(['vL*', 'p0', 'q0'], ['vR', 'p', 'q'])
        B_L /= renormalize  # re-normalize to <psi|psi> = 1
        self.psi.set_SR(i0, S)
        self.psi.set_B(i0, B_L, form='B')
        self.psi.set_B(i1, B_R, form='B')
        self._trunc_err_bonds[i] = self._trunc_err_bonds[i] + trunc_err
        return trunc_err

    def update_bond_imag(self, i, U_bond):
        """Update a bond with a (possibly non-unitary) `U_bond`.

        Similar as :meth:`update_bond`; but after the SVD just keep the `A, S, B` canonical form.
        In that way, one can sweep left or right without using old singular values,
        thus preserving the canonical form during imaginary time evolution.

        Parameters
        ----------
        i : int
            Bond index; we update the matrices at sites ``i-1, i``.
        U_bond : :class:`~tenpy.linalg.np_conserved.Array`
            The bond operator which we apply to the wave function.
            We expect labels ``'p0', 'p1', 'p0*', 'p1*'``.

        Returns
        -------
        trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
            The error of the represented state which is introduced by the truncation
            during this update step.
        """
        i0, i1 = i - 1, i
        if self.verbose >= 100:
            print("Update sites ({0:d}, {1:d})".format(i0, i1))
        # Construct the theta matrix
        theta = self.psi.get_theta(i0, n=2)  # 'vL', 'vR', 'p0', 'q0', 'p1', 'q1'
        theta = npc.tensordot(U_bond, theta, axes=(['p0*', 'p1*'], ['p0', 'p1']))
        theta = theta.combine_legs([('vL', 'p0', 'q0'), ('vR', 'p1', 'q1')], qconj=[+1, -1])
        # Perform the SVD and truncate the wavefunction
        U, S, V, trunc_err, renormalize = svd_theta(theta,
                                                    self.trunc_params,
                                                    inner_labels=['vR', 'vL'])
        # Split legs and update matrices
        B_R = V.split_legs(1).ireplace_labels(['p1', 'q1'], ['p', 'q'])
        A_L = U.split_legs(0).ireplace_labels(['p0', 'q0'], ['p', 'q'])
        self.psi.set_SR(i0, S)
        self.psi.set_B(i0, A_L, form='A')
        self.psi.set_B(i1, B_R, form='B')
        self._trunc_err_bonds[i] = self._trunc_err_bonds[i] + trunc_err
        return trunc_err

    def disentangle(self, theta):
        r"""Disentangle `theta` before splitting with svd.

        For the purification we write :math:`\rho_P = Tr_Q{|\psi_{P,Q}><\psi_{P,Q}|}`. Thus, we
        can actually apply any unitary to the auxiliar `Q` space of :math:`|\psi>` without
        changing the result.

        .. note ::
            We have to apply the *same* unitary to the 'bra' and 'ket' used for expectation values
            / correlation functions!

        The behaviour of this function is set by :attr:`used_disentangler`,
        which in turn is obtained from ``get_disentangler(options['disentangle'])``,
        see :func:`get_disentangler` for details on the syntax.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Wave function to disentangle, with legs ``'vL', 'vR', 'p0', 'p1', 'q0', 'q1'``.

        Returns
        -------
        theta_disentangled : :class:`~tenpy.linalg.np_conserved.Array`
            Disentangled `theta`; ``npc.tensordot(U, theta, axes=[['q0*', 'q1*'], ['q0', 'q1']])``.
        U : :class:`~tenpy.linalg.conserved.Array`
            The unitary used to disentangle `theta`, with labels ``'q0', 'q1', 'q0*', 'q1*'``.
            If no unitary was found/applied, it might also be ``None``.
        """
        theta, U = self.used_disentangler(theta)
        U_idx_dt, i = self._update_index
        if U_idx_dt is not None:
            self._guess_U_disent[U_idx_dt][i] = U  # save result as guess for `LastDisentangler`
        return theta, U

    def disentangle_global(self, pair=None):
        """Try global disentangling by determining the maximally entangled pairs of sites.

        Caclulate the mutual information (in the auxiliar space) between two sites and determine
        where it is maximal. Disentangle these two sites with :meth:`disentangle`
        """
        max_range = self.options.get('disent_gl_maxrange', 10)
        if pair is None:
            coords, mutinf = self.psi.mutinf_two_site(max_range, legs='q')
            # TODO: recalculate mutinf only as necessary and do multiple steps at once...
            sorted = np.argsort(mutinf)
            pair = coords[sorted[-1]]
        i, j = pair
        #  for i, j in coords[sorted[-1:]]:
        if self.verbose > 10:
            print('disentangle global pair ' + repr((i, j)))
        self._disentangle_two_site(i, j)
        return i, j  # TODO
        # done

    def disentangle_global_nsite(self, n=2):
        """Perform a sweep through the system and disentangle with :meth:`disentangle_n_site`.

        Parameters
        ----------
        n: int
            maximal number of sites to disentangle at once.
        """
        for i in range(0, self.psi.L - n + 1):  # sweep left to right
            self._update_index = None, i
            theta = self.psi.get_theta(i, n=n)
            self.disentangle_n_site(i, n, theta)  # works recursively
        for i in range(self.psi.L - n, -1, -1):  # sweep right to left
            self._update_index = None, i
            theta = self.psi.get_theta(i, n=n)
            self.disentangle_n_site(i, n, theta)  # works recursively
        self._update_index = None

    def disentangle_n_site(self, i, n, theta):
        r"""Generalization of :meth:`disentangle` to `n` sites.

        Simply group left and right `n`/2 physical legs, adjust labels, and
        apply :meth:`disentangle` to disentangle the central bond.
        Recursively proceed to disentangle left and right parts afterwards.
        Scales (for even `n`) as :math:`O(\chi^3 d^n d^{n/2})`.
        """
        assert (n >= 2)
        n1 = n // 2
        n2 = n - n1
        p = ['p' + str(j) for j in range(n)]  # labels of theta to be separated
        q = ['q' + str(j) for j in range(n)]
        pL, pR = p[:n1], p[n1:]
        qL, qR = q[:n1], q[n1:]
        theta = theta.combine_legs([pL, qL, pR, qR], qconj=[+1, -1, +1, -1], new_axes=[1, 2, 3, 4])
        _, p0, q0, p1, q1, _ = theta.get_leg_labels()  # keep the labels for later
        theta.ireplace_labels([p0, q0, p1, q1], ['p0', 'q0', 'p1', 'q1'])
        theta, _ = self.disentangle(theta)  # apply two-site disentangling
        theta = theta.combine_legs([('vL', 'p0', 'q0'), ('vR', 'p1', 'q1')], qconj=[+1, -1])

        # Perform the SVD and truncate the wavefunction
        U, S, V, trunc_err, renormalize = svd_theta(theta,
                                                    self.trunc_params,
                                                    inner_labels=['vR', 'vL'])
        self.psi.set_SL(i + n1, S)  # update S
        if n1 == 1:
            # save U as left B in psi
            U = U.split_legs(0).ireplace_labels(['p0', 'q0'], ['(p)', '(q)'])
            U = U.split_legs(['(p)', '(q)'])
            self.psi.set_B(i, U, form='A')  # TODO: might want to do this inversion-free?
        else:
            # disentangle left n1-site wave function recursively
            theta_L = U.iscale_axis(S, 1).split_legs(0)
            theta_L = theta_L.ireplace_labels(['p0', 'q0'], [p0, q0]).split_legs([p0, q0])
            self.disentangle_n_site(i, n1, theta_L)
        if n2 == 1:
            # save V as right B in psi
            V = V.split_legs(1).ireplace_labels(['p1', 'q1'], ['(p)', '(q)'])
            V = V.split_legs(['(p)', '(q)'])
            self.psi.set_B(i + n1, V, form='B')
        else:
            # disentangle right n2-site wave function recursively
            theta_R = V.iscale_axis(S, 0).split_legs(1)
            theta_R = theta_R.ireplace_labels(['p1', 'q1'], [p1, q1]).split_legs([p1, q1])
            theta_R.ireplace_labels(pR, p[:n2]).ireplace_labels(qR, q[:n2])
            self.disentangle_n_site(i + n1, n2, theta_R)

    def _disentangle_two_site(self, i, j):
        """swap until i and j are next to each other and use :meth:`disentangle`; swap back."""
        on_way = self.options.get('disent_gl_on_swap', False)
        if not self.psi.finite:
            raise NotImplementedError  # adjust: what's the shortest path?
        assert (i < j)
        for j0 in range(j, i + 1, -1):  # j0 = current site of `j`
            # originial leg `j` is at j0
            self._update_index = None, j0
            self._swap_disentangle_bond(j0, swap=True, disentangle=False)  # swap j0-1, j0
            # originial leg is at `j0-1`
        # disentangle i, i+1
        self._update_index = None, i + 1
        self._swap_disentangle_bond(i + 1, swap=False, disentangle=True)
        for j0 in range(i + 1, j):  # j0 = current site of `j`
            # originial leg `j` is at j0
            self._update_index = None, j0 + 1
            self._swap_disentangle_bond(j0 + 1, disentangle=on_way)  # swap j0, j0+1
            # originial leg is at `j0+1`
        self._update_index = None  # done

    def _swap_disentangle_bond(self, i, swap=True, disentangle=False):
        """swap sites (i-1, i) (if swap = True) """
        # very similar to update_bond
        i0, i1 = i - 1, i
        if self.verbose >= 30:
            print("Update sites ({0:d}, {1:d}), swap={2!s}, disentangle={3!s}".format(
                i0, i1, swap, disentangle))
        # Construct the theta matrix
        theta = self.psi.get_theta(i0, n=2)  # 'vL', 'vR', 'p0', 'p1', 'q0', 'q1'
        if swap:
            theta.ireplace_labels(['p0', 'q0', 'p1', 'q1'], ['p1', 'q1', 'p0', 'q0'])
        if disentangle:
            theta, U_disent = self.disentangle(theta)
        theta = theta.combine_legs([('vL', 'p0', 'q0'), ('vR', 'p1', 'q1')], qconj=[+1, -1])

        # Perform the SVD and truncate the wavefunction
        U, S, V, trunc_err, renormalize = svd_theta(theta,
                                                    self.trunc_params,
                                                    inner_labels=['vR', 'vL'])

        # bring back to right-canonical 'B' form and update matrices
        B_R = V.split_legs(1).ireplace_labels(['p1', 'q1'], ['p', 'q'])
        C = self.psi.get_theta(i0, n=2, formL=0.)
        if swap:
            C.ireplace_labels(['p0', 'q0', 'p1', 'q1'], ['p1', 'q1', 'p0', 'q0'])
        if disentangle and U_disent is not None:
            C = npc.tensordot(U_disent, C, axes=[['q0*', 'q1*'], ['q0', 'q1']])
        B_L = npc.tensordot(C.combine_legs(('vR', 'p1', 'q1'), pipes=theta.legs[1]),
                            V.conj(),
                            axes=['(vR.p1.q1)', '(vR*.p1*.q1*)'])
        B_L.ireplace_labels(['vL*', 'p0', 'q0'], ['vR', 'p', 'q'])
        B_L /= renormalize  # re-normalize to <psi|psi> = 1
        self.psi.set_SR(i0, S)
        self.psi.set_B(i0, B_L, form='B')
        self.psi.set_B(i1, B_R, form='B')
        self._trunc_err_bonds[i] = self._trunc_err_bonds[i] + trunc_err
        return trunc_err


class PurificationTEBD2(PurificationTEBD):
    """Similar as PurificationTEBD, but perform sweeps instead of brickwall.

    Instead of the A-B pattern of even/odd bonds used in TEBD, perform sweeps similar as in DMRG
    for real-time evolution (similar as :meth:`~tenpy.algorithms.tebd.TEBDEngine.update_imag` does
    for imaginary time evolution).
    """
    def update(self, N_steps):
        """Evolve by ``N_steps * U_param['dt']``.

        Parameters
        ----------
        N_steps : int
            The number of steps for which the whole lattice should be updated.

        Returns
        -------
        trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
            The error of the represented state which is introduced due to the truncation during
            this sequence of update steps.
        """
        trunc_err = TruncationError()
        order = self._U_param['order']
        assert (order == 2 and self.psi.finite)
        for i in range(N_steps):
            trunc_err += self.update_step(0, False)
            trunc_err += self.update_step(0, True)
        self.evolved_time = self.evolved_time + N_steps * self._U_param['tau']
        self.trunc_err = self.trunc_err + trunc_err  # not += : make a copy!
        # (this is done to avoid problems of users storing self.trunc_err after each `update`)
        return trunc_err

    def update_step(self, U_idx_dt, odd):
        """Updates bonds in unit cell.

        Depending on the choice of `odd`, perform a sweep to the left or right,
        updating once per site with a time step given by U_idx_dt.

        Parameters
        ----------
        U_idx_dt : int
            Time step index in ``self._U``,
            evolve with ``Us[i] = self.U[U_idx_dt][i]`` at bond ``(i-1,i)``.
        odd : bool/int
            Indication of whether to update even (``odd=False,0``) or even (``odd=True,1``) sites

        Returns
        -------
        trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
            The error of the represented state which is introduced due to the truncation
            during this sequence of update steps.
        """
        Us = self._U[U_idx_dt]
        trunc_err = TruncationError()
        if odd:
            sweep = range(1, self.psi.L)  # start with 1: only finite!
        else:
            sweep = range(self.psi.L - 1, 0, -1)
        for i_bond in sweep:
            if Us[i_bond] is None:
                if self.verbose >= 10:
                    print("Skip U_bond element:", i_bond)
                continue  # handles finite vs. infinite boundary conditions
            if self.verbose >= 10:
                print("Apply U_bond element", i_bond)
            self._update_index = (U_idx_dt, i_bond)
            trunc_err += self.update_bond(i_bond, Us[i_bond])
        self._update_index = None
        return trunc_err
