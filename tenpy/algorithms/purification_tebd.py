r"""Time evolving block decimation (TEBD) for MPS of purification.

See introduction in :mod:`~tenpy.networks.purification_mps`.
Time evolution for finite-temperature ensembles.
This can be used to obtain correlation functions in time.
"""

from __future__ import division

from . import tebd
from ..linalg import np_conserved as npc
from .truncation import svd_theta
from ..tools.params import get_parameter
from ..tools.math import entropy

import numpy as np


class PurificationTEBD(tebd.Engine):
    r"""Time evolving block decimation (TEBD) for purification MPS.


    Parameters
    ----------
    psi : :class:`~tenpy.networs.purification_mps.PurificationMPS`
        Initial state to be time evolved. Modified in place.
    model : :class:`~tenpy.models.NearestNeighborModel`
        The model representing the Hamiltonian for which we want to find the ground state.
    TEBD_params : dict
        Further optional parameters as described in the following table.
        Use ``verbose=1`` to print the used parameters during runtime.
        See :func:`run` and :func:`run_GS` for more details.

    Attributes
    ----------
    disent_iterations
    _disent_iterations : 1D ndarray
        Number of iterations performed on all bonds, including trivial bonds; lenght `L`.
    _guess_U_disent : list of list of npc.Array
        Same index strucuture as `self._U`: for each two-site U of the physical time evolution
        the disentangler from the last application. Initialized to identities.
    """
    def __init__(self, psi, model, TEBD_params):
        super(PurificationTEBD, self).__init__(psi, model, TEBD_params)
        self._disent_iterations = np.zeros(psi.L)
        self._guess_U_disent = None  # will be set in calc_U

    def run_imaginary(self, beta):
        """Run imaginary time evolution to cool down to the given `beta`.

        Parameters
        ----------
        beta : float
            The inverse temperature `beta = 1/T`, by which we should cool down.
            We evolve to the closest multiple of TEBD_params['dt'], c.f. :attr:`evolved_time,
            c.f. :attr:`evolved_time`.
        """
        delta_t = get_parameter(self.TEBD_params, 'dt', 0.1, 'PurificationTEBD')
        TrotterOrder = 2  # TODO: currently, imaginary time evolution works only for second order.
        self.calc_U(TrotterOrder, delta_t, type_evo='imag')
        self.update_imag(N_steps=int(beta / delta_t + 0.5))
        if self.verbose >= 1:
            E = np.average(self.model.bond_energies(self.psi))
            S = np.average(self.psi.entanglement_entropy())
            print "--> time={t:.6f}, E_bond={E:.10f}, S={S:.10f}".format(
                t=self.evolved_time, E=E.real, S=S.real)

    @property
    def disent_iterations(self):
        """For each bond the number of iterations in :meth:`disentangle_renyi`"""
        return self._disent_iterations[self.psi.nontrivial_bonds]

    def calc_U(self, order, delta_t, type_evo='real', E_offset=None):
        """see :meth:`~tenpy.algorithms.tebd.eng.calc_U`"""
        super(PurificationTEBD, self).calc_U(order, delta_t, type_evo, E_offset)
        psi = self.psi
        Id_bonds = []
        for i in range(psi.L):
            Id_L = npc.diag(1., psi.sites[(i-1) % psi.L].leg.conj()).set_leg_labels(['q0', 'q0*'])
            Id_R = npc.diag(1., psi.sites[i].leg.conj()).set_leg_labels(['q1', 'q1*'])
            Id_bonds.append(npc.outer(Id_L, Id_R))
        self._guess_U_disent = [list(Id_bonds) for _ in range(len(self._U))]

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
        U_bond : :class:~tenpy.linalg.np_conserved.Array`
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
            print "Update sites ({0:d}, {1:d})".format(i0, i1)
        # Construct the theta matrix
        theta = self.psi.get_theta(i0, n=2)  # 'vL', 'vR', 'p0', 'p1', 'q0', 'q1'
        theta = npc.tensordot(U_bond, theta, axes=(['p0*', 'p1*'], ['p0', 'p1']))
        # ##### new hook compared to tebd.Engine.calc_U
        theta, U_disent = self.disentangle(theta)
        # ####
        theta = theta.combine_legs([('vL', 'p0', 'q0'), ('vR', 'p1', 'q1')], qconj=[+1, -1])

        # Perform the SVD and truncate the wavefunction
        U, S, V, trunc_err, renormalize = svd_theta(
            theta, self.trunc_params, inner_labels=['vR', 'vL'])

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
        B_L = npc.tensordot(
            C.combine_legs(('vR', 'p1', 'q1'), pipes=theta.legs[1]),
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
        U_bond : :class:~tenpy.linalg.np_conserved.Array`
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
            print "Update sites ({0:d}, {1:d})".format(i0, i1)
        # Construct the theta matrix
        theta = self.psi.get_theta(i0, n=2)  # 'vL', 'vR', 'p0', 'q0', 'p1', 'q1'
        theta = npc.tensordot(U_bond, theta, axes=(['p0*', 'p1*'], ['p0', 'p1']))
        theta = theta.combine_legs([('vL', 'p0', 'q0'), ('vR', 'p1', 'q1')], qconj=[+1, -1])
        # Perform the SVD and truncate the wavefunction
        U, S, V, trunc_err, renormalize = svd_theta(
            theta, self.trunc_params, inner_labels=['vR', 'vL'])
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
        have actually apply any unitary to the auxiliar `Q` space of :math:`|\psi>` without
        changing the result.

        .. note :
            We have to apply the *same* unitary to the 'bra' and 'ket' used for expectation values
            / correlation functions!

        Thus function reads out ``TEBD_params['disentangle']``.
        By default (``None``) it does nothing,
        otherwise it calls one of the other `disentangle_*` methods.

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
        """
        disentangle = get_parameter(self.TEBD_params, 'disentangle', None, 'PurificationTEBD')
        if disentangle is None:
            return theta, None
        elif disentangle == 'backwards':
            return self.disentangle_backwards(theta)
        elif disentangle == 'renyi':
            return self.disentangle_renyi(theta)
        elif disentangle == 'diag':
            return self.disentangle_diag(theta)
        # else
        raise ValueError("Invalid 'disentangle': got " + repr(disentangle))

    def disentangle_backwards(self, theta):
        """Disentangle with backwards time evolution.

        See [Karrasch2013]_.

        For the infinite temperature state, ``theta = delta_{p0, q0}*delta_{p1, q1}``.
        Thus, an application of `U_bond` to ``p0, p1`` can be reverted completely by applying
        ``U_bond^{dagger}`` to ``q0, q1``, resulting in the same state.
        This works also for finite temperatures, since `exp(-beta H)` and `exp(-i H t)` commute.
        Once we apply an operator to measure correlation function, the disentangling
        breaks down -- though, for a local operator only in it's light-cone.

        Arguments and return values are the same as for :meth:`disentangle`.
        """
        if self._U_param['type_evo'] == 'imag':
            return theta, None  # doesn't work for this...
        U_idx_dt, i = self._update_index
        U = self._U[U_idx_dt][i].conj()
        U.ireplace_labels(['p0*', 'p1*', 'p0', 'p1'], ['q0', 'q1', 'q0*', 'q1*'])
        theta = npc.tensordot(U, theta, axes=[['q0*', 'q1*'], ['q0', 'q1']])
        return theta, U

    def disentangle_renyi(self, theta):
        """Find optimal `U` which minimizes the second Renyi entropy.

        Reads of the following `TEBD_params` as break criteria for the iteration:

        ================ ====== ======================================================
        key              type   description
        ================ ====== ======================================================
        disent_eps       float  Break, if the change in the Renyi entropy ``S(n=2)``
                                per iteration is smaller than this value.
        ---------------- ------ ------------------------------------------------------
        disent_max_iter  float  Maximum number of iterations to perform.
        ================ ====== ======================================================

        Arguments and return values are the same as for :meth:`disentangle`.
        """
        max_iter = get_parameter(self.TEBD_params, 'disent_max_iter', 20, 'PurificationTEBD')
        eps = get_parameter(self.TEBD_params, 'disent_eps', 1.e-10, 'PurificationTEBD')
        U_idx_dt, i = self._update_index
        if U_idx_dt is not None:
            U = self._guess_U_disent[U_idx_dt][i]  # recover last result
        else:
            #  U = npc.outer(npc.eye_like(theta, 'q0').set_leg_labels(['q0', 'q0*']),
            #                npc.eye_like(theta, 'q1').set_leg_labels(['q1', 'q1*']))
            # TODO: is this a good idea?
            _, U = self.disentangle_diag(theta)
        Sold = np.inf
        S0 = None
        for j in xrange(max_iter):
            S, U = self.disentangle_renyi_iter(theta, U)
            if S0 is None:
                S0 = S
            if abs(Sold - S) < eps:
                break
            Sold, S = S, Sold
        theta = npc.tensordot(U, theta, axes=[['q0*', 'q1*'], ['q0', 'q1']])
        self._disent_iterations[i] += j  # save the number of iterations performed
        if self.verbose >= 10:
            print "disentangle renyi: {j:d} iterations, Sold-S = {DS:.3e}".format(j=j, DS=S0-Sold)
        if U_idx_dt is not None:
            self._guess_U_disent[U_idx_dt][i] = U  # save result as next guess
        return theta, U

    def disentangle_renyi_iter(self, theta, U):
        r"""Given `theta` and `U`, find another `U` which reduces the 2nd Renyi entropy.

        Temporarily view the different `U` as independt and mimizied one of them -
        this corresponds to a linearization of the cost function.
        Defining `Utheta` as the application of `U` to `theata`, and combining the `p` legs of
        `theta` with ``'vL', 'vR'``, this function contracts:

            |     .----theta----.
            |     |    |   |    |
            |     |    q0  q1   |
            |     |             |
            |     |        q1*  |
            |     |        |    |
            |     |  .-Utheta*-.
            |     |  | |
            |     |  .-Utheta--.
            |     |        |    |
            |     |    q0* |    |
            |     |    |   |    |
            |     .----Utheta*-.

        The trace yields the second Renyi entropy `S2`. Further, we calculate the unitary `U`
        with maximum overlap with this network.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Two-site wave function to be disentangled.
        U

        Returns
        -------
        S2 : float
            Renyi entopy (n=2), :math:`S2 = \frac{1}{1-2} \log tr(\rho_L^2)` of `U theta`.
        new_U : :class:`~tenpy.linalg.np_conserved.Array`
            Unitary with legs ``'q0', 'q1', 'q0*', 'q1*'``, which should disentangle `theta`.
        """
        U_theta = npc.tensordot(U, theta, axes=[['q0*', 'q1*'], ['q0', 'q1']])
        # same legs as theta: 'vL', 'p0', 'q0', 'p1', 'q1', 'vR'
        # contract diagram from bottom to top
        dS = npc.tensordot(U_theta, U_theta.conj(), axes=[['p1', 'q1', 'vR'],
                                                          ['p1*', 'q1*', 'vR*']])
        # dS has legs 'vL', 'p0', 'q0', 'vL*', 'p0*', 'q0*'
        dS = npc.tensordot(U_theta.conj(), dS, axes=[['vL*', 'p0*', 'q0*'], ['vL', 'p0', 'q0']])
        # dS has legs 'vL', 'p0', 'q0', 'vR', 'p1', 'q1'
        dS = npc.tensordot(theta, dS, axes=[['vL', 'p0', 'vR', 'p1'],
                                            ['vL*', 'p0*', 'vR*', 'p1*']])
        S2 = npc.inner(U, dS, axes=[['q0', 'q1', 'q0*', 'q1*'], ['q0*', 'q1*', 'q0', 'q1']])
        # dS has legs 'q0', 'q1', 'q0*', 'q1*'
        dS = dS.combine_legs([['q0', 'q1'], ['q0*', 'q1*']], qconj=[+1, -1])
        # Find unitary which maximizes `trace(U dS)`.
        W, Y, VH = npc.svd(dS)
        new_U = npc.tensordot(W, VH, axes=[1, 0]).conj()  # == V W^dagger.
        # this yields trace(U dS) = trace(Y), which is maximal.
        return -np.log(S2.real), new_U.split_legs([0, 1])

    def disentangle_renyi_dU(self, theta):
        """Find optimal `U` which minimizes the second Renyi entropy.

        Very similar to :meth:`disentangle_renyi`,
        but use :meth:`disentangle_renyi_dU_iter` for the iteration.

        Arguments and return values are the same as for :meth:`disentangle`.

        .. todo :
            This should give *exactly* (as far as the SVD is unique, i.e. up to phases)
            the same result as :meth:`disentangle_renyi`.
            Keep only one of them, namely the one which is faster....
        """
        max_iter = get_parameter(self.TEBD_params, 'disent_max_iter', 20, 'PurificationTEBD')
        eps = get_parameter(self.TEBD_params, 'disent_eps', 1.e-10, 'PurificationTEBD')
        U_idx_dt, i = self._update_index
        S0 = self._entropy_theta(theta, n=2)
        if U_idx_dt is not None:
            # during real time evolution
            U = self._guess_U_disent[U_idx_dt][i]  # recover last result
        else:
            #  U = npc.outer(npc.eye_like(theta, 'q0').set_leg_labels(['q0', 'q0*']),
            #                npc.eye_like(theta, 'q1').set_leg_labels(['q1', 'q1*']))
            # TODO XXX: do also for self.disentangle_reny()
            _, U = self.disentangle_diag(theta)
        theta = npc.tensordot(U, theta, axes=[['q0*', 'q1*'], ['q0', 'q1']])
        Sold = np.inf
        for j in xrange(max_iter):
            S, u = self.disentangle_renyi_dU_iter(theta)
            U = npc.tensordot(u, U, axes=[['q0*', 'q1*'], ['q0', 'q1']])
            theta = npc.tensordot(u, theta, axes=[['q0*', 'q1*'], ['q0', 'q1']])
            if abs(Sold - S) < eps:
                break
            Sold, S = S, Sold
        self._disent_iterations[i] += j  # save the number of iterations performed
        if self.verbose >= 10:
            print "disentangle renyi: {j:d} iterations, Sold-S = {DS:.3e}, S0-S={DS2:.3e}".format(
                j=j, DS=S-Sold, DS2=S0-Sold)
        if U_idx_dt is not None:
            self._guess_U_disent[U_idx_dt][i] = U  # save result as next guess
        return theta, U

    def disentangle_renyi_dU_iter(self, theta):
        r"""given theta and `U`, find another `U` which reduces the 2nd Renyi entropy.

        Combining the `p` legs of `theta` with ``'vL', 'vR'``, this function contracts:

            |     .----theta---.
            |     |    |   |   |
            |     |    q0  |   |
            |     |        |   |
            |     |  .-theta*--.
            |     |  | |
            |     |  .-theta---.
            |     |        |   |
            |     |        q1  |
            |     |            |
            |     |    q0* q1* |
            |     |    |   |   |
            |     .----theta*--.

        The trace yields the second Renyi entropy `S2`. Further, we calculate the unitary `U`
        with maximum overlap with this network.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Two-site wave function to be disentangled.

        Returns
        -------
        S2 : float
            Renyi entopy (n=2), :math:`S2 = \frac{1}{1-2} \log tr(\rho_L^2)` of `theta`.
        U : :class:`~tenpy.linalg.np_conserved.Array`
            Unitary (with legs ``'q0', 'q1', 'q0*', 'q1*'``, which should disentangle `theta`.)
        """
        dS = npc.tensordot(theta, theta.conj(), axes=[['p1', 'q1', 'vR'], ['p1*', 'q1*', 'vR*']])
        # dS has legs 'vL', 'p0', 'q0', 'vL*', 'p0*', 'q0*'
        dS = npc.tensordot(dS, theta, axes=[['vL*', 'p0*', 'q0*'], ['vL', 'p0', 'q0']])
        # dS has legs 'vL', 'p0', 'q0', 'vR', 'p1', 'q1'
        dS = npc.tensordot(dS, theta.conj(), axes=[['vL', 'p0', 'vR', 'p1'],
                                                   ['vL*', 'p0*', 'vR*', 'p1*']])
        # dS has legs 'q0', 'q1', 'q0*', 'q1*'
        dS = dS.combine_legs([['q0', 'q1'], ['q0*', 'q1*']], qconj=[+1, -1])
        S2 = npc.trace(dS)
        # Find unitary which approximates `dS` optimally.
        # This corresponds to a polar decomposition dS = U P with P >= 0
        W, Y, VH = npc.svd(dS)
        U = npc.tensordot(W, VH, axes=[1, 0])  # P = V Y VH  is actually not needed.
        # NOTE: no conj: we contracted the conjugate compared to disentangle_renyi_iter
        return -np.log(S2.real), U.split_legs([0, 1])

    def disentangle_diag(self, theta):
        """Disentangle by diagonalizing the two-site density matrix in the auxiliar space.

        .. todo :
            See Appendix B in arXiv:1704.01974.
            Problem: Sorting by eigenvalues breaks the charge conservation!
            Instead we just sort within the charge blocks and check, wether we increased
            total entanglement.

        Arguments and return values are the same as for :meth:`disentangle`.
        """
        rho = npc.tensordot(theta, theta.conj(), axes=(['vL', 'vR', 'p0', 'p1'],
                                                       ['vL*', 'vR*', 'p0*', 'p1*']))
        S0 = self._entropy_theta(theta)
        # TODO: eigh sorts only within the charge blocks...
        # TODO: the phase of the eigenvectors is arbitrary, but might increase the entanglement!!!
        #       how should it be chosen?
        E, V = npc.eigh(rho.combine_legs((['q0', 'q1'], ['q0*', 'q1*']), qconj=[+1, -1]))
        # the phase of the eigenvectors is not well defined. Thus, even if V is the identity,
        # we might actually increase the entanglement due to the random phases!
        # Try to get rid of them by choosing the phase of the maximal element.
        V_flat = V.to_ndarray()
        phases = V_flat[np.argmax(np.abs(V_flat), axis=0), np.arange(len(V_flat))]   # max values
        phases = phases / np.abs(phases)  # divided by absolute value
        V.iscale_axis(np.conj(phases), 'eig')
        V.ireplace_label('eig', '(q0*.q1*)')
        V = V.split_legs()
        # test that we actually reduce the *global* entanglement (not only I_ij)
        Vd = V.conj()
        theta1 = npc.tensordot(Vd, theta, axes=(['q0*', 'q1*'], ['q0', 'q1']))
        S1 = self._entropy_theta(theta1)
        if S1 < S0 - 1.e-14:
            return theta1, Vd
        else:
            return theta1, Vd # TODO: still return theta1 although we increase S TODO XXX
            assert(False) # TODO XXX
            # discard: we increased the entanglement
            return (theta, npc.outer(npc.eye_like(theta, 'q0').set_leg_labels(['q0', 'q0*']),
                                     npc.eye_like(theta, 'q1').set_leg_labels(['q1', 'q1*'])))

    def disentangle_global(self):
        """Try global disentangling by determining the maximally entangled pairs of sites.

        Caclulate the mutual information (in the auxiliar space) between two sites
        and determine where it is maximal. Disentangle these two sites with :meth:`disentangle`
        """
        max_range = get_parameter(self.TEBD_params, 'disent_gl_maxrange', 10, 'PurificationTEBD')
        coords, mutinf = self.psi.mutinf_two_site(max_range, legs='q')
        # TODO: recalculate mutinf only as necessary and do multiple steps at once...
        sorted = np.argsort(mutinf)
        for i, j in coords[sorted[-1:]]:
            if self.verbose > 10:
                print 'disentangle global pair ' + repr((i, j))
            self._disentangle_two_site(i, j)
        # done

    def _disentangle_two_site(self, i, j):
        """swap until i and j are next to each other and use :meth:`_disentangle_renyi`."""
        # TODO: should we also disentangle 'on the way' of swapping?
        on_way = get_parameter(self.TEBD_params, 'disent_gl_on_swap', False, 'PurificationTEBD')
        if not self.psi.finite:
            raise NotImplementedError  # adjust: what's the shortest path?
        assert(i < j)
        for j0 in range(j, i+1, -1):  # j0 = current site of `j`
            # originial leg `j` is at j0
            self._update_index = None, j0
            self._swap_disentangle_bond(j0, swap=True, disentangle=False)  # swap j0-1, j0
            # originial leg is at `j0-1`
        # disentangle i, i+1
        self._update_index = None, i+1
        self._swap_disentangle_bond(i+1, swap=False, disentangle=True)
        for j0 in range(i+1, j):  # j0 = current site of `j`
            # originial leg `j` is at j0
            self._update_index = None, j0+1
            self._swap_disentangle_bond(j0+1, disentangle=on_way)  # swap j0, j0+1
            # originial leg is at `j0+1`
        self._update_index = None  # done

    def _swap_disentangle_bond(self, i, swap=True, disentangle=False):
        """swap sites (i-1, i) (if swap = True) """
        # very similar to update_bond
        i0, i1 = i - 1, i
        if self.verbose >= 30:
            print "Update sites ({0:d}, {1:d}), swap={2!s}, disentangle={3!s}".format(
                i0, i1, swap, disentangle)
        # Construct the theta matrix
        theta = self.psi.get_theta(i0, n=2)  # 'vL', 'vR', 'p0', 'p1', 'q0', 'q1'
        if swap:
            theta.ireplace_labels(['p0', 'q0', 'p1', 'q1'], ['p1', 'q1', 'p0', 'q0'])
        if disentangle:
            theta, U_disent = self.disentangle(theta)
        theta = theta.combine_legs([('vL', 'p0', 'q0'), ('vR', 'p1', 'q1')], qconj=[+1, -1])

        # Perform the SVD and truncate the wavefunction
        U, S, V, trunc_err, renormalize = svd_theta(
            theta, self.trunc_params, inner_labels=['vR', 'vL'])

        # bring back to right-canonical 'B' form and update matrices
        B_R = V.split_legs(1).ireplace_labels(['p1', 'q1'], ['p', 'q'])
        C = self.psi.get_theta(i0, n=2, formL=0.)
        if swap:
            C.ireplace_labels(['p0', 'q0', 'p1', 'q1'], ['p1', 'q1', 'p0', 'q0'])
        if disentangle and U_disent is not None:
            C = npc.tensordot(U_disent, C, axes=[['q0*', 'q1*'], ['q0', 'q1']])
        B_L = npc.tensordot(
            C.combine_legs(('vR', 'p1', 'q1'), pipes=theta.legs[1]),
            V.conj(),
            axes=['(vR.p1.q1)', '(vR*.p1*.q1*)'])
        B_L.ireplace_labels(['vL*', 'p0', 'q0'], ['vR', 'p', 'q'])
        B_L /= renormalize  # re-normalize to <psi|psi> = 1
        self.psi.set_SR(i0, S)
        self.psi.set_B(i0, B_L, form='B')
        self.psi.set_B(i1, B_R, form='B')
        self._trunc_err_bonds[i] = self._trunc_err_bonds[i] + trunc_err
        return trunc_err

    def _entropy_theta(self, theta, n=1):
        """calculate entropy of theta via SVD."""
        theta = theta.combine_legs([['p0', 'q0', 'vL'], ['p1', 'q1', 'vR']], qconj=[+1, -1])
        _, S, _ = npc.svd(theta)
        return entropy(S**2, n)
