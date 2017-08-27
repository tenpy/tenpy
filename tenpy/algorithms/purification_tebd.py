r"""Time evolving block decimation (TEBD) for MPS of purification.

See introduction in :mod:`~tenpy.networks.purification_mps`.
Time evolution for finite-temperature ensembles.
This can be used to obtain correlation functions in time.
"""

from __future__ import division

from . import tebd
from ..linalg import np_conserved as npc
from .truncation import svd_theta, TruncationError
from ..tools.params import get_parameter
from ..tools.math import entropy
from ..linalg import random_matrix as rand_mat

import numpy as np
import re  # regex


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
    _MATCH_DISENTANGLE : compiled regex
        Regex to match the TEBD parameter 'disentangle', which specifies the method used for
        disentangling.
    """
    _MATCH_DISENTANGLE = re.compile(r"min2?\([^)]+\)|[^-]+")

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
        self._guess_U_disent = [[None]*len(Us) for Us in self._U]

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
        By default (``None`` or ``'None'``) it does nothing.
        Valid strings are of the form ``'method1-method2-min(method3-method4,method5)'``.
        Here, each of the `method#` stands for a call of the coresponding `disentangle_{method}`,
        Methods separated by ``-`` are successively applied as reading from left to right.
        `min2` is an alternative to `min`, comparing the second Renyi entropy instead of the
        von-Neumann entropy.

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
        disentangle = get_parameter(self.TEBD_params, 'disentangle', None, 'PurificationTEBD')
        if disentangle is None:
            return theta, None
        theta, U = self._parse_disentangle(disentangle, theta)
        U_idx_dt, i = self._update_index
        if U_idx_dt is not None:
            self._guess_U_disent[U_idx_dt][i] = U  # save result as next guess
        return theta, U

    def _parse_disentangle(self, disentangle, theta):
        """Parse the parameter `disentangle` to disentangle `theta` with the chosen methods.

        """
        methods = self._MATCH_DISENTANGLE.findall(disentangle)
        Utot = None
        for method in methods:
            if method == 'None':
                U = None  # do nothing
            elif method == 'backwards':
                theta, U = self.disentangle_backwards(theta)
            elif method == 'renyi':
                theta, U = self.disentangle_renyi(theta)
            elif method == 'norm':
                theta, U = self.disentangle_norm(theta)
            elif method == 'diag':
                theta, U = self.disentangle_diag(theta)
            elif method == 'noise':
                theta, U = self.disentangle_noise(theta)
            elif method == 'last':
                theta, U = self.disentangle_last(theta)
            elif method.startswith('min('):
                theta, U = self.disentangle_min(method[4:-1], theta)
            elif method.startswith('min2('):
                theta, U = self.disentangle_min(method[5:-1], theta, n=2)
            else:
                raise ValueError("Invalid choice in 'disentangle': " + repr(disentangle))
            if Utot is None:
                Utot = U
            elif U is not None:  # neither Utot nor U are None: multiply together
                Utot = npc.tensordot(U, Utot, axes=[['q0*', 'q1*'], ['q0', 'q1']])
        return theta, Utot

    def disentangle_min(self, methods, theta, n=1):
        """Try different methods for disentangling and take the theta with minimal final entropy.

        Parameters
        ----------
        methods: string
            Methods to try, separated by ``,``.
            As in :meth:`disentangle`, methods separated by ``-`` are applied successively.
            Thus, ``method3-method4,method4`` compares the two possibilities of
            1) apply `method3`, then `method4` or 2) apply `method5`.
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Wave function to disentangle, with legs ``'vL', 'vR', 'p0', 'p1', 'q0', 'q1'``.
        n : int
            Which entropy to choose for selecting the 'minimum'.

        Returns
        -------
        theta_disentangled : :class:`~tenpy.linalg.np_conserved.Array`
            Disentangled `theta`; ``npc.tensordot(U, theta, axes=[['q0*', 'q1*'], ['q0', 'q1']])``.
        U : :class:`~tenpy.linalg.conserved.Array`
            The unitary used to disentangle `theta`, with labels ``'q0', 'q1', 'q0*', 'q1*'``.
            If no unitary was found/applied, it might also be ``None``.
        """
        methods = methods.split(',')
        theta_min, U_min = self._parse_disentangle(methods[0], theta)
        S_min = self._entropy_theta(theta_min, n)
        for method in methods[1:]:
            theta2, U2 = self._parse_disentangle(method, theta)
            S2 = self._entropy_theta(theta2, n)
            if S2 < S_min:
                S_min = S2
                theta_min = theta2
                U_min = U2
        return theta_min, U_min

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
        U = npc.outer(npc.eye_like(theta, 'q0').set_leg_labels(['q0', 'q0*']),
                      npc.eye_like(theta, 'q1').set_leg_labels(['q1', 'q1*']))
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
        U : :class:`~tenpy.linalg.np_conserved.Array`
            The previous guess for `U`; with legs ``'q0', 'q1', 'q0*', 'q1*'``.

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

    def disentangle_norm(self, theta):
        """Find optimal `U` for which the truncation of U|theta> has maximal overlap with U|theta>.

        Reads of the following `TEBD_params` as break criteria for the iteration:

        ================ ====== ======================================================
        key              type   description
        ================ ====== ======================================================
        disent_eps       float  Break, if the change in the Renyi entropy ``S(n=2)``
                                per iteration is smaller than this value.
        ---------------- ------ ------------------------------------------------------
        disent_max_iter  float  Maximum number of iterations to perform.
        ---------------- ------ ------------------------------------------------------
        disent_trunc_par dict   Truncation parameters; defaults to `trunc_params`.
        ================ ====== ======================================================

        Arguments and return values are the same as for :meth:`disentangle`.
        """
        max_iter = get_parameter(self.TEBD_params, 'disent_max_iter', 20, 'PurificationTEBD')
        eps = get_parameter(self.TEBD_params, 'disent_eps', 1.e-10, 'PurificationTEBD')
        trunc_par = get_parameter(self.TEBD_params, 'disent_trunc_par',
                                  self.trunc_params, 'PurificationTEBD')
        _, i = self._update_index
        U = npc.outer(npc.eye_like(theta, 'q0').set_leg_labels(['q0', 'q0*']),
                      npc.eye_like(theta, 'q1').set_leg_labels(['q1', 'q1*']))
        #  trunc_par['chi_max'] = max([int(len(self.psi._S[i+1])*0.9), 1]) # TODO XXX HACK
        err = None
        for j in xrange(max_iter):
            err2, U = self.disentangle_norm_iter(theta, U, trunc_par)
            if err is not None and abs(err.eps - err2.eps) <= err.eps * eps:
                break
            err = err2
        theta = npc.tensordot(U, theta, axes=[['q0*', 'q1*'], ['q0', 'q1']])
        self._disent_iterations[i] += j  # save the number of iterations performed
        if self.verbose >= 10:
            print "disentangle norm: {j:d} iterations, err={err!s}".format(j=j, err=err)
        return theta, U

    def disentangle_norm_iter(self, theta, U, trunc_params):
        r"""Given `theta` and `U`, find `U2` maximizing ``<theta|U2 truncate(U |theta>)``.

        Finds unitary `U2` which maximizes Tr(U

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Two-site wave function to be disentangled.
        U : :class:`~tenpy.linalg.np_conserved.Array`
            The previous guess for `U`; with legs ``'q0', 'q1', 'q0*', 'q1*'``.
        trunc_params : dict
            The truncation parameters (similar as `self.trunc_params`) used to truncate `U|theta>`.

        Returns
        -------
        trunc_err : TruncationError
            Norm error discarded during the truncation of ``U|theta>``.
        new_U : :class:`~tenpy.linalg.np_conserved.Array`
            Unitary with legs ``'q0', 'q1', 'q0*', 'q1*'``.
            Chosen such that ``new_U|theta>`` has maximal overlap with the truncated ``U|theta>``.
        """
        U_theta = npc.tensordot(U, theta, axes=[['q0*', 'q1*'], ['q0', 'q1']])
        lambda_ = U_theta.combine_legs([['vL', 'p0', 'q0'], ['vR', 'p1', 'q1']])
        X, Y, Z, err, _ = svd_theta(lambda_, trunc_params)
        lambda_ = npc.tensordot(X.scale_axis(Y), Z, axes=1).split_legs()
        dS = npc.tensordot(theta, lambda_.conj(), axes=[['vL', 'vR', 'p0', 'p1'],
                                                        ['vL*', 'vR*', 'p0*', 'p1*']])
        # dS has legs 'q0', 'q1', 'q0*', 'q1*'
        dS = dS.combine_legs([['q0', 'q1'], ['q0*', 'q1*']], qconj=[+1, -1])
        # Find unitary U2 which maximizes `trace(U dS)`.
        W, Y, VH = npc.svd(dS)
        new_U = npc.tensordot(W, VH, axes=[1, 0]).conj()  # == V W^dagger.
        # this yields trace(U dS) = trace(Y), which is maximal.
        return err, new_U.split_legs([0, 1])

    def disentangle_last(self, theta):
        "Use the last total 'U' used in :meth:`disentangle` for the same _update_index as guess."
        U = None
        U_idx_dt, i = self._update_index
        if U_idx_dt is not None:
            U = self._guess_U_disent[U_idx_dt][i]  # result saved in :meth:`disentangle`
        if U is not None:
            theta = npc.tensordot(U, theta, axes=[['q0*', 'q1*'], ['q0', 'q1']])
        return theta, U

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
        return theta1, Vd

    def disentangle_noise(self, theta):
        """Apply a little bit of random noise. Useful as pre-step to disentangle_renyi.

        Arguments and return values are the same as for :meth:`disentangle`.
        """
        a = get_parameter(self.TEBD_params, 'disent_noiselevel', 0.01, 'PurificationTEBD')
        leg = theta.make_pipe(['q0', 'q1'])
        if a is None:
            U = npc.Array.from_func_square(rand_mat.CUE, leg).split_legs()
        else:
            U = npc.Array.from_func_square(rand_mat.U_close_1, leg, func_args=[a]).split_legs()
        U.set_leg_labels(['q0', 'q1', 'q0*', 'q1*'])
        theta = npc.tensordot(U, theta, axes=[['q0*', 'q1*'], ['q0', 'q1']])
        return theta, U

    def disentangle_global(self, pair=None):
        """Try global disentangling by determining the maximally entangled pairs of sites.

        Caclulate the mutual information (in the auxiliar space) between two sites
        and determine where it is maximal. Disentangle these two sites with :meth:`disentangle`
        """
        max_range = get_parameter(self.TEBD_params, 'disent_gl_maxrange', 10, 'PurificationTEBD')
        if pair is None:
            coords, mutinf = self.psi.mutinf_two_site(max_range, legs='q')
            # TODO: recalculate mutinf only as necessary and do multiple steps at once...
            sorted = np.argsort(mutinf)
            pair = coords[sorted[-1]]
        i, j = pair
        #  for i, j in coords[sorted[-1:]]:
        if self.verbose > 10:
            print 'disentangle global pair ' + repr((i, j))
        self._disentangle_two_site(i, j)
        return i, j   # TODO
        # done

    def disentangle_global_nsite(self, n=2):
        """Perform a sweep through the system and disentangle with :meth:`disentangle_n_site`.

        Parameters
        ----------
        n: int
            maximal number of sites to disentangle at once.
        """
        for i in range(0, self.psi.L-n+1):  # sweep left to right
            self._update_index = None, i
            theta = self.psi.get_theta(i, n=n)
            self.disentangle_n_site(i, n, theta)  # works recursively
        for i in range(self.psi.L-n, -1, -1):  # sweep right to left
            self._update_index = None, i
            theta = self.psi.get_theta(i, n=n)
            self.disentangle_n_site(i, n, theta)  # works recursively
        self._update_index = None

    def disentangle_n_site(self, i, n, theta):
        r"""Generalization of :meth:`disentangle` to `n` sites.

        Simply group left and right `n`/2 physical legs, adjust labels, and
        apply :meth:`disentangle_renyi` to disentangle the central bond.
        Recursively proceed to disentangle left and right parts afterwards.
        Scales (for even `n`) as :math:`O(\chi^3 d^n d^{n/2})`.
        """
        assert(n >= 2)
        n1 = n // 2
        n2 = n - n1
        p = ['p'+str(j) for j in range(n)]  # labels of theta to be separated
        q = ['q'+str(j) for j in range(n)]
        pL, pR = p[:n1], p[n1:]
        qL, qR = q[:n1], q[n1:]
        theta = theta.combine_legs([pL, qL, pR, qR], qconj=[+1, -1, +1, -1], new_axes=[1, 2, 3, 4])
        _, p0, q0, p1, q1, _ = theta.get_leg_labels()  # keep the labels for later
        theta.ireplace_labels([p0, q0, p1, q1], ['p0', 'q0', 'p1', 'q1'])
        theta, _ = self.disentangle(theta)  # apply two-site disentangling
        theta = theta.combine_legs([('vL', 'p0', 'q0'), ('vR', 'p1', 'q1')], qconj=[+1, -1])

        # Perform the SVD and truncate the wavefunction
        U, S, V, trunc_err, renormalize = svd_theta(
            theta, self.trunc_params, inner_labels=['vR', 'vL'])
        self.psi.set_SL(i+n1, S)  # update S
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
            self.psi.set_B(i+n1, V, form='B')
        else:
            # disentangle right n2-site wave function recursively
            theta_R = V.iscale_axis(S, 0).split_legs(1)
            theta_R = theta_R.ireplace_labels(['p1', 'q1'], [p1, q1]).split_legs([p1, q1])
            theta_R.ireplace_labels(pR, p[:n2]).ireplace_labels(qR, q[:n2])
            self.disentangle_n_site(i+n1, n2, theta_R)

    def _disentangle_two_site(self, i, j):
        """swap until i and j are next to each other and use :meth:`_disentangle_renyi`."""
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


class PurificationTEBD2(PurificationTEBD):
    """similar as PurificationTEBD, but perform sweeps instead of brickwall
    for real and imaginary time evolution"""

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
        assert(order == 2 and self.psi.finite)
        # sweep right orde

        for i in range(N_steps):
            trunc_err += self.update_step(0, False)
            trunc_err += self.update_step(0, True)
        self.evolved_time = self.evolved_time + N_steps * self._U_param['tau']
        self.trunc_err = self.trunc_err + trunc_err  # not += : make a copy!
        # (this is done to avoid problems of users storing self.trunc_err after each `update`)
        return trunc_err

    def update_step(self, U_idx_dt, odd):
        """Updates either even *or* odd bonds in unit cell.

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
            sweep = range(self.psi.L-1, 0, -1)
        for i_bond in sweep:
            if Us[i_bond] is None:
                if self.verbose >= 10:
                    print "Skip U_bond element:", i_bond
                continue  # handles finite vs. infinite boundary conditions
            if self.verbose >= 10:
                print "Apply U_bond element", i_bond
            self._update_index = (U_idx_dt, i_bond)
            trunc_err += self.update_bond(i_bond, Us[i_bond])
        self._update_index = None
        return trunc_err
