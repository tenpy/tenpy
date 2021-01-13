r"""Disentanglers.

The Disentanglers can be used to obtain a unitary reducing the entanglement between left and
right while only acting on a subspace of the left and right Hilbert space.

For now, this is written for disentangling purifications; could be generalized to allow more legs.

.. autodata:: disentanglers_atom_parse_dict
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np
from ..linalg import np_conserved as npc
from .truncation import svd_theta
from ..tools.math import entropy
from ..linalg import random_matrix as rand_mat

__all__ = [
    'Disentangler', 'BackwardDisentangler', 'RenyiDisentangler', 'NormDisentangler',
    'DiagonalizeDisentangler', 'GradientDescentDisentangler', 'NoiseDisentangler',
    'LastDisentangler', 'CompositeDisentangler', 'MinDisentangler',
    'disentanglers_atom_parse_dict', 'get_disentangler'
]


class Disentangler:
    r"""Prototype for a disentangler. Trivial, does nothing.

    In purification, we write :math:`\rho_P = Tr_Q{|\psi_{P,Q}><\psi_{P,Q}|}`. Thus, we
    can actually apply any unitary to the auxiliar `Q` space of :math:`|\psi>` without
    changing the physical expectation values.

    .. note ::
        We have to apply the *same* unitary to the 'bra' and 'ket' used for expectation values
        / correlation functions!

    However, the unitary can strongly influence the entanglement structure of :math:`|\psi>`.
    Therefore, the :class:`PurificationTEBD` includes a hook in
    :meth:`PurificationTEBD.update_bond` (and similar methods) to find and apply a disentangling
    unitary to the auxiliar indices of a two-site wave function by calling (``__call__`` method)
    a `Disentangler`.

    This class is a 'trivial' disentangler which does *nothing* to the two-site wave function;
    derived classes use different strategies to find various disentanglers.

    Parameters
    ----------
    parent : :class:`~tenpy.algorithms.purification.PurificationTEBD`
        The parent class calling the disentangler. Mostly used to read out extra options.

    Attributes
    ----------
    parent : :class:`~tenpy.algorithms.purification.PurificationTEBD`
        The parent class calling the disentangler.
    """
    def __init__(self, parent):
        self.parent = parent

    def __call__(self, theta):
        """Find and apply a unitary to disentangle `theta`.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Wave function to disentangle, with legs ``'vL', 'vR', 'p0', 'p1', 'q0', 'q1'``.

        Returns
        -------
        theta_disentangled : :class:`~tenpy.linalg.np_conserved.Array`
            Disentangled `theta`; ``npc.tensordot(U, theta, axes=[['q0*', 'q1*'], ['q0', 'q1']])``.
        U : :class:`~tenpy.linalg.conserved.Array` | None
            The unitary used to disentangle `theta`, with labels ``'q0', 'q1', 'q0*', 'q1*'``.
            If no unitary was found/applied, it might also be ``None``.
        """
        # do nothing
        return theta, None


class BackwardDisentangler(Disentangler):
    """Disentangle with backward time evolution.

    See [Karrasch2013]_ for details; only useful during real-time evolution.

    For the infinite temperature state, ``theta = delta_{p0, q0}*delta_{p1, q1}``.
    Thus, an application of `U_bond` to ``p0, p1`` can be reverted completely by applying
    ``U_bond^{dagger}`` to ``q0, q1``, resulting in the same state.
    This works also for finite temperatures, since `exp(-beta H)` and `exp(-i H t)` commute.
    Once we apply an operator to measure correlation function, the disentangling
    breaks down, yet for a local operator only in it's light-cone.

    Arguments and return values are the same as for :class:`Disentangler`.
    """
    def __init__(self, parent):
        self.parent = parent
        from . import purification
        if not isinstance(parent, purification.PurificationTEBD):
            raise ValueError("BackwardsDisentangler works only with PurificationTEBD")

    def __call__(self, theta):
        eng = self.parent
        if eng._U_param['type_evo'] == 'imag':
            return theta, None  # doesn't work for this...
        U_idx_dt, i = eng._update_index
        U = eng._U[U_idx_dt][i].conj()
        U.ireplace_labels(['p0*', 'p1*', 'p0', 'p1'], ['q0', 'q1', 'q0*', 'q1*'])
        theta = npc.tensordot(U, theta, axes=[['q0*', 'q1*'], ['q0', 'q1']])
        return theta, U


class RenyiDisentangler(Disentangler):
    """Iterative find `U` which minimized the second Renyi entropy.

    See [Hauschild2018]_

    Reads of the following `options` as break criteria for the iteration:

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
    def __init__(self, parent):
        self.max_iter = parent.options.get('disent_max_iter', 20)
        self.eps = parent.options.get('disent_eps', 1.e-10)
        self.parent = parent

    def __call__(self, theta):
        """Find optimal `U` which minimizes the second Renyi entropy."""
        U_idx_dt, i = self.parent._update_index
        U = npc.outer(npc.eye_like(theta, 'q0', labels=['q0', 'q0*']),
                      npc.eye_like(theta, 'q1', labels=['q1', 'q1*']))
        Sold = np.inf
        S0 = None
        for j in range(self.max_iter):
            S, U = self.iter(theta, U)
            if S0 is None:
                S0 = S
            if abs(Sold - S) < self.eps:
                break
            Sold, S = S, Sold
        theta = npc.tensordot(U, theta, axes=[['q0*', 'q1*'], ['q0', 'q1']])
        self.parent._disent_iterations[i] += j  # save the number of iterations performed
        if self.parent.verbose >= 10:
            print("disentangle renyi: {j:d} iterations, Sold-S = {DS:.3e}".format(j=j,
                                                                                  DS=S0 - Sold))
        return theta, U

    def iter(self, theta, U):
        r"""Given `theta` and `U`, find another `U` which reduces the 2nd Renyi entropy.

        Temporarily view the different `U` as independt and mimizied one of them -
        this corresponds to a linearization of the cost function.
        Defining `Utheta` as the application of `U` to `theata`, and combining the `p` legs of
        `theta` with ``'vL', 'vR'``, this function contracts::

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
        dS = npc.tensordot(U_theta,
                           U_theta.conj(),
                           axes=[['p1', 'q1', 'vR'], ['p1*', 'q1*', 'vR*']])
        # dS has legs 'vL', 'p0', 'q0', 'vL*', 'p0*', 'q0*'
        dS = npc.tensordot(U_theta.conj(), dS, axes=[['vL*', 'p0*', 'q0*'], ['vL', 'p0', 'q0']])
        # dS has legs 'vL', 'p0', 'q0', 'vR', 'p1', 'q1'
        dS = npc.tensordot(theta,
                           dS,
                           axes=[['vL', 'p0', 'vR', 'p1'], ['vL*', 'p0*', 'vR*', 'p1*']])
        S2 = npc.inner(U, dS, axes=[['q0', 'q1', 'q0*', 'q1*'], ['q0*', 'q1*', 'q0', 'q1']])
        # dS has legs 'q0', 'q1', 'q0*', 'q1*'
        dS = dS.combine_legs([['q0', 'q1'], ['q0*', 'q1*']], qconj=[+1, -1])
        # Find unitary which maximizes `trace(U dS)`.
        W, Y, VH = npc.svd(dS)
        new_U = npc.tensordot(W, VH, axes=[1, 0]).conj()  # == V W^dagger.
        # this yields trace(U dS) = trace(Y), which is maximal.
        return -np.log(S2.real), new_U.split_legs([0, 1])


class NormDisentangler(Disentangler):
    """Find optimal `U` for which the truncation of U|theta> has maximal overlap with U|theta>.

    Reads of the following `options` as break criteria for the iteration:

    ================ ========= ======================================================
    key              type      description
    ================ ========= ======================================================
    disent_eps       float     Break, if the change in the Renyi entropy ``S(n=2)``
                               per iteration is smaller than this value.
    ---------------- --------- ------------------------------------------------------
    disent_max_iter  float     Maximum number of iterations to perform.
    ---------------- --------- ------------------------------------------------------
    disent_trunc_par dict      Truncation parameters; defaults to `trunc_params`.
    ---------------- --------- ------------------------------------------------------
    disent_norm_chi  iterable  To find the optimal U it can help to increase `chi_max`
                               of `disent_trunc_par` slowly, the default is
                               ``range(1, disent_trunc_par['chi_max']+1)``.
                               However, that's **very** slow for large `chi_max`,
                               so we allow to change it. (In fact, it makes the
                               disentangler *scale* worse than the rest of TEBD.)
    ================ ========= ======================================================

    Arguments and return values are the same as for :meth:`disentangle`.
    """
    def __init__(self, parent):
        self.max_iter = parent.options.get('disent_max_iter', 20)
        self.eps = parent.options.get('disent_eps', 1.e-10)
        self.trunc_par = parent.options.subconfig('disent_trunc_par', parent.trunc_params)
        self.chi_max = self.trunc_par.get('chi_max', 100)
        self.trunc_cut = self.trunc_par.get('trunc_cut', None)
        self.chi_range = self.trunc_par.get('disent_norm_chi', range(1, self.chi_max + 1))
        self.parent = parent

    def __call__(self, theta):
        _, i = self.parent._update_index
        U = npc.outer(npc.eye_like(theta, 'q0', labels=['q0', 'q0*']),
                      npc.eye_like(theta, 'q1', labels=['q1', 'q1*']))
        err = None
        trunc_par = self.trunc_par.copy()
        for chi_opt in self.chi_range:
            trunc_par['chi_max'] = chi_opt
            for j in range(self.max_iter):
                err2, U = self.iter(theta, U, trunc_par)
                if err is not None and abs(err.eps - err2.eps) <= err.eps * self.eps:
                    break
                err = err2
            if self.trunc_cut is not None:
                if err2.eps < self.trunc_cut * self.trunc_cut:
                    break
        theta = npc.tensordot(U, theta, axes=[['q0*', 'q1*'], ['q0', 'q1']])
        self.parent._disent_iterations[i] += j  # save the number of iterations performed
        if self.parent.verbose >= 10:
            print("disentangle norm: {j:d} iterations, err={err!s}".format(j=j, err=err))
        return theta, U

    def iter(self, theta, U, trunc_params):
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
        lambda_ = U_theta.combine_legs([['vL', 'p0', 'q0'], ['vR', 'p1', 'q1']], qconj=[+1, -1])
        X, Y, Z, err, _ = svd_theta(lambda_, trunc_params)
        lambda_ = npc.tensordot(X.scale_axis(Y), Z, axes=1).split_legs()
        dS = npc.tensordot(theta,
                           lambda_.conj(),
                           axes=[['vL', 'vR', 'p0', 'p1'], ['vL*', 'vR*', 'p0*', 'p1*']])
        # dS has legs 'q0', 'q1', 'q0*', 'q1*'
        dS = dS.combine_legs([['q0', 'q1'], ['q0*', 'q1*']], qconj=[+1, -1])
        # Find unitary U2 which maximizes `trace(U dS)`.
        W, Y, VH = npc.svd(dS)
        new_U = npc.tensordot(W, VH, axes=[1, 0]).conj()  # == V W^dagger.
        # this yields trace(U dS) = trace(Y), which is maximal.
        return err, new_U.split_legs([0, 1])


class GradientDescentDisentangler(Disentangler):
    """Gradient-descent optimization, similar to :class:`RenyiDisentangler`.

    Reads of the following `TEBD_params`:

    ================ ====== ======================================================
    key              type   description
    ================ ====== ======================================================
    disent_eps       float  Break, if the change in the Renyi entropy ``S(n=2)``
                            per iteration is smaller than this value.
    ---------------- ------ ------------------------------------------------------
    disent_max_iter  float  Maximum number of iterations to perform.
    ---------------- ------ ------------------------------------------------------
    disent_n         float  Renyi index of the entropy to be used.
                            ``n=1`` for von-Neumann entropy.
    ================ ====== ======================================================

    Arguments and return values are the same as for :class:`Disentangler`.
    """
    def __init__(self, parent):
        self.max_iter = parent.options.get('disent_max_iter', 20)
        self.eps = parent.options.get('disent_eps', 1.e-10)
        self.n = parent.options.get('disent_n', 1.)
        self.stepsizes = parent.options.get('disent_stepsizes', [0.2, 1., 2.])
        self.parent = parent

    def __call__(self, theta):
        U_idx_dt, i = self.parent._update_index
        Utot = None
        Sold = np.inf
        S0 = None
        for j in range(self.max_iter):
            S, theta, U = self.iter(theta)
            if Utot is None:
                Utot = U
            else:
                Utot = npc.tensordot(U, Utot, axes=[['q0*', 'q1*'], ['q0', 'q1']])
            if S0 is None:
                S0 = S
            if abs(Sold - S) < self.eps:
                break
            Sold, S = S, Sold
        theta = npc.tensordot(U, theta, axes=[['q0*', 'q1*'], ['q0', 'q1']])
        self.parent._disent_iterations[i] += j  # save the number of iterations performed
        if self.parent.verbose >= 10:
            print("disentangle renyi: {j:d} iterations, Sold-S = {DS:.3e}".format(j=j,
                                                                                  DS=S0 - Sold))
        return theta, U

    def iter(self, theta):
        r"""Given `theta`, find a unitary `U` towards minimizing the n-th Renyi entropy.

        This function calulates the gradiant :math:`dS = \partial S(U theta, n) /\partial U`.
        and then ``U(t) = exp(-t*dS)``, where we choose the `t` from stepsizes which
        minimizes the entropy of ``U(t) theta``.

        When ``R[i]`` is the derivative :math:`\partial S(Y, n)/ \partial Y_i` of the (n-th Renyi)
        entropy, ``dS`` is given by::

            |     .----X--R--Z----.
            |     |    |     |    |
            |     |    q0    q1   |
            |     |               |
            |     |    q0*   q1*  |
            |     |    |     |    |
            |     .----X*-Y--Z*---.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Two-site wave function to be disentangled

        Returns
        -------
        S : float
            n-th Renyi entopy of new_theta
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            The *disentangled* wave function ``new_U theta``.
        new_U : :class:`~tenpy.linalg.np_conserved.Array`
            Unitary with legs ``'q0', 'q1', 'q0*', 'q1*'``, which was used to disentangle `theta`.
        """
        theta2 = theta.combine_legs([('vL', 'p0', 'q0'), ('vR', 'p1', 'q1')], qconj=[+1, -1])
        X, Y, Z = npc.svd(theta2, inner_labels=['vR', 'vL'])
        n = self.n
        if n == 1:
            r = Y * np.log(Y) * 2
            r[Y < 1.e-14] = 0.
            #  S = -np.inner(Y**2, np.log(Y**2))
        else:
            Y[Y < 1.e-20] = 1.e-20
            tr_pn = np.sum(Y**(2 * n))
            ss = Y**(2 * (n - 1))
            r = Y * ss * (n / (n - 1.) / tr_pn)  # TODO: why?
            #  r = Y*ss *(1 - n.)  # TODO: why not?
            #  S = np.log(tr_pn)/(1 - n)
        XrZ = npc.tensordot(X.scale_axis(r, 'vR'), Z, axes=['vR', 'vL']).split_legs()
        dS = npc.tensordot(theta,
                           XrZ.conj(),
                           axes=[['vL', 'p0', 'p1', 'vR'], ['vL*', 'p0*', 'p1*', 'vR*']])
        dS = dS.combine_legs([['q0', 'q1'], ['q0*', 'q1*']], qconj=[1, -1])
        dS = dS - dS.conj().transpose(['(q0.q1)', '(q0*.q1*)'])  # project: anti-hermitian part
        new_Ss = []
        new_thetas = []
        new_Us = []
        for t in self.stepsizes:
            U = npc.expm((-t) * dS).split_legs()  # dS anti-hermitian => exp(-tdS) unitary
            new_theta = npc.tensordot(U, theta, axes=[['q0*', 'q1*'], ['q0', 'q1']])
            new_Ss.append(self._entropy_theta(new_theta, n))
            new_thetas.append(new_theta)
            new_Us.append(U)
        a = np.argmin(new_Ss)
        return new_Ss[a], new_thetas[a], new_Us[a]

    def _entropy_theta(self, theta):
        """Calculate entropy of theta via SVD."""
        theta = theta.combine_legs([('vL', 'p0', 'q0'), ('vR', 'p1', 'q1')], qconj=[+1, -1])
        _, S, _ = npc.svd(theta)
        return entropy(S**2, self.n)


class NoiseDisentangler(Disentangler):
    """Apply a little bit of random noise. Useful as pre-step to :class:`RenyiDisentangler`.

    Arguments and return values are the same as for :class:`Disentangler`.
    """
    def __init__(self, parent):
        self.a = parent.options.get('disent_noiselevel', 0.01)

    def __call__(self, theta):
        a = self.a
        leg = theta.make_pipe(['q0', 'q1'])
        if a is None:
            U = npc.Array.from_func_square(rand_mat.CUE, leg).split_legs()
        else:
            U = npc.Array.from_func_square(rand_mat.U_close_1, leg, func_args=[a]).split_legs()
        U.iset_leg_labels(['q0', 'q1', 'q0*', 'q1*'])
        theta = npc.tensordot(U, theta, axes=[['q0*', 'q1*'], ['q0', 'q1']])
        return theta, U


class LastDisentangler(Disentangler):
    """Use the last total 'U' used in :meth:`disentangle` for the same _update_index as guess.

    Useful as a starting point in a :class:`CompositeDisentangler` to reduce the number of
    iterations for a following disentangler.
    """
    def __call__(self, theta):
        # result was saved in :meth:`PurificationTEBD.disentangle`
        U = None
        U_idx_dt, i = self.parent._update_index
        if U_idx_dt is not None:
            U = self.parent._guess_U_disent[U_idx_dt][i]
        if U is not None:
            theta = npc.tensordot(U, theta, axes=[['q0*', 'q1*'], ['q0', 'q1']])
        return theta, U


class DiagonalizeDisentangler(Disentangler):
    """Disentangle by diagonalizing the two-site density matrix in the auxiliar space.

    See :arxiv:`1704.01974`.
    Problem: Sorting by eigenvalues breaks the charge conservation!
    Instead we just sort within the charge blocks.
    For non-trivial charges, this might increase the entropy!

    Arguments and return values are the same as for :class:`Disentangler`.
    """
    def __call__(self, theta):
        rho = npc.tensordot(theta,
                            theta.conj(),
                            axes=(['vL', 'vR', 'p0', 'p1'], ['vL*', 'vR*', 'p0*', 'p1*']))
        # eigh sorts only within the charge blocks...
        E, V = npc.eigh(rho.combine_legs((['q0', 'q1'], ['q0*', 'q1*']), qconj=[+1, -1]))
        # the phase of the eigenvectors is not well defined. Thus, even if V is the identity,
        # we might actually increase the entanglement due to the random phases!
        # Try to get rid of them by choosing the phase of the maximal element.
        V_flat = V.to_ndarray()
        phases = V_flat[np.argmax(np.abs(V_flat), axis=0), np.arange(len(V_flat))]  # max values
        phases = phases / np.abs(phases)  # divided by absolute value
        V.iscale_axis(np.conj(phases), 'eig')
        V.ireplace_label('eig', '(q0*.q1*)')
        V = V.split_legs()
        Vd = V.conj()
        theta1 = npc.tensordot(Vd, theta, axes=(['q0*', 'q1*'], ['q0', 'q1']))
        return theta1, Vd


class CompositeDisentangler(Disentangler):
    """Concatenate multiple disentanglers.

    Applies multiple disentanglers, one after another (in iteration order).

    Parameters
    ----------
    disentanglers : list of :class:`Disentangler`
        The disentanglers to be used.

    Attributes
    ----------
    disentanglers : list of :class:`Disentangler`
        The disentanglers to be used.
    """
    def __init__(self, disentanglers):
        self.disentanglers = disentanglers

    def __call__(self, theta):
        Utot = None
        for disent in self.disentanglers:
            theta, U = disent(theta)
            if Utot is None:
                Utot = U
            elif U is not None:  # neither Utot nor U are None: multiply together
                Utot = npc.tensordot(U, Utot, axes=[['q0*', 'q1*'], ['q0', 'q1']])
        return theta, Utot


class MinDisentangler(Disentangler):
    """Chose the disentangler giving the smallest entropy.

    Apply each of the disentanglers to the given `theta`, use the result with smallest entropy.
    Reads the TEBD_param ``'disent_min_n'`` which selects the :func:`~tenpy.tools.math.entropy`
    to be used for comparison.

    Parameters
    ----------
    disentanglers : list of :class:`Disentangler`
        The disentanglers to be used.
    parent : :class:`~tenpy.algorithms.purification.PurificationTEBD`
        The parent class calling the disentangler.

    Attributes
    ----------
    n : float
        Selects the entropy to be used for comparison.
    disentanglers : list of :class:`Disentangler`
        The disentanglers to be used.
    """
    def __init__(self, disentanglers, parent):
        self.disentanglers = disentanglers
        self.n = parent.options.get('disent_min_n', 1.)

    def __call__(self, theta):
        theta_min, U_min = self.disentanglers[0](theta)
        S_min = self._entropy_theta(theta_min)
        for disent in self.disentanglers[1:]:
            theta2, U2 = disent(theta)
            S2 = self._entropy_theta(theta2)
            if S2 < S_min:
                S_min = S2
                theta_min = theta2
                U_min = U2
        return theta_min, U_min

    def _entropy_theta(self, theta):
        """Calculate entropy of theta via SVD."""
        theta = theta.combine_legs([('vL', 'p0', 'q0'), ('vR', 'p1', 'q1')], qconj=[+1, -1])
        _, S, _ = npc.svd(theta)
        return entropy(S**2, self.n)


disentanglers_atom_parse_dict = {
    'None': Disentangler,
    'backwards': BackwardDisentangler,
    'renyi': RenyiDisentangler,
    'norm': NormDisentangler,
    'graddesc': GradientDescentDisentangler,
    'noise': NoiseDisentangler,
    'last': LastDisentangler,
    'diag': DiagonalizeDisentangler
}
"""Dictionary to translate the 'disentangle' TEBD parameter into a :class:`Disentangler`.

If you define your own disentanglers, you can dynamically append them to this dictionary.
CompositeDisentangler and MinDisentangler separate: they have non-default constructor and
special syntax.
"""


def get_disentangler(method, parent):
    """Parse the parameter `method` and construct a :class:`Disentangler` instance.

    Parameters
    ----------
    method : str | ``None``
        The method to be used, of the form 'method1-method2-min(method3,method4-method5)'.
        The usage should be clear from the examples, the precise rule follows:
        We parse the full `method` string as a `composite`, and define
        ``composite := min_atom ['-' min_atom ...]``,
        ``min_atom := { 'min(' composite [',' composite ...] ')' } | atom``, and
        ``atom := {any key of `disentanglers_atom_parse_dict`}``.
    parent : :class:`~tenpy.algorithms.purification.PurificationTEBD`
        The parent class calling the disentangler.

    Returns
    -------
    disentangler : :class:`Disentangler`
        Disentangler instance, which can be called to disentangle a 2-site `theta`
        with the specified `method`.

    Examples
    --------
    >>> get_disentangler(None, p)
    Disentangler(p)
    >>> get_disentangler('last-renyi', p)
    Disentangler([LastDisentangler(p), RenyiDisentangler(p)], p)
    >>> get_disentangler('min(None,noise-renyi,min(backwards,last)-graddesc)')
    MinDisentangler([Disentangler,
                     CompositeDisentangler([NoiseDisentangler(p), RenyiDisentangler(p)], p),
                     CompositeDisentangler([MinDisentangler([BackwardDisentangler(p),
                                                             LastDisentangler(p)]),
                                            GradientDescentDisentangler(p)], p), p)
    """
    try:
        disent, unparsed = _parse_composite(str(method), parent)
        if len(unparsed) > 0:
            raise _ParseError
    except _ParseError:
        raise
        #  raise ValueError("Error while parsing disentangle method: " + repr(method))
    return disent


def _parse_composite(unparsed, parent):
    disentanglers = []
    while True:
        disent, unparsed = _parse_min_atom(unparsed, parent)
        disentanglers.append(disent)
        if len(unparsed) == 0 or unparsed[0] != '-':
            break  # end of composite
        # else: unparsed[0] == '-'
        unparsed = unparsed[1:]
        # -> continue with while loop
    if len(disentanglers) == 1:
        # just a min_atom
        return disentanglers[0], unparsed
    return CompositeDisentangler(disentanglers), unparsed


def _parse_min_atom(unparsed, parent):
    if unparsed.startswith('min('):
        disentanglers = []
        unparsed = unparsed[4:]
        while True:
            disent, unparsed = _parse_composite(unparsed, parent)
            disentanglers.append(disent)
            if len(unparsed) == 0 or unparsed[0] != ',':
                break  # parsed the expected part
            # else: unparsed[0] == ','
            unparsed = unparsed[1:]
            # -> continue with while loop
        if len(unparsed) == 0 or unparsed[0] != ')':
            raise _ParseError
        # else: unparsed[0] == ')'
        return MinDisentangler(disentanglers, parent), unparsed[1:]
    else:  # expect atom
        return _parse_atom(unparsed, parent)


def _parse_atom(unparsed, parent):
    for key, disent in disentanglers_atom_parse_dict.items():
        if unparsed.startswith(key):
            return disent(parent), unparsed[len(key):]
    raise _ParseError


class _ParseError(ValueError):
    pass
