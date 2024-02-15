r"""Time evolving block decimation (TEBD).

The TEBD algorithm (proposed in :cite:`vidal2004`) uses a trotter decomposition of the
Hamiltonian to perform a time evolution of an MPS. It works only for nearest-neighbor hamiltonians
(in tenpy given by a :class:`~tenpy.models.model.NearestNeighborModel`),
which can be written as :math:`H = H^{even} + H^{odd}`,  such that :math:`H^{even}` contains the
the terms on even bonds (and similar :math:`H^{odd}` the terms on odd bonds).
In the simplest case, we apply first :math:`U=\exp(-i*dt*H^{even})`,
then :math:`U=\exp(-i*dt*H^{odd})` for each time step :math:`dt`.
This is correct up to errors of :math:`O(dt^2)`, but to evolve until a time :math:`T`, we need
:math:`T/dt` steps, so in total it is only correct up to error of :math:`O(T*dt)`.
Similarly, there are higher order schemata (in dt) (for more details see
:meth:`TEBDEngine.update`).

Remember, that bond `i` is between sites `(i-1, i)`, so for a finite MPS it looks like::

    |     - B0 - B1 - B2 - B3 - B4 - B5 - B6 -
    |       |    |    |    |    |    |    |
    |       |----|    |----|    |----|    |
    |       | U1 |    | U3 |    | U5 |    |
    |       |----|    |----|    |----|    |
    |       |    |----|    |----|    |----|
    |       |    | U2 |    | U4 |    | U6 |
    |       |    |----|    |----|    |----|
    |                   .
    |                   .
    |                   .

After each application of a `Ui`, the MPS needs to be truncated - otherwise the bond dimension
`chi` would grow indefinitely. A bound for the error introduced by the truncation is returned.

If one chooses imaginary :math:`dt`, the exponential projects
(for sufficiently long 'time' evolution) onto the ground state of the Hamiltonian.

.. note ::
    The application of DMRG is typically much more efficient than imaginary TEBD!
    Yet, imaginary TEBD might be useful for cross-checks and testing.

"""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np
import time
import warnings
import logging
logger = logging.getLogger(__name__)

from .algorithm import TimeEvolutionAlgorithm, TimeDependentHAlgorithm
from ..linalg import np_conserved as npc
from .truncation import svd_theta, TruncationError, truncate
from ..linalg import random_matrix

__all__ = ['TEBDEngine', 'Engine', 'QRBasedTEBDEngine', 'RandomUnitaryEvolution', 'TimeDependentTEBD']


class TEBDEngine(TimeEvolutionAlgorithm):
    """Time Evolving Block Decimation (TEBD) algorithm.

    Parameters are the same as for :class:`~tenpy.algorithms.algorithm.Algorithm`.

    .. deprecated :: 0.6.0
        Renamed parameter/attribute `TEBD_params` to :attr:`options`.

    Options
    -------
    .. cfg:config :: TEBDEngine
        :include: TimeEvolutionAlgorithm

        start_trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
            Initial truncation error for :attr:`trunc_err`.
        order : int
            Order of the algorithm. The total error for evolution up to a fixed time `t`
            scales as ``O(t*dt^order)``.
        E_offset : None | list of float
            Energy offset to be applied in :meth:`calc_U`, see doc there.
            Only used for real-time evolution!

    Attributes
    ----------
    trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
        The error of the represented state which is introduced due to the truncation during
        the sequence of update steps.
    psi : :class:`~tenpy.networks.mps.MPS`
        The MPS, time evolved in-place.
    model : :class:`~tenpy.models.model.NearestNeighborModel`
        The model defining the Hamiltonian.
    _U : list of list of :class:`~tenpy.linalg.np_conserved.Array`
        Exponentiated `H_bond` (bond Hamiltonians), i.e. roughly ``exp(-i H_bond dt_i)``.
        First list for different `dt_i` as necessary for the chosen `order`,
        second list for the `L` different bonds.
    _U_param : dict
        A dictionary containing the information of the latest created `_U`.
        We don't recalculate `_U` if those parameters didn't change.
    _trunc_err_bonds : list of :class:`~tenpy.algorithms.truncation.TruncationError`
        The *local* truncation error introduced at each bond, ignoring the errors at other bonds.
        The `i`-th entry is left of site `i`.
    _update_index : None | (int, int)
        The indices ``i_dt,i_bond`` of ``U_bond = self._U[i_dt][i_bond]`` during update_step.
    """
    def __init__(self, psi, model, options, **kwargs):
        TimeEvolutionAlgorithm.__init__(self, psi, model, options, **kwargs)
        self._trunc_err_bonds = [TruncationError() for i in range(psi.L + 1)]
        self._U = None
        self._U_param = {}
        self._update_index = None

    @property
    def TEBD_params(self):
        warnings.warn("renamed self.TEBD_params -> self.options", FutureWarning, stacklevel=2)
        return self.options

    @property
    def trunc_err_bonds(self):
        """truncation error introduced on each non-trivial bond."""
        return self._trunc_err_bonds[self.psi.nontrivial_bonds]

    # run() as defined in TimeEvolutionAlgorithm

    def run_GS(self):
        """TEBD algorithm in imaginary time to find the ground state.

        .. note ::
            It is almost always more efficient (and hence advisable) to use DMRG.
            This algorithms can nonetheless be used quite well as a benchmark and for comparison.

        .. cfg:configoptions :: TEBDEngine

            delta_tau_list : list
                A list of floats: the timesteps to be used.
                Choosing a large timestep `delta_tau` introduces large (Trotter) errors,
                but a too small time step requires a lot of steps to reach
                ``exp(-tau H) --> |psi0><psi0|``.
                Therefore, we start with fairly large time steps for a quick time evolution until
                convergence, and then gradually decrease the time step.
            order : int
                Order of the Suzuki-Trotter decomposition.
            N_steps : int
                Number of steps before measurement can be performed
        """
        # initialize parameters
        delta_tau_list = self.options.get(
            'delta_tau_list',
            [0.1, 0.01, 0.001, 1.e-4, 1.e-5, 1.e-6, 1.e-7, 1.e-8, 1.e-9, 1.e-10, 1.e-11, 0.])
        max_error_E = self.options.get('max_error_E', 1.e-13)
        N_steps = self.options.get('N_steps', 10)
        TrotterOrder = self.options.get('order', 2)

        Eold = np.mean(self.model.bond_energies(self.psi))
        Sold = np.mean(self.psi.entanglement_entropy())
        start_time = time.time()

        for delta_tau in delta_tau_list:
            logger.info("delta_tau=%e", delta_tau)
            self.calc_U(TrotterOrder, delta_tau, type_evo='imag')
            DeltaE = 2 * max_error_E
            step = 0
            while (DeltaE > max_error_E):
                if self.psi.finite and TrotterOrder == 2:
                    self.update_imag(N_steps)
                else:
                    self.evolve(N_steps, delta_tau)
                step += N_steps
                E = np.mean(self.model.bond_energies(self.psi))
                DeltaE = abs(Eold - E)
                Eold = E
                S = self.psi.entanglement_entropy()
                max_S = max(S)
                S = np.mean(S)
                DeltaS = S - Sold
                Sold = S
                logger.info(
                    "--> step=%(step)6d, beta=%(beta)3.3f, max(chi)=%(max_chi)d,"
                    "DeltaE=%(dE).2e, E_bond=%(E).10f, Delta_S=%(dS).4e, "
                    "max(S)=%(max_S).10f, time simulated: %(wall_time).1fs", {
                        'step': step,
                        'beta': -self.evolved_time.imag,
                        'max_chi': max(self.psi.chi),
                        'dE': DeltaE,
                        'E': E.real,
                        'dS': DeltaS,
                        'max_S': max_S,
                        'wall_time': time.time() - start_time,
                    })
        # done

    @staticmethod
    def suzuki_trotter_time_steps(order):
        """Return time steps of U for the Suzuki Trotter decomposition of desired order.

        See :meth:`suzuki_trotter_decomposition` for details.

        Parameters
        ----------
        order : int
            The desired order of the Suzuki-Trotter decomposition.

        Returns
        -------
        time_steps : list of float
            We need ``U = exp(-i H_{even/odd} delta_t * dt)`` for the `dt` returned in this list.
        """
        if order == 1:
            return [1.]
        elif order == 2:
            return [0.5, 1.]
        elif order == 4:
            t1 = 1. / (4. - 4.**(1 / 3.))
            t3 = 1. - 4. * t1
            return [t1 / 2., t1, (t1 + t3) / 2., t3]
        elif order == '4_opt':
            # Eq (30a) of arXiv:1901.04974
            a1 = 0.095848502741203681182
            b1 = 0.42652466131587616168
            a2 = -0.078111158921637922695
            b2 = -0.12039526945509726545
            return [a1, b1, a2, b2, 0.5 - a1 - a2, 1. - 2 * (b1 + b2)]  # a1 b1 a2 b2 a3 b3
        # else
        raise ValueError("Unknown order %r for Suzuki Trotter decomposition" % order)

    @staticmethod
    def suzuki_trotter_decomposition(order, N_steps):
        r"""Returns list of necessary steps for the suzuki trotter decomposition.

        We split the Hamiltonian as :math:`H = H_{even} + H_{odd} = H[0] + H[1]`.
        The Suzuki-Trotter decomposition is an approximation
        :math:`\exp(t H) \approx prod_{(j, k) \in ST} \exp(d[j] t H[k]) + O(t^{order+1 })`.

        Parameters
        ----------
        order : ``1, 2, 4, '4_opt'``
            The desired order of the Suzuki-Trotter decomposition.
            Order ``1`` approximation is simply :math:`e^A a^B`.
            Order ``2`` is the "leapfrog" `e^{A/2} e^B e^{A/2}`.
            Order ``4`` is the fourth-order from :cite:`suzuki1991` (also referenced in
            :cite:`schollwoeck2011`), and ``'4_opt'`` gives the optimized version of Equ. (30a) in
            :cite:`barthel2020`.

        Returns
        -------
        ST_decomposition : list of (int, int)
            Indices ``j, k`` of the time-steps ``d = suzuki_trotter_time_step(order)`` and
            the decomposition of `H`.
            They are chosen such that a subsequent application of ``exp(d[j] t H[k])`` to a given
            state ``|psi>`` yields ``(exp(N_steps t H[k]) + O(N_steps t^{order+1}))|psi>``.
        """
        even, odd = 0, 1
        if N_steps == 0:
            return []
        if order == 1:
            a = (0, odd)
            b = (0, even)
            return [a, b] * N_steps
        elif order == 2:
            a = (0, odd)  # dt/2
            a2 = (1, odd)  # dt
            b = (1, even)  # dt
            # U = [a b a]*N
            #   = a b [a2 b]*(N-1) a
            return [a, b] + [a2, b] * (N_steps - 1) + [a]
        elif order == 4:
            a = (0, odd)  # t1/2
            a2 = (1, odd)  # t1
            b = (1, even)  # t1
            c = (2, odd)  # (t1 + t3) / 2 == (1 - 3 * t1)/2
            d = (3, even)  # t3 = 1 - 4 * t1
            # From Schollwoeck 2011 (:arxiv:`1008.3477`):
            # U = U(t1) U(t2) U(t3) U(t2) U(t1)
            # with U(dt) = U(dt/2, odd) U(dt, even) U(dt/2, odd) and t1 == t2
            # Using above definitions, we arrive at:
            # U = [a b a2 b c d c b a2 b a] * N
            #   = [a b a2 b c d c b a2 b] + [a2 b a2 b c d c b a2 b a] * (N-1) + [a]
            steps = [a, b, a2, b, c, d, c, b, a2, b]
            steps = steps + [a2, b, a2, b, c, d, c, b, a2, b] * (N_steps - 1)
            steps = steps + [a]
            return steps
        elif order == '4_opt':
            # symmetric: a1 b1 a2 b2 a3 b3 a2 b2 a2 b1 a1
            steps = [(0, odd), (1, even), (2, odd), (3, even), (4, odd),  (5, even),
                     (4, odd), (3, even), (2, odd), (1, even), (0, odd)]  # yapf: disable
            return steps * N_steps
        # else
        raise ValueError("Unknown order {0!r} for Suzuki Trotter decomposition".format(order))

    def prepare_evolve(self, dt):
        order = self.options.get('order', 2)
        E_offset = self.options.get('E_offset', None)
        self.calc_U(order, dt, type_evo='real', E_offset=E_offset)

    def calc_U(self, order, delta_t, type_evo='real', E_offset=None):
        """Calculate ``self.U_bond`` from ``self.model.H_bond``.

        This function calculates

        * ``U_bond = exp(-i dt (H_bond-E_offset_bond))`` for ``type_evo='real'``, or
        * ``U_bond = exp(- dt H_bond)`` for ``type_evo='imag'``.

        For first order (in `delta_t`), we need just one ``dt=delta_t``.
        Higher order requires smaller `dt` steps, as given by :meth:`suzuki_trotter_time_steps`.

        Parameters
        ----------
        order : int
            Trotter order calculated U_bond. See update for more information.
        delta_t : float
            Size of the time-step used in calculating U_bond
        type_evo : ``'imag' | 'real'``
            Determines whether we perform real or imaginary time-evolution.
        E_offset : None | list of float
            Possible offset added to `H_bond` for real-time evolution.
        """
        U_param = dict(order=order, delta_t=delta_t, type_evo=type_evo, E_offset=E_offset)
        if type_evo == 'real':
            U_param['tau'] = delta_t
        elif type_evo == 'imag':
            U_param['tau'] = -1.j * delta_t
        else:
            raise ValueError("Invalid value for `type_evo`: " + repr(type_evo))
        if self._U_param == U_param and not self.force_prepare_evolve:
            logger.debug("Skip recalculation of U with same parameters as before")
            return  # nothing to do: U is cached
        self._U_param = U_param
        logger.info("Calculate U for %s", U_param)

        L = self.psi.L
        self._U = []
        for dt in self.suzuki_trotter_time_steps(order):
            U_bond = [
                self._calc_U_bond(i_bond, dt * delta_t, type_evo, E_offset) for i_bond in range(L)
            ]
            self._U.append(U_bond)
        self.force_prepare_evolve = False

    def evolve(self, N_steps, dt):
        """Evolve by ``dt * N_steps``.

        Parameters
        ----------
        N_steps : int
            The number of steps for which the whole lattice should be updated.
        dt : float
            The time step; but really this was already used in :meth:`prepare_evolve`.

        Returns
        -------
        trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
            The error of the represented state which is introduced due to the truncation during
            this sequence of evolution steps.
        """
        if dt is not None:
            assert dt == self._U_param['delta_t']
        trunc_err = TruncationError()
        order = self._U_param['order']
        for U_idx_dt, odd in self.suzuki_trotter_decomposition(order, N_steps):
            trunc_err += self.evolve_step(U_idx_dt, odd)
        self.evolved_time = self.evolved_time + N_steps * self._U_param['tau']
        self.trunc_err = self.trunc_err + trunc_err  # not += : make a copy!
        # (this is done to avoid problems of users storing self.trunc_err after each `evolve`)
        return trunc_err

    def evolve_step(self, U_idx_dt, odd):
        """Updates either even *or* odd bonds in unit cell.

        Depending on the choice of p, this function updates all even (``E``, odd=False,0)
        **or** odd (``O``) (odd=True,1) bonds::

        |     - B0 - B1 - B2 - B3 - B4 - B5 - B6 -
        |       |    |    |    |    |    |    |
        |       |    |----|    |----|    |----|
        |       |    |  E |    |  E |    |  E |
        |       |    |----|    |----|    |----|
        |       |----|    |----|    |----|    |
        |       |  O |    |  O |    |  O |    |
        |       |----|    |----|    |----|    |

        Note that finite boundary conditions are taken care of by having ``Us[0] = None``.

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
        for i_bond in np.arange(int(odd) % 2, self.psi.L, 2):
            if Us[i_bond] is None:
                continue  # handles finite vs. infinite boundary conditions
            self._update_index = (U_idx_dt, i_bond)
            trunc_err += self.update_bond(i_bond, Us[i_bond])
        self._update_index = None
        return trunc_err

    def update_bond(self, i, U_bond):
        """Updates the B matrices on a given bond.

        Function that updates the B matrices, the bond matrix s between and the
        bond dimension chi for bond i. The corresponding tensor networks look like this::

        |           --S--B1--B2--           --B1--B2--
        |                |   |                |   |
        |     theta:     U_bond        C:     U_bond
        |                |   |                |   |

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
        logger.debug("Update sites (%d, %d)", i0, i1)
        # Construct the theta matrix
        C = self.psi.get_theta(i0, n=2, formL=0.)  # the two B without the S on the left
        C = npc.tensordot(U_bond, C, axes=(['p0*', 'p1*'], ['p0', 'p1']))  # apply U
        C.itranspose(['vL', 'p0', 'p1', 'vR'])
        theta = C.scale_axis(self.psi.get_SL(i0), 'vL')
        # now theta is the same as if we had done
        #   theta = self.psi.get_theta(i0, n=2)
        #   theta = npc.tensordot(U_bond, theta, axes=(['p0*', 'p1*'], ['p0', 'p1']))  # apply U
        # but also have C which is the same except the missing "S" on the left
        # so we don't have to apply inverses of S (see below)

        theta = theta.combine_legs([('vL', 'p0'), ('p1', 'vR')], qconj=[+1, -1])
        # Perform the SVD and truncate the wavefunction
        U, S, V, trunc_err, renormalize = svd_theta(theta,
                                                    self.trunc_params,
                                                    [self.psi.get_B(i0, None).qtotal, None],
                                                    inner_labels=['vR', 'vL'])

        # Split tensor and update matrices
        B_R = V.split_legs(1).ireplace_label('p1', 'p')

        # In general, we want to do the following:
        #     U = U.iscale_axis(S, 'vR')
        #     B_L = U.split_legs(0).iscale_axis(self.psi.get_SL(i0)**-1, 'vL')
        #     B_L = B_L.ireplace_label('p0', 'p')
        # i.e. with SL = self.psi.get_SL(i0), we have ``B_L = SL**-1 U S``
        #
        # However, the inverse of SL is problematic, as it might contain very small singular
        # values.  Instead, we use ``C == SL**-1 theta == SL**-1 U S V``,
        # such that we obtain ``B_L = SL**-1 U S = SL**-1 U S V V^dagger = C V^dagger``
        # here, C is the same as theta, but without the `S` on the very left
        # (Note: this requires no inverse if the MPS is initially in 'B' canonical form)
        B_L = npc.tensordot(C.combine_legs(('p1', 'vR'), pipes=theta.legs[1]),
                            V.conj(),
                            axes=['(p1.vR)', '(p1*.vR*)'])
        B_L.ireplace_labels(['vL*', 'p0'], ['vR', 'p'])
        B_L /= renormalize  # re-normalize to <psi|psi> = 1
        self.psi.norm *= renormalize
        self.psi.set_SR(i0, S)
        self.psi.set_B(i0, B_L, form='B')
        self.psi.set_B(i1, B_R, form='B')
        self._trunc_err_bonds[i] = self._trunc_err_bonds[i] + trunc_err
        return trunc_err

    def update_imag(self, N_steps):
        """Perform an update suitable for imaginary time evolution.

        Instead of the even/odd brick structure used for ordinary TEBD,
        we 'sweep' from left to right and right to left, similar as DMRG.
        Thanks to that, we are actually able to preserve the canonical form.

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
        # allow only second order evolution
        if order != 2 or not self.psi.finite:
            # Would lead to loss of canonical form. What about DMRG?
            raise NotImplementedError("Use DMRG instead...")
        U_idx_dt = 0  # always with dt=0.5
        assert (self.suzuki_trotter_time_steps(order)[U_idx_dt] == 0.5)
        assert (self.psi.finite)  # finite or segment bc
        Us = self._U[U_idx_dt]
        for _ in range(N_steps):
            # sweep right
            for i_bond in range(self.psi.L):
                if Us[i_bond] is None:
                    continue  # handles finite vs. infinite boundary conditions
                self._update_index = (U_idx_dt, i_bond)
                trunc_err += self.update_bond_imag(i_bond, Us[i_bond])
            # sweep left
            for i_bond in range(self.psi.L - 1, -1, -1):
                if Us[i_bond] is None:
                    continue  # handles finite vs. infinite boundary conditions
                self._update_index = (U_idx_dt, i_bond)
                trunc_err += self.update_bond_imag(i_bond, Us[i_bond])
        self._update_index = None
        self.evolved_time = self.evolved_time + N_steps * self._U_param['tau']
        self.trunc_err = self.trunc_err + trunc_err  # not += : make a copy!
        # (this is done to avoid problems of users storing self.trunc_err after each `update`)
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
        logger.debug("Update sites (%d, %d)", i0, i1)
        # Construct the theta matrix
        theta = self.psi.get_theta(i0, n=2)  # 'vL', 'vR', 'p0', 'p1'
        theta = npc.tensordot(U_bond, theta, axes=(['p0*', 'p1*'], ['p0', 'p1']))
        theta = theta.combine_legs([('vL', 'p0'), ('vR', 'p1')], qconj=[+1, -1])
        # Perform the SVD and truncate the wavefunction
        U, S, V, trunc_err, renormalize = svd_theta(theta,
                                                    self.trunc_params,
                                                    inner_labels=['vR', 'vL'])
        self.psi.norm *= renormalize
        # Split legs and update matrices
        B_R = V.split_legs(1).ireplace_label('p1', 'p')
        A_L = U.split_legs(0).ireplace_label('p0', 'p')
        self.psi.set_SR(i0, S)
        self.psi.set_B(i0, A_L, form='A')
        self.psi.set_B(i1, B_R, form='B')
        self._trunc_err_bonds[i] = self._trunc_err_bonds[i] + trunc_err
        return trunc_err

    def _calc_U_bond(self, i_bond, dt, type_evo, E_offset):
        """Calculate exponential of a bond Hamiltonian.

        * ``U_bond = exp(-i dt (H_bond-E_offset_bond))`` for ``type_evo='real'``, or
        * ``U_bond = exp(- dt H_bond)`` for ``type_evo='imag'``.
        """
        h = self.model.H_bond[i_bond]
        if h is None:
            return None  # don't calculate exp(i H t), if `H` is None
        H2 = h.combine_legs([('p0', 'p1'), ('p0*', 'p1*')], qconj=[+1, -1])
        if type_evo == 'imag':
            H2 = (-dt) * H2
        elif type_evo == 'real':
            if E_offset is not None:
                H2 = H2 - npc.diag(E_offset[i_bond], H2.legs[0])
            H2 = (-1.j * dt) * H2
        else:
            raise ValueError("Expect either 'real' or 'imag'inary time, got " + repr(type_evo))
        U = npc.expm(H2)
        assert (tuple(U.get_leg_labels()) == ('(p0.p1)', '(p0*.p1*)'))
        return U.split_legs()


class Engine(TEBDEngine):
    """Deprecated old name of :class:`TEBDEngine`.

    .. deprecated : v0.8.0
        Renamed the `Engine` to `TEBDEngine` to have unique algorithm class names.
    """
    def __init__(self, psi, model, options):
        msg = "Renamed `Engine` class to `TEBDEngine`."
        warnings.warn(msg, category=FutureWarning, stacklevel=2)
        TEBDEngine.__init__(self, psi, model, options)


class QRBasedTEBDEngine(TEBDEngine):
    r"""Version of TEBD that relies on QR decompositions rather than SVD.

    As introduced in :arxiv:`2212.09782`.

    .. todo ::
        To use `use_eig_based_svd == True`, which makes sense on GPU only, we need to implement
        the `_eig_based_svd` for "non-square" matrices.
        This means that :math:`M^{\dagger} M` and :math:`M M^{\dagger}` dont have the same size,
        and we need to disregard those eigenvectors of the larger one, that have eigenvalue zero,
        since we dont have corresponding eigenvalues of the smaller one.

    Options
    -------
    .. cfg:config :: QRBasedTEBDEngine
        :include: TEBDEngine

        cbe_expand : float
            Expansion rate. The QR-based decomposition is carried out at an expanded bond dimension
            ``eta = (1 + cbe_expand) * chi``, where ``chi`` is the bond dimension before the time step.
            Default is `0.1`.
        cbe_expand_0 : float
            Expansion rate at low ``chi``.
            If given, the expansion rate decreases linearly from ``cbe_expand_0`` at ``chi == 1``
            to ``cbe_expand`` at ``chi == trunc_params['chi_max']``, then remains constant.
            If not given, the expansion rate is ``cbe_expand`` at all ``chi``.
        cbe_min_block_increase : int
            Minimum bond dimension increase for each block. Default is `1`.
        use_eig_based_svd : bool
            Whether the SVD of the bond matrix :math:`\Xi` should be carried out numerically via
            the eigensystem. This is faster on GPUs, but less accurate.
            It makes no sense to do this on CPU. It is currently not supported for update_imag.
            Default is `False`.
        compute_err : bool
            Whether the truncation error should be computed exactly.
            Compared to SVD-based TEBD, computing the truncation error is significantly more expensive.
            If `True` (default), the full error is computed.
            Otherwise, the truncation error is set to NaN.
    """

    def _expansion_rate(self, i):
        """get expansion rate for updating bond i"""
        expand = self.options.get('cbe_expand', 0.1)
        expand_0 = self.options.get('cbe_expand_0', None)

        if expand_0 is None or expand_0 == expand:
            return expand

        chi_max = self.trunc_params.get('chi_max', None)
        if chi_max is None:
            raise ValueError('Need to specify trunc_params["chi_max"] in order to use cbe_expand_0.')

        chi = min(self.psi.get_SL(i).shape)
        return max(expand_0 - chi / chi_max * (expand_0 - expand), expand)

    def update_bond(self, i, U_bond):
        i0, i1 = i - 1, i
        expand = self._expansion_rate(i)
        logger.debug(f'Update sites ({i0}, {i1}). CBE expand={expand}')
        # Construct the theta matrix
        C = self.psi.get_theta(i0, n=2, formL=0.)  # the two B without the S on the left
        C = npc.tensordot(U_bond, C, axes=(['p0*', 'p1*'], ['p0', 'p1']))  # apply U
        C.itranspose(['vL', 'p0', 'p1', 'vR'])
        theta = C.scale_axis(self.psi.get_SL(i0), 'vL')
        theta = theta.combine_legs([('vL', 'p0'), ('p1', 'vR')], qconj=[+1, -1])

        min_block_increase = self.options.get('cbe_min_block_increase', 1)
        Y0 = _qr_tebd_cbe_Y0(B_L=self.psi.get_B(i0, 'B'), B_R=self.psi.get_B(i1, 'B'), theta=theta,
                             expand=expand, min_block_increase=min_block_increase)
        A_L, S, B_R, trunc_err, renormalize = _qr_based_decomposition(
            theta=theta, Y0=Y0, use_eig_based_svd=self.options.get('use_eig_based_svd', False),
            need_A_L=False, compute_err=self.options.get('compute_err', True),
            trunc_params=self.trunc_params
        )
        B_L = npc.tensordot(C.combine_legs(('p1', 'vR'), pipes=theta.legs[1]),
                            B_R.conj(),
                            axes=[['(p1.vR)'], ['(p*.vR*)']]) / renormalize
        B_L.ireplace_labels(['p0', 'vL*'], ['p', 'vR'])
        B_R = B_R.split_legs(1)
        self.psi.norm *= renormalize
        self.psi.set_B(i0, B_L, form='B')
        self.psi.set_SL(i1, S)
        self.psi.set_B(i1, B_R, form='B')
        self._trunc_err_bonds[i] = self._trunc_err_bonds[i] + trunc_err
        return trunc_err

    def update_bond_imag(self, i, U_bond):
        i0, i1 = i - 1, i
        expand = self._expansion_rate(i)
        logger.debug(f'Update sites ({i0}, {i1}). CBE expand={expand}')
        # Construct the theta matrix
        theta = self.psi.get_theta(i0, n=2)
        theta = npc.tensordot(U_bond, theta, axes=(['p0*', 'p1*'], ['p0', 'p1']))
        theta.itranspose(['vL', 'p0', 'p1', 'vR'])
        theta = theta.combine_legs([('vL', 'p0'), ('p1', 'vR')], qconj=[+1, -1])

        use_eig_based_svd = self.options.get('use_eig_based_svd', False)

        if use_eig_based_svd:
            # see todo comment in _eig_based_svd
            raise NotImplementedError('update_bond_imag does not (yet) support eig based SVD')

        min_block_increase = self.options.get('cbe_min_block_increase', 1)
        Y0 = _qr_tebd_cbe_Y0(B_L=self.psi.get_B(i0, 'B'), B_R=self.psi.get_B(i1, 'B'), theta=theta,
                             expand=expand, min_block_increase=min_block_increase)
        A_L, S, B_R, trunc_err, renormalize = _qr_based_decomposition(
            theta=theta, Y0=Y0, use_eig_based_svd=use_eig_based_svd,
            need_A_L=True, compute_err=self.options.get('compute_err', True),
            trunc_params=self.trunc_params
        )
        A_L = A_L.split_legs(0)
        B_R = B_R.split_legs(1)

        self.psi.norm *= renormalize
        self.psi.set_B(i0, A_L, form='A')
        self.psi.set_SL(i1, S)
        self.psi.set_B(i1, B_R, form='B')
        self._trunc_err_bonds[i] = self._trunc_err_bonds[i] + trunc_err

        return trunc_err


def _qr_tebd_cbe_Y0(B_L: npc.Array, B_R: npc.Array, theta: npc.Array, expand: float, min_block_increase: int):
    """Generate the initial guess Y0 for the right isometry in QR based TEBD.

    Parameters
    ----------
    B_L : Array with legs [vL, p, vR]
    B_R : Array with legs [vL, p, vR]
    theta : Array with legs [(vL.p0), (p1.vR)]
    expand : float or None

    Returns
    -------
    Y0 : Array with legs [vL, (p1.vR)]
    """
    if expand is None or expand == 0:
        return B_R.combine_legs(['p', 'vR']).ireplace_labels('(p.vR)', '(p1.vR)')

    assert min_block_increase >= 0

    Y0 = theta.copy(deep=False)
    Y0.legs[0] = Y0.legs[0].to_LegCharge()
    Y0.ireplace_label('(vL.p0)', 'vL')
    if any(B_L.qtotal != 0):
        Y0.gauge_total_charge('vL', new_qtotal=B_R.qtotal)
    vL_old = B_R.get_leg('vL')
    if not vL_old.is_blocked():
        vL_old = vL_old.sort()[1]
    vL_new = Y0.get_leg('vL')  # is blocked, since created from pipe

    # vL_old is guaranteed to be a slice of vL_new by charge rule in B_L
    piv = np.zeros(vL_new.ind_len, dtype=bool)  # indices to keep in vL_new
    increase_per_block = max(min_block_increase, int(vL_old.ind_len * expand // vL_new.block_number))
    sizes_old = vL_old.get_block_sizes()
    sizes_new = vL_new.get_block_sizes()
    # iterate over charge blocks in vL_new and vL_old at the same time
    j_old = 0
    q_old = vL_old.charges[j_old, :]
    qdata_order = np.argsort(Y0._qdata[:, 0])
    qdata_idx = 0
    for j_new, q_new in enumerate(vL_new.charges):
        if all(q_new == q_old):  # have charge block in both vL_new and vL_old
            s_new = sizes_old[j_old] + increase_per_block
            # move to next charge block in next loop iteration
            j_old += 1
            if j_old < len(vL_old.charges):
                q_old = vL_old.charges[j_old, :]
        else:  # charge block only in vL_new
            s_new = increase_per_block
        s_new = min(s_new, sizes_new[j_new])  # don't go beyond block

        if Y0._qdata[qdata_order[qdata_idx], 0] != j_new:
            # block does not exist
            # while we could set corresponding piv entries to True, it would not help, since
            # the corresponding "entries" of Y0 are zero anyway
            continue

        # block has axis [vL, (p1.vR)]. want to keep the s_new slices of the vL axis
        #  that have the largest norm
        norms = np.linalg.norm(Y0._data[qdata_order[qdata_idx]], axis=1)
        kept_slices = np.argsort(-norms)[:s_new]  # negative sign so we sort large to small
        start = vL_new.slices[j_new]
        piv[start + kept_slices] = True

        qdata_idx += 1
        if qdata_idx >= Y0._qdata.shape[0]:
            break

    Y0.iproject(piv, 'vL')
    return Y0


def _qr_based_decomposition(theta: npc.Array, Y0: npc.Array, use_eig_based_svd: bool, trunc_params,
                            need_A_L: bool, compute_err: bool):
    """Perform the decomposition step of QR based TEBD

    Parameters
    ----------
    theta : Array with legs [(vL.p0), (p1.vR)]
    Y0 : Array with legs [vL, (p1.vR)]
    ...

    Returns
    -------
    A_L : array with legs [(vL.p), vR] or None
    S : 1D numpy array
    B_R : array with legs [vL, (p.vR)]
    trunc_err : TruncationError
    renormalize : float
    """

    if compute_err:
        need_A_L = True

    # QR based updates
    theta_i0 = npc.tensordot(theta, Y0.conj(), ['(p1.vR)', '(p1*.vR*)']).ireplace_label('vL*', 'vR')
    A_L, _ = npc.qr(theta_i0, inner_labels=['vR', 'vL'])
    # A_L: [(vL.p0), vR]
    theta_i1 = npc.tensordot(A_L.conj(), theta, ['(vL*.p0*)', '(vL.p0)']).ireplace_label('vR*', 'vL')
    theta_i1.itranspose(['(p1.vR)', 'vL'])
    B_R, Xi = npc.qr(theta_i1, inner_labels=['vL', 'vR'], inner_qconj=-1)
    B_R.itranspose(['vL', '(p1.vR)'])
    Xi.itranspose(['vL', 'vR'])

    # SVD of bond matrix Xi
    if use_eig_based_svd:
        U, S, Vd, trunc_err, renormalize = _eig_based_svd(
            Xi, inner_labels=['vR', 'vL'], need_U=need_A_L, trunc_params=trunc_params
        )
    else:
        U, S, Vd, _, renormalize = svd_theta(Xi, trunc_params)
    B_R = npc.tensordot(Vd, B_R, ['vR', 'vL'])
    if need_A_L:
        A_L = npc.tensordot(A_L, U, ['vR', 'vL'])
    else:
        A_L = None

    if compute_err:
        theta_approx = npc.tensordot(A_L.scale_axis(S, axis='vR'), B_R, ['vR', 'vL'])
        eps = npc.norm(theta - theta_approx) ** 2
        trunc_err = TruncationError(eps, 1. - 2. * eps)
    else:
        trunc_err = TruncationError(np.nan, np.nan)

    B_R = B_R.ireplace_label('(p1.vR)', '(p.vR)')
    if need_A_L:
        A_L = A_L.ireplace_label('(vL.p0)', '(vL.p)')

    return A_L, S, B_R, trunc_err, renormalize


def _eig_based_svd(A, need_U: bool = True, need_Vd: bool = True, inner_labels=[None, None],
                   trunc_params=None):
    """Computes the singular value decomposition of a matrix A via eigh

    Singular values and vectors are obtained by diagonalizing the "square" A.hc @ A and/or A @ A.hc,
    i.e. with two eigh calls instead of an svd call.

    Truncation if performed if and only if trunc_params are given.
    This performs better on GPU, but is not really useful on CPU.
    If isometries U or Vd are not needed, their computation can be omitted for performance.

    Does not (yet) support computing both U and Vd
    """
    warnings.warn('_eig_based_svd is nonsensical on CPU!!')
    assert A.rank == 2

    if need_U and need_Vd:
        # TODO (JU) just doing separate eighs for U, S and for S, Vd is not sufficient
        #  the phases of U / Vd are arbitrary.
        #  Need to put in more work in that case...
        raise NotImplementedError

    if need_U:
        Vd = None
        A_Ahc = npc.tensordot(A, A.conj(), [1, 1])
        L, U = npc.eigh(A_Ahc, sort='>')
        S = np.sqrt(np.abs(L))  # abs to avoid `nan` due to accidentally negative values close to zero
        U = U.ireplace_label('eig', inner_labels[0])
    elif need_Vd:
        U = None
        Ahc_A = npc.tensordot(A.conj(), A, [0, 0])
        L, V = npc.eigh(Ahc_A, sort='>')
        S = np.sqrt(np.abs(L))  # abs to avoid `nan` due to accidentally negative values close to zero
        Vd = V.iconj().itranspose().ireplace_label('eig*', inner_labels[1])
    else:
        U = None
        Vd = None
        # use the smaller of the two square matrices -- they have the same eigenvalues
        if A.shape[1] >= A.shape[0]:
            A2 = npc.tensordot(A, A.conj(), [1, 0])
        else:
            A2 = npc.tensordot(A.conj(), A, [1, 0])
        L = npc.eigvalsh(A2)
        S = np.sqrt(np.abs(L))  # abs to avoid `nan` due to accidentally negative values close to zero

    if trunc_params is not None:
        piv, renormalize, trunc_err = truncate(S, trunc_params)
        S = S[piv]
        S /= renormalize
        if need_U:
            U.iproject(piv, 1)
        if need_Vd:
            Vd.iproject(piv, 0)
    else:
        renormalize = np.linalg.norm(S)
        S /= renormalize
        trunc_err = TruncationError()

    return U, S, Vd, trunc_err, renormalize


class RandomUnitaryEvolution(TEBDEngine):
    """Evolution of an MPS with random two-site unitaries in a TEBD-like fashion.

    Instead of using a model Hamiltonian, this TEBD engine evolves with random two-site unitaries.
    These unitaries are drawn according to the Haar measure on unitaries obeying the conservation
    laws dictated by the conserved charges. If no charge is preserved, this distribution is called
    circular unitary ensemble (CUE), see :func:`~tenpy.linalg.random_matrix.CUE`.
    The distribution can be changed through the
    :cfg:option:`RandomUnitaryEvolution.distribution_function`.

    On one hand, such an evolution is of interest in recent research (see eg. :arxiv:`1710.09827`).
    On the other hand, it also comes in handy to "randomize" an initial state, e.g. for DMRG.
    Note that the entanglement grows very quickly, choose the truncation parameters accordingly!

    Options
    -------
    .. cfg:config :: RandomUnitaryEvolution
        :include: TEBDEngine

        N_steps : int
            Number of two-site unitaries to be applied on each bond.
        trunc_params : dict
            Truncation parameters as described in :cfg:config:`truncate`

    Examples
    --------
    One can initialize a "random" state with total Sz = L//2 as follows:

    .. doctest :: RandomUnitaryEvolution

        >>> from tenpy.algorithms.tebd import RandomUnitaryEvolution
        >>> from tenpy.networks.mps import MPS
        >>> L = 8
        >>> spin_half = tenpy.networks.site.SpinHalfSite(conserve='Sz')
        >>> psi = MPS.from_product_state([spin_half]*L, ["up", "down"]*(L//2), bc='finite')  # Neel
        >>> print(psi.chi)
        [1, 1, 1, 1, 1, 1, 1]
        >>> options = dict(N_steps=2, trunc_params={'chi_max':10})
        >>> eng = RandomUnitaryEvolution(psi, options)
        >>> eng.run()
        >>> print(psi.chi)
        [2, 4, 8, 10, 8, 4, 2]
        >>> psi.canonical_form()  # a good idea if there was a truncation necessary.

    The "random" unitaries preserve the specified charges, e.g. here we have Sz-conservation.
    If you start in a sector of all up spins, the random unitaries can only apply a phase:

    .. doctest :: RandomUnitaryEvolution

        >>> psi2 = MPS.from_product_state([spin_half]*L, ["up"]*L, bc='finite')  # all spins up
        >>> print(psi2.chi)
        [1, 1, 1, 1, 1, 1, 1]
        >>> eng2 = RandomUnitaryEvolution(psi2, options)
        >>> eng2.run()  # random unitaries respect Sz conservation -> we stay in all-up sector
        >>> print(psi2.chi)  # still a product state, not really random!!!
        [1, 1, 1, 1, 1, 1, 1]
    """

    def __init__(self, psi, options, **kwargs):
        TEBDEngine.__init__(self, psi, None, options, **kwargs)

    def run(self):
        """Time evolution with TEBD and random two-site unitaries (possibly conserving charges)."""
        dt = self.options.get('dt', 1)
        if dt != 1:
            warnings.warn(f"dt={dt!s} != 1 for RandomUnitaryEvolution "
                          "is only used as unit for evolved_time")
        super().run()

    def prepare_evolve(self, dt):
        "Do nothing, as we call :meth:`calc_U` directly in :meth:`update`."
        pass

    def calc_U(self):
        """Draw new random two-site unitaries replacing the usual `U` of TEBD.

        The parameter `dt` is only there for compatibility with parent classes and is ignored.

        .. cfg:configoptions :: RandomUnitaryEvolution

            distribution_func : str | function
                Function or name for one of the matrix ensembles in
                :mod:`~tenpy.linalg.random_matrix` which generates unitaries (or a subset of them).
                To be used as `func` for generating unitaries with
                :meth:`~tenpy.linalg.np_conserved.Array.from_func_square`, i.e. the `U` still
                preserves the charge block structure!
            distribution_func_kwargs : dict
                Extra keyword arguments for `distribution_func`.
        """
        func = self.options.get('distribution_func', "CUE")
        if isinstance(func, str):
            if func not in ["CUE", "CRE", "COE", "O_close_1", "U_close_1"]:
                raise ValueError("distribution_func should generate unitaries")
            func = getattr(random_matrix, func, None)
            assert func is not None
        func_kwargs = self.options.get('distribution_func_kwargs', {})
        sites = self.psi.sites
        L = len(sites)
        U_bonds = []
        for i in range(L):
            if i == 0 and self.psi.finite:
                U_bonds.append(None)
            else:
                leg_L = sites[i - 1].leg
                leg_R = sites[i].leg
                pipe = npc.LegPipe([leg_L, leg_R])
                U = npc.Array.from_func_square(func, pipe, func_kwargs=func_kwargs)
                U = U.split_legs()
                U.iset_leg_labels(['p0', 'p1', 'p0*', 'p1*'])
                U_bonds.append(U)
        self._U = [U_bonds]

    def evolve(self, N_steps, dt):
        """Apply ``N_steps`` random two-site unitaries to each bond (in even-odd pattern).

        Parameters
        ----------
        dt : float
            Mostly ignored, but used as unit to update :attr:`evolved_time`.
        N_steps : int
            The number of steps for which the whole lattice should be updated.

        Returns
        -------
        trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
            The error of the represented state which is introduced due to the truncation during
            this sequence of update steps.
        """
        trunc_err = TruncationError()
        for _ in range(N_steps):
            self.calc_U()  # draw new random unitaries
            for odd in [1, 0]:
                trunc_err += self.evolve_step(0, odd)
        self.evolved_time = self.evolved_time + N_steps * dt
        self.trunc_err = self.trunc_err + trunc_err  # not += : make a copy!
        # (this is done to avoid problems of users storing self.trunc_err after each `update`)
        return trunc_err


class TimeDependentTEBD(TimeDependentHAlgorithm,TEBDEngine):
    """Variant of :class:`TEBDEngine` that can handle time-dependent Hamiltonians.

    See details in :class:`~tenpy.algorithms.algorithm.TimeDependentHAlgorithm` as well.
    """
    # uses run_evolution from TimeDependentHAlgorithm
    # so nothing to redefine here
