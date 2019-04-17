r"""Time evolving block decimation (TEBD).

The TEBD algorithm (proposed in [Vidal2004]_) uses a trotter decomposition of the
Hamiltonian to perform a time evoltion of an MPS. It works only for nearest-neigbor hamiltonians
(in tenpy given by a :class:`~tenpy.models.model.NearestNeighborModel`),
which can be written as :math:`H = H^{even} + H^{odd}`,  such that :math:`H^{even}` contains the
the terms on even bonds (and similar :math:`H^{odd}` the terms on odd bonds).
In the simplest case, we apply first :math:`U=\exp(-i*dt*H^{even})`,
then :math:`U=\exp(-i*dt*H^{odd})` for each time step :math:`dt`.
This is correct up to errors of :math:`O(dt^2)`, but to evolve until a time :math:`T`, we need
:math:`T/dt` steps, so in total it is only correct up to error of :math:`O(T*dt)`.
Similarly, there are higher order schemata (in dt) (for more details see :meth:`Engine.update`).

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
    Yet, imaginary TEBD might be usefull for cross-checks and testing.

"""
# Copyright 2018 TeNPy Developers

import numpy as np
import time

from ..linalg import np_conserved as npc
from .truncation import svd_theta, TruncationError
from ..tools.params import get_parameter, unused_parameters
from ..linalg.random_matrix import CUE

__all__ = ['Engine', 'RandomUnitaryEvolution']


class Engine:
    """Time Evolving Block Decimation (TEBD) 'engine'.

    Parameters
    ----------
    psi : :class:`~tenpy.networs.mps.MPS`
        Initial state to be time evolved. Modified in place.
    model : :class:`~tenpy.models.model.NearestNeighborModel`
        The model representing the Hamiltonian for which we want to find the ground state.
    TEBD_params : dict
        Further optional parameters as described in the tables in
        :func:`run` and :func:`run_GS` for more details.
        Use ``verbose=1`` to print the used parameters during runtime.

    Attributes
    ----------
    verbose : int
        Level of verbosity (i.e. how much status information to print); higher=more output.
    evolved_time : float | complex
        Indicating how long `psi` has been evolved, ``psi = exp(-i * evolved_time * H) psi(t=0)``.
    trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
        The error of the represented state which is introduced due to the truncation during
        the sequence of update steps.
    psi : :class:`~tenpy.networks.mps.MPS`
        The MPS, time evolved in-place.
    model : :class:`~tenpy.models.model.NearestNeigborModel`
        The model defining the Hamiltonian.
    TEBD_params: dict
        Optional parameters, see :func:`run` and :func:`run_GS` for more details.
    _bond_eig_vals : list of 1D ndarray
        Eigenvalues for each of `model.H_bond`; necessary to calculate `_U`.
    _bond_eig_vecs : list of :class:`~tenpy.linalg.np_conserved.Array`
        Eigenvectors for each of `model.H_bond`; necessary to calculate `_U`.
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

    def __init__(self, psi, model, TEBD_params):
        self.verbose = get_parameter(TEBD_params, 'verbose', 1, 'TEBD')
        self.TEBD_params = TEBD_params
        self.trunc_params = get_parameter(TEBD_params, 'trunc_params', {}, 'TEBD')
        self.trunc_params.setdefault('verbose', self.verbose / 10)  # reduced verbosity
        self.psi = psi
        self.model = model
        self.evolved_time = get_parameter(TEBD_params, 'start_time', 0., 'TEBD')
        self.trunc_err = get_parameter(TEBD_params, 'start_trunc_err', TruncationError(), 'TEBD')
        self._calc_bond_eig()  # calculates `self._bond_eig_vals`, `self._bond_eig_vecs`.
        self._U = None
        self._U_param = {}
        self._trunc_err_bonds = [TruncationError() for i in range(psi.L + 1)]
        self._update_index = None

    def __del__(self):
        unused_parameters(self.TEBD_params['trunc_params'], "TEBD trunc_params")
        unused_parameters(self.TEBD_params, "TEBD")

    @property
    def trunc_err_bonds(self):
        """truncation error introduced on each non-trivial bond"""
        return self._trunc_err_bonds[self.psi.nontrivial_bonds]

    def run(self):
        """(Real-)time evolution with TEBD (time evolving block decimation).

        The following (optional) parameters are read out from the :attr:`TEBD_params`.

        ============== ====== ======================================================
        key            type   description
        ============== ====== ======================================================
        dt             float  Time step.
        -------------- ------ ------------------------------------------------------
        order          int    Order of the algorithm.
                                The total error scales as O(t, dt^order).
        -------------- ------ ------------------------------------------------------
        N_steps        int    Number of time steps `dt` to evolve.
                              (The Trotter decompositions of order > 1 are slightly
                              more efficient if more than one step is performed at
                              once.)
        -------------- ------ ------------------------------------------------------
        trunc_params   dict   Truncation parameters as described in
                              :func:`~tenpy.algorithms.truncation.truncate`.
        ============== ====== ======================================================
        """
        # initialize parameters
        delta_t = get_parameter(self.TEBD_params, 'dt', 0.1, 'TEBD')
        N_steps = get_parameter(self.TEBD_params, 'N_steps', 10, 'TEBD')
        TrotterOrder = get_parameter(self.TEBD_params, 'order', 2, 'TEBD')

        self.calc_U(TrotterOrder, delta_t, type_evo='real', E_offset=None)

        if self.verbose >= 1:
            Sold = np.average(self.psi.entanglement_entropy())
            start_time = time.time()
        self.update(N_steps)
        if self.verbose >= 1:
            S = np.average(self.psi.entanglement_entropy())
            DeltaS = np.abs(Sold - S)
            msg = ("--> time={t:3.3f}, max_chi={chi:d}, "
                   "Delta_S={dS:.4e}, S={S:.10f}, since last update: {time:.1f} s")
            print(
                msg.format(
                    t=self.evolved_time,
                    chi=max(self.psi.chi),
                    dS=DeltaS,
                    S=S.real,
                    time=time.time() - start_time,
                ))

    def run_GS(self):
        """TEBD algorithm in imaginary time to find the ground state.

        .. note ::
            It is almost always more efficient (and hence advisable) to use DMRG.
            This algorithms can nonetheless be used quite well as a benchmark and for comparison.

        The following (optional) parameters are read out from the :attr:`TEBD_params`.
        Use ``verbose=1`` to print the used parameters during runtime.

        ============== ====== =============================================
        key            type   description
        ============== ====== =============================================
        delta_tau_list list   A list of floats: the timesteps to be used.
                              Choosing a large timestep `delta_tau`
                              introduces large (Trotter) errors, but a too
                              small time step requires a lot of steps to
                              reach  ``exp(-tau H) --> |psi0><psi0|``.
                              Therefore, we start with fairly large time
                              steps for a quick time evolution until
                              convergence, and the gradually decrease the
                              time step.
        -------------- ------ ---------------------------------------------
        order          int    Order of the Suzuki-Trotter decomposition.
        -------------- ------ ---------------------------------------------
        N_steps        int    Number of steps before measurement can be
                              performed
        -------------- ------ ---------------------------------------------
        trunc_params   dict   Truncation parameters as described in
                              :func:`~tenpy.algorithms.truncation.truncate`
        ============== ====== =============================================
        """
        # initialize parameters
        delta_tau_list = get_parameter(
            self.TEBD_params, 'delta_tau_list',
            [0.1, 0.01, 0.001, 1.e-4, 1.e-5, 1.e-6, 1.e-7, 1.e-8, 1.e-9, 1.e-10, 1.e-11, 0.],
            'run_GS')
        max_error_E = get_parameter(self.TEBD_params, 'max_error_E', 1.e-13, 'run_GS')
        N_steps = get_parameter(self.TEBD_params, 'N_steps', 10, 'run_GS')
        TrotterOrder = get_parameter(self.TEBD_params, 'order', 2, 'run_GS')

        Eold = np.average(self.model.bond_energies(self.psi))
        if self.verbose >= 1:
            Sold = np.average(self.psi.entanglement_entropy())
            start_time = time.time()

        for delta_tau in delta_tau_list:
            if self.verbose >= 1:
                print("delta_tau = {dt:e}".format(dt=delta_tau))
            self.calc_U(TrotterOrder, delta_tau, type_evo='imag')
            DeltaE = 2 * max_error_E
            DeltaS = 2 * max_error_E
            step = 0
            while (DeltaE > max_error_E):
                if self.psi.finite and TrotterOrder == 2:
                    self.update_imag(N_steps)
                else:
                    self.update(N_steps)
                step += N_steps
                E = np.average(self.model.bond_energies(self.psi))
                DeltaE = np.abs(Eold - E)
                Eold = E
                if self.verbose >= 1:
                    S = np.average(self.psi.entanglement_entropy())
                    DeltaS = np.abs(Sold - S)
                    Sold = S
                    msg = ("--> step={step:6d}, time={t:3.3f}, max chi={chi:d}, " +
                           "Delta_E={dE:.2e}, E_bond={E:.10f}, Delta_S={dS:.4e}, " +
                           "S={S:.10f}, time simulated: {time:.1f} s")
                    print(
                        msg.format(
                            step=step,
                            t=self.evolved_time,
                            chi=max(self.psi.chi),
                            dE=DeltaE,
                            dS=DeltaS,
                            E=E.real,
                            S=S.real,
                            time=time.time() - start_time,
                        ))
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
        # else
        raise ValueError("Unknown order {0!r} for Suzuki Trotter decomposition".format(order))

    @staticmethod
    def suzuki_trotter_decomposition(order, N_steps):
        r"""Returns list of necessary steps for the suzuki trotter decomposition.

        We split the Hamiltonian as :math:`H = H_{even} + H_{odd} = H[0] + H[1]`.
        The Suzuki-Trotter decomposition is an approximation
        :math:`\exp(t H) \approx prod_{(j, k) \in ST} \exp(d[j] t H[k]) + O(t^{order+1 })`.

        Parameters
        ----------
        order : int
            The desired order of the Suzuki-Trotter decomposition.

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
            # Uusing above definitions, we arrive at:
            # U = [a b a2 b c d c b a2 b a] * N
            #   = [a b a2 b c d c b a2 b] + [a2 b a2 b c d c b a2 b a] * (N-1) + [a]
            steps = [a, b, a2, b, c, d, c, b, a2, b]
            steps = steps + [a2, b, a2, b, c, d, c, b, a2, b] * (N_steps - 1)
            steps = steps + [a]
            return steps
        # else
        raise ValueError("Unknown order {0!r} for Suzuki Trotter decomposition".format(order))

    def calc_U(self, order, delta_t, type_evo='real', E_offset=None):
        """Calculate ``self.U_bond`` from ``self.bond_eig_{vals,vecs}``.

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
        if self._U_param == U_param:  # same keys and values as cached
            if self.verbose >= 10:
                print("Skip recalculation of U with same parameters as before: ", U_param)
            return  # nothing to do: U is cached
        self._U_param = U_param
        if self.verbose >= 1:
            print("Calculate U for ", U_param)

        L = self.psi.L
        self._U = []
        for dt in self.suzuki_trotter_time_steps(order):
            U_bond = [
                self._calc_U_bond(i_bond, dt * delta_t, type_evo, E_offset) for i_bond in range(L)
            ]
            self._U.append(U_bond)
        # done

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
        for U_idx_dt, odd in self.suzuki_trotter_decomposition(order, N_steps):
            trunc_err += self.update_step(U_idx_dt, odd)
        self.evolved_time = self.evolved_time + N_steps * self._U_param['tau']
        self.trunc_err = self.trunc_err + trunc_err  # not += : make a copy!
        # (this is done to avoid problems of users storing self.trunc_err after each `update`)
        return trunc_err

    def update_step(self, U_idx_dt, odd):
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
                if self.verbose >= 10:
                    print("Skip U_bond element:", i_bond)
                continue  # handles finite vs. infinite boundary conditions
            if self.verbose >= 10:
                print("Apply U_bond element", i_bond)
            self._update_index = (U_idx_dt, i_bond)
            trunc_err += self.update_bond(i_bond, Us[i_bond])
        self._update_index = None
        return trunc_err

    def update_bond(self, i, U_bond):
        """Updates the B matrices on a given bond.

        Function that updates the B matrices, the bond matrix s between and the
        bond dimension chi for bond i. The correponding tensor networks look like this::

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
        if self.verbose >= 100:
            print("Update sites ({0:d}, {1:d})".format(i0, i1))
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
                    if self.verbose >= 10:
                        print("Skip U_bond element:", i_bond)
                    continue  # handles finite vs. infinite boundary conditions
                if self.verbose >= 10:
                    print("Apply U_bond element", i_bond)
                self._update_index = (U_idx_dt, i_bond)
                trunc_err += self.update_bond_imag(i_bond, Us[i_bond])
            # sweep left
            for i_bond in range(self.psi.L - 1, -1, -1):
                if Us[i_bond] is None:
                    if self.verbose >= 10:
                        print("Skip U_bond element:", i_bond)
                    continue  # handles finite vs. infinite boundary conditions
                if self.verbose >= 10:
                    print("Apply U_bond element", i_bond)
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
        if self.verbose >= 100:
            print("Update sites ({0:d}, {1:d})".format(i0, i1))
        # Construct the theta matrix
        theta = self.psi.get_theta(i0, n=2)  # 'vL', 'vR', 'p0', 'p1'
        theta = npc.tensordot(U_bond, theta, axes=(['p0*', 'p1*'], ['p0', 'p1']))
        theta = theta.combine_legs([('vL', 'p0'), ('vR', 'p1')], qconj=[+1, -1])
        # Perform the SVD and truncate the wavefunction
        U, S, V, trunc_err, renormalize = svd_theta(theta,
                                                    self.trunc_params,
                                                    inner_labels=['vR', 'vL'])
        # Split legs and update matrices
        B_R = V.split_legs(1).ireplace_label('p1', 'p')
        A_L = U.split_legs(0).ireplace_label('p0', 'p')
        self.psi.set_SR(i0, S)
        self.psi.set_B(i0, A_L, form='A')
        self.psi.set_B(i1, B_R, form='B')
        self._trunc_err_bonds[i] = self._trunc_err_bonds[i] + trunc_err
        return trunc_err

    def _calc_bond_eig(self):
        """Calculate ``self._bond_eig_{vals,vecs}`` from ``self.model.H_bond``.

        Raises ValueError is 2-site Hamiltonian could not be diagonalized.
        """
        self._bond_eig_vals = []
        self._bond_eig_vecs = []
        for h in self.model.H_bond:
            if h is None:
                w = v = None
            else:
                H2 = h.combine_legs([('p0', 'p1'), ('p0*', 'p1*')], qconj=[+1, -1])
                w, v = npc.eigh(H2)
            self._bond_eig_vals.append(w)
            self._bond_eig_vecs.append(v)
        # done

    def _calc_U_bond(self, i_bond, dt, type_evo, E_offset):
        """Calculate exponential of a bond Hamitonian.

        * ``U_bond = exp(-i dt (H_bond-E_offset_bond))`` for ``type_evo='real'``, or
        * ``U_bond = exp(- dt H_bond)`` for ``type_evo='imag'``.
        """
        V = self._bond_eig_vecs[i_bond]
        E = self._bond_eig_vals[i_bond]
        if V is None:
            return None  # don't calculate exp(i H t), if `H` is None
        if type_evo == 'imag':
            diag = np.exp(-dt * E)
        elif type_evo == 'real':
            if E_offset is not None:
                E = E - E_offset[i_bond]
            diag = np.exp(-1.j * dt * E)
        else:
            raise ValueError("Expect either 'real' or 'imag'inary time, got " + repr(type_evo))
        # U = V s V^dag, s = e^(- tau E )
        U = V.scale_axis(diag, axis=1)
        U = npc.tensordot(U, V.conj(), axes=(1, 1))
        assert (tuple(U.get_leg_labels()) == ('(p0.p1)', '(p0*.p1*)'))
        return U.split_legs()


class RandomUnitaryEvolution(Engine):
    """Evolution of an MPS with random two-site unitaries in a TEBD-like fashion.

    Instead of using a model Hamiltonian, this TEBD engine evolves with random two-site unitaries.
    These unitaries are drawn according to the Haar measure on unitaries obeying the conservation
    laws dictated by the conserved charges. If no charge is preserved, this distribution is called
    circular unitary ensemble (CUE), see :func:`~tenpy.linalg.random_matrix.CUE`.

    On one hand, such an evolution is of interest in recent research (see eg. :arxiv:`1710.09827`).
    On the other hand, it also comes in handy to "randomize" an initial state, e.g. for DMRG.
    Note that the entanglement grows very quickly, choose the truncation paramters accordingly!

    Parameters
    ----------
    psi : :class:`~tenpy.networs.mps.MPS`
        Initial state to be time evolved. Modified in place.
    TEBD_params : dict
        Use ``verbose=1`` to print the used parameters during runtime.
        See :func:`run` and :func:`run_GS` for more details.

    Examples
    --------
    One can initialize a "random" state with total Sz = L//2 as follows:

    >>> L = 8
    >>> spin_half = SpinHalfSite(conserve='Sz')
    >>> psi = MPS.from_product_state([spin_half]*L, [0, 1]*(L//2), bc='finite')  # Neel state
    >>> print(psi.chi)
    [1, 1, 1, 1, 1, 1, 1]
    >>> TEBD_params = dict(N_steps=2, trunc_params={'chi_max':10})
    >>> eng = RandomUnitaryEvolution(psi, TEBD_params)
    >>> eng.run()
    >>> print(psi.chi)
    [2, 4, 8, 10, 8, 4, 2]
    >>> psi.canonical_form()  # necessary if you need to truncate (strongly) during the evolution

    The "random" unitaries preserve the specified charges, e.g. here we have Sz-conservation.
    If you start in a sector of all up spins, the random unitaries can only apply a phase:

    >>> psi2 = MPS.from_product_state([spin_half]*L, [0]*L, bc='finite')  # all spins up
    >>> print(psi2.chi)
    [1, 1, 1, 1, 1, 1, 1]
    >>> eng2 = RandomUnitaryEvolution(psi2, TEBD_params)
    >>> eng2.run()  # random unitaries respect Sz conservation -> we stay in all-up sector
    >>> print(psi2.chi)  # still a product state, not really random!!!
    [1, 1, 1, 1, 1, 1, 1]
    """

    def __init__(self, psi, TEBD_params):
        Engine.__init__(self, psi, None, TEBD_params)

    def run(self):
        """Time evolution with TEBD (time evolving block decimation) and random two-site unitaries.

        The following (optional) parameters are read out from the :attr:`TEBD_params`.

        ============== ====== ======================================================
        key            type   description
        ============== ====== ======================================================
        N_steps        int    Number of two-site unitaries to be applied on each
                              bond.
        -------------- ------ ------------------------------------------------------
        trunc_params   dict   Truncation parameters as described in
                              :func:`~tenpy.algorithms.truncation.truncate`
        ============== ====== ======================================================
        """
        N_steps = get_parameter(self.TEBD_params, 'N_steps', 1, 'TEBD')
        if self.verbose >= 1:
            Sold = np.average(self.psi.entanglement_entropy())
            start_time = time.time()
        self.update(N_steps)
        if self.verbose >= 1:
            S = np.average(self.psi.entanglement_entropy())
            DeltaS = np.abs(Sold - S)
            msg = ("--> time={t:3.3f}, max_chi={chi:d}, "
                   "Delta_S={dS:.4e}, S={S:.10f}, since last update: {time:.1f} s")
            print(
                msg.format(
                    t=self.evolved_time,
                    chi=max(self.psi.chi),
                    dS=DeltaS,
                    S=S.real,
                    time=time.time() - start_time,
                ))

    def calc_U(self):
        """Draw new random two-site unitaries replacing the usual `U` of TEBD."""
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
                U = npc.Array.from_func_square(CUE, pipe).split_legs()
                U.iset_leg_labels(['p0', 'p1', 'p0*', 'p1*'])
                U_bonds.append(U)
        self._U = [U_bonds]

    def update(self, N_steps):
        """Apply ``N_steps`` random two-site unitaries to each bond (in even-odd pattern).

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
        for i in range(N_steps):
            self.calc_U()  # draw new random unitaries
            for odd in [1, 0]:
                trunc_err += self.update_step(0, odd)
        self.evolved_time = self.evolved_time + N_steps
        self.trunc_err = self.trunc_err + trunc_err  # not += : make a copy!
        # (this is done to avoid problems of users storing self.trunc_err after each `update`)
        return trunc_err

    def _calc_bond_eig(self):
        pass  # do nothing
