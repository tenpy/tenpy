r"""Time evolving block decimation (TEBD).


The TEBD algorithm (proposed in [1]_) uses a trotter decomposition of the
Hamiltonian to perform a time evoltion of an MPS. It works only for nearest-neigbor hamiltonians
(in tenpy given by a :class:`~tenpy.models.NearestNeighborModel`),
which can be written as :math:`H = H^{even} + H^{odd}`,  such that :math:`H^{even}` contains the
the terms on even bonds (and similar :math:`H^{odd}` the terms on odd bonds).
In the simplest case, we apply first :math:`U=\exp(-i*dt*H^{even})`,
then :math:`U=\exp(-i*dt*H^{odd})` for each time step :math:`dt`.
This is correct up to errors of :math:`O(dt^2)`, but to evolve until a time :math:`T`, we need
:math:`T/dt` steps, so in total it is only correct up to error of :math:`O(T*dt)`.
Similarly, there are higher order schemata (in dt).

Remember, that bond `i` is between sites `(i-1, i)`, so for a finite MPS it looks like::

    |     - B0 - B1 - B2 - B3 - B4 - B5 - B6 -
    |       |    |    |    |    |    |    |
    |       |    |----|    |----|    |----|
    |       |    | U2 |    | U4 |    | U6 |
    |       |    |----|    |----|    |----|
    |       |----|    |----|    |----|    |
    |       | U1 |    | U3 |    | U5 |    |
    |       |----|    |----|    |----|    |
    |                   .
    |                   .
    |                   .

After each application of a `Ui`, the MPS needs to be truncated - otherwise the bond dimension
`chi` would grow indefinitely. A bound for the error introduced by the truncation is returned.

If one chooses imaginary :math:`dt`, the exponential projects
(for sufficiently long 'time' evolution) onto the ground state of the Hamiltonian.

.. Note ::
    The application of DMRG is typically much more efficient than imaginary TEBD!
    Yet, imaginary TEBD might be usefull for cross-checks and testing.

References
----------
.. [1] G. Vidal, Phys. Rev. Lett. 93, 040502 (2004), arXiv:quant-ph/0310089
"""

from __future__ import division
import numpy as np
import copy

from ..linalg import np_conserved as npc
from .truncation import svd_theta, TruncationError
from ..tools.params import get_parameter

class Engine(object):
    """TEBD (time evolving block decimation) 'engine'.

    Parameters
    ----------
    psi : MPS
        Initial state. Modified in place.
    model : :class:`~tenpy.models.MPOModel`
        The model representing the Hamiltonian for which we want to find the ground state.
    TEBD_params : dict
        Further optional parameters as described in the following table.
        Use ``verbose=1`` to print the used parameters during runtime.
        See :func:`run` and :func:`run_GS` for more details.

    Attributes
    ----------
    verbose : int
        Level of verbosity (i.e. how much status information to
        print); higher=more output.
    U_bond : list
        A list of exponentiated H_bond (bond Hamiltonian), i.e.
        exp[-i H_bond dt], with appropriately chosen dt
    U_param : dict
        A dictionary containing the information of the latest created U_bond
    bond_eig_vals : list
        A list of eigenvalues of H_bond[i]
    bond_eig_vecs : list
        A list of eigenvectors of H_bond[i]
    real_time : float
        A float indicating how much the the wavefunction has been evolved in
        real_time = dt*(TEBD_steps)
    psi: MPS
        see parameters
    model: :class:`~tenpy.models.MPOModel`
        see parameters
    H_bond: list
        From model.H_bond (for convenience)
    TEBD_params: dict
        see parameters

    """
    def __init__(self, psi, model, TEBD_params):
        self.verbose = get_parameter(TEBD_params,'verbose',2,'TEBD')
        self.psi = psi
        self.model = model
        self.H_bond = model.H_bond
        self.calc_bond_eig()
        self.U_bond = None
        self.U_param = None
        self.TEBD_params = TEBD_params
        self.real_time = None

    def run(self):
        """Time evolution with TEBD (time evolving block decimation).

        Parameters
        ----------
        TEBD_params : dict
            The optional parameters that are used are described in the
            following table.
            Use ``verbose=1`` to print the used parameters during runtime.

            ======= ====== ==============================================
            key     type   description
            ======= ====== ==============================================
            dt      float  time step.
            ------- ------ ----------------------------------------------
            order   int    Order of the algorithm.
                           The total error scales as O(t, dt^order).
            ------- ------ ----------------------------------------------
            N_steps int    Number of steps before measurement can be performed,
                           number of steps that are interlinked for all
                           Trotter decompositions of order > 1.
            ------- ------ ----------------------------------------------
            ...            Truncation parameters as described in
                           :func:`~tenpy.algorithms.truncation.truncate`
            ======= ====== ==============================================

        Returns
        -------
        """
        # initialize parameters
        delta_t = get_parameter(self.TEBD_params, 'dt',0.1, 'run')
        N_steps = get_parameter(self.TEBD_params, 'N_steps', 10, 'run')
        TrotterOrder = get_parameter(self.TEBD_params, 'order', 2, 'run')

        U_param = {'dt': delta_t, 'type_evo': 'REAL', 'order': TrotterOrder}
        if set(self.U_param.items()) != set(U_param.items()):
            self.calc_U(TrotterOrder, delta_t, type_evo = 'REAL')

        Eold = np.average(self.model.bond_energies(self.psi))
        Sold = np.average(self.psi.entanglement_entropy())
        if self.real_time is None:
            self.real_time = 0.


        self.update(N_steps)
        E = np.average(self.model.bond_energies(self.psi))
        S = np.average(self.psi.entanglement_entropy())
        DeltaE = np.abs(Eold - E)
        DeltaS = np.abs(Sold - S)
        Eold = E
        Sold = S
        self.real_time += N_steps*delta_t

        if self.verbose > 1:
            print "--> time = %6.0f " % (self.real_time),
            # print " Chi ", psi.chi,
            print " Delta_tau = %.5f " % delta_t,
            print " Delta_E = %.10f " % DeltaE,
            print " Ebond = %.10f " % E.real,
            print " Delta_S = %.10f " %DeltaS,
            print " Sbond = %.10f "  %S.real



    def run_GS(self):
        """TEBD algorithm in imaginary time to find the ground state.

        An algorithm that finds the ground state and its corresponding energy by
        imaginary time TEBD.

        Note that it is almost always more efficient (and hence advisable) to use
        DMRG. It can nonetheless be used quite well as a benchmark!

        Parameters
        ----------
        TEBD_params : dict
            The optional parameters that are used are described in the
            following table.
            Use ``verbose=1`` to print the used parameters during runtime.

            ============== ====== =============================================
            key            type   description
            ============== ====== =============================================
            delta_tau_list list   A list of floats describing
            -------------- ------ ---------------------------------------------
            order          int    Order of the algorithm.
                                  The total error scales as O(t, dt^order).
            -------------- ------ ---------------------------------------------
            N_steps        int    Number of steps before measurement can be
                                  performed
            -------------- ------ ---------------------------------------------
            ...                   Truncation parameters as described in
                                  :func:`~tenpy.algorithms.truncation.truncate`
            ============== ====== =============================================

        Returns
        -------

        """

        # initialize parameters
        delta_tau_list = get_parameter(self.TEBD_params, 'delta_tau_list',
                                       [0.1, 0.01, 0.001, 1.e-4, 1.e-5, 1.e-6,
                                        1.e-7,1.e-8,1.e-9,1.e-10,1.e-11, 0.],
                                         'run_GS')
        max_error_E = get_parameter(self.TEBD_params, 'max_error_E', 1.e-13, 'run_GS')
        N_steps = get_parameter(self.TEBD_params, 'N_steps', 10, 'run_GS')
        TrotterOrder = get_parameter(self.TEBD_params, 'order', 2, 'run_GS')

        for delta_t in delta_tau_list:
            self.calc_U(TrotterOrder, delta_t, type_evo = 'IMAG')
            DeltaE = 2 * max_error_E
            DeltaS = 2 * max_error_E

            Eold = np.average(self.model.bond_energies(self.psi))
            Sold = np.average(self.psi.entanglement_entropy())
            step = 1
            while (DeltaE > max_error_E):
                self.update(N_steps)
                E = np.average(self.model.bond_energies(self.psi))
                S = np.average(self.psi.entanglement_entropy())
                DeltaE = np.abs(Eold - E)
                DeltaS = np.abs(Sold - S)
                Eold = E
                Sold = S

                step += N_steps

                if self.verbose > 1:
                    print "--> step = %6.0f " % (step - 1),
                    # print " Chi ", psi.chi,
                    print " Delta_tau = %.10f " % delta_t,
                    print " Delta_E = %.10f " % DeltaE,
                    print " Ebond = %.10f " % E.real,
                    print " Delta_S = %.10f " %DeltaS,
                    print " Sbond = %.10f "  %S.real

    def calc_bond_eig(self):
        """Calculate ``self.bond_eig_{vals,vecs}`` from ``self.H_bond``.

        Raises ValueError is 2-site Hamiltonian could not be diagonalized.

        Parameters
        ----------

        Returns
        -------
        """
        self.bond_eig_vals = []
        self.bond_eig_vecs = []
        for h in self.H_bond:
            #TODO: Check if hermitian?!
            if h is None:
                w = v = None
            else:
                H2 = h.combine_legs([('pL', 'pR'), ('pL*', 'pR*')], qconj=[+1, -1])
                w, v = npc.eigh(H2)
            self.bond_eig_vals.append(w)
            self.bond_eig_vecs.append(v)
        # done

    def calc_U(self, order, delta_t,type_evo):
        """Calculate ``self.U_bond`` from ``self.bond_eig_{vals,vecs}``

        Parameters
        ----------
        order : int
            Trotter order calculated U_bond. See update for more information.
        delta_t: float
            Size of the time-step used in calculating U_bond
        type_evo: string
            Has to be 'IMAG' or 'REAL' and determines whether we choose real or
            imaginary time-evolution

        Returns
        -------
        """

        #TODO: Old TenPy has E_offset
        if order == 1:
            self.U_bond = [[None] * len(self.H_bond)]
            for i_bond in range(len(self.H_bond)):
                dt = delta_t
                if self.bond_eig_vals[i_bond] is not None:
                    if (type_evo == 'IMAG'):
                        s = np.exp(-dt * self.bond_eig_vals[i_bond])
                    elif (type_evo == 'REAL'):
                        s = np.exp(-1j * dt * (self.bond_eig_vals[i_bond]))
                    else:
                        raise ValueError(
                            "Need to have either real time (REAL) or imaginary time (IMAG)")
                    V = self.bond_eig_vecs[i_bond]
                    # U = V s V^dag, s = e^(- tau E )
                    U = V.scale_axis(s, axis=1)
                    U = npc.tensordot(U, V.conj(), axes=(1, 1))
                    labels = tuple(('(pL.pR)','(pL*.pR*)'))
                    U.set_leg_labels(labels)
                    self.U_bond[0][i_bond] = U.split_legs()
        elif order == 2:
            self.U_bond = [[None] * len(self.H_bond), [None] * len(self.H_bond)]
            for i_bond in range(len(self.H_bond)):
                dt = delta_t/2.
                if self.bond_eig_vals[i_bond] is not None:
                    if (type_evo == 'IMAG'):
                        s = np.exp(-dt * self.bond_eig_vals[i_bond])
                    elif (type_evo == 'REAL'):
                        s = np.exp(-1j * dt * (self.bond_eig_vals[i_bond]))
                    else:
                        raise ValueError(
                            "Need to have either real time (REAL) or imaginary time (IMAG)")
                    V = self.bond_eig_vecs[i_bond]

                    # U = V s V^dag, s = e^(- tau E )

                    U = V.scale_axis(s, axis=1)
                    U = npc.tensordot(U, V.conj(), axes=(1, 1))
                    labels = tuple(('(pL.pR)','(pL*.pR*)'))
                    U.set_leg_labels(labels)
                    self.U_bond[0][i_bond] = U.split_legs()

                    s = s*s
                    U = V.scale_axis(s, axis=1)
                    U = npc.tensordot(U, V.conj(), axes=(1, 1))
                    labels = tuple(('(pL.pR)','(pL*.pR*)'))
                    U.set_leg_labels(labels)
                    self.U_bond[1][i_bond] = U.split_legs()
        elif order == 4:
            # adapted from Schollwock2011 notation ...
            # U operators in the following order :
            # a  : exp( Hodd t1 / 2   )
            # 2a : exp( Heven t1    )
            # b  : exp(  Hodd t1    )
            # c  : exp(  Hodd (t-3*t1)/2  )
            # d  : exp(  Hodd t3  )
            # 2a and b use the same slot!
            self.U_bond = [[None] * len(self.H_bond),
                [None] * len(self.H_bond),[None] * len(self.H_bond),
                [None] * len(self.H_bond)]

            for i_bond in range(len(self.H_bond)):
                dt1 = 1. / (4. - 4.**(1/3) ) * delta_t /2.
                dt3 = delta_t - 4* (dt1*2)
                if self.bond_eig_vals[i_bond] is not None:
                    if (type_evo == 'IMAG'):
                        s1 = np.exp(-dt1 * self.bond_eig_vals[i_bond])
                        s13 = np.exp(- ( dt3 + 2*dt1)/2. * self.bond_eig_vals[i_bond])
                        s3 = np.exp(-dt3 * self.bond_eig_vals[i_bond])
                    elif (type_evo == 'REAL'):
                        s1 = np.exp(-1j * dt1 * self.bond_eig_vals[i_bond])
                        s13 = np.exp(-1j * ( dt3 + 2*dt1)/2. * self.bond_eig_vals[i_bond])
                        s3 = np.exp(-1j *dt3 * self.bond_eig_vals[i_bond])
                    else:
                        raise ValueError(
                            "Need to have either real time (REAL) or imaginary time (IMAG)")
                    V = self.bond_eig_vecs[i_bond]

                    # U = V s V^dag, s = e^(- tau E )

                    # a
                    U = V.scale_axis(s1, axis=1)
                    U = npc.tensordot(U, V.conj(), axes=(1, 1))
                    labels = tuple(('(pL.pR)','(pL*.pR*)'))
                    U.set_leg_labels(labels)
                    self.U_bond[0][i_bond] = U.split_legs()

                    # 2a
                    s1 = s1*s1
                    U = V.scale_axis(s1, axis=1)
                    U = npc.tensordot(U, V.conj(), axes=(1, 1))
                    labels = tuple(('(pL.pR)','(pL*.pR*)'))
                    U.set_leg_labels(labels)
                    self.U_bond[1][i_bond] = U.split_legs()

                    # c
                    U = V.scale_axis(s13, axis=1)
                    U = npc.tensordot(U, V.conj(), axes=(1, 1))
                    labels = tuple(('(pL.pR)','(pL*.pR*)'))
                    U.set_leg_labels(labels)
                    self.U_bond[2][i_bond] = U.split_legs()

                    # d
                    U = V.scale_axis(s3, axis=1)
                    U = npc.tensordot(U, V.conj(), axes=(1, 1))
                    labels = tuple(('(pL.pR)','(pL*.pR*)'))
                    U.set_leg_labels(labels)
                    self.U_bond[3][i_bond] = U.split_legs()
        else:
            raise NotImplementedError('Only 4th order Trotter has been implemented')

        self.U_param = {'dt': delta_t, 'type_evo': type_evo
            , 'order': order}
        if self.verbose > 1:
            print "Calculated U_bond for:",self.U_param

    def update(self, N_steps):
        """Update a single time step with a given U

        The form of self.U_bond depends on desired Trotter Order:

        1st order - self.U_bond = [  [U_bond]*L ], a len1 list, whose element is list of
        bond ops

        2nd order - self.U_bond = [ [U_bond]*L, [U_bond**2]*L], len2 list, whose elements
        are list of bond ops and bond ops squared

        4th order - self.U_bond = [ [U_bond]*L,[U_bond]*L,[U_bond]*L, [U_bond**2]*L],
        len4 list, whose elements are list of bond ops and bond ops squared for the
        two dts needed for the 4th o. Trotter decomposion

        Parameters
        ----------
        N_steps: int
            The number of steps for which the whole lattice should be updated

        Returns
        -------
        truncErr : :class:`TruncationError`
            The error of the represented state which is introduced due to the truncation during
            this sequence of update steps.
        """
        truncErr = TruncationError()
        # TODO: for p, U_list in enumerate(self.U_bond):
        if len(self.U_bond) == 1:  #First Order Trotter
            for i_step in xrange(N_steps):
                for p in xrange(2):
                    truncErr += self.update_step(self.U_bond[0], p)

        elif len(self.U_bond) == 2:  #Second Order Trotter
            # Scheme: [a 2b a]*N = a 2b [2a 2b]*(N-1) a

            # self.U_bond[0] a  : exp( H t1 / 2   )
            # self.U_bond[1] b  : exp( H t1    )

            truncErr = self.update_step(self.U_bond[0], 0)
            truncErr = self.update_step(self.U_bond[1], 1)

            for i_step in xrange(N_steps - 1):
                for p in xrange(2):
                    truncErr = self.update_step(self.U_bond[1], p)
            truncErr = self.update_step(self.U_bond[0], 0)

        elif len(self.U_bond) == 4: #Fourth Order Trotter
            #Scheme: [ a b 2a b c d c b 2a b a] * N =
            #  a b 2a b c d c b 2a b [ 2a b 2a b c d c b 2a b ] * (N-1) a

            # self.U_bond[0] a  : exp(  Hodd t1 / 2   )
            # self.U_bond[1] 2a : exp(  Hodd t1    )
            # self.U_bond[1] b  : exp(  Heven t1    )
            # self.U_bond[2] c  : exp(  Hodd (t3+t1)/2  )
            # self.U_bond[3] d  : exp(  Heven t3  )

            truncErr = self.update_step(self.U_bond[0], 0)
            for p in xrange(3):
                truncErr = self.update_step(self.U_bond[1], np.mod(p+1,2))

            truncErr = self.update_step(self.U_bond[2], 0)
            truncErr = self.update_step(self.U_bond[3], 1)
            truncErr = self.update_step(self.U_bond[2], 0)

            for p in xrange(3):
                truncErr = self.update_step(self.U_bond[1], np.mod(p+1,2))

            for i_step in xrange(N_steps - 1):
                for p in xrange(4):
                    truncErr = self.update_step(self.U_bond[1], np.mod(p,2))

                truncErr = self.update_step(self.U_bond[2], 0)
                truncErr = self.update_step(self.U_bond[3], 1)
                truncErr = self.update_step(self.U_bond[2], 0)

                for p in xrange(3):
                    truncErr = self.update_step(self.U_bond[1], np.mod(p+1,2))

            truncErr = self.update_step(self.U_bond[0], 0)

    def update_step(self, U, p):
        """Updates all even OR odd bonds in unit cell.

        Depending on the choice of p, this function updates all even (E) (p = 0)
        OR odd (O) (p=1) bonds::

        |     - B0 - B1 - B2 - B3 - B4 - B5 - B6 -
        |       |    |    |    |    |    |    |
        |       |    |----|    |----|    |----|
        |       |    |  E |    |  E |    |  E |
        |       |    |----|    |----|    |----|
        |       |----|    |----|    |----|    |
        |       |  O |    |  O |    |  O |    |
        |       |----|    |----|    |----|    |

        Note that boundary conditions are taken care of by having U[0] = None
        or otherwise.

        Parameters
        ----------
        U: list
            The list of bond operator with which we update the MPS
        p: int
            Indication of whether to update even (p = 0) or odd (p = 1) sites

        Returns
        -------
        truncErr : :class:`TruncationError`
            The error of the represented state which is introduced due to the truncation
            during this sequence of update steps.
        """
        truncErr = TruncationError()
        for i_bond in np.arange(p % 2, self.psi.L, 2):
            if U[i_bond] is None:
                if self.verbose > 10:
                    print "Skipped U_bond element:",i_bond
                continue  # handles finite vs. infinite boundary conditions
            truncErr += self.update_bond(i_bond, U[i_bond])
            if self.verbose/100 > 1:
                print "Took U_bond element",(i_bond)
        return truncErr

    def update_bond(self, i, U_bond):
        """Updates the B matrices on a given bond.

        Function that updates the B matrices, the bond matrix s between and the
        bond dimension chi for bond i. This would look something like::

        |     ... - B1  -  s  -  B2 - ...
        |           |             |
        |           |-------------|
        |           |      U      |
        |           |-------------|
        |           |             |

        Parameters
        ----------
        i: int
            Bond index; we update the matrices at sites ``i-1, i``.
        U_bond:
            The bond operator which we apply to the wave function.

        Returns
        -------
        trunc_err : :class:`TruncationError`
            The error of the represented state which is introduced by the truncation
            during this update step.
        """
        i0, i1 = i - 1, i
        # Construct the theta matrix
        theta = self.psi.get_theta(i0, n=2)  # 'vL', 'vR', 'p0', 'p1'
        theta = npc.tensordot(U_bond, theta, axes=(['pL*', 'pR*'], ['p0', 'p1']))
        theta = theta.combine_legs([('vL', 'pL'), ('vR', 'pR')], qconj=[+1, -1])

        # Perform the SVD and truncate the wavefunction
        U, S, V, truncErr, renormalize = svd_theta(theta, self.TEBD_params, inner_labels=['vR', 'vL'])

        # Split tensor and update matrices
        B_R = V.split_legs(1).ireplace_label('pR', 'p')
        #  U = U.iscale_axis(S, 'vR')
        #  B_L = U.split_legs(0).iscale_axis(self.psi.get_SL(i0)** -1, 'vL').ireplace_label('pL', 'p')
        # In general, we want to do the following:
        #     U = U.iscale_axis(S, 'vR')
        #     B_L = U.split_legs(0).iscale_axis(self.psi.get_SL(i0)** -1, 'vL').ireplace_label('pL', 'p')
        # i.e. with SL = self.psi.get_SL(i0), we have   B_L = SL**(-1) U S
        # However, the inverse of SL is problematic, as it might contain very small singular values.
        # instead, we calculate C == SL**-1 theta == SL**-1 U S V,
        # such that we obtain B_L = SL**-1 U S = SL**-1 U S V V^dagger = C V^dagger
        C = self.psi.get_theta(i0, n=2, formL=0.)  # same as theta, but without the `S` on the very left
        # (Note: this requires no inverse if the MPS is initiall in 'B' canonical form)
        C = npc.tensordot(U_bond, C, axes=(['pL*', 'pR*'], ['p0', 'p1']))  # apply U as for theta
        B_L = npc.tensordot(
            C.combine_legs(
                ('vR', 'pR'), pipes=theta.legs[1]), V.conj(), axes=['(vR.pR)', '(vR*.pR*)'])
        B_L.ireplace_labels(['vL*', 'pL'], ['vR', 'p'])
        B_L /= renormalize  # re-normalize to <psi|psi> = 1
        self.psi.set_SR(i0, S)
        self.psi.set_B(i0, B_L, form='B')
        self.psi.set_B(i1, B_R, form='B')

        if self.verbose/100 > 1:
            print "Update sites",i0," and ",i1

        return truncErr
