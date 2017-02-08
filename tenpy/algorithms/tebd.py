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


def update_bond(psi, i, U_bond, truncation_par):
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
    psi : :class:`MPS`
        The wavefunction represented in the form of an MPS.
    i: int
        Bond index; we update the matrices at sites ``i-1, i``.
    U_bond:
        The bond operator which we apply to the wave function.
    truncation_par: dict
        The truncation parameters as explained in :func:`~tenpy.algorithms.truncation.truncate`.

    Returns
    -------
    trunc_err : :class:`TruncationError`
        The error of the represented state which is introduced by the truncation
        during this update step.
    """
    i0, i1 = i - 1, i
    # Construct the theta matrix
    theta = psi.get_theta(i0, n=2)  # 'vL', 'vR', 'p0', 'p1'
    theta = npc.tensordot(U_bond, theta, axes=(['pL*', 'pR*'], ['p0', 'p1']))
    theta = theta.combine_legs([('vL', 'pL'), ('vR', 'pR')], qconj=[+1, -1])

    # Perform the SVD and truncate the wavefunction
    U, S, V, truncErr, renormalize = svd_theta(theta, truncation_par, inner_labels=['vR', 'vL'])

    # Split tensor and update matrices
    B_R = V.split_legs(1).ireplace_label('pR', 'p')
    #  U = U.iscale_axis(S, 'vR')
    #  B_L = U.split_legs(0).iscale_axis(psi.get_SL(i0)** -1, 'vL').ireplace_label('pL', 'p')
    # In general, we want to do the following:
    #     U = U.iscale_axis(S, 'vR')
    #     B_L = U.split_legs(0).iscale_axis(psi.get_SL(i0)** -1, 'vL').ireplace_label('pL', 'p')
    # i.e. with SL = psi.get_SL(i0), we have   B_L = SL**(-1) U S
    # However, the inverse of SL is problematic, as it might contain very small singular values.
    # instead, we calculate C == SL**-1 theta == SL**-1 U S V,
    # such that we obtain B_L = SL**-1 U S = SL**-1 U S V V^dagger = C V^dagger
    C = psi.get_theta(i0, n=2, formL=0.)  # same as theta, but without the `S` on the very left
    # (Note: this requires no inverse if the MPS is initiall in 'B' canonical form)
    C = npc.tensordot(U_bond, C, axes=(['pL*', 'pR*'], ['p0', 'p1']))  # apply U as for theta
    B_L = npc.tensordot(
        C.combine_legs(
            ('vR', 'pR'), pipes=theta.legs[1]), V.conj(), axes=['(vR.pR)', '(vR*.pR*)'])
    B_L.ireplace_labels(['vL*', 'pL'], ['vR', 'p'])
    B_L /= renormalize  # re-normalize to <psi|psi> = 1
    psi.set_SR(i0, S)
    psi.set_B(i0, B_L, form='B')
    psi.set_B(i1, B_R, form='B')
    return truncErr


def update_step(psi, U, p, truncation_par):
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
    psi : :class:`MPS`
        The wavefunction represented in the form of an MPS
    U: list
        The list of bond operator with which we update the MPS
    p: int
        Indication of whether to update even (p = 0) or odd (p = 1) sites
    truncation_par: dict
        The truncation parameters as explained in truncate


    Returns
    -------
    truncErr : :class:`TruncationError`
        The error of the represented state which is introduced due to the truncation
        during this sequence of update steps.
    """
    truncErr = TruncationError()
    for i_bond in np.arange(p % 2, psi.L, 2):
        if U[i_bond] is None:
            continue  # handles finite vs. infinite boundary conditions
        truncErr += update_bond(psi, i_bond, U[i_bond], truncation_par)
        # TODO : if verbose > ...
        # print "Update sites",i_bond," and ",i_bond+1
        # print "Took U_bond element",(i_bond+1)
    return truncErr


def update(psi, model, N_steps, truncation_par):
    """Update a single time step with a given U

    The form of model.U_bond depends on desired Trotter Order:

    1st order - model.U_bond = [  [U_bond]*L ], a len1 list, whose element is list of
    bond ops

    2nd order - model.U_bond = [ [U_bond]*L, [U_bond**2]*L], len2 list, whose elements
    are list of bond ops and bond ops squared

    4th order - model.U_bond = [ [U_bond]*L,[U_bond]*L,[U_bond]*L, [U_bond**2]*L],
    len4 list, whose elements are list of bond ops and bond ops squared for the
    two dts needed for the 4th o. Trotter decomposion

    Parameters
    ----------
    psi : :class:`MPS`
        The wavefunction represented in the form of an MPS
    model: :class:`NearestNeighborModel`
        The model from which the bond operators are taken
    U_bond:
        The bond operator with which we update the bond
    truncation_par: dict
        The truncation parameters as explained in truncate

    Returns
    -------
    truncErr : :class:`TruncationError`
        The error of the represented state which is introduced due to the truncation during
        this sequence of update steps.
    """
    truncErr = TruncationError()
    # TODO: for p, U_list in enumerate(model.U_bond):
    if len(model.U_bond) == 1:  #First Order Trotter
        for i_step in xrange(N_steps):
            for p in xrange(2):
                truncErr += update_step(psi, model.U_bond[0], p, truncation_par)

    elif len(model.U_bond) == 2:  #Second Order Trotter
        # Scheme: [a 2b a]*N = a 2b [2a 2b]*(N-1) a

        # model.U_bond[0] a  : exp( H t1 / 2   )
        # model.U_bond[1] b  : exp( H t1    )

        truncErr = update_step(psi, model.U_bond[0], 0, truncation_par)
        truncErr = update_step(psi, model.U_bond[1], 1, truncation_par)

        for i_step in xrange(N_steps - 1):
            for p in xrange(2):
                truncErr = update_step(psi, model.U_bond[1], p, truncation_par)
        truncErr = update_step(psi, model.U_bond[0], 0, truncation_par)

    elif len(model.U_bond) == 4: #Fourth Order Trotter
        #Scheme: [ a b 2a b c d c b 2a b a] * N =
        #  a b 2a b c d c b 2a b [ 2a b 2a b c d c b 2a b ] * (N-1) a

        # model.U_bond[0] a  : exp(  Hodd t1 / 2   )
        # model.U_bond[1] 2a : exp(  Hodd t1    )
        # model.U_bond[1] b  : exp(  Heven t1    )
        # model.U_bond[2] c  : exp(  Hodd (t3+t1)/2  )
        # model.U_bond[3] d  : exp(  Heven t3  )

        truncErr = update_step(psi, model.U_bond[0], 0, truncation_par)
        for p in xrange(3):
            truncErr = update_step(psi, model.U_bond[1], np.mod(p+1,2), truncation_par)

        truncErr = update_step(psi, model.U_bond[2], 0, truncation_par)
        truncErr = update_step(psi, model.U_bond[3], 1, truncation_par)
        truncErr = update_step(psi, model.U_bond[2], 0, truncation_par)

        for p in xrange(3):
            truncErr = update_step(psi, model.U_bond[1], np.mod(p+1,2), truncation_par)

        for i_step in xrange(N_steps - 1):
            for p in xrange(4):
                truncErr = update_step(psi, model.U_bond[1], np.mod(p,2), truncation_par)

            truncErr = update_step(psi, model.U_bond[2], 0, truncation_par)
            truncErr = update_step(psi, model.U_bond[3], 1, truncation_par)
            truncErr = update_step(psi, model.U_bond[2], 0, truncation_par)

            for p in xrange(3):
                truncErr = update_step(psi, model.U_bond[1], np.mod(p+1,2), truncation_par)

        truncErr = update_step(psi, model.U_bond[0], 0, truncation_par)


def ground_state(psi, model, TEBD_par):
    """TEBD algorithm in imaginary time to find the ground state.

    An algorithm that finds the ground state and its corresponding energy by
    imaginary time TEBD.

    Note that it is almost always more efficient (and hence advisable) to use
    DMRG. It can nonetheless be used quite well as a benchmark!

    Parameters
    ----------
    psi : :class:`MPS`
        The wavefunction represented in the form of an MPS. Modified in place.
    model: :class:`NearestNeighborModel`
        The model from which the bond operators are taken
    TEBD_par: dict
        Some parameters, #TODO: decide later how to connect to time evo.


    Returns
    -------

    """
    delta_tau_list = get_parameter(TEBD_par, 'delta_tau_list',
                                   [0.1, 0.01, 0.001, 1.e-4, 1.e-5, 1.e-6, 1.e-7], 'imag. time GS')
    max_error_E = get_parameter(TEBD_par, 'max_error_E', 1.e-12, 'imag. time GS')

    N_steps = get_parameter(TEBD_par, 'N_steps', 10, 'imag. time GS')

    # Take away for now and directly pass TEBD_par
    # truncation_par = {'chi_max': TEBD_par['chi_max'],
    #                   'chi_min': TEBD_par['chi_min'],
    #                   'symmetry_tol': TEBD_par['symmetry_tol'],
    #                   'svd_min': TEBD_par['svd_min'],
    #                   'trunc_cut': TEBD_par['trunc_cut']}
    # TODO: N_STEPS, verbose etc.
    for delta_t in delta_tau_list:
        model.calc_U(TEBD_par,type_evo = 'IMAG')
        DeltaE = 2 * TEBD_par['max_error_E']
        DeltaS = 2 * TEBD_par['max_error_E']

        Eold = np.average(model.bond_energies(psi))
        #TODO: what if different leg order?
        # Sold= np.average(psi.entanglement_entropy())
        step = 1
        while (DeltaE > max_error_E):
            update(psi, model, N_steps, TEBD_par)
            E = np.average(model.bond_energies(psi))
            # S = np.average(psi.entanglement_entropy())
            DeltaE = np.abs(Eold - E)
            # DeltaS=np.abs(Sold-S)
            Eold = E
            # Sold=S

            step += N_steps

            if True:  #TEBD_par['VERBOSE']:#TODO: How to incorporate
                print "--> step = %6.0f " % (step - 1),
                # print " Chi ", psi.chi,
                print " Delta_tau = %.10f " % delta_t,
                print " Delta_E = %.10f " % DeltaE,
                print " Ebond = %.10f " % E.real
                # print " Delta_S = %.10f " %DeltaS,
                # print " Sbond = %.10f "  %S.real


def time_evolution(psi, TEBD_par):
    """Time evolution with TEBD (time evolving block decimation).

    Parameters
    ----------
    psi : MPS
        Initial state. Modified in place.
    TEBD_par : dict
        Further optional parameters as described in the following table.
        Use ``verbose=1`` to print the used parameters during runtime.

        ======= ====== ==============================================
        key     type   description
        ======= ====== ==============================================
        dt      float  time step.
        ------- ------ ----------------------------------------------
        order   int    Order of the algorithm.
                       The total error scales as O(t, dt^order).
        ------- ------ ----------------------------------------------
        type    string Imaginary or real time evolution (IMAG,REAL)
        ------- ------ ----------------------------------------------
        ...            Truncation parameters as described in
                       :func:`~tenpy.algorithms.truncation.truncate`
        ======= ====== ==============================================
    """
