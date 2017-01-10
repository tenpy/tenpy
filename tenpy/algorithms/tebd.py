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

from ..linalg import np_conserved as npc
from ..networks import mps
from .truncation import truncate


def update_bond(psi,i,U_bond,truncation_par):
    """Updates the B matrices on a given bond.

    Function that updates the B matrices, the bond matrix s between and the bond dimension chi for bond i. This would look something like::

    |     ... - B1  -  s  -  B2 - ...
    |           |             |
    |           |-------------|
    |           |      U      |
    |           |-------------|
    |         - B1* -  s* -  B2* - ...

    Parameters
    ----------
    psi : MPS class
        The wavefunction represented in the form of an MPS
    i: int
        The bond which will be updated
    U_bond:
        The bond operator with which we update the bond
    truncation_par: dict
        The truncation parameters as explained in truncate


    Returns
    -------
    truncErr : :class:`TruncationError`
        The error of the represented state which is introduced due to the truncation during this update step.
    norm : float
        The norm of the truncated Schmidt values, ``np.linalg.norm(S[mask])``.
        Useful for re-normalization.
    """
    #TODO: Did not include the protocol distinction

    #Construct the theta matrix
    theta = psi.get_theta(i).replace_label('p0','pL').ireplace_label('p1','pR')
    theta = npc.tensordot(theta,U_bond,axes=(['pL', 'pR'], ['pL*', 'pR*']))

    #Perform the SVD and truncate the wavefunction
    theta = theta.combine_legs([('vL', 'pL'), ('vR', 'pR')], qconj=[+1, -1])
    U, S, V = npc.svd(theta, inner_labels=['vR', 'vL'])
    piv,norm,truncErr = truncate(S,truncation_par)

    #Split tensor and update matrices
    #s
    S = S[piv]
    psi.set_SR(i, S/norm)
    #B_L
    U = U.iscale_axis(S/norm, 'vR')
    B_L = U.split_legs(0).iscale_axis(psi.get_SL(i)**-1, 'vL').ireplace_label('pL', 'p')
    #B_R
    B_R = V.split_legs(1).ireplace_label('pR', 'p')
    psi.set_B(i, B_L)
    psi.set_B(i + 1, B_R)

    return truncErr,norm

def update(psi,model,N_steps,truncation_par):
    """Update a single time step with a given U

        The form of M.U depends on desired Trotter Order:

        1st order - M.U = [  [U_bond]*L ], a len1 list, whose element is list of bond ops

        2nd order - M.U = [ [U_bond]*L, [U_bond**2]*L], len2 list, whose elements are list of bond ops and bond ops squared

        4th order - M.U = [ [U_bond]*L,[U_bond]*L,[U_bond]*L, [U_bond**2]*L], len4 list, whose elements are list of bond ops and bond ops squared for the two dts needed for the 4th o. Trotter decomposion

        #TODO: more to write here!
    """

def update_step(psi,U,p,truncation_par):
    """Updates all bonds in unit cell.

        #TODO: More to write here!

    """
    


def time_evolution(psi, TEBD_params):
    """Time evolution with TEBD (time evolving block decimation).

    Parameters
    ----------
    psi : MPS
        Initial state. Modified in place.
    TEBD_parameters : dict
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
