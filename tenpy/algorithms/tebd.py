from __future__ import division
import numpy as np

from ..linalg import np_conserved as npc
from ..networks import MPS


def time_evolution(psi, TEBD_params):
    """time evolution with TEBD.

    Parameters
    ----------
    psi : MPS
        Initial state. Modified in place.
    TEBD_parameters : dict
        Further parameters as described in the following table.
        Use ``verbose=1`` to print the used parameters during runtime.

        ======= ====== ==============================================
        key     type   description
        ======= ====== ==============================================
        dt      float  time step.
        ------- ------ ----------------------------------------------
        order   int    Order of the algorithm.
                       The total error scales as O(dt^order).
        ------- ------ ----------------------------------------------
        type    string Imaginary or real time evolution (IMAG,REAL)
        ------- ------ ----------------------------------------------
        ...            Truncation parameters as described in
                       :func:`~tenpy.algorithms.truncation.truncate`
        ======= ====== ==============================================
    """
