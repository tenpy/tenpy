"""This module contains a base class for a Matrix Product State (MPS)

.. todo : funktion getSite(), getsL etc!!
.. todo : BC als Argument konsistent machen, was mit segment?!
.. todo : wie man indizes zaehlt?
.. todo : Funktion kanonisch
.. todo : Check sanity implementieren
.. todo : much much more ....
.. todo : proper documentation

"""

from __future__ import division
import numpy as np
import itertools

from ..linalg import np_conserved as npc
from ..models import lattice as lat
"""Canonical form conventions: B = s**nu[0] Gamma s**nu[1], nu[0] + nu[1] = 1 """
nu = {'A': np.array([1., 0.]), 'C': np.array([0.5, 0.5]), 'B': np.array([0., 1.])}


class MPS(object):
    """ A Matrix Product State (MPS) class that contains the sites (B) and bonds (s)

    We store in this MPS class the tensors corresponding to sites (labelled B),
    and the matrices corresponding to bonds (labelled s). Note that we have
    len(B)+1 entries for s, with the first corresponding to the bond to the left
    of the first site.

    Boundary conditions: 'periodic','finite','segment'.

    .. todo : Write more documentation.
    """

    def __init__(self, L=None, bc='finite', form='B'):
        self.B = None
        self.s = None
        self.form = nu[
            form].copy()  # self.form = [1., 0.], [0.5, 0.5], [0., 1.] for A C B respectively.

        self.bc = bc  # 'finite', 'periodic', 'segment', will be overwritten from lattice if provided

        #convenience info that can be read from B:
        self.L = L
        # self.chi = None #Do we want this, since it can change so easily
        self.d = None
        self.chinfo = None
        self.dtype = None

        #####We so far don't include the old grouped,can_keep

    def check_sanity(self):
        print "No sanity check implemented yet!! You moron..."

    def init_from_B(self):
        """Initializes L, d, (chi),chinfo, dtype from self.B
            """
        self.L = len(self.B)
        self.d = self.B[0].shape[0]
        self.dtype = self.B[0].dtype  #Necessary to "promote all others??"
        self.chinfo = self.B[0].chinfo

    @classmethod
    def product_imps(cls, d, p_state, dtype=np.float, lattice=None, form='B', charge_l=None):
        """ Construct a matrix product state from a given product state

            d: site dimension
            p_state: array which sets the product state
                if p_state[i] is an integer, then site 'i' is in state p_state[i]
                if p_state[i] is an array, then site 'i''s wavefunction is p_state[i]
            dtype: float or complex

            lattice: for charge conservation and for bc

            bc: periodic, finite, or segment

            chargeL: Bond charges at bond -1 (purely conventional) #Do we need this??

        """
        L = np.product(lattice.shape)
        if len(p_state) != L: raise ValueError("Length of p_state does not match lattice")

        psi = cls()
        psi.B = []
        psi.s = []
        psi.form = nu[form].copy()
        ci = lattice.chinfo
        psi.bc = lattice.bc_MPS

        if charge_l is None:
            charge_l = np.zeros((ci.qnumber), np.int)
        else:
            charge_l = np.array(charge_l)

        for i2 in xrange(L):
            try:
                iter(p_state[i2])
                if len(p_state[i2]) != d: raise ValueError, p_state[i2]
                B = np.array(p_state[i2], dtype).reshape((d, 1, 1))
            except TypeError:
                B = np.zeros((d, 1, 1), dtype)
                B[p_state[i2], 0, 0] = 1.0
            p_leg = lattice.site(i2).leg
            if i2 == 0:
                v_leg_left = npc.LegCharge.from_qflat(ci, charge_l)
            else:
                v_leg_left = v_leg_right
            charge_r = np.array(lattice.site(i2).leg.charges[p_state[i2]] + v_leg_left.charges[0])
            v_leg_right = npc.LegCharge.from_qflat(ci, charge_r)

            B = npc.Array.from_ndarray(B, ci, [p_leg, v_leg_left, v_leg_right.conj()])
            B.set_leg_labels(['p', 'vL', 'vR'])
            psi.B.append(B)
            s = np.ones(1, dtype=np.float)
            psi.s.append(s)
        psi.s.append(s)

        psi.init_from_B()
        psi.check_sanity()
        return psi

    # @classmethod
    # def product_af_example(cls,L,form = 'B'):
    #     psi = cls()
    #     psi.B = []
    #     psi.s = []
    #     psi.form = nu[form].copy()
    #     # create a ChargeInfo to specify the nature of the charge
    #     ci = npc.ChargeInfo([1], ['2*Sz'])
    #
    #     # create LegCharges on physical leg and even/odd bonds
    #     p_leg = npc.LegCharge.from_qflat(ci, [[1], [-1]])  # charges for up, down
    #     v_leg_even = npc.LegCharge.from_qflat(ci, [[0]])
    #     v_leg_odd = npc.LegCharge.from_qflat(ci, [[1]])
    #
    #     B_even = npc.zeros(ci, [v_leg_even, v_leg_odd.conj(), p_leg])
    #     B_odd = npc.zeros(ci, [v_leg_odd, v_leg_even.conj(), p_leg])
    #     B_even[0, 0, 0] = 1.  # up
    #     B_odd[0, 0, 1] = 1.  # down
    #
    #     for B in [B_even, B_odd]:
    #         B.set_leg_labels(['vL', 'vR', 'p'])  # virtual left/right, physical
    #
    #     psi.B = [B_even, B_odd] * (L // 2) + [B_even] * (L % 2)  # (right-canonical)
    #     psi.s = [np.ones(1)] * L
    #     return psi
