"""test of :class:`tenpy.models.XXZChain`.

.. todo ::
    more tests...
"""

from __future__ import division

import numpy as np
import numpy.testing as npt
import pprint

import tenpy.linalg.np_conserved as npc
from tenpy.models.xxz_chain import XXZChain


def test_XXZChain():
    pars = dict(L=4, Jxx=1., Jz=1., hz=0., bc_MPS='finite')
    chain = XXZChain(pars)
    chain.test_sanity()
    # check bond eigenvalues
    for Hb in chain.H_bond[1:]:
        Hb2 = Hb.combine_legs([['pL', 'pR'], ['pL*', 'pR*']], qconj=[+1, -1])
        W = npc.eigvalsh(Hb2)
        # TODO check eigenvalues for nozero hz, ...?
        npt.assert_array_almost_equal_nulp(np.sort(W), [-0.75, 0.25, 0.25, 0.25], 16**3)

    for L in [2, 3, 4, 5, 6]:
        print "L =", L
        pars['L'] = L
        pars['bc_MPS'] = 'infinite'
        chain = XXZChain(pars)
        pprint.pprint(chain.coupling_terms)
        assert len(chain.H_bond) == L
        Hb0 = chain.H_bond[0]
        for Hb in chain.H_bond[1:]:
            assert(npc.norm(Hb - Hb0, np.inf) == 0.)  # exactly equal
