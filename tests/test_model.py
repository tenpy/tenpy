"""A collection of tests for (classes in) :module:`tenpy.models.model`.

.. todo ::
    move example sites to models.lattice
    A lot more to test: conversions
"""

from __future__ import division

import numpy as np
import numpy.testing as npt
import nose.tools as nst

import tenpy.linalg.np_conserved as npc
from tenpy.models import model, lattice

# TODO: generate these in models/lattice
leg_spin_half = npc.LegCharge.from_qflat(npc.ChargeInfo([1], ['2*Sz']), [1, -1])
Sp = [[0., 1.], [0., 0.]]
Sm = [[0., 0.], [1., 0.]]
Sz = [[0.5, 0.], [0., -0.5]]
site_spin_half = lattice.Site(leg_spin_half, ['up', 'down'], Sp=Sp, Sm=Sm, Sz=Sz)

leg_ferm = npc.LegCharge.from_qflat(npc.ChargeInfo([1], ['n']), [0, 1])
c = [[0., 1.], [0., 0.]]
cdag = [[0., 0.], [1., 0.]]
JW = [[1., 0.], [0., -1.]]
n = [[0., 0.], [0., 1.]]
site_spin_half = lattice.Site(leg_ferm, ['0', '1'], Sp=Sp, Sm=Sm, Sz=Sz)

lat_spin_half = lattice.Chain(2, site_spin_half)

def test_CouplingModel():
    M = model.CouplingModel(lat_spin_half, 'open')
    M.test_sanity()
