"""A collection of tests for (classes in) :mod:`tenpy.models.model`.

.. todo ::
    A lot more to test, e.g. conversions of the different models
"""

from __future__ import division

import numpy as np
import numpy.testing as npt
import nose.tools as nst

import tenpy.linalg.np_conserved as npc
from tenpy.models import model, lattice
import tenpy.networks.site

spin_half_site = tenpy.networks.site.SpinHalfSite('Sz')
spin_half_lat = lattice.Chain(2, spin_half_site)

fermion_site = tenpy.networks.site.FermionSite('N')
fermion_lat = lattice.Chain(5, fermion_site)


def test_CouplingModel():
    for bc in ['open', 'periodic']:
        M = model.CouplingModel(spin_half_lat, bc)
        M.test_sanity()
        M = model.CouplingModel(fermion_lat, bc)
        M.test_sanity()
