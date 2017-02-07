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

site_spin_half = tenpy.networks.site.spin_half_site('Sz')
lat_spin_half = lattice.Chain(2, site_spin_half)

site_fermion = tenpy.networks.site.fermion_site('N')
lat_spin_half = lattice.Chain(5, site_fermion)


def test_CouplingModel():
    for bc in ['open', 'periodic']:
        M = model.CouplingModel(lat_spin_half, 'open')
        M.test_sanity()
