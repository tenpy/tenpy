"""A collection of tests for :module:`tenpy.networks.uniform_mps`."""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np
from tenpy.networks import mps, site, uniform_mps


def test_to_and_from_mps():
    # create MPS, transform to uniform MPS and back to MPS
    spin_half = site.SpinHalfSite(conserve='Sz', sort_charge=False)
    psi = mps.MPS.from_product_state([spin_half] * 4, [0, 1, 0, 1], bc='infinite')
    psi.test_sanity()

    uniform_psi = uniform_mps.UniformMPS.from_MPS(psi)
    assert np.max(uniform_psi.test_sanity()) < 1e-10
    assert np.allclose(uniform_psi.expectation_value("Sz"), [0.5, -0.5, 0.5, -0.5])

    psi2 = uniform_psi.to_MPS(check_overlap=True)
    ov = psi.overlap(psi2, understood_infinite=True)
    assert (abs(ov - 1.) < 1.e-15)



