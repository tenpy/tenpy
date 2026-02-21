"""A collection of tests for :module:`tenpy.networks.momentum_mps`."""

# Copyright (C) TeNPy Developers, Apache license
from tenpy.networks import momentum_mps, mps, site, uniform_mps


def test_MomentumMPS():
    # only very basic check, that it can be initialized.
    # TODO extend, as functionality extends

    spin_half = site.SpinHalfSite(conserve='None', sort_charge=False)
    psi = mps.MPS.from_product_state([spin_half] * 4, [0, 1, 0, 1], bc='infinite', unit_cell_width=4)
    psi.test_sanity()

    uniform_psi = uniform_mps.UniformMPS.from_MPS(psi)
    excitations = []
    for B in uniform_psi._AC:
        B = B.copy()
        B[0, 0, 0] += 1
        excitations.append(B / B.norm())
    excitation_psi = momentum_mps.MomentumMPS(excitations, uniform_psi, p=0.5)
    excitation_psi.test_sanity()
