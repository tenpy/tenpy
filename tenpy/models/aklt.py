"""Prototypical example of a 1D quantum model constructed from bond terms: the AKLT chain.

The AKLT model is famous for having a very simple ground state MPS of bond dimension 2.
Writing down the Hamiltonian is easiest done in terms of bond couplings.
This class thus serves as an example how this can be done.
"""

# Copyright (C) TeNPy Developers, Apache license
from typing import Literal

import lasyten as lasy
import numpy as np

from ..networks.mps import MPS
from .lattice import Chain
from .model import ModelBuilder

__all__ = ['AKLTChain']


class AKLTChain(ModelBuilder):
    r"""A simple implementation of the AKLT model.

    Here we define the Hamiltonian on a chain of S=1 spins as originally defined by
    Affleck, Kennedy, Lieb, Tasaki in :cite:`affleck1987`, but
    dropping the constant parts of 1/3 per bond and rescaling with a factor of 2,
    such that we expect a ground state energy of ``E_0 = - (L-1) 2/3 * J``.

    .. math ::
        H = J \sum_i 2* P^{S=2}_{i,i+1} + const
          = J \sum_i (\vec{S}_i \cdot \vec{S}_{i+1}
                     +\frac{1}{3} (\vec{S}_i \cdot \vec{S}_{i+1})^2)
    """

    default_lattice = Chain
    force_default_lattice = True
    conserve: Literal['SU2', 'Sz', 'parity', None]

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'best', str)
        if conserve == 'best':
            conserve = 'SU2'
        assert conserve in ['SU2', 'Sz', 'parity', 'None', None]
        self.conserve = conserve
        return lasy.sites.SpinSite(S=1, conserve=conserve, backend=self.backend)

    def init_couplings(self, model_params):
        site, *_ = self.lattice.unit_cell
        self.define_coupling(lasy.models.aklt_coupling([site, site], name='AKLT'), hc='AKLT', permute='invariant')

    def init_terms(self, model_params):
        J = model_params.get('J', 1.0, 'real_or_array')
        self.add_two_site_coupling(J, 'AKLT', dx=1)

    def psi_AKLT(self):
        """Initialize the chi=2 MPS which is exact ground state of the AKLT model.

        Only implemented for ``conserve='SU2'`` and infinite boundary conditions.

        The AKLT state is built from a spin-1/2 singlet on each bond and a projector from two
        spin-1/2 to a spin-1 on each site. Hence, all bonds carry a single spin-1/2 multiplet.
        With SU(2) symmetry, the (right canonical) tensor with this virtual space is unique up to
        a gauge transformation, such that a random symmetric tensor with spin-1/2 bonds is the
        AKLT state.

        Returns
        -------
        psi_aklt : :class:`~tenpy.networks.mps.MPS`
            The AKLT groundstate

        Raises
        ------
        NotImplementedError
            For all other symmetries and for finite boundary conditions.
            A finite SU(2) symmetric chain has no trivial virtual spaces at the ends
            (spin-1/2 times spin-1 contains no singlet), so the bond spaces would have to be
            chosen differently and the tensors constructed explicitly.

        """
        if self.conserve != 'SU2' or self.lattice.bc_MPS != 'infinite':
            raise NotImplementedError('psi_AKLT is only ported for conserve="SU2" and bc_MPS="infinite"')
        sites = self.lattice.mps_sites()
        sym = sites[0].symmetry
        # the spin-1/2 sector. TODO: use named sectors once https://github.com/tenpy/lasyten/issues/314 is done
        spin_half = lasy.ElementarySpace.from_defining_sectors(sym, np.array([[1, 0]]), multiplicities=[1])
        return MPS.from_desired_virtual_spaces(
            sites,
            [spin_half] * len(sites),
            bc='infinite',
            total_charge=sym.trivial_sector,
            unit_cell_width=self.lattice.mps_unit_cell_width,
        )
