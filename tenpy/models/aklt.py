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

        Returns
        -------
        psi_aklt : :class:`~tenpy.networks.mps.MPS`
            The AKLT groundstate

        """
        # build each B tensor of the MPS as contraction of
        #    --sR  sL--
        #       |  |
        #       proj
        #         |
        #
        # where  sL--sR forms a singlet between neighboring spin-1/2,
        # and proj is the projector from two spin-1/2 to spin-1.
        # indexing of basis states:
        # spin-1/2: 0=|m=-1/2>, 1=|m=+1/2>;  spin-1: 0=|m=-1>, 1=|m=0> 2=|m=+1>
        raise NotImplementedError('not yet ported to v2')
        sL = np.sqrt(0.5) * np.array([[1.0, 0.0], [0.0, -1.0]])  # p2 vR
        sR = np.array([[0.0, 1.0], [1.0, 0.0]])  # vL p1
        proj = np.zeros([3, 2, 2])  # p p1 p2
        proj[0, 0, 0] = 1.0  # |m=-1> = |down down>
        proj[1, 0, 1] = proj[1, 1, 0] = np.sqrt(0.5)  # |m=0> = (|up down> + |down, up>)/sqrt(2)
        proj[2, 1, 1] = 1.0  # |m=+1> = |up up>
        B = np.tensordot(sR, proj, axes=[1, 1])  # vL [p1], p [p1] p2
        B = np.tensordot(B, sL, axes=[2, 0])  # vL p [p2], [p2] vR
        B = B * np.sqrt(4.0 / 3.0)  # normalize after projection
        B = B.transpose([1, 0, 2])  # p vL vR
        # it's easy to check that B is right-canonical:
        BB = np.tensordot(B, B.conj(), axes=[[0, 2], [0, 2]])
        assert np.linalg.norm(np.eye(2) - BB) < 1.0e-14, 'BB=' + str(BB).replace('\n', '  ')

        L = self.lattice.N_sites
        Bs = [B] * L

        if self.lattice.bc_MPS == 'finite':
            # project onto one of the two virtual states on the left/right most state.
            # It's a ground state whatever you choose here,
            # but we project to different indices to allow Sz conservation
            # and fix overall Sz=0 sector
            Bs[0] = Bs[0][:, :1, :]
            Bs[-1] = Bs[-1][:, :, :-1]
            form = None  # Bs[-1] is not right canonical
        else:
            form = 'B'

        spin1 = self.lattice.unit_cell[0]
        if self.lattice.bc_MPS != 'finite' and spin1.conserve in ['Sz', 'parity']:
            charges = [[0], [2]] if spin1.conserve == 'Sz' else [[0], [1]]
            legL = npc.LegCharge.from_qflat(spin1.leg.chinfo, charges)
        else:
            legL = None

        res = MPS.from_Bflat(
            self.lattice.mps_sites(),
            Bs,
            bc=self.lattice.bc_MPS,
            permute=True,
            form=form,
            legL=legL,
            unit_cell_width=self.lattice.mps_unit_cell_width,
        )
        if form is None:
            res.canonical_form()
        return res
