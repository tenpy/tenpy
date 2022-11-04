"""Prototypical example of a 1D quantum model constructed from bond terms: the AKLT chain.

The AKLT model is famous for having a very simple ground state MPS of bond dimension 2.
Writing down the Hamiltonian is easiest done in terms of bond couplings.
This class thus serves as an example how this can be done.
"""
# Copyright 2021 TeNPy Developers, GNU GPLv3

import numpy as np

import tenpy.linalg.np_conserved as npc
from tenpy.networks.site import SpinSite, kron
from tenpy.networks.mps import MPS
from tenpy.models.lattice import Chain
from tenpy.models.model import NearestNeighborModel, MPOModel
from tenpy.tools.params import asConfig

__all__ = ['AKLTChain']


class AKLTChain(NearestNeighborModel, MPOModel):
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
    def __init__(self, model_params):
        model_params = asConfig(model_params, "AKLTModel")
        L = model_params.get('L', 2)
        site = SpinSite(S=1., conserve='Sz')

        # lattice
        bc_MPS = model_params.get('bc_MPS', 'finite')
        bc = 'open' if bc_MPS == 'finite' else 'periodic'
        lat = Chain(L, site, bc=bc, bc_MPS=bc_MPS)

        Sp, Sm, Sz = site.Sp, site.Sm, site.Sz
        S_dot_S = 0.5 * (kron(Sp, Sm) + kron(Sm, Sp)) + kron(Sz, Sz)
        S_dot_S_square = npc.tensordot(S_dot_S, S_dot_S, [['(p0*.p1*)'], ['(p0.p1)']])

        H_bond = S_dot_S + S_dot_S_square / 3.
        # P_2 = H_bond * 0.5 + 1/3 * npc.eye_like(S_dot_S)

        J = model_params.get('J', 1.)
        H_bond = J * H_bond.split_legs().transpose(['p0', 'p1', 'p0*', 'p1*'])
        H_bond = [H_bond] * L
        # H_bond[i] acts on sites (i-1, i)
        if bc_MPS == "finite":
            H_bond[0] = None
        # 7) initialize H_bond (the order of 7/8 doesn't matter)
        NearestNeighborModel.__init__(self, lat, H_bond)
        # 9) initialize H_MPO
        MPOModel.__init__(self, lat, self.calc_H_MPO_from_bond())


    def psi_AKLT(self):
        """Initialize the chi=2 MPS which is exact ground state of the AKLT model."""
        # build each B tensor of the MPS as contraction of
        #    --sR  sL--
        #       |  |
        #       proj
        #         |
        #
        # where  SL--SR forms a singlet between neighboring spin-1/2,
        # and projS1 is the projector from two spin-1/2 to spin-1.
        # indexing of basis states:
        # spin-1/2: 0=|m=-1/2>, 1=|m=+1/2>;  spin-1: 0=|m=-1>, 1=|m=0> 2=|m=+1>
        sL = np.sqrt(0.5) * np.array([[1., 0.], [0., -1.]]) # p2 vR
        sR = np.array([[0., 1.], [1., 0.]]) # vL p1
        proj = np.zeros([3, 2, 2]) # p p1 p2
        proj[0, 0, 0] = 1.  # |m=-1> = |down down>
        proj[1, 0, 1] = proj[1, 1, 0] = np.sqrt(0.5)  # |m=0> = (|up down> + |down, up>)/sqrt(2)
        proj[2, 1, 1] = 1.  # |m=+1> = |up up>
        B = np.tensordot(sR, proj, axes=[1, 1]) # vL [p1], p [p1] p2
        B = np.tensordot(B, sL, axes=[2, 0]) # vL p [p2], [p2] vR
        B = B * np.sqrt(4./3.)  # normalize after projection
        B = B.transpose([1, 0, 2]) # p vL vR
        S = np.sqrt(0.5) * np.ones([2])
        # it's easy to check that B is right-canonical:
        BB = np.tensordot(B, B.conj(), axes=[[0, 2], [0, 2]])
        assert np.linalg.norm(np.eye(2) - BB) < 1.e-14

        L = self.lat.N_sites
        Bs = [B] * L
        Ss = [S] * (L + 1)

        spin1 = self.lat.unit_cell[0]
        legL = None  # default
        if self.lat.bc_MPS == 'finite':
            # project onto one of the two virtual states on the left/right most state.
            # It's a ground state whatever you choose here,
            # but we project to different indices to allow Sz convservation
            # and fix overall Sz=0 sector
            Bs[0] = Bs[0][:, :1, :]
            Bs[-1] = Bs[-1][:, -1:, :]
            Ss[0] = Ss[-1] = np.ones([1.])
        elif spin1.conserve in ['Sz', 'parity']:
            chinfo = spin1.leg.chinfo
            legL = npc.LegCharge.from_qflat(chinfo, [[0], [2]])
        return MPS.from_Bflat(self.lat.mps_sites(),
                              Bs,
                              bc=self.lat.bc_MPS,
                              permute=True,
                              form='B',
                              legL=legL)
