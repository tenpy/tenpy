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
        r"""Initialize an MPS which is an exact ground state of the AKLT model.

        The AKLT state is built from a spin-1/2 singlet on each bond and a projector from two
        spin-1/2 to a spin-1 on each site. Hence, all bonds carry a spin-1/2, and the state has
        bond dimension ``chi=2``.

        **Infinite chain.**
        The AKLT state is the unique ground state, with energy ``-2/3`` per site.
        For ``conserve='SU2'``, the (right canonical) tensor with spin-1/2 bonds is unique up to a
        gauge transformation, such that a random symmetric tensor with spin-1/2 bonds is the AKLT
        state.
        For the abelian symmetries (``None``, ``'parity'`` and ``'Sz'``), the tensor is
        constructed explicitly. The two states of the virtual spin-1/2 carry the charges
        ``Sz=-1/2`` and ``Sz=+1/2`` (up to an overall shift).

        **Finite chain.**
        An open chain of ``L`` sites has a free spin-1/2 at each of its two ends, such that the
        ground space is 4-fold degenerate: the two edge spins form a singlet (spin 0) or a
        triplet (spin 1), and all four states have the energy ``-(L-1) * 2/3``.
        Which of these states is returned depends on `conserve`:

        - ``conserve=None``, ``'parity'`` or ``'Sz'``: The left edge spin is up and the right edge
          spin is down, which is a superposition of the singlet and the ``S^z=0`` triplet state.
          It has total ``Sz=0`` and bond dimension 2 on all bonds, and it is not invariant under SU(2).
        - ``conserve='SU2'``: The two edge spins are in a singlet, such that the state has total
          spin 0. It is the AKLT state on a periodic ring of ``L`` sites (the open chain has no
          term on the closing bond), written as an open MPS by carrying the closing bond along the
          chain: the bond dimension is 4 on the inner bonds and 3 on the bonds next to the two
          edge sites.

        A finite chain needs ``L >= 2`` sites.

        Returns
        -------
        psi_aklt : :class:`~tenpy.networks.mps.MPS`
            The AKLT groundstate

        Raises
        ------
        ValueError
            For a finite chain with ``L < 2`` sites.
        NotImplementedError
            For ``conserve='parity'`` on an infinite chain with an odd number ``L`` of sites.
            The state has a non-trivial total charge per unit cell then, which
            :meth:`~tenpy.networks.mps.MPS.canonical_form` does not support yet, see https://github.com/tenpy/tenpy/issues/765.

        """
        sites = self.lattice.mps_sites()
        L = len(sites)
        sym = sites[0].symmetry
        bc = self.lattice.bc_MPS
        unit_cell_width = self.lattice.mps_unit_cell_width
        if bc == 'finite' and L < 2:
            raise ValueError("psi_AKLT needs at least two sites for bc_MPS='finite'")

        if self.conserve == 'SU2':
            spin_half = lasy.ElementarySpace(sym, sym.get_sector('[1 0]'))
            # the tensor with spin-1/2 bonds is unique up to a gauge transformation
            psi = MPS.from_desired_virtual_spaces(
                sites,
                [spin_half] * L,
                bc='infinite',
                total_charge=sym.trivial_sector,
                unit_cell_width=unit_cell_width,
            )
            if bc == 'infinite':
                return psi
            return self._psi_AKLT_SU2_finite(psi)

        # build the B tensor of the MPS as contraction of
        #    --sR  sL--
        #       |  |
        #       proj
        #         |
        #
        # where  sL--sR forms a singlet between neighboring spin-1/2,
        # and proj is the projector from two spin-1/2 to spin-1.
        # indexing of basis states:
        # spin-1/2: 0=|m=-1/2>, 1=|m=+1/2>;  spin-1: 0=|m=-1>, 1=|m=0> 2=|m=+1>
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
        Bs = [B] * L

        if bc == 'finite':
            # project onto the first virtual state on the left-most and onto the first virtual state on the
            # right-most site (left edge spin up, right edge spin down). We obtain total Sz=0.
            Bs[0] = Bs[0][:, :1, :]
            Bs[-1] = Bs[-1][:, :, :1]
            # the projected edge tensors are not right canonical; from_Bflat brings the state into canonical form
            form = None
            legL = None  # trivial charge
            total_charge = None  # ignored for finite bc
        else:
            form = 'B'
            # The two virtual states are the spin-1/2 states with Sz=-1/2 and Sz=+1/2, up to an overall shift of
            # the charge: 2*Sz = 0 and 2 for Sz, and the parities 0 and 1. Without symmetry, the charges are trivial.
            charges = {'Sz': [[0], [2]], 'parity': [[0], [1]]}.get(self.conserve, [[0], [0]])
            legL = lasy.ElementarySpace.from_defining_sectors(sym, charges, unique_sectors=False)
            # Once the virtual charges return to their initial value, the charge of the unit cell is L times the
            # charge of the Sz=0 state. For 'parity', the Sz=0 state of a spin-1 site has the charge 1 (the parity of
            # a site is (Sz+1) mod 2), such that the unit cell has the parity L mod 2.
            sz_zero_charge = sites[0].leg.idx_to_sector(1)  # basis states are ordered Sz=-1, 0, 1
            total_charge = sym.trivial_sector
            for _ in range(L):
                total_charge = sym.fusion_outcomes(total_charge, sz_zero_charge)[0]

        return MPS.from_Bflat(
            sites,
            Bs,
            bc=bc,
            form=form,
            legL=legL,
            total_charge=total_charge,
            unit_cell_width=unit_cell_width,
        )

    def _psi_AKLT_SU2_finite(self, psi_ring):
        """Write the periodic AKLT ring as an open (finite) MPS.

        The closing bond (`vL` of the first site) is carried along the chain as an extra strand next
        to the bonds, and traced with `vR` of the last site.

        Parameters
        ----------
        psi_ring : :class:`~tenpy.networks.mps.MPS`
            Infinite MPS with ``L`` sites in the unit cell and spin-1/2 bonds.

        Returns
        -------
        psi : :class:`~tenpy.networks.mps.MPS`
            The same (ring) state as a finite MPS in canonical form. All inner bonds are the product of
            the spin-1/2 bond and the spin-1/2 strand, i.e. spin 0 and spin 1.

        """
        sites = self.lattice.mps_sites()
        L = len(sites)
        Bs = [psi_ring.get_B(i, 'B') for i in range(L)]
        spin_half = Bs[0].get_leg('vL')
        # the dual space makes the strand compatible with the leg `e` of the first site, which is bent from vL
        strand = lasy.eye(spin_half.dual, Bs[0].backend, labels=['e', 'e*'], dtype=Bs[0].dtype, diagonal=False)
        tensors = []
        for i, B in enumerate(Bs):
            if i == 0:
                B = B.copy()
                B.labels = ['e', 'p', 'vR']
                B = lasy.planar_permute_legs(B, codomain=['p'], domain=['e', 'vR'])
                B = lasy.add_trivial_leg(B, codomain_pos=0, label='vL')
                B = lasy.planar_combine_legs(B, ['vR', 'e'], pipe_dualities=[True])
            else:
                B = lasy.outer(strand, B)  # legs e, vL, p, vR, e*
                if i == L - 1:
                    B = lasy.planar_partial_trace(B, ['vR', 'e*'])
                    B = lasy.planar_combine_legs(B, ['e', 'vL'])
                    B = lasy.add_trivial_leg(B, domain_pos=0, label='vR')
                else:
                    B = lasy.planar_combine_legs(B, ['e', 'vL'], ['vR', 'e*'], pipe_dualities=[False, True])
            B.labels = ['vL', 'p', 'vR']
            B = lasy.flatten_pipe_leg(B, 'vL')  # bonds are ElementarySpaces, not LegPipes
            B = lasy.flatten_pipe_leg(B, 'vR')
            tensors.append(B)
        SVs = [lasy.DiagonalTensor.from_eye(B.get_leg('vL'), B.backend, ['vL', 'vR'], B.dtype.to_real) for B in tensors]
        last = tensors[-1]
        SVs.append(
            lasy.DiagonalTensor.from_eye(last.get_leg('vR').dual, last.backend, ['vL', 'vR'], last.dtype.to_real)
        )
        res = MPS(sites, tensors, SVs, bc='finite', form=None, unit_cell_width=self.lattice.mps_unit_cell_width)
        res.canonical_form()
        return res
