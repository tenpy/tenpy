"""Prototypical example of a 1D quantum model constructed from bond terms: the AKLT chain.

The AKLT model is famous for having a very simple ground state MPS of bond dimension 2.
Writing down the Hamiltonian is easiest done in terms of bond couplings.
This class thus serves as an example how this can be done in more gen
The XXZ chain is contained in the more general :class:`~tenpy.models.spins.SpinChain`; the idea of
this module is more to serve as a pedagogical example for a model.
"""
# Copyright 2021 TeNPy Developers, GNU GPLv3

import tenpy.linalg.np_conserved as npc
from tenpy.networks.site import SpinSite, kron
from tenpy.models.lattice import Chain
from tenpy.models.model import NearestNeighborModel, MPOModel
from tenpy.tools.params import asConfig


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
        bc = 'periodic' if bc_MPS == 'infinite' else 'open'
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
