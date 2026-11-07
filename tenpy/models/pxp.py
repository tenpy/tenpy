"""Implementation of the PXP model on a chain."""

# Copyright (C) TeNPy Developers, Apache license
from typing import Literal

import lasyten as lasy

from .lattice import Chain
from .model import ModelBuilder

__all__ = ['PXPChain']


class PXPChain(ModelBuilder):
    r"""The PXP model as (approximately) implemented by a chain of Rydberg-blockaded atoms.

    The Hamiltonian reads:

    .. math ::
        H =  \mathtt{J} \sum_{i} P_{i-1} X_i P_{i+1}
            + \mathtt{J_boundary} X_0 P_1 + P_{L-2} X_{L-1}

    where we only add the boundary terms for open boundaries with `J_boundary` defaulting to `J`.
    `P` is the projector onto the up state of the site, which corresponds to the ground state
    of the atom.

    The model arises from the strong-interaction limit of Rydberg atom chains in the seminal
    experiment of :doi:`10.1038/nature24622`, which found long oscillations now attributed to
    quantum many-body scars in the PXP model.

    Options
    -------
    .. cfg:config :: PXPChain
        :include: CouplingMPOModel

        conserve : 'best' | None
            What should be conserved. See :class:`~lasyten.sites.SpinSite`.
        J, J_boundary : float | array
            Couplings as defined for the Hamiltonian above.
    """

    default_lattice = Chain
    force_default_lattice = True  # we implicitly assume a 1D chain,
    # otherwise more P's need to be added
    conserve: Literal[None]

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'best', None)
        if conserve == 'best':
            conserve = None
        self.conserve = conserve
        assert conserve in [None, 'None']
        return lasy.sites.SpinSite(S=0.5, conserve=conserve)

    def init_couplings(self, model_params):
        site, *_ = self.lattice.unit_cell
        P = site.get_op('Sz') + 0.5 * site.get_op('Id')  # projector onto up state
        P = P.as_SymmetricTensor()
        X = 2 * site.get_op('Sx')
        px = lasy.outer(P, X, relabel1={'p': 'p0', 'p*': 'p0*'}, relabel2={'p': 'p1', 'p*': 'p1*'})
        xp = lasy.outer(X, P, relabel1={'p': 'p0', 'p*': 'p0*'}, relabel2={'p': 'p1', 'p*': 'p1*'})
        pxp = lasy.outer(px, P, relabel2={'p': 'p2', 'p*': 'p2*'})
        self.define_coupling(
            lasy.Coupling.from_tensor(pxp, [site, site, site], name='p.x.p'), hc='p.x.p', permute=None
        )  # permutations should not be needed
        self.define_coupling(
            lasy.Coupling.from_tensor(xp, [site, site], name='x.p'), hc='x.p', permute=None
        )  # permutations should not be needed
        self.define_coupling(
            lasy.Coupling.from_tensor(px, [site, site], name='p.x'), hc='p.x', permute=None
        )  # permutations should not be needed

    def init_terms(self, model_params):
        J = model_params.get('J', 2.0)
        self.add_multi_site_coupling(J, 'p.x.p', dx=[[0], [1], [2]])
        if model_params['bc_x'] == 'open':
            L = model_params['L']
            J_boundary = model_params.get('J_boundary', J)
            self.add_two_site_coupling(J_boundary, 'x.p', dx=1, x1=0)
            self.add_two_site_coupling(J_boundary, 'p.x', dx=1, x1=L - 2)
