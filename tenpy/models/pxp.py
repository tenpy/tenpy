"""Implementation of the PXP model on a chain."""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np

from ..networks.site import SpinHalfSite
from .lattice import Chain
from .model import CouplingMPOModel

__all__ = ['PXPChain']


class PXPChain(CouplingMPOModel):
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

    .. note ::

        This Hamiltonian does **not** conserve :math:`S^z` or its parity: each bulk term
        contains a single :math:`X` and thus flips one spin.  In particular,
        :math:`C=\prod_i Z_i` *anticommutes* with :math:`H` (spectral mirroring
        :math:`E\leftrightarrow -E`), so ``conserve='parity'`` must not be used.

    Options
    -------
    .. cfg:config :: PXPChain
        :include: CouplingMPOModel

        conserve : 'best' | None
            No charge conservation is possible for this model; ``'best'`` maps to ``None``.
            ``'parity'`` and ``'Sz'`` raise :exc:`ValueError`.
        J, J_boundary : float | array
            Couplings as defined for the Hamiltonian above.
    """

    default_lattice = Chain
    force_default_lattice = True  # we implicitly assume a 1D chain,
    # otherwise more P's need to be added

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'best', None)
        if conserve == 'best':
            conserve = None
            self.logger.info('%s: set conserve to %s', self.name, conserve)
        if conserve not in [None, 'None']:
            model_params.unused.clear()  # aborting init; avoid unused-option warnings
            raise ValueError(
                'PXPChain does not conserve Sz or Sz-parity '
                '(each PXP term flips one spin; prod_i Z_i anticommutes with H). '
                'Use conserve=None.'
            )
        s = SpinHalfSite(conserve=conserve)
        s.add_op('X', s.get_op('Sigmax'), hc='X')  # X is already defined under other name
        # but P is not, so we add it (as projector onto the state 0, i.e. the up spin).
        s.add_op('P', np.array([[1.0, 0.0], [0.0, 0.0]]), hc='P', permute_dense=True)
        return s

    def init_terms(self, model_params):
        J = model_params.get('J', 2.0)
        self.add_multi_coupling(J, [('P', [-1], 0), ('X', [0], 0), ('P', [1], 0)])

        if model_params['bc_x'] == 'open':
            L = model_params['L']
            J_boundary = model_params.get('J_boundary', J)
            self.add_coupling_term(J_boundary, 0, 1, 'X', 'P')
            self.add_coupling_term(J_boundary, L - 2, L - 1, 'P', 'X')
