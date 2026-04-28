"""Toy code implementing the 2-site DMRG algorithm."""

# Copyright (C) TeNPy Developers, Apache license
import numpy as np

import cyten as ct
from toycodes.tenpy_toycodes.a_mps import SimpleMPS, split_truncate_theta
from toycodes.tenpy_toycodes.b_model import TFIModel


class HEffective(ct.sparse.LinearOperator):
    """Class for the effective Hamiltonian.

    To be diagonalized in `DMRGEngine.update_bond`. Looks like this::

        .--vL             vR--.
        |       i     j       |
        |       |     |       |
        (LP)---(W1)--(W2)----(RP)
        |       |     |       |
        |       i*    j*      |
        .--vL*           vR*--.

    Parameters
    ----------
    LP : ct.SymmetricTensor
        Left environment with legs `'vL', 'wL*', 'vL*'`, where `'wL*'` is contrated with the MPO
        tensors and `'vL*'` with the MPS tensors. The leg `'vL'` is expected to be in the codomain,
        `'wL*', 'vL*'` in the domain.
    RP : ct.SymmetricTensor
        Right environment with legs `'vR', 'wR*', 'vR*'`, where `'wR*'` is contrated with the MPO
        tensors and `'vR*'` with the MPS tensors. The leg `'vR'` is expected to be in the codomain,
        `'wR*', 'vR*'` in the domain.
    W1, W2 : ct.SymmetricTensor
        MPO tensors of the two-site effective Hamiltonian, each with legs ``'wL', 'p', 'wR', 'p*'``
        with the codomain and domain consisting of `'wL', 'p'` and `'wR', 'p*'`, respectively.

    Attributes
    ----------
        LP, RP, W1, W2: same as parameters, but with the following changed legs in the (co)domain:
        LP : codomain: `'vL', 'wL*'`, domain: `'vL*'`.
        RP : codomain: `'vR*', 'wR*'`, domain: `'vR'`.
        W1 : codomain: `'p0', 'wC'`, domain: 'wL', 'p0*'
            (with renaming `'p', 'p*', 'wR'` to `'p0', 'p0*', 'wC'`)
        W2 : codomain: `'p1', 'wR'`, domain: 'wC', 'p1*'
            (with renaming `'p', 'p*', 'wL'` to `'p1', 'p1*', 'wC'`)

    """

    def __init__(self, LP, RP, W1, W2):
        # bend such that we can directly compose it with theta
        self.LP = ct.permute_legs(LP, ['vL', 'wL*'], ['vL*'], bend_right=True)  # vL wL* vL*
        self.RP = ct.permute_legs(
            RP, ['vR*', 'wR*'], ['vR'], bend_right=[True, False, False]
        )  # vR vR* wR* -> vR* wR* vR
        self.W1 = ct.permute_legs(
            W1, ['p', 'wR'], ['wL', 'p*'], bend_right=[False, None, True, None]
        )  # wL i wC i* -> i wC i* wL
        self.W1.relabel({'p': 'p0', 'p*': 'p0*', 'wR': 'wC'})
        self.W2 = ct.permute_legs(
            W2, ['p', 'wR'], ['wL', 'p*'], bend_right=[False, None, True, None]
        )  # wC j wR j* -> j wR j* wC
        self.W2.relabel({'p': 'p1', 'p*': 'p1*', 'wL': 'wC'})

    def matvec(self, theta: ct.Tensor) -> ct.Tensor:
        """Calculate |theta'> = H_eff |theta>"""
        # get_theta2(i) has 2 legs in codomain and 2 legs domain
        x = ct.permute_legs(theta, ['vL'], ['vR', 'p1', 'p0'], bend_right=True)  # vL p0 p1 vR
        x = ct.compose(self.LP, x)  # vL wL* p0 p1 vR
        x = ct.permute_legs(
            x, ['wL*', 'p0'], ['vL', 'vR', 'p1'], bend_right=[False, None, True, None, None]
        )  # wL* p0 p1 vR vL
        x = ct.compose(self.W1, x)  # p0 wC p1 vR vL
        x = ct.permute_legs(
            x, ['wC', 'p1'], ['p0', 'vL', 'vR'], bend_right=[False, None, True, None, None]
        )  # wC p1 vR vL p0
        x = ct.compose(self.W2, x)  # p1 wR vR vL p0
        x = ct.permute_legs(
            x, ['vL', 'p0', 'p1'], ['vR', 'wR'], bend_right=[None, True, None, False, False]
        )  # vL p0 p1 wR vR
        x = ct.compose(x, self.RP)  # vL p0 p1 vR
        x = ct.permute_legs(x, domain=['vR', 'p1'], bend_right=True)  # vL p0 p1 vR
        return x

    def to_tensor(self) -> ct.Tensor:
        raise NotImplementedError


class DMRGEngine:
    """DMRG algorithm, implemented as class holding the necessary data.

    Parameters
    ----------
    psi, model, chi_max, max_E_err, eps:
        See attributes

    Attributes
    ----------
    psi : MPS
        The current ground-state (approximation).
    model :
        The model of which the groundstate is to be calculated.
    chi_max, eps:
        Truncation parameters, see :func:`a_mps.split_truncate_theta`.
    LPs, RPs : list of ct.SymmetricTensor
        Left and right parts ("environments") of the effective Hamiltonian.
        ``LPs[i]`` is the contraction of all parts left of site `i` in the network ``<psi|H|psi>``,
        and similar ``RPs[i]`` for all parts right of site `i`.
        Each ``LPs[i]`` has legs ``vL wL* vL*``, ``RPs[i]`` has legs ``vR wR* vR*``
    max_E_err : float
        Convergence criterion for the energy difference between two consecutive sweeps.
    energies : list of float
        Energies after every update.

    """

    def __init__(
        self, psi: SimpleMPS, model: TFIModel, chi_max: int = 100, max_E_err: float = 1.0e-12, eps: float = 1.0e-12
    ):
        assert psi.L == model.L  # ensure compatibility
        assert psi.bc == model.bc
        self.H_mpo = model.H_mpo
        self.psi = psi
        self.LPs = [None] * psi.L
        self.RPs = [None] * psi.L
        self.chi_max = chi_max
        self.eps = eps
        self.max_E_err = max_E_err
        self.n_sweeps = 0
        self.energies = []
        # initialize left and right environment
        self.LPs[0] = self.init_LP()
        self.RPs[-1] = self.init_RP()
        # initialize necessary RPs
        for i in range(psi.L - 1, 1, -1):
            self.update_RP(i)

    def init_LP(self):
        mps_left_leg = self.psi.Bs[0].codomain[0]
        mpo_left_leg = self.H_mpo[0].codomain[0]
        sym = mps_left_leg.symmetry
        left_codom = ct.TensorProduct([mps_left_leg], sym)
        left_dom = ct.TensorProduct([mps_left_leg, mpo_left_leg], sym)
        tree_pairs = {}
        for tree, _, mults, _ in left_dom.iter_tree_blocks(mps_left_leg.sector_decomposition):
            if not np.all(tree.uncoupled[1] == sym.trivial_sector):
                continue
            # add the second MPS leg
            shape = np.append(mults[:1], mults[::-1])
            block = np.zeros(shape)
            block[:, 0, :] += np.eye(shape[0], shape[2])
            codom_tree = ct.FusionTree.from_sector(sym, tree.uncoupled[0], tree.are_dual[0])
            tree_pairs[(codom_tree, tree)] = block
        return ct.SymmetricTensor.from_tree_pairs(
            tree_pairs, left_codom, left_dom, self.psi.backend, labels=['vL', 'wL*', 'vL*']
        )

    def init_RP(self):
        mps_right_leg = self.psi.Bs[-1].domain[0].dual
        mpo_right_leg = self.H_mpo[-1].domain[1].dual
        sym = mps_right_leg.symmetry
        right_codom = ct.TensorProduct([mps_right_leg], sym)
        right_dom = ct.TensorProduct([mpo_right_leg, mps_right_leg], sym)
        tree_pairs = {}
        for tree, _, mults, _ in right_dom.iter_tree_blocks(mps_right_leg.sector_decomposition):
            if not np.all(tree.uncoupled[0] == sym.trivial_sector):
                continue
            # add the second MPS leg
            shape = np.append(mults[1:], mults[::-1])
            block = np.zeros(shape)
            block[:, :, -1] += np.eye(*shape[:-1])
            codom_tree = ct.FusionTree.from_sector(sym, tree.uncoupled[1], tree.are_dual[1])
            tree_pairs[(codom_tree, tree)] = block
        return ct.SymmetricTensor.from_tree_pairs(
            tree_pairs, right_codom, right_dom, self.psi.backend, labels=['vR', 'vR*', 'wR*']
        )

    def sweep(self):
        # sweep from left to right
        for i in range(self.psi.nbonds - 1):
            self.update_bond(i)
        # sweep from right to left
        for i in range(self.psi.nbonds - 1, 0, -1):
            self.update_bond(i)
        self.n_sweeps += 1

    def update_bond(self, i):
        j = i + 1
        # get effective Hamiltonian
        Heff = HEffective(self.LPs[i], self.RPs[j], self.H_mpo[i], self.H_mpo[j])
        # Diagonalize Heff, find ground state `theta`
        theta0 = self.psi.get_theta2(i)
        e, theta, _ = ct.krylov_based.lanczos(Heff, theta0)
        self.energies.append(e)
        # split and truncate
        Ai, Sj, Bj = split_truncate_theta(theta, self.chi_max, self.eps)
        # put back into MPS
        Gi = ct.scale_axis(Ai, ct.pinv(self.psi.Ss[i], cutoff=self.eps), leg='vL')
        Bi = ct.scale_axis(Gi, Sj, leg='vR')
        self.psi.Bs[i] = Bi
        self.psi.Ss[j] = Sj
        self.psi.Bs[j] = Bj
        self.update_LP(i)
        self.update_RP(j)

    def update_RP(self, i):
        """Calculate RP right of site `i-1` from RP right of site `i`."""
        j = i - 1
        RP = self.RPs[i]  # vR vR* wR*
        B = self.psi.Bs[i]  # vL p vR
        Bc = B.hc  # vR* p* vL*
        W = self.H_mpo[i]  # wL p wR p*

        Bc = ct.permute_legs(Bc, ['p*', 'vL*'], ['vR*'], bend_right=[True, False, False])  # p* vL* vR*
        RP = ct.compose(Bc, RP)  # p* vL* vR* wR*
        RP = ct.permute_legs(RP, ['vL*', 'vR*'], ['p*', 'wR*'], bend_right=[False, None, True, None])  # vL* vR* wR* p*
        W_ = ct.permute_legs(W, ['p', 'wR'], ['wL', 'p*'], bend_right=[False, None, True, None])  # p wR p* wL
        RP = ct.compose(RP, W_)  # vL* vR* p* wL
        RP = ct.permute_legs(RP, ['wL', 'vL*'], ['p*', 'vR*'], bend_right=[None, True, None, False])  # wL vL* vR* p*
        B_ = ct.permute_legs(B, ['p', 'vR'], ['vL'], bend_right=[False, None, True])  # p vR vL
        RP = ct.compose(RP, B_, relabel1={'vL*': 'vR', 'wL': 'wR*'}, relabel2={'vL': 'vR*'})
        # wL vL* vL == wR* vR vR*
        RP = ct.permute_legs(RP, ['vR'], ['wR*', 'vR*'], bend_right=[False, None, None])
        self.RPs[j] = RP

    def update_LP(self, i):
        """Calculate LP left of site `i+1` from LP left of site `i`."""
        j = i + 1
        LP = self.LPs[i]  # vL wL* vL*
        B = self.psi.Bs[i]  # vL p vR
        G = ct.scale_axis(B, ct.pinv(self.psi.Ss[j], cutoff=self.eps), leg='vR')
        A = ct.scale_axis(G, self.psi.Ss[i], leg='vL')
        Ac = A.hc
        W = self.H_mpo[i]  # wL p wR p*

        Ac = ct.permute_legs(Ac, codomain=['vR*', 'p*'], bend_right=True)
        LP = ct.compose(Ac, LP)  # vR* p* wL* vL*
        LP = ct.permute_legs(LP, ['vL*', 'vR*'], ['wL*', 'p*'], bend_right=[None, True, None, False])  # vL* vR* p* wL*
        LP = ct.compose(LP, W)  # vL* vR* wR p*
        LP = ct.permute_legs(LP, ['vR*', 'wR'], ['vL*', 'p*'], bend_right=[False, None, True, None])  # vR* wR* p* vL*
        # vR* wR vR == vL wL* vL*
        LP = ct.compose(LP, A, relabel1={'vR*': 'vL', 'wR': 'wL*'}, relabel2={'vR': 'vL*'})
        LP = ct.permute_legs(LP, domain=['vL*', 'wL*'], bend_right=True)
        self.LPs[j] = LP

    def run(self) -> float:
        self.sweep()
        e_new = self.energies[-1]
        e_old = e_new + 2 * self.max_E_err
        while abs(e_new - e_old) > self.max_E_err:
            e_old = e_new
            self.sweep()
            e_new = self.energies[-1]
        return self.energies[-1]
