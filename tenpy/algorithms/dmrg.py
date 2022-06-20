"""Density Matrix Renormalization Group (DMRG).

Although it was originally not formulated with tensor networks,
the DMRG algorithm (invented by Steven White in 1992 :cite:`white1992`) opened the whole field
with its enormous success in finding ground states in 1D.

We implement DMRG in the modern formulation of matrix product states :cite:`schollwoeck2011`,
both for finite systems (``'finite'`` or ``'segment'`` boundary conditions)
and in the thermodynamic limit (``'infinite'`` b.c.).

The function :func:`run` - well - runs one DMRG simulation.
Internally, it generates an instance of an :class:`Sweep`.
This class implements the common functionality like defining a `sweep`,
but leaves the details of the contractions to be performed to the derived classes.

Currently, there are two derived classes implementing the contractions: :class:`SingleSiteDMRGEngine`
and :class:`TwoSiteDMRGEngine`. They differ (as their name implies) in the number of sites which
are optimized simultaneously.
They should both give the same results (up to rounding errors). However, if started from a product
state, :class:`SingleSiteDMRGEngine` depends critically on the use of a :class:`Mixer`, while
:class:`TwoSiteDMRGEngine` is in principle more computationally expensive to run and has
occasionally displayed some convergence issues..
Which one is preffered in the end is not obvious a priori and might depend on the used model.
Just try both of them.

A :class:`Mixer` should be used initially to avoid that the algorithm gets stuck in local energy
minima, and then slowly turned off in the end. For :class:`SingleSiteDMRGEngine`, using a mixer is
crucial, as the one-site algorithm cannot increase the MPS bond dimension by itself.

A generic protocol for approaching a physics question using DMRG is given in
:doc:`/intro/dmrg-protocol`.
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np
import time
import warnings
import logging
logger = logging.getLogger(__name__)

from ..linalg import np_conserved as npc
from ..networks.mps import MPSEnvironment
from ..linalg.lanczos import lanczos, lanczos_arpack
from .truncation import truncate, svd_theta
from ..tools.params import asConfig
from ..tools.math import entropy
from ..tools.misc import find_subclass
from ..tools.process import memory_usage
from .mps_common import Sweep, OneSiteH, TwoSiteH

__all__ = [
    'run',
    'DMRGEngine',
    'SingleSiteDMRGEngine',
    'TwoSiteDMRGEngine',
    'Mixer',
    'SubspaceExpansion',
    'DensityMatrixMixer',
    'chi_list',
    'full_diag_effH',
    'SingleSiteMixer',
    'TwoSiteMixer',
    'EngineCombine',
    'EngineFracture',
]


def run(psi, model, options, **kwargs):
    r"""Run the DMRG algorithm to find the ground state of the given model.

    Parameters
    ----------
    psi : :class:`~tenpy.networks.mps.MPS`
        Initial guess for the ground state, which is to be optimized in-place.
    model : :class:`~tenpy.models.MPOModel`
        The model representing the Hamiltonian for which we want to find the ground state.
    options : dict
        Further optional parameters as described in :cfg:config:`DMRGEngine`.
    **kwargs :
        Further keyword arguments for the algorithm classes :class:`TwoSiteDMRGEngine` or
        :class:`SingleSiteDMRGEngine`.

    Returns
    -------
    info : dict
        A dictionary with keys ``'E', 'shelve', 'bond_statistics', 'sweep_statistics'``

    Options
    -------
    .. cfg:config :: DMRG
        :include: SingleSiteDMRGEngine, TwoSiteDMRGEngine

        active_sites : 1 | 2
            The number of active sites to be used by DMRG.
            If set to 1, :class:`SingleSiteDMRGEngine` is used.
            If set to 2, DMRG is handled by :class:`TwoSiteDMRGEngine`.

    """
    # initialize the engine
    options = asConfig(options, 'DMRG')
    active_sites = options.get('active_sites', 2)
    if active_sites == 1:
        engine = SingleSiteDMRGEngine(psi, model, options, **kwargs)
    elif active_sites == 2:
        engine = TwoSiteDMRGEngine(psi, model, options, **kwargs)
    else:
        raise ValueError("For DMRG, can only use 1 or 2 active sites, not {}".format(active_sites))
    E, _ = engine.run()
    return {
        'E': E,
        'shelve': engine.shelve,
        'bond_statistics': engine.update_stats,
        'sweep_statistics': engine.sweep_stats
    }


class Mixer:
    """Base class of a general Mixer.

    Since DMRG performs only local updates of the state, it can get stuck in "local minima",
    in particular if the Hamiltonain is long-range -- which is the case if one
    maps a 2D system ("infinite cylinder") to 1D -- or if one wants to do single-site updates.
    The idea of the mixer is to perturb the state with the terms of the Hamiltonian
    which have contributions in both the "left" and "right" side of the system.
    In that way, it adds fluctuation of the quantum numbers and non-zero contributions of the
    long-range terms - leading to a significantly improved convergence of DMRG.

    The strength of the perturbation is given by the `amplitude` of the mixer.
    A good strategy is to choose an initially significant amplitude and let it decay until
    the perturbation becomes completely irrelevant and the mixer gets disabled.

    This original idea of the mixer was introduced in :cite:`white2005`, implemented as
    :class:`DensityMatrixMixer`.
    More recently, :cite:`hubig2015` discussed the mixer and provided an improved version
    based on an svd, which turns out to give the same results up to numerical errors;
    it's implemented as the :class:`SubspaceExpansion`.

    Parameters
    ----------
    options : dict
        Optional parameters as described in the following table.
        see :cfg:config:`Mixer`
    sweep_activated : int
        The first sweep where the mixer was activated; `disable_after` is relative to that.

    Options
    -------
    .. cfg:config :: Mixer

        amplitude : float
            Initial strength of the mixer. (Should be sufficiently smaller than 1.)
        decay : float
            To slowly turn off the mixer, we divide `amplitude` by `decay`
            after each sweep. (Should be >= 1.)
        disable_after : int
            We disable the mixer completely after this number of sweeps.

    Attributes
    ----------
    amplitude : float
        Current amplitude for mixing. Singular values are perturbed on that order of magnitude.
    decay : float
        Factor by which `amplitude` is divided after each sweep.
    disable_after : int
        The number of sweeps after which the mixer should be disabled, relative to `disable_after`.
        Note that DMRG might repeatedly activate the mixer if you gradually increase `chi` with
        a :cfg:configoption`DMRGEngine.chi_list`.
    """
    #: how many sites the `theta` in `perturb_svd` should have
    update_sites = 2

    def __init__(self, options, sweep_activated):
        self.options = options = asConfig(options, 'Mixer')
        self.amplitude = options.get('amplitude', 1.e-5)
        assert self.amplitude <= 1.
        self.decay = options.get('decay', 2.)
        assert self.decay >= 1.
        if self.decay == 1.:
            warnings.warn("Mixer with decay=1. doesn't decay")
        self.disable_after = options.get('disable_after', 15)
        self.sweep_activated = sweep_activated

    def update_amplitude(self, sweeps):
        """Update the amplitude, possibly disable the mixer.

        Parameters
        ----------
        sweeps : int
            The number of performed sweeps, to check if we need to disable the mixer.

        Returns
        -------
        mixer : :class:`Mixer` | None
            Returns `self` if we should continue mixing, or ``None``, if the mixer
            should be disabled.
        """
        self.amplitude /= self.decay
        if (sweeps >= self.disable_after + self.sweep_activated
                or self.amplitude <= np.finfo('float').eps):
            logger.info("disable mixer after %(sweeps)d sweeps, final amplitude %(amp).2e", {
                'sweeps': sweeps,
                'amp': self.amplitude
            })
            return None  # disable mixer
        return self

    def perturb_svd(self, engine, theta, i0, update_LP, update_RP):
        """Perturb the wave function and perform an SVD with truncation.

        The call structure is slightly different depending on :attr:`update_sites`;
        see :meth:`SubspaceExpansion.perturb_svd` and :meth:`DensityMatrixMixer.perturb_svd`.
        """
        raise NotImplementedError("This function should be implemented in derived classes")

    def _mix_LR(self, H, i0, sqrt=False):
        """Return `mixL, mixR, IdL, IdR` on bond ``i0:i0+1``."""
        chi_MPO = H.get_W(i0).get_leg('wR').ind_len
        IdL, IdR = H.get_IdL(i0 + 1), H.get_IdR(i0)
        amplitude = np.sqrt(self.amplitude) if sqrt else self.amplitude
        mix_L = np.full((chi_MPO, ), amplitude)
        mix_R = np.full((chi_MPO, ), amplitude)
        one = 1. if not H.explicit_plus_hc else 0.5
        if IdL is not None:
            mix_L[IdL] = one
            mix_R[IdL] = 0.
        if IdR is not None:
            mix_L[IdR] = 0.
            mix_R[IdR] = one
        return mix_L, mix_R, IdL, IdR, H.explicit_plus_hc


def _get_LHeff(env, i, eff_H):
    # return LHeff with p0 labels on site `i`
    if i == eff_H.i0 and hasattr(eff_H, 'LHeff'):
        return eff_H.LHeff
    # else:
    return env._contract_LHeff(i)


def _get_RHeff(env, i, eff_H):
    # return RHeff with 'p1' labels on site `i`
    if i == eff_H.i0 + eff_H.length - 1 and hasattr(eff_H, 'RHeff'):
        if eff_H.length == 1:
            return eff_H.RHeff.replace_labels(['(p0.vL*)', '(p0*.vL)'], ['(p1.vL*)', '(p1*.vL)'])
        return eff_H.RHeff
    # else:
    return env._contract_RHeff(i)


class SubspaceExpansion(Mixer):
    """Mixer of a direct subspace expansion for both single-site DMRG and two-site DMRG.

    Performs a subspace expansion following :cite:`hubig2015`.
    It views `theta` as a single-site wave function.

    It is actually not necessary to fill the `next_B` with zeros as described in Hubig's paper;
    rather we directly project the `wR` leg of `VH` onto the `IdL` index, which corresponds to
    taking the original `theta` (up to truncation).

    Pictorially for a subspace expansion of the left `U` while moving right::

        |  --theta---            .-theta---                                  --U---S---VH---
        |     |                  |   |                                         |       |
        |             =dot=>    LP---H0--mix_L--     =SVD=>                    |       .---[IdL]
        |                        |   |          (vL.p0),(wR.vR)

    For a left-move::

        |  --theta---            --theta--.                         ---U---S---VH--
        |     |                      |    |                            |       |
        |            =dot=>  --mix_R-H0---RP         =SVD=>      [IdR]-.       |
        |                            |    |     (vL.wL),(p0.vR)


    Note that only the `U` during the right move (or `VH` during left-move) is guaranteed to be
    an isometry as expected in the canonical form; `VH` during the right-move contains a
    "subspace expansion" and does not fulfill the canonical ``VH.dot(VH.conj().T) == eye``.
    Moreover, the `U` constructed from a two-site `theta` viewing the ``'(p1.vR)`` leg as just `vR`
    in the right-move is (mathematically) equivalent to the `U` returned by the
    :class:`DensityMatrixMixer` (up to degenerate singular values).

    In other words, the :class:`SubspaceExpansion` and :class:`DensityMatrixMixer`
    should produce equivalent results; they only differ in the way they calculate `U` and `V`
    internally.
    """
    update_sites = 1

    def perturb_svd(self, engine, theta, i0, move_right):
        """Preform a subspace expansion of a single-site wave function on one side.

        Parameters
        ----------
        engine : :class:`DMRGEngine`
            The DMRG engine calling the mixer.
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            The optimized wave function, prepared for svd, with labels ``'(vL.p0)', 'vR'`` for
            right move, or ``'vL', '(p0.vR)'`` for left move.
        i0 : int
            The site index where `theta` lives.
        move_right : bool
            Whether we move to the right (``True``) or left (``False``).

        Returns
        -------
        U, VH : :class:`~tenpy.linalg.np_conserved.Array`
            Left and right part of the subspace-expanded svd.
            Always such that the contraction ``U.S.VH`` resembles the original `theta` up to
            truncation error.
            `U` has labels ``'(vL.p0)', 'vR'`` (right move) or ``'vL', 'vR'`` (left move).
            `V` has labels ``'vL', 'vR'`` (right move) or ``'(vL.p0)', 'vR'`` (left move).
            For a right move, only `U` is canonical; for a left-move only `VH` is canonical.
        S : 1D ndarray
            (Perturbed) singular values on the new bond.
        err : :class:`~tenpy.algorithms.truncation.TruncationError`
            The truncation error introduced.
        S_approx : ndarray
            Same as `S`.
        """
        bond = i0 if move_right else i0 - 1
        mix_L, mix_R, IdL, IdR, explicit_plus_hc = self._mix_LR(engine.env.H, bond, sqrt=True)

        if move_right:
            LHeff = _get_LHeff(engine.env, i0, engine.eff_H)
            LHeff = LHeff.transpose(['(vR*.p0)', 'wR', '(vR.p0*)'])
            if not explicit_plus_hc and IdL is not None:
                theta_expand = npc.tensordot(LHeff.iscale_axis(mix_L, 'wR'), theta,
                                             ['(vR.p0*)', '(vL.p0)'])
                theta_expand.ireplace_label('(vR*.p0)', '(vL.p0)')
            else:
                # need to stack different parts of the wR leg
                wR = LHeff.get_leg('wR')
                stack = [theta.add_trivial_leg(1, 'wR', wR.qconj)]  # explicitly add the identity
                proj = np.ones(wR.ind_len - (IdL is not None) - (IdR is not None), bool)
                if IdL is not None:
                    proj[IdL] = False
                if IdR is not None:
                    proj[IdR] = False
                LHeff.iproject(proj, 'wR')
                LHeff = LHeff * np.sqrt(self.amplitude)
                stack.append(npc.tensordot(LHeff, theta, ['(vR.p0*)', '(vL.p0)']))
                if explicit_plus_hc:
                    # apply (LHeff^dagger theta) = conj(dot(LHeff.T, theta.conj()))
                    th = npc.tensordot(LHeff, theta.conj(), ['(vR*.p0)', '(vL*.p0*)'])
                    stack.append(th.itranspose(['(vR.p0*)', 'wR', 'vR*']).iconj())
                theta_expand = npc.concatenate(stack, axis='wR')
                IdL = 0  # of the new, concatenated leg.
            theta_expand = theta_expand.combine_legs(['wR', 'vR'], qconj=-1)
            U, S, VH, err, _ = svd_theta(theta_expand,
                                         engine.trunc_params,
                                         qtotal_LR=[theta.qtotal, None],
                                         inner_labels=['vR', 'vL'])
            VH = VH.split_legs('(wR.vR)')
            VH = VH.take_slice(IdL, 'wR')  # project back such that U-S-VH is original theta
        else:  # move left
            RHeff = _get_RHeff(engine.env, i0, engine.eff_H)  # on site i0, but with p1 label
            RHeff = RHeff.transpose(['(p1*.vL)', 'wL', '(p1.vL*)'])
            if not explicit_plus_hc and IdR is not None:
                theta_expand = npc.tensordot(theta, RHeff.iscale_axis(mix_R, 'wL'),
                                             ['(p0.vR)', '(p1*.vL)'])
                theta_expand.ireplace_label('(p1.vL*)', '(p0.vR)')
            else:
                # need to stack different parts of the wR leg
                wL = RHeff.get_leg('wL')
                stack = [theta.add_trivial_leg(1, 'wL', wL.qconj)]  # explicitly add the identity
                proj = np.ones(wL.ind_len - (IdL is not None) - (IdR is not None), bool)
                if IdL is not None:
                    proj[IdL] = False
                if IdR is not None:
                    proj[IdR] = False
                RHeff.iproject(proj, 'wR')
                stack.append(npc.tensordot(theta, RHeff, ['(p0.vR)', '(p1*.vL)']))
                if explicit_plus_hc:
                    # apply (RHeff^dagger theta) = conj(dot(RHeff.T, theta.conj()))
                    th = npc.tensordot(theta.conj(), RHeff, ['(p0*.vR*)', '(p1.vL*)'])
                    stack.append(th.itranspose(['vL*', 'wL', '(p1*.vL*)']).iconj())
                theta_expand = npc.concatenate(stack, axis='wR')
                IdR = 0  # of the new, concatenated leg.
            theta_expand = theta_expand.combine_legs(['vL', 'wL'], qconj=+1)
            U, S, VH, err, _ = svd_theta(theta_expand,
                                         engine.trunc_params,
                                         qtotal_LR=[theta.qtotal, None],
                                         inner_labels=['vR', 'vL'])
            U = U.split_legs('(vL.wL)')
            U = U.take_slice(IdR, 'wL')  # project back such that U-S-VH is original theta
        return U, S, VH, err, S


class SingleSiteMixer(SubspaceExpansion):
    r"""Deprecated name for the :class:`SubspaceExpansion` class.

    .. deprecated :: 0.5.0
       Instead of `SingleSiteMixer` and `TwoSiteMixer`, directly use :class:`SubspaceExpansion`
       which is compatible with both single-site and two-site DMRG.
    """
    def __init__(self, *args, **kwargs):
        msg = ("The `SingleSiteMixer` and `TwoSiteMixer` have been replaced by the unified "
               "`SubspaceExpansion` class, and\n"
               "all mixers are compatible with both SingleSiteDMRGEngine and TwoSiteDMRGEngine.")
        warnings.warn(msg, category=FutureWarning, stacklevel=2)
        super().__init__(*args, **kwargs)


class TwoSiteMixer(SingleSiteMixer):
    # Both DMRG engines have code in mixed_svd to support both single-site and two-site mixers
    pass


class DensityMatrixMixer(Mixer):
    r"""Mixer based on density matrices.

    This mixer constructs density matrices as described in the original paper :cite:`white2005`.

    The mixer interjects at the svd ``theta = U S VH`` with ``U-> A[i0]`` and `VH -> B[i0+1]``
    being the new tensors in the MPS.
    Given `theta`, one way to get the `U` is to calculate and diagonalize the reduced
    density matrices ``rho_L = tr_R |theta><theta|``,  and similarly diagonalize `rho_R` for `VH`.

    With the mixer, we perturb the `rho_L` when the left environment needs to be updated (i.e.,
    we're moving to the right), and similarly perturb `rho_R` when updating the right environment.
    Note that for iDMRG there are cases where both `rho_R` and `rho_L` are perturbed.

    The perturbation of `rho_L` is

    .. math ::

        rho_L = tr_R(|\theta><\theta|)
        \rightarrow  tr_R(|\theta><\theta|) + a \sum_l h_l tr_R(|\theta><\theta|) h_l^\dagger

    where `a` is the (small) perturbation :attr:`amplitude` and `h_l` are the left parts of
    the Hamiltonian going accross the center bond (i0, i0+1).
    This perturbs singular values on the order of that amplitude.

    Pictorially, the left density matrix `rho_L` is given by::

        |     update_LP=False           update_LP=True
        |
        |    .---theta---.            .---theta----.
        |    |   |   |   |            |   |    \   |
        |            |   |           LP---H0-.  \  |
        |    |   |   |   |            |   |   \  | |
        |    .---theta*--.                  mixL | |
        |                             |   |   /  | |
        |                            LP*--H0*-  /  |
        |                             |   |    /   |
        |                             .---theta*---.

    Here, the `mixL` is a diagonal matrix with mostly the :attr:`amplitude` on the diagonal,
    except for the `IdL` and `IdR` indices of the MPO, which are 1. and 0., respectively.

    The right density matrix `rho_R` is mirrored accordingly.

    Note that the :class:`SubspaceExpansion` mixer does mathematically the same,
    but circumvents the explicit contraction of the

    """
    update_sites = 2

    def perturb_svd(self, engine, theta, i0, update_LP, update_RP):
        """Mix extra terms to theta and perform an SVD.

        We calculate the left and right reduced density using the mixer
        (which might include applications of `H`).
        These density matrices are diagonalized and truncated such that we effectively perform
        a svd for the case ``mixer.amplitude=0``.

        Parameters
        ----------
        engine : :class:`SingleSiteDMRGEngine` | :class:`TwoSiteDMRGEngine`
            The DMRG engine calling the mixer.
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            The optimized wave function, prepared for svd.
        i0 : int
            Site index; `theta` lives on ``i0, i0+1``.
        update_LP : bool
            Whether to calculate the next ``env.LP[i0+1]``, i.e. whether to perturb `rho_L`.
        update_RP : bool
            Whether to calculate the next ``env.RP[i0]``, i.e., whether to perturb `rho_R`.

        Returns
        -------
        U : :class:`~tenpy.linalg.np_conserved.Array`
            Left-canonical part of `theta`. Labels ``'(vL.p0)', 'vR'``.
        S : 2D :class:`~tenpy.linalg.np_conserved.Array`
            General center matrix such that ``theta = U.S.VH``
        VH : :class:`~tenpy.linalg.np_conserved.Array`
            Right-canonical part of `theta`. Labels ``'vL', '(p1.vR)'``.
        err : :class:`~tenpy.algorithms.truncation.TruncationError`
            The truncation error introduced.
        S_a : 1D ndarray
            Approximation of the actual singular values of `theta`.
        """
        rho_L, rho_R = self.mix_rho(engine, theta, i0, update_LP, update_RP)
        return self.svd_from_rho(engine, rho_L, rho_R, theta, i0)

    def svd_from_rho(self, engine, rho_L, rho_R, theta, i0):
        r"""Diagonalize ``rho_L, rho_R`` to rewrite `theta` as ``U S V`` with canonical U/V.

        If `rho_L` and `rho_R` were the actual density matrices of `theta`, this function
        just performs an SVD by diagonalizing `rho_L` with U and `rho_R` with `VH` and then
        rewriting `theta == U (U^\dagger theta VH^\dagger VH) = U S V``.
        Since the actual `rho_L` and `rho_R` passed as arguments are perturbed by `mix_rho`

        Returns
        -------
        U, S, VH, err, S_a:
            As defined in :meth:`perturb_svd`.
        """
        rho_L.itranspose(['(vL.p0)', '(vL*.p0*)'])  # just to be sure of the order
        rho_R.itranspose(['(p1.vR)', '(p1*.vR*)'])  # just to be sure of the order
        # consider the SVD `theta = U S V^H` (with real, diagonal S>0)
        # rho_L ~=  theta theta^H = U S V^H V S U^H = U S S U^H  (for mixer -> 0)
        # Thus, rho_L U = U S S, i.e. columns of U are the eigenvectors of rho_L,
        # eigenvalues are S^2.
        val_L, U = npc.eigh(rho_L)
        U.legs[1] = U.legs[1].to_LegCharge()  # explicit conversion: avoid warning in `iproject`
        U.iset_leg_labels(['(vL.p0)', 'vR'])
        val_L[val_L < 0.] = 0.  # for stability reasons
        val_L /= np.sum(val_L)
        S_a = np.sqrt(val_L)
        keep_L, _, err_L = truncate(S_a, engine.trunc_params)
        U.iproject(keep_L, axes='vR')  # in place
        U = U.gauge_total_charge(1, engine.psi.get_B(i0, form=None).qtotal)
        # rho_R ~=  theta^T theta^* = V^* S U^T U* S V^T = V^* S S V^T  (for mixer -> 0)
        # Thus, rho_R V^* = V^* S S, i.e. columns of V^* are eigenvectors of rho_R
        val_R, Vc = npc.eigh(rho_R)
        Vc.legs[1] = Vc.legs[1].to_LegCharge()
        Vc.iset_leg_labels(['(p1.vR)', 'vL'])
        VH = Vc.itranspose(['vL', '(p1.vR)'])
        val_R[val_R < 0.] = 0.  # for stability reasons
        val_R /= np.sum(val_R)
        keep_R, _, err_R = truncate(np.sqrt(val_R), engine.trunc_params)
        VH.iproject(keep_R, axes='vL')
        VH = VH.gauge_total_charge(0, engine.psi.get_B(i0 + 1, form=None).qtotal)

        # calculate S = U^H theta V
        theta = npc.tensordot(U.conj(), theta, axes=['(vL*.p0*)', '(vL.p0)'])  # axes 0, 0
        theta = npc.tensordot(theta, VH.conj(), axes=['(p1.vR)', '(p1*.vR*)'])  # axes 1, 1
        theta.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
        # normalize `S` (as in svd_theta) to avoid blowing up numbers
        theta /= theta.norm()  # norm(singular values) = norm(whole array)
        S_a = S_a[keep_L]
        return U, theta, VH, err_L + err_R, S_a

    def mix_rho(self, engine, theta, i0, update_LP, update_RP):
        r"""Calculated reduced density matrices of theta with a perturbation by the mixer.

        Parameters
        ----------
        engine : :class:`DMRGEngine`
            The DMRG engine calling the mixer.
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Ground state of the effective Hamiltonian, prepared for svd.
        i0 : int
            Site index; `theta` lives on sites ``i0, i0+1``.
        update_LP, update_RP : bool
            Whether to perturb `rho_L` and `rho_R`, respectively.
            (At least one of them is True when the mixer is enabled.)

        Returns
        -------
        rho_L, rho_R : :class:`~tenpy.linalg.np_conserved.Array`
            A (hermitian) square array with labels ``'(vL.p0)', '(vL*.p0*)'``,
            or ``'(p1.vR)', '(p1*.vR*)'``, respectively.

        """
        eff_H = engine.eff_H
        mix_L, mix_R, IdL, IdR, explicit_plus_hc = self._mix_LR(engine.env.H, i0, sqrt=False)

        if update_LP:
            LHeff = _get_LHeff(engine.env, i0, eff_H)
            rho_L = npc.tensordot(LHeff, theta, axes=['(vR.p0*)', '(vL.p0)'])
            rho_L.ireplace_label('(vR*.p0)', '(vL.p0)')
            rho_c = rho_L.conj()
            rho_L.iscale_axis(mix_L, 'wR')
            rho_L = npc.tensordot(rho_L, rho_c, axes=[['wR', '(p1.vR)'], ['wR*', '(p1*.vR*)']])
            if explicit_plus_hc:
                rho_L = rho_L + rho_L.conj().itranspose()
            if IdL is None:  # can't set mix_L[IdL] = 1.
                rho_L = rho_L + npc.tensordot(theta, theta.conj(), axes=['(p1.vR)', '(p1*.vR*)'])
        else:
            rho_L = npc.tensordot(theta, theta.conj(), axes=['(p1.vR)', '(p1*.vR*)'])

        if update_RP:
            RHeff = _get_RHeff(engine.env, i0 + 1, eff_H)
            rho_R = npc.tensordot(theta, RHeff, axes=['(p1.vR)', '(p1*.vL)'])
            rho_R.ireplace_label('(p1.vL*)', '(p1.vR)')
            rho_c = rho_R.conj()
            rho_R.iscale_axis(mix_R, 'wL')
            rho_R = npc.tensordot(rho_c, rho_R, axes=[['wL*', '(vL*.p0*)'], ['wL', '(vL.p0)']])
            if explicit_plus_hc:
                rho_R = rho_R + rho_R.conj().itranspose()
            if IdR is None:
                rho_R = rho_R + npc.tensordot(theta.conj(), theta, axes=['(vL*.p0*)', '(vL.p0)'])
        else:
            rho_R = npc.tensordot(theta.conj(), theta, axes=['(vL*.p0*)', '(vL.p0)'])
        return rho_L, rho_R


class DMRGEngine(Sweep):
    """DMRG base class with common methods for the TwoSiteDMRG and SingleSiteDMRG.

    This engine is implemented as a subclass of :class:`~tenpy.algorithms.mps_common.Sweep`.
    It contains all methods that are generic between
    :class:`SingleSiteDMRGEngine` and :class:`TwoSiteDMRGEngine`.
    Use the latter two classes for actual DMRG runs.

    A generic protocol for approaching a physics question using DMRG is given in
    :doc:`/intro/dmrg-protocol`.

    .. deprecated :: 0.5.0
        Renamed parameter/attribute `DMRG_params` to :attr:`options`.

    Options
    -------
    .. cfg:config :: DMRGEngine
        :include: Sweep

    Attributes
    ----------
    EffectiveH : class type
        Class for the effective Hamiltonian, i.e., a subclass of
        :class:`~tenpy.algorithms.mps_common.EffectiveH`. Has a `length` class attribute which
        specifies the number of sites updated at once (e.g., whether we do single-site vs. two-site
        DMRG).
    chi_list : dict | ``None``
        See :cfg:option:`DMRGEngine.chi_list`
    eff_H : :class:`~tenpy.algorithms.mps_common.EffectiveH`
        Effective two-site Hamiltonian.
    mixer : :class:`Mixer` | ``None``
        If ``None``, no mixer is used (anymore), otherwise the mixer instance.
    shelve : bool
        If a simulation runs out of time (`time.time() - start_time > max_seconds`), the run will
        terminate with `shelve = True`.
    sweeps : int
        The number of sweeps already performed. (Useful for re-start).
    time0 : float
        Time marker for the start of the run.
    update_stats : dict
        A dictionary with detailed statistics of the convergence at local update-level.
        For each key in the following table, the dictionary contains a list where one value is
        added each time :meth:`DMRGEngine.update_bond` is called.

        =========== ===================================================================
        key         description
        =========== ===================================================================
        i0          An update was performed on sites ``i0, i0+1``.
        ----------- -------------------------------------------------------------------
        age         The number of physical sites involved in the simulation.
        ----------- -------------------------------------------------------------------
        E_total     The total energy before truncation.
        ----------- -------------------------------------------------------------------
        N_lanczos   Dimension of the Krylov space used in the lanczos diagonalization.
        ----------- -------------------------------------------------------------------
        time        Wallclock time evolved since :attr:`time0` (in seconds).
        ----------- -------------------------------------------------------------------
        ov_change   ``1. - abs(<theta_guess|theta_diag>)``, where ``|theta_guess>`` is
                    the initial guess for the wave function and ``|theta_diag>`` is the
                    *untruncated* wave function returned by :meth:`diag`.
        =========== ===================================================================

    sweep_stats : dict
        A dictionary with detailed statistics at the sweep level.
        For each key in the following table, the dictionary contains a list where one value is
        added each time :meth:`Engine.sweep` is called (with ``optimize=True``).

        ============= ===================================================================
        key           description
        ============= ===================================================================
        sweep         Number of sweeps (excluding environment sweeps) performed so far.
        ------------- -------------------------------------------------------------------
        N_updates     Number of updates (including environment sweeps) performed so far.
        ------------- -------------------------------------------------------------------
        E             The energy *before* truncation (as calculated by Lanczos).
        ------------- -------------------------------------------------------------------
        S             Mean entanglement entropy (over bonds).
        ------------- -------------------------------------------------------------------
        max_S         Max entanglement entropy (over bonds).
        ------------- -------------------------------------------------------------------
        time          Wallclock time evolved since :attr:`time0` (in seconds).
        ------------- -------------------------------------------------------------------
        max_trunc_err The maximum truncation error in the last sweep
        ------------- -------------------------------------------------------------------
        max_E_trunc   Maximum change or Energy due to truncation in the last sweep.
        ------------- -------------------------------------------------------------------
        max_chi       Maximum bond dimension used.
        ------------- -------------------------------------------------------------------
        norm_err      Error of canonical form ``np.linalg.norm(psi.norm_test())``.
        ============= ===================================================================

    _entropy_approx : list of {None, 1D array}
        While the mixer is on, the `S` stored in the MPS is a non-diagonal 2D array.
        To check convergence, we use the approximate singular values based on which we truncated
        instead to calculate the entanglement entropy and store it inside this list.
    """
    EffectiveH = None
    DefaultMixer = None

    def __init__(self, psi, model, options, **kwargs):
        options = asConfig(options, self.__class__.__name__)
        self.mixer = None
        self.diag_method = options.get('diag_method', 'default')
        self._entropy_approx = [None] * psi.L  # always left of a given site
        super().__init__(psi, model, options, **kwargs)

    @property
    def DMRG_params(self):
        warnings.warn("renamed self.DMRG_params -> self.options", FutureWarning, stacklevel=2)
        return self.options

    def run(self):
        """Run the DMRG simulation to find the ground state.

        Returns
        -------
        E : float
            The energy of the resulting ground state MPS.
        psi : :class:`~tenpy.networks.mps.MPS`
            The MPS representing the ground state after the simluation,
            i.e. just a reference to :attr:`psi`.

        Options
        -------
        .. cfg:configoptions :: DMRGEngine

            diag_method : str
                Method to be used for diagonalzation, default ``'default'``.
                For possible arguments see :meth:`DMRGEngine.diag`.
            E_tol_to_trunc : float
                It's reasonable to choose the Lanczos convergence criteria
                ``'E_tol'`` not many magnitudes lower than the current
                truncation error. Therefore, if `E_tol_to_trunc` is not
                ``None``, we update `E_tol` of `lanczos_params` to
                ``max_E_trunc*E_tol_to_trunc``,
                restricted to the interval [`E_tol_min`, `E_tol_max`],
                where ``max_E_trunc`` is the maximal energy difference due to
                truncation right after each Lanczos optimization during the
                sweeps.
            E_tol_max : float
                See `E_tol_to_trunc`
            E_tol_min : float
                See `E_tol_to_trunc`
            max_E_err : float
                Convergence if the change of the energy in each step
                satisfies ``-Delta E / max(|E|, 1) < max_E_err``. Note that
                this is also satisfied if ``Delta E > 0``,
                i.e., if the energy increases (due to truncation).
            max_hours : float
                If the DMRG took longer (measured in wall-clock time),
                'shelve' the simulation, i.e. stop and return with the flag
                ``shelve=True``.
            max_S_err : float
                Convergence if the relative change of the entropy in each step
                satisfies ``|Delta S|/S < max_S_err``
            max_sweeps : int
                Maximum number of sweeps to be performed.
            min_sweeps : int
                Minimum number of sweeps to be performed.
                Defaults to 1.5*N_sweeps_check.
            N_sweeps_check : int
                Number of sweeps to perform between checking convergence
                criteria and giving a status update.
            norm_tol : float
                After the DMRG run, update the environment with at most
                `norm_tol_iter` sweeps until
                ``np.linalg.norm(psi.norm_err()) < norm_tol``.
            norm_tol_iter : float
                Perform at most `norm_tol_iter`*`update_env` sweeps to
                converge the norm error below `norm_tol`.
            norm_tol_final : float
                After performing `norm_tol_iter`*`update_env` sweeps, if
                ``np.linalg.norm(psi.norm_err()) < norm_tol_final``, call
                :meth:`~tenpy.networks.mps.canonical_form` to canonicalise
                instead. This tolerance should be stricter than `norm_tol`
                to ensure canonical form even if DMRG cannot fully converge.
            P_tol_to_trunc : float
                It's reasonable to choose the Lanczos convergence criteria
                ``'P_tol'`` not many magnitudes lower than the current
                truncation error. Therefore, if `P_tol_to_trunc` is not
                ``None``, we update `P_tol` of `lanczos_params` to
                ``max_trunc_err*P_tol_to_trunc``,
                restricted to the interval [`P_tol_min`, `P_tol_max`],
                where ``max_trunc_err`` is the maximal truncation error
                (discarded weight of the Schmidt values) due to truncation
                right after each Lanczos optimization during the sweeps.
            P_tol_max : float
                See `P_tol_to_trunc`
            P_tol_min : float
                See `P_tol_to_trunc`
            update_env : int
                Number of sweeps without bond optimizaiton to update the
                environment for infinite boundary conditions,
                performed every `N_sweeps_check` sweeps.
        """
        options = self.options
        start_time = self.time0
        self.shelve = False
        # parameters for lanczos
        p_tol_to_trunc = options.get('P_tol_to_trunc', 0.05)
        if p_tol_to_trunc is not None:
            svd_min = self.trunc_params.silent_get('svd_min', 0.)
            svd_min = 0. if svd_min is None else svd_min
            trunc_cut = self.trunc_params.silent_get('trunc_cut', 0.)
            trunc_cut = 0. if trunc_cut is None else trunc_cut
            p_tol_min = max(1.e-30, svd_min**2 * p_tol_to_trunc, trunc_cut**2 * p_tol_to_trunc)
            p_tol_min = options.get('P_tol_min', p_tol_min)
            p_tol_max = options.get('P_tol_max', 1.e-4)
        e_tol_to_trunc = options.get('E_tol_to_trunc', None)
        if e_tol_to_trunc is not None:
            e_tol_min = options.get('E_tol_min', 5.e-16)
            e_tol_max = options.get('E_tol_max', 1.e-4)

        # parameters for DMRG convergence criteria
        N_sweeps_check = options.get('N_sweeps_check', 1 if self.psi.finite else 10)
        min_sweeps = int(1.5 * N_sweeps_check)
        if self.chi_list is not None:
            min_sweeps = max(max(self.chi_list.keys()), min_sweeps)
        min_sweeps = options.get('min_sweeps', min_sweeps)
        max_sweeps = options.get('max_sweeps', 1000)
        max_E_err = options.get('max_E_err', 1.e-8)
        max_S_err = options.get('max_S_err', 1.e-5)
        max_seconds = 3600 * options.get('max_hours', 24 * 365)
        if not self.finite:
            update_env = options.get('update_env', N_sweeps_check // 2)
        E_old, S_old = np.nan, np.mean(self.psi.entanglement_entropy())  # initial dummy values
        E, Delta_E, Delta_S = 1., 1., 1.
        self.diag_method = options['diag_method']

        self.mixer_activate()
        is_first_sweep = True
        # loop over sweeps
        while True:
            loop_start_time = time.time()
            # check convergence criteria
            if self.sweeps >= max_sweeps:
                break
            if (self.sweeps > min_sweeps and -Delta_E < max_E_err * max(abs(E), 1.)
                    and abs(Delta_S) < max_S_err):
                if self.mixer is None:
                    break
                else:
                    logger.info("Convergence criterium reached with enabled mixer. "
                                "Disable mixer and continue")
                    self.mixer = None
                    self.S_inv_cutoff = 1.e-15
            if loop_start_time - start_time > max_seconds:
                self.shelve = True
                logger.warning("DMRG: maximum time limit reached. Shelve simulation.")
                break
            if not is_first_sweep:
                self.checkpoint.emit(self)
            # --------- the main work --------------
            logger.info('Running sweep with optimization')
            for i in range(N_sweeps_check - 1):
                self.sweep(meas_E_trunc=False)
            max_trunc_err = self.sweep(meas_E_trunc=True)
            max_E_trunc = np.max(self.E_trunc_list)
            # --------------------------------------
            # update lancos_params depending on truncation error(s)
            if p_tol_to_trunc is not None and max_trunc_err > p_tol_min:
                P_tol = max(p_tol_min, min(p_tol_max, max_trunc_err * p_tol_to_trunc))
                self.lanczos_params['P_tol'] = P_tol
                self.lanczos_params.touch('P_tol')  # don't warn about unused P_tol, since
                # the optimization might not even use the normal lanczos function.
                logger.debug("set lanczos_params['P_tol'] = %.2e", P_tol)
            if e_tol_to_trunc is not None and max_E_trunc > e_tol_min:
                E_tol = max(e_tol_min, min(e_tol_max, max_E_trunc * e_tol_to_trunc))
                self.lanczos_params['E_tol'] = E_tol
                self.lanczos_params.touch('E_tol')
                logger.debug("set lanczos_params['E_tol'] = %.2e", E_tol)
            # update environment
            if not self.finite:
                self.environment_sweeps(update_env)

            # update values for checking the convergence
            entropy_bonds = self._entropy_approx
            if self.finite:
                entropy_bonds = entropy_bonds[1:]
            max_S = max(entropy_bonds)
            S = sum(entropy_bonds) / len(entropy_bonds)  # mean
            Delta_S = (S - S_old) / N_sweeps_check
            S_old = S
            if not self.finite:  # iDMRG: need energy density
                Es = self.update_stats['E_total']
                age = self.update_stats['age']
                delta = min(1 + 2 * self.env.L, len(age))
                growth = (age[-1] - age[-delta])
                E = (Es[-1] - Es[-delta]) / growth
            else:
                E = self.update_stats['E_total'][-1]
            Delta_E = (E - E_old) / N_sweeps_check
            E_old = E
            norm_err = np.linalg.norm(self.psi.norm_test())

            # update statistics
            self.sweep_stats['sweep'].append(self.sweeps)
            self.sweep_stats['N_updates'].append(len(self.update_stats['i0']))
            self.sweep_stats['E'].append(E)
            self.sweep_stats['S'].append(S)
            self.sweep_stats['max_S'].append(max_S)
            self.sweep_stats['time'].append(time.time() - start_time)
            self.sweep_stats['max_trunc_err'].append(max_trunc_err)
            self.sweep_stats['max_E_trunc'].append(max_E_trunc)
            self.sweep_stats['max_chi'].append(np.max(self.psi.chi))
            self.sweep_stats['norm_err'].append(norm_err)

            # status update
            logger.info(
                "checkpoint after sweep %(sweeps)d\n"
                "energy=%(E).16f, max S=%(S).16f, age=%(age)d, norm_err=%(norm_err).1e\n"
                "Current memory usage %(mem).1fMB, wall time: %(wall_time).1fs\n"
                "Delta E = %(dE).4e, Delta S = %(dS).4e (per sweep)\n"
                "max trunc_err = %(trunc_err).4e, max E_trunc = %(E_trunc).4e\n"
                "chi: %(chi)s\n"
                "%(sep)s", {
                    'sweeps': self.sweeps,
                    'E': E,
                    'S': max_S,
                    'age': self.update_stats['age'][-1],
                    'norm_err': norm_err,
                    'mem': memory_usage(),
                    'wall_time': time.time() - loop_start_time,
                    'dE': Delta_E,
                    'dS': Delta_S,
                    'trunc_err': max_trunc_err,
                    'E_trunc': max_E_trunc,
                    'chi': self.psi.chi if self.psi.L < 40 else max(self.psi.chi),
                    'sep': "=" * 80,
                })
            is_first_sweep = False

        # clean up from mixer
        self.mixer_cleanup()

        self._canonicalize(True)
        logger.info("DMRG finished after %d sweeps, max chi=%d", self.sweeps, max(self.psi.chi))
        return E, self.psi

    def _canonicalize(self, warn=False):
        #Update environment until norm_tol is reached. If norm_tol_final
        #is not reached, call canonical_form.
        if self.mixer is not None:
            return
        norm_err = np.linalg.norm(self.psi.norm_test())
        norm_tol = self.options.get('norm_tol', 1.e-5)
        norm_tol_final = self.options.get('norm_tol_final', 1.e-10)
        if not self.finite:
            update_env = self.options['update_env']
            norm_tol_iter = self.options.get('norm_tol_iter', 5)
        if norm_tol is None or (norm_err < norm_tol and norm_err < norm_tol_final):
            return
        if warn and norm_err > norm_tol:
            logger.warning(
                "final DMRG state not in canonical form up to "
                "norm_tol=%.2e: norm_err=%.2e", norm_tol, norm_err)
        if norm_err > norm_tol and not self.finite:
            for _ in range(norm_tol_iter):
                self.environment_sweeps(update_env)
                norm_err = np.linalg.norm(self.psi.norm_test())
                if norm_err <= norm_tol:
                    break
            else:
                logger.warning(
                    "norm_err=%.2e still too high after environment_sweeps", norm_err)
        if norm_err > norm_tol_final:
            self._resume_psi = self.psi.copy()
            if warn and not self.finite:
                logger.warning(
                "final DMRG state not in canonical form up to "
                "norm_tol_final=%.2e: norm_err=%.2e, "
                "calling psi.canonical_form()", norm_tol_final, norm_err)
            self.psi.canonical_form()

    def reset_stats(self, resume_data=None):
        """Reset the statistics, useful if you want to start a new sweep run.

        .. cfg:configoptions :: DMRGEngine

            chi_list : dict | None
                A dictionary to gradually increase the `chi_max` parameter of
                `trunc_params`. The key defines starting from which sweep
                `chi_max` is set to the value, e.g. ``{0: 50, 20: 100}`` uses
                ``chi_max=50`` for the first 20 sweeps and ``chi_max=100``
                afterwards. Overwrites `trunc_params['chi_list']``.
                By default (``None``) this feature is disabled.
            sweep_0 : int
                The number of sweeps already performed. (Useful for re-start).
        """
        super().reset_stats(resume_data)
        self.update_stats = {
            'i0': [],
            'age': [],
            'E_total': [],
            'N_lanczos': [],
            'time': [],
            'err': [],
            'E_trunc': [],
            'ov_change': []
        }
        self.sweep_stats = {
            'sweep': [],
            'N_updates': [],
            'E': [],
            'S': [],
            'max_S': [],
            'time': [],
            'max_trunc_err': [],
            'max_E_trunc': [],
            'max_chi': [],
            'norm_err': []
        }

    def sweep(self, optimize=True, meas_E_trunc=False):
        """One 'sweep' of a the algorithm.

        Iteratate over the bond which is optimized, to the right and
        then back to the left to the starting point.

        Parameters
        ----------
        optimize : bool, optional
            Whether we actually optimize to find the ground state of the effective Hamiltonian.
            (If False, just update the environments).
        meas_E_trunc : bool, optional
            Whether to measure truncation energies.

        Options
        -------
        .. cfg:configoptions :: DMRGEngine

            chi_list_reactivates_mixer : bool
                If True, the mixer is reset/reactivated each time the bond dimension growths
                due to :cfg:option:`DMRGEngine.chi_list`.

        Returns
        -------
        max_trunc_err : float
            Maximal truncation error introduced.
        max_E_trunc : ``None`` | float
            ``None`` if meas_E_trunc is False, else the maximal change of the energy due to the
            truncation.
        """
        # wrapper around tenpy.algorithms.mps_common.Sweep.sweep()
        self._meas_E_trunc = meas_E_trunc
        if (self.options.get('chi_list_reactivates_mixer', True) and optimize
                and self.chi_list is not None):
            new_chi_max = self.chi_list.get(self.sweeps, None)
            if new_chi_max is not None:
                # growing the bond dimension with chi_list, so we should also reactivate the mixer
                self.mixer_activate()
        res = super().sweep(optimize)
        if optimize:
            # update mixer
            if self.mixer is not None:
                self.mixer = self.mixer.update_amplitude(self.sweeps)
                if self.mixer is None:  # deactivated
                    self.S_inv_cutoff = 1.e-15
        return res

    def update_local(self, theta, optimize=True):
        """Perform site-update on the site ``i0``.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Initial guess for the ground state of the effective Hamiltonian.
        optimize : bool
            Whether we actually optimize to find the ground state of the effective Hamiltonian.
            (If False, just update the environments).

        Returns
        -------
        update_data : dict
            Data computed during the local update, as described in the following:

            E0 : float
                Total energy, obtained *before* truncation (if ``optimize=True``),
                or *after* truncation (if ``optimize=False``) (but never ``None``).
            N : int
                Dimension of the Krylov space used for optimization in the lanczos algorithm.
                0 if ``optimize=False``.
            age : int
                Current size of the DMRG simulation: number of physical sites involved
                into the contraction.
            U, VH: :class:`~tenpy.linalg.np_conserved.Array`
                `U` and `VH` returned by :meth:`mixed_svd`.
            ov_change: float
                Change in the wave function ``1. - abs(<theta_guess|theta>)``
                induced by :meth:`diag`, *not* including the truncation!
        """
        i0 = self.i0
        n_opt = self.n_optimize
        age = self.env.get_LP_age(i0) + n_opt + self.env.get_RP_age(i0 + n_opt - 1)
        if optimize:
            E0, theta, N, ov_change = self.diag(theta)
        else:
            E0, N, ov_change = None, 0, 0.
        theta = self.prepare_svd(theta)
        U, S, VH, err, S_approx = self.mixed_svd(theta)
        self._entropy_approx[(i0 + n_opt - 1) % self.psi.L] = entropy(S_approx**2)
        self.set_B(U, S, VH)
        update_data = {
            'E0': E0,
            'err': err,
            'N': N,
            'age': age,
            'U': U,
            'VH': VH,
            'ov_change': ov_change
        }
        return update_data

    def post_update_local(self, E0, age, N, ov_change, err, **update_data):
        """Perform post-update actions.

        Compute truncation energy and collect statistics.

        Parameters
        ----------
        **update_data : dict
            What was returned by :meth:`update_local`.
        """
        E0 = E0
        i0 = self.i0
        E_trunc = None
        if self._meas_E_trunc or E0 is None:
            i = i0 if self.n_optimize == 2 or self.move_right else i0 - 1
            E_trunc = self.env.full_contraction(i).real  # uses updated LP/RP (if calculated)
            if E0 is None:
                E0 = E_trunc
            E_trunc = E_trunc - E0

        # collect statistics
        self.update_stats['i0'].append(i0)
        self.update_stats['age'].append(age)
        self.update_stats['E_total'].append(E0)
        self.update_stats['E_trunc'].append(E_trunc)
        self.update_stats['N_lanczos'].append(N)
        self.update_stats['ov_change'].append(ov_change)
        self.update_stats['err'].append(err)
        self.update_stats['time'].append(time.time() - self.time0)
        self.trunc_err_list.append(err.eps)
        self.E_trunc_list.append(E_trunc)

    def diag(self, theta_guess):
        """Diagonalize the effective Hamiltonian represented by self.

        .. cfg:configoptions :: DMRGEngine

            max_N_for_ED : int
                Maximum matrix dimension of the effective hamiltonian
                up to which the ``'default'`` `diag_method` uses ED instead of
                Lanczos.
            diag_method : str
                One of the folloing strings:

                'default'
                      Same as ``'lanczos'`` for large bond dimensions, but if the
                      total dimension of the effective Hamiltonian does not exceed
                      the DMRG parameter ``'max_N_for_ED'`` it uses ``'ED_block'``.
                'lanczos'
                      :func:`~tenpy.linalg.lanczos.lanczos`
                      Default, the Lanczos implementation in TeNPy.
                'arpack'
                      :func:`~tenpy.linalg.lanczos.lanczos_arpack`
                      Based on :func:`scipy.linalg.sparse.eigsh`.
                      Slower than 'lanczos', since it needs to convert the npc arrays
                      to numpy arrays during *each* matvec, and possibly does many
                      more iterations.
                'ED_block'
                      :func:`full_diag_effH`
                      Contract the effective Hamiltonian to a (large!) matrix and
                      diagonalize the block in the charge sector of the initial state.
                      Preserves the charge sector of the explicitly conserved charges.
                      However, if you don't preserve a charge explicitly, it can break
                      it.
                      For example if you use a ``SpinChain({'conserve': 'parity'})``,
                      it could change the total "Sz", but not the parity of 'Sz'.
                'ED_all'
                      :func:`full_diag_effH`
                      Contract the effective Hamiltonian to a (large!) matrix and
                      diagonalize it completely.
                      Allows to change the charge sector *even for explicitly
                      conserved charges*.
                      For example if you use a ``SpinChain({'conserve': 'Sz'})``,
                      it **can** change the total "Sz".

        Parameters
        ----------
        theta_guess : :class:`~tenpy.linalg.np_conserved.Array`
            Initial guess for the ground state of the effective Hamiltonian.

        Returns
        -------
        E0 : float
            Energy of the found ground state.
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Ground state of the effective Hamiltonian.
        N : int
            Number of Lanczos iterations used. ``-1`` if unknown.
        ov_change : float
            Change in the wave function ``1. - abs(<theta_guess|theta_diag>)``
        """
        N = -1  # (unknown)

        if self.diag_method == 'default':
            # use ED for small matrix dimensions, but lanczos by default
            max_N = self.options.get('max_N_for_ED', 400)
            if self.eff_H.N < max_N:
                E, theta = full_diag_effH(self.eff_H, theta_guess, keep_sector=True)
            else:
                E, theta, N = lanczos(self.eff_H, theta_guess, self.lanczos_params)
        elif self.diag_method == 'lanczos':
            E, theta, N = lanczos(self.eff_H, theta_guess, self.lanczos_params)
        elif self.diag_method == 'arpack':
            E, theta = lanczos_arpack(self.eff_H, theta_guess, self.lanczos_params)
        elif self.diag_method == 'ED_block':
            E, theta = full_diag_effH(self.eff_H, theta_guess, keep_sector=True)
        elif self.diag_method == 'ED_all':
            E, theta = full_diag_effH(self.eff_H, theta_guess, keep_sector=False)
        else:
            raise ValueError("Unknown diagonalization method: " + repr(self.diag_method))
        ov_change = 1. - abs(npc.inner(theta_guess, theta, 'labels', do_conj=True))
        return E, theta, N, ov_change

    def plot_update_stats(self, axes, xaxis='time', yaxis='E', y_exact=None, **kwargs):
        """Plot :attr:`update_stats` to display the convergence during the sweeps.

        Parameters
        ----------
        axes : :class:`matplotlib.axes.Axes`
            The axes to plot into. Defaults to :func:`matplotlib.pyplot.gca()`
        xaxis : ``'N_updates' | 'sweep'`` | keys of :attr:`update_stats`
            Key of :attr:`update_stats` to be used for the x-axis of the plots.
            ``'N_updates'`` is just enumerating the number of bond updates,
            and ``'sweep'`` corresponds to the sweep number (including environment sweeps).
        yaxis : ``'E'`` | keys of :attr:`update_stats`
            Key of :attr:`update_stats` to be used for the y-axisof the plots.
            For 'E', use the energy (per site for infinite systems).
        y_exact : float
            Exact value for the quantity on the y-axis for comparison.
            If given, plot ``abs((y-y_exact)/y_exact)`` on a log-scale yaxis.
        **kwargs :
            Further keyword arguments given to ``axes.plot(...)``.
        """
        if axes is None:
            import matplotlib.pyplot as plt
            axes = plt.gca()
        stats = self.update_stats
        L = self.psi.L
        kwargs.setdefault('marker', 'x')
        kwargs.setdefault('linestyle', '-')

        E = np.array(stats['E_total'])
        schedule = list(self.get_sweep_schedule())
        N = len(schedule)  # bond updates per sweep
        if xaxis is None or xaxis == 'N_updates' or xaxis == 'index':
            xaxis = 'N_updates'
            x = np.arange(len(E))
        elif xaxis == 'sweep':
            x = np.arange(1, len(E) + 1) / N
        else:
            x = np.array(stats[xaxis])
        if yaxis == 'E':
            if not self.psi.finite:
                # use energy per site instead of total energy
                age = np.array(stats['age'])
                d_age = age[N:] - age[:-N]
                d_E = E[N:] - E[:-N]
                y = d_E / d_age
                x = x[N:]
            else:
                y = E
        else:
            y = np.array(stats[yaxis])
        if y_exact is not None:
            y = np.abs(y - y_exact) / np.abs(y_exact)
            axes.set_yscale('log')
        axes.plot(x, y, **kwargs)
        axes.set_xlabel(xaxis)
        axes.set_ylabel(yaxis)

    def plot_sweep_stats(self, axes=None, xaxis='time', yaxis='E', y_exact=None, **kwargs):
        """Plot :attr:`sweep_stats` to display the convergence with the sweeps.

        Parameters
        ----------
        axes : :class:`matplotlib.axes.Axes`
            The axes to plot into. Defaults to :func:`matplotlib.pyplot.gca()`
        xaxis, yaxis : key of :attr:`sweep_stats`
            Key of :attr:`sweep_stats` to be used for the x-axis and y-axis of the plots.
        y_exact : float
            Exact value for the quantity on the y-axis for comparison.
            If given, plot ``abs((y-y_exact)/y_exact)`` on a log-scale yaxis.
        **kwargs :
            Further keyword arguments given to ``axes.plot(...)``.
        """
        if axes is None:
            import matplotlib.pyplot as plt
            axes = plt.gca()
        stats = self.sweep_stats
        L = self.psi.L
        kwargs.setdefault('marker', 'x')
        kwargs.setdefault('linestyle', '-')

        x = np.array(stats[xaxis])
        y = np.array(stats[yaxis])
        if y_exact is not None:
            y = np.abs(y - y_exact) / np.abs(y_exact)
            axes.set_yscale('log')
        axes.plot(x, y, **kwargs)
        axes.set_xlabel(xaxis)
        axes.set_ylabel(yaxis)

    def mixer_activate(self):
        """Set `self.mixer` to the class specified by `options['mixer']`.

        .. cfg:configoptions :: DMRGEngine

            mixer : str | class | bool
                Chooses the :class:`Mixer` to be used.
                A string stands for one of the mixers defined in this module,
                a class is used as custom mixer.
                Default (``None``) uses no mixer, ``True`` uses
                :class:`DensityMatrixMixer` for the 2-site case and
                :class:`SubspaceExpansion` for the 1-site case.
                :class:`TwoSiteDMRGEngine` only supports two-site mixers,
                but :class:`SingleSiteDMRGEngine` supports both single-site and two-site mixers.
            mixer_params : dict
                Mixer parameters as described in :cfg:config:`Mixer`.
        """
        default = True if isinstance(self, SingleSiteDMRGEngine) else None
        Mixer_class = self.options.get('mixer', default)
        if Mixer_class:
            if Mixer_class is True:
                Mixer_class = self.DefaultMixer
            if isinstance(Mixer_class, str):
                if Mixer_class == "Mixer":
                    msg = 'Use `True` instead of "Mixer" for DMRG parameter "mixer"'
                    warnings.warn(msg, FutureWarning)
                    Mixer_class = self.DefaultMixer
                else:
                    Mixer_class = find_subclass(Mixer, Mixer_class)
            mixer_params = self.options.subconfig('mixer_params')
            self.mixer = Mixer_class(mixer_params, self.sweeps)
            self.S_inv_cutoff = 1.e-8
            logger.info("activate %s with initial amplitude %.1e", Mixer_class.__name__,
                        self.mixer.amplitude)

    def mixer_cleanup(self):
        """Cleanup the effects of a mixer.

        A :meth:`sweep` with an enabled :class:`Mixer` leaves the MPS `psi` with 2D arrays in `S`.
        To recover the originial form, this function simply performs one sweep with disabled mixer.
        """
        if any([self.psi.get_SL(i).ndim > 1 for i in range(self.psi.L)]):
            mixer = self.mixer
            self.mixer = None  # disable the mixer
            self.sweep(optimize=False)  # (discard return value)
            self.mixer = mixer  # recover the original mixer


class TwoSiteDMRGEngine(DMRGEngine):
    """Engine for the two-site DMRG algorithm.

    Parameters
    ----------
    psi : :class:`~tenpy.networks.mps.MPS`
        Initial guess for the ground state, which is to be optimized in-place.
    model : :class:`~tenpy.models.MPOModel`
        The model representing the Hamiltonian for which we want to find the ground state.
    options : dict
        Further optional parameters.

    Options
    -------
    .. cfg:config :: TwoSiteDMRGEngine
        :include: DMRGEngine

    Attributes
    ----------
    eff_H : :class:`~tenpy.algorithms.mps_common.EffectiveH`
        Effective two-site Hamiltonian.
    mixer : :class:`Mixer` | ``None``
        If ``None``, no mixer is used (anymore), otherwise the mixer instance.
    shelve : bool
        If a simulation runs out of time (`time.time() - start_time > max_seconds`), the run will
        terminate with ``shelve = True``.
    sweeps : int
        The number of sweeps already performed. (Useful for re-start).
    time0 : float
        Time marker for the start of the run.
    update_stats : dict
        A dictionary with detailed statistics of the convergence.
        For each key in the following table, the dictionary contains a list where one value is
        added each time :meth:`DMRGEngine.update_bond` is called.

        =========== ===================================================================
        key         description
        =========== ===================================================================
        i0          An update was performed on sites ``i0, i0+1``.
        ----------- -------------------------------------------------------------------
        age         The number of physical sites involved in the simulation.
        ----------- -------------------------------------------------------------------
        E_total     The total energy before truncation.
        ----------- -------------------------------------------------------------------
        N_lanczos   Dimension of the Krylov space used in the lanczos diagonalization.
        ----------- -------------------------------------------------------------------
        time        Wallclock time evolved since :attr:`time0` (in seconds).
        =========== ===================================================================

    sweep_stats : dict
        A dictionary with detailed statistics of the convergence.
        For each key in the following table, the dictionary contains a list where one value is
        added each time :meth:`DMRGEngine.sweep` is called (with ``optimize=True``).

        ============= ===================================================================
        key           description
        ============= ===================================================================
        sweep         Number of sweeps performed so far.
        ------------- -------------------------------------------------------------------
        E             The energy *before* truncation (as calculated by Lanczos).
        ------------- -------------------------------------------------------------------
        S             Maximum entanglement entropy.
        ------------- -------------------------------------------------------------------
        time          Wallclock time evolved since :attr:`time0` (in seconds).
        ------------- -------------------------------------------------------------------
        max_trunc_err The maximum truncation error in the last sweep
        ------------- -------------------------------------------------------------------
        max_E_trunc   Maximum change or Energy due to truncation in the last sweep.
        ------------- -------------------------------------------------------------------
        max_chi       Maximum bond dimension used.
        ------------- -------------------------------------------------------------------
        norm_err      Error of canonical form ``np.linalg.norm(psi.norm_test())``.
        ============= ===================================================================
    """
    EffectiveH = TwoSiteH
    DefaultMixer = DensityMatrixMixer

    def prepare_svd(self, theta):
        """Transform theta into matrix for svd."""
        if self.combine:
            return theta  # Theta is already combined.
        else:
            return theta.combine_legs([['vL', 'p0'], ['p1', 'vR']],
                                      new_axes=[0, 1],
                                      qconj=[+1, -1])

    def mixed_svd(self, theta):
        """Get (truncated) `B` from the new theta (as returned by diag).

        The goal is to split theta and truncate it::

            |   -- theta --   ==>    -- U -- S --  VH -
            |      |   |                |          |

        Without a mixer, this is done by a simple svd and truncation of Schmidt values.

        With a mixer, the state is perturbed before the SVD.
        The details of the perturbation are defined by the :class:`Mixer` class.

        Note that the returned `S` is a general (not diagonal) matrix, with labels ``'vL', 'vR'``.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            The optimized wave function, prepared for svd.

        Returns
        -------
        U : :class:`~tenpy.linalg.np_conserved.Array`
            Left-canonical part of `theta`. Labels ``'(vL.p)', 'vR'``.
        S : 1D ndarray | 2D :class:`~tenpy.linalg.np_conserved.Array`
            Without mixer just the singluar values of the array; with mixer it might be a general
            matrix with labels ``'vL', 'vR'``; see comment above.
        VH : :class:`~tenpy.linalg.np_conserved.Array`
            Right-canonical part of `theta`. Labels ``'vL', '(p.vR)'``.
        err : :class:`~tenpy.algorithms.truncation.TruncationError`
            The truncation error introduced.
        S_approx : ndarray
            Just the `S` if a 1D ndarray, or an approximation of the correct S (which was used for
            truncation) in case `S` is 2D Array.
        """
        i0 = self.i0
        update_LP, update_RP = self.update_LP_RP
        mixer = self.mixer
        if mixer is None:
            # simple case: real svd, defined elsewhere.
            qtotal_i0 = self.env.bra.get_B(i0, form=None).qtotal
            U, S, VH, err, _ = svd_theta(theta,
                                         self.trunc_params,
                                         qtotal_LR=[qtotal_i0, None],
                                         inner_labels=['vR', 'vL'])
            S_a = S
        elif mixer.update_sites == 2:
            U, S, VH, err, S_a = mixer.perturb_svd(self, theta, self.i0, update_LP, update_RP)
        elif mixer.update_sites == 1:
            if update_LP and update_RP:
                # sub-space expand left site by treating p1 as part of vR leg
                theta_L = theta.replace_label('(p1.vR)', 'vR')
                U, _, _, err_L, S_a = mixer.perturb_svd(self, theta_L, self.i0, True)
                U = U.gauge_total_charge(1, self.psi.get_B(i0, form=None).qtotal)
                # sub-space expand right site by treating p0 as part of vL leg
                theta_R = theta.replace_labels(['(vL.p0)', '(p1.vR)'], ['vL', '(p0.vR)'])
                _, _, VH, err_R, S_a = mixer.perturb_svd(self, theta_R, self.i0 + 1, False)
                VH = VH.gauge_total_charge(0, self.psi.get_B(i0 + 1, form=None).qtotal)
                # calculate S = U^H theta V
                theta = npc.tensordot(U.conj(), theta, axes=['(vL*.p0*)', '(vL.p0)'])
                theta = npc.tensordot(theta, VH.conj(), axes=['(p1.vR)', '(p0*.vR*)'])
                theta.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
                theta /= np.linalg.norm(npc.svd(theta, compute_uv=False))
                S = theta
                err = err_L + err_R
                VH.ireplace_label('(p0.vR)', '(p1.vR)')
            elif update_LP:
                # sub-space expand left site by treating p1 as part of vR leg
                theta.ireplace_label('(p1.vR)', 'vR')
                U, S, VH, err, S_a = mixer.perturb_svd(self, theta, self.i0, True)
                # note: VH is not isometry, but we don't update_RP
                VH.ireplace_label('vR', '(p1.vR)')
            elif update_RP:
                # sub-space expand right site by treating p0 as part of vL leg
                theta.ireplace_labels(['(vL.p0)', '(p1.vR)'], ['vL', '(p0.vR)'])
                U, S, VH, err, S_a = mixer.perturb_svd(self, theta, self.i0 + 1, False)
                # note: U not isometry, but we don't update_LP
                U.ireplace_label('vL', '(vL.p0)')
                VH.ireplace_label('(p0.vR)', '(p1.vR)')
            else:
                assert False
        else:
            assert False, "mixer acting on wired number of sites"
        U.ireplace_label('(vL.p0)', '(vL.p)')
        VH.ireplace_label('(p1.vR)', '(p.vR)')
        return U, S, VH, err, S_a

    def set_B(self, U, S, VH):
        """Update the MPS with the ``U, S, VH`` returned by `self.mixed_svd`.

        Parameters
        ----------
        U, VH : :class:`~tenpy.linalg.np_conserved.Array`
            Left and Right-canonical matrices as returned by the SVD.
        S : 1D array | 2D :class:`~tenpy.linalg.np_conserved.Array`
            The middle part returned by the SVD, ``theta = U S VH``.
            Without a mixer just the singular values, with enabled `mixer` a 2D array.
        """
        B0 = U.split_legs(['(vL.p)'])
        B1 = VH.split_legs(['(p.vR)'])
        i0 = self.i0
        self.psi.set_B(i0, B0, form='A')  # left-canonical
        self.psi.set_B(i0 + 1, B1, form='B')  # right-canonical
        self.psi.set_SR(i0, S)
        # environments are cleaned/updated in :meth:`update_env`


class SingleSiteDMRGEngine(DMRGEngine):
    """Engine for the single-site DMRG algorithm.

    Parameters
    ----------
    psi : :class:`~tenpy.networks.mps.MPS`
        Initial guess for the ground state, which is to be optimized in-place.
    model : :class:`~tenpy.models.MPOModel`
        The model representing the Hamiltonian for which we want to find the ground state.
    options : dict
        Further optional parameters.

    Options
    -------
    .. cfg:config :: SingleSiteDMRGEngine
        :include: DMRGEngine

    Attributes
    ----------
    chi_list : dict | ``None``
        A dictionary to gradually increase the `chi_max` parameter of `trunc_params`. The key
        defines starting from which sweep `chi_max` is set to the value, e.g. ``{0: 50, 20: 100}``
        uses ``chi_max=50`` for the first 20 sweeps and ``chi_max=100`` afterwards. Overwrites
        `trunc_params['chi_list']``. By default (``None``) this feature is disabled.
    eff_H : :class:`~tenpy.algorithms.mps_common.EffectiveH`
        Effective two-site Hamiltonian.
    mixer : :class:`Mixer` | ``None``
        If ``None``, no mixer is used (anymore), otherwise the mixer instance.
    shelve : bool
        If a simulation runs out of time (`time.time() - start_time > max_seconds`), the run will
        terminate with ``shelve = True``.
    sweeps : int
        The number of sweeps already performed. (Useful for re-start).
    time0 : float
        Time marker for the start of the run.
    update_stats : dict
        A dictionary with detailed statistics of the convergence.
        For each key in the following table, the dictionary contains a list where one value is
        added each time :meth:`DMRGEngine.update_bond` is called.

        =========== ===================================================================
        key         description
        =========== ===================================================================
        i0          An update was performed on sites ``i0, i0+1``.
        ----------- -------------------------------------------------------------------
        age         The number of physical sites involved in the simulation.
        ----------- -------------------------------------------------------------------
        E_total     The total energy before truncation.
        ----------- -------------------------------------------------------------------
        N_lanczos   Dimension of the Krylov space used in the lanczos diagonalization.
        ----------- -------------------------------------------------------------------
        time        Wallclock time evolved since :attr:`time0` (in seconds).
        =========== ===================================================================

    sweep_stats : dict
        A dictionary with detailed statistics of the convergence.
        For each key in the following table, the dictionary contains a list where one value is
        added each time :meth:`DMRGEngine.sweep` is called (with ``optimize=True``).

        ============= ===================================================================
        key           description
        ============= ===================================================================
        sweep         Number of sweeps performed so far.
        ------------- -------------------------------------------------------------------
        E             The energy *before* truncation (as calculated by Lanczos).
        ------------- -------------------------------------------------------------------
        S             Maximum entanglement entropy.
        ------------- -------------------------------------------------------------------
        time          Wallclock time evolved since :attr:`time0` (in seconds).
        ------------- -------------------------------------------------------------------
        max_trunc_err The maximum truncation error in the last sweep
        ------------- -------------------------------------------------------------------
        max_E_trunc   Maximum change or Energy due to truncation in the last sweep.
        ------------- -------------------------------------------------------------------
        max_chi       Maximum bond dimension used.
        ------------- -------------------------------------------------------------------
        norm_err      Error of canonical form ``np.linalg.norm(psi.norm_test())``.
        ============= ===================================================================
    """
    EffectiveH = OneSiteH
    DefaultMixer = SubspaceExpansion

    def prepare_svd(self, theta):
        """Transform theta into matrix for svd.

        In contrast with the 2-site engine, the matrix here depends on the direction we move, as we
        need `'p'` to point away from the direction we are going in.
        """
        if self.combine:
            if self.move_right:
                theta.itranspose(['(vL.p0)', 'vR'])  # ensure the order.
            else:
                theta.itranspose(['vL', '(p0.vR)'])  # ensure the order.
        else:
            if self.move_right:
                theta = theta.combine_legs(['vL', 'p0'], qconj=+1, new_axes=0)
            else:
                theta = theta.combine_legs(['p0', 'vR'], qconj=-1, new_axes=1)
        return theta

    def mixed_svd(self, theta):
        """Get (truncated) `B` from the new theta (as returned by diag).

        The goal is to split theta and truncate it. For a move to the right::

            |             -- theta -- next_B --   ==>    -- U -- S -- VH --
            |                  |        |                   |         |

        For a move to the left::

            |   -- next_A -- theta --   ==>    -- U -- S -- VH --
            |        |         |                  |         |

        Note that `theta` lives on the same site :attr:`i0` in both cases,
        but the sites of `next_A` and `next_B` depend on whether we move right or left.
        The returned `U` and `VH` have the same labels independent of that.

        Without a mixer, this is done by a simple svd and truncation of Schmidt values of theta
        followed by the absorption of `VH` into `next_B` (`U` into `next_A`).

        With a mixer, the state/density matrix is perturbed before the SVD.
        The details of the perturbation are defined by the :class:`Mixer` class.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            The optimized wave function, prepared for svd with :meth:`prepare_svd`,
            i.e., with combined legs.

        Returns
        -------
        U : :class:`~tenpy.linalg.np_conserved.Array`
            Left-canonical part of `theta`. Labels ``'(vL.p)', 'vR'``
        S : 1D ndarray | 2D :class:`~tenpy.linalg.np_conserved.Array`
            Without mixer just the singluar values of the array; with mixer it might be a general
            matrix with labels ``'vL', 'vR'``; see comment above.
        VH : :class:`~tenpy.linalg.np_conserved.Array`
            Right-canonical part of `theta`. Labels ``'vL', '(p.vR)'``.
        err : :class:`~tenpy.algorithms.truncation.TruncationError`
            The truncation error introduced.
        S_approx : ndarray
            Just the `S` if a 1D ndarray, or an approximation of the correct S (which was used for
            truncation) in case `S` is 2D Array.
        """
        mixer = self.mixer
        move_right = self.move_right
        update_LP, update_RP = self.update_LP_RP
        if self.move_right:
            next_B = self.psi.get_B(self.i0 + 1, form='B')
            next_B = next_B.combine_legs(['p', 'vR'], qconj=-1, new_axes=1)
            if update_RP:
                # make sure that `next_B` is in right-canonical form
                assert self.psi.form[(self.i0 + 1) % self.psi.L] == (0., 1.)
        else:
            next_A = self.psi.get_B(self.i0 - 1, form='A')
            next_A = next_A.combine_legs(['vL', 'p'], qconj=1, new_axes=0)
            if update_LP:
                # make sure that `next_A` is in left-canonical form
                assert self.psi.form[(self.i0 - 1) % self.psi.L] == (1., 0.)

        if mixer is None:
            qtotal = [theta.qtotal, None] if move_right else [None, theta.qtotal]
            U, S, VH, err, _ = svd_theta(theta,
                                         self.trunc_params,
                                         qtotal_LR=qtotal,
                                         inner_labels=['vR', 'vL'])
            S_a = S
            # absorb VH/U into next_B/next_A for right/left move
            if move_right:
                # VH is at most truncation, so VH-next_B is still right-canonical,
                # (unless next_B wasn't, but then we don't need to update_RP)
                VH = npc.tensordot(VH, next_B, ['vR', 'vL'])
                U.ireplace_label('(vL.p0)', '(vL.p)')
            else:
                # U is at most truncation, so next_A-U is still left-canonical,
                # (unless next_A wasn't, but then we don't need to update_RP)
                U = npc.tensordot(next_A, U, ['vR', 'vL'])
                VH.ireplace_label('(p0.vR)', '(p.vR)')
        elif mixer.update_sites == 1:
            # single-site mixer
            U, S, VH, err, S_a = mixer.perturb_svd(self, theta, self.i0, move_right)
            # absorb VH/U into S
            if move_right:
                # note: if update_RP, the `next_B` is a right-canonical B from the MPS.
                # Hence we *did* a subspace expansion on it, during the update when we put it
                # into the MPS.
                if isinstance(S, npc.Array):
                    S = npc.tensordot(S, VH, ['vR', 'vL'])
                else:
                    S = VH.iscale_axis(S, 'vL')
                VH = next_B
                U.ireplace_label('(vL.p0)', '(vL.p)')
            else:
                if isinstance(S, npc.Array):
                    S = npc.tensordot(U, S, ['vR', 'vL'])
                else:
                    S = U.iscale_axis(S, 'vR')
                U = next_A
                VH.ireplace_label('(p0.vR)', '(p.vR)')
        elif mixer.update_sites == 2:
            # two-site mixer -> just use two-site theta
            if self.move_right:
                next_B.ireplace_label('(p.vR)', '(p1.vR)')
                theta = npc.tensordot(theta, next_B, axes=['vR', 'vL'])
                i0 = self.i0
            else:
                next_A.ireplace_label('(vL.p)', '(vL.p0)')
                theta.ireplace_label('(p0.vR)', '(p1.vR)')
                theta = npc.tensordot(next_A, theta, axes=['vR', 'vL'])
                i0 = self.i0 - 1
            # and do usual mixer
            U, S, VH, err, S_a = mixer.perturb_svd(self, theta, i0, update_LP, update_RP)
            U.ireplace_label('(vL.p0)', '(vL.p)')
            VH.ireplace_label('(p1.vR)', '(p.vR)')
        else:
            assert False
        return U, S, VH, err, S_a

    def set_B(self, U, S, VH):
        """Update the MPS with the ``U, S, VH`` returned by `self.mixed_svd`.

        Parameters
        ----------
        U, VH : :class:`~tenpy.linalg.np_conserved.Array`
            Left and Right-canonical matrices as returned by the SVD.
        S : 1D array | 2D :class:`~tenpy.linalg.np_conserved.Array`
            The middle part returned by the SVD, ``theta = U S VH``.
            Without a mixer just the singular values, with enabled `mixer` a 2D array.
        """
        i_L, i_R = self._update_env_inds()  # left and right updated sites
        A0 = U.split_legs(['(vL.p)'])
        B1 = VH.split_legs(['(p.vR)'])
        self.psi.set_B(i_L, A0, form='A')  # left-canonical
        self.psi.set_B(i_R, B1, form='B')  # right-canonical
        self.psi.set_SR(i_L, S)
        # environments are cleaned/updated in :meth:`update_env`


class EngineCombine(TwoSiteDMRGEngine):
    r"""Engine which combines legs into pipes as far as possible.

    This engine combines the virtual and physical leg for the left site and right site into pipes.
    This reduces the overhead of calculating charge combinations in the contractions,
    but one :meth:`matvec` is formally more expensive, :math:`O(2 d^3 \chi^3 D)`.

    .. deprecated :: 0.5.0
       Directly use the :class:`TwoSiteDMRGEngine` with the DMRG parameter ``combine=True``.
    """
    def __init__(self, psi, model, DMRG_params):
        msg = ("Old-style engines are deprecated in favor of `Sweep` subclasses.\n"
               "Use `TwoSiteDMRGEngine` with parameter `combine=True` "
               "instead of `EngineCombine`.")
        warnings.warn(msg, category=FutureWarning, stacklevel=2)
        DMRG_params['combine'] = True  # to reproduces old-style engine
        super().__init__(psi, model, DMRG_params)


class EngineFracture(TwoSiteDMRGEngine):
    r"""Engine which keeps the legs separate.

    Due to a different contraction order in :meth:`matvec`, this engine might be faster than
    :class:`EngineCombine`, at least for large physical dimensions and if the MPO is sparse.
    One :meth:`matvec` is :math:`O(2 \chi^3 d^2 W + 2 \chi^2 d^3 W^2 )`.

    .. deprecated :: 0.5.0
       Directly use the :class:`TwoSiteDMRGEngine` with the DMRG parameter ``combine=False``.
    """
    def __init__(self, psi, model, DMRG_params):
        msg = ("Old-style engines are deprecated in favor of `Sweep` subclasses.\n"
               "Use `TwoSiteDMRGEngine` with parameter `combine=False` "
               "instead of `EngineFracture`.")
        warnings.warn(msg, category=FutureWarning, stacklevel=2)
        DMRG_params['combine'] = False  # to reproduces old-style engine
        super().__init__(psi, model, DMRG_params)


def chi_list(chi_max, dchi=20, nsweeps=20):
    """Compute a 'ramping-up' chi_list.

    The resulting chi_list allows to increases `chi` by `dchi` every `nsweeps` sweeps up to a given
    maximal `chi_max`.

    Parameters
    ----------
    chi_max : int
        Final value for the bond dimension.
    dchi :int
        Step size how to increase chi
    nsweeps : int
        Step size for sweeps

    Returns
    -------
    chi_list : dict
        To be used as `chi_list` parameter for DMRG, see :func:`run`.
        Keys increase by `nsweeps`, values by `dchi`, until a maximum of `chi_max` is reached.
    """
    chi_max = int(chi_max)
    nsweeps = int(nsweeps)
    if chi_max < dchi:
        return {0: chi_max}
    chi_list = {}
    for i in range(chi_max // dchi):
        chi = int(dchi * (i + 1))
        chi_list[nsweeps * i] = chi
    if chi < chi_max:
        chi_list[nsweeps * (i + 1)] = chi_max
    return chi_list


def full_diag_effH(effH, theta_guess, keep_sector=True):
    """Perform an exact diagonalization of `effH`.

    This function offers an alternative to :func:`~tenpy.linalg.lanczos.lanczos`.

    Parameters
    ----------
    effH : :class:`~tenpy.algorithms.mps_common.EffectiveH`
        The effective Hamiltonian.
    theta_guess : :class:`~tenpy.linalg.np_conserved.Array`
        Current guess to select the charge sector. Labels as specified by ``effH.acts_on``.
    """
    theta_guess = theta_guess.combine_legs(effH.acts_on, qconj=+1)
    fullH = effH.to_matrix()
    if keep_sector:
        # diagonalize only the block of the charge sector in which `theta_guess` is.
        leg = theta_guess.legs[0]
        qi = leg.get_qindex_of_charges(theta_guess.qtotal)
        block = fullH.get_block(np.array([qi, qi], np.intp))
        if block is None:
            warnings.warn("H is zero in the given block, nothing to diagonalize."
                          "We just return the initial state again.")
            E0 = 0
            theta = theta_guess
        else:
            E, V = np.linalg.eigh(block)
            E0 = E[0]
            theta = theta_guess.zeros_like()
            theta.dtype = np.find_common_type([fullH.dtype, theta_guess.dtype], [])
            theta_block = theta.get_block(np.array([qi], np.intp), insert=True)
            theta_block[:] = V[:, 0]  # copy data into theta
    else:  # allow to change charge sector!
        E, V = npc.eigh(fullH)
        i0 = np.argmin(E)
        E0 = E[i0]
        theta = V.take_slice(i0, 1)
    theta = theta.split_legs([0]).iset_leg_labels(effH.acts_on)
    return E0, theta
