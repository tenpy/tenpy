"""Algorithms for using Purification.

# TODO: include example

"""

# Copyright 2020 TeNPy Developers, GNU GPLv3
import tenpy.linalg.np_conserved as npc
from .mps_common import VariationalApplyMPO, TwoSiteH
from .truncation import svd_theta

__all__ = ['PurificationTwoSiteU', 'PurificationApplyMPO']


class PurificationTwoSiteU(TwoSiteH):
    """Variant of `TwoSiteH` suitable for purification.

    The MPO gets only applied to the physical legs `p0`, `p1`, the ancialla legs `q0`, `q1` of
    `theta` are ignored.
    """
    length = 2
    acts_on = ['vL', 'p0', 'q0', 'p1', 'q1', 'vR']

    # initialization, matvec, combine_theta and adjoint derived from `TwoSiteH` work.
    # to_matrix() should in general multiply with identity on q0/q1; but it isn't used anyways.

    def combine_Heff(self):
        super().combine_Heff()  # almost correct
        self.acts_on = ['(vL.p0)', 'q0', 'q1', '(p1.vR)']  # overwrites class attribute!


class PurificationApplyMPO(VariationalApplyMPO):
    """Variant of `VariationalApplyMPO` suitable for purification."""
    EffectiveH = PurificationTwoSiteU

    def update_local(self, _, optimize=True):
        """Perform local update.

        This simply contracts the environments and `theta` from the `ket` to get an updated
        `theta` for the bra `self.psi` (to be changed in place).
        """
        i0 = self.i0
        self.make_eff_H()
        th = self.env.ket.get_theta(i0, n=2)  # ket is old psi
        th = self.eff_H.combine_theta(th)
        th = self.eff_H.matvec(th)
        if self.eff_H.combine:
            th = th.split_legs()
        th = th.combine_legs([['vL', 'p0', 'q0'], ['p1', 'q1', 'vR']], qconj=[+1, -1])
        return self.update_new_psi(th)

    def update_new_psi(self, theta):
        i0 = self.i0
        new_psi = self.psi
        qtotal_i0 = new_psi.get_B(i0, form=None).qtotal
        U, S, VH, err, renormalize = svd_theta(theta,
                                               self.trunc_params,
                                               qtotal_LR=[qtotal_i0, None],
                                               inner_labels=['vR', 'vL'])
        self.renormalize.append(renormalize)
        # TODO: up to the `renormalize`, we could use `new_psi.set_svd_theta`.
        B0 = U.split_legs(['(vL.p0.q0)']).replace_labels(['p0', 'q0'], ['p', 'q'])
        B1 = VH.split_legs(['(p1.q1.vR)']).replace_labels(['p1', 'q1'], ['p', 'q'])
        new_psi.set_B(i0, B0, form='A')  # left-canonical
        new_psi.set_B(i0 + 1, B1, form='B')  # right-canonical
        new_psi.set_SR(i0, S)
        # the old stored environments are now invalid
        # => delete them to ensure that they get calculated again in :meth:`update_LP` / RP
        for o_env in self.ortho_to_envs:
            o_env.del_LP(i0 + 1)
            o_env.del_RP(i0)
        self.env.del_LP(i0 + 1)
        self.env.del_RP(i0)
        return {'U': U, 'VH': VH, 'err': err}
