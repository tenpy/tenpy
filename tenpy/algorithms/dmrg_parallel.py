"""Parallelized version of DMRG.

.. warning ::
    This module is still under active development. Use with care!
"""
# Copyright 2021 TeNPy Developers, GNU GPLv3

from ..tools.thread import Worker

from ..linalg import np_conserved as npc
from .dmrg import TwoSiteDMRGEngine, SingleSiteDMRGEngine
from .mps_common import OneSiteH, TwoSiteH

__all__ = ["DMRGThreadPlusHC", "TwoSiteHThreadPlusHC"]


class TwoSiteHThreadPlusHC(TwoSiteH):
    """Version of `TwoSiteH` that parallelizes matvec with threads.

    Using threads instead of e.g. MPI parallelization means we don't need to make explicit copies
    of (at least one of) the environment tensors and communication is much cheaper.
    """
    def __init__(self, *args, plus_hc_worker=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._plus_hc_worker = plus_hc_worker
        if not self.combine:
            raise NotImplementedError("works only with combine=True")
        self.RHeff_for_hc = self.RHeff.transpose(['(p1*.vL)', '(p1.vL*)', 'wL'])

    def matvec(self, theta):
        assert self._plus_hc_worker is not None
        res = {}
        self._plus_hc_worker.put_task(self.matvec_hc, theta, return_dict=res, return_key="theta")
        theta = super().matvec(theta)
        self._plus_hc_worker.join_tasks()
        theta_hc = res["theta"]
        return theta + theta_hc

    def matvec_hc(self, theta):
        labels = theta.get_leg_labels()
        theta = theta.conj()  # copy!
        theta = npc.tensordot(theta, self.LHeff, axes=['(vL*.p0*)', '(vR*.p0)'])
        theta = npc.tensordot(self.RHeff_for_hc,
                              theta,
                              axes=[['(p1.vL*)', 'wL'], ['(p1*.vR*)', 'wR']])
        theta.iconj().itranspose()
        theta.ireplace_labels(['(vR*.p0)', '(p1.vL*)'], ['(vL.p0)', '(p1.vR)'])
        return theta

    def to_matrix(self):
        mat = super().to_matrix()
        mat_hc = mat.conj().itranspose()
        mat_hc.iset_leg_labels(mat.get_leg_labels())
        return mat + mat_hc

    def adjoint(self):
        return self


class DMRGThreadPlusHC(TwoSiteDMRGEngine):

    EffectiveH = TwoSiteHThreadPlusHC

    def __init__(self, psi, model, options, **kwargs):
        self._plus_hc_worker = None
        if not model.H_MPO.explicit_plus_hc:
            raise ValueError("works only with `explicit_plus_hc` set!")
        super().__init__(psi, model, options, **kwargs)

    def make_eff_H(self):
        assert self.env.H.explicit_plus_hc
        self.eff_H = self.EffectiveH(self.env,
                                     self.i0,
                                     self.combine,
                                     self.move_right,
                                     plus_hc_worker=self._plus_hc_worker)
        if len(self.ortho_to_envs) > 0:
            self._wrap_ortho_eff_H()

    def run(self):
        # re-initialize worker to allow calling `run()` multiple times
        self._plus_hc_worker = Worker("EffectiveHPlusHC worker", max_queue_size=1, daemon=False)
        with self._plus_hc_worker:
            res = super().run()
        self._plus_hc_worker = None
        return res
