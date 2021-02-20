"""Parallelization with MPI.

This module implements parallelization of DMRG with the MPI framework [MPI]_.
It's based on the python interface of `mpi4py <https://mpi4py.readthedocs.io/>`_,
which needs to be installed when you want to use classes in this module.

.. note ::
    This module is not imported by default, since just importing mpi4py already initializes MPI.
    Hence, if you want to use it, you need to explicitly call
    ``import tenpy.simulation.mpi_parallel`` in your python script.
"""
# Copyright 2021 TeNPy Developers, GNU GPLv3

import warnings
from enum import Enum

try:
    from mpi4py import MPI
except ImportError:
    warnings.warn("mpi4py not installed.")
    MPI = None

from ..linalg.sparse import NpcLinearOperatorWrapper
from ..algorithms.dmrg import SingleSiteDMRGEngine, TwoSiteDMRGEngine
from ..simulations.ground_state_search import GroundStateSearch
from ..tools.misc import get_recursive

__all__ = [
    'ReplicaAction', 'ParallelPlusHcNpcLinearOperator', 'ParallelTwoSiteDMRG', 'ParallelDMRGSim'
]


class ReplicaAction(Enum):
    DONE = 0
    RECV_H = 1
    MATVEC = 2


class ParallelPlusHcLinOp(NpcLinearOperatorWrapper):
    def __init__(self, orig_operator, comm):
        super().__init__(orig_operator)
        self.comm = comm
        assert self.comm.size == 2
        self.rank = self.comm.rank
        self.comm.bcast(ReplicaAction.RECV_H)
        self.comm.bcast(orig_operator)

    def matvec(self, theta):
        self.comm.bcast(ReplicaAction.MATVEC)
        self.comm.bcast(theta)
        theta = self.orig_operator.matvec(theta)
        theta = self.comm.reduce(theta, op=MPI.SUM)
        return theta

    def to_matrix(self):
        return self.orig_operator.to_matrix() + self.orig_operator.adjoint().to_matrix()


class ParallelDMRGPlusHc:
    def __init__(self, psi, model, options, comm, **kwargs):
        self.comm_plus_hc = comm
        # TODO: how to handle for multiple parallel layers?
        super().__init__(psi, model, options, **kwargs)

    def make_eff_H(self):
        assert self.env.H.explicit_plus_hc
        self.eff_H = self.EffectiveH(self.env, self.i0, self.combine, self.move_right)
        # note: this order of wrapping is most effective.
        self.eff_H = ParallelPlusHcLinOp(self.eff_H, self.comm_plus_hc)
        if len(self.ortho_to_envs) > 0:
            raise NotImplementedError("Not supported")


class ParallelSingleSiteDMRG(ParallelDMRGPlusHc, SingleSiteDMRGEngine):
    # note: order is important: fixes MRO
    pass


class ParallelTwoSiteDMRG(ParallelDMRGPlusHc, TwoSiteDMRGEngine):
    pass


class ParallelDMRGSim(GroundStateSearch):

    default_algorithm = "ParallelTwoSiteDMRG"

    def __init__(self, options, comm=None):
        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm_plus_hc = comm.Dup()
        if self.comm_plus_hc.size != 2:
            warnings.warn(f"unexpected size of MPI communicator: {self.comm_plus_hc.size:d}")
        if self.comm_plus_hc.rank == 0:
            super().__init__(options)  # TODO: how to handle for multiple parallel layers?
            if not get_recursive(options, "model_params/explicit_plus_hc"):
                raise ValueError("need explicit_plus_hc!")
        else:
            # HACK: monkey-patch `[resume_]run` by `run_replica_plus_hc`
            self.run = self.resume_run = self.run_replica_plus_hc
            # don't __init__() to avoid locking files

    def __delete__(self):
        self.comm_plus_hc.Free()
        super().__delete__()

    def _init_algorithm(self, AlgorithmClass):
        params = self.options.subconfig('algorithm_params')
        self.engine = AlgorithmClass(self.psi, self.model, params, comm=self.comm_plus_hc)

    def run(self):
        res = super().run()
        self.comm_plus_hc.bcast(ReplicaAction.DONE)
        return res

    def run_replica_plus_hc(self):
        """Replacement for :meth:`run` used for replicas (as opposed to primary MPI process)."""
        comm = self.comm_plus_hc
        self.effH = None
        while True:
            action = comm.bcast(None)
            if action is ReplicaAction.DONE:  # allow to gracefully terminate
                return
            elif action is ReplicaAction.RECV_H:
                del self.effH
                self.effH = comm.bcast(None)
                self.effH = self.effH.adjoint()
            elif action is ReplicaAction.MATVEC:
                theta = comm.bcast(None)
                theta = self.effH.matvec(theta)
                comm.reduce(theta, op=MPI.SUM)
                del theta
            else:
                raise ValueError("recieved invalid action: " + repr(action))
        # done

    def resume_run(self):
        res = super().resume_run()
        self.comm_plus_hc.bcast(ReplicaAction.DONE)
        return res
