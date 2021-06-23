"""Parallelization with MPI.

This module implements parallelization of DMRG with the MPI framework [MPI]_.
It's based on the python interface of `mpi4py <https://mpi4py.readthedocs.io/>`_,
which needs to be installed when you want to use classes in this module.

.. note ::
    This module is not imported by default, since just importing mpi4py already initializes MPI.
    Hence, if you want to use it, you need to explicitly call
    ``import tenpy.simulation.mpi_parallel`` in your python script.

.. warning ::
    This module is still under active development.
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
from ..algorithms.mps_common import TwoSiteH
from ..simulations.ground_state_search import GroundStateSearch
from ..tools.misc import get_recursive
from ..networks.mpo import MPOEnvironment

__all__ = [
    'ReplicaAction', 'ParallelPlusHcNpcLinearOperator', 'ParallelTwoSiteDMRG', 'ParallelDMRGSim'
]


def split_MPO_leg(leg, N_nodes):
    # TODO: make this more clever
    D = leg.ind_len
    res = []
    for i in range(N_nodes):
        proj = np.zeros(D, dtype=bool)
        proj[D//N *i : D// N * (i+1)] = True
        res.append(proj)
    return res


class ReplicaAction(Enum):
    DONE = 0
    DISTRIBUTE_H = 0
    CALC_LHeff = 1
    CALC_RHeff = 2
    MATVEC = 3


class ParallelTwoSiteH(TwoSiteH):
    def __init__(self, , comm):
        self.comm = comm
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


class ParallelMPOEnvironment(MPOEnvironment):
    """

    The environment is completely distributed over the different nodes; each node only has its
    fraction of the MPO wL/wR legs.
    Only the main node initializes this class,
    the other nodes save stuff in their :class:`NodeLocalData`.

    """

    def get_LP(self, i0):
        # change this to return only the part for the main node
        raise NotImplementedError("TODO")

    def get_RP(self, i0):
        # change this to return only the part for the main node
        raise NotImplementedError("TODO")

    def _contract_LP(self, i0):
        raise NotImplementedError("TODO")

    def _contract_RP(self, i0):
        raise NotImplementedError("TODO")


class ParallelTwoSiteDMRG:
    def __init__(self, psi, model, options, *, comm_H, **kwargs):
        self.comm_H = comm_H
        self.main_node_local = NodeLocalData(self.comm_H.size)
        super().__init__(psi, model, options, **kwargs)


    def make_eff_H(self):

        self.eff_H = ParallelTwoSiteH(self.env, self.i0, self.combine, self.move_right, self.comm)

        if len(self.ortho_to_envs) > 0:
            raise NotImplementedError("TODO: Not supported (yet)")

    def _init_MPO_env(self, H, init_env_data):
        self.comm_H.bcast(ReplicaAction.DISTRIBUTE_H)
        self.comm_H.bcast(H)
        # Initialize custom ParallelMPOEnvironment
        cache = self.cache.create_subcache('env')
        self.env = ParallelMPOEnvironment(self.psi, H, self.psi, cache=cache, **init_env_data)


class NodeLocalData:
    def __init__(self, N_nodes):
        # TODO initialize cache
        self.N_nodes = N_nodes
        pass

    def add_H(self, H):
        self.H = H
        N = self.N_nodes
        self.W_blocks = []
        for i in range(H.L):
            W = H.get_W(i).copy()
            projs_L = split_MPO_leg(W.get_leg('wL'), N)
            projs_R = split_MPO_leg(W.get_leg('wR'), N)
            blocks = []
            for b_L in range(N):
                row = []
                for b_R in range(N):
                    Wblock = W.iproject([projs_L[b_L], projs_R[b_R]], ['wL', 'wR'])
                    if Wblock.norm() < 1.e-14:
                        Wblock = None
                    row.append(Wblock)
                blocks.append(row)
            self.W_blocks.append(blocks)


class ParallelDMRGSim(GroundStateSearch):

    default_algorithm = "ParallelTwoSiteDMRG"

    def __init__(self, options, *, comm=None, **kwargs):
        # TODO: generalize such that it works if we have sequential simulations
        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm_H = comm.Dup()
        if self.comm_H.rank == 0:
            super().__init__(options, **kwargs)
        else:
            # HACK: monkey-patch `[resume_]run` by `run_replica_plus_hc`
            self.run = self.resume_run = self.run_replica_plus_hc
            # don't __init__() to avoid locking files
            self.node_local = NodeLocalData(self.comm_H.size)

    def __delete__(self):
        self.comm_H.Free()
        super().__delete__()

    def init_algorithm(self, **kwargs):
        kwargs.setdefault('comm_H', self.comm_H)
        super().init_algorithm(**kwargs)

    def init_model(self):
        super().init_model()

    def run(self):
        res = super().run()
        self.comm_H.bcast(ReplicaAction.DONE)
        return res

    def run_replica_plus_hc(self):
        """Replacement for :meth:`run` used for replicas (as opposed to primary MPI process)."""
        comm = self.comm_H
        self.effH = None
        # TODO: initialize environment nevertheless
        # TODO: initialize how MPO legs are split
        while True:
            action = comm.bcast(None)
            if action is ReplicaAction.DONE:  # allow to gracefully terminate
                return
            elif action is ReplicaAction.DISTRIBUTE_H:
                H = comm.bcast(None)
                self.node_local.add_H(H)
                # TODO init cache etc
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
        self.comm_H.bcast(ReplicaAction.DONE)
        return res
