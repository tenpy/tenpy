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
from ..tools.params import asConfig
from ..tools.misc import get_recursive
from ..tools.cache import CacheFile
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
    DISTRIBUTE_H = 1
    CALC_LHeff = 2
    CALC_RHeff = 3
    MATVEC = 4


class ParallelTwoSiteH(TwoSiteH):
    def __init__(self, env, i0, combine=True, move_right=True, comm=None):
        assert comm is not None
        assert combine, 'not implemented for other case'
        self.comm = comm
        self.rank = self.comm.rank
        super().__init__(env, i0, combine, move_right)

    def matvec(self, theta):
        self.comm.bcast(ReplicaAction.MATVEC)
        self.comm.bcast(theta)
        theta = self.orig_operator.matvec(theta)
        theta = self.comm.reduce(theta, op=MPI.SUM)
        return theta

    def to_matrix(self):
        return self.orig_operator.to_matrix() + self.orig_operator.adjoint().to_matrix()


def attach_W_to_LP():
    raise NotImplementedError("TODO")

class DistributedArray:
    """Represents a npc Array which is distributed over the MPI nodes.

    Each node only saves a fraction `local_part` of the actual represented Tensor.
    """
    def __init__(self, key, local_part, comm):
        self.key = key
        self.local_part = local_part
        self.comm = comm

    @classmethod
    def from_scatter(cls, all_parts, node_local):
        comm = node_local.comm
        assert len(all_parts) == comm.size
        local_part = comm.scatter(all_parts, root=0)




class ParallelMPOEnvironment(MPOEnvironment):
    """

    The environment is completely distributed over the different nodes; each node only has its
    fraction of the MPO wL/wR legs.
    Only the main node initializes this class,
    the other nodes save stuff in their :class:`NodeLocalData`.

    Compared to the usual MPOEnvironment, we actually only save the
    ``LP--W`` contractions which already include the W on the next site to the right
    (or left for `RP`).
    """
    def __init__(self, node_local, bra, H, ket, cache=None, **init_env_data):
        self.node_local = node_local
        comm_H = self.node_local.comm
        comm_H.bcast(ReplicaAction.DISTRIBUTE_H)
        comm_H.bcast(H)
        self.node_local.add_H(H)

        super().__init__(*args, **kwargs)


    def get_LP(self, i):
        """Returns only the part for the main node"""
        # find nearest available LP to the left.
        for i0 in range(i, i - self.L, -1):
            key = self._LP_keys[self._to_valid_index(i0)]
            LP = self.cache.get(key, None)
            if LP is not None:
                break
            # (for finite, LP[0] should always be set, so we should abort at latest with i0=0)
        else:  # no break called
            raise ValueError("No left part in the system???")
        # communicate to other nodes to
        age = self.get_LP_age(i0)
        for j in range(i0, i):
            LP = self._contract_LP(j, LP)
            age = age + 1
            if store:
                self.set_LP(j + 1, LP, age=age)
        return LP
        #  LP = super().get_LP(i0)
        # TODO XXX
        return DistributedArray("LP_{i:d}")

    def get_RP(self, i0):
        # change this to return only the part for the main node
        raise NotImplementedError("TODO")


    def set_LP(self, i, LP, age):
        # later on: `LP` is a DistributedArray
        if isinstance(LP, DistributedArray):
            # during the DMRG run
            raise NotImplementedError("TODO")
        else:
            # during __init__: `LP` is what's loaded/generated from `init_LP`
            proj = self.node_local.projs_L[i]
            splits = [LP.project(p, axes='wR') for p in proj]
            comm = self.node_local.comm
            comm.bcast(ReplicaAction.INIT_LP)
            my_part = DistributedArray.from_scatter(splits, comm)
            my_new_part = attach_W_to_LP(my_part, self.node_local)
            super().set_LP(i, LP, age)
            # split LP into N_nodes part and scatter

    def _contract_LP(self, i0):
        """Now also immediately save LP"""
        #     i0         A
        #  LP W  ->   LP W  W
        #                A*


        raise NotImplementedError("TODO")

    def _contract_RP(self, i0):
        raise NotImplementedError("TODO")


class ParallelTwoSiteDMRG(TwoSiteDMRGEngine):
    def __init__(self, psi, model, options, *, comm_H, **kwargs):
        options.setdefault('combine', True)
        self.comm_H = comm_H
        self.main_node_local = NodeLocalData(self.comm_H, kwargs['cache'])
        super().__init__(psi, model, options, **kwargs)
        print('combine = ', self.options.get('combine', 'not initialized'))


    def make_eff_H(self):
        assert self.combine
        self.eff_H = ParallelTwoSiteH(self.env, self.i0, True, self.move_right, self.comm_H)

        if len(self.ortho_to_envs) > 0:
            raise NotImplementedError("TODO: Not supported (yet)")

    def _init_MPO_env(self, H, init_env_data):
        # Initialize custom ParallelMPOEnvironment
        cache = self.cache.create_subcache('env')
        self.env = ParallelMPOEnvironment(self.main_node_local,
                                          self.psi, H, self.psi, cache=cache, **init_env_data)


class NodeLocalData:
    def __init__(self, comm, cache):
        # TODO initialize cache
        self.comm = comm
        i = comm.rank
        self.cache = cache

    def add_H(self, H):
        self.H = H
        N = self.comm.size
        self.W_blocks = []
        self.projs_L = []
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
            self.projs_L.append(projs_L)
            self.W_blocks.append(blocks)


class ParallelDMRGSim(GroundStateSearch):

    default_algorithm = "ParallelTwoSiteDMRG"

    def __init__(self, options, *, comm=None, **kwargs):
        # TODO: generalize such that it works if we have sequential simulations
        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm_H = comm.Dup()
        print("MPI rank %d reporting for duty" % comm.rank)
        if self.comm_H.rank == 0:
            super().__init__(options, **kwargs)
        else:
            # HACK: monkey-patch `[resume_]run` by `run_replica_plus_hc`
            self.options = asConfig(options, "replica node sim_params")
            self.run = self.resume_run = self.run_replica_plus_hc
            # don't __init__() to avoid locking files
            # but allow to use context __enter__ and __exit__ to initialize cache
            self.cache = CacheFile.open()

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
        print("main is done")
        self.comm_H.bcast(ReplicaAction.DONE)
        return res

    def run_replica_plus_hc(self):
        """Replacement for :meth:`run` used for replicas (as opposed to primary MPI process)."""
        comm = self.comm_H
        self.effH = None
        self.node_local = NodeLocalData(self.comm_H, self.cache)  # cache is initialized node-local
        # TODO: initialize environment nevertheless
        # TODO: initialize how MPO legs are split
        while True:
            action = comm.bcast(None)
            print(f"replic {comm.rank:d} got action {action!s}")
            if action is ReplicaAction.DONE:  # allow to gracefully terminate
                print("finish")
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
            elif action is ReplicaAction.INIT_LP:
                local_part = comm.scatter(None, root=0)

            else:
                raise ValueError("recieved invalid action: " + repr(action))
        # done

    def resume_run(self):
        res = super().resume_run()
        self.comm_H.bcast(ReplicaAction.DONE)
        return res
