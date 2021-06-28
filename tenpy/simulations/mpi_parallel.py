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
from enum import auto as enum_auto
import numpy as np


try:
    from mpi4py import MPI
except ImportError:
    warnings.warn("mpi4py not installed.")
    MPI = None

from ..linalg import np_conserved as npc
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
        proj[D//N_nodes *i : D// N_nodes * (i+1)] = True
        res.append(proj)
    return res


class ReplicaAction(Enum):
    DONE = enum_auto()
    DISTRIBUTE_H = enum_auto()
    SCATTER_DA = enum_auto()
    GATHER_DA = enum_auto()
    ATTACH_B = enum_auto()
    ATTACH_A = enum_auto()
    CALC_LHeff = enum_auto()
    CALC_RHeff = enum_auto()
    MATVEC = enum_auto()


class ParallelTwoSiteH(TwoSiteH):
    def __init__(self, env, i0, combine=True, move_right=True, comm=None):
        assert comm is not None
        assert combine, 'not implemented for other case'
        self.comm = comm
        self.rank = self.comm.rank
        # TODO: this fails due to LP/RP.get_leg(...) for self.N
        super().__init__(env, i0, combine, move_right)

    def matvec(self, theta):
        self.comm.bcast((ReplicaAction.MATVEC, theta))
        self.comm.bcast(theta)
        theta = self.orig_operator.matvec(theta)
        theta = self.comm.reduce(theta, op=MPI.SUM)
        return theta

    def to_matrix(self):
        return self.orig_operator.to_matrix() + self.orig_operator.adjoint().to_matrix()


class DistributedArray:
    """Represents a npc Array which is distributed over the MPI nodes.

    Each node only saves a fraction `local_part` of the actual represented Tensor.

    Never pickle/save this!
    """
    def __init__(self, key, node_local, in_cache):
        self.key = key
        self.node_local = node_local
        self.in_cache = in_cache

    @property
    def local_part(self):
        if self.in_cache:
            return self.node_local.cache[self.key]
        else:
            return self.node_local.distributed[self.key]

    @local_part.setter
    def local_part(self, local_part):
        if self.in_cache:
            self.node_local.cache[self.key] = local_part
        else:
            self.node_local.distributed[self.key] = local_part

    def __getstate__(self):
        raise ValueError("Never pickle/copy this!")

    @classmethod
    def from_scatter(cls, all_parts, node_local, key, in_cache=True):
        comm = node_local.comm
        assert len(all_parts) == comm.size
        comm.bcast((ReplicaAction.SCATTER_DA, (key, in_cache)))
        local_part = comm.scatter(all_parts, root=0)
        res = cls(key, node_local, in_cache)
        res.local_part = local_part
        return res

    def gather(self):
        comm = self.node_local.comm
        comm.bcast((ReplicaAction.GATHER_DA, (self.key, self.in_cache)))
        all_data = comm.gather(self.local_part, root=0)
        return all_data



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
        comm_H.bcast((ReplicaAction.DISTRIBUTE_H, H))
        self.node_local.add_H(H)
        super().__init__(bra, H, ket, cache, **init_env_data)
        assert self.L == bra.L == ket.L == H.L
        assert bra is ket, "could be generalized...."

    def get_LP(self, i, store=True):
        """Returns only the part for the main node"""
        assert store, "TODO: necessary to fix this? right now we always store!"
        # find nearest available LP to the left.
        for i0 in range(i, i - self.L, -1):
            key = self._LP_keys[self._to_valid_index(i0)]
            if key in self.cache:
                LP = DistributedArray(key, self.node_local, True)
                break
            # (for finite, LP[0] should always be set, so we should abort at latest with i0=0)
        else:  # no break called
            raise ValueError("No left part in the system???")
        # communicate to other nodes to
        age = self.get_LP_age(i0)
        for j in range(i0, i):
            LP = self._contract_LP(j, LP)  # TODO: store keyword for _contract_LP/RP?
            age = age + 1
            if store:
                self.set_LP(j + 1, LP, age=age)
        return LP

    def get_RP(self, i, store=True):
        """Returns only the part for the main node"""
        assert store, "TODO: necessary to fix this? right now we always store!"
        # find nearest available RP to the left.
        for i0 in range(i, i + self.L):
            key = self._RP_keys[self._to_valid_index(i0)]
            if key in self.cache:
                RP = DistributedArray(key, self.node_local, True)
                break
        else:  # no break called
            raise ValueError("No left part in the system???")
        # communicate to other nodes to
        age = self.get_RP_age(i0)
        for j in range(i0, i, -1):
            RP = self._contract_RP(j, RP)
            age = age + 1
            if store:
                self.set_RP(j - 1, RP, age=age)
        return RP

    def set_LP(self, i, LP, age):
        if not isinstance(LP, DistributedArray):
            # during __init__: `LP` is what's loaded/generated from `init_LP`
            proj = self.node_local.projs_L[i]
            splits = []
            for p in proj:
                LP_part = LP.copy()
                LP_part.iproject(p, axes='wR')
                splits.append(LP_part)
            LP = DistributedArray.from_scatter(splits, self.node_local, self._LP_keys[i], True)
            # from_scatter already puts local part in cache
        else:
            # we got a DistributedArray, so this is already in cache!
            assert self._LP_keys[i] in self.cache
        i = self._to_valid_index(i)
        self._LP_age[i] = age

    def set_RP(self, i, RP, age):
        if not isinstance(RP, DistributedArray):
            # during __init__: `RP` is what's loaded/generated from `init_RP`
            proj = self.node_local.projs_L[(i+1) % len(self.node_local.projs_L)]
            splits = []
            for p in proj:
                RP_part = RP.copy()
                RP_part.iproject(p, axes='wL')
                splits.append(RP_part)
            RP = DistributedArray.from_scatter(splits, self.node_local, self._RP_keys[i], True)
            # from_scatter already puts local part in cache
        else:
            # we got a DistributedArray, so this is already in cache!
            assert self._RP_keys[i] in self.cache
        i = self._to_valid_index(i)
        self._RP_age[i] = age

    def _contract_LP(self, i, LP):
        """Now also immediately save LP"""
        #     i0         A
        #  LP W  ->   LP W  W
        #                A*
        assert isinstance(LP, DistributedArray)

        raise NotImplementedError("TODO")  # TODO continue here :)
        LP = npc.tensordot(LP, self.ket.get_B(i, form='A'), axes=('vR', 'vL'))
        axes = (self.ket._get_p_label('*') + ['vL*'], self.ket._p_label + ['vR*'])
        # for a ususal MPS, axes = (['p*', 'vL*'], ['p', 'vR*'])
        LP = npc.tensordot(self.bra.get_B(i, form='A').conj(), LP, axes=axes)
        return LP  # labels 'vR*', 'vR'

    def _contract_RP(self, i, RP):
        assert isinstance(RP, DistributedArray)
        W_RP =  self._contract_W_RP_dumb(i, RP)
        return self._attach_B_to_W_RP(i, W_RP)

    def _contract_W_RP_dumb(self, i, RP):
        RP_parts = RP.gather()
        W_block = self.node_local.W_blocks[i % self.L]
        W_RP = []
        for b_L, row in enumerate(W_block):
            block = None  # contraction of W_RP in row i
            for b_R, W in enumerate(row):
                if W is None:
                    continue
                Wb = npc.tensordot(W, RP_parts[b_R], ['wR', 'wL'])
                if block is None:
                    block = Wb
                else:
                    block = block + Wb
            assert block is not None
            pipeR = block.make_pipe(['p', 'vL*'], qconj=-1)
            #  for Left: pipeL = block.make_pipe(['vR*', 'p'], qconj=+1)
            block = block.combine_legs([['p', 'vL*'], ['p*', 'vL']], pipes=[pipeR, pipeR.conj()])
            W_RP.append(block)
        return DistributedArray.from_scatter(W_RP, self.node_local, "W_RP", False)

    def _attach_B_to_W_RP(self, i, W_RP):
        comm = self.node_local.comm
        B = self.ket.get_B(i, "B")
        new_key = self._RP_keys[(i-1) % self.L]
        comm.bcast((ReplicaAction.ATTACH_B, (W_RP.key, new_key, B)))
        local_part = W_RP.local_part
        B = B.combine_legs(['p', 'vR'], pipes=local_part.get_leg('(p.vL*)'))
        local_part = npc.tensordot(B, local_part, axes=['(p.vR)', '(p*.vL)'])
        local_part = npc.tensordot(B.conj(), local_part, axes=['(p*.vR*)', '(p.vL*)'])
        res = DistributedArray(new_key, self.node_local, True)
        res.local_part = local_part
        return res


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
        cache = self.main_node_local.cache
        # TODO: don't use separte cache for MPSEnvironment, only use it for the MPOEnvironment
        self.env = ParallelMPOEnvironment(self.main_node_local,
                                          self.psi, H, self.psi, cache=cache, **init_env_data)


class NodeLocalData:
    def __init__(self, comm, cache):
        # TODO initialize cache
        self.comm = comm
        i = comm.rank
        self.cache = cache
        self.distributed = {}  # data from DistributedArray that is not in_cache

    def add_H(self, H):
        self.H = H
        N = self.comm.size
        self.W_blocks = [] # index: site
        self.projs_L = []
        for i in range(H.L):
            W = H.get_W(i).copy()
            self.projs_L.append(split_MPO_leg(W.get_leg('wL'), N))
        for i in range(H.L):
            W = H.get_W(i).copy()
            projs_L = self.projs_L[i]
            projs_R = self.projs_L[(i+1) % H.L]
            blocks = []
            for b_L in range(N):
                row = []
                for b_R in range(N):
                    Wblock = W.copy()
                    Wblock.iproject([projs_L[b_L], projs_R[b_R]], ['wL', 'wR'])
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
        print("MPI rank %d reporting for duty" % comm.rank)
        if self.comm_H.rank == 0:
            super().__init__(options, **kwargs)
        else:
            # HACK: monkey-patch `[resume_]run` by `replica_run`
            self.options = asConfig(options, "replica node sim_params")
            self.run = self.resume_run = self.replica_run
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
        self.comm_H.bcast((ReplicaAction.DONE, None))
        return res

    def replica_run(self):
        """Replacement for :meth:`run` used for replicas (as opposed to primary MPI process)."""
        comm = self.comm_H
        self.effH = None
        self.node_local = NodeLocalData(self.comm_H, self.cache)  # cache is initialized node-local
        # TODO: initialize environment nevertheless
        # TODO: initialize how MPO legs are split
        while True:
            action, meta = comm.bcast(None)
            print(f"replic {comm.rank:d} got action {action!s}")
            if action is ReplicaAction.DONE:  # allow to gracefully terminate
                print("finish")
                return
            elif action is ReplicaAction.DISTRIBUTE_H:
                H = meta
                self.node_local.add_H(H)
                # TODO init cache etc
            elif action is ReplicaAction.MATVEC:
                theta = meta
                theta = self.effH.matvec(theta)
                comm.reduce(theta, op=MPI.SUM)
                del theta
            elif action is ReplicaAction.SCATTER_DA:
                key, in_cache = meta
                local_part = comm.scatter(None, root=0)
                if in_cache:
                    self.node_local.cache[key] = local_part
                else:
                    self.node_local.distributed[key] = local_part
            elif action is ReplicaAction.GATHER_DA:
                key, in_cache = meta
                if in_cache:
                    local_part = self.node_local.cache[key]
                else:
                    local_part = self.node_local.distributed[key]
                comm.gather(local_part, root=0)
            elif action is ReplicaAction.ATTACH_B:
                (old_key, new_key, B) = meta
                local_part = self.node_local.distributed[old_key]
                B = B.combine_legs(['p', 'vR'], pipes=local_part.get_leg('(p.vL*)'))
                local_part = npc.tensordot(B, local_part, axes=['(p.vR)', '(p*.vL)'])
                local_part = npc.tensordot(B.conj(), local_part, axes=['(p*.vR*)', '(p.vL*)'])
                self.node_local.cache[new_key] = local_part
            else:
                raise ValueError("recieved invalid action: " + repr(action))
        # done

    def resume_run(self):
        res = super().resume_run()
        self.comm_H.bcast(ReplicaAction.DONE)
        return res
