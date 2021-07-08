"""Parallelization with MPI by splitting the MPO bond amongst several nodes.

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
import numpy as np

from . import mpi_parallel_actions as action

try:
    from mpi4py import MPI
except ImportError:
    warnings.warn("mpi4py not installed.")
    MPI = None

from ..linalg import np_conserved as npc
from ..linalg.sparse import NpcLinearOperatorWrapper
from ..algorithms.truncation import truncate
from ..algorithms.dmrg import SingleSiteDMRGEngine, TwoSiteDMRGEngine, Mixer
from ..algorithms.mps_common import TwoSiteH
from ..simulations.ground_state_search import GroundStateSearch
from ..tools.params import asConfig
from ..tools.misc import get_recursive, set_recursive, transpose_list_list
from ..tools.thread import Worker
from ..tools.cache import CacheFile
from ..networks.mpo import MPOEnvironment

__all__ = [
    'ParallelPlusHcNpcLinearOperator', 'ParallelTwoSiteDMRG', 'ParallelDMRGSim'
]


def split_MPO_leg(leg, N_nodes):
    ''' Split MPO leg indices as evenly as possible amongst N_nodes nodes.'''
    # TODO: make this more clever
    D = leg.ind_len
    res = []
    for i in range(N_nodes):
        proj = np.zeros(D, dtype=bool)
        proj[(D*i)//N_nodes: (D*(i+1))// N_nodes] = True
        res.append(proj)
    return res


def index_in_blocks(block_projs, index):
    if index is None:
        return (-1, None)
    for j, proj in enumerate(block_projs):
        if proj[index]:
            return (j, np.sum(proj[:index]))  # (block index,  index within block)
    assert False, "None of block_projs has `index` True"


class ParallelTwoSiteH(TwoSiteH):
    def __init__(self, env, i0, combine=True, move_right=True):
        assert combine, 'not implemented for other case'
        # super().__init__(env, i0, combine, move_right)
        self.i0 = i0
        self.LP = env.get_LP(i0)
        self.RP = env.get_RP(i0 + 1)
        self.W0 = env.H.get_W(i0).replace_labels(['p', 'p*'], ['p0', 'p0*'])
        # 'wL', 'wR', 'p0', 'p0*'
        self.W1 = env.H.get_W(i0 + 1).replace_labels(['p', 'p*'], ['p1', 'p1*'])
        # 'wL', 'wR', 'p1', 'p1*'
        self.dtype = env.H.dtype
        self.combine = combine
        self.N = (self.LP.local_part.get_leg('vR').ind_len * self.W0.get_leg('p0').ind_len *
                  self.W1.get_leg('p1').ind_len * self.RP.local_part.get_leg('vL').ind_len)
        self.combine_Heff(i0, env)
        env._eff_H = self # HACK to give env.full_contraction access to LHeff, RHeff, i0

    def combine_Heff(self, i0, env):
        self.LHeff = env._contract_LP_W_dumb(i0, self.LP)
        self.pipeL = self.LHeff.local_part.get_leg('(vR*.p0)')
        self.RHeff = env._contract_W_RP_dumb(i0+1, self.RP)
        self.pipeR = self.RHeff.local_part.get_leg('(p1.vL*)')
        self.acts_on = ['(vL.p0)', '(p1.vR)']

    def matvec(self, theta):
        LHeff = self.LHeff
        RHeff = self.RHeff
        return action.run(action.matvec, LHeff.node_local,
                          (theta, LHeff.key, RHeff.key))

    def to_matrix(self):
        mat = action.run(action.effh_to_matrix, self.LHeff.node_local,
                         (self.LHeff.key, self.RHeff.key))
        if self.LHeff.node_local.H.explicit_plus_hc:
            mat_hc = mat.conj().itranspose()
            mat_hc.iset_leg_labels(mat.get_leg_labels())
            mat = mat + mat_hc
        return mat

    def update_LP(self, env, i, U=None):
        assert i == self.i0 + 1
        assert self.combine
        LP = env._attach_A_to_LP_W(i - 1, self.LHeff, A=U)
        env.set_LP(i, LP, age=env.get_LP_age(i - 1) + 1)

    def update_RP(self, env, i, VH=None):
        assert self.combine
        assert i == self.i0
        RP = env._attach_B_to_W_RP(i + 1, self.RHeff, B=VH)
        env.set_RP(i, RP, age=env.get_RP_age(i + 1) + 1)


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
        raise ValueError("Never pickle/copy/save this!")

    @classmethod
    def from_scatter(cls, all_parts, node_local, key, in_cache=True):
        """initialize DsitrubtedArray from all parts on the main node.

        Scatter all_parts array to each node and instruct recipient node to save either in cache
        (for parts that will be saved to disk) or in the node_local.distributed dictionary
        (for parts that will only be used once)."""
        assert len(all_parts) == node_local.comm.size
        action.run(action.scatter_distr_array, node_local, (key, in_cache), all_parts)
        res = cls(key, node_local, in_cache)
        return res

    def gather(self):
        """Gather all parts of distributed array to the root node."""
        return action.run(action.gather_distr_array, self.node_local, (self.key, self.in_cache))


class ParallelMPOEnvironment(MPOEnvironment):
    """

    The environment is completely distributed over the different nodes; each node only has its
    fraction of the MPO wL/wR legs.
    Only the main node initializes this class.
    :meth:`get_RP` and :meth:`set_RP` return/expect a :class:`DistributedArray` representing the
    and saves instances of :class:`DistributedArray`
    the other nodes save stuff in their :class:`NodeLocalData`.
    """
    def __init__(self, node_local, bra, H, ket, cache=None, **init_env_data):
        self.node_local = node_local
        comm_H = self.node_local.comm
        action.run(action.distribute_H, node_local, (H, ))
        super().__init__(bra, H, ket, cache, **init_env_data)
        assert self.L == bra.L == ket.L == H.L
        assert bra is ket, "could be generalized...."

    def get_LP(self, i, store=True):
        """Returns DistributedArray containing the part for the main node"""
        assert store, "necessary to fix this? right now we always store..."
        # find nearest available LP to the left.
        for i0 in range(i, i - self.L, -1):
            key = self._LP_keys[self._to_valid_index(i0)]
            if key in self.cache:
                LP = DistributedArray(key, self.node_local, True)
                break
            # (for finite, LP[0] should always be set, so we should abort at latest with i0=0)
        else:  # no break called
            raise ValueError("No left part in the system???")
        # Contract the found env with MPO and MPS tensors to get the desired env
        age = self.get_LP_age(i0)
        for j in range(i0, i):
            LP = self._contract_LP(j, LP)
            age = age + 1
            if store:
                self.set_LP(j + 1, LP, age=age)
        return LP

    def get_RP(self, i, store=True):
        """Returns DistributedArray containing the part for the main node"""
        assert store, "necessary to fix this? right now we always store..."
        # find nearest available RP to the right.
        for i0 in range(i, i + self.L):
            key = self._RP_keys[self._to_valid_index(i0)]
            if key in self.cache:
                RP = DistributedArray(key, self.node_local, True)
                break
        else:  # no break called
            raise ValueError("No right part in the system???")
        # Contract the found env with MPO and MPS tensors to get the desired env
        age = self.get_RP_age(i0)
        for j in range(i0, i, -1):
            RP = self._contract_RP(j, RP)
            age = age + 1
            if store:
                self.set_RP(j - 1, RP, age=age)
        return RP

    def set_LP(self, i, LP, age):
        i = self._to_valid_index(i)
        if not isinstance(LP, DistributedArray): # This should only happen upon initialization
            # during __init__: `LP` is what's loaded/generated from `init_LP`
            proj = self.node_local.projs_L[i] # At site i, how the left MPO leg is split.
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
        self._LP_age[i] = age

    def set_RP(self, i, RP, age):
        i = self._to_valid_index(i)
        if not isinstance(RP, DistributedArray): # This should only happen upon initialization
            # during __init__: `RP` is what's loaded/generated from `init_RP`
            proj = self.node_local.projs_L[(i+1) % len(self.node_local.projs_L)] # At site i + 1, how the left MPO leg is split, which is the right MPO leg of site i.
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
        self._RP_age[i] = age

    def del_LP(self, i):
        """Delete stored part strictly to the left of site `i`."""
        i = self._to_valid_index(i)
        action.run(action.cache_del, self.node_local, (self._LP_keys[i], ))
        self._LP_age[i] = None

    def del_RP(self, i):
        """Delete stored part scrictly to the right of site `i`."""
        i = self._to_valid_index(i)
        action.run(action.cache_del, self.node_local, (self._RP_keys[i], ))
        self._RP_age[i] = None

    def clear(self):
        """Delete all partial contractions except the left-most `LP` and right-most `RP`."""
        keys = [key for key in self._LP_keys[1:] + self._RP_keys[:-1] if key in self.cache]
        action.run(action.cache_del, self.node_local, keys)
        self._LP_age[1:] = [None] * (self.L - 1)
        self._RP_age[:-1] = [None] * (self.L - 1)

    def full_contraction(self, i0):
        eff_H = getattr(self, "_eff_H", None)   # HACK to have access to previous envs
        if eff_H is None or i0 != eff_H.i0:
            raise NotImplementedError("this shouldn't be needed for DMRG")
        i1 = i0 + 1
        meta = []
        if self.has_LP(i1):
            LP = self.get_LP(i1)
            if self.has_RP(i0):
                case = 0b11
                RP = self.get_RP(i0)
                S_ket = self.ket.get_SR(i0)
                meta = (case, LP.key, LP.in_cache, RP.key, RP.in_cache, S_ket)
            else:
                case = 0b10
                RHeff = eff_H.RHeff
                theta = self.ket.get_theta(i1, 1).replace_label('p0', 'p1')
                theta = theta.combine_legs(['p1', 'vR'], pipes=eff_H.pipeR)
                meta = (case, LP.key, LP.in_cache, RHeff.key, RHeff.in_cache, theta)
        else:
            LHeff = eff_H.LHeff
            if self.has_RP(i0):
                case = 0b01
                RP = self.get_RP(i0)
                theta = self.ket.get_theta(i0, 1)
                theta = theta.combine_legs(['vL', 'p0'], pipes=eff_H.pipeL)
                meta = (case, LHeff.key, LHeff.in_cache, RP.key, RP.in_cache, theta)
            else: # case = 0b00
                assert False, "Not needed!?"
        res = action.run(action.full_contraction, self.node_local, meta)
        return res

    def _contract_LP(self, i, LP):
        """Now also immediately save LP"""
        #          site i0
        #  .-        .-         .- A--
        #  |         |  |       |  |
        #  LP-   ->  LP-W- ->   LP-W--
        #  |         |  |       |  |
        #  .-        .-         .- A*-
        assert isinstance(LP, DistributedArray)
        LP_W =  self._contract_LP_W_dumb(i, LP)
        return self._attach_A_to_LP_W(i, LP_W)

    def _contract_RP(self, i, RP):
        assert isinstance(RP, DistributedArray)
        W_RP =  self._contract_W_RP_dumb(i, RP)
        return self._attach_B_to_W_RP(i, W_RP)

    def _contract_LP_W_dumb(self, i, LP):
        LP_parts = LP.gather()
        W_block = self.node_local.W_blocks[i % self.L] # Get blocks of W[i], the MPO tensor at site i.
        W_block_T = transpose_list_list(W_block)
        LP_W = []
        for b_R, col in enumerate(W_block_T):
            block = None  # contraction of W_RP in row i
            for b_L, W in enumerate(col):
                if W is None:
                    continue
                Wb = npc.tensordot(LP_parts[b_L], W, ['wR', 'wL']).replace_labels(['p', 'p*'], ['p0', 'p0*'])
                if block is None:
                    block = Wb
                else:
                    block = block + Wb
            assert block is not None
            #pipeR = block.make_pipe(['p', 'vL*'], qconj=-1)
            pipeL = block.make_pipe(['vR*', 'p0'], qconj=+1)
            block = block.combine_legs([['vR*', 'p0'], ['vR', 'p0*']], pipes=[pipeL, pipeL.conj()], new_axes=[0, 2]) # vR*.p, wR, vR.p*
            LP_W.append(block)
        return DistributedArray.from_scatter(LP_W, self.node_local, "LP_W", False)

    def _contract_W_RP_dumb(self, i, RP):
        RP_parts = RP.gather()
        W_block = self.node_local.W_blocks[i % self.L]
        W_RP = []
        for b_L, row in enumerate(W_block):
            block = None  # contraction of W_RP in row i
            for b_R, W in enumerate(row):
                if W is None:
                    continue
                Wb = npc.tensordot(W, RP_parts[b_R], ['wR', 'wL']).replace_labels(['p', 'p*'], ['p1', 'p1*'])
                if block is None:
                    block = Wb
                else:
                    block = block + Wb
            assert block is not None
            pipeR = block.make_pipe(['p1', 'vL*'], qconj=-1)
            #  for Left: pipeL = block.make_pipe(['vR*', 'p'], qconj=+1)
            block = block.combine_legs([['p1', 'vL*'], ['p1*', 'vL']], pipes=[pipeR, pipeR.conj()], new_axes=[2, 1])
            W_RP.append(block)
        return DistributedArray.from_scatter(W_RP, self.node_local, "W_RP", False)

    def _attach_A_to_LP_W(self, i, LP_W, A=None):
        comm = self.node_local.comm
        local_part = LP_W.local_part
        if A is None:
            A = self.ket.get_B(i, "A").replace_labels(['p'], ['p0'])
        if A.ndim == 3:
            A = A.combine_legs(['vL', 'p0'], pipes=local_part.get_leg('(vR*.p0)'))
        elif A.ndim != 2:
            raise ValueError("'A' tensor has neither 2 nor 3 legs")

        new_key = self._LP_keys[(i+1) % self.L]
        action.run(action.attach_A, self.node_local, (LP_W.key, new_key, A))
        res = DistributedArray(new_key, self.node_local, True)
        return res

    def _attach_B_to_W_RP(self, i, W_RP, B=None):
        comm = self.node_local.comm
        local_part = W_RP.local_part
        if B is None:
            B = self.ket.get_B(i, "B").replace_labels(['p'], ['p1'])
        if B.ndim == 3:
            B = B.combine_legs(['p1', 'vR'], pipes=local_part.get_leg('(p1.vL*)'))
        elif B.ndim != 2:
            raise ValueError("'B' tensor has neither 2 nor 3 legs")

        new_key = self._RP_keys[(i-1) % self.L]
        action.run(action.attach_B, self.node_local, (W_RP.key, new_key, B))
        res = DistributedArray(new_key, self.node_local, True)
        return res

    def get_initialization_data(self, first=0, last=None):
        data = super().get_initialization_data(first, last)
        data['init_LP'] = data['init_LP'].gather()
        data['init_RP'] = data['init_RP'].gather()
        return data

    def cache_optimize(self, short_term_LP=[], short_term_RP=[], preload_LP=None, preload_RP=None):
        # need to update cache on all nodes
        LP_keys = self._LP_keys
        RP_keys = self._RP_keys
        preload = []
        if preload_LP is not None:
            preload.append(LP_keys[self._to_valid_index(preload_LP)])
        if preload_RP is not None:
            preload.append(RP_keys[self._to_valid_index(preload_RP)])

        short_term_keys = tuple([LP_keys[self._to_valid_index(i)] for i in short_term_LP] +
                                [RP_keys[self._to_valid_index(i)] for i in short_term_RP])

        action.run(action.cache_optimize, self.node_local, (short_term_keys, preload))


class ParallelDensityMatrixMixer(Mixer):
    def perturb_svd(self, engine, theta, i0, update_LP, update_RP):
        rho_L, rho_R = self.mix_rho(engine, theta, i0, update_LP, update_RP)
        # TODO: remainder is copy-pasted from DensityMatrixMixer.perturb_svd ....
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
        keep_L, _, errL = truncate(np.sqrt(val_L), engine.trunc_params)
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
        theta.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])  # for left/right
        # normalize `S` (as in svd_theta) to avoid blowing up numbers
        theta /= np.linalg.norm(npc.svd(theta, compute_uv=False))
        return U, theta, VH, errL + err_R

    def mix_rho(self, engine, theta, i0, update_LP, update_RP):
        assert update_LP or update_RP
        LHeff = engine.eff_H.LHeff
        RHeff = engine.eff_H.RHeff
        data = (theta, i0, self.amplitude, update_LP, update_RP, LHeff.key, RHeff.key)
        rho_LR = action.run(action.mix_rho, engine.main_node_local, data)
        if update_LP and update_RP:
            rho_L, rho_R = rho_LR
        elif update_LP:
            rho_L = rho_LR
            rho_R = npc.tensordot(theta, theta.conj(), axes=[['(vL.p0)'], ['(vL*.p0*)']])
        elif update_RP:
            rho_R = rho_LR
            rho_L = npc.tensordot(theta, theta.conj(), axes=['(p1.vR)', '(p1*.vR*)'])
        return rho_L, rho_R


class ParallelTwoSiteDMRG(TwoSiteDMRGEngine):
    DefaultMixer = ParallelDensityMatrixMixer

    def __init__(self, psi, model, options, *, comm_H, **kwargs):
        options.setdefault('combine', True)
        self.comm_H = comm_H
        self.main_node_local = NodeLocalData(self.comm_H, kwargs['cache'])
        super().__init__(psi, model, options, **kwargs)
        self._plus_hc_worker = None
        self.use_threading_plus_hc = self.options.get('thread_plus_hc', False)
        if self.use_threading_plus_hc and not model.H_MPO.explicit_plus_hc:
            raise ValueError("can't use threading+hc if the model doesn't have explicit_plus_hc.")


    def make_eff_H(self):
        assert self.combine
        self.eff_H = ParallelTwoSiteH(self.env, self.i0, True, self.move_right)
        if len(self.ortho_to_envs) > 0:
            self._wrap_ortho_eff_H()

    def _init_MPO_env(self, H, init_env_data):
        # Initialize custom ParallelMPOEnvironment
        cache = self.main_node_local.cache
        self.env = ParallelMPOEnvironment(self.main_node_local,
                                          self.psi, H, self.psi, cache=cache, **init_env_data)

    def run(self):
        # re-initialize worker to allow calling `run()` multiple times
        if self.use_threading_plus_hc:
            print("using a worker on rank ", self.comm_H.rank)
            worker = Worker("EffectiveHPlusHC worker", max_queue_size=1, daemon=False)
            self.main_node_local.worker = worker
            with worker:
                res = super().run()
            self.main_node_local.worker = None
        else:
            self.main_node_local.worker = None
            res = super().run()
        return res


class NodeLocalData:
    def __init__(self, comm, cache):
        self.comm = comm
        i = comm.rank
        self.cache = cache
        self.distributed = {}  # data from DistributedArray that is not in_cache

    def add_H(self, H):
        self.H = H
        N = self.comm.size
        self.W_blocks = [] # index: site
        self.projs_L = []  # index: bond s.t. projs_L[i] is on bond left of site i
        self.IdLR_blocks = []  # index: bond left of site i
        self.local_MPO_chi = []  # index: bond left of site i
        for i in range(H.L):
            W = H.get_W(i).copy()
            projs = split_MPO_leg(W.get_leg('wL'), N)
            self.projs_L.append(projs)
            self.local_MPO_chi.append(np.sum(projs[self.comm.rank]))
            IdL, IdR = H.IdL[i], H.IdR[i]
            IdLR = (index_in_blocks(projs, IdL), index_in_blocks(projs, IdR))
            self.IdLR_blocks.append(IdLR)
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
                if all([b is None for b in row]):
                    print(row)
                    print(W.to_ndarray())
                    print(projs_L, projs_R)
                    #  assert False

            self.W_blocks.append(blocks)


class ParallelDMRGSim(GroundStateSearch):

    default_algorithm = "ParallelTwoSiteDMRG"

    def __init__(self, options, *, comm=None, **kwargs):
        # TODO: generalize such that it works if we have sequential simulations
        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm_H = comm
        print("MPI rank %d reporting for duty" % comm.rank)
        if self.comm_H.rank == 0:
            super().__init__(options, **kwargs)
        else:
            # don't __init__() to avoid locking other files
            # TODO: how to safely log? should replica create a separate log file?
            self.options = options  # don't convert to options to avoid unused warnings
            # but allow to use context __enter__ and __exit__ to initialize cache
            self.cache = CacheFile.open()

    def init_cache(self):
        # make sure we have a unique dir/file for each rank
        directory = get_recursive(self.options, 'cache_params.directory', default=None)
        filename = get_recursive(self.options, 'cache_params.filename', default=None)
        if filename is not None or directory is not None:
            if filename is not None:
                filename = filename + f"_rank_{self.comm_H.rank:2d}"
                set_recursive(self.options, 'cache_params.filename', filename)
            if directory is not None:
                directory = directory + f"_rank_{self.comm_H.rank:2d}"
                set_recursive(self.options, 'cache_params.directory', directory)
        super().init_cache()

    def __exit__(self, exc_type, exc_value, traceback):
        # Simulation.__enter__ is fine for all ranks
        self.cache.__exit__(exc_type, exc_value, traceback)
        if exc_type is not None and self.comm_H.rank == 0:
            self.logger.exception("simulation abort with the following exception",
                                  exc_info=(exc_type, exc_value, traceback))
        if self.comm_H.rank == 0:
            self.options.warn_unused(True)

    def init_algorithm(self, **kwargs):
        kwargs.setdefault('comm_H', self.comm_H)
        super().init_algorithm(**kwargs)

    def run(self):
        if self.comm_H.rank == 0:
            res = super().run()
            self.comm_H.bcast((action.DONE, None))
            return res
        else:
            self.replica_run()

    def resume_run(self):
        if self.comm_H.rank == 0:
            res = super().resume_run()
            self.comm_H.bcast((action.DONE, None))
            return res
        else:
            self.replica_run()

    def replica_run(self):
        """Replacement for :meth:`run` used for replicas (as opposed to primary MPI process)."""
        node_local = NodeLocalData(self.comm_H, self.cache)  # cache is initialized node-local
        use_threading_plus_hc = get_recursive(self.options, 'algorithm_params.thread_plus_hc',
                                              default=False)
        if use_threading_plus_hc:
            print("using a worker on rank ", self.comm_H.rank)
            worker = Worker("EffectiveHPlusHC worker", max_queue_size=1, daemon=False)
            node_local.worker = worker
            with worker:
                action.replica_main(node_local)
        else:
            node_local.worker = None
            action.replica_main(node_local)
        # done
