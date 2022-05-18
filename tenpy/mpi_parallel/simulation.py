"""Extension of other tenpy classes to employ MPI parallelization."""
# Copyright 2021 TeNPy Developers, GNU GPLv3

#: flag to select contraction method for attaching W to LP/RP, which requires much communication
CONTRACT_W = "sparse"  # you can't change this dynamically....

import warnings
import numpy as np
import os

from ..linalg import np_conserved as npc
from ..algorithms.dmrg import TwoSiteDMRGEngine, DensityMatrixMixer
from ..algorithms.algorithm import Algorithm
from ..algorithms.mps_common import TwoSiteH
from ..simulations.ground_state_search import GroundStateSearch
from ..simulations.simulation import Skip
from ..simulations.simulation import resume_from_checkpoint as _simulation_resume_from_checkpoint
from ..tools.params import asConfig
from ..tools.misc import get_recursive, set_recursive, transpose_list_list
from ..tools.thread import Worker
from ..tools.cache import CacheFile
from ..tools import hdf5_io
from ..networks.mpo import MPOEnvironment

from . import actions
from .helpers import EPSILON
from .distributed import DistributedArray, NodeLocalData

try:
    from mpi4py import MPI
except ImportError:
    pass  # error/warning in mpi_parallel.py


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
        if self.LP.local_part is not None and self.RP.local_part is not None:
            self.N = (self.LP.local_part.get_leg('vR').ind_len * self.W0.get_leg('p0').ind_len *
                      self.W1.get_leg('p1').ind_len * self.RP.local_part.get_leg('vL').ind_len)
        else:
            self.N = 0
        self.combine_Heff(i0, env)
        env._eff_H = self # HACK to give env.full_contraction access to LHeff, RHeff, i0

    def combine_Heff(self, i0, env):
        self.LHeff = env._contract_LP_W(i0, self.LP)
        self.pipeL = self.LHeff.local_part.get_leg('(vR*.p0)')
        self.RHeff = env._contract_W_RP(i0+1, self.RP)
        self.pipeR = self.RHeff.local_part.get_leg('(p1.vL*)')
        self.acts_on = ['(vL.p0)', '(p1.vR)']

    def matvec(self, theta):
        LHeff = self.LHeff
        RHeff = self.RHeff
        return actions.run(actions.matvec, LHeff.node_local,
                           (theta, LHeff.key, RHeff.key))

    def to_matrix(self):
        mat = actions.run(actions.effh_to_matrix, self.LHeff.node_local,
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


class ParallelMPOEnvironment(MPOEnvironment):
    """MPOEnvironment where each node only has its fraction of the MPO wL/wR legs.

    Only the main node initializes this class.
    The other nodes save tensors in their :class:`NodeLocalData`.
    :meth:`get_RP` and :meth:`set_RP` return/expect a
    :class:`~tenpy.mpi_parallel.distributed.DistributedArray`.
    """
    def __init__(self, node_local, mpi_split_params, bra, H, ket, cache=None, **init_env_data):
        assert bra is ket, "could be generalized...."
        self.node_local = node_local
        comm_H = self.node_local.comm
        actions.run(actions.distribute_H, node_local, (H, mpi_split_params))
        super().__init__(bra, H, ket, cache, **init_env_data)
        assert self.L == bra.L == ket.L == H.L

    def _check_compatible_legs(self, init_LP, init_RP, start_env_sites):
        if isinstance(init_LP, DistributedArray):
            assert isinstance(init_RP, DistributedArray)
            # TODO: might be a good idea to check the compatibility nevertheless?!
            return init_LP, init_RP
        return super()._check_compatible_legs(init_LP, init_RP, start_env_sites)

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
                if LP_part.norm() < EPSILON:
                    LP_part = None
                splits.append(LP_part)
            LP = DistributedArray.from_scatter(splits, self.node_local, self._LP_keys[i], True)
            # from_scatter already puts local part in cache
        else:
            # we got a DistributedArray, so this should already be in cache!
            assert self._LP_keys[i] in self.cache
        self._LP_age[i] = age

    def set_RP(self, i, RP, age):
        i = self._to_valid_index(i)
        if not isinstance(RP, DistributedArray): # This should only happen upon initialization
            # during __init__: `RP` is what's loaded/generated from `init_RP`
            proj = self.node_local.projs_L[(i+1) % len(self.node_local.projs_L)]
            splits = []
            for p in proj:
                RP_part = RP.copy()
                RP_part.iproject(p, axes='wL')
                if RP_part.norm() < EPSILON:
                    RP_part = None
                splits.append(RP_part)
            RP = DistributedArray.from_scatter(splits, self.node_local, self._RP_keys[i], True)
            # from_scatter already puts local part in cache
        else:
            # we got a DistributedArray, so this should already be in cache!
            assert self._RP_keys[i] in self.cache
        self._RP_age[i] = age

    def del_LP(self, i):
        """Delete stored part strictly to the left of site `i`."""
        i = self._to_valid_index(i)
        actions.run(actions.cache_del, self.node_local, (self._LP_keys[i], ))
        self._LP_age[i] = None

    def del_RP(self, i):
        """Delete stored part scrictly to the right of site `i`."""
        i = self._to_valid_index(i)
        actions.run(actions.cache_del, self.node_local, (self._RP_keys[i], ))
        self._RP_age[i] = None

    def clear(self):
        """Delete all partial contractions except the left-most `LP` and right-most `RP`."""
        keys = [key for key in self._LP_keys[1:] + self._RP_keys[:-1] if key in self.cache]
        actions.run(actions.cache_del, self.node_local, keys)
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
        res = actions.run(actions.full_contraction, self.node_local, meta)
        return res

    def _contract_LP(self, i, LP):
        """Now also immediately save LP"""
        #          site i
        #  .-        .-         .- A--
        #  |         |  |       |  |
        #  LP-   ->  LP-W- ->   LP-W--
        #  |         |  |       |  |
        #  .-        .-         .- A*-
        assert isinstance(LP, DistributedArray)
        LP_W =  self._contract_LP_W(i, LP)
        return self._attach_A_to_LP_W(i, LP_W)

    def _contract_RP(self, i, RP):
        assert isinstance(RP, DistributedArray)
        W_RP =  self._contract_W_RP(i, RP)
        return self._attach_B_to_W_RP(i, W_RP)

    if CONTRACT_W == "sparse":
        def _contract_LP_W(self, i, LP):
            actions.run(actions.contract_LP_W_sparse, self.node_local, (i, LP.key, "LP_W"), None)
            return DistributedArray("LP_W", self.node_local, False)

        def _contract_W_RP(self, i, RP):
            actions.run(actions.contract_W_RP_sparse, self.node_local, (i, RP.key, "W_RP"), None)
            return DistributedArray("W_RP", self.node_local, False)

    elif CONTRACT_W == "dumb":
        def _contract_LP_W(self, i, LP):
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
                if block is not None:    # I think this will be an issue for finite DMRG and many nodes.  # TODO: is this fixed?
                    pipeL = block.make_pipe(['vR*', 'p0'], qconj=+1)
                    block = block.combine_legs([['vR*', 'p0'], ['vR', 'p0*']], pipes=[pipeL, pipeL.conj()], new_axes=[0, 2]) # vR*.p, wR, vR.p*
                LP_W.append(block)
            return DistributedArray.from_scatter(LP_W, self.node_local, "LP_W", False)

        def _contract_W_RP(self, i, RP):
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
                if block is not None:        # I think this will be an issue for finite DMRG and many nodes.  # TODO ?
                    pipeR = block.make_pipe(['p1', 'vL*'], qconj=-1)
                    block = block.combine_legs([['p1', 'vL*'], ['p1*', 'vL']], pipes=[pipeR, pipeR.conj()], new_axes=[2, 1])
                W_RP.append(block)
            return DistributedArray.from_scatter(W_RP, self.node_local, "W_RP", False)
    else:
        raise ValueError("invalid CONTRACT_W = " + repr(CONTRACT_W))

    def _attach_A_to_LP_W(self, i, LP_W, A=None):
        comm = self.node_local.comm
        local_part = LP_W.local_part
        if A is None:
            A = self.ket.get_B(i, "A")
        if A.ndim == 3:
            A = A.combine_legs(['vL', 'p'], pipes=local_part.get_leg('(vR*.p0)'))
        elif A.ndim != 2:
            raise ValueError("'A' tensor has neither 2 nor 3 legs")

        new_key = self._LP_keys[(i+1) % self.L]
        actions.run(actions.attach_A, self.node_local, (LP_W.key, new_key, A))
        res = DistributedArray(new_key, self.node_local, True)
        return res

    def _attach_B_to_W_RP(self, i, W_RP, B=None):
        comm = self.node_local.comm
        local_part = W_RP.local_part
        if B is None:
            B = self.ket.get_B(i, "B")
        if B.ndim == 3:
            B = B.combine_legs(['p', 'vR'], pipes=local_part.get_leg('(p1.vL*)'))
        elif B.ndim != 2:
            raise ValueError("'B' tensor has neither 2 nor 3 legs")

        new_key = self._RP_keys[(i-1) % self.L]
        actions.run(actions.attach_B, self.node_local, (W_RP.key, new_key, B))
        res = DistributedArray(new_key, self.node_local, True)
        return res

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

        actions.run(actions.cache_optimize, self.node_local, (short_term_keys, preload))


class ParallelDensityMatrixMixer(DensityMatrixMixer):
    def mix_rho(self, engine, theta, i0, update_LP, update_RP):
        assert update_LP or update_RP
        LHeff = engine.eff_H.LHeff
        RHeff = engine.eff_H.RHeff
        data = (theta, i0, self.amplitude, update_LP, update_RP, LHeff.key, RHeff.key)
        rho_L, rho_R = actions.run(actions.mix_rho, engine.main_node_local, data)
        return rho_L, rho_R


class ParallelTwoSiteDMRG(TwoSiteDMRGEngine):
    """

    Options
    -------
    thread_plus_hc: bool
        Whether to use multi-threading to parallelize the matvec.
    mpi_split_params : dict
        Parameters for :func:`~tenpy.mpi_parallel.helpers.split_MPO_leg`.
    """
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
        # finish loading DistributedArray from hdf5 if not done yet.
        for key in ['init_LP', 'init_RP']:
            if key in init_env_data:
                T = init_env_data[key]
                if isinstance(T, DistributedArray):
                    T.finish_load_from_hdf5(self.main_node_local)
        actions.run(actions.node_local_close_hdf5_file,  # does nothing if no file was opened
                   self.main_node_local,
                   ("hdf5_import_file", ))

        mpi_split_params = self.options.subconfig("mpi_split_params")  # see helpers.split_MPO_leg

        # Initialize custom ParallelMPOEnvironment
        cache = self.main_node_local.cache
        self.env = ParallelMPOEnvironment(self.main_node_local, mpi_split_params,
                                          self.psi, H, self.psi, cache=cache, **init_env_data)

    def get_resume_data(self, sequential_simulations=False):
        data = super().get_resume_data(sequential_simulations)
        save_init_env_data = self.options.get('save_init_env_data', 'per_node')
        if 'init_LP' not in data.get('init_env_data', {}):
            return data
        # data['init_env_data']['init_LP'] and 'init_RP' are DistributedArray
        if save_init_env_data == 'gather':
            for key, axis in [('init_LP', 'wR'), ('init_RP', 'wL')]:
                gathered = npc.concatenate(data['init_env_data'][key].gather(), axis=axis)
                perm, gathered = gathered.sort_legcharge(sort=False, bunch=True)
                data['init_env_data'][key] = gathered
        elif save_init_env_data == 'per_node':
            if sequential_simulations:
                # the cache for the next sequential simulation is going to be different!
                # -> need to preserve just the init_LP/init_RP cache data
                data['init_env_data']['init_LP']._keep_alive_beyond_cache()
                data['init_env_data']['init_RP']._keep_alive_beyond_cache()
            else:
                pass  # already in correct form
        elif save_init_env_data == 'drop' or not save_env_data:
            del data['init_env_data']['init_LP']
            del data['init_env_data']['init_RP']
        else:
            raise ValueError(f"don't understand option save_init_env_data={save_init_env_data!r}")
        return data

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


class ParallelDMRGSim(GroundStateSearch):
    default_algorithm = "ParallelTwoSiteDMRG"

    def __init__(self, options, *, comm=None, **kwargs):
        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm_H = comm
        print(f"MPI rank {comm.rank:d} reporting for duty", flush=True)
        if self.comm_H.rank == 0:
            try:
                super().__init__(options, **kwargs)
            except Skip:
                self.comm_H.bcast((actions.DONE, None))
                raise
        else:
            # don't __init__() to avoid locking other files
            # consequence: logging doesn't work on replicas; fall back to print if necessary!
            self.options = options  # don't convert to options to avoid unused warnings
            cwd = self.options.get("directory", None)
            if cwd is not None:
                os.chdir(cwd)  # needed fore relative cache filenames
            # but allow to use context __enter__ and __exit__ to initialize cache
            self.cache = CacheFile.open()

    def init_cache(self):
        # make sure we have a unique dir/file for each rank
        directory = get_recursive(self.options, 'cache_params.directory', default=None)
        filename = get_recursive(self.options, 'cache_params.filename', default=None)
        if filename is not None or directory is not None:
            if filename is not None:
                filename = filename + f"_rank_{self.comm_H.rank:02d}"
                set_recursive(self.options, 'cache_params.filename', filename)
            if directory is not None:
                directory = directory + f"_rank_{self.comm_H.rank:02d}"
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
            if exc_type is None:
                self.comm_H.bcast((actions.DONE, None))
        print(f"MPI rank {self.comm_H.rank:d} signing off", flush=True)

    def init_algorithm(self, **kwargs):
        kwargs.setdefault('comm_H', self.comm_H)
        super().init_algorithm(**kwargs)

    def run(self):
        if self.comm_H.rank == 0:
            return super().run()  # actions.DONE
        else:
            self.replica_run()

    def resume_run(self):
        if self.comm_H.rank == 0:
            return super().resume_run()
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
                actions.replica_main(node_local)
        else:
            node_local.worker = None
            actions.replica_main(node_local)

        # HACK: make sure that `run_sequential_simulations` still works with dummy
        self.engine = _DummyAlgorithm()

    def _save_to_file(self, results, output_filename):
        super()._save_to_file(results, output_filename)
        # close any remaining node_local open hdf5 files (opened by DistributedArray.save_hdf5)
        actions.run(actions.node_local_close_hdf5_file,
                   self.engine.main_node_local,
                   ("hdf5_export_file", ))


class _DummyAlgorithm(Algorithm):
    """Replacement for actual algorithms on the replica nodes.

    This one is only used on replica nodes to make sure
    :func:`~tenpy.simulations.simulation.run_simulation` and
    :func:`~tenpy.simulations.simulation.run_seq_simulations` work without an error on the replica
    nodes.
    """
    def __init__(self, *args, **kwargs):
        pass

    def get_resume_data(self, *args, **kwargs):
        return {}
