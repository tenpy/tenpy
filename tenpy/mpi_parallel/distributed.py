"""Classes/functions for memory management on the different nodes."""

import os
from . import actions


class DistributedArray:
    """Represents a npc Array which is distributed over the MPI nodes.

    Each node only saves a fraction `local_part` of the actual represented Tensor.

    We explicitly disallow to pickle this class! However, we allow to export to hdf5,
    which will generate one HDF5 file per node, see :meth:`save_hdf5`.
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
        actions.run(actions.distr_array_scatter, node_local, (key, in_cache), all_parts)
        res = cls(key, node_local, in_cache)
        return res

    def gather(self):
        """Gather all parts of distributed array to the root node."""
        return actions.run(actions.distr_array_gather, self.node_local, (self.key, self.in_cache))

    def save_hdf5(self, hdf5_saver, h5gr, subpath):
        """Export the distributed Array to Hdf5.

        Very limited functionality: have to have the *same* splitting into different MPI
        ranks (in particular: same MPI size) upon re-loading!

        Hdf5 in general supports writing from multiple MPI threads to a single file, but
        (at least regarding h5py) it seems to be limited to parallely writing a single dataset;
        all MPI threads need to write the same group structure. This is not what we want here...

        Hence, we keep it simple and just open one HDF5 file per MPI rank.
        The filenames are taken to be the same as the main Hdf5 file, except ending with
        "_mpi_{rank}.h5". Note that
        :func:`~tenpy.simulations.mpi_parallel_actions.node_local_close_hdf5_file` needs to be
        called after finishing the safe.
        """
        filename, ext = os.path.splitext(h5gr.file.filename)
        filename_template = filename + '_mpirank_{mpirank:d}' + ext
        hdf5_saver.save(filename_template, subpath + "filename_template")
        hdf5_saver.save(filename_template, subpath + "subpath")
        h5gr.attrs["mpi_size"] = self.node_local.comm.size
        h5gr.attrs["in_cache"] = self.in_cache
        h5gr.attrs["key"] = self.key
        actions.run(actions.distr_array_save_hdf5,
                    self.node_local,
                    (self.key, self.in_cache, filename_template, subpath))

    def _keep_alive_beyond_cache(self):
        """Copy local_part into a node-local but python-global class dictionary.

        Same strucuture as `from_hdf5` such that :meth:`finish_load_from_hdf5` puts the local_part
        back into the (new) cache.
        """
        self._unfinished_load = True
        self._mpi_size = self.node_local.comm.size
        self._subpath = self.key
        self._filename_template = None  # this tells actions.distr_array_load_hdf5 to not use HDF5
        actions.run(actions.distr_array_keep_alive, self.node_local, (self.key, self.in_cache))

    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):
        """*Partially* load instance from a HDF5 file.

        Initializes the class, but does *not* load `node_local` data.

        To finish initialization of the DistributedArray, one needs to explicitly call
        :meth:`finish_load_from_hdf5` with the `node_local` data, which is when we will actually
        load the distributed data.
        """
        res = cls(key=hdf5_loader.get_attr(h5gr, "key"),
                  node_local=None,
                  in_cache=hdf5_loader.get_attr(h5gr, "in_cache"))
        res._filename_template = hdf5_loader.load(subpath + "filename_template")
        res._subpath = hdf5_loader.load(subpath + "subpath")
        res._mpi_size = hdf5_loader.get_attr(h5gr, "mpi_size")
        res._unfinished_load = True
        return res

    def finish_load_from_hdf5(self, node_local):
        """Finish loading the distributed array.

        Note that :func:`~tenpy.simulations.mpi_parallel_actions.node_local_close_hdf5_file`
        needs to be called after all calls to this method.
        """
        self.node_local = node_local
        if not getattr(self, '_unfinished_load', False):
            return  # nothing to do
        if self._mpi_size != node_local.comm.size:
            # note: if necessary, we can generalize to allow at least distribution over more nodes
            raise NotImplementedError("loading from hdf5 with different MPI rank size!")
        actions.run(actions.distr_array_load_hdf5,
                    self.node_local,
                    (self.key, self.in_cache, self._filename_template, self._subpath))
        del self._filename_template
        del self._subpath
        del self._mpi_size
        del self._unfinished_load


class NodeLocalData:
    """Data local to each node.

    Attributes
    ----------
    comm : MPI_Comm
        MPI Communicator.
    cache : :class:`~tenpy.tools.cache.DictCache`
        Local cache exclusive to the node.
    distributed : dict
        Additional node-local "cache" for things to be kept in RAM only.
    projs : list of list of numpy arrays
        For each MPO bond (index `i` left of site `i`) the projection arrays splitting the leg
        over the different nodes.
    IdLR_blocks : list of tuple of tuple
        For each bond the MPO the :meth:`index_in_blocks` of `IdL`, `IdR`.
    local_MPO_chi : list of int
        Local part of the MPO bond dimension on each bond.
    W_blocks : list of list of list of {None, Array}
        For each site the matrix W split by node blocks, with `None` enties for zero blocks.
    sparse_comm_schedule : list of (schedule_L, schedule_R)
        For each site the :func:`~tenpy.simulations.mpi_parallel_actions.sparse_comm_schedule`
        for attaching ``W_blocks[i]`` to the neighboring LP/RP.
        See e.g., :func:`~tenpy.simulations.mpi_parallel_actions.contract_W_RP_sparse`.
    """

    def __init__(self, comm, cache):
        self.comm = comm
        i = comm.rank
        self.cache = cache
        self.distributed = {}  # data from DistributedArray that is not in_cache
        # other class attributes are only added once we have H in actions.distribute_H
