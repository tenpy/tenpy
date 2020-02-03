[latest]
========

Release Notes
-------------
This release contains a major update of the documenation.
Apart from that, it introduces a format how to save and load data (in particular TeNPy classes) to HDF5 files.
See :doc:`/intro/input_output` for more details.

Changelog
---------

Backwards incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- nothing yet
- Remove argument `leg0` from :class:`~tenpy.networks.mpo.MPOGraph.build_MPO`.
- Remove argument `leg0` from :class:`~tenpy.networks.mpo.MPO.from_grids`, instead optionally give *all* `legs` as argument.

Added
^^^^^
- :mod:`tenpy.tools.hdf5_io` with convenience functions for import and output with pickle, as well as an implementation 
  allowing to save and load objects to HDF5 files in the format specified in :doc:`/intro/input_output`.
- `save_hdf5` and `load_hdf5` methods to support saving/loading to HDF5 for the following classes:
  - :class:`~tenpy.linalg.charges.ChargeInfo`
  - :class:`~tenpy.linalg.charges.LegCharge`
  - :class:`~tenpy.linalg.charges.LegPipe`
  - :class:`~tenpy.linalg.np_conserved.Array`
  - :class:`~tenpy.networks.mps.MPS`
  - :class:`~tenpy.networks.mpo.MPO`
- Argument `insert_all_id` for :meth:`tenpy.networks.mpo.MPOGraph.from_terms` and :meth:`~tenpy.networks.mpo.MPOGraph.from_term_list`

Changed
^^^^^^^
- DEFAULT DMRG paramter ``'diag_method'`` from ``'lanczos'`` to ``'default'``, which is the same for large bond
  dimensions, but performs a full exact diagonalization if the effective Hamiltonian has small dimensions.
  The threshold introduced is the new DMRG parameter ``'max_N_for_ED'``.
- Derive the following classes from the new :class:`~tenpy.tools.hdf5_io.Hdf5Exportable` to support saving
  of sites to HDF5:
  - :class:`~tenpy.networks.site.Site`
- By default, for an usual MPO define `IdL` and `IdR` on all bonds. This can generate "dead ends" in the MPO graph of
  finite systems, but it is useful for the `make_WI`/`make_WII` for MPO-exponentiation.


Fixed
^^^^^
- Adjust the default DMRG parameter `min_sweeps` if `chi_list` is set.
- Avoid some unnecessary transpositions in MPO environments for MPS sweeps (e.g. in DMRG).
- :class:`~tenpy.linalg.charges.LegPipe` did not initialize ``self.bunched`` correctly.
