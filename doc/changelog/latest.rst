[latest]
========

Release Notes
-------------
This release contains a major update of the documentation, which is now hosted by "Read the Docs" at https://tenpy.readthedocs.io/.
Update your bookmark :-)

Apart from that, this release introduces a format how to save and load data (in particular TeNPy classes) to HDF5 files.
See :doc:`/intro/input_output` for more details.
To use that feature, you need to **install** the h5py package (and therefore some version of the HDF5 library).
This is easy with anaconda, ``conda install h5py``, but might be cumbersome on your local computing cluster.
(However, many university computing clusters have some version of HDF5 installed already. Check with your local sysadmin.)

Changelog
---------

Backwards incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Don't save `H_MPO_graph` as model attribute anymore - this also wasn't documented.

Added
^^^^^
- :mod:`tenpy.tools.hdf5_io` with convenience functions for import and output with pickle, as well as an implementation 
  allowing to save and load objects to HDF5 files in the format specified in :doc:`/intro/input_output`.
- human-readable `boundary_conditions` property in :class:`~tenpy.models.lattice.Lattice`.
- `save_hdf5` and `load_hdf5` methods to support saving/loading to HDF5 for the following classes (and their subclasses):
  - :class:`~tenpy.linalg.charges.ChargeInfo`
  - :class:`~tenpy.linalg.charges.LegCharge`
  - :class:`~tenpy.linalg.charges.LegPipe`
  - :class:`~tenpy.linalg.np_conserved.Array`
  - :class:`~tenpy.networks.mps.MPS`
  - :class:`~tenpy.networks.mpo.MPO`
  - :class:`~tenpy.models.lattice.Lattice`

Changed
^^^^^^^
- DEFAULT DMRG paramter ``'diag_method'`` from ``'lanczos'`` to ``'default'``, which is the same for large bond
  dimensions, but performs a full exact diagonalization if the effective Hamiltonian has small dimensions.
  The threshold introduced is the new DMRG parameter ``'max_N_for_ED'``.
- Derive the following classes (and their subclasses) from the new :class:`~tenpy.tools.hdf5_io.Hdf5Exportable`
  to support saving to HDF5:
  - :class:`~tenpy.networks.site.Site`
  - :class:`~tenpy.networks.terms.Terms`
  - :class:`~tenpy.networks.terms.OnsiteTerms`
  - :class:`~tenpy.networks.terms.CouplingTerms`
  - :class:`~tenpy.models.model.Model`, i.e., all model classes.


Fixed
^^^^^
- Adjust the default DMRG parameter `min_sweeps` if `chi_list` is set.
- Avoid some unnecessary transpositions in MPO environments for MPS sweeps (e.g. in DMRG).
- :class:`~tenpy.linalg.charges.LegPipe` did not initialize ``self.bunched`` correctly.
