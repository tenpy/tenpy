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

Changed
^^^^^^^
- DEFAULT DMRG paramter ``'diag_method'`` from ``'lanczos'`` to ``'default'``, which is the same for large bond
  dimensions, but performs a full exact diagonalization if the effective Hamiltonian has small dimensions.
  The threshold introduced is the new DMRG parameter ``'max_N_for_ED'``.

Added
^^^^^
- :mod:`tenpy.tools.hdf5_io` with convenience functions for import and output with pickle, as well as an implementation 
  allowing to save and load objects to HDF5 files in the format specified in :doc:`/intro/input_output`.

Fixed
^^^^^
- Adjust the default DMRG parameter `min_sweeps` if `chi_list` is set.
- Avoid some unnecessary transpositions in MPO environments for MPS sweeps (e.g. in DMRG).
