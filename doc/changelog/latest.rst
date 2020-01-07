[latest]
========

Release Notes
-------------
This release contains a major update of the documenation.

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
- :mod:`tenpy.tools.io` for easier input and output to pickle.

Fixed
^^^^^
- Adjust the default DMRG parameter `min_sweeps` if `chi_list` is set.
- Avoid some unnecessary transpositions in MPO environments for MPS sweeps (e.g. in DMRG).
