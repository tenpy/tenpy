[latest]
========

Release Notes
-------------
TODO: Summarize the most important changes

Changelog
---------

Backwards incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- nothing yet

Added
^^^^^
- wrappers for the helical and irregular lattice (removing sites) in :meth:`~tenpy.models.model.CouplingMPOModel.init_lattice`.
- nothing yet

Changed
^^^^^^^
- nothing yet

Fixed
^^^^^
- :issue:`145` that :func:`~tenpy.networks.mpo.make_W_II` failed for MPOs with trivial virtual bonds.
- Make :func:`~tenpy.linalg.np_conserved.detect_qtotal` more stable: use the maximal entry instead of the first non-zero one.
