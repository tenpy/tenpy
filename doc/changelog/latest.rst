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
- Renamed ``tenpy.networks.mpo.MPOGraph.add_string`` to :meth:`~tenpy.networks.mpo.MPOGraph.add_string_left_to_right`
  as part of the fix for :issue:`148`. Added similar :meth:`~tenpy.networks.mpo.MPOGraph.add_string_left_to_right`.

Fixed
^^^^^
- :issue:`145` that :func:`~tenpy.networks.mpo.make_W_II` failed for MPOs with trivial virtual bonds.
- Make :func:`~tenpy.linalg.np_conserved.detect_qtotal` more stable: use the maximal entry instead of the first non-zero one.
- :issue:`148` that generating MPOs with long-range couplings over multiple MPS unit cells and multi-couplings raised errors.
