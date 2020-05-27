[latest]
========

Release Notes
-------------
TODO: Summarize the most important changes

Changelog
---------

Backwards incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Remove argument `leg0` from :class:`~tenpy.networks.mpo.MPOGraph.build_MPO`.
- Remove argument `leg0` from :class:`~tenpy.networks.mpo.MPO.from_grids`, instead optionally give *all* `legs` as argument.


Added
^^^^^
- Argument `insert_all_id` for :meth:`tenpy.networks.mpo.MPOGraph.from_terms` and :meth:`~tenpy.networks.mpo.MPOGraph.from_term_list`
- implemented the :class:`~tenpy.models.lattice.IrregularLattice`.


Changed
^^^^^^^
- By default, for an usual MPO define `IdL` and `IdR` on all bonds. This can generate "dead ends" in the MPO graph of
  finite systems, but it is useful for the `make_WI`/`make_WII` for MPO-exponentiation.

Fixed
^^^^^
- Wrong results of :meth:`tenpy.networks.mps.MPS.get_total_charge` with ``only_physical_legs=True``.
