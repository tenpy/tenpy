[latest]
========

Release Notes
-------------
TODO: Summarize the most important changes

Changelog
---------

Backwards incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- The :class:`~tenpy.models.lattice.Kagome` lattice did not include all `next_next_nearest_neighbors`.
  (It had only the ones across the hexagon, missing those maiking up a bow-tie.)

Added
^^^^^
- nothing yet

Changed
^^^^^^^
- nothing yet

Fixed
^^^^^
- The :class:`~tenpy.models.lattice.IrregularLattice` used the ``'default'`` order of the regular lattice instead of
  whatever the order of the regular lattice was.
