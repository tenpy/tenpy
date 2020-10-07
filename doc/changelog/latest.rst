[latest]
========

Release Notes
-------------
We've added a list of all papers using (and citing) TeNPy, see :doc:`/papers_using_tenpy`. Feel free to include your own works!

Changelog
---------

Backwards incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- The :class:`~tenpy.models.lattice.Kagome` lattice did not include all `next_next_nearest_neighbors`.
  (It had only the ones across the hexagon, missing those maiking up a bow-tie.)

Added
^^^^^
- :meth:`~tenpy.networks.mps.MPS.term_correlation_function_right` and 
  :meth:`~tenpy.networks.mps.MPS.term_correlation_function_left`
  for correlation functions with more than one operator on each end.
- Allow to include jupyter notebooks into the documentation; collect example notebooks in [TeNPyNotebooks]_.

Changed
^^^^^^^
- nothing yet

Fixed
^^^^^
- The :class:`~tenpy.models.lattice.IrregularLattice` used the ``'default'`` order of the regular lattice instead of
  whatever the order of the regular lattice was.
- :meth:`~tenpy.networks.mps.MPS.charge_variance` did not work for more than 1 charge.
