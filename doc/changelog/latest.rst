[latest]
========

Release Notes
-------------
TODO: Summarize the most important changes

Changelog
---------

Backwards incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Drop official support for Python 3.5

Added
^^^^^
- :meth:`~tenpy.networks.mps.MPS.entanglement_entropy_segment2`

Changed
^^^^^^^
- For finite DMRG, :cfg:option:`DMRGEngine.N_sweeps_check` now defaults to 1 instead of 10 (which is still the default for infinite MPS).


Fixed
^^^^^
- nothing yet
