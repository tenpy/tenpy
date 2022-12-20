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
- nothing yet

Changed
^^^^^^^
- nothing yet

Fixed
^^^^^
- :meth:`~tenpy.networks.purification_mps.PurificationMPS.from_infiniteT_canonical` should now work with arbitrary
  charges of the original model, and has the option to double the number of charges to separately conserve the charges
  on each the physical and ancialla legs.
