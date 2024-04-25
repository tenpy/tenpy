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
- MPS classmethod :meth:`~tenpy.networks.mps.MPS.project_onto_charge_sector` added. 
  This makes it possible to project a product state (with entries given by a list or ndarray)
  onto a given ``charge_sector`` (similar to :meth:`~tenpy.networks.purification_mps.PurificationMPS.from_infiniteT_canonical` 
  for a :class:`~tenpy.networks.purification_mps.PurificationMPS`).

Fixed
^^^^^
- The classmethod :meth:`~tenpy.networks.purification_mps.PurificationMPS.from_infiniteT_canonical` 
  can now deal with two independent charges in the `charge_sector`, i.e. as in the :class:`~tenpy.networks.site.SpinHalfFermionSite`.
