[latest]
========

Release Notes
-------------
TODO: Summarize the most important changes

If you have ever defined a custom model and used :meth:`~tenpy.models.model.CouplingModel.add_multi_coupling_term` with `plus_hc=True`,
please note :issue:`218`!

Changelog
---------

Backwards incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- nothing yet

Added
^^^^^
- :class:`~tenpy.models.tj_model.tJModel` and :class:`~tenpy.networks.site.SpinHalfHoleSite`
- :class:`~tenpy.algorithms.tebd.QRBasedTEBDEngine`
- :class:`~tenpy.models.clock.ClockModel`, :class:`~tenpy.models.clock.ClockChain` and :class:`~tenpy.models.sites.ClockSite` 

Changed
^^^^^^^
- nothing yet

Fixed
^^^^^
- :meth:`~tenpy.networks.purification_mps.PurificationMPS.from_infiniteT_canonical` should now work with arbitrary
  charges of the original model, and has the option to double the number of charges to separately conserve the charges
  on each the physical and ancialla legs.
- Fix a wrong total charge in :meth:`~tenpy.networks.mpo.MPO.apply_zipup`.
- Fix :issue:`218` that :meth:`~tenpy.models.model.CouplingModel.add_multi_coupling_term` with `plus_hc=True` didn't
  correctly add the hermitian conjugate.
- Fix :issue:`210` that :meth:`~tenpy.models.aklt.AKLTChain.psi_AKLT` had typos and wrong canonical form for finite systems.
