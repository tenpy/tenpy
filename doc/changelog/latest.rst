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
- Simulation parameters :cfg:option:`Simulation.measure_at_algorithm_checkpoints` and
  :cfg:option:`Simulation.canonicalize_before_measurement'

Changed
^^^^^^^
- Change the build system and metadata declaration to ``pyproject.toml`` format.
  This makes installation more future-proof and stable, but should not affect how tenpy is used,
  once installed.
- Allow `couplings` parameters in the :class:`~tenpy.models.mixed_xk.MixedXKModel` methods 
  :meth:`~tenpy.models.mixed_xk.MixedXKModel.add_inter_ring_hopping`,
  :meth:`~tenpy.models.mixed_xk.MixedXKModel.add_intra_ring_hopping`,
  :meth:`~tenpy.models.mixed_xk.MixedXKModel.add_inter_ring_interaction`, and
  :meth:`~tenpy.models.mixed_xk.MixedXKModel.add_intre_ring_interaction` to vary with `x`.

Fixed
^^^^^
- Potentially serious bug :issue:`260` that the `sorted` flag of :class:`~tenpy.linalg.charges.LegCharge` was not set
  correctly in :func:`~tenpy.linalg.np_conserved.qr`.
- :meth:`~tenpy.networks.purification_mps.PurificationMPS.from_infiniteT_canonical` should now work with arbitrary
  charges of the original model, and has the option to double the number of charges to separately conserve the charges
  on each the physical and ancialla legs.
- Fix a wrong total charge in :meth:`~tenpy.networks.mpo.MPO.apply_zipup`.
- Fix :issue:`218` that :meth:`~tenpy.models.model.CouplingModel.add_multi_coupling_term` with `plus_hc=True` didn't
  correctly add the hermitian conjugate.
- Fix :issue:`210` that :meth:`~tenpy.models.aklt.AKLTChain.psi_AKLT` had typos and wrong canonical form for finite systems.
- Fix that the MPS :meth:`~tenpy.networks.mps.MPS.apply_local_op` with local multi-site operators didn't correctly track the
  norm with `renormalize=False`.
- We now update the norm of the MPS in :meth:`~tenpy.algorithms.tebd.TEBDEngine.update_bond`.
  If the parameter ``preserve_norm`` is ``True`` (which is the default for real time evolution)
  this has no effect when using :meth:`~tenpy.algorithms.tebd.TEBDEngine.run` or similat, 
  since the MPS norm is reset after the timestep anyway.
  It does, however, change the behavior if ``preserve_norm`` is ``False``.
