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
  :cfg:option:`Simulation.canonicalize_before_measurement`
- :class:`~tenpy.networks.mps.BaseMPSExpectationValues` parent class to unify the framework of computing expectation values and
  correlation functions in :class:`~tenpy.networks.mps.MPS` and :class:`~tenpy.networks.mps.MPSEnvironment`
- Abstract :class:`~tenpy.networks.mps.BaseEnvironment` parent class for :class:`~tenpy.networks.mps.MPSEnvironment`
  and :class:`~tenpy.networks.mpo.MPOEnvironment`
- Add `phi_ext` parameter to :class:`~tenpy.models.fermions_spinless.FermionModel`,
  :class:`~tenpy.models.hubbard.BoseHubbardModel` and :class:`~tenpy.models.hubbard.FermiHubbardModel`.
- Option `allow_incommensurate` for :meth:`~tenpy.networks.mps.MPS.from_lat_product_state`.
- Most "important" objects (this obvious involves some judgement calls, regarding what counts as important)
  are now directly exposed in the top-level namespace of the tenpy package, i.e. you can now
  ``from tenpy import MPS, tensordot, TwoSiteDMRGEngine`` or ``import tenpy as tp`` and then use
  ``tp.tensordot`` etc. All objects which are not "private" and/or "implementation details" are
  exposed in the subpackage namespace, e.g. you can ``from tenpy.networks import MPOGraph``.

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
- Renamed the module :mod:`~tenpy.linalg.lanczos` to :mod:`tenpy.linalg.krylov_based`.
- The :attr:`~tenpy.algorithms.mps_common.Sweep.move_right` attribute of
  :class:`~tenpy.algorithms.mps_common.Sweep` now supports a third value, ``None``, in addition
  to ``True, False``. ``None`` means that the sweep will not move at all, i.e. the next update
  will be at the same position than the current one. This happens e.g. in TDVP.
- Mixers have been generalized and are no longer specialized for use in DMRG.
  Method names and signatures have been changed.
  The mixer classes are now implemented in :mod:`tenpy.linalg.algorithms.mps_common`.
  Backwads-compatible wrappers with the old method names and signatures will be kept in
  :mod:`tenpy.linalg.algorithms.dmrg` until v1.0.

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
  this has no effect when using :meth:`~tenpy.algorithms.tebd.TEBDEngine.run` or similar,
  since the MPS norm is reset after the timestep anyway.
  It does, however, change the behavior if ``preserve_norm`` is ``False``.
- :issue:`265` that MPO methods :meth:`~tenpy.networks.mpo.MPO.make_U_I`, `make_U_II`, `apply_naively` and `apply_zipup` 
  just ignored the `explicit_plus_hc` flag of the MPO, possibly giving completely wrong results without raising errors.
- Make sure that :func:`~tenpy.linalg.np_conserved.eigh` doesn't have a :class:`~tenpy.linalg.charges.LegPipe` on the second (=new) leg.
- :issue:`289` that :meth:`~tenpy.networks.mps.MPS.correlation_length` raised errors for `charge_sector` np ndarrays.
  Further, allow to pass multiplie charge sectors to loop over at once, add argument `return_charges`.
  Also, provide a :meth:`~tenpy.networks.mps.MPS.correlation_length_charge_sectors` convenience function to return valid charge sectors.
