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
- Added consistency checks, see :func:`~tenpy.tools.misc.consistency_check`, as well as
  * :cfg:option:`Algorithm.max_cylinder_width`
  * :cfg:option:`TimeEvolutionAlgorithm.max_trunc_err`
  * :cfg:option:`IterativeSweeps.max_trunc_err`
  * :cfg:option:`ExpMPOEvolution.max_dt`
  * :cfg:option:`TEBDEngine.max_delta_t`

Changed
^^^^^^^
- nothing yet

Fixed
^^^^^
- MPO methods :meth:`~tenpy.networks.mpo.MPO.dagger`, :meth:`~tenpy.networks.mpo.MPO.is_hermitian`,
  and :meth:`~tenpy.networks.mpo.MPO.__add__` now respect
  the :attr:`:~tenpy.networks.mpo.MPO.explicit_plus_hc` flag.
