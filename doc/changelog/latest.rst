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
- Added class :class:`~tenpy.models.lattice.SimpleBZ` for the Brillouin zone of a Lattice and convenience functions for
  plotting it.
- Add simulation classes 
  :class:`~tenpy.simulations.time_evolution.SpectralSimulation`,
  :class:`~tenpy.simulations.time_evolution.TimeDependentCorrelation`,
  :class:`~tenpy.simulations.time_evolution.TimeDependentCorrelationEvolveBraKet`,
  :class:`~tenpy.simulations.time_evolution.SpectralSimulationEvolveBraKet`.
- Add module :mod:`tenpy.simulations.post_processing` with a :class:`~tenpy.simulations.post_processing.DataLoader` class,
  which can be used at the end of a simulation (or in a separate script) to calculate additional results from measurment data or plot something.
- Functions to perform a linear prediction in :mod:`tenpy.tools.prediction`.
- Functions for fourier transform and plotting of spectral functions in :mod:`tenpy.tools.spectral_function_tools`.


Changed
^^^^^^^
- safeguard measurments with try-except.

Fixed
^^^^^
- MPO methods :meth:`~tenpy.networks.mpo.MPO.dagger`, :meth:`~tenpy.networks.mpo.MPO.is_hermitian`,
  and :meth:`~tenpy.networks.mpo.MPO.__add__` now respect
  the :attr:`:~tenpy.networks.mpo.MPO.explicit_plus_hc` flag.
- Handle Jordan wigner strings better, see :issue:`355`.

