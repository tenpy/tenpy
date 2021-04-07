[latest]
========

Release Notes
-------------
TODO: Summarize the most important changes

Changelog
---------

Backwards incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- change default separator for :func:`tenpy.tools.misc.get_recursive`, :func:`~tenpy.tools.misc.set_recursive`, :func:`~tenpy.tools.misc.update_recursive`, and
  :func:`~tenpy.tools.misc.flatten` to ``'.'`` instead of ``'/'``.

Added
^^^^^
- :func:`tenpy.simulations.simulation.run_sequential_simulations`

Changed
^^^^^^^
- :func:`tenpy.tools.misc.find_subclass` now directly raises an error if no subclass with that name is found.
- Renamed the `logging_params` to `log_params`.
- :func:`tenpy.simulations.measurement.correlation_length` now supports a `unit` keyword.
  If it is not given explicitly, a warning is raised.

Fixed
^^^^^
- Use logging in simulation only after calling :func:`~tenpy.tools.misc.setup_logging`.
- Missing ``+ h.c.`` in :meth:`tenpy.networks.mpo.MPOEnvironment.full_contraction` when `H.explicit_plus_hc` was True.
  This caused wrong energies being reported during DMRG when `explicit_plus_hc` was used.
