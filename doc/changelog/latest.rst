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
- replace the `orthogonal_to` option of :class:`tenpy.algorithms.mps_common.Sweep` by an `orthogonal_to` keyword argument for the class and it's :meth:`~tenpy.algorithms.mps_common.Sweep.init_env`.

Added
^^^^^
- :func:`tenpy.simulations.simulation.run_sequential_simulations`
- :meth:`tenpy.networks.mps.MPSEnvironment.init_LP` and :meth:`~tenpy.networks.mps.MPSEnvironment.init_RP`, and
  :meth:`tenpy.networks.mpo.MPOEnvironment.init_LP` and :meth:`~tenpy.networks.mpo.MPOEnvironment.init_RP` additionally
  support the argument `start_env_sites`, which can now be part of the `init_env_data`.
  This allows to converge MPO environments from scratch, given only the MPO and MPS.

Changed
^^^^^^^
- :func:`tenpy.tools.misc.find_subclass` now directly raises an error if no subclass with that name is found.
- Renamed the `logging_params` to `log_params`.
- :func:`tenpy.simulations.measurement.correlation_length` now supports a `unit` keyword.
  If it is not given explicitly, a warning is raised.
- :func:`tenpy.networks.mps.MPS.canonical_form` now supports an argument `envs_to_update` to allow keeping
  MPS/MPOEnvironments consistent.
- keyword argument `sequential_simulations` for :meth:`tenpy.algorithms.algorithm.Algorithm.get_resume_data`.

Fixed
^^^^^
- Use logging in simulation only after calling :func:`~tenpy.tools.misc.setup_logging`.
- Missing ``+ h.c.`` in :meth:`tenpy.networks.mpo.MPOEnvironment.full_contraction` when `H.explicit_plus_hc` was True.
  This caused wrong energies being reported during DMRG when `explicit_plus_hc` was used.
- :issue:`99` and :issue:`113` by allowing to either reinitialize the environment from scratch, 
  and/or to updating the environments in psi.canonical_form().
