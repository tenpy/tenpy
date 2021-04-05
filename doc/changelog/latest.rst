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
^^^^^^
- :func:`tenpy.tools.misc.find_subclass` now directly raises an error if no subclass with that name is found.
- Renamed the `logging_params` to `log_params`.

Fixed
^^^^^
- Use logging in simulation only after calling :func:`~tenpy.tools.misc.setup_logging`.
