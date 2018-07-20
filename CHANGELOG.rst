CHANGELOG
=========

All notable changes to the project will be documented in this file.
The project adheres `semantic versioning <http://semver.org/spec/v2.0.0.html>`_

[Unreleased]
------------

Added
^^^^^
- :meth:`tenpy.networks.mps.MPS.canonical_form_infinite`.
- :meth:`tenpy.linalg.np_conserved.Array.extend` and :meth:`tenpy.linalg.charges.LegCharge.extend`,
  allowing to extend an Array with zeros.
- DMRG parameter ``'orthogonal_to'`` allows to calculate excited states for finite systems.
- possibility to change the number of charges after creating LegCharges/Arrays
- more general way to specify the order of sites in a :class:`tenpy.models.lattice.Lattice`.
- new :class:`tenpy.models.lattice.Honeycomb`

Changed
^^^^^^^
- moved toycodes from the folder ``examples/`` to a new folder ``toycodes/`` to separate them clearly.
- Restructured lanczos into a class, added time evolution calculating exp(A*dt)|psi0>
- Warning for poorly conditioned Lanczos; to overcome this enable the new parameter "reortho"
- By default, make deep copies of npc Arrays.
- Restructured :mod:`tenpy.algorithms.dmrg`:

  - :func:`~tenpy.algorithms.dmrg.run` is now just a wrapper around the new 
    :meth:`~tenpy.algorithms.dmrg.Engine.run`, ``run(psi, model, pars)`` is roughly equivalent to
    ``eng = EngineCombine(psi, model, pars); eng.run()``.
  - Added :meth:`~tenpy.algorithms.dmrg.Engine.init_env` and :meth:`~tenpy.algorithms.dmrg.Engine.reset_stats`
    to allow a simple restart of DMRG with slightly different parameters.
  - call ``MPS.canonical_form()`` for infinite systems if the final state is not in canonical form.

- Changed **default values** for some parameters:

  - increase ``Lanczos_params['N_cache'] = N_max`` (i.e. keep all states)
  - set ``DMRG_params['P_tol_to_trunc'] = 0.05`` and provide reasonable ..._min and ..._max values.
  - increased (default) DMRG accuracy by setting
    ``DMRG_params['max_E_err'] = 1.e-5`` and ``DMRG_params['max_S_err'] = 1.e-3``.

- don't print the energy during real-time TEBD evolution - it's preserved up to truncation errors.
- Renamed the `SquareLattice` class to :class:`tenpy.models.lattice.Square` for better consistency.

Fixed
^^^^^
- avoid error in MPS.apply_local_op()
- Don't carry around total charge when using DMRG with a mixer
- Corrected couplings of the FermionicHubbardChain
- issue #2: memory leak in cython parts when using intelpython/anaconda
- issue #4: incompatible data types.
- issue #6: the CouplingModel generated wrong Couplings in some cases
- more reasonable traceback in case of wrong labels


[0.3.0] - 2018-02-19
--------------------
This is the first version published on github.

Added
^^^^^
- Cython modules for np_conserved and charges, which can optionally compiled for speed-ups
- tools.optimization for dynamical optimization
- Various models.
- More predefined lattice sites.
- Example toy-codes.
- Network contractor for general networks

Changed
^^^^^^^
- Switch to python3

Removed
^^^^^^^
- Python 2 support.


[0.2.0] - 2017-02-24
--------------------
- Compatible with python2 and python3 (using the 2to3 tool).
- Development version.
- Includes TEBD and DMRG.


Changes compared to previous TeNPy
----------------------------------
This library is based on a previous (closed source) version developed mainly by
Frank Pollmann, Michael P. Zaletel and Roger S. K. Mong.
While allmost all files are completely rewritten and not backwards compatible, the overall structure is similar.
In the following, we list only the most important changes.

Global Changes
^^^^^^^^^^^^^^
- syntax style based on PEP8. Use ``$>yapf -r -i ./`` to ensure consitent formatting over the whole project.
  Special comments ``# yapf: disable`` and ``# yapf: enable`` can be used for manual formatting of some regions in code.
- Following PEP8, we distinguish between 'private' functions, 
  indicated by names starting with an underscore and to be used only within the library, and the public API. 
  The puplic API should be backwards-compatible with different releases, while private functions might change at any time.
- all modules are in the folder ``tenpy`` to avoid name conflicts with other libraries.
- withing the library, relative imports are used, e.g., ``from ..tools.math import (toiterable, tonparray)``
  Exception: the files in `tests/` and `examples/` run as ``__main__`` and can't use relative imports

  Files outside of the library (and in `tests/`, `examples/`) should use
  absolute imports, e.g. ``import tenpy.algorithms.tebd``
- renamed `tenpy/mps/` to `tenpy/networks`, since it containes various tensor networks.
- added :class:`~tenpy.networks.site.Site` describing the local physical sites by providing the physical LegCharge and
  onsite operators.

np_conserved
^^^^^^^^^^^^
- pure python, no need to compile!
- in module :mod:`tenpy.linalg` instead of ``algorithms/linalg``.
- moved functionality for charges to :mod:`~tenpy.linalg.charges`
- Introduced the classes :class:`~tenpy.linalg.charges.ChargeInfo` (basically the old ``q_number``, and ``mod_q``)
  and :class:`~tenpy.linalg.charges.LegCharge` (the old ``qind, qconj``).
- Introduced the class :class:`~tenpy.linalg.charges.LegPipe` to replace the old ``leg_pipe``.
  It is derived from ``LegCharge`` and used as a leg in the `array` class. Thus any inherited array (after
  ``tensordot`` etc still has all the necessary information to split the legs.
  (The legs are shared between different arrays, so it's saved only once in memory)
- Enhanced indexing of the array class to support slices and 1D index arrays along certain axes
- more functions, e.g. :func:`~tenpy.linalg.np_conserved.grid_outer`

TEBD
^^^^
- Introduced TruncationError for easy handling of total truncation error.
- some truncation parameters are renamed and may have a different meaning, e.g. `svd_max` -> `svd_min` 
  has no 'log' in the definition.

DMRG
^^^^
- separate Lanczos module in `tenpy/linalg/`. Strangely, the old version orthoganalized
  against the complex conjugates of `orthogonal_to` (contrary to it's doc string!)
  (and thus calculated 'theta_o' as bra, not ket).
- cleaned up, provide prototypes for DMRG engine and mixer.

Tools
^^^^^
- added :mod:`tenpy.tools.misc`, which contains 'random stuff' from old ``tools.math``
  like ``to_iterable`` and ``to_array`` (renamed to follow PEP8, documented)
- moved stuff for fitting to :mod:`tenpy.tools.fit`
- enhanced :func:`tenpy.tools.string.vert_join` for nice formatting
- moved (parts of) old `cluster/omp.py` to :mod:`tenpy.tools.process`
- added :mod:`tenpy.tools.params` for a simplified handling of parameter/arguments for models and/or algorithms.
  Similar as the old `models.model.set_var`, but use it also for algorithms. Also, it may modify the given dictionary.
