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
