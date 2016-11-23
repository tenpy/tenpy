Changes compared to previous TenPy
==================================

Global Changes
--------------
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


np_conserved
------------
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


tools
-----
- added :mod:`tenpy.tools.misc`, which contains 'random stuff' from old ``tools.math``
  like ``to_iterable`` and ``to_array`` (renamed to follow PEP8, and documented)
- moved stuff for fitting to :mod:`tenpy.tools.fit`
- enhanced :func:`tenpy.tools.string.vert_join` for nice formatting
- moved (parts of) old `cluster/omp.py` to :mod:`tenpy.tools.process`
