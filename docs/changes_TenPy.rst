Changes compared to previous TenPy2
===================================


Global changes
--------------
- syntax style based on PEP8. Use ``$>yapf -r -i ./`` to ensure consitent formatting over the whole project.
  Special comments ``# yapf: disable`` and ``# yapf: enable`` can be used for manual formatting of some regions in code.
- relative imports, e.g., ``from ..tools.math import (toiterable, tonparray)``
  Exception: the files in `tests/` and `examples/` run as ``__main__`` and can't use relative imports

  Files outside of the library (and in `tests/`, `examples/`) should use
  absolute imports , e.g., ``import TenPyLight.algorithms.TEBD``

- re-implemented large parts of `np_conserved`.
  - moved functionality for charges to `charges.py`, but import it in np_conserveded.
  - Introduced the classes :class:`~tenpy.linalg.charges.ChargeInfo` (old q_number, mod_q)
    and :class:`~tenpy.linalg.charges.LegCharge` (old qind, qconj).
  - Complete version in pure python, optimize only special functions.




