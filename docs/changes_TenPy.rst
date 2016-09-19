Changes compared to previous TenPy2
===================================


Global changes
--------------
- syntax style based on PEP8. Use `$>yapf -r -i ./` to ensure consitent formatting over the whole project.
  Special comments `# yapf: disable` and `# yapf: enable` can be used for manual formatting of some regions in code.
- relative imports, e.g., `from ..tools.math import (toiterable, tonparray)`
  Exception: the files in tests/ and examples/ run as __main__ and can't use relative imports
  Files outside of the library (and in tests/, examples/) should use
  absolute imports as 



