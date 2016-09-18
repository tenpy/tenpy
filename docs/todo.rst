To-Do list
===============

Primary goals for the coming release
----------------------------------
- convert TenPy2 code with autopep8, restrict to necessary parts.
  Method:
    - add $FILE from old git repo without changes, git commit
    - autopep8 $FILE
    - Use the editor of your choice to expand tabs, remove trailing spaces
    - relative imports. In tests/, use absolute imports with
    - git commit
- run tests, extend tests
- simplify: no translate_Q in np_conserved; what's obselete?


Concrete things to be fixed in different files
----------------------------------------------
- docs/setup.tex: translate to reStructuredText
- algorithms/linalg/npc_setup: document possible variables/setups in docs/setup.tex
- which of algorithms/linalg/svd_* are necessary? np_conserved use svd_{d,z}gesvd.
  I also have a svd_robust.py in my TenPy; is that used?
- add doc strings in __init__.py for different folders



To be done at some point for the next releases
----------------------------------------------
- update setup.tex
- setup of the library
- documentation on the idea of algorithms, references in doc strings.
  - np_conserved needs an introduction for newbies


Other
-----
- Rename TenPyLight: too long? 


Wish-list
---------
- ED code using the symmetries, including example/test
- HTML documentation with sphinx
- Open Source on GitHub? -> Licence?
- possible to convert to python3 ? 
