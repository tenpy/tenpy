To-Do list
==========
Update docs/changes_TenPy.rst if you've done something.

Primary goals for the coming release
------------------------------------
- convert TenPy2 code with autopep8, restrict to necessary parts. 

    - add files from old git repo without changes, git commit
    - run `$> yapf -r -i`. See docs/changes_TenPy.rst for detailed command
    - relative imports (except in tests)
    - git commit

- run tests, extend tests
- include minimal library with mps, mpo, model, 




Concrete things to be fixed in different files
----------------------------------------------
- np_conserved:

  - documentation of general idea
  - simplify: remove translate_Q (?)
  - introduce new class leg_charge with charge data for a single leg

- model:

  - separate class for nearest neighbour models.
  - every model should define Hmpo
  - generalize to non-uniform [d]
  - introduce basic lattice class;
    derive MPS-chain fixing index notation for accessing different sites
    How to handle different mappings lattice->chain?

- docs/setup.tex: translate to reStructuredText
- algorithms/linalg/npc_setup: document possible variables/setups in docs/setup.tex
- which of algorithms/linalg/svd_* are necessary? np_conserved use svd_{d,z}gesvd.
  I also have a svd_robust.py in my TenPy; is that used anywhere?
- add doc strings in __init__.py for different folders, explaining the most important parts of the modules


To be done at some point for the next releases
----------------------------------------------
- update setup.tex
- setup of the library
- documentation on the idea of algorithms, references in doc strings.

  - np_conserved needs an introduction for newbies
  - usage introduction with very simple (few-line) examples for newbies.


Other
-----
- Rename TenPyLight: too long?  TeNetLight?
- how much speedup does npc_helper give? 
  Maybe, a portable (python-only) np_conserved without npc_helper would be possible?

Wish-list
---------
- ED code using the symmetries, including example/test
- HTML documentation with sphinx
- Open Source on GitHub? -> Licence?
- possible to convert to python3 ? 

.. _buglist:
BUGS
----
Here, you can report Bugs that need to be fixed.


Known limitations
-----------------
TenPyLight is meant to be a simple and readably library. Therefore, it may not handle every special case correctly.
Here, we list some known 'bugs' that won't be fixed (at least not in the near future).

