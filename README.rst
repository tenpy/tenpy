README
======

Introduction
------------
This is a Python Library for simulation of strongly correlated quantum systems with tensor networks.
It's based on TenPy, but restricts to simple code.

How do I get set up?
--------------------
- Requirements:
  - python 2.7 with fairly recent versions of:
    - numpy
    - scipy 
    - Cython
    - and optionally:
      - pylab might be useful for plotting and debugging, but is not required.
      - nose for testing
  - Lapack library (the -devel version including the headers. Included in MKL.)
  - MKL is recommended for optimization:
    MKL = Intel's math kernel library. Use it in combination with Intels icc compiler.
    TenPy works also without MKL. However, if MKL is set up correctly with icc,
    it uses multiple cores for `numpy.tensordot()` and similar routines.
    In addition, it can use processor specific optimizations of Intel CPUs.
- A detailed instruction for the setup with MKL can be found in docs/setup.tex
- Place this library as whole in a folder included in your `$PYTHON_PATH`,
  say ~/PyLibs/TenPyLight/, and add `export PYTHON_PATH=$HOME/PyLibs/` to your ~/.bashrc.
- additional documentation on various parts of the library is found in docs/


Contribution guidelines
-----------------------
- the code is maintained in a git repository.
- style guide for code: PEP8_ . Try to use tools like autopep8 before importing new code.
- use 4-space indents; break lines with more than 120 characters.
- *Every* function/class/module should be documented by its doc-string, following PEP 257:
  - for small functions a one-liner, else a one-liner summary followed by a blank line, followed by the description.
  - document at most one argument per line. If appropriate, give expected type and default values.
  - Use reStructuredText_ for long doc strings.
  - Formulas in the documentation are enclosed in $$ and written as in LaTeX. 
    However, we may also use as combination of Python and Latex, e.g. `$ EE = \sum_i sing_val[i]**2 $`
- During development, locations of work may be marked with a `# TODO` comment.
  Unfinished functions should raise a `NotImplementedError`.
  Bugs may be marked with `# BUG` 
- Use relative imports within TenPy, e.g. `from ..algorithms.linalg import np_conserved as npc`
- a To-Do list can be found in docs/.
- We use nose_ for testing. Simply install nose and run the command `nosetests` in tests.
  This should work for every stable version of the library.



References
----------
.. _PEP8: https://www.python.org/dev/peps/pep-0008/
.. _reStructuredText: https://en.wikipedia.org/wiki/ReStructuredText
.. _nose: https://nose.readthedocs.io/en/latest/
