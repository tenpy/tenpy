README
======

Introduction
------------
This is a Python Library for simulation of strongly correlated quantum systems with tensor networks.
It's based on TenPy, but restricts to simple code.

A detailed list of changes compared to the previous TenPy can be found in docs/changes_TenPy.rst

How do I get set up?
--------------------
- Requirements:
  - python 2.7 with fairly recent versions of:
    - numpy
    - scipy 
    - Cython
    - and optionally:
      - pylab might be useful for plotting and debugging, but is not required.
    - for contribution, make sure you have:
      - `nose <https://nose.readthedocs.io/en/latest/>`_ for testing
      - `yapf <https://github.com/google/yapf>`_ to ensure a consistent style
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
- Style guide for code: based on `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_.
  Details in the file `.style.yapf`. Just run `$>yapf -i FILE` for formatting.
  We use yapf_ for code-formatting, detailed style config is in .style.yapf
  `$> yapf -r -i ./` auto-formats all files in the project.
  Special comments `# yapf: disable` and `# yapf: enable` can be used for manual formatting of some regions in code.
- *Every* function/class/module should be documented by its doc-string, following PEP257_:
  - for small functions a one-liner, else a one-liner summary followed by a blank line, followed by the description.
  - document at most one argument per line. If appropriate, give expected type and default values.
  - Use reStructuredText_ for long doc strings and extra documentation files.
  - Formulas in the documentation are enclosed in $$ and written as in LaTeX. 
    However, we may also use as combination of Python and Latex, e.g. '$ EE = \sum_i `sing_val[i]**2` $'
- A To-Do list can be found in docs/todo.rst. It also contains a bug list.
- During development, you can introduce `# TODO` comments. But also try to remove them again later!
  Unfinished functions should raise a `NotImplementedError`.
  Bugs may be marked with `# BUG`. Also add them to the bug list in docs/todo.rst
- Use relative imports within TenPy, e.g. `from ..algorithms.linalg import np_conserved as npc`
- We use nose_ for testing. Simply install nose and run the command `nosetests` in tests.
  This should work for every stable version of the library.


I found a bug
-------------
- Fix it, if that's possible. git commit with a message starting as 'BUG: descrition'
- Alternatively, report it in the "BUGS" section of docs/todo.rst
