Welcome to TenPyLight's documentation!
======================================

Introduction
------------
This is a Python Library for simulation of strongly correlated quantum systems with tensor networks.
It's based on TenPy2, but tries to be readable and easy to understand for new peoples.
A detailed list of changes compared to the previous TenPy can be found in docs/changes_TenPy.rst

How do I get set up?
--------------------
Right now, this documentetion is only distributed with the code, so I guess you already have the code somehow.

Minimal Requirements:
Python 2.7 with fairly recent versions of `NumPy <http://www.numpy.org>`_ and `SciPy <http://www.scipy.org>`_.
Further `Cython <http://www.cython.org>`_ and some kind of LaPack library (including the headers).

In addition, some code uses `MatPlotLib <http://www.matplotlib.org>`_ for plotting and visualization.
However, this is not necessary for running simulations.

If you want to run larger simulations, we recommend the use of Intels MKL and icc.
It ships with a Lapack library, and uses optimization for Intel CPUs.
Moreover, it uses parallelization of the Lapack routines, which makes execution much faster...
However, that requires compiling python with the same icc, which is a bit complicated. 
Details can be found in :doc:`docs/install.rst <install>`

Place this library as whole in a folder included in your `$PYTHONPATH`.
For example, save it as `$HOME/PyLibs/TenPyLight`, and add the following line in your `~/.bashrc`::

    export PYTHONPATH=$HOME/PyLibs/

You can then include it from python with::

    import TenPyLight as tp

How to read the documentation
-----------------------------
The documentation is based on Python's docstrings, and some additional .rst files are located in docs/.
It should be formated as `reStructuredText <http://www.sphinx-doc.org/en/stable/rest.html>`_.
Thus, you can  for example intective python help(), Python IDEs, or just read the source to get some documentation.
However, you can also use `Sphinx <http://www.sphinx-doc.org>`_ to generate the full API in various formats including HTML.

You can install `sphinx`_ with::

    pip install sphinx

Afterwards, go to `docs` and run the following command::

    sphinx-build -b html -d sphinx_build/doctrees . sphinx_build/html
    # alternatively, simply call `make html`

This should generate the html documentation in `docs/sphinx_build/html`.
Simply open this folder (to be precise: it's `index.html`) in your webbroser and enjoy this and other documentation 
beatifully rendered and with cross links :-).


Further documentation
=====================
.. toctree::
   :maxdepth: 1

   install
   changes_TenPy
   todo

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Contribution
============
The code is maintained in a git repository. You're welcome to contribute and push changes back.
However, to keep consistency, we ask you to comply with the following guidelines:

- Use code style based on `PEP8 <http://www.python.org/dev/peps/pep-0008/>`_.
  The git repo includes a config file `.style.yapf` for `yapf <http://github.com/google/yapf>`_.
  `yapf` is a tool to auto-format code, e.g., by the command `yapf -i some/file`.
  Run the following command to ensure consitency over the whole project::

      yapf -r -i ./

  Since no tool is perfect, you can format some regions of code manually and enclose them 
  with the special comments `# yapf: disable` and `# yapf: enable`.
- Every function/class/module should be documented by its doc-string, following `PEP257 <http://www.python.org/dev/peps/pep-0257/>`_:
  - for small functions a one-liner, else a one-liner summary followed by a blank line, followed by the description.
  - document at most one argument per line. If appropriate, give expected type and default values.
  - Use `reStructuredText`_ for the doc strings and extra documentation files.
  - Math Formulas in the documentation can be written in Latex style ::

        Inline formulas as :math:` H |\Psi\rangle = E |\Psi\rangle` or displayed as 
        .. math :
            e^{i\pi} + 1 = 0

- Use the python package `nose <http://nose.readthedocs.io/en/latest/>`_ for testing.
  Run it simply with `nosetest` in `tests`.
- A To-Do list can be found in :doc:`docs/todo.rst <todo>`. It also contains a bug list.
- During development, you can introduce `# TODO` comments. But also try to remove them again later!
  During development, unfinished functions should raise a `NotImplementedError`.
  Bugs may be marked with `# BUG`. Also add them to the bug list in docs/todo.rst
- Use relative imports within TenPy, e.g. ::

      from ..algorithms.linalg import np_conserved as npc

Thank You for helping with the development!


I found a bug
-------------
Fix it, if that's possible. git commit with a message starting as 'BUG: descrition'

Alternatively, report it in the "BUGS" section of :doc:`docs/todo.rst <todo>`.
