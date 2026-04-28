# Cyten - a combination of Cytnx and TeNPy

The name Cyten is pronounced like sci-ten, and refers to scientific tensor networks implmented in C++ with a focus on the provided Python bindings, and the python-style library in modern C++.

## About: combining forces of the TeNPy and Cytnx team
This repo originates from a collaboration between the [Cytnx](https://github.com/cytnx-dev/cytnx) and [TeNPy](https://github.com/tenpy/tenpy) developers.
Cytnx is a C++ library for Tensors with Abelian symmetries with pybind11 bindings to Python.
The goal is to use that as a basis to translate the refactoring and implementation of non-Abelian symmetries from the `v2_alpha` branch in the tenpy repository (currenlty in pure Python) into C++, thereby extending the capabilities of the cytnx library, and providing a backend for TeNpy which will then focus on the higher-levels (defining MPS and algorithms like DMRG  etc).
At the same time, the code from cytnx to be included will be refactored and cleaned a bit.

## Setup
Once released, we will provide pre-compiled packages on conda/pip.
Until then, you need to build the package yourself on your local machine, as detailed in `docs/INSTALL.rst`.

## Testing
The python interface is tested with `pytest` run from the `tests/` folder in the repo.
To test the C++ code, run `gtest` (from GoogleTest).

## Documentation
Will eventually be online, but so far you also need to build it locally.
See `README.md` in the `docs/` folder on how - essentially just install `cyten`, run `doxygen` and then `make html` in the `docs/` folder.

## Code style and linting
Please follow these guidelines when contributing code

- Use a code style based on :pep:`8`.
  The git repo includes a config file ``.flake8`` for the python package `flake8 <https://flake8.pycqa.org/en/latest/>`_.
  `flake8` is a tool that lints python code, i.e. it checks a number of rules for code formatting and
  reports any violations. Install it using either `pip` or `conda`.
  Run e.g. `flake8 .` in the repository root.
  Feel free to use the git hooks, see `docs/contributing/code_style.rst`.
  It is recommended to also install the `flake8-docstring <https://github.com/pycqa/flake8-docstrings>`
  extension.

- Every function/class/module should be documented by its doc-string, see :pep:`257`.
  Exception: If you override a method of a parent class, only add a docstring if it adds value.

  Additional documentation for the user guide is in the folder ``doc/``.

  The documentation uses `reStructuredText`. If you are new to `reStructuredText`, read this `introduction <http://www.sphinx-doc.org/en/stable/rest.html>`_.
  We use the `numpy` style for doc-strings (with the `napoleon <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_ extension to sphinx).
  You can read about them in these `Instructions for the doc strings <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
  In addition, you can take a look at the following `example file <https://github.com/numpy/numpydoc/blob/main/doc/example.py>`_.
  Helpful hints on top of that::

        r"""<- this r makes me a raw string, thus '\' has no special meaning.
        Otherwise you would need to escape backslashes, e.g. in math formulas.

        You can include cross references to classes, methods, functions, modules like
        :class:`~tenpy.linalg.np_conserved.Array`, :meth:`~tenpy.linalg.np_conserved.Array.to_ndarray`,
        :func:`tenpy.tools.math.toiterable`, :mod:`tenpy.linalg.np_conserved`.
        The ~ in the beginning makes only the last part of the name appear in the generated documentation.
        Documents of the userguide can be referenced with :doc:`/intro_npc` even from inside the doc-strings.
        You can also cross-link to other documentations, e.g. :class:`numpy.ndarray`, :func`scipy.linalg.svd` and :mod: will work.

        Moreover, you can link to github issues, arXiv papers, dois, and topics in the community forum with
        e.g. :issue:`5`, :arxiv:`1805.00055`, :doi:`10.1000/1` and :forum:`3`.

        Citations from the literature list can be cited as :cite:`white1992` using the bibtex key.

        Write inline formulas as :math:`H |\Psi\rangle = E |\Psi\rangle` or displayed equations as
        .. math ::

           e^{i\pi} + 1 = 0

        In doc-strings, math can only be used in the Notes section.
        To refer to variables within math, use `\mathtt{varname}`.

        .. todo ::

           This block can describe things which need to be done and is automatically included in a section of :doc:`todo`.
        """

- Use relative imports within the package. Example::

      from .symmetries import no_symmetry

- If you write new functions, please also include suitable tests!

- During development, you might introduce ``# TODO`` comments.
  Please use exactly this format, to make searching for them easier, and include your initials or name.
  Try to remove/resolve them as soon as possible.
  If you're not 100% sure that you will remove it soon, please add a doc-string with a
  ``.. todo ::`` block, such that we can keep track of it.

  Unfinished functions should ``raise NotImplementedError()``.

- We may start keeping a changelog in the future, so far we do not.
