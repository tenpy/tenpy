Contributing
============

The code is maintained in a git repository, the official repository is on `github <https://github.com/tenpy/tenpy>`.
You're welcome to contribute and submit pull requests on github. You are also welcome to become a member of the developer team, just ask nicely :)
To keep consistency, we ask you to comply with the following guidelines for contributions:

- Use a code style based on :pep:`8`.
  The git repo includes a config file `.style.yapf` for the python package `yapf <http://github.com/google/yapf>`_.
  `yapf` is a tool to auto-format code, e.g., by the command ``yapf -i some/file``.
  We run the following command from time to time to ensure consitency over the whole project::

      yapf -r -i ./

.. note ::

  Since no tool is perfect, you can format some regions of code manually and enclose them 
  with the special comments ``# yapf: disable`` and ``# yapf: enable``.

- Every function/class/module should be documented by its doc-string (c.f. :pep:`257`),
  additional documentation is in ``doc/``.
  The documentation uses `reStructuredText`. If you're new to `reStructuredText`, consider reading this
  `introduction <http://www.sphinx-doc.org/en/stable/rest.html>`_.
  We use the `numpydoc` extension to sphinx, so please read and follow these
  `Instructions for the doc strings <http://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.
  In addition, you can take a look at the following
  `example file <http://github.com/numpy/numpy/blob/master/doc/example.py>`_.
  Helpful hints on top of that::

        r"""<- this r makes me a raw string, thus '\' has no special meaning.
        Otherwise you would need to escape backslashes, e.g. in math formulas.

        You can include cross references to classes, methods, functions etc, like
        :class:`~tenpy.linalg.np_conserved.Array`, :meth:`~tenpy.linalg.np_conserved.Array.to_ndarray` or :func:`tenpy.tools.math.toiterable`.
        Within the python docstrings, documents of the userguides can be references like :doc:`../IntroNpc`.

        Write inline formulas as :math:`H |\Psi\rangle = E |\Psi\rangle` or displayed equations as 
        .. math ::

           e^{i\pi} + 1 = 0

        In doc-strings, math can only be used in the Notes section.
        To refer to variables within math, use `\mathtt{varname}`.

        .. todo ::

           This block can describe things which need to be done and is automatically included in a section of :doc:`todo`.
        """

  Instructions for building the documentation are in :doc:`INSTALL`.

- Use relative imports within TeNPy. Example::

      from ..linalg import np_conserved as npc

- If you changed the file strucuture of tenpy, run ``make src2rst`` in the ``doc/`` folder
  to update the documentation index for the reference.
- Use the python package `nose <http://nose.readthedocs.io/en/latest/>`_ for testing.
  Run it simply with ``nosetest`` in `tests/`.
  You should make sure that all tests run through, before you ``git push`` back into the public repo.
  Long-running tests are marked with the attribute `slow`; for a quick check you can also run
  ``nosetest -a '!slow'``.
- Reversely, if you write new functions, please also include suitable tests!
- During development, you might introduce ``# TODO`` comments.  But also try to remove them again later!
  If you're not 100% sure that you will remove it soon, please add a doc-string with a 
  ``.. todo ::`` block, such that we can keep track of it as explained in the previous point.

  Unfinished functions should ``raise NotImplementedError()``.
- if you want to try out new things in temporary files: any folder named ``playground`` is ignored by `git`.

**Thank You** for helping with the development!


Bulding the documentation
-------------------------
You can use `Sphinx <http://www.sphinx-doc.org>`_ to generate the full documentation 
in various formats (including HTML or PDF) yourself, as described in the following.
First, install `Sphinx`_ and the extension `numpydoc <http://pypi.python.org/pypi/numpydoc>`_ with::

    sudo pip install --upgrade sphinx numpydoc

Afterwards, simply go to the folder `doc/` and run the following command::

    make html

This should generate the html documentation in the folder `doc/sphinx_build/html`.
Open this folder (or to be precise: the file `index.html` in it) in your webbroser
and enjoy this and other documentation beautifully rendered, with cross links, math formulas
and even a search function.
Other output formats are available as other make targets, e.g., ``make latexpdf``.

.. note ::

   Building the (html) documentation requires loading the modules.
   Thus make sure that the folder `tenpy` is included in your ``$PYTHONPATH``,
   as described in :doc:`doc/INSTALL.rst <INSTALL>`.
