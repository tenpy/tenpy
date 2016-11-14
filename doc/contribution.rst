Contributing
============

The code is maintained in a git repository. You're welcome to contribute and push changes back.
However, to keep consistency, we ask you to comply with the following guidelines:

- Use code style based on :pep:`8`.
  The git repo includes a config file `.style.yapf` for the python package `yapf <http://github.com/google/yapf>`_.
  `yapf` is a tool to auto-format code, e.g., by the command ``yapf -i some/file``.
  Run the following command to ensure consitency over the whole project::

      yapf -r -i ./

.. note ::

  Since no tool is perfect, you can format some regions of code manually and enclose them 
  with the special comments ``# yapf: disable`` and ``# yapf: enable``.

- Every function/class/module should be documented by its doc-string (c.f. :pep:`257`),
  additional documentation is in ``doc/``
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

- Every file should (after the doc-string) include::

      from __future__ import division

  This reduces the ambiguity of the ``/`` operator to give floating points even for integer division.
  If you want true integer division, make it explicit with the new ``//`` operator.
  See :pep:`238` for details. Example::

      >>> print 5 / 4    # without __future__ division, this would be 1
      1.25
      >>> print 5 // 4
      1
- Use relative imports within TenPy. Example::

      from ..linalg import np_conserved as npc

- If you changed the file strucuture of tenpy, run ``make src2rst`` in the ``doc/`` folder
  to update the documentation index for the reference.
- Use the python package `nose <http://nose.readthedocs.io/en/latest/>`_ for testing.
  Run it simply with ``nosetest`` in `tests/`.
  You should make sure that all tests run through, before you ``git push`` back into the public repo.
- Reversely, if you write new functions, please also include suitable tests!
- A To-Do list can be found in :doc:`doc/todo.rst <todo>`. It also contains a bug list.
  Sphinx also extract blocks following ``.. todo ::`` from doc-strings, 
  collecting them in a list in the html documentation.
- During development, you might introduce ``# TODO`` comments.  But also try to remove them again later!
  If you're not 100% sure that you will remove it soon, please add a doc-string with a 
  ``.. todo ::`` block, such that we can keep track of it as explained in the previous point.

  Unfinished functions should ``raise NotImplementedError()``.
  Locations of bugs may be marked with `# BUG`. But also add them to the bug list in :doc:`doc/todo.rst <todo>`

- if you want to try out new things in temporary files: any folder named ``playground`` is ignored by `git`.

**Thank You** for helping with the development!


Additional information
----------------------

.. toctree::
   :maxdepth: 2

   todo

