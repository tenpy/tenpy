Contributing
============
The code is maintained in a git repository. You're welcome to contribute and push changes back.
However, to keep consistency, we ask you to comply with the following guidelines:

- Use code style based on :pep:`8`.
  The git repo includes a config file `.style.yapf` for `yapf <http://github.com/google/yapf>`_.
  `yapf` is a tool to auto-format code, e.g., by the command `yapf -i some/file`.
  Run the following command to ensure consitency over the whole project::

      yapf -r -i ./

  Since no tool is perfect, you can format some regions of code manually and enclose them 
  with the special comments `# yapf: disable` and `# yapf: enable`.
- Every function/class/module should be documented by its doc-string, following :pep:`257`:

  - for small functions a one-liner, else a one-liner summary followed by a blank line, followed by the description.
  - document at most one argument per line. If appropriate, give expected type and default values.
    I would suggest following numpys style, see this `example file <http://github.com/numpy/numpy/blob/master/doc/example.py>`_.
  - Use `reStructuredText` for the doc strings and extra documentation files.
    Its fairly intuitive, but if you don't know reStructeredText, read this `introduction <http://www.sphinx-doc.org/en/stable/rest.html>`_.
    Useful tips on top of that:

        r"""<- this r makes me a raw string, thus '\' has no special meaning.
        Otherwise you would need to escape the following backslashes.

        Write inline formulas as :math:`H |\Psi\rangle = E |\Psi\rangle` or displayed equations as 

        .. math ::
            e^{i\pi} + 1 = 0

        You can also include cross references to various things, e.g.,
        :class:`tenpy.algorithms.linalg.np_conserved.array` or :func:`tenpy.tools.math.tondarray`
        """

- Use the python package `nose <http://nose.readthedocs.io/en/latest/>`_ for testing.
  Run it simply with `nosetest` in `tests`.
- A To-Do list can be found in :doc:`docs/todo.rst <todo>`. It also contains a bug list.
- During development, you can introduce `# TODO` comments. But also try to remove them again later!
  During development, unfinished functions should raise a `NotImplementedError`.
  Bugs may be marked with `# BUG`. Also add them to the bug list in docs/todo.rst
- Use relative imports within TenPy, e.g. ::

      from ..algorithms.linalg import np_conserved as npc

Thank You for helping with the development!

.. toctree::
   :maxdepth: 1

   todo
   changes_TenPy

