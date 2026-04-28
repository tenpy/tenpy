Coding Guidelines
=================
We ask you to comply with the following guidelines for contributions
Most of our coding guidelines are explicitly checked by the GitHub actions that run on every pull
request, but there are some extra guidelines below.

We have a `pre-commit <https://pre-commit.com/>`_ configuration in the repository that should make
it easy for somewhat experienced developers to adhere to the automatically checked rules.
If it detects any violation it stops the commit, reports them and automatically fixes them if possible.
In addition, follow the list of guidelines below, which is not automatically checked by tools.

If you are having trouble with the guidelines, please don't let that stop you from sharing your contribution.
Someone from the team can finish up.


Guidelines
~~~~~~~~~~

- Summarize the changes you have made in the changelog.
  Make a new file, and e.g. name it after the PR, ``doc/changelog/latest/pr_401.txt`` or similar.
  Make sure to use ``.txt`` suffix.
  It should contain only bullet points.
  See e.g. `this example <https://github.com/tenpy/tenpy/blob/b49485e7cfdfe9ec4fe740e6dbeea3451783840b/doc/changelog/latest/pr_520.txt>`_.

- If you add a new toycode or example: add a reference to include it in the documentation.

- Use relative imports within cyten. Example::

      from ..spaces import ElementarySpace


- Include documentation, Put a docstring on every new module, class and function.
  See the section on docs below.

- Include tests for your new features. See the existing ones in the ``tests/`` folder.

- Long running tests are marked with ``@pytest.mark.slow``. You can exclude them by running only
  ``pytest -m "not slow"``. If your new tests are slow (``> 10s`` total), mark them accordingly.

- Prefer GitHub issues over todo comments in the code.
  Do not leave todo comments unless you take responsibility to take care of the todo later.
  Even if you do, GitHub issues are a better place to keep track of open todos.

- Unfinished functions should ``raise NotImplementedError()``.



Linter rules
~~~~~~~~~~~~
The following rules are explicitly checked both by GitHub actions, as well as by the pre-commit
configuration

- Basic sanity checks (dont push private keys, yaml files should be valid, no debugging leftovers, ...)

- Linting of the python code using ``ruff check``, with the rules configured in ``pyproject.toml``.

- Linting of docstrings using ``flake8-rst-docstrings``, with the rules configured in ``.flake8``

- Autoformatting using ``ruff format``. The pre-commit simply does these changes and amends them,
  the GitHub action only checks that another run would not change anything but does not propose
  these changes. You can either read the diff in the action logs, or better run ``ruff`` locally,
  e.g. via ``pre-commit``.

- Check that text files do not contain the specific strings ``FIXME`` and ``DONTSHIP``.
  You may use them in your workflow as reminders to do something before committing/pushing.


Workarounds
~~~~~~~~~~~
Automated tools are never perfect.
If the tool complains and you are reasonably sure that it is wrong, you can use the following workarounds.
Use them *responsibly* and *sparingly*.

- You can disable linters locally using ``# noqa <rule>`` comments.
  See `ruff: Error suppression <https://docs.astral.sh/ruff/linter/#error-suppression>`_.

- You can ignore linter rules in ``pyproject.toml``. Prefer per-file ignores over global ignores.
  See `ruff: Error suppression <https://docs.astral.sh/ruff/linter/#error-suppression>`_.

- You can disable autoformatting locally by using
  `pragma comments <https://docs.astral.sh/ruff/formatter/#format-suppression>`_.

- You can exclude files for ruff in ``pyproject.toml``. Do this only with a solid reason!
  See `ruff: exclude <https://docs.astral.sh/ruff/settings/#exclude>`_.


Documentation
~~~~~~~~~~~~~
- Every function/class/module should be documented by its doc-string, see :pep:`257`.

  Additional documentation for the user guide is in the folder ``docs/``.

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

        To refer to variables within math, use `\mathtt{varname}`.

        """

- We've created a sphinx extensions for `documenting config-option dictionaries <https://sphinx-cfg-options.readthedocs.io/en/latest/>`_.
  If a class takes a dictionary of options, we usually call it ``options``,
  convert it to a :class:`~tenpy.tools.params.Config` at the very beginning of the ``__init__`` with
  :func:`~tenpy.tools.params.asConfig`, save it as ``self.options``,
  and document it in the class doc-string with a ``.. cfg:config ::`` directive.
  The name of the ``config`` should usually be the class-name (if that is sufficiently unique),
  or for algorithms directly the common name of the algorithm, e.g. "DMRG"; use the same name for the
  use the same name for the documentation of the ``.. cfg:config ::`` directive as for the
  :class:`~tenpy.tools.params.Config` class instance.
  Attributes which are simply read-out options should be documented by just referencing the options with the
  ``:cfg:option:`configname.optionname``` role.
