Welcome to TenPy!
=================================

Introduction
------------
TenPy is a Python Library for the simulation of strongly correlated quantum systems with tensor networks.
It's based on TenPy2, but tries to be readable and easy to understand for new peoples.
A detailed list of changes compared to the previous TenPy can be found in docs/changes_TenPy.rst

How do I get set up?
--------------------
Follow the instructions in :doc:`docs/INSTALL.rst <INSTALL>`.

How to read the documentation
-----------------------------
The documentation is based on Python's docstrings, and some additional .rst files located in docs/.

All documentation should be formated as `reStructuredText <http://www.sphinx-doc.org/en/stable/rest.html>`_,
This means it's readable in the source plain text, but one can also convert it to other formats.
If you like it simply, you can just use intective python `help()`, Python IDEs of your choice, or just read the source.

Alternatively, you can also use `Sphinx <http://www.sphinx-doc.org>`_ to generate the full documentation 
in various formats including HTML or PDF.
You can install `Sphinx`_ and the extension `numpydoc <http://pypi.python.org/pypi/numpydoc>`_ with::

    sudo pip install --upgrade sphinx numpydoc

.. note ::
   If you don't have super user rights, try ``pip install --upgrade --user sphinx numpydoc`` to 
   install the packages to your home directory.

Afterwards, go to the folder `docs` and run the following command::

    make html

This should generate the html documentation in `docs/sphinx_build/html`.
Simply open this folder (or to be precise: the file `index.html` in it) in your webbroser
and enjoy this and other documentation beautifully rendered, with cross links, Math forumulas
and even a search function :-).

.. note ::
   Building the html documentation requires loading the modules.
   Thus make sure that the folder tenpy is included in you `$PYTHONPATH`.



Contents
--------
.. toctree::
   :maxdepth: 2

   userguide
   reference/tenpy


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



I found a bug
-------------
If you can fix it, do it. git commit with a message starting as 'BUG: description'.

Alternatively, you can report it in the `BUGS` section of :doc:`docs/todo.rst <todo>`.
