Welcome to TeNPy!
=================

Introduction
------------
TeNPy (short for TEnsor Networks Python) is a Python library for the simulation of strongly correlated quantum systems with tensor networks.
It originated from an earlier version of the library (which is not open source). 
However, that early version grew over the years and became quite unreadable for newcommers.
The philosophy of this version is to get a new balance of readability and at the same time powerful algorithms.
Therefore, next to the code it includes an extensive documentation (both in Python doc strings and separately as "user
guides") as well as simple example codes, and even some toy codes,
which demonstrate various algorithms (like TEBD and DMRG) in ~100 Lines per file.


How do I get set up?
--------------------
Follow the instructions in :doc:`doc/INSTALL.rst <INSTALL>`.

How to read the documentation
-----------------------------
The documentation is based on Python's docstrings, and some additional ``*.rst`` files located in `doc/`.

All documentation should be formated as `reStructuredText <http://www.sphinx-doc.org/en/stable/rest.html>`_,
This means it's readable in the source plain text, but one can also convert it to other formats.
If you like it simple, you can just use intective python `help()`, Python IDEs of your choice, or just read the source.

Alternatively, you can also use `Sphinx <http://www.sphinx-doc.org>`_ to generate the full documentation 
in various formats including HTML or PDF.
You can install `Sphinx`_ and the extension `numpydoc <http://pypi.python.org/pypi/numpydoc>`_ with::

    sudo pip install --upgrade sphinx numpydoc

.. note ::

   If you don't have super user rights, try ``pip install --upgrade --user sphinx numpydoc`` to 
   install the packages to your home directory.

Afterwards, go to the folder `doc/` and run the following command::

    make html

This should generate the html documentation in `doc/sphinx_build/html`.
Simply open this folder (or to be precise: the file `index.html` in it) in your webbroser
and enjoy this and other documentation beautifully rendered, with cross links, math formulas
and even a search function :-).

.. note ::

   Building the (html) documentation requires loading the modules.
   Thus make sure that the folder `tenpy` is included in your ``$PYTHONPATH``,
   as described in :doc:`doc/INSTALL.rst <INSTALL>`.


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
If you know how to fix it, just do it. ``git commit`` with a message containing ``bug`` in the description.

Alternatively, you can report it in the `BUGS` section of :doc:`doc/todo.rst <todo>`.

License
-------
The license for this code is given in the file ``LICENSE`` of the code, in the online documentation included in  
:doc:`this page <license>`.
