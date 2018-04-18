Welcome to Tensor Networks Python (TeNPy)!
==========================================

TeNPy (short for TEnsor Networks Python) is a Python library for the simulation of strongly correlated quantum systems with tensor networks.

The philosophy of this library is to get a new balance of a good readability and usability for new-comers, and at the same time powerful algorithms and fast development of new algorithms for experts.
For good readability, next to the code we include an extensive documentation, both in Python doc strings and separately as `user guides`, as well as simple example codes and even toy codes, which just demonstrate various algorithms (like TEBD and DMRG) in ~100 Lines per file.

How do I get set up?
--------------------
Follow the instructions in :doc:`doc/INSTALL.rst <INSTALL>`.

How to read the documentation
-----------------------------
The **documentation is available online** at `github pages <https://tenpy.github.io/documentation/>`_.

The documentation is based on Python's docstrings, and some additional ``*.rst`` files located in `doc/`.

All documentation should be formated as `reStructuredText <http://www.sphinx-doc.org/en/stable/rest.html>`_,
This means it is readable in the source plain text, but can also be converted to other formats.
If you like it simple, you can just use intective python `help()`, Python IDEs of your choice (or jupyter notebooks), or just read the source.

The documentation is converted into HTML using `Sphinx <http://www.sphinx-doc.org>`_, as mentioned above available online at `github pages <https://tenpy.github.io/documentation/>`_.
You can also build the documentation yourself, as described in :doc:`doc/contributing.rst <contributing>`.
Sphinx can generate the documentation in various output formats, including HTML and LATEX/PDF.

The documentation is roughly split in two parts: the full reference containing the documentation of all functions,
classes, methods, etc, and the `userguide` containing some introductions and additional explanations.

I found a bug
-------------
You might want to check the `github issues <https://github.com/tenpy/tenpy/issues>`_, if someone else already reported the same problem.
To report a new bug, just `open a new issue <https://github.com/tenpy/tenpy/issues/new>`_ on github.
If you already know how to fix it, you can just create a pull request :)

License
-------
The license for this code is given in the file ``LICENSE`` of the code, in the online documentation included in  
:doc:`this page <license>`.
