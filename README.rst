Welcome to Tensor Network Python (TeNPy)!
=========================================

TeNPy (short for Tensor Network Python) is a Python library for the simulation of strongly correlated quantum systems with tensor networks.

The philosophy of this library is to get a new balance of a good readability and usability for new-comers, and at the same time powerful algorithms and fast development of new algorithms for experts.
For good readability, we include an extensive documentation next to the code, both in Python doc strings and separately as `user guides`, as well as simple example codes and even toy codes, which just demonstrate various algorithms (like TEBD and DMRG) in ~100 Lines per file.

How do I get set up?
--------------------
Follow the instructions in the file ``doc/INSTALL.rst``, online `here <https://tenpy.github.io/INSTALL.html>`_.

How to read the documentation
-----------------------------
The **documentation is available online** at https://tenpy.github.io.

The documentation is based on Python's docstrings, and some additional ``*.rst`` files located in the folder `doc/` of the repository.
All documentation should (like this file) be formated as `reStructuredText <http://www.sphinx-doc.org/en/stable/rest.html>`_,
which means it is quite readable in the source plain text, but can also be converted to other formats.
If you like it simple, you can just use intective python `help()`, Python IDEs of your choice (or jupyter notebooks), or just read the source.
The documentation is nightly converted into HTML using `Sphinx <http://www.sphinx-doc.org>`_, and made available online at https://tenpy.github.io/.
The big advantage of the HTML documentation are a lot of cross-links between different functions, and even a search function.
If you prefer yet another format, you can try to build the documentation yourself, as described in :doc:`doc/contributing.rst <contributing>`.

The documentation is roughly split in two parts: on one hand the full `reference` containing the documentation of all functions,
classes, methods, etc., and on the other hand the `userguide` containing some introductions and additional explanations.

Help - I looked at the documentation, but I don't understand how ...?
---------------------------------------------------------------------
We have set up a **community forum** at https://tenpy.johannes-hauschild.de/, where you can post questions and hopefully find answers.
Once you got some experience, you might also be able to contribute to the community and answer some questions yourself ;-)
We will also use this forum for official annoucements, for example when we release a new version.

I found a bug
-------------
You might want to check the `github issues <https://github.com/tenpy/tenpy/issues>`_, if someone else already reported the same problem.
To report a new bug, just `open a new issue <https://github.com/tenpy/tenpy/issues/new>`_ on github.
If you already know how to fix it, you can just create a pull request :)

License
-------
The code is licensed under GPL-v3.0 given in the file ``LICENSE`` of the repository, 
in the online documentation readable at https://tenpy.github.io/license.html.
