User Guide
==========

First a short warning: the term 'user guide' might be a bit misleading: 
this part of the documentation simply covers everything but what is documented directly in the source - 
the latter can be found in the :doc:`reference <reference/tenpy>`.

Library overview
----------------
The root directory of this git repository contains the following folders:

README.rst
    Starting point for reading the documentation; explains how to build html documentation.
doc
    a folder containing the documentation: the user guide is contained in the ``*.rst`` files.
    Contains a make file for building the documentation, run ``make help`` for the different options.
    The necessary files for the reference in `doc/reference` can be auto-generated/updated with ``make src2html``.
tenpy
    The actual source code of the library.
    Every subfolder contains an ``__init__.py`` file with a summary what the modules in it are good for.
    (This file is also necessary to mark the folder as part of the python package.
    Consequently, other subfolders of the git repo should not include a ``__init__.py`` file.)
examples
    some example files demonstrating the usage of the library.
tests
    Contains files with test routines, to be used with `nose`. 
    If you are set up correctly and have `nose` installed, you can run the test suite with
    ``nosetests`` from within the `tests/` folder.


Content
-------

.. toctree::
   :maxdepth: 1

   INSTALL
   literature
   IntroNpc
   changes_TenPy
   contribution
   todo
