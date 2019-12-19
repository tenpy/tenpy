Tensor Network Python (TeNPy)
=============================

.. image:: https://img.shields.io/github/last-commit/tenpy/tenpy
    :alt: GitHub last commit
    :target: https://github.com/tenpy/tenpy
.. image:: https://readthedocs.org/projects/tenpy/badge/?version=latest
    :alt: Documentation Status
    :target: https://tenpy.readthedocs.io/en/latest/?badge=latest
.. image:: https://travis-ci.org/tenpy/tenpy.svg?branch=master
    :alt: Build
    :target: https://travis-ci.org/tenpy/tenpy
.. image:: https://img.shields.io/github/issues/tenpy/tenpy
    :alt: GitHub issues
    :target: https://github.com/tenpy/tenpy/issues
.. image:: https://img.shields.io/pypi/v/physics-tenpy
    :alt: PyPi
    :target: https://pypi.org/project/physics-tenpy/

TeNPy (short for 'Tensor Network Python') is a Python library for the simulation of strongly correlated quantum systems with tensor networks.

The philosophy of this library is to get a new balance of a good readability and usability for new-comers, and at the same time powerful algorithms and fast development of new algorithms for experts.
For good readability, we include an extensive documentation next to the code, both in Python doc strings and separately as `user guides`, as well as simple example codes and even toy codes, which just demonstrate various algorithms (like TEBD and DMRG) in ~100 lines per file.

How do I get set up?
--------------------
Follow the instructions in the file ``doc/INSTALL.rst``, online at https://tenpy.github.io/INSTALL.html.
The latest version of the source code can be obtained from https://github.com/tenpy/tenpy.

How to read the documentation
-----------------------------
The **documentation is available online** at https://tenpy.github.io.
The documentation is roughly split in two parts: on one hand the full "reference" containing the documentation of all functions,
classes, methods, etc., and on the other hand the "user guide" containing some introductions and additional explanations.

The documentation is based on Python's docstrings, and some additional ``*.rst`` files located in the folder `doc/` of the repository.
All documentation is formated as `reStructuredText <http://www.sphinx-doc.org/en/stable/rest.html>`_,
which means it is quite readable in the source plain text, but can also be converted to other formats.
If you like it simple, you can just use intective python `help()`, Python IDEs of your choice or jupyter notebooks, or just read the source.
Moreover, the documentation is nightly converted into HTML using `Sphinx <http://www.sphinx-doc.org>`_, and made available online at https://tenpy.github.io/.
The big advantages of the (online) HTML documentation are a lot of cross-links between different functions, and even a search function.
If you prefer yet another format, you can try to build the documentation yourself, as described in ``doc/contributing.rst``, online at https://tenpy.github.io/contributing.html.

Help - I looked at the documentation, but I don't understand how ...?
---------------------------------------------------------------------
We have set up a **community forum** at https://tenpy.johannes-hauschild.de/,
where you can post questions and hopefully find answers.
Once you got some experience with TeNPy, you might also be able to contribute to the community and answer some questions yourself ;-)
We also use this forum for official annoucements, for example when we release a new version.

Citing TeNPy
------------
When you use TeNPy for a work published in an academic journal, you can cite `this paper <https://dx.doi.org/10.21468/SciPostPhysLectNotes.5>`_  to acknowledge the work put into the development of TeNPy.
(The license of TeNPy does not force you, however.)
For example, you could add the sentence ``"Calculations were performed using the TeNPy Library (version X.X.X)\cite{tenpy}."`` in the acknowledgements or in the main text.

The corresponding BibTex Entry would be the following (the ``\url{...}`` requires ``\usepackage{hyperref}`` in the LaTeX preamble.)::

    @Article{tenpy,
        title={{Efficient numerical simulations with Tensor Networks: Tensor Network  Python (TeNPy)}},
        author={Johannes Hauschild and Frank Pollmann},
        journal={SciPost Phys. Lect. Notes},
        pages={5},
        year={2018},
        publisher={SciPost},
        doi={10.21468/SciPostPhysLectNotes.5},
        url={https://scipost.org/10.21468/SciPostPhysLectNotes.5},
        archiveprefix={arXiv},
        eprint={1805.00055},
        note={Code available from \url{https://github.com/tenpy/tenpy}},
    }


I found a bug
-------------
You might want to check the `github issues <https://github.com/tenpy/tenpy/issues>`_, if someone else already reported the same problem.
To report a new bug, just `open a new issue <https://github.com/tenpy/tenpy/issues/new>`_ on github.
If you already know how to fix it, you can just create a pull request :)
If you are not sure whether your problem is a bug or a feature, you can also ask for help in the `TeNPy forum <https://tenpy.johannes-hauschild.de/>`_.

License
-------
The code is licensed under GPL-v3.0 given in the file ``LICENSE`` of the repository, 
in the online documentation readable at https://tenpy.github.io/license.html.
