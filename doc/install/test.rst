Checking the installation
^^^^^^^^^^^^^^^^^^^^^^^^^

The first check of whether tenpy is installed successfully, is to try to import it from within python::

    >>> import tenpy

.. note ::

    If this raises a warning ``Couldn't load compiled cython code. Code will run a bit slower.``, something went wrong with
    the compilation of the Cython parts (or you didn't compile at all). 
    While the code might run slower, the results should still be the same.

The function :func:`tenpy.show_config()` prints information about the used versions of tenpy, numpy and
scipy, as well on the fact whether the Cython parts were compiled and could be imported.

As a further check of the installation you can try to run (one of) the python files in the `examples/` subfolder;
hopefully all of them should run without error.

You can also run the automated testsuite with `pytest <http://pytest.org>`_ to make sure everything works fine.
If you have ``pytest`` installed, you can go to the `tests` folder of the repository, and run the tests with::

    cd tests
    pytest

In case of errors or failures it gives a detailed traceback and possibly some output of the test.
At least the stable releases should run these tests without any failures.

If you can run the examples but not the tests, check whether `pytest` actually uses the correct python version.

The test suite is also run automatically by `github actions <https://github.com/tenpy/tenpy/actions>`_ and with `travis-ci <https://travis-ci.org>`_, results can be inspected `here <https://travis-ci.org/tenpy/tenpy>`_.
