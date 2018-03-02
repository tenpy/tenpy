"""Optimization options for this library.

Let me start with a `quote <http://wiki.c2.com/?RulesOfOptimization>`_ of "Micheal Jackson"
(a programmer, not the musician)::

    First rule of optimization: "Don't do it."
    Second rule of optimization (for experts only): "Don't do it yet."
    Third rule of optimization: "Profile before optimizing."

That being said, I actually did profile and optimize (parts of) the library; and there are a few
knobs you can turn to tweak the most out of this library, explained in the following.

1) Simply install the 'bottleneck' python package, which allows to optimize slow parts of numpy,
   most notably 'NaN' checking.
2) Figure out which numpy/scipy/python you are using. As explained in :doc:`../INSTALL`,
   we recommend to use the Python distributed provided by Intel or Anaconda. They ship with numpy
   and scipy which use Intels MKL library, such that e.g. ``np.tensordot`` is parallelized to use
   multiple cores.
3) In case you didn't do that yet: some parts of the library are written in both python and Cython
   with the same interface, so you can simply compile the Cython code, as explained in
   :doc:`../INSTALL`. Then everything should work the same way from a user perspective,
   while internally the faster, pre-compiled cython code is used.
   This should also be a safe thing to do.
4) One of the great things about python is its dynamical nature - anything can be done at runtime.
   In that spirit, this module allows to set a global  "optimization level" which can be changed
   *dynamically* (i.e., during runtime) with :func:`set_level`. The library will then try some
   extra optimiztion, e.g., skip sanity checks.
   See :class:`OptimizationFlag` and :func:`set_level` for more details.

   .. warning ::
        When this optimizing is enabled, we skip (some) sanity checks.
        Thus, errors will not be detected that easily, and debugging is much harder!
        Use this kind of optimization only for code which you succesfully ran before
        with (very) similar parmeters!
        Enable this optimization only during the parts of the code where it is really necessary.
        Check whether it actually helps - if it doesn't, keep the optimization disabled!

5) You might want to try some different compile time options for the cython code, set in the
   `setup.py` in the top directory.  Since the `setup.py` reads out the `TENPY_OPTIMIZE`
   environment variable, you can simple use an ``export TENPY_OPTIMIZE=1`` (in your bash/terminal)
   right before compilation. An ``export TENPY_OPTIMIZE=-1`` activates profiling hooks instead.

   .. warning ::
       This increases the probability of getting segmentation faults and anyway might not
       help that much; in the crucial parts of the cython code, these optimizations are already
       applied. We do *not* recommend using this!

6) If you're still not satisfied with the speed of this library: this code is written in python,
   which is 'slow' due to its dynamical nature. Use a library written in another programming
   language, there are plenty of them out there...

Finally, let me note that following the third optimization rule, namely profiling code, is
fairly simple in python, see the `documentation <https://docs.python.org/3/library/profile.html>`_.
If you have a python skript running your code, you can simply call it with
``python -m "cProfile" -s "tottime" your_skript.py``. Alternative, save the profiling statistics
with ``python -m "cProfile" -o "profile_data.stat" your_skript.py`` and
run these few lines of python code::

    import pstats
    p = pstats.Pstats("profile_data.stat")
    p.sort_stats('cumtime')  # sort by 'cumtime' column
    p.print_stats(30)   # prints first 30 entries
"""
# Copyright 2018 TeNPy Developers

from enum import IntEnum

all = ['bottleneck', 'OptimizationFlag', 'set_level', 'get_level', 'optimize', 'debug']

try:
    import bottleneck
except:
    bottleneck = None


class OptimizationFlag(IntEnum):
    """Options for the global 'optimization level' used for dynamical optimizations.

    Whether we optimize dynamically is decided by comparison of the global "optimization level"
    with one of the following flags. A higher level *includes* all the previous optimizations.

    ===== ================ ========================================================================
    Level Flag             Description
    ===== ================ ========================================================================
    0     none             Don't do any optimizations, i.e., run many sanity checks.
                           Used for testing.
    ----- ---------------- ------------------------------------------------------------------------
    1     default          Skip really unnecessary sanity checks, but also don't try any
                           optional optimizations if they might give an overhead.
    ----- ---------------- ------------------------------------------------------------------------
    2     safe             Activate safe optimizations in algorithms, even if they might
                           give a small overhead.
                           Example: Try to compress the MPO representing the hamiltonian.
    ----- ---------------- ------------------------------------------------------------------------
    3     skip_arg_checks  Unsafe! Skip (some) class sanity tests and (function) argument checks.
    ===== ================ ========================================================================

   .. warning ::
        When unsafe optimizations are enabled, errors will not be detected that easily,
        debugging is much harder, and you might even get segmentation faults in the compiled parts.
        Use this kind of optimization only for code which you succesfully ran before
        with (very) similar parmeters and disabled optimiztions!
        Enable this optimization only during the parts of the code where it is really necessary.
        Check whether it actually helps - if it doesn't, keep the optimization disabled!
    """
    none = 0
    default = 1
    safe = 2
    skip_arg_checks = 3


def set_level(new_level=1):
    """Set the global optimization level.

    Parameters
    ----------
    new_level : int | OptimizationFlag
        The new global optimization level to be set.
    """
    new_level = OptimizationFlag(new_level)
    global _level
    _level = new_level


def get_level():
    """Return the global optimization level."""
    global _level
    return _level


def optimize(level_compare=OptimizationFlag.default):
    """Called by algorithms to check whether it should (try to) do some optimizations.

    Parameters
    ----------
    level_compare : OptimizationFlag
        At which level to start optimization, i.e., how safe the suggested optimization is.

    Returns
    -------
    optimize : bool
        True if the algorithms should try to optimize, i.e., i.e. whether the global
        "optimization level" is equal or higher than the level to compare to.
    """
    global _level
    return (_level >= level_compare)


_level = OptimizationFlag.none  # by default don't optimize
