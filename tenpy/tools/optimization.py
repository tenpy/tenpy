"""Optimization options for this library.

Let me start with a `quote <http://wiki.c2.com/?RulesOfOptimization>`_ of "Micheal Jackson"
(a programmer, not the musician)::

    First rule of optimization: "Don't do it."
    Second rule of optimization (for experts only): "Don't do it yet."
    Third rule of optimization: "Profile before optimizing."

Luckily, following the third optimization rule, namely profiling code, is
fairly simple in python, see the `documentation <https://docs.python.org/3/library/profile.html>`_.
If you have a python skript running your code, you can simply call it with
``python -m "cProfile" -s "tottime" your_skript.py``. Alternatively, save the profiling statistics
with ``python -m "cProfile" -o "profile_data.stat" your_skript.py`` and
run these few lines of python code::

    import pstats
    p = pstats.Pstats("profile_data.stat")
    p.sort_stats('cumtime')  # sort by 'cumtime' column
    p.print_stats(30)   # prints first 30 entries

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
   while internally the faster, pre-compiled cython code from ``tenpy/linalg/_npc_helper.pyx``
   is used. This should also be a safe thing to do.
   The replacement of the optimized functions is done by the decorator :func:`use_cython`.
4) One of the great things about python is its dynamical nature - anything can be done at runtime.
   In that spirit, this module allows to set a global  "optimization level" which can be changed
   *dynamically* (i.e., during runtime) with :func:`set_level`. The library will then try some
   extra optimiztion, most notably skip sanity checks of arguments.
   The possible choices for this global level are given by the :class:`OptimizationFlag`.
   The default initial value for the global optimization level can be adjusted by the environment
   variable `TENPY_OPTIMIZE`.

   .. warning ::
        When this optimizing is enabled, we skip (some) sanity checks.
        Thus, errors will not be detected that easily, and debugging is much harder!
        We recommend to use this kind of optimization only for code which you succesfully have run
        before with (very) similar parmeters!
        Enable this optimization only during the parts of the code where it is really necessary.
        The context manager :class:`temporary_level` can help with that.
        Check whether it actually helps - if it doesn't, keep the optimization disabled!
        Some parts of the library already do that as well (e.g. DMRG after the first sweep).

5) You might want to try some different compile time options for the cython code, set in the
   `setup.py` in the top directory of the repository.
   Since the `setup.py` reads out the `TENPY_OPTIMIZE`
   environment variable, you can simple use an ``export TENPY_OPTIMIZE=3`` (in your bash/terminal)
   right before compilation. An ``export TENPY_OPTIMIZE=0`` activates profiling hooks instead.

   .. warning ::
       This increases the probability of getting segmentation faults and anyway might not
       help that much; in the crucial parts of the cython code, these optimizations are already
       applied. We do *not* recommend using this!
"""
# Copyright 2018 TeNPy Developers

from enum import IntEnum
import warnings
import os

all = [
    'bottleneck', 'OptimizationFlag', 'temporary_level', 'set_level', 'get_level', 'optimize',
    'use_cython', 'have_cython_functions'
]

try:
    import bottleneck
except:
    bottleneck = None
"""bool whether the import of the cython file tenpy/linalg/_npc_helper.pyx succeeded"""
have_cython_functions = None  # set to True or False in the first call of `use_cython`


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


class temporary_level:
    """Context manager to temporarily set the optimization level to a different value.

    Parameters
    ----------
    temporary_level : int | OptimizationFlag | str | None
        The optimization level to be set during the context.
        `None` defaults to the current value of the optimization level.

    Attributes
    ----------
    temporary_level : None | OptimizationFlag
        The optimization level to be set during the context.
    _old_level : OptimizationFlag
        Optimization level to be restored at the end of the context manager.

    Examples
    --------
    It is recommended to use this context manager in a ``with`` statement::

        # optimization level default
        with temporary_level(OptimizationFlag.safe):
            do_some_stuff()  # temporarily have Optimization level `safe`
            # you can even change the optimization level to something else:
            set_level(OptimizationFlag.skip_args_check)
            do_some_really_heavy_stuff()
        # here we are back to the optimization level as before the ``with ...`` statement
    """

    def __init__(self, temporary_level):
        self.temporary_level = temporary_level

    def __enter__(self):
        "enter the context manager"
        self._old_level = get_level()
        if self.temporary_level is not None:
            set_level(self.temporary_level)

    def __exit__(self, exc_type, exc_value, traceback):
        "exit the context manager"
        set_level(self._old_level)


def to_OptimizationFlag(level):
    "Convert strings and int to a valid OptimizationFlag. ``None`` defaults to the current level."
    if level is None:
        return get_level()
    if isinstance(level, str):
        try:
            level = int(level)
        except ValueError:
            level = OptimizationFlag[level]
    return OptimizationFlag(level)


def set_level(level=1):
    """Set the global optimization level.

    Parameters
    ----------
    level : int | OptimizationFlag | str | None
        The new global optimization level to be set.
        ``None`` defaults to keeping the current level.
    """
    global _level
    _level = to_OptimizationFlag(level)


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
        True if the algorithms should try to optimize, i.e., whether the global
        "optimization level" is equal or higher than the level to compare to.
    """
    global _level
    return (_level >= level_compare)


def use_cython(func=None, replacement=None, check_doc=True):
    """Decorator to replace a function with a Cython-equivalent from _npc_helper.pyx.

    This is a `decorator <https://docs.python.org/3.7/glossary.html#term-decorator>`_, which is
    supposed to be used in front of function definitions with an ``@`` sign, for example::

        @use_cython
        def my_slow_function(a):
            "some example function with slow python loops"
            result = 0.
            for i in range(a.shape[0]):
                for j in range(a.shape[1]):
                    #... heavy calculations ...
                    result += np.cos(a[i, j]**2) * (i + j)
            return result

    This decorator indicates that there is a `Cython <https://cython.org>`_ implementation in
    the file ``tenpy/linalg/_npc_helper.pyx``, which should have the same signature (i.e. same
    arguments and return values) as the decorated function, and can be used as a replacement for
    the decorated function. However, if the cython code could not be compiled on your system
    (or if the environment variable ``TENPY_OPTIMIZE`` is set to negative values),
    we just pass the previous function.

    Note: in case that the decorator is used for a class method, the corresponding Cython version
    needs to have an ``@cython.binding(True)``.

    Parameters
    ----------
    func : function
        The defined function
    replacement : string | None
        The name of the function defined in ``tenpy/linalg/_npc_helper.pyx`` which should
        replace the decorated function.
        ``None`` defaults to the name of the decorated function,
        e.g., in the above example `my_slow_function`.
    check_doc : bool
        If True, we check that the cython version of the function has the exact same doc string
        (up to a possible first line containing the function signature) to exclude typos and
        inconsistent versions.

    Returns
    -------
    replacement_func : function
        The function replacing the decorated function `func`.
        If the cython code can not be loaded, this is just `func`,
        otherwise it's the cython version specified by `replacement`.
    """
    if func is None:
        # someone used ``@use_cython(replacement=...)``
        # so we need to return another decorator function
        def _decorator(func):
            return use_cython(func, replacement, check_doc)

        return _decorator
    global _npc_helper_module
    global have_cython_functions
    if have_cython_functions is None:
        if optimize(OptimizationFlag.default):
            try:
                from ..linalg import _npc_helper
                _npc_helper_module = _npc_helper
                have_cython_functions = True
            except ImportError:
                warnings.warn("Couldn't load compiled cython code. Code will run a bit slower.")
                have_cython_functions = False
        else:
            warnings.warn("Don't load compiled cython code due to TENPY_OPTMIZE. "
                          "Code will run a bit slower.")
            have_cython_functions = False
    if not have_cython_functions:
        # can't provide a faster version: cython module not available
        return func
    if replacement is None:
        replacement = func.__name__
    fast_func = _npc_helper_module.__dict__.get(replacement, None)
    if fast_func is None:
        msg = "can't find cython function {0!s} to replace python function {1!s} in {2!r}"
        msg = msg.format(replacement, func.__name__, func.__module__)
        raise ValueError(msg)
    if check_doc:
        import inspect
        clean_fdoc = inspect.getdoc(func)
        clean_cdoc = inspect.getdoc(fast_func)
        cdoc = fast_func.__doc__
        # if the cython compiler directive 'embedsignature' is used, the first line contains the
        # function signature, so the doc string starts only with the second line
        clean_cdoc2 = inspect.cleandoc(cdoc[cdoc.find("\n") + 1:])
        if clean_fdoc != clean_cdoc and clean_fdoc != clean_cdoc2:
            msg = "cython version of {0!s} has different doc-string".format(func.__name__)
            raise ValueError(msg)
    return fast_func


# private global variables
_level = OptimizationFlag.default  # set default optimization level
set_level(os.getenv("TENPY_OPTIMIZE", default=None))  # update from environment variable
_npc_helper_module = None
