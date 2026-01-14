"""Optimization options for this library.

Let me start with a `quote <http://wiki.c2.com/?RulesOfOptimization>`_ of "Micheal Jackson"
(a programmer, not the musician)::

    First rule of optimization: "Don't do it."
    Second rule of optimization (for experts only): "Don't do it yet."
    Third rule of optimization: "Profile before optimizing."

Luckily, following the third optimization rule, namely profiling code, is
fairly simple in python, see the `documentation <https://docs.python.org/3/library/profile.html>`_.
If you have a python script running your code, you can simply call it with
``python -m "cProfile" -s "tottime" your_script.py``. Alternatively, save the profiling statistics
with ``python -m "cProfile" -o "profile_data.stat" your_script.py`` and
run these few lines of python code::

    import pstats

    p = pstats.Pstats('profile_data.stat')
    p.sort_stats('cumtime')  # sort by 'cumtime' column
    p.print_stats(30)  # prints first 30 entries

That being said, I actually did profile and optimize (parts of) the library; and there are a few
knobs you can turn to tweak the most out of this library, explained in the following.

1) Figure out which numpy/scipy/python you are using. As explained in :doc:`../INSTALL`,
   we recommend to use the Python distributed provided by Intel or Anaconda. They ship with numpy
   and scipy which use Intels MKL library, such that e.g. ``np.tensordot`` is parallelized to use
   multiple cores.

#) One of the great things about python is its dynamical nature - anything can be done at runtime.
   In that spirit, this module allows to set a global  "optimization level" which can be changed
   *dynamically* (i.e., during runtime) with :func:`set_level`. The library will then try some
   extra optimization, most notably skip sanity checks of arguments.
   The possible choices for this global level are given by the :class:`OptimizationFlag`.
   The default initial value for the global optimization level can be adjusted by the environment
   variable `TENPY_OPTIMIZE`.

   .. warning ::
        When this optimizing is enabled, we skip (some) sanity checks.
        Thus, errors will not be detected that easily, and debugging is much harder!
        We recommend to use this kind of optimization only for code which you successfully have run
        before with (very) similar parameters!
        Enable this optimization only during the parts of the code where it is really necessary.
        The context manager :class:`temporary_level` can help with that.
        Check whether it actually helps - if it doesn't, keep the optimization disabled!
        Some parts of the library already do that as well (e.g. DMRG after the first sweep).

"""
# Copyright (C) TeNPy Developers, Apache license

import os
from enum import IntEnum

__all__ = [
    'OptimizationFlag',
    'temporary_level',
    'to_OptimizationFlag',
    'set_level',
    'get_level',
    'optimize',
]


class OptimizationFlag(IntEnum):
    """Options for the global 'optimization level' used for dynamical optimizations.

    Whether we optimize dynamically is decided by comparison of the global "optimization level"
    with one of the following flags. A higher level *includes* all the previous optimizations.

    ===== ================ =======================================================================
    Level Flag             Description
    ===== ================ =======================================================================
    0     none             Don't do any optimizations, i.e., run many sanity checks.
                           Used for testing.
    ----- ---------------- -----------------------------------------------------------------------
    1     default          Skip really unnecessary sanity checks, but also don't try any
                           optional optimizations if they might give an overhead.
    ----- ---------------- -----------------------------------------------------------------------
    2     safe             Activate safe optimizations in algorithms, even if they might
                           give a small overhead.
                           Example: Try to compress the MPO representing the hamiltonian.
    ----- ---------------- -----------------------------------------------------------------------
    3     skip_arg_checks  Unsafe! Skip (some) class sanity tests and (function) argument checks.
    ===== ================ =======================================================================

    .. warning ::
        When unsafe optimizations are enabled, errors will not be detected that easily,
        debugging is much harder, and you might even get segmentation faults in the compiled parts.
        Use this kind of optimization only for code which you successfully ran before
        with (very) similar parameters and disabled optimizations!
        Enable this optimization only during the parts of the code where it is really necessary.
        Check whether it actually helps - if it doesn't, keep the optimization disabled!
    """

    none = 0
    default = 1
    safe = 2
    skip_arg_checks = 3

    @classmethod
    def from_bytes(cls, bytes, byteorder, *, signed=False):
        """Like ``int.from_bytes``, which has a docstring which sphinx cant parse"""
        return super(OptimizationFlag, cls).from_bytes(bytes, byteorder, signed=signed)

    def to_bytes(self, length=1, byteorder='big', *, signed=False):
        """Like ``int.to_bytes``, which has a docstring which sphinx cant parse"""
        return super().to_bytes(length, byteorder, signed=signed)


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
        """Enter the context manager."""
        self._old_level = get_level()
        if self.temporary_level is not None:
            set_level(self.temporary_level)

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager."""
        set_level(self._old_level)


def to_OptimizationFlag(level):
    """Convert strings and int to a valid OptimizationFlag.

    ``None`` defaults to the current level.
    """
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
    return _level >= level_compare


# private global variables
_level = OptimizationFlag.default  # set default optimization level
set_level(os.getenv('TENPY_OPTIMIZE', default=None))  # update from environment variable
