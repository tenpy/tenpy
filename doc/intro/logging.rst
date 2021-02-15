Logging and terminal output
===========================

By default, calling (almost) any function in TeNPy will not print output, appart from error messages, tracebacks, and warnings.

Of course, when you get an error message, you should be concerned to find out what it is about and how to fix it. 
(If you believe it is a bug, `report <https://github.com/tenpy/tenpy/issues/new/choose>`_ it.)
Warnings can be reported either using ``warnings.warn(...)`` or with the logging mechanis ``logger.warn(...)``.
The former is used for warnings about things in your setup that you *should* fix.
The latter give you notifications about bad things that can happen in calculations, e.g. bad conditioning of a matrix, but there
is not much you can do about it. 
Those warnings indicate that you should take your results with a grain of salt and carefully ensure everything is well behaved.




.. todo ::
    explain more!

Configuring logging
-------------------

.. todo ::
    ????

You can also filter out messages from specific loggers. For example, you can suppress the parameter prints like this::

    ???


.. note ::

    You might also want to use ``logging.captureWarnings(True)``.



How to write logging (and warning) code
---------------------------------------
You can read the `official logging tutorial <https://docs.python.org/3/howto/logging.html>`_, 
but it's actually straight-forward.

1.  Add the following lines at the top of your module::

        import warnings
        import logging
        logger = logging.getLogger(__name__)

    .. note ::

        Some classes that you might want to subclass (e.g. all models) provide a `logger` as 
        ``self.logger`` class attribute. It's recommended to use that one instead from inside methods.

2.  Inside your funtions/methods/..., make calls like this::

        if is_likely_bad(options['parameter']):
            # fixable!
            warnings.warn("This is a bad parameter, you shouldn't do this!")
        if "old_parameter" in options:
            warnings.warn("Use `new_parameter` instead of `old_parameter`", FutureWarning, 2)
        logger.info("starting some lengthy calculation")
        n_steps = do_calculation()
        if something_bad_happened():
            # can't do anything about it
            logger.warn("Something bad happend")
        logger.info("calculation finished after %d steps", n_steps)

    You can use `printf-formatting <https://docs.python.org/3/library/stdtypes.html#printf-style-string-formatting>`_
    for the arguments of ``logger.debug(...), logger.info(...), logger.warn(...)``.
