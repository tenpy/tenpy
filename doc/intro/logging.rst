Logging and terminal output
===========================

By default, calling (almost) any function in TeNPy will not print output, appart from error messages, tracebacks, and warnings.
Instead, we use Python's :mod:`logging` module to allow fine-grained redirecting of status messages etc.

Of course, when you get an error message, you should be concerned to find out what it is about and how to fix it. 
(If you believe it is a bug, `report <https://github.com/tenpy/tenpy/issues/new/choose>`_ it.)
Warnings can be reported either using ``warnings.warn(...)`` or with the logging mechanism ``logger.warn(...)``.
The former is used for warnings about things in your setup that you *should* fix.
The latter give you notifications about bad things that can happen in calculations, e.g. bad conditioning of a matrix, but there
is not much you can do about it. Those warnings indicate that you should take your results with a grain of salt and carefully double-check them.


Configuring logging
-------------------
If you also want to see status messages (e.g. during a DMRG run whenever a checkpoint is reached), you can use
set the :mod:`logging` level to `logging.INFO` with the following, basic setup::

    import logging
    logging.basicConfig(level=logging.INFO)

We use this snippet in our examples to activate the printing of info messages to the standard output stream.
For really detailed output, you can even set the level to `logging.DEBUG`.
:func:`logging.basicConfig` also takes a `filename` argument, which allows to redirect the output to a file
instead of stdout. 


More detailed configurations can be made through :mod:`logging.config`.
For example, the following both prints log messages to stdout and saves them to`ouput_filename.log`::

    import logging.config
    conf = {
        'version': 1
        'disable_existing_loggers': False,
        'formatters': {'brief': {'format': '%(levelname)-8s: %(message)s'},
                       'detailed': {'format': '%(asctime)s %(levelname)-8s: %(message)s'}},
        'handlers': {'to_file': {'class': 'logging.FileHandler',
                                 'filename': 'output_filename.log',
                                 'formatter': 'detailed',
                                 'level': 'INFO',
                                 'mode': 'w'},
                    'to_stdout': {'class': 'logging.StreamHandler',
                                  'formatter': 'brief',
                                  'level': 'INFO',
                                  'stream': 'ext://sys.stdout'}},
        'root': {'handlers': ['to_stdout', 'to_file'], 'level': 'DEBUG'},
    }
    logging.config.dictConfig(conf)

.. note ::

    Whether you use :func:`logging.config.fileConfig` or the :func:`logging.config.dictConfig`,
    make sure that you also set ``disable_existing_loggers=False``.
    Otherwise, it will not work as expected in the case where you ``import tenpy`` before setting up the logging.

To also capture warnings, you might also want to call :func:`logging.captureWarnings`.

In fact, the above is the default configuration used by :func:`tenpy.tools.misc.setup_logging`.
If you use a :class:`~tenpy.simulations.simulation.Simulation` class, it will automatically 
call :meth:`~tenpy.tools.misc.setup_logging` for you, saving the log to the same filename as the :cfg:option:`Simulation.output_filename` but with a ``.log`` ending.
Moreover, you can easily adjust the log levels with simple parameters, for example with the following configuration (using [yaml]_ notation):

.. code-block :: yaml

    logging_params:
        to_stdout:     # nothing in yaml -> None in python => no logging to stdout
        to_file: INFO
        log_levels:
            tenpy.tools.params : WARNING  # suppres INFO/DEBUG output for any logging of parameters 


How to write your own logging (and warning) code
------------------------------------------------
You can read the `official logging tutorial <https://docs.python.org/3/howto/logging.html>`_, 
but it's actually straight-forward, and just requires two steps.

1.  Import the necessary modules and create a logger at the top of your module::

        import warnings
        import logging
        logger = logging.getLogger(__name__)

    .. note ::

        Some classes that you might want to subclass ,e.g., all models, provide a :class:`~logging.Logger` as 
        ``self.logger`` class attribute. It's recommended to use that one instead from inside methods.

2.  Inside your funtions/methods/..., make calls like this::

        if is_likely_bad(options['parameter']):
            # this can be fixed by the user!
            warnings.warn("This is a bad parameter, you shouldn't do this!")
        if "old_parameter" in options:
            warnings.warn("Use `new_parameter` instead of `old_parameter`", FutureWarning, 2)

        logger.info("starting some lengthy calculation")
        n_steps = do_calculation()
        if something_bad_happened():
            # the user can't do anything about it
            logger.warn("Something bad happend")
        logger.info("calculation finished after %d steps", n_steps)

    You can use `printf-formatting <https://docs.python.org/3/library/stdtypes.html#printf-style-string-formatting>`_
    for the arguments of ``logger.debug(...), logger.info(...), logger.warn(...)``, as illustrated in the last line.
