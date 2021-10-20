Logging and terminal output
===========================

By default, calling (almost) any function in TeNPy will not print output, appart from error messages, tracebacks, and warnings.
Instead, we use Python's :mod:`logging` module to allow fine-grained redirecting of status messages etc.

Of course, when you get an error message, you should be concerned to find out what it is about and how to fix it. 
(If you believe it is a bug, `report <https://github.com/tenpy/tenpy/issues/new/choose>`_ it.)
Warnings can be reported either using ``warnings.warn(...)`` or with the logging mechanism ``logger.warning(...)``.
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
instead of stdout. Note that you should call `basicConfig` only once; subsequent calls have no effect.


More detailed configurations can be made through :mod:`logging.config`.
For example, the following both prints log messages to stdout and saves them to`ouput_filename.log`::

    import logging.config
    conf = {
        'version': 1
        'disable_existing_loggers': False,
        'formatters': {'custom': {'format': '%(levelname)-8s: %(message)s'}},
        'handlers': {'to_file': {'class': 'logging.FileHandler',
                                 'filename': 'output_filename.log',
                                 'formatter': 'custom',
                                 'level': 'INFO',
                                 'mode': 'a'},
                    'to_stdout': {'class': 'logging.StreamHandler',
                                  'formatter': 'custom',
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

    log_params:
        to_stdout:     # nothing in yaml -> None in python => no logging to stdout
        to_file: INFO
        logger_levels:
            tenpy.tools.params : WARNING  # suppres INFO/DEBUG output for any logging of parameters

Of course, you can also explicilty call the :func:`~tenpy.tools.misc.setup_logging` yourself, if you don't use the `Simulation` classes::

    tenpy.tools.misc.setup_logging({'to_stdout': None, 'to_file': 'INFO', 'filename': 'my_log.txt',
                                    'log_levels': {'tenpy.tools.params': 'WARNING'}})


How to write your own logging (and warning) code
------------------------------------------------
Of course, you can still use simple ``print(...)`` statements in your code, and they will just appear on your screen.
In fact, this is one of the benefits of logging: you can make sure that you *only* get the print statements you have put
yourself, and at the same time redirect the logging messages of tenpy to a file, if you want.

However, these ``print(...)`` statements are not re-directed to the log-files.
Therefore, if you write your own sub-classes like Models, I would recommended that you also use the loggers instead of
simple print statements.
You can read the `official logging tutorial <https://docs.python.org/3/howto/logging.html>`_ for details, 
but it's actually straight-forward, and just requires at most two steps.

1.  If necessary, import the necessary modules and create a logger at the top of your module::

        import warnings
        import logging
        logger = logging.getLogger(__name__)

    .. note ::

        Most TeNPy classes that you might want to subclass, like models, algorithm engines or simulations,
        provide a :class:`~logging.Logger` as ``self.logger`` class attribute. 
        In that case you can even **skip** this step and just use ``self.logger`` instead of ``logger`` in the snippets
        below.

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
            logger.warning("Something bad happend")
        logger.info("calculation finished after %d steps", n_steps)

    You can use `printf-formatting <https://docs.python.org/3/library/stdtypes.html#printf-style-string-formatting>`_
    for the arguments of ``logger.debug(...), logger.info(...), logger.warning(...)``, as illustrated in the last line.

In summary, instead of just ``print("do X")`` statements, use ``self.logger.info("do X")`` inside TeNPy classes, or just
``logger.info("do X")`` for the module-wide logger, which you can initialize right at the top of your file with the import
statements. If you have non-string arguments, add a formatter string, e.g. replace ``print(max(psi.chi))`` with
``logger.info("%d", max(psi.chi))``, or even better, ``logger.info("max(chi)=%d", max(psi.chi))``.
For genereic types, use ``"%s"`` or ``"%r"``, which converts the other arguments to strings with ``str(...)`` or ``repr(...)``, respectively.
