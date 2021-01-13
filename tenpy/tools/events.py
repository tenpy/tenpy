"""Event handler.

The :class:`EventHandler` is basically just holds a list of functions
which can get called once a certain "event" happens.
Examples are given in the class doc-string.
"""
# Copyright 2020-2021 TeNPy Developers, GNU GPLv3

from collections import namedtuple
import warnings
import functools

from .hdf5_io import find_global

__all__ = ['Listener', 'EventHandler']

Listener = namedtuple('Listener', "listener_id, callback, priority")


class EventHandler:
    """Handler for an event represented by an instance of this class.

    All in all, events provide a flexible extension mechanism for classes to define "checkpoints"
    in the code where the user of a class might want to run something else,
    for example doing some measurements or saving intermediate results.

    Parameters
    ----------
    arg_descr : str
        An informative description how the callback function is called.
        An empty string indicates no arguments.

    Attributes
    ----------
    arg_descr : str
        An informative description how the callback function is called.
    listeners : list of (int, function, int)
        Entries are tuples ``(listener_id, callback, priority)``.

    Examples
    --------
    Instances of this class typically get defined during class initialization and define an event.
    The event "happens" each time :meth:`emit` or :meth:`emit_until_result` is called,
    typically inside a method of the class defining the event. Example:

    .. testsetup :: EventHandler

        from tenpy.tools.events import EventHandler

    .. doctest :: EventHandler

        >>> class MyAlgorithm:
        ...     def __init__(self):
        ...         self.checkpoint = EventHandler("algorithm, iteration")
        ...         self.data = 0
        ...     def run(self):
        ...         for i in range(4):
        ...             self.data += i # do some complicated stuff
        ...             self.checkpoint.emit(self, i)

    Other code with access to the event can then connect a `listener` to the event, i.e.,
    give a function to the event that should be called each time the event is :meth:`emit`-ed.

    .. doctest :: EventHandler

        >>> my_alg = MyAlgorithm()
        >>> def my_listener(algorithm, iteration):
        ...     print("my_listener called: iteration", iteration, "with data", algorithm.data)
        >>> my_alg.checkpoint.connect(my_listener)  # doctest: +ELLIPSIS
        <function my_listener at 0x...>
        >>> my_alg.run()
        my_listener called: iteration 0 with data 0
        my_listener called: iteration 1 with data 1
        my_listener called: iteration 2 with data 3
        my_listener called: iteration 3 with data 6

    As you can see, the function `my_listener` has been called during the ``MyAlgorithm.run()``
    and had full access to the current status of the algorithm class.
    This is convenient to e.g. perform measurements of the state so far, print a status message
    of the progress or save intermediate results.

    If the EventHandler is already initialized when you define the function, you can also
    use :meth:`connect` as a function property like this:

    .. doctest :: EventHandler

        >>> @my_alg.checkpoint.connect
        ... def another_one(algorithm, iteration):
        ...     print("another_one called: iteration", iteration)
        >>> @my_alg.checkpoint.connect(priority=5)
        ... def high_priority(algorithm, iteration):
        ...     print("high_priority call: iteration", iteration)
        >>> my_alg.run()
        high_priority call: iteration 0
        my_listener called: iteration 0 with data 6
        another_one called: iteration 0
        high_priority call: iteration 1
        my_listener called: iteration 1 with data 7
        another_one called: iteration 1
        high_priority call: iteration 2
        my_listener called: iteration 2 with data 9
        another_one called: iteration 2
        high_priority call: iteration 3
        my_listener called: iteration 3 with data 12
        another_one called: iteration 3

    """
    def __init__(self, arg_descr=None):
        self.arg_descr = arg_descr
        self.listeners = []
        self._id_counter = 0

    def copy(self):
        """Make a (shallow) copy."""
        cp = EventHandler(self.arg_descr)
        cp.listeners = self.listeners[:]
        cp._id_counter = self._id_counter
        return cp

    @property
    def id_of_last_connected(self):
        if self._id_counter == 0:
            raise ValueError("connect() hasn't been called yet!'")
        return self._id_counter - 1

    def connect(self, callback=None, priority=0):
        """Register a `callback` function as a listener to the event.

        You can either call this function directly or use it as a function decorator,
        see the example in :class:`EventHandler`.

        If you ever plan to :meth:`disconnect` again, you can read it out with
        :attr:`id_of_last_connected` right after connecting, i.e., right after calling this method.

        Parameters
        ----------
        callback : callable
            A function to be called during each :meth:`emit` of the event.
        priority : int
            Higher priority indicates that the callback function should be called before other
            possibly registered callback functions.

        Returns
        -------
        callback : callable
            The callback function exactly as given.
        """
        if callback is None:
            # hande the case that we got called as property like this::
            # @ev_handler.connect(priority=2)
            # def my_function():
            #     pass
            def property(callback):
                self.connect(callback, priority)
                return callback

            return property
        listener_id = self._id_counter
        self._id_counter += 1
        self.listeners.append(Listener(listener_id, callback, priority))
        return callback

    def connect_by_name(self, module_name, func_name, kwargs=None, priority=0):
        """Connect to a function given by the name in a module, optionally inserting arguments.

        Parameters
        ----------
        module_name : str
            The name of the module containing the function to be used. Gets imported.
        func_name : str
            The (qualified) name of the function inside the module.
        kwargs : dict
            Optional extra keyword-arguments to be given to the function.
        priority : int
            Higher priority indicates that the callback function should be called before other
            possibly registered callback functions.
        """
        func = find_global(module_name, func_name)
        if kwargs is not None:
            func = functools.partial(func, **kwargs)
        self.connect(func, priority)

    def disconnect(self, listener_id):
        """De-register a listener.

        Parameters
        ----------
        listener_id : int
            The id of the listener, as given by :attr:`id_of_last_connected`
            right after calling :meth:`connect`.
        """
        for i, listener in enumerate(self.listeners):
            if listener.listener_id == 0:
                del self.listeners[i]
                return
        warnings.warn("No listener with listener_id {id_:d} found".format(id_=listener_id))

    def emit(self, *args, **kwargs):
        """Call the `callback` functions of all listeners.

        Returns
        -------
        results : list
            List of results returned by the individual callback functions.
        """
        self._prepare_emit()
        results = []
        for _, callback, _ in self.listeners:
            res = callback(*args, **kwargs)
            results.append(res)
        return results

    def emit_until_result(self, *args, **kwargs):
        """Call the listeners `callback` until one returns not `None`."""
        self._prepare_emit()
        for _, callback, _ in self.listeners:
            res = callback(*args, **kwargs)
            if res is not None:
                return res
        return None

    def _prepare_emit(self):
        # TODO: logging?
        # sort listeners: highest priority first
        self.listeners = sorted(self.listeners, key=lambda listener: -listener.priority)
