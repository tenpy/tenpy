"""Event handler.

.. todo ::
    Support HDF5 export -> allow to export functions!
"""
# Copyright 2020 TeNPy Developers, GNU GPLv3

from collections import defaultdict, namedtuple

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
    listeners : list
        Entries are tuples ``(listener_id, callback, priority)``.

    Examples
    --------
    Instances of this class typically get defined during class initialization and define an event.
    The event "happens" each time  :meth:`emit` or :meth:`emit_until_result` is called,
    typically inside a method of the class defining the event. Example:

    >>> class MyAlgorithm:
    ...     def __init__(self):
    ...         self.checkpoint = EventHandler("algorithm, iteration")
    ...         self.data = 0
    ...     def run(self):
    ...         for i in range(3):
    ...              self.data += i # do some complicated stuff
    ...              self.checkpoint.emit(self, i)

    Other code with access to the event can then connect a `listener` to the event, i.e.,
    give a function to the event that should be called each time the event is :meth:`emit`-ed.

    >>> my_alg = MyAlgorithm()
    >>> def first_listener(algorithm, iteration):
    ...     print("iteration={0:d}, data={1:d}".format(iteration, algorithm.data))
    >>> my_alg.checkpoint.connect(first_listener)
    0
    >>> my_alg.run()
    iteration=0, data=0
    iteration=1, data=1
    iteration=2, data=3

    """
    def __init__(self, arg_descr=None):
        self.arg_descr = arg_descr
        self.listeners = []
        self._id_counter = 0

    def connect(self, callback, priority=0):
        """Register a `callback` function as a listener to the event.

        Parameters
        ----------
        callback : callable
            A function to be called during each :meth:`emit` of the event.
        priority : int
            Higher priority indicates that the callback function should be called before other
            possibly registered callback functions.

        Returns
        -------
        listener_id : int
            Id of the listener
        """
        listener_id = self._id_counter
        self._id_counter += 1
        self.listeners.append(Listener(listener_id, callback, priority))
        return listener_id

    def disconnect(self, listener_id):
        """De-register a listener.

        Parameters
        ----------
        listener_id : int
            The id of the listener returned by :meth:`connect`.
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
