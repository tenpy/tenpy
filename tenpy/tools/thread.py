"""Tools for thread-based parallelization."""

# Copyright 2021 TeNPy Developers, GNU GPLv3

import threading
import queue
import logging
logger = logging.getLogger(__name__)

__all__ = ["WorkerDied", "Worker"]


class WorkerDied(Exception):
    """Exception thrown if the main thread detects that the worker subthread died."""
    pass


class Worker:
    """Manager for a worker thread.

    Should be used as a context manager in a ``with`` statement, see the example below.

    Parameters
    ----------
    name : str
        Descriptive name for the worker thread.
    max_queue_size : int
        The `maxsize` for the :class:`queue.Queue`.

    Attributes
    ----------
    name : str
        The `name` parameter.
    tasks : :class:`queue.Queue`
        The queue with tasks to be done by the worker.
    exit : :class:`threading.Event`
        Set by the worker or main thread to indicate that the other thread should terminate.

    Example
    -------
    .. testsetup :: Worker

        from tenpy.tools.thread import *

    .. doctest :: Worker

        >>> def work_to_be_done(a, b):
        ...     # do something
        ...     return a + b
        >>> with Worker("example") as worker:
        ...    results = {}
        ...    worker.put_task(work_to_be_done, 2, 2, return_dict=results, return_key="2+2")
        ...    # go do something else, then add another task
        ...    worker.put_task(work_to_be_done, a=3, b=4, return_dict=results, return_key="3+4")
        ...    # "2+2" might be in results, but doesn't have to be yet
        ...    worker.join_tasks()  # block/wait until all the tasks have been done
        ...    assert "3+4" in results   # now we can be sure that we got all results
        >>> results
        {'2+2': 4, '3+4': 7}
    """
    def __init__(self, name="tenpy worker", max_queue_size=0, daemon=None):
        self.name = name
        self.tasks = queue.Queue(maxsize=max_queue_size)
        self.exit = threading.Event()  # set by both threads to tell each other to terminate
        self.worker_exception = None
        self.worker_thread = threading.Thread(target=self.run, name=name, daemon=daemon)
        self._entered = False

    def __enter__(self):
        if self._entered:
            raise ValueError("Can't reuse Worker multiple times!")
        self._entered = True
        self.worker_thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.worker_thread.is_alive():
            # no error occured in subthread, terminate it gracefully
            self.exit.set()
            self.worker_thread.join()

    def run(self):
        """Main function for worker thread."""
        logger.info("%s thread starting", self.name)
        try:
            while True:
                if self.exit.is_set():  # main thread wants to finish
                    logger.info("%s thread finishes", self.name)
                    return
                try:
                    task = self.tasks.get(timeout=1.)
                except queue.Empty:  # hit timeout
                    continue
                try:
                    fct, args, kwargs, return_dict, return_key = task
                    logger.debug("task for %s thread: %s, return=%s", self.name, fct.__qualname__,
                                 return_dict is not None)
                    res = fct(*args, **kwargs)
                    if return_dict is not None:
                        return_dict[return_key] = res
                finally:
                    self.tasks.task_done()
        except:
            self.exit.set()
            logger.exception("%s thread dies with following exception", self.name)
        finally:
            # drain the queue such that Queue.join() doesn't block
            while not self.tasks.empty():
                self.tasks.get()
                self.tasks.task_done()
        logger.info("%s thread terminates after exception", self.name)

    def _test_worker_alive(self):
        """Check wether worker thread is still alive."""
        if not self._entered:
            raise ValueError("Worker needs to be started in `with` statement.")
        if self.exit.is_set() or not self.worker_thread.is_alive():
            raise WorkerDied(self.name + ": either exception occured or close() was called.")

    def put_task(self, fct, *args, return_dict=None, return_key=None, **kwargs):
        """Add a task to be done by the worker.

        The worker will eventually do::

            res = fct(*args, **kwargs)
            if return_dict is not None:
                return_dict[return_key] = res

        It is unclear at which exact moment this happens, but after :meth:`join_tasks` was called,
        it is guaranteed to be done (or an exception was raised that the workder died).
        """
        task = (fct, args, kwargs, return_dict, return_key)
        while True:
            self._test_worker_alive()
            try:
                self.tasks.put(task, timeout=1.)
                return
            except queue.Full:  # hit timeout
                continue

    def join_tasks(self):
        """Block until all worker tasks are finished."""
        self._test_worker_alive()
        self.tasks.join()
        self._test_worker_alive()
