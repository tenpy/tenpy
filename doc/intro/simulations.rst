Simulations
===========

Simulations provide the highest-level interface in TeNPy.
They represent one simulation from start (initializing the various classes from given parameters) to end (saving the results to a file).
The idea is that they contain the full package of code that you run by a job on a computing cluster.
(You don't have to stick to that rule, of course.)
In fact, any simulation can be run from the command line, given only a parameter file as input, like this::

   python -m tenpy -c SimulationClassName parameters.yml
   # or alternatively, if tenpy is installed correctly:
   tenpy-run -c SimulationClassName parameters.yml

Of course, you should replace `SimulationClassName` with the class name of the simulation class you want to use, for
example :class:`~tenpy.simulations.ground_state_search.GroundStateSearch` or
:class:`~tenpy.simulations.time_evolution.RealTimeEvolution`. For more details, see :func:`tenpy.run_commandline`.

In some cases, this might not be enough, and you want to do some pre- or post-processing, or just do something a litte
bit differently during the simulation. In that case, you can also define your own simulation class (as subclass of one
the existing ones).
