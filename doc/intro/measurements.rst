Measurements for Simulations
============================

Rationale
---------

When we run a simulation performing a time evolution, we are interested in measurements
after every (n-th) time step, but it would be too costly (in terms of disk space) to save the
full psi at each time step; we only have the ``|psi(t)>`` *during* the simulation, not afterwards.
Hence, we need to define what measurements we want to perform for a given simulation **before**
running it.

.. note ::
    For variational ground state searches, e.g. DMRG, the situation is better: we're not
    interested in how we got to the ground state, but only properties of the ground state itselft.
    In this case, we can first run DMRG, save the state, and then perform additional
    measurements and analysis *after* finishing the simulation, so it is not crucial to
    define all the measurements before the simulation.

The setup for simulations in TeNPy is as follows.

1) For each measurement that is to be done, we need a measurement function that evaluates
   whatever we want to measure, e.g., the expectation value or correlation function of some operators.
   If needed, you can define your own, custom functions.
2) For a given simulation, we specify the list of measurement functions in the simulation parameter
   :cfg:option:`Simulation.connect_measurements`.
3) When the simulation runs, it calls the :meth:`~tenpy.simulations.Simulation.make_measurements` method
   each time a set of measurements should be performed, e.g. on the initial state, during the time 
   evolution, and on the final state.
   This causes a call to each of the measurement functions specified in
   the :cfg:option:`Simulation.connect_measurements` parameter, passing the current state
   ``psi, model, simulation`` as arguments (possibly amongst other keyword arguments 
   also specified in :cfg:option:`Simulation.connect_measurements`).
   Moreover, it passes a dictionary ``results``, in which measurement results should be saved.
   At the end of `make_measurements`, the simulation class merges the obtained results 
   into the collection :attr:`~tenpy.simulations.Simulation.results` of all previous measurements
4) At the end of simulation, the `results` are saved and returned for further analysis (e.g. plotting).


Measurement functions
---------------------

In the simplest case, a measurement function is just a function, which can take the keyword arguments
``results, psi, model, simulation`` and saves the measurement results in the dictionary `results`.
The other arguments `psi` and `model` are the current MPS and model that can be used for measurements, 
and `simulation` gives access to the full simulation class, in case other addiontal data is needed.

Within TeNPy, we use the convention that measurement functions (taking these arguments and saving to `results` instead
of simply returning values) start with an ``m_`` in their name.
A few generic measurement functions are defined in :mod:`~tenpy.simulations.measurement`.

As a first, somewhat trivial example, let us look at the source code of
:func:`tenpy.simulations.measurement.m_entropy`::

    def m_entropy(results, psi, model, simulation, results_key='entropy'):
        results[results_key] = psi.entanglement_entropy()

As you can see, it's a simple wrapper around the MPS method :meth:`~tenpy.networks.mps.MPS.entanglement_entropy`.
Note that usually the `psi` and `model` arguments are the same as the simulation attributes 
``simulation.psi`` and ``simulation.model``, but they can be different in certain cases, e.g. when grouping sites.
In most cases, you should directly use the passed `psi` and `model`.

Of course, you can also do some actual calculations in the measurement functions.
A good example of this is the :func:`tenpy.simulations.measurement.m_onsite_expectation_value` - take a look at it's
source code. Another example could be the `m_pollmann_turner_inversion` measurement function defined in the
:doc:`/examples/model_custom` example from the :doc:`/intro/simulations` guide.


The connect_measurements parameter
----------------------------------

The :cfg:option:`Simulation.connect_measurements` parameter is a list with one entry for each measurment function to be
used. Each function is specified by a tuple ``module, func_name, extra_kwargs, priority``.
Here, `module` and `func` specfiy the module and name of the function, `extra_kwargs` are (optional) additional keyword
arguments to be given to the function, and `priority` allows to control the order in which the measurement functions get
called. The latter is usefull if you want to "post-process" results of another measurement function.

For example, say you want to measure local expectation values of both `Sz` and `Sx` with
:func:`tenpy.simulations.measurment.m_onsite_expectation_value`, then you could use

.. code :: yaml

    connect_measurement
        - - tenpy.simulations.measurement
          - m_onsite_expectation_value
          - opname: Sx
        - - tenpy.simulations.measurement
          - m_onsite_expectation_value
          - opname: Sz

These measurement functions have default `results_key` under which they save values in the `results`, so you can then
read out ``results['<Sx>']`` and ``results['<Sz>']`` in the simulation results.
If you want other keys, you can explicitly specify them with the `results_key` argument of the function, e.g.,

.. code :: yaml

    connect_measurement:
        - - tenpy.simulations.measurement
          - m_onsite_expectation_value
          - opname: Sx
            results_key: X_i     # save as results['X_i']
        - - tenpy.simulations.measurement
          - m_onsite_expectation_value
          - opname: Sz
            results_key: Z_i     # save as results['Z_i']


Some measurements are actually that common that they get added by default to the simulations (unless you explicitly
disable them with :cfg:option:`Simulation.use_default_measurements`); for example the :func:`tenpy.simulations.measurement.m_entropy`
is measured for any simulation, as it appears in :attr:`~tenpy.simulations.simulation.Simulation.default_measurements`.

Often, what you want to measure is just calling a method of the state `psi`, so there is a special syntax in the
`connect_measurement` parameter:
if you **specify the first entry to be** ``psi_method``, ``model_method`` or ``simulation_method``, you can call a method of the
corresponding classes. 
As for global measurement functions, we pass the corresponding ``results, psi, model, simulation`` keyword arguments,
e.g. `psi_method` measurement functions need to accept ``results, model, simulation`` as arguments, and
`simulation_method` measurement functions should accept ``results, psi, model``.

This is already very usefull to call measurement functions defined inside (custom) models or simulation classes, 
yet methods of `psi` don't follow the measurement function call structure, but simply return values.
For those cases, you can use another special syntax, namely to **simply add `wrap` before the function name**.
In this case, we don't pass ``results, psi, model, simulation``, but simply save the return values of the function
in the results, under the `results_key` that gets passed as extra keyword argument,
see :func:`~tenpy.simulations.measurment.measurement_wrapper`.
The `results_key` defaults to the function name.

To make this clearer, let's extend the example above with more measurements:

.. code :: yaml

    connect_measurement:
        - - tenpy.simulations.measurement
          - m_onsite_expectation_value
          - opname: Sx
        - - tenpy.simulations.measurement
          - m_onsite_expectation_value
          - opname: Sz
        - - psi_method
          - wrap correlation_function   # call psi.correlation_function()
          - results_key: '<Sz Sz>'      # save returned value as results["<Sz Sz>"]
            ops1: Sz                    # other (necessary) arguments to psi.correlation_function
            ops2: Sz
        - - simulation_method
          - wrap walltime               # "measure" wall clock time it took to run so far
        - - tenpy.tools.process
          - wrap memory_usage           # "measure" the current RAM usage in MB


.. note ::

   The `*_method` and `wrap` syntax are (currently) special to the :cfg:option:`Simulation.connect_measurements`
   parameter, and do not apply to e.g. :cfg:option:`Simulation.connect_algorithm_checkpoint`, which uses an analogous
   setup to allow calling functions at each algorithm checkpoint.
