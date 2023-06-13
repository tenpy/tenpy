Simulations
===========

What is a simulation?
---------------------

Simulations provide the highest-level interface in TeNPy.
They represent one simulation from start (initializing the various classes from given parameters) to end (saving the results to a file).
The idea is that they contain the full package of code that you run by a job on a computing cluster.
(You don't have to stick to that rule, of course.)
In fact, any simulation can be run from the command line, given only a parameter file as input, like this::

   python -m tenpy parameters.yml
   # or alternatively, if tenpy is installed correctly:
   tenpy-run parameters.yml

Of course, you need to specify somewhere what type of simulation you want to run. Often, one of the predefined ones like
the :class:`~tenpy.simulations.ground_state_search.GroundStateSearch` for running DMRG or
:class:`~tenpy.simulations.time_evolution.RealTimeEvolution` for running e.g. TEBD or TDVP will suffice.
The :class:`~tenpy.simulations.simulation.Simulation` class can be specified with the `simulation_class` option in the yaml file, or directly as a command line
argument, e.g. ``tenpy-run -C GroundStateSearch parameters.yml``.
Note that command line arguments possibly override entries in the yaml files.
For more details, see :func:`tenpy.console_main` for the command-line interface.

Of course, you can also directly run the simulation from inside python, the command line call is essentially just a wrapper around the :func:`tenpy.run_simulation` python interface::

    import tenpy
    import yaml

    simulation_params = yaml.load("parameters.yml")
    # instead of using yaml, you can also define a usual python dictionary
    tenpy.run_simulation(**simulation_params)


An minimal example to run finite DMRG for a Spin-1/2 Heisenberg :class:`~tenpy.models.spins.SpinChain` could be given by

.. literalinclude:: /../examples/userguide/i_dmrg_parameters.yml


Parallelization: controlling the number of threads
--------------------------------------------------
Almost all of the TeNPy code is "only" using thread-based paralellization provided by the underlying LAPACK/BLAS package linked to by Numpy/Scipy, and/or TeNPy's Cython code when you compile it.
(A notable exception is the :class:`~tenpy.tools.cache.ThreadedStorage` for caching.)
In practice, you can control the number of threads in the same way as if you use just plain numpy - by default, this uses all the CPU cores on a given machine. 

If you run things on a cluster, it is often required to only use a fixed number of cores. Assuming a standard Linux cluster, the easiest way to control the used number of threads is usually the OMP_NUM_THREADS environment variable, which you can set in your cluster submission script:

.. code :: bash

    export OMP_NUM_THREADS=4
    python -m tenpy parameters.yml

If you linked against MKL, you can use ``export MKL_NUM_THREADS=4`` instead. In some cases, it might also be necessary to additionaly ``export MKL_DYNAMIC=FALSE``.
Universities usually have some kind of local cluster documentation with examples - try to follow those, and doulbe check
that you only use the cores you request.


Customizing parameters
----------------------

The most straight-forward way to customize a simulation is to tweak and adjust the parameters to your needs.
As you can see in the above example, the parameters are organized in a hierarchical structure, following roughly the
same level structure as discussed in the :doc:`/intro/overview`.

The allowed options on the top level are documented in the corresponding simulation class, e.g. the
:class:`~tenpy.simulations.ground_state_search.GroundStateSearch`.

The allowed entries in the `model_params` section depend on the `model_class`:
Clearly, the :class:`~tenpy.models.spins.SpinChain` in the example above requires a different set of specified 
coupling parameters than, e.g., the :class:`~tenpy.models.hubbard.FermiHubbardModel`.
The base model classes like the :class:`~tenpy.models.models.model.CouplingMPOModel` have a common set of parameters
usually read out, but custom model implementations can override this and/or add additional parameters. 
The list of allowed parameters can hence be found in the documentation of the most specialized class that you use, e.g.,
the :class:`~tenpy.models.tf_ising.TFIChain`` above.

Similarly, allowed values in the `algorithm_params` section depend on the used `algorithm_class`.


To get the full set of used options, it can be convenient to simply run the algorithm 
(for debugging parameters to allow a very quick run) and look at the ``results['simulation_parameters']``
returned by the simulation (or saved to file):

.. code-block :: yaml

    import tenpy
    from pprint import pprint
    import yaml

    with open('parameters.yml', 'r') as f:
        simulation_parameters = yaml.safe_load(f)
    results = tenpy.run_simulation(simulation_parameters)
    pprint(results['simulation_parameters'])

.. note ::

    You can find a **list of all the different configs** in the :ref:`cfg-config-index`, and a **list of all parameters** in :ref:`cfg-option-index`.

.. note ::

    If you add extra options to your configuration that TeNPy doesn't read out by the end of the simulation, it will (usually) issue a warning.
    Getting such a warnings is often an indicator for a typo in your configuration, or an option being in the wrong section.


Adjusting the output
--------------------
If specified, output files are saved in a given :cfg:option:`Simulation.directory`.
As shown in the parameter example above, you can simply give an :cfg:option:`Simulation.output_filename` parameter.
Alternatively, one can specify the :cfg:option:`Simulation.output_filename_params` to make the filename depend on other
simulation paramters (specified as keys of the `parts`), e.g:

.. code-block :: yaml

    directory: results
    output_filename_params:
        prefix: dmrg
        parts:
            algorithm_params.trunc_params.chi_max: 'chi_{0:04d}'
            model_params.L: 'L_{0:d}'
        suffix: .h5

With the above example parameters, this would yield the output filename ``results/dmrg_chi_0100_L_16.h5``; further examples in the
documentation of :func:`~tenpy.simulations.simulation.output_filename_from_dict`.

Note that TeNPy will not overwrite output unless you explicitly set :cfg:option:`Simulation.overwrite_output` to ``True``.
Rather, it will modify the filename with extra numbers, e.g., ``file.h5, file_1.h5, file_2.h5, ...``, or it will raise a
specific :class:`~tenpy.simulations.simulation.Skip` exception if :cfg:option:`Simulation.skip_if_output_exists` is set.
Further, temporary ``.backup.h5`` files are used while saving to avoid loosing previous results in case of a crash during the save.

The option :cfg:option:`Simulation.save_psi` allows to enable (default) or disable saving the full tensor network at the end of the simulation - note this drastically influences the size of the output file!
For long-running simulations you can decide to save intermediate checkpoints with the option :cfg:option:`Simulation.save_every_x_seconds`; see the resume_details_ section below.

Log files by default use the same filename as the output but with the extension ``.log``, see :doc:`/intro/logging` for more details.
In practice, it is useful only print warnings and errors to stdout to allow a simple check for errors, while the ``.log`` files can then be used to follow the details and progress of the simulation:

.. code-block :: yaml

    log_params:
        to_file: INFO
        to_stdout: WARN
        # format: "{levelname:.4s} {asctime} {message}"

.. note ::

    Always check errors and warnings! In most simulations, there shouldn't be any warnings left.


Analyzing the results post simulation: output structure
-------------------------------------------------------
A simulation usually generates an output file that can be loaded with the :func:`~tenpy.tools.hdf5_io.load` function.
It is usually either in the pickle or HDF5 format, see :doc:`/intro/input_output` for more details.

The ability to keep code snippets and plots together in [jupyter]_ notebooks makes them a very convenient environment for analyzing results.
There are a bunch of jupyter notebooks in the :doc:`/examples` that you can look at for inspiration.

The `results` returned by :func:`~tenpy.run_simulation` are a (nested) dictionary.
The general structure is listed in :attr:`~tenpy.simulations.simulation.Simulation.results`.
Possible entries depend on the simulation class run, and some options like `save_psi` or specified measurements.

Let us consider our initial DMRG example.
The :class:`~tenpy.simulations.ground_state_search.GroundStateSearch` performs two measurements: one on the inital
state (unless disabled with :cfg:option:`measure_initial` and one on the final state.
Further, MPS-based simulations by default measure the entanglement entropies for cutting at the various MPS bonds, 
such that we can read out the final half-chain entanglement entropy like this::

    >>> import tenpy
    >>> results = tenpy.tools.hdf5_io.load('results/dmrg_chi_0100_L_32.h5')
    >>> L = results['simulation_parameters']['model_params']['L']
    >>> L
    32
    >>> print(results['measurements']['entropy'].shape)
    (2, 31)
    >>> print(results['measurements']['entropy'][-1, (L-1)//2])

Here, the shape of the entropy array is ``(2, 31)`` since 2 is the number of measurements 
(one on the initial state, one on the final ground state), and 31=L-1 the number of bonds.
Note that you can easily read out the simulation parameters, even default ones that are only implicitly defined
somewhere in the code!


Adding more measurements
------------------------
Most simulation classes have only a few :attr:`~tenpy.simulations.Simulation.default_measurements`, but you can easily
add more with the :cfg:option:`Simulation.connect_measurements` parameters. Each measurement is simply a function that is
called whenever the simulation wants to measure, e.g. with the initial state, at the end of the simulation, and for time
evolutions also during the evolution. The default measurement functions are defined in
the module :mod:`tenpy.simulations.measurement`; :func:`~tenpy.simulations.measurement.measurement_index` documents what
arguments a measurement function should have. 
In the simplest case, you just specify the module and function name, but you can also add more arguments, as the
following example shows.

.. code-block :: yaml

    connect_measurements:
      - - tenpy.simulations.measurement
        - m_onsite_expectation_value
        - opname: Sz
      - - psi_method
        - wrap correlation_function
        - results_key: '<Sp_i Sm_j>'
          ops1: Sp
          ops2: Sm

Note the indentation and minus signs here: this yaml syntax is equivalent to the following python structure:

.. code-block :: python

    {'connect_measurements': [['tenpy.simulations.measurement',
                               'm_onsite_expectation_value',
                               {'opname': 'Sz'}],
                              ['psi_method',
                               'wrap correlation_function',
                               {'results_key': '<Sp_i Sm_j>',
                                'ops1': 'Sp',
                                'ops2': 'Sm'}]]}

The measurement functions add the values under the specified `key` to the `results` returned and saved by the
simulation, e.g. for the above measurements you can now read out ``results['measurements']['<Sz>']`` (default key) and ``results['measurements']['<Sp_i Sm_j>']``.

For more details, see the extra guide :doc:`/intro/measurements`.

A full example with custom python code
--------------------------------------

While there are plenty of predefined models and algorithms, there is a good chance that you need to tweak and adjust
them further by writing your own python code. Examples could be custom models and/or lattices, measurment functions, or
even adjustments to any other class (tensor networks, algorithms, simulations...).

As a concrete example, let's try to reproduce some results of :cite:`pollmann2012`, namely the :math:`\mathcal{O}_I` defined in eq. (15) of that paper.
A new model class is not strictly necessary, one can also select appropriate parameters for the :class:`~tenpy.models.spins.SpinChain`, but we include it here for completeness.
Details on how to define a custom model class can be found in :doc:`/intro/model`.

.. literalinclude::  /../examples/model_custom.py

The corresponding `simulation_custom.yml` parameter file, collecting the snippets above, could then look like this:

.. literalinclude::  /../examples/simulation_custom.yml

Note that we explicitly specified the module `model_custom` for the additional measurement; you need to adjust that if
you rename the `model_custom.py` file.
You can then run this simulation, say for three different `D` values specified directly on the command line::

    tenpy-run -i model_custom simulation_custom.yml -o model_params.D 0.
    tenpy-run -i model_custom simulation_custom.yml -o model_params.D 1.5
    tenpy-run -i model_custom simulation_custom.yml -o model_params.D -1.0


.. note ::

    If you use the setup from the [TeNPyProjectTemplate]_ repository, the ``cluster_jobs.py`` helps to manage submiting
    jobs with similar parameters to a computing cluster; 
    it includes this very example as a starting point for customization.


.. _resume_details:

Checkpoints for resuming a simulation
-------------------------------------
As mentioned above, you can save intermediate results with the option :cfg:option:`save_every_x_seconds`.
Moreover, you need to have :cfg:option:`Simulation.save_psi` and :cfg:option:`Simulation.save_resume_data` enabled::

    save_every_x_seconds: 1800
    save_psi: True
    save_resume_data: True

If this is the case, the simulation will save the current status at certain "checkpoints" defined by the algorithm, 
e.g., in DMRG at the end of a sweep.
The checkpoints are saved to the same filename as the desired final output file, and get overwritten by each following save at a checkpoint.
You can check ``results['finished']`` in the output file to see whether it finished.

You can then resume the simulation using the function :func:`tenpy.resume_from_checkpoint`.

Note that you can also adjust parameters for the resume. 
For example, if you find that a DMRG result (even a finished one) is not yet fully converged in bond dimension, you can "resume" the simulation
with a larger bond dimension and a new output filename.
For DMRG, this is roughly equivalent to starting a new simulation with the initial state loaded
:meth:`~tenpy.networks.mps.InitialStateBuilder.from_file`; but it can reuse more than just the state, e.g., environments and already performed measurements, or the `evolved_time` of a time evolution.


Sequential simulations
----------------------
Instead of waiting for one simulation to finish and "resuming" another one with slightly different parameters, you can
also directly specify a set of "sequential" simulations where the output/results of one simulation are reused for the
next one. This can be particularly useful to "adiabatically" follow the ground state when tuning model parameters, in
particular for flux pump experiments, or to get a stable scaling with bond dimension.

To achieve this, you need to call :func:`~tenpy.run_seq_simulations` instead of just :func:`~tenpy.run_simulation`, and
specify the :cfg:config:`sequential` parameters for the simulation (at the top level of the yaml files), in particular
the `recursive_keys` for the parameters to be changed. The values for those parameters can be specified as 
:cfg:option:`sequential.value_lists`, or as lists in the original localtion of the yaml file.

.. code-block :: yaml

    sequential:
        recursive_keys:
            - algorithm_params.trunc_params.chi_max

    algorithm_params:
        trunc_params:
            chi_max: [128, 256, 512]
