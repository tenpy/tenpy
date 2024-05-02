Parameters and options
======================

(We use `parameter` and `option` synonymously. See also the section on parameters in :doc:`/intro/simulations`.

Standard simulations in TeNPy can be defined by just a set of options collected in a dictionary (possibly containing
other parameter dictionaries).
It can be convenient to represent these options in a [yaml]_ file, say ``parameters.yml``, which might look like this:

.. literalinclude:: /../examples/userguide/i_dmrg_parameters.yml

Note that the default values and even the allowed/used option names often depend on other parameters.
For example, the `model_class` parameter above given to a :class:`~tenpy.simulations.simulation.Simulation` selects a model class,
and different model classes might have completely different parameters.
This gives you freedom to easily define your own parameters when you implement a model, 
but it also makes it a little bit harder to keep track of allowed values.

In the TeNPy documentation, we use the ``Options`` sections of doc-strings to define parameters that are read out.
Each documented parameter is attributed to one set of parameters, called "config", and managed in a :class:`~tenpy.tools.params.Config` class at runtime.
The above example represents the config for a `Simulation`, with the `model_params` representing the config given as
`options` to the model for initialization.
Sometimes, there is also a structure of one `config` including the parameters from another one:
For example, the generic parameters for time evolution algorithms, :cfg:config:`TimeEvolutionAlgorithm` are included
into the :cfg:config:`TEBDEngine` config, similarly to the sub-classing used.

During runtime, the :class:`~tenpy.tools.params.Config` class logs the first use of any parameter (with DEBUG log-level, if
the default is used, and with INFO log-level, if it is non-default). Moreover, the default is saved into the parameter
dictionary. Hence, it will contain the *full set of all used parameters*, default and non-default, at the end of a
simulation, e.g., in the `sim_params` of the `results` returned by :meth:`~tenpy.simulations.Simulation.run`.

.. note ::

    You can find a **list of all the different configs** in the :ref:`cfg-config-index`, and a **list of all parameters** in :ref:`cfg-option-index`.

.. note ::

    If you add extra options to your configuration that TeNPy doesn't read out by the end of the simulation, it will (usually) issue a warning.
    Getting such a warnings is an indicator for a typo in your configuration, or an option being in the wrong config dictionary.


Python snippets in yaml files
-----------------------------
When defining the parameters in the yaml file, you might want to evaluate small formulas e.g., set a parameter to a certain fraction of $\pi$,
or expanding a long list ``[2**i for i in range(5, 10)]`` without explicitly writing all the entries.
For those cases, it can be convenient to have small python snippets inside the yaml file, which we allow by loading the
yaml files with :func:`tenpy.tools.params.load_yaml_with_py_eval`.

It defines a ``!py_eval`` yaml tag, which should be followed by a string of python code to be evaluated with python's ``eval()`` function.
A good method to pass the python code is to use a literal string in yaml, as shown in the simple examples below.

.. code :: yaml

    a: !py_eval |
        2**np.arange(6, 10)
    b: !py_eval |
        [10, 15] + list(range(20, 31, 2)) + [35, 40]
    c: !py_eval "2*np.pi * 0.3"
