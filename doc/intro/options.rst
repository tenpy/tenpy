Parameters and options
======================

(We use `parameter` and `option` synonymously. See also the section on parameters in :doc:`/intro/simulations`.

Standard simulations in TeNPy can be defined by just set of options collected in a dictionary (possibly containing
other parameter dictionaries).
It can be convenient to represent these options in a [yaml]_ file, say ``parameters.yml``, which might look like this:

.. literalinclude:: /../examples/userguide/i_dmrg_parameters.yml

Note that the default values and even the allowed/used option names often depend on other parameters.
For example, the `model_class` parameter above given to a :class:`~tenpy.simulations.Simulation` selects a model class,
and diffent model classes might have completely different parameters.
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
simulation, e.g., in the `sim_params` of the `results` returned by :meth:`tenpy.simulations.Simulation.run`.

.. note ::

    You can find a **list of all the different configs** in the :ref:`cfg-config-index`, and a **list of all parameters** in :ref:`cfg-option-index`.

.. note ::

    If you add extra options to your configuration that TeNPy doesn't read out by the end of the simulation, it will (usually) issue a warning.
    Getting such a warnings is an indicator for a typo in your configuration, or an option being in the wrong config dictionary.
