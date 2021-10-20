Troubleshooting and FAQ
=======================

I updated to a new version and now I get an error/warning.
    Take a look at the section "Backwards incomptatible changes" in the :doc:`/releases` of the corresponding versions
    since when you updated.

Where did all the output go?
    Take a look at :doc:`/intro/logging`.

What are possible parameters for ...?
    See :doc:`/intro/options`.

How can I set the number of threads TeNPy is using?
    Most algorithms in TeNPy don't use any parallelization besides what the underlying BLAS provides,
    so that depends on how you installed TeNPy, numpy and scipy!
    Using for example an ``export OMP_NUM_THREADS=4`` should limit it to 4 threads under usual setups,
    but you might also want to ``export MKL_NUM_THREADS=4`` instead, if you are sure that you are using MKL.

Why is TeNPy not respecting MKL_NUM_THREADS?
    It might be that it is not using MKL.
    On linux, check whether you have installed a pip version of numpy or scipy in $HOME/.local/lib/python3.*
    Those packages do not use MKL - you would need to install numpy and scipy from conda.
    If you use the `conda-forge` channel as recommended in the installation, also make sure that you select
    the BLAS provided by MKL, see the note in :doc:`/install/conda`.

How can I double check the installed TeNPy version?
    You can call :func:`tenpy.show_config` to print details about the installed tenpy version.
    If you have multiple TeNPy/Python versions on your computer, 
    just calling ``print(tenpy)`` after an ``import tenpy`` will print the path of the used tenpy and can thus help
    you identify which of the TeNPy installations you use.


I get an error when ...
-----------------------
... I try to measure ``Sx_i Sx_j`` correlations in a state with `Sz` conseration.
    Right now this is not possible. See the basic facts in :doc:`/intro/npc`.


I get a warning about ...
-------------------------
... an unused parameter.
    Make sure that you don't have a typo and that it is in the right parameter set!
    Also, check the logging output whether the parameter was actually used.
    For further details, see :doc:`/intro/options`
