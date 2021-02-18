Parameters and options
======================

(We use `parameter` and `option` synonymously.)

Standard simulations in TeNPy can be defined by just set of options collected in a dictionary (possibly containing
other parameter dictionaries).
It can be convenient to represent these options in a [yaml]_ file, which might look like this:

.. code-block :: yaml

    output_filename : params_output.h5
    overwrite_output : True
    model_class :  SpinChain
    model_params :
        L : 14
        bc_MPS : finite
        explicit_plus_hc : True

    initial_state_params:
        method : lat_product_state
        product_state : [[up], [down]]

    algorithm_params:
        trunc_params:
            chi_max: 120
            svd_min: 1e.-8
        chi_list: 
            0: 20
            3: 50
        algorithm_class: TwoSiteDMRG
        algorithm_params:
            max_sweeps: 20
            mixer: True
            trunc_params:
                chi_max: 100
                svd_min: 1.e-8
