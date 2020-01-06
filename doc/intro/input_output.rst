Saving to disk: input/output
============================

Using pickle
------------

A simple and pythonic way to store data of TeNPy arrays is to use :mod:`pickle` from the Python standard library.
Pickle allows to store (almost) arbitrary python objects,
and the :class:`~tenpy.linalg.np_conserved.Array` is no exception (and neither are other TeNPy classes).

Say that you have run DMRG to get a ground state `psi` as an :class:`~tenpy.networks.mps.MPS`.
With pickle, you can save it to disk as follows::

    import pickle
    with open('my_psi_file.pkl', 'wb') as f:
        pickle.dump(psi, f)

Here, the ``with ... :`` structure ensures that the file gets closed after the pickle dump.
Reading the data from disk is as easy as::

    with open('my_psi_file.pkl', 'rb') as f:
        psi = pickle.load(f)

.. note ::
    It is a good (scientific) practice to include meta-data to the file, like the parameters you used to generate that state.
    Instead of just the `psi`, you can simply store a dictionary containing `psi` and other data, e.g., 
    ``data = {'psi': psi, 'dmrg_params': dmrg_params, 'model_params': model_params}``.
    This can *save you a lot of pain*, when you come back looking at the files a few month later and forgot what you've done to generate them!

In some cases, compression can significantly reduce the space needed to save the data.
This can for example be done with :mod:`gzip` (as well in the Python standard library).
However, be warned that it might cause longer loading and saving times, i.e. it comes at the penalty of more CPU usage for the input/output.
In Python, this requires only small adjustments::

    import pickle
    import gzip

    # to save:
    with gzip.open('my_data_file.pkl', 'wb') as f:
        pickle.dump(data, f)
    # and to load:
    with gzip.open('my_data_file.pkl', 'rb') as f:
        data = pickle.load(data, f)


