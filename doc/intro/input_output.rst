Saving to disk: input/output
============================

.. This file is maintained in the repository https://github.com/tenpy/hdf5_io.git
.. It can be read online at https://tenpy.readthedocs.io/en/latest/intro/input_output.html

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

Here, the ``with ... :`` structure ensures that the file gets closed after the pickle dump, and the ``'wb'`` indicates
the file opening mode "write binary".
Reading the data from disk is as easy as (``'rb'`` for reading binary)::

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


Using HDF5 with h5py
--------------------

While :mod:`pickle` is great for simple input/output of python objects, it also has disadvantages. The probably most
dramatic one is the limited portability: saving data on one PC and loading it on another one might fail!
Even exporting data from Python 2 to load them in Python 3 on the same machine can give quite some troubles.
Moreover, pickle requires to load the whole file at once, which might be unnecessary if you only need part of the data,
or even lead to memory problems if you have more data on disk than fits into RAM.

Hence, we support saving to `HDF5 <https://portal.hdfgroup.org/display/HDF5/HDF5>`_ files as an alternative.
The `h5py <https://docs.h5py.org>`_ package provides a dictionary-like interface for the file/group objects with
numpy-like data sets, and is quite easy to use. 
If you don't know about HDF5, read the :ref:`quickstart <h5py:quick>` of the `h5py`_ documentation (and this guide).

The implementation can be found in the :mod:`tenpy.tools.hdf5_io` module with the
:class:`~tenpy.tools.hdf5_io.Hdf5Saver` and :class:`~tenpy.tools.hdf5_io.Hdf5Loader` classes
and the wrapper functions :func:`~tenpy.tools.hdf5_io.save_to_hdf5`, :func:`~tenpy.tools.hdf5_io.load_from_hdf5`.

The usage is very similar to pickle::

    import h5py
    from tenpy.tools import hdf5_io

    data = {"psi": psi,  # e.g. an MPS
            "model": my_model,
            "parameters": {"L": 6, "g": 1.3}}

    with h5py.File("file.h5", 'w') as f:
        hdf5_io.save_to_hdf5(f, data)
    # ...
    with h5py.File("file.h5", 'r') as f:
        data = hdf5_io.load_from_hdf5(f)
        # or for partial reading:
        pars = hdf5_io.load_from_hdf5(f, "/parameters")

.. warning ::
    Like loading a pickle file, loading data from a manipulated HDF5 file with the functions
    described has the potential to cause arbitrary code execution.
    Only load data from trusted sources!


.. note ::
    The `hickle <https://github.com/telegraphic/hickle>`_ package imitates the pickle functionality 
    while saving the data to HDF5 files.
    However, since it aims to be close to pickle, it results in a more complicated data structure than we want here.

.. note ::
    To use the export/import features to HDF5, you need to install the `h5py`_ python package 
    (and hence some version of the HDF5 library).


Data format specification for saving to HDF5
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section motivates and defines the format how we save data of TeNPy-defined classes.
The goal is to have the :func:`~tenpy.tools.hdf5_io.save_to_hdf5` function for saving sufficiently simple enough python
objects (supported by the format) to disk in an HDF5 file, such that they can be reconstructed with the
:func:`~tenpy.tools.hdf5_io.load_from_hdf5` function, as outlined in the example code above.

Guidelines of the format:

0. Store enough data such that :func:`~tenpy.tools.hdf5_io.load_from_hdf5` can reconstruct a copy of the object
   (provided that the save did not fail with an error).
1. Objects of a type supported by the HDF5 datasets (with the `h5py`_ interface) should be directly stored as h5py :class:`Dataset`.
   Such objects are for example numpy arrays (of non-object `dtype`), scalars and strings.
2. Allow to save (nested) python lists, tuples and dictionaries with values (and keys) which can be saved.
3. Allow user-defined classes to implement a well-defined interface which allows to save instances of that class, hence extending what data can be saved.
   An instance of a class supporting the interface gets saved as an HDF5 :class:`Group`.
   Class attributes are stored as entries of the group, metadata like the type should be stored in HDF5 attributes, see :ref:`attributes <h5py:attributes>`.
4. Simple and intuitive, human-readable structure for the HDF5 paths.
   For example, saving a simple dictionary ``{'a': np.arange(10), 'b': 123.45}`` should result in an
   HDF5 file with just the two data sets ``/a`` and ``/b``. 
5. Allow loading only a subset of the data by specifying the `path` of the HDF5 group to be loaded.
   For the above example, specifying the path ``/b`` should result in loading the float ``123.45``, not the array.
6. Avoid unnecessary copies if the same python object is referenced by different names, e.g,
   for the data ``{'c': large_obj, 'd': large_obj}`` with to references to the same `large_obj`, save it only once and use HDF5 hard-links
   such that ``/c`` and ``/d`` are the same HDF5 dataset/group.
   Also avoid the copies during the loading, i.e., the loaded dictionary should again have two references to a single object `large_obj`.
   This is also necessary to allow saving and loading of objects with cyclic references.


The full format specification is given by the what the code in :mod:`~tenpy.tools.hdf5_io` does...
Since this is not trivial to understand, let me summarize it here:

- Following 1), simple scalars, strings and numpy arrays are saved as :class:`Dataset`. 
  Other objects are saved as a HDF5 :class:`Group`, with the actual data being saved as group members (as sub-groups and
  sub-datasets) or as attributes (for metadata or simple data).
- The type of the object is stored in the HDF5 attribute ``'type'``, which is one of the global ``REPR_*`` variables in
  :mod:`tenpy.tools.hdf5_io`. The type determines the format for saving/loading of builtin types (list, ...)
- Userdefined classes which should be possible to export/import need to implement the methods ``save_hdf5`` and ``from_hdf5``
  as specified in :class:`~tenpy.tools.hdf5_io.Hdf5Exportable`.
  When saving such a class, the attribute ``'type'`` is automatically set to ``'instance'``, and the class name and
  module are saved under the attributes ``'module'`` and ``'class'``. During loading, this information is used to 
  automatically import the module, get the class and call the classmethod ``from_hdf5`` for reconstruction.
  This can only work if the class definition already exists, i.e., you can only save class instances, not classes itself.
- For most (python) classes, simply subclassing :class:`~tenpy.tools.hdf5_io.Hdf5Exportable` should work to make the class exportable.
  The latter saves the contents of :attr:`~object.__dict__`, with the extra attribute ``'format'`` specifying 
  whether the dictionary is "simple" (see below.).
- The ``None`` object is saved as a group with the attribute ``'type'`` being ``'None'`` and no subgroups.
- For iterables (list, tuple and set), we simple enumerate the entries and save entries as group members under the
  names ``'0', '1', '2', ...``, and a maximum ``'len'`` attribute.
- The format for dictionaries depends on whether all keys are "simple", which we define as being strings which are valid
  path names in HDF5, see :func:`~tenpy.tools.hdf5_io.valid_hdf5_path_component`. 
  Following 4), the keys of a simple dictionary are directly used as names for group members, 
  and the values being whatever object the group member represents.
- Partial loading along 5) is possible by directly specifying the subgroup or the path to :func:`~tenpy.tools.hdf5_io.load_from_hdf5`.
- Guideline 6) is ensured as much as possible. However, there is a bug/exception: 
  tuples with cyclic references are not re-constructed correctly; the inner objects will be lists instead of tuples
  (but with the same object entries).

Finally, we have to mention that many TeNPy classes are :class:`~tenpy.tools.hdf5_io.Hdf5Exportable`.
In particular, the :class:`~tenpy.linalg.np_conserved.Array` supports this. 
To see what the exact format for those classes is, look at the `save_hdf5` and `from_hdf5` methods of those classes.

.. note ::
    There can be multiple possible output formats for the same object.
    The dictionary -- with the format for simple keys or general keys -- is such an example, 
    but userdefined classes can use the same technique in their `from_hdf5` method.
    The user might also explicitly choose a "lossy" output format (e.g. "flat" for np_conserved Arrays and LegCharges).

.. tip ::
    The above format specification is quite general and not bound to TeNPy. Feel free to use it in your own projects ;-)
    To separate the development, versions and issues of the format clearly from TeNPy, we maintain the code for it in a separate git repository,
    https://github.com/tenpy/hdf5_io
