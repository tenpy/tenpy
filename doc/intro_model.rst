Introduction to models
======================

A `model` stands for some physical (quantum) system to be described.
Tensor networks are useful if the model is specified as a Hamiltonian written in terms of second quantization, written in terms of local operators.

Usually, some kind of lattice structure is defined implicitly in the Hamiltonian, e.g. by the sum over next neighbours
:math:`\sum_{<i, j>}`. Of course, DMRG and related MPS methods are most successfull for 1D chains; yet they can be
generalized to cylinders with small circumference.

The basic building block of the lattice is a :class:`~tenpy.networks.site.Site`. The site collects both the local
Hilbert space and local operators. 
The most common sites (for spins, spin-less or spin-full fermions, or bosons) are predefined in :mod:`tenpy.networks.site`, 
but if necessary you can easily extend them (e.g. by adding further local operators) or completely write your own sites.

A :class:`~tenpy.models.lattice.Lattice` specifies the geometry how the sites are arranged. In general, it consists of a unit
cell which is repeated periodically. Again, there are multiple pre-defined lattices, but you can also define your own
lattice if needed (More details on that can be found in the doc-string of :class:`~tenpy.models.lattice.Lattice`).
In the simplest case -- namely a :class:`~tenpy.models.lattice.Chain` -- a unit-cell with just a single site is repeated periodicaly in one dimension.

A :class:`~tenpy.models.model.CouplingModel` is a general, abstract way to specify a Hamiltonian of two-site couplings.
Given a lattice, it allows to :meth:`~tenpy.models.model.CouplingModel.add_coupling` between the repeated unit cells.
In that way, we obtain an abstract way of defining an Hamiltonian by its coupling terms. 
Given these couplings, it is possible to automatically build an MPO with :meth:`~tenpy.models.model.CouplingModel.calc_H_MPO`, which in turn can be used to initialize
the :class:`~tenpy.models.model.MPOModel`. 
If the model consists only of nearest-neighbor terms (in terms of a 1D chain used for the MPS), 
you can also build the bond terms for TEBD with :meth:`~tenpy.models.model.CouplingModel.calc_H_bond` and initialize a 
:class:`~tenpy.models.model.NearestNeighborModel`.

The :class:`~tenpy.models.model.MPOModel` and :class:`~tenpy.models.model.NearestNeighborModel` store the Hamiltonian
exactly in the form needed for the various algorithms like DMRG (:mod:`~tenpy.algorithms.dmrg`) and TEBD (:mod:`~tenpy.algorithms.tebd`).
Of course, an MPO is all you need to initialize a :class:`~tenpy.models.model.MPOModel` to be used for DMRG; you don't have to use
the :class:`~tenpy.models.model.CouplingModel`. 
For example exponentially decaying long-range interactions are not supported by the coupling model but straight-forward to include to an MPO.

We suggest writing the model to take a single parameter dicitionary for the initialization, which is to be read out inside the class with
:func:`~tenpy.tools.params.get_parameter`. Read the doc-string of this function for more details why this is a good idea.

If the model you're interested in contains Fermions, read the user-guide to :doc:`intro_JordanWigner`.

When you write a model, don't forget to write some tests. The file ``tests/test_model.py`` provides basic functionality for that.


The :class:`tenpy.models.xxz_chain.XXZChain` demonstrates nicely how a new model can be defined using the :class:`~tenpy.models.model.CouplingModel`.
The complete source code for this model is included in the following and should be straight-forward to understand and
generalize.

.. literalinclude:: ../tenpy/models/xxz_chain.py
