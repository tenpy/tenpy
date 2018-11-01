Introduction to models
======================

What is a model?
----------------

Abstractly, a **model** stands for some physical (quantum) system to be described.
For tensor networks algorithms, the model is usually specified as a Hamiltonian written in terms of second quantization.
For example, let us consider a spin-1/2 Heisenberg model described by the Hamiltonian

.. math ::

    H = J \sum_i S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + S^z_i S^z_{i+1}

Note that a few things are defined more or less implicitly.

- The local Hilbert space: it consists of Spin-1/2 degrees of freedom with the usual spin-1/2 operators :math:`S^x, S^y, S^z`.
- The geometric (lattice) strucuture: above, we spoke of a 1D "chain".
- The boundary conditions: do we have open or periodic boundary conditions?
  The "chain" suggests open boundaries, which are in most cases preferable for MPS-based methods.
- The range of `i`: How many sites do we consider (for a 2D system: in each direction)?

Obviously, these things need to be specified in TeNPy in one way or another, if we want to define a model.

Ultimately, our goal is to run some algorithm. Each algorithm requires the model and Hamiltonian to be specified in a particular form.
We have one class for each such required form.
For example :mod:`~tenpy.algorithms.dmrg` requires an :class:`~tenpy.models.model.MPOModel`,
which contains the Hamiltonian written as an :class:`~tenpy.networks.mpo.MPO`.
On the other hand, if we want to evolve a state with :mod:`~tenpy.algorithms.tebd`
we need a :class:`~tenpy.models.model.NearestNeighborModel`, in which the Hamiltonian is written in terms of
two-site bond-terms to allow a Suzuki-Trotter decomposition of the time-evolution operator.

Implmenting you own model ultimatley means to get an instance of :class:`~tenpy.models.model.MPOModel` or :class:`~tenpy.models.model.NearestNeighborModel`.
The predefined classes in the other modules under :mod:`~tenpy.models` are subclasses of at least one of those,
you will see examples later down below.

The Hilbert space
-----------------

The **local Hilbert** space is represented by a :class:`~tenpy.networks.site.Site` (read its doc-string!).
In particular, the `Site` contains the local :class:`~tenpy.linalg.charges.LegCharge` and hence the meaning of each
basis state needs to be defined.
Beside that, the site contains the local operators - those give the real meaning to the local basis.
Having the local operators in the site is very convenient, because it makes them available by name for example when you want to calculate expectation values.
The most common sites (e.g. for spins, spin-less or spin-full fermions, or bosons) are predefined
in the module :mod:`tenpy.networks.site`, but if necessary you can easily extend them
by adding further local operators or completely write your own subclasses of :class:`~tenpy.networks.site.Site`.

The full Hilbert space is a tensor product of the local Hilbert space on each site.

.. note ::

    The :class:`~tenpy.linalg.charges.LegCharge` of all involved sites need to have a common
    :class:`~tenpy.linalg.np_conserved.ChargeInfo` in order to allow the contraction of tensors acting on the various sites.
    This can be ensured with the function :func:`~tenpy.networks.site.multi_sites_combine_charges`.


An example where :func:`~tenpy.networks.site.multi_sites_combine_charges` is needed would be a coupling of different
types of sites, e.g., when a tight binding chain of fermions is coupled to some local spin degrees of freedom.
Another use case of this function would be a model with a $U(1)$ symmetry involving only half the sites, say :math:`\sum_{i=0}^{L/2} n_{2i}`.

.. note ::

    If you don't know about the charges and `np_conserved` yet, but want to get started with models right away,
    you can set ``conserve=None`` in the existing sites or use
    ``leg = tenpy.linalg.np_conserved.LegCharge.from_trivial(d)`` for an implementation of your custom site,
    where `d` is the dimension of the local Hilbert space.
    Alternatively, you can find some introduction to the charges in the :doc:`/intro_npc`.


The geometry : lattices
-----------------------

The geometry is usually given by some kind of **lattice** structure how the sites are arranged,
e.g. implicitly with the sum over nearest neighbours :math:`\sum_{<i, j>}`.
In TeNPy, this is specified by a :class:`~tenpy.models.lattice.Lattice` class, which contains a unit cell of
a few :class:`~tenpy.networks.site.Site` which are shifted periodically by its basis vectors to form a regular lattice.
Again, we have pre-defined some basic lattices like a :class:`~tenpy.models.lattice.Chain`,
two chains coupled as a :class:`~tenpy.models.lattice.Ladder` or 2D lattices like the
:class:`~tenpy.models.lattice.Square`, :class:`~tenpy.models.lattice.Honeycomb` and
:class:`~tenpy.models.lattice.Kagome` lattices; but you are also free to define your own generalizations.
(More details on that can be found in the doc-string of :class:`~tenpy.models.lattice.Lattice`, read it!)

**Visualization** of the lattice can help a lot to understand which sites are connected by what couplings.
The methods ``plot_...`` of the :class:`~tenpy.models.lattice.Lattice` can do a good job for a quick illustration.
We include a small image in the documation of each of the lattices.
For example, the following small script can generate the image of the Kagome lattice shown below::

    import matplotlib.pyplot as plt
    from tenpy.models.lattice import Kagome

    ax = plt.gca()
    lat = Kagome(4, 4, None, bc='periodic')
    lat.plot_coupling(ax, lat.nearest_neighbors, linewidth=3.)
    lat.plot_order(ax=ax, linestyle=':')
    lat.plot_sites()
    lat.plot_basis(ax, color='g', linewidth=2.)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()

.. image :: images/lattices/Kagome.*

The lattice contains also the **boundary conditions** `bc` in each direction. It can be one of the usual ``'open'`` or
``'periodic'`` in each direcetion. Instead of just saying "periodic", you can also specify a `shift` (except in the
first direction). This is easiest to understand at its standard usecase: DMRG on a infinite cylinder.
Going around the cylinder, you have a degree of freedom which sites to connect.
The orange markers in the following figures illustrates sites identified for a Square lattice with ``bc=['periodic', shift]`` (see :meth:`~tenpy.models.lattice.Lattice.plot_bc_shift`):

.. image :: images/lattices/square_bc_shift.*

Note that the "cylinder" axis (and direction for :math:`k_x`) is perpendicular to the orange line connecting these
sites. The line where the cylinder is "cut open" therefore winds around the the cylinder for a non-zero `shift` (or
more complicated lattices without perpendicular basis).


MPS based algorithms like DMRG always work on purely 1D systems. Even if our model "lives" on a 2D lattice,
these algorithms require to map it onto a 1D chain (probably at the cost of longer-range interactions).
This mapping is also done in by the lattice, as it defines an **order** (:attr:`~tenpy.models.lattice.Lattice.order`) of the sites.
The methods :meth:`~tenpy.models.lattice.Lattice.mps2lat_idx` and :meth:`~tenpy.models.lattice.Lattice.lat2mps_idx` map
indices of the MPS to and from indices of the lattice. If you obtained and array with expectation values for a given MPS,
you can use :meth:`~tenpy.models.lattice.Lattice.mps2lat_values` to map it to lattice indices, thereby reverting the ordering.

.. note ::

    A suitable ordering is critical for the efficiency of MPS-based algorithms.
    On one hand, different orderings can lead to different MPO bond-dimensions, with direct impact on the complexity scaling.
    Moreover, it influences how much entanglement needs to go through each bonds of the underlying MPS,
    e.g., the ground strate to be found in DMRG, and therefore influence the required MPS bond dimensions.
    For the latter reason, the "optimal" bond dimension can not be known a priori, but one needs to try different
    orderings.

Performing this mapping of the Hamiltonain from a 2D lattice to a 1D chain by hand can be a tideous process.
Therefore, we have automated this mapping in TeNPy as explained in the next section.
(Nevertheless it's a good exercise you should do at least once in your life to understand how it works!)

Implementing you own model
--------------------------
When you want to simulate a model not provided in :mod:`~tenpy.models`, you need to implement your own model class,
lets call it ``MyNewModel``.
The idea is that you define a new subclass of one or multiple of the model base classes.
For example, when you plan to do DMRG, you have to provide an MPO in a :class:`~tenpy.models.MPOModel`,
so your model class should look like this::

    class MyNewModel(MPOModel):
        """General strucutre for a model suitable for DMRG.

        Here is a good place to document the represented Hamiltonian and parameters.

        In the models of TeNPy, we usually take a single dictionary `model_params`
        containing all parameters, and read values out with ``tenpy.tools.params.get_parameter(...)``,
        The model needs to provide default values if the parameters was not specified.
        """
        def __init__(self, model_params):
            # some code here to read out model parameters and generate H_MPO
            lattice = somehow_generate_lattice(model_params)
            H_MPO = somehow_generate_MPO(lattice, model_params)
            # initialize MPOModel
            MPOModel.__init__(self, lattice, H_MPO)

TEBD requires another representation of H in terms of bond terms `H_bond` given to a
:class:`~tenpy.models.NearestNeighborModel`, so in this case it would look so like this instead::

    class MyNewModel2(NearestNeighborModel):
        """General strucutre for a model suitable for TEBD."""
        def __init__(self, model_params):
            # some code here to read out model parameters and generate H_bond
            lattice = somehow_generate_lattice(model_params)
            H_bond = somehow_generate_H_bond(lattice, model_params)
            # initialize MPOModel
            NearestNeighborModel.__init__(self, lattice, H_bond)

.. note :

    The :class:`~tenpy.models.model.NearestNeighborModel` is only suitable for models which are "nearest-neighbor"
    in the sense of the 1D MPS "snake", not in the sense of the lattice,
    i.e., it only works for nearest-neigbor models on a 1D chain.

Of course, the difficult part in these examples is to generate the ``H_MPO`` and ``H_bond``.
Moreover, it's quite annoying to write every model multiple times,
just because we need different representations of the same Hamiltonian.
Luckily, there is a way out in TeNPy: the CouplingModel!


The easy way to new models: the (Multi)CouplingModel
----------------------------------------------------

The :class:`~tenpy.models.model.CouplingModel` provides a general, quite abstract way to specify a Hamiltonian
of two-site couplings on a given lattice.
Once initialized, its methods :meth:`~tenpy.models.CouplingModel.add_onsite` and
:meth:`~tenpy.models.model.CouplingModel.add_coupling` allow to add onsite and coupling terms repeated over the different
unit cells of the lattice.
In that way, it basically allows a straight-forward translation of the Hamiltonian given as a math forumla
:math:`H = \sum_{i} A_i B_{i+dx} + ...` with onsite operators `A`, `B`,... into a model class.

The general structure for a new model based on the :class:`~tenpy.models.model.CouplingModel` is then::

    class MyNewModel3(CouplingModel,MPOModel,NearestNeighborModel):
        def __init__(self, ...):
            ...  # follow the basic steps explained below


In the initialization method ``__init__(self, ...)`` of this class you can then follow these basic steps:

0. Read out the parameters.
1. Given the parameters, determine the charges to be conserved.
   Initialize the :class:`~tenpy.linalg.charges.LegCharge` of the local sites accordingly.
2. Define (additional) local operators needed.
3. Initialize the needed :class:`~tenpy.networks.site.Site`.

   .. note ::

      Using pre-defined sites like the :class:`~tenpy.networks.site.SpinHalfSite` is recommended and
      can replace steps 1-3.

4. Initialize the lattice (or if you got the lattice as a parameter, set the sites in the unit cell).
5. Initialize the :class:`~tenpy.models.model.CouplingModel` with ``CouplingModel.__init__(self, lat)``.
6. Use :meth:`~tenpy.models.CouplingModel.add_onsite` and :meth:`~tenpy.models.model.CouplingModel.add_coupling`
   to add all terms of the Hamiltonian. Here, the :attr:`~tenpy.models.lattice.Lattice.nearest_neighbors` of the lattice
   (and its friends for next nearest neighbors) can come in handy, for example::

       self.add_onsite(-np.asarray(h), 0, 'Sz')
       for u1, u2, dx in self.lat.nearest_neighbors:
           self.add_coupling(J, u1, 'Sz', u2, 'Sz', dx)

   .. note ::

      The method :meth:`~tenpy.models.model.CouplingModel.add_coupling` adds the coupling only in one direction, i.e.
      not switching `i` and `j` in a :math:`\sum_{\langle i, j\rangle}`.
      If you have terms like :math:`c^\dagger_i c_j` in your Hamiltonian, you *need* to add it in both directions to get
      a hermitian Hamiltonian! Simply add another line ``self.add_coupling(J, u1, 'Sz', u2, 'Sz', -dx)``.

   Note that the `strength` arguments of these functions can be (numpy) arrays for site-dependent couplings.
   If you need to add or multipliy some parameters of the model for the `strength` of certain terms,
   it is recommended use ``np.asarray`` beforehand -- in that way lists will also work fine.
7. Finally, if you derived from the :class:`~tenpy.models.model.MPOModel`, you can call
   :meth:`~tenpy.models.model.CouplingModel.calc_H_MPO` to build the MPO and use it for the initialization
   as ``MPOModel.__init__(self, lat, self.calc_H_MPO())``.
8. Similarly, if you derived from the :class:`~tenpy.models.model.NearestNeighborModel`, you can call
   :meth:`~tenpy.models.model.CouplingModel.calc_H_MPO` to initialze it
   as ``NearestNeighborModel.__init__(self, lat, self.calc_H_bond())``.
   Calling ``self.calc_H_bond()`` will fail for models which are not nearest-neighbors (with respect to the MPS ordering),
   so you should only subclass the :class:`~tenpy.models.model.NearestNeighborModel` if the lattice is a simple
   :class:`~tenpy.models.lattice.Chain`.

The :class:`~tenpy.models.model.CouplingModel` works for Hamiltonians which are a sum of terms involving at most two sites.
The generalization :class:`~tenpy.models.model.MultiCouplingModel` can be used for Hamlitonians with
coupling terms acting on more than 2 sites at once. Follow the exact same steps in the initialization, and just use the
:meth:`~tenpy.models.model.MultiCouplingModel.add_multi_coupling` instead or in addition to the
:meth:`~tenpy.models.model.CouplingModel.add_coupling`.
A prototypical example is the exactly solvable :class:`~tenpy.models.toric_code.ToricCode`.

The code of the module :mod:`tenpy.models.xxz_chain` is included below as an illustrative example how to implemnet a
Model. The implementation of the :class:`~tenpy.models.xxz_chain.XXZChain` directly follows the steps
outline above.
The :class:`~tenpy.models.model.xxz_chain.XXZChain2` implements the very same model, but based on the
:class:`~tenpy.models.model.CouplingMPOModel` explained in the next section.


.. literalinclude:: ../tenpy/models/xxz_chain.py

The easy easy way: the CouplingMPOModel
---------------------------------------
Since many of the basic steps above are always the same, we don't need to repeat them all the time.
So we have yet another class helping to structure the initialization of models: the :class:`~tenpy.models.model.CouplingMPOModel`.
The general structure of the  class is like this::

    class CouplingMPOModel(CouplingModel,MPOModel):
        def __init__(self, model_param):
            # ... follow the basic steps 1-8 using the methods
            lat = self.init_lattice(self, model_param)  # for step 4
            # ...
            self.init_terms(self, model_param) # for step 6
            # ...

        def init_sites(self, model_param):
            # You should overwrite this

        def init_lattice(self, model_param):
            sites = self.init_sites(self, model_param) # for steps 1-3
            # initialize an arbitrary pre-defined lattice
            # using model_params['lattice']

        def init_terms(self, model_param):
            # does nothing.
            # You should overwrite this

The :class:`~tenpy.models.model.xxz_chain.XXZChain2` included above illustrates, how it can be used.
You need to implement steps 1-3) by overwriting the method :meth:`~tenpy.models.model.CouplingMPOModel.init_sites`
Step 4) is performed in the method :meth:`~tenpy.models.model.CouplingMPOModel.init_lattice`, which initializes arbitrary 1D or 2D
lattices; by default a simple 1D chain.
If your model only works for specific lattices, you can overwrite this method in your own class.
Step 6) should be done by overwriting the method :meth:`~tenpy.models.model.CouplingMPOModel.init_terms`.
Steps 5,7,8 and calls to the `init_...` methods for the other steps are done automatically if you just call the
``CouplingMPOModel.__init__(self, model_param)``.

The :class:`~tenpy.models.model.xxz_chain.XXZChain` and :class:`~tenpy.models.model.xxz_chain.XXZChain2` work only with the
:class:`~tenpy.models.lattice.Chain` as lattice, since they are derived from the :class:`~tenpy.models.model.NearestNeighborModel`.
This allows to use them for TEBD in 1D (yeah!), but we can't get the MPO for DMRG on a e.g. a :class:`~tenpy.models.lattice.Square`
lattice cylinder - although it's intuitively clear, what the hamiltonian there should be: just put the nearest-neighbor
coupling on each bond of the 2D lattice.

It's not possible to generalize a :class:`~tenpy.models.model.NearestNeighborModel` to an arbitrary lattice where it's
no longer nearest Neigbors in the MPS sense, but we can go the other way around:
first write the model on an arbitrary 2D lattice and then restrict it to a 1D chain to make it a :class:`~tenpy.models.model.NearestNeighborModel`.

Let me illustrate this with another standard example model: the transverse field Ising model, imlemented in the module
:mod:`tenpy.models.tf_ising` included below.
The :class:`~tenpy.models.tf_ising.TFIsingModel` works for arbitrary 1D or 2D lattices.
The :class:`~tenpy.models.tf_ising.TFIsingCHain` is then taking the exact same model making a :class:`~tenpy.models.model.NearestNeighborModel`,
which only works for the 1D chain.

.. literalinclude:: ../tenpy/models/tf_ising.py


Some final remarks
------------------

- Needless to say that we have also various predefined models under :mod:`tenpy.models`.

- Of course, an MPO is all you need to initialize a :class:`~tenpy.models.model.MPOModel` to be used for DMRG; you don't have to use the :class:`~tenpy.models.model.CouplingModel`
  or :class:`~tenpy.models.model.CouplingMPOModel`.
  For example an exponentially decaying long-range interactions are not supported by the coupling model but straight-forward to include to an MPO, as demonstrated in the example ``examples/mpo_exponentially_decaying.py``.

- If the model of your interest contains Fermions, you should read the :doc:`/intro_JordanWigner`.

- We suggest writing the model to take a single parameter dicitionary for the initialization, which is to be read out
  inside the class with :func:`~tenpy.tools.params.get_parameter`.
  Read the doc-string of this function for more details on why this is a good idea.
  Th `CouplingMPOModel.__init__()` calls :func:`~tenpy.tools.params.unused_parameters`, helping to avoid typos in the specified parameters.

- When you write a model and want to include a test that it can be at least constructed,
  take a look at ``tests/test_model.py``.
