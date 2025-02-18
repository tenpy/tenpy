Models (Introduction)
=====================

This is an introduction to models in TeNPy, intended to guide new-comers towards defining
their own custom models.

We go step by step to introduce the relevant concepts and "derive" how you could have come up
with the following example code to implement an anisotropic Heisenberg model on a square lattice::

    class MyModel(CouplingMPOModel):

        def init_sites(self, model_params):
                conserve = model_params.get('conserve', 'best')
                if conserve == 'best':
                    if model_params.get('Jx', 1) == 0:
                        conserve = 'Sz'
                    else:
                        conserve = 'parity'
                return SpinHalfSite(conserve=conserve)

        def init_terms(self, model_params):
            Jx = model_params.get('Jx', 1.)
            Jz = model_params.get('Jz', 1.)
            h = model_params.get('h', 0.)

            for u in range(len(self.lat.unit_cell)):
                self.add_onsite(-h, u, 'Sz')

            for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
                self.add_coupling(Jz, u1, 'Sz', u2, 'Sz', dx)
                self.add_coupling(.5 * Jx, u1, 'Sp', u2, 'Sm', dx, plus_hc=True)

    model = MyModel({'lattice': 'Square', 'Lx': 2, 'Ly': 4, 'Jx': 0})


What is a model?
----------------

Abstractly, a **model** stands for some physical (quantum) system, described by a Hamiltonian.
For example, let us consider an anisotropic spin-1/2 Heisenberg model in a field, described by

.. math ::

    H = J_x \sum_i (S^x_i S^x_{i+1} + S^y_i S^y_{i+1}) + J_z \sum_i S^z_i S^z_{i+1} - h \sum_i S^z_i

The main features that need to be defined for a model are

1. The local Hilbert space. In this example it is a Spin-1/2 degree of freedom with the usual spin operators :math:`S^x, S^y, S^z`.
2. The problem geometry, in terms of lattice type, size and boundary conditions.
3. The Hamiltonian itself. Here, it is naturally expressed as a sum of couplings.

In the following, we guide you towards defining your own custom model, with the above case as an example.
Note that this particular Hamiltonian is already implemented, in terms of the more general
:class:`~tenpy.models.spins.SpinModel`.
We first follow the most direct route, using the :class:`~tenpy.models.model.CouplingMPOModel` framework,
and present alternatives later.

The first step is to identify what the **parameters** of your model are.
In this case, we have the coupling constants :math:`J_x, J_z, h`, and parameters that specify
the lattice geometry (discussed later). In the TeNPy ecosystem, these parameters are
gathered into dictionary-like objects, and for the rest of this guide you can think of
``model_params`` as a dictionary of these parameters, e.g. ``model_params = {Jx: 0.5, Jz: 1}``.
Note that every parameter should have a reasonable default value.

We start implementing our custom model by defining a class for it::

    class MyModel(CouplingMPOModel):
        r"""The anisotropic spin-1/2 Heisenberg model in an external field.

        Hamiltonian::
            H = Jx \sum_i (S^x_i S^x_{i+1} + S^y_i S^y_{i+1}) + Jz \sum_i S^z_i S^z_{i+1} - h \sum_i S^z_i

        Model parameters: ``Jx, Jz, h``. Defaults ``Jx=1, Jz=1, h=0``.
        """
        pass  # content will be added later


Note that we define our model as a subclass of :class:`~tenpy.models.model.CouplingMPOModel`.
This means our model inherits all the machinery to build Hamiltonians etc, and we only need
to implement the code that is specific to our model.


The local Hilbert space
-----------------------
The **local Hilbert** space is represented by a :class:`~tenpy.networks.site.Site` (read its doc-string!).

A site allows defines the meaning of each basis state (e.g. by fixing an order, to define e.g.
that the state are ``spin_down, spin_up``). Additionally, it stores common local operators, such as
:math:`S^x` and makes them accessible by name.

We need to tell our model, what its local Hilbert space is.
This is done by implementing the :meth:`~tenpy.models.model.CouplingMPOModel.init_sites` method.
It needs to take the ``model_params`` as input and return one :class:`~tenpy.networks.site.Site`
per site in the unit cell of the lattice (see lattice section below, here this is one site).
The most common sites (e.g. for spins, spin-less or spin-full fermions, or bosons) are predefined
in the module :mod:`tenpy.networks.site`, and in this example we can use one of them directly::

    class MyModel(CouplingMPOModel):

        def init_sites(self, model_params):
            # simple version: no charge conservation
            return SpinHalfSite(conserve='None')


If necessary, you can easily extend a pre-defined site by adding further local operators or
completely write your own subclasses of :class:`~tenpy.networks.site.Site`.

If you want to use charge conservation (and you probably should, if possible), we need to specify
what charges are conserved at this point already, i.e. we should give a value to the ``conserve``
argument of the site.

.. note ::

    If you don't know about :doc:`/intro/npc` yet, but want to get started with models right away,
    you can set ``conserve=None`` in the existing sites as above and skip the rest of this
    section.

    If you need a custom site, you can use 
    ``leg = tenpy.linalg.np_conserved.LegCharge.from_trivial(d)`` for an implementation of your
    custom site, where `d` is the dimension of the local Hilbert space.


In many cases, the possible symmetries we may exploit depend on the
values of the parameters, which is why they are an input to ``init_sites``.
In our example, we can conserve :math:`S^z` if :math:`J = 0`, and only its parity otherwise.

    class MyModel(CouplingMPOModel):

        def init_sites(self, model_params):
            conserve = model_params.get('conserve', 'best')
            if conserve == 'best':
                if model_params.get('Jx', 1) == 0:
                    conserve = 'Sz'
                else:
                    conserve = 'parity'
            return SpinHalfSite(conserve=conserve)


Note that we added ``conserve`` as a model parameters, such that we can later turn charge
conservation on or off. The possible values for ``conserve`` are documented in the site class,
here :class:`~tenpy.networks.site.SpinHalfSite`, and it is common to support ``'best'``
as a value for the ``conserve`` model parameter and translate it to the largest possible symmetry,
given the other parameters.

.. note ::

    The :class:`~tenpy.linalg.charges.LegCharge` of all involved sites need to have a common
    :class:`~tenpy.linalg.np_conserved.ChargeInfo` in order to allow the contraction of tensors
    acting on the various sites.
    This can be ensured with the function :func:`~tenpy.networks.site.set_common_charges`.

    An example where :func:`~tenpy.networks.site.set_common_charges` is needed would be a coupling
    of different types of sites, e.g., when a tight binding chain of fermions is coupled to some
    local spin degrees of freedom. Another use case of this function would be a model with a $U(1)$
    symmetry involving only half the sites, say :math:`\sum_{i=0}^{L/2} n_{2i}`.


The geometry (lattice)
----------------------
The geometry is usually given by some kind of **lattice** structure how the sites are arranged,
e.g. implicitly with the sum over nearest neighbors :math:`\sum_{<i, j>}`.
In TeNPy, this is specified by a :class:`~tenpy.models.lattice.Lattice` class, which contains a unit cell of
a few :class:`~tenpy.networks.site.Site` which are shifted periodically by its basis vectors to form a regular lattice.
Again, we have pre-defined some basic lattices like a :class:`~tenpy.models.lattice.Chain`,
two chains coupled as a :class:`~tenpy.models.lattice.Ladder` or 2D lattices like the
:class:`~tenpy.models.lattice.Square`, :class:`~tenpy.models.lattice.Honeycomb` and
:class:`~tenpy.models.lattice.Kagome` lattices; but you are also free to define your own generalizations.

.. note ::

    Further details on the lattice geometry can be found in :doc:`/intro/lattices`.


By default, the :class:`~tenpy.models.model.CouplingMPOModel` puts your model on
a :class:`~tenpy.models.lattice.Chain`, and looks for its length as ``model_params['L']``.

If you want to use a different pre-defined lattice, you can put it into the parameters, e.g.
as ``model_params['lattice'] = 'Square'``, and the size is taken from ``model_params['Lx']``
and ``model_params['Ly']``, while the boundary conditions are ``model_params['bc_x']``
and ``model_params['bc_y']``.

Of course, simply changing the lattice only makes sense if the Hamiltonian is defined in a lattice
independent language, e.g. in terms of "nearest neighbor pairs".
As we will explore in the next section, this is in fact the natural way to define Hamiltonians in TeNPy.

It is also common to have specialized classes for special lattices::

    class MyModelKagome(MyModel):
        default_lattice = Kagome
        force_default_lattice = True

        def init_sites(self, model_params):
            # note: Kagome has three sites per unit-cell
            site = MyModel.init_site(model_params)
            return (site, site, site)


For custom lattices, or more complicated code, you can overwrite the
:meth:`~tenpy.models.model.CouplingMPOModel.init_lattice` method, similar to how we did
for ``init_sites`` above.


The Hamiltonian
---------------
The last ingredient we need to implement for a custom model is its Hamiltonian.
To that end, we override the :meth:`~tenpy.models.model.CouplingMPOModel.init_terms` method.
At this point during model initialization, the lattice is already initialized, and we
may access ``self.lat`` and use e.g. the :attr:`~tenpy.models.lattice.Lattice.pairs` attribute
for convenient definition of couplings between e.g. nearest-neighbor pairs.

There are a bunch of convenience methods implemented in :class:`~tenpy.models.model.CouplingModel`,
which make this easy. Let us summarize them here:

- :meth:`~tenpy.models.model.CouplingModel.add_onsite`
    for onsite terms :math:`\sum_i h_i \hat{A}_i`.
- :meth:`~tenpy.models.model.CouplingModel.add_coupling`
    for two-body couplings :math:`\sum_i J_i \hat{A}_i \hat{B}_{i+n}`
- :meth:`~tenpy.models.model.CouplingModel.add_multi_coupling`
    for multi-body couplings :math:`\sum_i J_i \hat{A}_i \hat{B}_{i+n} ... \hat{F}_{i+m}`

.. note ::
    A single call to each of these methods adds an extensive number of terms to your Hamiltonian,
    as it includes a sum over all sites in the definition.
    This means that a Hamiltonian like ``H = -3 \sum_i S_i^z`` is realized as a **single** call to 
    :meth:`~tenpy.models.model.CouplingModel.add_onsite`, **without**  an explicit loop over `i`.

.. note ::
    These methods allow the prefactors to be site-dependent; you can either give a single number
    as the prefactor, or a list/array that is tiled to fit the size.
    E.g. if a coupling ``strength=1`` gives you a ferromagnet, ``strength=[1, -1]`` gives you
    the corresponding anti-ferromagnet, assuming a chain of even length.

For each of those methods, there is a version that adds just a single term, i.e. without
the sum over lattice sites, but is less convenient, since it takes MPS indices instead of
lattice positions as inputs. They are :meth:`~tenpy.models.model.CouplingModel.add_onsite_term`,
:meth:`~tenpy.models.model.CouplingModel.add_coupling_term` and
:meth:`~tenpy.models.model.CouplingModel.add_multi_coupling_term`.

See also :meth:`~tenpy.models.model.CouplingModel.add_exponentially_decaying_coupling`

For our example, we define the Hamiltonian by implementing::

    class MyModel(CouplingMPOModel):

        def init_sites(self, model_params):
            ...

        def init_terms(self, model_params):
            Jx = model_params.get('Jx', 1.)
            Jz = model_params.get('Jz', 1.)
            h = model_params.get('h', 0.)

            for u in range(len(self.lat.unit_cell)):
                self.add_onsite(-h, u, 'Sz')

            for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
                self.add_coupling(Jz, u1, 'Sz', u2, 'Sz', dx)

                # Sx and Sy violate parity conservation, but Sx.Sx and Sy.Sy do not.
                # need to define them using Sp = Sx + i Sy, Sm = Sx - i Sy
                # Sx.Sx + Sy.Sy = .5 * (Sp.Sm + Sm.Sp) = .5 * (Sp.Sm + hc)
                self.add_coupling(.5 * Jx, u1, 'Sp', u2, 'Sm', dx, plus_hc=True)

.. note ::
    If we did not care about charge conservation, we could have also done
    ``add_coupling(Jx, u1, 'Sx', u2, 'Sx', dx)`` and 
    ``add_coupling(Jx, u1, 'Sy', u2, 'Sy', dx)``.
    This only works if we set ``conserve='None'``, as otherwise the site does not even
    define ``'Sx'``.


At this point we are done defining our model, and have reproduced the result at the very top
of the chapter. We should, however, make sure that we defined the model correctly.


Verifying models
----------------
Especially when you define custom models, we strongly recommend you triple-check if you correctly
implemented the model you are interested in (i.e. have the correct couplings at between correct sites).
This is a crucial step to make sure you are in fact simulating the model that you are thinking
about and not some random other model with entirely different physics.

.. note ::
    If the model contains Fermions, you should read the introduction to :doc:`/intro/JordanWigner`.


To verify that you have added the correct terms, initialize the model on a small lattice,
e.g.::

    model = MyModel({'lattice': 'Square', 'Lx': 2, 'Ly': 3, 'Jx': 0})


Now, run ``print(model.all_coupling_terms().to_TermList())`` to print a list of all coupling
terms that the model has. It gives them in terms of site indices, which may be hard to read.
To visualize the site order of the lattice, run the following snippet::

    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 6))
    ax = plt.gca()
    model.lat.plot_coupling(ax)
    model.lat.plot_sites(ax)
    model.lat.plot_order(ax)
    plt.show()


You may be surprised to find a coupling ``1.00000 * Sz_0 Sz_2``. 
We have this coupling, because the default boundary conditions in y-direction are periodic.
Note how this coupling is not present for ``MyModel({'lattice': 'Square', 'Lx': 2, 'Ly': 3, 'Jx': 0, 'bc_y': 'open'})``.


Contribute your model?
----------------------
If you have implemented a your model, it may be useful to the broader community.
If you like, consider contributing it to TeNPy via a pull request.
We have coding guidelines at :doc:`contributing`, and you can have a look at the implementation
of e.g. the :class:`~tenpy.models.spins.SpinModel` for documentation style, but do not let that
stop you from sharing your code, we can always address any nitpicks ourselves.


Details on the implementation of Models
=======================================
In this chapter, we provide some more detail on how models work, and how you might customize them.


The CouplingModel: general structure
------------------------------------

The :class:`~tenpy.models.model.CouplingModel` provides a general, quite abstract way to specify a Hamiltonian
of couplings on a given lattice.
Once initialized, its methods :meth:`~tenpy.models.CouplingModel.add_onsite` and
:meth:`~tenpy.models.model.CouplingModel.add_coupling` allow to add onsite and coupling terms repeated over the different
unit cells of the lattice.
In that way, it basically allows a straight-forward translation of the Hamiltonian given as a math formula
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
6. Use :meth:`~tenpy.models.model.CouplingModel.add_onsite` and :meth:`~tenpy.models.model.CouplingModel.add_coupling`
   to add all terms of the Hamiltonian. Here, the :attr:`~tenpy.models.lattice.Lattice.pairs` of the lattice
   can come in handy, for example::

       self.add_onsite(-np.asarray(h), 0, 'Sz')
       for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
           self.add_coupling(0.5*J, u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
           self.add_coupling(    J, u1, 'Sz', u2, 'Sz', dx)

   .. note ::

      The method :meth:`~tenpy.models.model.CouplingModel.add_coupling` adds the coupling only in one direction, i.e.
      not switching `i` and `j` in a :math:`\sum_{\langle i, j\rangle}`.
      If you have terms like :math:`c^\dagger_i c_j` or :math:`S^{+}_i S^{-}_j` in your Hamiltonian, 
      you *need* to add it in both directions to get a Hermitian Hamiltonian!
      The easiest way to do that is to use the `plus_hc` option of
      :meth:`~tenpy.models.model.CouplingModel.add_onsite` and :meth:`~tenpy.models.model.CouplingModel.add_coupling`,
      as we did for the :math:`J/2 (S^{+}_i S^{-}_j + h.c.)` terms of the Heisenberg model above.
      Alternatively, you can add the hermitian conjugate terms explicitly, see the examples in 
      :meth:`~tenpy.models.model.CouplingModel.add_coupling` for more details.

   Note that the `strength` arguments of these functions can be (numpy) arrays for site-dependent couplings.
   If you need to add or multiply some parameters of the model for the `strength` of certain terms,
   it is recommended use ``np.asarray`` beforehand -- in that way lists will also work fine.
7. Finally, if you derived from the :class:`~tenpy.models.model.MPOModel`, you can call
   :meth:`~tenpy.models.model.CouplingModel.calc_H_MPO` to build the MPO and use it for the initialization
   as ``MPOModel.__init__(self, lat, self.calc_H_MPO())``.
8. Similarly, if you derived from the :class:`~tenpy.models.model.NearestNeighborModel`, you can call
   :meth:`~tenpy.models.model.CouplingModel.calc_H_bond` to initialize it
   as ``NearestNeighborModel.__init__(self, lat, self.calc_H_bond())``.
   Calling ``self.calc_H_bond()`` will fail for models which are not nearest-neighbors (with respect to the MPS ordering),
   so you should only subclass the :class:`~tenpy.models.model.NearestNeighborModel` if the lattice is a simple
   :class:`~tenpy.models.lattice.Chain`.

.. note ::

    The method :meth:`~tenpy.models.model.CouplingModel.add_coupling` works only for terms involving operators on 2
    sites. If you have couplings involving more than two sites, you can use the
    :meth:`~tenpy.models.model.CouplingModel.add_multi_coupling` instead.
    A prototypical example is the exactly solvable :class:`~tenpy.models.toric_code.ToricCode`.


The code of the module :mod:`tenpy.models.xxz_chain` is included below as an illustrative example how to implement a
Model. The implementation of the :class:`~tenpy.models.xxz_chain.XXZChain` directly follows the steps
outline above.
The :class:`~tenpy.models.xxz_chain.XXZChain2` implements the very same model, but based on the
:class:`~tenpy.models.model.CouplingMPOModel` explained in the next section.


.. literalinclude:: /../tenpy/models/xxz_chain.py

The easiest way: the CouplingMPOModel
-------------------------------------
Since many of the basic steps above are always the same, we don't need to repeat them all the time.
So we have yet another class helping to structure the initialization of models: the :class:`~tenpy.models.model.CouplingMPOModel`.
The general structure of this class is like this::

    class CouplingMPOModel(CouplingModel,MPOModel):
        default_lattice = "Chain"
        "

        def __init__(self, model_param):
            # ... follows the basic steps 1-8 using the methods
            lat = self.init_lattice(self, model_param)  # for step 4
            # ...
            self.init_terms(self, model_param) # for step 6
            # ...

        def init_sites(self, model_param):
            # You should overwrite this in most cases to ensure
            # getting the site(s) and charge conservation you want
            site = SpinSite(...)  # or FermionSite, BosonSite, ...
            return site  # (or tuple of sites)

        def init_lattice(self, model_param):
            sites = self.init_sites(self, model_param) # for steps 1-3
            # and then read out the class attribute `default_lattice`,
            # initialize an arbitrary pre-defined lattice
            # using model_params['lattice']
            # and ensure it's the default lattice if the class attribute
            # `force_default_lattice` is True.

        def init_terms(self, model_param):
            # does nothing.
            # You should overwrite this

The :class:`~tenpy.models.xxz_chain.XXZChain2` included above illustrates, how it can be used.
You need to implement steps 1-3) by overwriting the method :meth:`~tenpy.models.model.CouplingMPOModel.init_sites`
Step 4) is performed in the method :meth:`~tenpy.models.model.CouplingMPOModel.init_lattice`, which initializes arbitrary 1D or 2D
lattices; by default a simple 1D chain.
If your model only works for specific lattices, you can overwrite this method in your own class.
Step 6) should be done by overwriting the method :meth:`~tenpy.models.model.CouplingMPOModel.init_terms`.
Steps 5,7,8 and calls to the `init_...` methods for the other steps are done automatically if you just call the
``CouplingMPOModel.__init__(self, model_param)``.

The :class:`~tenpy.models.xxz_chain.XXZChain` and :class:`~tenpy.models.xxz_chain.XXZChain2` work only with the
:class:`~tenpy.models.lattice.Chain` as lattice, since they are derived from the :class:`~tenpy.models.model.NearestNeighborModel`.
This allows to use them for TEBD in 1D (yeah!), but we can't get the MPO for DMRG on (for example) a :class:`~tenpy.models.lattice.Square`
lattice cylinder - although it's intuitively clear, what the Hamiltonian there should be: just put the nearest-neighbor
coupling on each bond of the 2D lattice.

It's not possible to generalize a :class:`~tenpy.models.model.NearestNeighborModel` to an arbitrary lattice where it's
no longer nearest Neighbors in the MPS sense, but we can go the other way around:
first write the model on an arbitrary 2D lattice and then restrict it to a 1D chain to make it a :class:`~tenpy.models.model.NearestNeighborModel`.

Let me illustrate this with another standard example model: the transverse field Ising model, implemented in the module
:mod:`tenpy.models.tf_ising` included below.
The :class:`~tenpy.models.tf_ising.TFIModel` works for arbitrary 1D or 2D lattices.
The :class:`~tenpy.models.tf_ising.TFIChain` is then taking the exact same model making a :class:`~tenpy.models.model.NearestNeighborModel`,
which only works for the 1D chain.

.. literalinclude:: /../tenpy/models/tf_ising.py


Automation of Hermitian conjugation
-----------------------------------
As most physical Hamiltonians are Hermitian, these Hamiltonians are fully determined when only half of the mutually conjugate terms is defined. For example, a simple Hamiltonian:

.. math ::
        H = \sum_{\langle i,j\rangle, i<j}
              - \mathtt{J} (c^{\dagger}_i c_j + c^{\dagger}_j c_i)

is fully determined by the term :math:`c^{\dagger}_i c_j` if we demand that Hermitian conjugates are included automatically.
In TeNPy, whenever you add a coupling using :meth:`~tenpy.models.model.CouplingModel.add_onsite`,
:meth:`~tenpy.models.model.CouplingModel.add_coupling()`, or :meth:`~tenpy.models.model.CouplingModel.add_multi_coupling()`,
you can use the optional argument `plus_hc` to automatically create and add the Hermitian conjugate of that coupling term - as shown above.

Additionally, in an MPO, explicitly adding both a non-Hermitian term and its conjugate increases the bond dimension of the MPO, which increases the memory requirements of the :class:`~tenpy.networks.mpo.MPOEnvironment`.
Instead of adding the conjugate terms explicitly, you can set a flag `explicit_plus_hc` in the :class:`~tenpy.models.model.MPOCouplingModel` parameters, which will ensure two things:

1. The model and the MPO will only store half the terms of each Hermitian conjugate pair added, but the flag `explicit_plus_hc` indicates that they *represent* `self + h.c.`.
   In the example above, only the term :math:`c^{\dagger}_i c_j` would be saved.
2. At runtime during DMRG, the Hermitian conjugate of the (now non-Hermitian) MPO will be computed and applied along with the MPO, so that the effective Hamiltonian is still Hermitian.

.. note ::

    The model flag `explicit_plus_hc` should be used in conjunction with the flag `plus_hc` in :meth:`~tenpy.models.model.CouplingModel.add_coupling()` or :meth:`~tenpy.models.model.CouplingModel.add_multi_coupling()`.
    If `plus_hc` is `False` while `explicit_plus_hc` is `True` the MPO bond dimension will not be reduced, but you will still pay the additional computational cost of computing the Hermitian conjugate at runtime.

Thus, we end up with several use cases, depending on your preferences. 
Consider the :class:`~tenpy.models.fermions_spinless.FermionModel`.
If you do not care about the MPO bond dimension, and want to add Hermitian conjugate terms manually, you would set `model_par['explicit_plus_hc'] = False` and write::

    self.add_coupling(-J, u1, 'Cd', u2, 'C', dx)
    self.add_coupling(np.conj(-J), u2, 'Cd', u1, 'C', -dx)

If you wanted to save the trouble of the extra line of code (but still did not care about MPO bond dimension), you would keep the `model_par`, but instead write::

    self.add_coupling(-J, u1, 'Cd', u2, 'C', dx, plus_hc=True)

Finally, if you wanted a reduction in MPO bond dimension, you would need to set `model_par['explicit_plus_hc'] = True`, and write::

    self.add_coupling(-J, u1, 'Cd', u2, 'C', dx, plus_hc=True)


Non-uniform terms and couplings
-------------------------------
The CouplingModel-methods :meth:`~tenpy.models.model.CouplingModel.add_onsite`, :meth:`~tenpy.models.model.CouplingModel.add_coupling`, 
and :meth:`~tenpy.models.model.CouplingModel.add_multi_coupling` add a sum over a "coupling" term shifted by lattice
vectors. However, some models are not that "uniform" over the whole lattice.

First of all, you might have some local term that gets added only at one specific location in the lattice.
You can add such a term for example with :meth:`~tenpy.models.model.CouplingModel.add_local_term`.

Second, if you have irregular lattices, take a look at the corresponding section in :doc:`/intro/lattices`.

Finally, note that the argument `strength` for the `add_onsite`, `add_coupling`, and `add_multi_coupling` methods 
can not only be a numpy scalar, but also a (numpy) array.
In general, the sum performed by the methods runs over the given term 
shifted by lattice vectors *as far as possible to still fit the term into the lattice*. 

For the :meth:`~tenpy.models.model.CouplingModel.add_onsite` case this criterion is simple: there is exactly one site in each lattice unit cell with the `u` specified as separate argument, so the correct shape for the `strength` array is simply given by :attr:`~tenpy.models.lattice.Lattice.Ls`.
For example, if you want the defacto standard model studied for many-body localization, a Heisenberg chain with random , uniform onsite field :math:`h^z_i \in [-W, W]`,

.. math ::

    H = J \sum_{i=0}^{L-1} \vec{S}_i \cdot \vec{S}_{i+1} - \sum_{i=0}^{L} h^z_i S^z_i

you can use the :class:`~tenpy.models.spins.SpinChain` with the following model parameters::

    L = 30 # or whatever you like...
    W = 5.  # MBL transition at W_c ~= 3.5 J
    model_params = {
        'L': L,
        'Jx': 1., 'Jy': 1., 'Jz': 1.,
        'hz': 2.*W*(np.random.random(L) - 0.5),  # random values in [-W, W], shape (L,)
        'conserve': 'best',
    }
    M = tenpy.models.spins.SpinChain(model_params)

For :meth:`~tenpy.models.model.CouplingModel.add_coupling` and :meth:`~tenpy.models.model.CouplingModel.add_multi_coupling`,
things become a little bit more complicated, and the correct shape of the `strength` array depends not only on the :attr:`~tenpy.models.lattice.Lattice.Ls`
but also on the boundary conditions of the lattice. Given a term, you can call
:meth:`~tenpy.models.lattice.Lattice.coupling_shape` and :meth:`~tenpy.models.lattice.Lattice.multi_coupling_shape` to find out the correct shape for `strength`.
To avoid any ambiguity, the shape of the `strength` always has to fit, at least after a tiling performed by :func:`~tenpy.tools.misc.to_array`.

For example, consider the Su-Schrieffer-Heeger model, a spin-less :class:`~tenpy.models.fermions.FermionChain` with hopping strength alternating between two values, say `t1` and `t2`.
You can generate this model for example like this::
    
    L = 30 # or whatever you like...
    t1, t2 = 0.5, 1.5
    t_array = np.array([(t1 if i % 2 == 0 else t2) for i in range(L-1)])
    model_params = {
        'L': L,
        't': t_array,
        'V': 0., 'mu': 0.,  # just free fermions, but you can generalize...
        'conserve': 'best'
    }
    M = tenpy.models.fermions.FermionChain(model_params)


Some random remarks on models
-----------------------------

- Needless to say that we have also various predefined models under :mod:`tenpy.models`.
- If you want to use random parameters, you should use ``model.rng`` as a random number generator and change ``model_params['random_seed']`` for different configurations.
- Of course, an MPO is all you need to initialize a :class:`~tenpy.models.model.MPOModel` to be used for DMRG; you don't have to use the :class:`~tenpy.models.model.CouplingModel`k
  or :class:`~tenpy.models.model.CouplingMPOModel`.
  For example an exponentially decaying long-range interactions are not supported by the coupling model but straight-forward to include to an MPO, as demonstrated in the example ``examples/mpo_exponentially_decaying.py``.
  The :class:`~tenpy.models.aklt.AKLTChain` is another example which is directly constructed from the `H_bond` terms.
- We suggest writing the model to take a single parameter dictionary for the initialization,
  as the :class:`~tenpy.models.model.CouplingMPOModel` does.
  The :class:`~tenpy.models.model.CouplingMPOModel` converts the dictionary to a dict-like 
  :class:`~tenpy.tools.params.Config` with some additional features before passing it on to the `init_lattice`,
  `init_site`, ... methods.
  It is recommended to read out providing default values with ``model_params.get("key", default_value)``, 
  see :meth:`~tenpy.tools.params.Config.get`.
- When you write a model and want to include a test that it can be at least constructed,
  take a look at ``tests/test_model.py``.
