Details on the implementation of Models
=======================================

In this chapter, we provide some more detail on how models work, and how you might customize them.
You should probably read :doc:`/intro/model` first.

We define abstract base classes for models:

- A :class:`~tenpy.models.model.EvolutionModel` must define time evolution operators, such that
  we can do dynamics simulations using e.g. TEBD or ExpMPOEvolution.

- A :class:`~tenpy.models.model.HamiltonianModel` must define a Hamiltonian operator, such that
  we can e.g. do groundstate search using DMRG, tangent space methods like TDVP. It defines the
  time evolution operator in terms of the Hamiltonian.

In the inheritance diagram of most models, there are two more classes that commonly appear.
They are, however, not abstract and can be directly initialized, should you so choose:

- In a :class:`~tenpy.models.model.CouplingModel`, the Hamiltonian is given symbolically, in
  the form of terms (see :mod:`~tenpy.networks.terms`), which are onsite operators and/or
  couplings that act on specific sites of a :class:`~tenpy.models.lattice.Lattice`.
  The :class:`~tenpy.models.model.CouplingModel` handles translating from lattice positions to
  MPS indices, and building an MPO and/or bond-operator representation of the Hamiltonian.

- The :class:`~tenpy.models.model.ModelBuilder` class takes care of a lot of boilerplate in model
  generation for you. It takes model parameters as a dict(-like) config and can e.g. automatically
  initialize the lattice for you.

In the vast majority of cases, models are defined by subclassing from
:class:`~tenpy.models.model.ModelBuilder` and overriding some methods.


The CouplingModel: general structure
------------------------------------

The :class:`~tenpy.models.model.CouplingModel` provides a general, quite abstract way to specify a
Hamiltonian of couplings on a given lattice.
Once initialized, its methods :meth:`~tenpy.models.CouplingModel.add_onsite_operator`,
:meth:`~tenpy.models.model.CouplingModel.add_two_site_coupling`
and :meth:`~tenpy.models.model.CouplingModel.add_multi_coupling`
allow to add onsite and coupling terms repeated over the different unit cells of the lattice.
In that way, it basically allows a straight-forward translation of the Hamiltonian given as a math formula
:math:`H = \sum_{i} h_{i, i+dx} + ...` with some local operators `h`, which we call couplings
into a model class.

The general structure for a new model based on the :class:`~tenpy.models.model.CouplingModel` is then::

    class MyNewModel3(CouplingModel):
        def __init__(self, ...):
            ...  # follow the basic steps explained below


In the initialization method ``__init__(self, ...)`` of this class you can then follow these basic steps:

0. Read out the parameters.
1. Given the parameters, determine the charges to be conserved.
   Commonly, we set a string attribute ``model.conserve`` of the model.
2. Initialize the backend, see :func:`lasyten.get_backend`.
3. Initialize the needed :class:`lasyten.Site`.
4. Initialize the lattice (or if you got the lattice as a parameter, set the sites in the unit cell).
5. Initialize the :class:`~tenpy.models.model.CouplingModel` with ``CouplingModel.__init__(self, lat)``.
6. Define the couplings required. Use :meth:`~tenpy.models.model.CouplingModel.define_coupling` to
   make the model know about the needed :class:`lasyten.Coupling`\s used, each with a name and defined
   behavior under dagger or site permutation.
7. Use :meth:`~tenpy.models.model.CouplingModel.add_onsite_operator`,
   :meth:`~tenpy.models.model.CouplingModel.add_two_site_coupling` and/or
   :meth:`~tenpy.models.model.CouplingModel.add_multi_coupling`
   to add all terms of the Hamiltonian.
   Here, the :attr:`~tenpy.models.lattice.Lattice.pairs` of the lattice can come in handy, for example::

       self.add_onsite_operator(-np.asarray(h), 'Sz')
       for u1, u2, dx in self.lattice.pairs['nearest_neighbors']:
           self.add_two_site_coupling(0.5*J, 'Sp.Sm', dx, u1, u2, plus_hc=True)
           self.add_two_site_coupling(    J, 'Sz.Sz', dx, u1, u2)

   .. note ::

      The method :meth:`~tenpy.models.model.CouplingModel.add_two_site_coupling` adds the coupling
      only in one direction, i.e. not switching `i` and `j` in a :math:`\sum_{\langle i, j\rangle}`.
      If you have terms like :math:`c^\dagger_i c_j` or :math:`S^{+}_i S^{-}_j` in your Hamiltonian,
      you *need* to add it in both directions to get a Hermitian Hamiltonian!
      The easiest way to do that is to use the `plus_hc` option of the "add" methods, e.g.
      :meth:`~tenpy.models.model.CouplingModel.add_onsite_operator`,
      as we did for the :math:`J/2 (S^{+}_i S^{-}_j + h.c.)` terms of the Heisenberg model above.
      Alternatively, you can add the hermitian conjugate terms explicitly.

   Note that the `strength` arguments of these functions can be (numpy) arrays for site-dependent couplings.
   If you need to add or multiply some parameters of the model for the `strength` of certain terms,
   it is recommended use ``np.asarray`` beforehand -- in that way lists will also work fine.

.. note ::

    The method :meth:`~tenpy.models.model.CouplingModel.add_two_site_coupling`, as the name suggests,
    only support two-site / two-body couplings. If you have couplings involving more than two sites,
    you can use the :meth:`~tenpy.models.model.CouplingModel.add_multi_coupling` instead.
    A prototypical example is the exactly solvable :class:`~tenpy.models.toric_code.ToricCode`.


The code of the module :mod:`tenpy.models.xxz_chain` is included below as an illustrative example
how to implement a Model. The implementation of the :class:`~tenpy.models.xxz_chain.XXZChain`
directly follows the steps outline above and serves as a pedagogical example.
The :class:`~tenpy.models.xxz_chain.XXZChain2` implements the very same model, but based on the
:class:`~tenpy.models.model.ModelBuilder` explained in the next section.


.. literalinclude:: /../tenpy/models/xxz_chain.py

The easiest way: the CouplingMPOModel
-------------------------------------
Since many of the basic steps above are always the same, we don't need to repeat them all the time.
So we have yet another class helping to structure the initialization of models:
the :class:`~tenpy.models.model.ModelBuilder`.
The general structure of this class is like this::

    class ModelBuilder(CouplingModel):
        default_lattice = "Chain"

        def __init__(self, model_param):
            # ... follows the basic steps 1-7 using the methods
            self.backend = backend = self.init_backend(model_params)  # for step 2
            lat = self.init_lattice(self, model_param)  # for step 1, 3 & 4
            # ...
            self.init_couplings(self, model_param) # for step 6
            self.init_terms(self, model_param) # for step 7
            # ...

        def init_sites(self, model_param):
            # should determine what to conserve and set self.conserve
            raise NotImplementedError('Subclasses should implement `init_sites`')

        def init_couplings(self, model_param):
            raise NotImplementedError('Subclasses should implement `init_couplings`')

        def init_lattice(self, model_param):
            sites = self.init_sites(self, model_param) # for steps 1-3
            # and then read out the class attribute `default_lattice`,
            # initialize an arbitrary pre-defined lattice
            # using model_params['lattice']
            # and ensure it's the default lattice if the class attribute
            # `force_default_lattice` is True.
            return lattice

        def init_terms(self, model_param):
            pass  # does nothing by default, you should overwrite this

The :class:`~tenpy.models.xxz_chain.XXZChain2` included above illustrates, how it can be used.
You need to implement steps 1 & 3 by overwriting the method
:meth:`~tenpy.models.model.CouplingMPOModel.init_sites`.
Step 4 is performed in the method :meth:`~tenpy.models.model.CouplingMPOModel.init_lattice`, which
initializes arbitrary 1D or 2D lattices; by default a simple 1D chain.
If your model only works for specific lattices, you can overwrite this method in your own class.
Step 6 should be done by overwriting the method :meth:`~tenpy.models.model.CouplingMPOModel.init_couplings`.
Step 7 should be done by overwriting the method :meth:`~tenpy.models.model.CouplingMPOModel.init_terms`.

It is a common pattern to define a model on a general lattice, such as e.g. the
:class:`~tenpy.models.tf_ising.TFIModel`, which works for arbitrary 1D or 2D lattices,
and then define a subclass on a specific lattice such as the :class:`~tenpy.models.tf_ising.TFIChain`.
This subclass then needs no code at all, just setting class attributes to indicate that the lattice
is a chain.

.. literalinclude:: /../tenpy/models/tf_ising.py


Automation of Hermitian conjugation
-----------------------------------
As most physical Hamiltonians are Hermitian, these Hamiltonians are fully determined when only half
of the mutually conjugate terms is defined. For example, a simple Hamiltonian:

.. math ::
        H = \sum_{\langle i,j\rangle, i<j}
              - \mathtt{J} (c^{\dagger}_i c_j + c^{\dagger}_j c_i)

is fully determined by the term :math:`c^{\dagger}_i c_j` if we demand that Hermitian conjugates
are included automatically.
In TeNPy, whenever you add a coupling using :meth:`~tenpy.models.model.CouplingModel.add_onsite_operator`,
:meth:`~tenpy.models.model.CouplingModel.add_two_site_coupling`, or
:meth:`~tenpy.models.model.CouplingModel.add_multi_coupling`,
you can use the optional argument `plus_hc` to automatically create and add the Hermitian conjugate
of that coupling term - as shown above.

Additionally, in an MPO, explicitly adding both a non-Hermitian term and its conjugate increases
the bond dimension of the MPO, which increases the memory requirements of the
:class:`~tenpy.networks.mpo.MPOEnvironment`.
Instead of adding the conjugate terms explicitly, you can set a flag `explicit_plus_hc` in the
:class:`~tenpy.models.model.MPOCouplingModel` parameters, which will ensure two things:

1. The model and the MPO will only store half the terms of each Hermitian conjugate pair added,
   but the flag `explicit_plus_hc` indicates that they *represent* `self + h.c.`.
   In the example above, only the term :math:`c^{\dagger}_i c_j` would be saved.
2. At runtime during DMRG, the Hermitian conjugate of the (now non-Hermitian) MPO will be computed
   and applied along with the MPO, so that the effective Hamiltonian is still Hermitian.

.. note ::

    The model flag `explicit_plus_hc` should be used in conjunction with the flag `plus_hc` in
    :meth:`~tenpy.models.model.CouplingModel.add_two_site_coupling` or
    :meth:`~tenpy.models.model.CouplingModel.add_multi_coupling`.
    If `plus_hc` is `False` while `explicit_plus_hc` is `True` the MPO bond dimension will not be
    reduced, but you will still pay the additional computational cost of computing the Hermitian
    conjugate at runtime.

Thus, we end up with several use cases, depending on your preferences.
Consider the :class:`~tenpy.models.fermions_spinless.FermionModel`.
If you do not care about the MPO bond dimension, and want to add Hermitian conjugate terms manually,
you would set `model_par['explicit_plus_hc'] = False` and write::

    self.add_coupling(-J, 'Cd.C', dx, u1, u2)
    self.add_coupling(np.conj(-J), 'Cd.C', -dx, u1, u2)

If you wanted to save the trouble of the extra line of code (but still did not care about MPO bond
dimension), you would keep the `model_par`, but instead write::

    self.add_coupling(-J, 'Cd.C', dx, u1, u2, plus_hc=True)

Finally, if you wanted a reduction in MPO bond dimension, you would need to set
`model_par['explicit_plus_hc'] = True`, and write::

    self.add_coupling(-J, 'Cd.C', dx, u1, u2, plus_hc=True)


Non-uniform terms and couplings
-------------------------------
The CouplingModel-methods :meth:`~tenpy.models.model.CouplingModel.add_onsite_operator`,
:meth:`~tenpy.models.model.CouplingModel.add_two_site_coupling`,
and :meth:`~tenpy.models.model.CouplingModel.add_multi_coupling` by default add an extensive sum
over the same "coupling" term, shifted by lattice vectors.
However, some models are not that "uniform" over the whole lattice.

First of all, you might have some local term that gets added only at one specific location in the lattice.
The "add" methods have optional positional args (``x`` or ``x1``) that allow you to do this.

Second, if you have irregular lattices, take a look at the corresponding section in :doc:`/intro/lattices`.

Finally, note that the argument `strength` for the "add" methods can not only be a scalar,
but also a sequence (e.g. list of numpy array).
In general, the sum performed by the methods runs over the given term
shifted by lattice vectors *as far as possible to still fit the term into the lattice*.

For the :meth:`~tenpy.models.model.CouplingModel.add_onsite_operator` case this criterion is simple:
there is exactly one site in each lattice unit cell with the `u` specified as separate argument,
so the correct shape for the `strength` array is simply given by :attr:`~tenpy.models.lattice.Lattice.Ls`.
For example, if you want the defacto standard model studied for many-body localization,
a Heisenberg chain with random, uniform onsite field :math:`h^z_i \in [-W, W]`,

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

For :meth:`~tenpy.models.model.CouplingModel.add_two_site_coupling`
and :meth:`~tenpy.models.model.CouplingModel.add_multi_coupling`,
things become a little bit more complicated, and the correct shape of the `strength` array depends
not only on the :attr:`~tenpy.models.lattice.Lattice.Ls`
but also on the boundary conditions of the lattice. Given a term, you can call
:meth:`~tenpy.models.lattice.Lattice.coupling_shape` and
:meth:`~tenpy.models.lattice.Lattice.multi_coupling_shape` to find out the correct shape for `strength`.
To avoid any ambiguity, the shape of the `strength` always has to fit, at least after a tiling
performed by :func:`~tenpy.tools.misc.to_array`.

For example, consider the Su-Schrieffer-Heeger model, a spin-less
:class:`~tenpy.models.fermions.FermionChain` with hopping strength alternating between two values,
say `t1` and `t2`.
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
- If you want to use random parameters, you should use ``model.rng`` as a random number generator
  and change ``model_params['random_seed']`` for different configurations.
- You can initialize a model from only an MPO of the Hamiltonian using the
  :class:`~tenpy.models.model.HamiltonianMPOModel`. This could be useful e.g. if the Hamiltonian
  comes from some other software, and/or from compression methods.
- We suggest writing the model to take a single parameter dictionary for the initialization,
  as the :class:`~tenpy.models.model.ModelBuilder` does.
  The :class:`~tenpy.models.model.ModelBuilder` converts the dictionary to a dict-like
  :class:`~tenpy.tools.params.Config` with some additional features before passing it on to the
  `init_lattice`, `init_site`, ... methods.
  It is recommended to read out providing default values with
  ``model_params.get("key", default_value)``, see :meth:`~tenpy.tools.params.Config.get`.
- When you write a model and want to include a test that it can be at least constructed,
  take a look at ``tests/test_model.py``.
