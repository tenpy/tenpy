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


Further Reading
---------------
- Details and ideas behind the implementation: :doc:`intro/model_details`
- Look at the implementation of the pre-defined models in :mod:`tenpy.models`.
  Most are based on the :class:`~tenpy.models.model.CouplingMPOModel` as discussed here.
- The :class:`~tenpy.models.aklt.AKLTChain` is a notable counter-example where it is actually
  easier to define ``H_bond`` than to write down couplings.
- If the Hamiltonian is already given in MPO form (e.g. because it comes from some other software),
  it can be used to directly build a model, as is done in ``examples/mpo_exponentially_decaying.py``.
