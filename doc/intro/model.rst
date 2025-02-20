Models (Introduction)
=====================

This is an introduction to models in TeNPy, intended to guide new-comers towards defining
their own custom models.

We go step by step to introduce the relevant concepts and "derive" how you could have come up
with the following example code to implement an anisotropic Heisenberg model on a square lattice::

    class MyModel(CouplingMPOModel):
        r"""The anisotropic spin-1/2 Heisenberg model in an external field.

        This is a pedagogical example, and you should probably use the SpinModel instead.
        The Hamiltonian is:

        .. math ::
            H = J_x \sum_i (S^x_i S^x_{i+1} + S^y_i S^y_{i+1})
                + J_z \sum_i S^z_i S^z_{i+1} - h \sum_i S^z_i

        """

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
We follow the most direct route, using the :class:`~tenpy.models.model.CouplingMPOModel` framework,
for more flexible alternatives see :doc:`/intro/model_details`.

.. note ::
    This Hamiltonian is already implemented as one of the pre-defined models.
    Here, we implement it from scratch as an example, but if you wanted to simulate this particular
    Hamiltonian in practice, you would use the :class:`~tenpy.models.spins.SpinModel`, which has a
    more general Hamiltonian and contains our example as a special case.


The first step is to identify what the **parameters** of your model are.
In this case, we have the coupling constants :math:`J_x, J_z, h`, and parameters that specify
the lattice geometry (discussed later). In the TeNPy ecosystem, these parameters are
gathered into dictionary-like :class:`~tenpy.tools.params.Config` objects, and for the rest of this
guide you can think of ``model_params`` as a dictionary of these parameters, e.g.
``model_params = {'Jx': 0.5, 'Jz': 1}``.
It is common practice to make all parameters optional, in which case you should think about
(and ideally document) the default values for the parameters.

We start implementing our custom model by defining a class for it::

    class MyModel(CouplingMPOModel):
        r"""The anisotropic spin-1/2 Heisenberg model in an external field.

        This is a pedagogical example, and you should probably use the SpinModel instead.
        The Hamiltonian is:

        .. math ::
            H = J_x \sum_i (S^x_i S^x_{i+1} + S^y_i S^y_{i+1})
                + J_z \sum_i S^z_i S^z_{i+1} - h \sum_i S^z_i

        """
        pass  # content will be added later


Note that we define our model as a subclass of :class:`~tenpy.models.model.CouplingMPOModel`.
This means our model inherits all the machinery to build Hamiltonians etc, and we only need
to implement the code that is specific to our model.


The local Hilbert space
-----------------------
The **local Hilbert** space is represented by a :class:`~tenpy.networks.site.Site` (read its doc-string!).
A site defines the meaning of each basis state (i.e. by fixing an order, to define e.g.
that the state are ``spin_down, spin_up``). Additionally, it stores common local operators, such as
:math:`S^z` and makes them accessible by name.

We need to tell our model, what its local Hilbert space is.
This is done by implementing the :meth:`~tenpy.models.model.CouplingMPOModel.init_sites` method.
It needs to take the ``model_params`` as input and returns one :class:`~tenpy.networks.site.Site`
instance per site in the unit cell of the lattice (see lattice section below, here this is one site).
The most common sites -- e.g. for spins, spin-less or spin-full fermions, or bosons -- are predefined
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
    section. If you need a custom site, you can use 
    ``leg = tenpy.linalg.np_conserved.LegCharge.from_trivial(d)`` for an implementation of your
    site, where `d` is the dimension of the local Hilbert space.


In many cases, the possible symmetries we may exploit depend on the
values of the parameters, which is why they are an input to ``init_sites``.
In our example, we can conserve :math:`S^z` if :math:`J = 0`, and only its parity otherwise.::

    class MyModel(CouplingMPOModel):

        def init_sites(self, model_params):
            conserve = model_params.get('conserve', 'best')
            if conserve == 'best':
                if model_params.get('Jx', 1) == 0:
                    conserve = 'Sz'
                else:
                    conserve = 'parity'
            return SpinHalfSite(conserve=conserve)


Note that we added ``conserve`` as a model parameter, such that we can later turn charge
conservation on or off. The possible values for ``conserve`` are documented in the site class,
here :class:`~tenpy.networks.site.SpinHalfSite`, and it is common to support ``'best'``
as a value for the ``conserve`` model parameter and translate it to the largest possible symmetry,
given the values of the coupling strengths.

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
The geometry is usually given by some kind of **lattice** structure that determines how the sites
are arranged spatially. This implicitly defines e.g. the meaning of a sum over nearest neighbors
:math:`\sum_{<i, j>}`.
In TeNPy, this is specified by a :class:`~tenpy.models.lattice.Lattice` class, which contains a unit cell of
a few :class:`~tenpy.networks.site.Site`\s which are repeated periodically according to the lattice
basis vectors, to form a regular lattice.
Again, we have pre-defined some basic lattices like a :class:`~tenpy.models.lattice.Chain`,
two chains coupled as a :class:`~tenpy.models.lattice.Ladder` or 2D lattices like the
:class:`~tenpy.models.lattice.Square`, :class:`~tenpy.models.lattice.Honeycomb` and
:class:`~tenpy.models.lattice.Kagome` lattices; but you are also free to define your own generalizations.
See :doc:`/intro/lattices`.


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


Setting ``default_lattice = Kagome`` means that the lattice defaults to Kagome, if ``'lattice' not in model_params``,
while setting ``force_default_lattice = True`` means that this model does not allow any other
lattice. Thus, ``MyModelKagome`` does what its name promises to do.

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

- :meth:`~tenpy.models.model.CouplingModel.add_onsite` for a sum of onsite terms :math:`\sum_i h_i \hat{A}_i`.
- :meth:`~tenpy.models.model.CouplingModel.add_coupling` for a sum of two-body couplings :math:`\sum_i J_i \hat{A}_i \hat{B}_{i+n}`.
- :meth:`~tenpy.models.model.CouplingModel.add_multi_coupling` for a sum of multi-body couplings :math:`\sum_i J_i \hat{A}_i \hat{B}_{i+n} ... \hat{F}_{i+m}`.

.. note ::
    A single call to each of these methods adds an extensive number of terms to your Hamiltonian,
    as it includes a sum over all sites in the definition.
    This means that a Hamiltonian like :math:`H = -3 \sum_i S_i^z` is realized as a **single** call to 
    :meth:`~tenpy.models.model.CouplingModel.add_onsite`, **without**  an explicit loop over `i`.

.. note ::
    These methods allow the prefactors to be site-dependent; you can either give a single number
    as the prefactor, or a list/array that is tiled to fit the size.
    E.g. if an onsite term with ``strength=1`` gives you a uniform magnetic field,
    ``strength=[1, -1]`` gives you the corresponding staggered field,
    assuming a chain of even length.

- :meth:`~tenpy.models.model.CouplingModel.add_local_term` for a single term :math:`\hat{A}_i` or :math:`\hat{A}_i \hat{B}_{j}`
  or :math:`\hat{A}_i \hat{B}_{j} ... \hat{F}_k`.

.. warning ::
    You probably should not directly use :meth:`~tenpy.models.model.CouplingModel.add_onsite_term`,
    :meth:`~tenpy.models.model.CouplingModel.add_coupling_term` and
    :meth:`~tenpy.models.model.CouplingModel.add_multi_coupling_term`.
    They do not handle Jordan-Wigner strings and they need MPS indices as inputs, not
    lattice positions.

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


To verify that you have added the correct terms, initialize the model on a small lattice (we also
set :math:`J_x=0` here for readability, but you should turn it on to verify the full model),
e.g.::

    model = MyModel({'lattice': 'Square', 'Lx': 2, 'Ly': 3, 'Jx': 0})


Now, print all couplings in the model to console::

    print(model.all_coupling_terms().to_TermList())


Which gives you the following output for our example::

    1.00000 * Sz_0 Sz_1 +
    1.00000 * Sz_0 Sz_2 +
    1.00000 * Sz_0 Sz_3 +
    1.00000 * Sz_1 Sz_2 +
    1.00000 * Sz_1 Sz_4 +
    1.00000 * Sz_2 Sz_5 +
    1.00000 * Sz_3 Sz_4 +
    1.00000 * Sz_3 Sz_5 +
    1.00000 * Sz_4 Sz_5


You may be surprised to get nine different couplings on this ``2 x 3`` square patch.
Let us look at the couplings in detail to figure out why this might be.
We need to understand the meaning of the site indices, i.e. where does ``Sz_4`` live spatially?
The convention for site indices comes from the MPS geometry and may be hard to read.
To visualize the site order of the lattice, you may run the following snippet::

    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 6))
    ax = plt.gca()
    model.lat.plot_coupling(ax)
    model.lat.plot_sites(ax)
    model.lat.plot_order(ax)
    plt.show()


.. plot ::

    from tenpy import SpinHalfSite
    from tenpy.models import CouplingMPOModel
    import matplotlib.pyplot as plt
    
    class MyModel(CouplingMPOModel):
        def init_sites(self, model_params):
            return SpinHalfSite()
        def init_terms(self, model_params):
            self.add_onsite(-1, 0, 'Sz')  # terms dont matter for this plot :: simplify

    model = MyModel({'lattice': 'Square', 'Lx': 2, 'Ly': 3, 'Jx': 0})
    plt.figure(figsize=(5, 6))
    ax = plt.gca()
    model.lat.plot_coupling(ax)
    model.lat.plot_sites(ax)
    model.lat.plot_order(ax)
    plt.show()


We see the lattice plotted in black. Concretely, we get a black line for each pair of nearest-neighbor
sites. The red line goes through the sites in order, and we see the site indices labelled.

In particular, we can now understand why we get nine different couplings; we see from the plot
that the lattice has open boundaries in x-direction but periodic boundaries in y-direction.
Try playing around with different boundary conditions, e.g.
``MyModel({'lattice': 'Square', 'Lx': 2, 'Ly': 3, 'Jx': 0, 'bc_y': 'open'})``
or
``MyModel({'lattice': 'Square', 'Lx': 2, 'Ly': 3, 'Jx': 0, 'bc_x': 'periodic'})``.
See :doc:`/intro/lattices` regarding boundary conditions.


Contribute your model?
----------------------
If you have implemented a model, and you think it may be useful to the broader community, consider
contributing it to TeNPy via a pull request.
We have :doc:`/contr/guidelines`, and you can have a look at the implementation
of e.g. the :class:`~tenpy.models.spins.SpinModel` as a guide, but do not let formalities
stop you from sharing your code, we can always address any nitpicks ourselves.


Further Reading
---------------
- Details and ideas behind the implementation: :doc:`/intro/model_details`
- Look at the implementation of the pre-defined models in :mod:`tenpy.models`.
  Most are based on the :class:`~tenpy.models.model.CouplingMPOModel` as discussed here.
- The :class:`~tenpy.models.aklt.AKLTChain` is a notable counter-example where it is actually
  easier to define ``H_bond`` than to write down couplings.
- If the Hamiltonian is already given in MPO form (e.g. because it comes from some other software),
  it can be used to directly build a model, as is done in ``examples/mpo_exponentially_decaying.py``.
