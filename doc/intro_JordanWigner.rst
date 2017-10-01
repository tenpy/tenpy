Fermions and the Jordan-Wigner transformation
=============================================

The `Jordan-Wigner tranformation <https://en.wikipedia.org/wiki/Jordan-Wigner_transformation>`_
maps spin-operators to fermionic creation and annihilation operators.

Spinless fermions in 1D
-----------------------
Let's start by explicitly writing down the transformation.
With the spin-1/2 operators :math:`\sigma^{x,y,z}_j` and :math:`\sigma^{\pm}_j = \sigma^x_j \pm \mathrm{i} \sigma^y_j` on each site,
we can map

.. math ::
    (\sigma^{z}_j + 1)/2 \leftrightarrow n_j                            \\
    (-1)^{\sum_{l < j} n_l} \sigma^{-}_j \leftrightarrow c_j            \\
    (-1)^{\sum_{l < j} n_l} \sigma^{+}_j \leftrightarrow c_j^\dagger

(The :math:`n_l` in the second and third row are defined in terms of spin operators according to the first row).

Note that this transformation maps :math:`c_j` and :math:`c_j^\dagger` to *global* operators; although they carry an index `j` indicating
a site, they actually act on all sites ``l <= j``!
Thus, clearly the operators ``C`` and ``Cd`` defined in the :class:`~tenpy.networks.site.FermionSite` do *not* directly correspond to :math:`c_j` and
:math:`c_j^\dagger`.
The part :math:`(-1)^{\sum_{l < j} n_l}` is called Jordan-Wigner string and in the :class:`~tenpy.networks.site.FermionSite` given by the local operator ``JW``.
Since this important, let me stress it again:

.. warning ::
    The fermionic operator :math:`c_j` (and similar :math:`c^\dagger,j`) maps to a *global* operator consisting of
    the Jordan-Wigner string build by the local operator ``JW`` on sites ``l < j`` *and* the local operator ``C`` (or ``Cd``, respectively) on site ``j``.

On the sites itself, the onsite operators ``C`` and ``Cd`` in the :class:`~tenpy.networks.site.FermionSite` fulfill the correct anti-commutation relation.
But the ``JW`` string is necessary to ensure the anti-commutation for operators acting on different sites.

Also note that ``JW`` squares to the identity, which is the reason that the Jordan-wigner string completely cancels in :math:`n_j = c_j^\dagger c_j`.
On different sites (say `i` < `j`) we have e.g.
:math:`c_i^\dagger c_j \leftrightarrow \sigma_i^{+} (-1)^{\sum_{i <=l < j} n_l}  \sigma_j^{-}`.
In (other) words, the Jordan-Wigner string appears only in the range ``i <= l < j``, i.e. between the two sites *and* on the smaller/left one of them.
(You can easily generalize this rule to cases with more than two :math:`c` or :math:`c^\dagger`.)


Higher dimensions
-----------------
For an MPO or MPS, you always have to define an ordering of all your sites. This ordering effectifely maps the
higher-dimensional lattice to a 1D chain, usually at the expence of long-range hopping/interactions.
With this mapping, the Jordan-Wigner transformation generalizes to higher dimensions in a straight-forward way.


Spinfull fermions
-----------------
You can think of spin-1/2 fermions on a chain as spinless fermions living on a ladder (and analogous mappings for higher dimensional lattices),
each rung forming a :class:`~tenpy.networks.site.SpinHalfFermionSite` being composed two :class:`~tenpy.networks.site.FermionSite` for each spin-up and spin-down.
More generally, each species of fermions appearing in your model gets a separate label, and its Jordan-Wigner string
includes the signs :math:`(-1)^n_l` of *all* species of Fermions 'left' of it.
In the case of spin-1/2 fermions labeled by :math:`\uparrow` and :math:`\downarrow`, the complete mapping is:

.. math ::
    (\sigma^{z}_{\uparrow,j} + 1)/2 \leftrightarrow n_{\uparrow,j}                                                                                  \\
    (\sigma^{z}_{\downarrow,j} + 1)/2 \leftrightarrow n_{\downarrow,j}                                                                              \\
    (-1)^{\sum_{l < j} n_{\uparrow,j} + n_{\downarrow,j}} \sigma^{-}_{\uparrow,j} \leftrightarrow c_{\uparrow,j}                                    \\
    (-1)^{\sum_{l < j} n_{\uparrow,j} + n_{\downarrow,j}} \sigma^{+}_{\uparrow,j} \leftrightarrow c_{\uparrow,j}^\dagger                            \\
    (-1)^{\sum_{l < j} n_{\uparrow,j} + n_{\downarrow,j}} (-1)^{n_{\uparrow,j}} \sigma^{-}_{\downarrow,j} \leftrightarrow c_{\downarrow,j}          \\
    (-1)^{\sum_{l < j} n_{\uparrow,j} + n_{\downarrow,j}} (-1)^{n_{\uparrow,j}} \sigma^{+}_{\downarrow,j} \leftrightarrow c_{\downarrow,j}^\dagger  \\

All of the operators on the left hand sides above commute; we can rewrite
:math:`(-1)^{\sum_{l < j} n_{\uparrow,l} + n_{\downarrow,l}} = \prod_{l < j} (-1)^{n_{\uparrow,l}} (-1)^{n_{\downarrow,l}}`,
which resembles the actual structure in the code more closely.
The parts of the operator acting on one site (i.e. one index `j` or `l`) are the 'onsite' operators in the :class:`~tenpy.networks.site.SpinHalfFermionSite`,
for example ``JW`` on site `j` is given by :math:`(-1)^{n_{\uparrow,j}} (-1)^{n_{\downarrow,j}}`, ``Cu`` is just the
:math:`\sigma^{-}_{\uparrow,j}` and ``Cd`` is :math:`(-1)^{n_{\uparrow,j}} \sigma^{-}_{\uparrow,j}`.
To summarize:

.. note ::
    Again, the fermionic operators :math:`c_{\downarrow,j}, c^\dagger_{\downarrow,j}, c_{\downarrow,j}, c^\dagger_{\downarrow,j}` correspond to  *global* operators consisting of
    the Jordan-Wigner string build by the local operator ``JW`` on sites ``l < j`` *and* the local operators ``'Cu', 'Cud', 'Cd', 'Cdd'`` on site ``j``.


How to handle Jordan-Wigner strings in practice
-----------------------------------------------
There are a only few pitfalls where you have to keep the mapping in mind:
When **building a model**, you map the physical fermionic operators to the usual spin/bosonic operators.
The algorithms don't care about the mapping, they just use the given Hamiltonian, be it given as MPO for DMRG or as nearest neighbor couplings for TEBD.
Only when you do a **measurement** (e.g. by calculating an expectation value or a correlation function), you have to reverse this mapping.
Be aware that in certain cases, e.g. when calculating the entanglement entropy on a certain bond,
you cannot reverse this mapping (in a straightforward way), and thus your results might depend on how you defined the Jordan-Wigner string.

Whatever you do, you should first think about if (and how much of) the Jordan-Wigner string cancels.
For example for many of the onsite operators (like the particle number operator ``N`` or the spin operators in the :class:`~tenpy.networks.site.SpinHalfFermionSite`)
the Jordan-Wigner string cancels and you can just ignore it both in onsite-terms and couplings.
The case that the operator string extends on the left is currently not really supported.


When **building a model** with the :class:`~tenpy.models.model.CouplingModel`,
*onsite* terms for which the Jordan-Wigner string cancels can be added directly.
Care has to be taken when adding *couplings* with :meth:`~tenpy.models.model.CouplingModel.add_coupling`.
When you need a Jordan-Wigner string inbetween the operators, set the optional arguments ``op_string='JW', str_on_first=True``.
Then, the function automatically takes care of the Jordan-Wigner string in the correct way, adding it on the left
operator.

Obviously, you should be careful about the convention which of the two coupling terms is applied first (in a physical
sense as an operator acting on a state), as this corresponds to a sign. We follow the convention that the operator given
as argument `op2` is applied first, independent of wheter it ends up left or right in the MPS ordering sense.

As a concrete example, let us specify a hopping
:math:`\sum_{\langle i, j\rangle} (c^\dagger_i c_j + h.c.) = \sum_{\langle i, j\rangle} (c^\dagger_i c_j + c^\dagger_j c_i)`
in a 1D chain of :class:`~tenpy.networks.site.FermionSite` with :meth:`~tenpy.models.model.CouplingModel.add_coupling`::

    add_coupling(strength, 0, 'Cd', 0, 'C', 1, 'JW', True)
    add_coupling(strength, 0, 'Cd', 0, 'C', -1, 'JW', True)

Slightly more complicated, to specify the hopping
:math:`\sum_{\langle i, j\rangle, s} (c^\dagger_{s,i} c_{s,j} + h.c.)`
in the Fermi-Hubbard model on a 2D square lattice, we would need more terms::

    for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        add_coupling(strength, 0, 'Cud', 0, 'Cu', (dx, dy), 'JW', True)
        add_coupling(strength, 0, 'Cdd', 0, 'Cd', (dx, dy), 'JW', True)

If you want to build a model directly as an MPO or with nearest-neighbor bonds, you have to worry yourself about how to handle the Jordan-Wigner string correctly.


The most important functions for doing **measurements** are probably :meth:`~tenpy.networks.mps.MPS.expectation_value`
and :meth:`~tenpy.networks.mps.MPS.correlation_function`. Again, if all the Jordan-Wigner strings cancel, you don't have
to worry about them at all, e.g. for many onsite operators or correlation functions involving only number operators.
If you measure operators involving multiple sites with `expectation_value`, take care to include the Jordan-Wigner
string correctly while building these operators.
The :meth:`~tenpy.networks.mps.MPS.correlation_function` supports an Jordan-Wigner string inbetween the two operators to
be measured; as for :meth:`~tenpy.models.model.CouplingModel.add_coupling`, you should set the optional arguments ``op_string='JW', str_on_first=True`` in that case.
