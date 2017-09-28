Fermions and the Jordan-Wigner transformation
=============================================

The `Jordan-Wigner tranformation <https://en.wikipedia.org/wiki/Jordan-Wigner_transformation>`_
maps spin-operators to fermionic creation and annihilation operators.

Spinless fermions in 1D
-----------------------
Let's start by explicitly writing down the transformation.
With the Pauli matrices :math:`\sigma^{x,y,z}_j` and :math:`\sigma^{\pm}_j = \sigma^x_j \pm \mathrm{i} \sigma^y_j` on each site,
we can map

.. math ::
    \sigma^{z}_j \leftrightarrow 2 n_j - 1                              \\
    (\sigma^{z}_j + 1)/2 \leftrightarrow n_j                            \\
    (-1)^{\sum_{l < j} n_l} \sigma^{+}_j \leftrightarrow c_j            \\
    (-1)^{\sum_{l < j} n_l} \sigma^{-}_j \leftrightarrow c_j^\dagger

Note that this makes :math:`c_j` and :math:`c_j^\dagger` *global* operators; although they carry an index `j` indicating
a site, they actually act on all sites ``l <= j``!
Thus, clearly the operators ``C`` and ``Cd`` defined in the :class:`~tenpy.networks.site.FermionSite` do *not* directly correspond to :math:`c_j` and
:math:`c_j^\dagger`.
The part :math:`(-1)^{\sum_{l < j} n_l}`` is called Jordan-Wigner string and in the :class:`~tenpy.networks.site.FermionSite` given by the local operator ``JW``.
Since this important, let me stress it again:

.. warning ::
    The fermionic operator :math:`c_j` corresponds to a *global* operator consisting of the Jordan-Wigner string build by the local operator ``JW`` on sites ``l < j`` and the local operator ``C`` on site ``j``.
    Similar, :math:`c_j^\dagger` corresponds to ``JW`` on sites ``l < j`` and ``C`` on site ``j``.

On the sites itself, ``C`` and ``Cd`` fulfill the correct anti-commutation relation, but the ``JW`` string is necessary
to ensure the anti-commutation between different sites.

Also note that ``JW`` squares to the identity. Thus, the Jordan-wigner string cancels in :math:`n_j = c_j^\dagger c_j`
completely. On different sites (say `i` < `j`) we have e.g.
:math:`c_i^\dagger c_j \leftrightarrow \sigma_i^{+} (-1)^{\sum_{i <=l < j} n_l}  \sigma_j^{-}`, i.e. the Jordan-Wigner
string appears only in the range ``i <= l < j``, i.e. between the two sites *and* on the smaller one of them.

.. todo ::
    discuss correct use of ``JW`` in CouplingModel/MPOs, MPS.expectation_value, MPS.corralation_function


Higher dimensions
-----------------
For an MPO or MPS, you always have to define an ordering of all your sites. This ordering effectifely maps the
higher-dimensional lattice to a 1D chain (with possibly longer-range interactions). With this mapping
the Jordan-Wigner transformation generalizes to higher dimensions in a straight-forward way.


Spinfull fermions
-----------------

.. todo :: write.
