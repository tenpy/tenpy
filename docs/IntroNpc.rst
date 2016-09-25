Introduction to np_conserved
============================

The basic idea is quickly summarized:
By inspecting the Hamiltonian, you can identify symmetries, which correspond to conserved quantities, called **charges**.
These charges divide the tensors into different sectors. This can be used to infer for example a block-diagonal structure
of certain matrices, which in turn speeds up SVD or diagonalization a lot.


Notations
---------
Lets fix the notation for this introduction and the doc-strings in np_conserved.

A :class:`~tenpy.linalg.np_conserved.Array` is a multi-dimensional array representing a **tensor** with the entries:

.. math ::
   T_{a_0, a_1, ... a_{rank-1}} \quad with \quad a_i \in \lbrace 0, ..., n_i-1 \rbrace

Each **leg** :math:`a_i` corresponds the a vector space of dimension `n_i`.

An **index** of a leg is a particular value :math:`a_i \in \lbrace 0, ... ,n_i-1\rbrace`.

The **rank** is the number of legs, the **shape** is :math:`(n_0, ..., n_{rank-1})`.

We restrict ourselfes to abelian charges with entries in :math:`\mathbb{Z}` or in :math:`\mathbb{Z}_m`.
The nature of a charge is specified by :math:`m`; we set :math:`m=1` for charges corresponding to :math:`\mathbb{Z}`.
The number of charges, and their :math:`m` (and possibly descriptive names) are saved in 
an instance of :class:`~tenpy.linalg.charges.ChargeInfo`.

To each index of each leg, a value of the charge(s) is associated.
A **block** is a contiguous slice corresponding to the same charge(s) of the leg.
A **qindex** is an index in the list of blocks for a certain leg.
A **charge sector** is for given charge(s) is the set of all qindices of that charge(s).
A leg is **blocked** if all charge sectors are contiguous (i.e., consist only of a single qindex); similarly the array
as whole is blocked, if all legs are blocked.

The charge data for a single leg is collected in the class :class:`~tenpy.linalg.charges.LegCharge`.

Physical Example
----------------
For concreteness, you can think of the Hamiltonian :math:`H = -t \sum_{<i,j>} (c^\dagger_i c_j + H.c.) + U n_i n_j` 
with :math:`n_i = c^\dagger_i c_i`.
This Hamiltonian has the global :math:`U(1)` gauge symmetry :math:`c_i \rightarrow c_i e^i\phi`.
The corresponding charge is the total number of particles :math:`N = \sum_i n_i` [1].

As a second example, consider BCS terms :math:`\sum_k (c^\dagger_k c^\dagger_{-k} + H.c.)`.
These terms break the total particle conservation,
but they preserve the total parity, i.e.,  :math:`N % 2` is conserved.

More details on the charge structure
------------------------------------



See also
--------
- The module :mod:`tenpy.linalg.np_conserved` should contain all the API needed 
  from the point of view of the algorithms.
  It contians the fundamental :class:`~tenpy.linalg.np_conserved.Array` class and functions
  for working with them (creating and manipulating).
- The module :mod:`tenpy.linalg.charges` contains the implementations of the classes 
  :class:`~tenpy.linalg.charges.ChargeInfo` and :class:`~tenpy.linalg.charges.LegCharge`.

References
----------
[1] Schollwöck: DMRG in the age of MPS
