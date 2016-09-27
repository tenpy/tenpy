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
The number of charges is refered to as **qnumber** as a short hand, and the collection of :math:`m` for each charge is called **qmod**.
The qnumber, qmod and possibly descriptive names of the charges are saved in an instance of :class:`~tenpy.linalg.charges.ChargeInfo`.

To each index of each leg, a value of the charge(s) is associated.
A **block** is a contiguous slice corresponding to the same charge(s) of the leg.
A **qindex** is an index in the list of blocks for a certain leg.
A **charge sector** is for given charge(s) is the set of all qindices of that charge(s).
A leg is **blocked** if all charge sectors are contiguous (i.e., consist only of a single qindex).
Finally, a leg is **sorted**, if the charges are sorted lexiographically.
Note that a `sorted` leg is always `blocked`.
We can also speak of the complete array to be blocked or **legcharge-sorted**,  which means that all of its legs are blocked or sorted, respectively.
The charge data for a single leg is collected in the class :class:`~tenpy.linalg.charges.LegCharge`.

For completeness, let us also summarize also the internal structure of an :class:`~tenpy.linalg.np_conserved.Array` here:
The array saves only non-zero blocks, collected as a list of `np.array` in ``self._data``.
The qindices necessary to map these blocks to the original leg indices are collected in ``self._qdat``
An array is said to be **qdat-sorted** if its ``self._qdat`` is lexiographically sorted.
More details on this follow later. However, note that you usually shouldn't access `_qdat` and `_data` directly - this
is only necessary from within `tensordot`, `svd`, etc.

Finally, a **leg pipe** (implemented in :class:`~tenpy.linalg.charges.LegPipe`)
is used to formally combine multiple legs into one leg. Again, more details follow later.

Physical Example
----------------
For concreteness, you can think of the Hamiltonian :math:`H = -t \sum_{<i,j>} (c^\dagger_i c_j + H.c.) + U n_i n_j` 
with :math:`n_i = c^\dagger_i c_i`.
This Hamiltonian has the global :math:`U(1)` gauge symmetry :math:`c_i \rightarrow c_i e^i\phi`.
The corresponding charge is the total number of particles :math:`N = \sum_i n_i` [1].
You would then introduce one charge with :math:`m=1`.

Note that the total charge is a sum of local terms, living on single sites.
Thus, you can infer the charge of a single physical site: it's just the value :math:`q_i = n_i \in \mathbb{N}` for each of the states.

Note that you can only assign integer charges. Consider for example the spin 1/2 Heisenberg chain.
Here, you can naturally identify the magnetization :math:`S^z = \sum_i S^z_i` as the conserved quantity, 
with values :math:`S^z_i = \pm \frac{1}{2}`. 
Obviously, if :math:`S^z` is conserved, then so is :math:`2 S^z`, so you can use the charges
:math:`q_ = 2 S^z_i \in \lbrace-1, +1 \rbrace` for the `down` and `up` states, respectively.
Alternatively, you can also use a shift and define :math:`q_i = S^z_i + \frac{1}{2} \in \lbrace 0, 1 \rbrace`.

As another example, consider BCS like terms :math:`\sum_k (c^\dagger_k c^\dagger_{-k} + H.c.)`.
These terms break the total particle conservation,
but they preserve the total parity, i.e., :math:`N % 2` is conserved. Thus, you would introduce a charge with :math:`m = 2` in this case.

In the above examples, we had only a single charge conserved at a time, but you might be lucky and have multiple
conserved quantities, e.g. if you have two chains coupled only by interactions. 
TenPy is designed to handle the general case of multiple charges.
When giving examples, we will restrict to one charge, but everything generalizes to multiple charges.

The different formats for LegCharge
-----------------------------------
As mentioned above, we assign charges to each index of each leg of a tensor.
This can be done in three formats: **q_flat**, as **q_ind** and as **q_dict**.
Let me explain them with examples, for simplicity considereing only a single charge (the most inner array has one entry
for each charge).

**qflat** form: simply a list of charges for each index. An example::

        qflat = [[-2], [-1], [-1], [0], [0], [0], [0], [3], [3]]

    This tells you that the leg has size 9, the charges for are ``[-2], [-1], [-1], ..., [3]`` for the indices ``0, 1, 2, 3,..., 8``.
    There are four charge blocks (with charges ``[-2], [-1], [0], [3]``), 
    and the qindex (``0, 1, 2, 3``) just enumerates these blocks. 

**qind** form: a table of slices (first two columns) and charges (remaining columns) for each qindex.
    In that way, qind is a map from the qindices (rows) to slice/charges (colum) on the leg.
    The first two columns specify `start` and `stop` of slices, the remaining `ChargeInfo.number` columns are the charge for
    that block. For the above example, you would have::

        qind = np.array([[0, 1, -2],
                         [1, 3, -1],
                         [3, 7,  0],
                         [7, 9,  3])

    By convention, qind should be sorted such that the slices are continuous, i.e., ``qind[i, 1] == qind[i+1, 0]``.
    Here, you can directly read of the blocks using the first two columns.

**qdict** form: a dictionary in the other direction as qind, taking charge tuples to slices.
    Again for the same example::

        {(-2,): slice(0, 1),
         (-1,): slice(1, 3),
         (0,) : slice(3, 7),
         (3,) : slice(7, 9)}

    Since the keys of a dictionary are unique, this includes all indices only if the leg is completely `blocked`.

    
The :class:`~tenpy.linalg.charges.LegCharge` uses saves the charge data of a leg internally in qind form.
It also provides convenient functions for conversion between from and to the flat and dict form.

Assigning charges to non-physical legs and the pesky LegCharge.conj()
---------------------------------------------------------------------
From the above physical charges, it should be clear how you assign charges to physical legs.
But what about other legs?
But what about the virtual bonds of an MPS? For simplicity, conside




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
