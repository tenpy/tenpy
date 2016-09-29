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
A :class:`~tenpy.linalg.charges.LegCharge` has also a flag **qconj**, which tells whether the charges
point *inward* (+1) or *outward* (-1). What that means, is explained later in :ref:`nonzero_entries`.

For completeness, let us also summarize also the internal structure of an :class:`~tenpy.linalg.np_conserved.Array` here:
The array saves only non-zero blocks, collected as a list of `np.array` in ``self._data``.
The qindices necessary to map these blocks to the original leg indices are collected in ``self._qdat``
An array is said to be **qdat-sorted** if its ``self._qdat`` is lexiographically sorted.
More details on this follow later. However, note that you usually shouldn't access `_qdat` and `_data` directly - this
is only necessary from within `tensordot`, `svd`, etc.
Also, an array has a **total charge**, defining which entries can be non-zero - details in :ref`nonzero_entries`.

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


.. _nonzero_entries

Which entries of the npc array can be non-zero?
-----------------------------------------------
The reason for the speedup with np_conserved lies in the fact
that it saves only the blocks 'compatible' with the charges. 
But how is this 'compatible' defined? 

Assume you have a tensor, call it :math:`T`, and the :class:`~tenpy.linalg.charges.LegCharge` for all of its legs, say :math:`a, b, c, ...`.

Remeber that the LegCharge associates to each index of the leg a charge value (for each of the charges, if `qnumber` > 1).
Let ``a.to_qflat()[ia]`` denote the charge(s) of index ``ia`` for leg ``a``, and similar for other legs.

In addition, the LegCharge has a flag :attr:`~tenpy.linalg.charges.LegCharge.qconj`. This flag **qconj** is only a sign,
saved as +1 or -1, specifying whether the charges point inward (+1, default) or outward (-1) of the tensor.

Then, the **total charge** of a single entry ``T[ia, ib, ic, ...]`` of the tensor is defined as::

   qtotal[ia, ib, ic, ...] = a.to_qflat()[ia] * a.qconj + b.to_qflat()[ib] * b.qconj + c.to_qflat()[ic] * c.qconj + ...  modulo qmod

In case of multiple charges, ``qnumber`` > 1, this equation holds for each of the different charges individually with the
corresponding ``qmod`` of the charge.

The rule which entries of the a :class:`~tenpy.linalg.np_conserved.Array` can be non-zero
(i.e., are 'compatible' with the charges), is then very simple:

.. topic :: Rule for non-zero entries

    An entry ``ia, ib, ic, ...`` of a :class:`~tenpy.linalg.np_conserved.Array` can only be non-zero,
    if ``qtotal[ia, ib, ic, ...]`` matches the :attr:`~tenpy.linalg.np_conserved.qtotal` attribute of the class.

Again, this must hold for each of the charges seperately in the case ``qnumber`` > 1.

The pesky qconj - contraction as an example
-------------------------------------------
Why did we introduce the ``qconj`` flag? Remember it's just a sign telling whether the charge points inward or outward.
So whats the reasoning?

The short answer is, that LegCharges actually live on bonds (i.e., legs which are to be contracted) 
rather than individual tensors. Thus, it is convenient to share the LegCharges between different legs and even tensors, 
and just adjust the sign.

As an example, consider the contraction of two tensors, :math:`C_{ia,ic} = \sum_{ib} A_{ia,ib} B_{ib,ic}`.
For simplicity, say that the total charge of all three tensors is zero.
What are the implications of the above rule for non-zero entries?
Or rather, how can we ensure that ``C`` complies with the above rule?
An entry ``C[ia,ic]`` will only be non-zero, 
if there is an ``ib`` such that both ``A[ia,ib]`` and ``B[ib,ic]`` are non-zero, i.e., both of the following equations are
fullfilled::

   A.qtotal == A.a.to_qflat()[ia] A.a.qconj_a + A.b.to_qflat()[ib] A.b.qconj  modulo qmod
   B.qtotal == B.b.to_qflat()[ib] B.b.qconj_b + B.c.to_qflat()[ic] B.c.qconj  modulo qmod

Here, the ``A.a`` should denotes the LegCharges for leg ``a`` of the tensor -- it is not directly accessible as an
attribute.

For the uncontracted legs, we just keep the charges as they are::

   C.a.qind = A.a.qind
   C.a.qconj = A.a.qconj
   C.c.qind = B.c.qind
   C.c.qconj = B.c.qconj

It is then straight-forward to check, that the rule is fullfilled for :math:`C`, if the following condition is met::

   A.qtotal + B.qtotal - C.qtotal == A.b.to_qflat()[ib] A.b.qconj + B.b.to_qflat()[ib] B.b.qconj  modulo qmod

The easiest way to meet this condition is, if ``A.b`` and ``B.b`` share the *same* charges ``b.to_qflat()``, but have
opposite ``qconj``, and defining ``C.qtotal = A.qtotal + B.qtotal``.
This justifies the introduction of ``qconj``:
when you define the tensors, you have to define the :class:`~tenpy.linalg.charges.LegCharge` only once, say ``A.b``.
For ``B.b`` you simply use ``A.b.conj()`` - this creates a copy with shared ``qind``, but opposite ``qconj``.
Or, as a more impressive example, all 'physical' legs of an MPS can usually share the same
:class:`~tenpy.linalg.charges.LegCharge`.


Assigning charges to non-physical legs
--------------------------------------
From the above physical examples, it should be clear how you assign charges to physical legs.
But what about other legs, e.g, the virtual bond of an MPS? 

The charge of these bonds must be derived by using the 'rule for non-zero entries', as far as they are not arbitrary.
As a concrete example, consider an MPS on just two spin 1/2 sites::

    |        _____         _____
    |   x->- | A | ->-y->- | B | ->-z
    |        -----         -----
    |          ^             ^
    |          |a            |b

The legs ``a`` and ``b`` are physical, say with indices :math:`\uparrow = 0` and :math:`downarrow = 1`.
As noted above, we can associate the charges 1 (up) and 0 (down), respectively.

The legs ``x`` and ``z`` are 'dummy' indices with just one index ``0``.
The charge on one of them, as well as the total charge of both ``A`` and ``B`` is somewhat arbitrary, so we make a simple choice: 
total charge 0 on both arrays, as well as charge 0 for `x` = 0.

Finally, we also have to define ``qconj`` values. We stick to the convention used in our MPS code: physical
legs incoming (qconj=1), and from left to right on the virtual bonds.

The charges on the bonds `y` and `z` then depend on the state the MPS represents.
Here, we consider a singlet as a the simplest non-trivial example.
A possible MPS representation is given by::

    A[up]   = [[1, 0]]     B[up]   = [[0], [-1]]
    A[down] = [[0, 1]]     B[down] = [[1], [0]]

There are two non-zero entries in ``A``, for the indices :math:`(a, x, y) = (\uparrow, 0, 0)` and :math:`(\downarrow, 0, 1)`.
To comply with the rules for non-zero entries, we then have to assign the charge 1 to `y` = 0, and the charge 0 to `y` = 1.
Again, we associate the same charge values of `y` to the ``A`` and ``B``, and just change the ``qconj``.
The non-zero entry :math:`(b, y, z) = (\uparrow, 1, 0)` then implies the charge 0 for `z` = 0.
Note, that the rule for :math:`(b, y, z) = (\downarrow, 0, 0)` is then automatically fullfilled:
this is an implication of the fact that the singlet has a well defined value for :math:`S^z_a + S^z_b`.
For other states without fixed magnetization (e.g., :math:`|\uparrow \upparrow> + |\downarrow \downarrow>`)
we could not use the charge conservation.


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
