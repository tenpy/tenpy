Charge conservation with np_conserved
=====================================

The basic idea is quickly summarized:
By inspecting the Hamiltonian, you can identify symmetries, which correspond to conserved quantities, called **charges**.
These charges divide the tensors into different sectors. This can be used to infer for example a block-diagonal structure
of certain matrices, which in turn speeds up SVD or diagonalization a lot.
Even for more general (non-square-matrix) tensors, charge conservation imposes restrictions which blocks of a tensor can
be non-zero. Only those blocks need to be saved, which ultimately (= for large enough arrays) leads to a speedup of many routines, e.g., tensordot.

This introduction covers our implementation of charges; explaining mathematical details of the underlying symmetry is beyond its scope.
We refer you to the corresponding chapter in our [TeNPyNotes]_ for a more general introduction of the idea (also stating
the "charge rule" introduced below).
:cite:`singh2010` explains why it works form a mathematical point of view, :cite:`singh2011` has the focus on a :math:`U(1)` symmetry and might be easier to read.

Notations
---------
Lets fix the notation of certain terms for this introduction and the doc-strings in :mod:`~tenpy.linalg.np_conserved`.
This might be helpful if you know the basics from a different context.
If you're new to the subject, keep reading even if you don't understand each detail,
and come back to this section when you encounter the corresponding terms again.

A :class:`~tenpy.linalg.np_conserved.Array` is a multi-dimensional array representing a **tensor** with the entries:

.. math ::
   T_{a_0, a_1, ... a_{rank-1}} \quad \text{ with } \quad a_i \in \lbrace 0, ..., n_i-1 \rbrace

Each **leg** :math:`a_i` corresponds the a vector space of dimension `n_i`.

An **index** of a leg is a particular value :math:`a_i \in \lbrace 0, ... ,n_i-1\rbrace`.

The **rank** is the number of legs, the **shape** is :math:`(n_0, ..., n_{rank-1})`.

We restrict ourselfes to abelian charges with entries in :math:`\mathbb{Z}` or in :math:`\mathbb{Z}_m`.
The nature of a charge is specified by :math:`m`; we set :math:`m=1` for charges corresponding to :math:`\mathbb{Z}`.
The number of charges is refered to as **qnumber** as a short hand, and the collection of :math:`m` for each charge is called **qmod**.
The qnumber, qmod and possibly descriptive names of the charges are saved in an instance of :class:`~tenpy.linalg.charges.ChargeInfo`.

To each index of each leg, a value of the charge(s) is associated.
A **charge block** is a contiguous slice corresponding to the same charge(s) of the leg.
A **qindex** is an index in the list of charge blocks for a certain leg.
A **charge sector** is for given charge(s) is the set of all qindices of that charge(s).
A leg is **blocked** if all charge sectors map one-to-one to qindices.
Finally, a leg is **sorted**, if the charges are sorted lexiographically.
Note that a `sorted` leg is always `blocked`.
We can also speak of the complete array to be **blocked by charges** or **legcharge-sorted**,  which means that all of its legs are blocked or sorted, respectively.
The charge data for a single leg is collected in the class :class:`~tenpy.linalg.charges.LegCharge`.
A :class:`~tenpy.linalg.charges.LegCharge` has also a flag **qconj**, which tells whether the charges
point *inward* (+1) or *outward* (-1). What that means, is explained later in :ref:`nonzero_entries`.

For completeness, let us also summarize also the internal structure of an :class:`~tenpy.linalg.np_conserved.Array` here:
The array saves only non-zero **blocks**, collected as a list of `np.array` in ``self._data``.
The qindices necessary to map these blocks to the original leg indices are collected in ``self._qdata``
An array is said to be **qdata-sorted** if its ``self._qdata`` is lexiographically sorted.
More details on this follow :ref:`later <array_storage_schema>`.
However, note that you usually shouldn't access `_qdata` and `_data` directly - this
is only necessary from within `tensordot`, `svd`, etc.
Also, an array has a **total charge**, defining which entries can be non-zero - details in :ref:`nonzero_entries`.

Finally, a **leg pipe** (implemented in :class:`~tenpy.linalg.charges.LegPipe`)
is used to formally combine multiple legs into one leg. Again, more details follow :ref:`later <leg_pipes>`.


Physical Example
----------------
For concreteness, you can think of the Hamiltonian :math:`H = -t \sum_{<i,j>} (c^\dagger_i c_j + H.c.) + U n_i n_j` 
with :math:`n_i = c^\dagger_i c_i`.
This Hamiltonian has the global :math:`U(1)` gauge symmetry :math:`c_i \rightarrow c_i e^{i\phi}`.
The corresponding charge is the total number of particles :math:`N = \sum_i n_i`.
You would then introduce one charge with :math:`m=1`.

Note that the total charge is a sum of local terms, living on single sites.
Thus, you can infer the charge of a single physical site: it's just the value :math:`q_i = n_i \in \mathbb{N}` for each of the states.

Note that you can only assign integer charges. Consider for example the spin 1/2 Heisenberg chain.
Here, you can naturally identify the magnetization :math:`S^z = \sum_i S^z_i` as the conserved quantity, 
with values :math:`S^z_i = \pm \frac{1}{2}`. 
Obviously, if :math:`S^z` is conserved, then so is :math:`2 S^z`, so you can use the charges
:math:`q_i = 2 S^z_i \in \lbrace-1, +1 \rbrace` for the `down` and `up` states, respectively.
Alternatively, you can also use a shift and define :math:`q_i = S^z_i + \frac{1}{2} \in \lbrace 0, 1 \rbrace`.

As another example, consider BCS like terms :math:`\sum_k (c^\dagger_k c^\dagger_{-k} + H.c.)`.
These terms break the total particle conservation,
but they preserve the total parity, i.e., :math:`N \mod 2` is conserved. Thus, you would introduce a charge with :math:`m = 2` in this case.

In the above examples, we had only a single charge conserved at a time, but you might be lucky and have multiple
conserved quantities, e.g. if you have two chains coupled only by interactions. 
TeNPy is designed to handle the general case of multiple charges.
When giving examples, we will restrict to one charge, but everything generalizes to multiple charges.


The different formats for LegCharge
-----------------------------------
As mentioned above, we assign charges to each index of each leg of a tensor.
This can be done in three formats: **qflat**, as **qind** and as **qdict**.
Let me explain them with examples, for simplicity considereing only a single charge (the most inner array has one entry
for each charge).

**qflat** form: simply a list of charges for each index. 
    An example::

        qflat = [[-2], [-1], [-1], [0], [0], [0], [0], [3], [3]]

    This tells you that the leg has size 9, the charges for are ``[-2], [-1], [-1], ..., [3]`` for the indices ``0, 1, 2, 3,..., 8``.
    You can identify four `charge blocks` ``slice(0, 1), slice(1, 3), slice(3, 7), slice(7, 9)`` in this example, which have charges ``[-2], [-1], [0], [3]``.
    In other words, the indices ``1, 2`` (which are in ``slice(1, 3)``) have the same charge value ``[-1]``.
    A `qindex` would just enumerate these blocks as ``0, 1, 2, 3``.

**qind** form: a 1D array `slices` and a 2D array `charges`.
    This is a more compact version than the `qflat` form: 
    the `slices` give a partition of the indices and the `charges` give the charge values. The same example as above
    would simply be::

        slices = [0, 1, 3, 7, 9]
        charges = [[-2], [-1], [0], [3]]


    Note that  `slices` includes ``0`` as first entry and the number of indices (here ``9``) as last entries.
    Thus it has len ``block_number + 1``, where ``block_number`` (given by :attr:`~tenpy.linalg.charges.LegCharge.block_number`) 
    is the number of charge blocks in the leg, i.e. a `qindex` runs from 0 to ``block_number-1``.
    On the other hand, the 2D array `charges` has shape ``(block_number, qnumber)``, where ``qnumber`` is the
    number of charges (given by :attr:`~tenpy.linalg.charges.ChargeInfo.qnumber`).

    In that way, the `qind` form maps an `qindex`, say ``qi``, to the indices ``slice(slices[qi], slices[qi+1])`` and
    the charge(s) ``charges[qi]``.


**qdict** form: a dictionary in the other direction than qind, taking charge tuples to slices.
    Again for the same example::

        {(-2,): slice(0, 1),
         (-1,): slice(1, 3),
         (0,) : slice(3, 7),
         (3,) : slice(7, 9)}

    Since the keys of a dictionary are unique, this form is only possible if the leg is `completely blocked`.


The :class:`~tenpy.linalg.charges.LegCharge` saves the charge data of a leg internally in `qind` form, 
directly in the attribute `slices` and `charges`.
However, it also provides convenient functions for conversion between from and to the `qflat` and `qdict` form.

The above example was nice since all charges were sorted and the charge blocks were 'as large as possible'.
This is however not required.

The following example is also a valid `qind` form::

    slices = [0, 1, 3, 5, 7, 9]
    charges = [[-2], [-1], [0], [0], [3]]

This leads to the *same* `qflat` form as the above examples, thus representing the same charges on the leg indices.
However, regarding our Arrays, this is quite different, since it diveds the leg into 5 (instead of previously 4)
charge blocks. We say the latter example is `not bunched`, while the former one is `bunched`.

To make the different notions of `sorted` and `bunched` clearer, consider the following (valid) examples:

================================  =========  =========  ==========
charges                           bunched    sorted     blocked
================================  =========  =========  ==========
``[[-2], [-1], [0], [1], [3]]``   ``True``   ``True``   ``True``
--------------------------------  ---------  ---------  ----------
``[[-2], [-1], [0], [0], [3]]``   ``False``  ``True``   ``False``
--------------------------------  ---------  ---------  ----------
``[[-2], [0], [-1], [1], [3]]``   ``True``   ``False``  ``True``
--------------------------------  ---------  ---------  ----------
``[[-2], [0], [-1], [0], [3]]``   ``True``   ``False``  ``False``
================================  =========  =========  ==========

If a leg is `bunched` and `sorted`, it is automatically `blocked` (but not vice versa). 
See also :ref:`below <blocking>` for further comments on that.


.. _nonzero_entries:

Which entries of the npc Array can be non-zero?
-----------------------------------------------
The reason for the speedup with np_conserved lies in the fact that it saves only the blocks 'compatible' with the charges. 
But how is this 'compatible' defined? 

Assume you have a tensor, call it :math:`T`, and the :class:`~tenpy.linalg.charges.LegCharge` for all of its legs, say :math:`a, b, c, ...`.

Remeber that the LegCharge associates to each index of the leg a charge value (for each of the charges, if `qnumber` > 1).
Let ``a.to_qflat()[ia]`` denote the charge(s) of index ``ia`` for leg ``a``, and similar for other legs.

In addition, the LegCharge has a flag :attr:`~tenpy.linalg.charges.LegCharge.qconj`. This flag **qconj** is only a sign,
saved as +1 or -1, specifying whether the charges point 'inward' (+1, default) or 'outward' (-1) of the tensor.

Then, the **total charge of an entry** ``T[ia, ib, ic, ...]`` of the tensor is defined as::

   qtotal[ia, ib, ic, ...] = a.to_qflat()[ia] * a.qconj + b.to_qflat()[ib] * b.qconj + c.to_qflat()[ic] * c.qconj + ...  modulo qmod

The rule which entries of the a :class:`~tenpy.linalg.np_conserved.Array` can be non-zero
(i.e., are 'compatible' with the charges), is then very simple:

.. topic :: Rule for non-zero entries

    An entry ``ia, ib, ic, ...`` of a :class:`~tenpy.linalg.np_conserved.Array` can only be non-zero,
    if ``qtotal[ia, ib, ic, ...]`` matches the *unique* :attr:`~tenpy.linalg.np_conserved.qtotal` attribute of the class.

In other words, there is a *single* **total charge** ``.qtotal`` attribute of a :class:`~tenpy.linalg.np_conserved.Array`.
All indices ``ia, ib, ic, ...`` for which the above defined ``qtotal[ia, ib, ic, ...]`` matches this `total charge`,
are said to be **compatible with the charges** and can be non-zero. 
All other indices are **incompatible with the charges** and must be zero.

In case of multiple charges, `qnumber` > 1, is a straigth-forward generalization:
an entry can only be non-zero if it is `compatible` with each of the defined charges.


The pesky qconj - contraction as an example
-------------------------------------------
Why did we introduce the ``qconj`` flag? Remember it's just a sign telling whether the charge points inward or outward.
So whats the reasoning?

The short answer is, that LegCharges actually live on bonds (i.e., legs which are to be contracted) 
rather than individual tensors. Thus, it is convenient to share the LegCharges between different legs and even tensors, 
and just adjust the sign of the charges with `qconj`.

As an example, consider the contraction of two tensors, :math:`C_{ia,ic} = \sum_{ib} A_{ia,ib} B_{ib,ic}`.
For simplicity, say that the total charge of all three tensors is zero.
What are the implications of the above rule for non-zero entries?
Or rather, how can we ensure that ``C`` complies with the above rule?
An entry ``C[ia,ic]`` will only be non-zero, 
if there is an ``ib`` such that both ``A[ia,ib]`` and ``B[ib,ic]`` are non-zero, i.e., both of the following equations are
fullfilled::

    A.qtotal == A.legs[0].to_qflat()[ia] * A.legs[0].qconj + A.legs[1].to_qflat()[ib] * A.legs[1].qconj  modulo qmod
    B.qtotal == B.legs[0].to_qflat()[ib] * B.legs[0].qconj + B.legs[1].to_qflat()[ic] * B.legs[1].qconj  modulo qmod

(``A.legs[0]`` is the :class:`~tenpy.linalg.charges.LegCharge` saving the charges of the first leg (with index ``ia``) of `A`.)

For the uncontracted legs, we just keep the charges as they are::

    C.legs = [A.legs[0], B.legs[1]]

It is then straight-forward to check, that the rule is fullfilled for :math:`C`, if the following condition is met::

   A.qtotal + B.qtotal - C.qtotal == A.legs[1].to_qflat()[ib] A.b.qconj + B.legs[0].to_qflat()[ib] B.b.qconj  modulo qmod

The easiest way to meet this condition is (1) to require that ``A.b`` and ``B.b`` share the *same* charges ``b.to_qflat()``, but have
opposite `qconj`, and (2) to define ``C.qtotal = A.qtotal + B.qtotal``.
This justifies the introduction of `qconj`:
when you define the tensors, you have to define the :class:`~tenpy.linalg.charges.LegCharge` for the `b` only once, say for ``A.legs[1]``.
For ``B.legs[0]`` you simply use ``A.legs[1].conj()`` which creates a copy of the LegCharge with shared `slices` and `charges`, but opposite `qconj`.
As a more impressive example, all 'physical' legs of an MPS can usually share the same
:class:`~tenpy.linalg.charges.LegCharge` (up to different ``qconj`` if the local Hilbert space is the same). 
This leads to the following convention:

.. topic :: Convention

   When an npc algorithm makes tensors which share a bond (either with the input tensors, as for tensordot, or amongst the output tensors, as for SVD),
   the algorithm is free, but not required, to use the **same** :class:`LegCharge` for the tensors sharing the bond, *without* making a copy.
   Thus, if you want to modify a LegCharge, you **must** make a copy first (e.g. by using methods of LegCharge for what you want to acchive).


Assigning charges to non-physical legs
--------------------------------------
From the above physical examples, it should be clear how you assign charges to physical legs.
But what about other legs, e.g, the virtual bond of an MPS (or an MPO)? 

The charge of these bonds must be derived by using the 'rule for non-zero entries', as far as they are not arbitrary.
As a concrete example, consider an MPS on just two spin 1/2 sites::

    |        _____         _____
    |   x->- | A | ->-y->- | B | ->-z
    |        -----         -----
    |          ^             ^
    |          |p            |p

The two legs ``p`` are the physical legs and share the same charge, as they both describe the same local Hilbert space.
For better distincition, let me label the indices of them by :math:`\uparrow=0` and :math:`\downarrow=1`.
As noted above, we can associate the charges 1 (:math:`p=\uparrow`) and -1 (:math:`p=\downarrow`), respectively, so we define::

    chinfo = npc.ChargeInfo([1], ['2*Sz'])
    p  = npc.LegCharge.from_qflat(chinfo, [1, -1], qconj=+1)

For the ``qconj`` signs, we stick to the convention used in our MPS code and indicated by the
arrows in above 'picture': physical legs are incoming (``qconj=+1``), and from left to right on the virtual bonds.
This is acchieved by using ``[p, x, y.conj()]`` as `legs` for ``A``, and ``[p, y, z.conj()]`` for ``B``, with the
default ``qconj=+1`` for all ``p, x, y, z``: ``y.conj()`` has the same charges as ``y``, but opposite ``qconj=-1``.

The legs ``x`` and ``z`` of an ``L=2`` MPS, are 'dummy' legs with just one index ``0``.
The charge on one of them, as well as the total charge of both ``A`` and ``B`` is arbitrary (i.e., a gauge freedom),
so we make a simple choice: total charge 0 on both arrays, as well as for :math:`x=0`, 
``x = npc.LegCharge.from_qflat(chinfo, [0], qconj=+1)``.

The charges on the bonds `y` and `z` then depend on the state the MPS represents.
Here, we consider a singlet :math:`\psi = (|\uparrow \downarrow\rangle  - |\downarrow \uparrow\rangle)/\sqrt{2}`
as a simple example. A possible MPS representation is given by::

    A[up, :, :]   = [[1/2.**0.5, 0]]     B[up, :, :]   = [[0], [-1]]
    A[down, :, :] = [[0, 1/2.**0.5]]     B[down, :, :] = [[1], [0]]

There are two non-zero entries in ``A``, for the indices :math:`(a, x, y) = (\uparrow, 0, 0)` and :math:`(\downarrow, 0, 1)`.
For :math:`(a, x, y) = (\uparrow, 0, 0)`, we want::

    A.qtotal = 0 = p.to_qflat()[up] * p.qconj + x.to_qflat()[0] * x.qconj + y.conj().to_qflat()[0] * y.conj().qconj 
                 = 1                * (+1)    + 0               * (+1)    + y.conj().to_qflat()[0] * (-1)

This fixes the charge of ``y=0`` to 1.
A similar calculation for :math:`(a, x, y) = (\downarrow, 0, 1)` yields the charge ``-1`` for ``y=1``.
We have thus all the charges of the leg ``y`` and can define ``y = npc.LegCharge.from_qflat(chinfo, [1, -1], qconj=+1)``.

Now take a look at the entries of ``B``. 
For the non-zero entry :math:`(b, y, z) = (\uparrow, 1, 0)`, we want::

    B.qtotal = 0 = p.to_qflat()[up] * p.qconj + y.to_qflat()[1] * y.qconj + z.conj().to_qflat()[0] * z.conj().qconj 
                 = 1                * (+1)    + (-1)            * (+1)    + z.conj().to_qflat()[0] * (-1)

This implies the charge 0 for `z` = 0, thus ``z = npc.LegCharge.form_qflat(chinfo, [0], qconj=+1)``.
Finally, note that the rule for :math:`(b, y, z) = (\downarrow, 0, 0)` is automatically fullfilled!
This is an implication of the fact that the singlet has a well defined value for :math:`S^z_a + S^z_b`.
For other states without fixed magnetization (e.g., :math:`|\uparrow \uparrow\rangle + |\downarrow \downarrow\rangle`)
this would not be the case, and we could not use charge conservation.

As an exercise, you can calculate the charge of `z` in the case that ``A.qtotal=5``, ``B.qtotal = -1`` and
charge ``2`` for ``x=0``. The result is -2.

.. note ::
    
    This section is meant be an pedagogical introduction. In you program, you can use the functions 
    :func:`~tenpy.linalg.np_conserved.detect_legcharge` (which does exactly what's described above) or
    :func:`~tenpy.linalg.np_conserved.detect_qtotal` (if you know all `LegCharges`, but not `qtotal`).

Array creation
--------------

Making an new :class:`~tenpy.linalg.np_conserved.Array` requires both the tensor entries (data) and charge data.

The default initialization ``a = Array(...)`` creates an empty Array, where all entries are zero
(equivalent to :func:`~tenpy.linalg.np_conserved.zeros`).
(Non-zero) data can be provided either as a dense `np.array` to :meth:`~tenpy.linalg.np_conserved.Array.from_ndarray`,
or by providing a numpy function such as `np.random`, `np.ones` etc. to :meth:`~tenpy.linalg.np_conserved.Array.from_func`.

In both cases, the charge data is provided by one :class:`~tenpy.linalg.charges.ChargeInfo`,
and a :class:`~tenpy.linalg.charges.LegCharge` instance for each of the legs.

.. note ::

    The charge data instances are not copied, in order to allow it to be shared between different Arrays.
    Consequently, you *must* make copies of the charge data, if you manipulate it directly.
    (However, methods like :meth:`~tenpy.linalg.charges.LegCharge.sort` do that for you.)

Of course, a new :class:`~tenpy.linalg.np_conserved.Array` can also created using the charge data from exisiting Arrays,
for examples with :meth:`~tenpy.linalg.np_conserved.Array.zeros_like` or creating a (deep or shallow) :meth:`~tenpy.linalg.np_conserved.Array.copy`.
Further, there are the higher level functions like :func:`~tenpy.linalg.np_conserved.tensordot` or :func:`~tenpy.linalg.np_conserved.svd`,
which also return new Arrays.

Further, new Arrays are created by the various functions like `tensordot` or `svd` in :mod:`~tenpy.linalg.np_conserved`.


.. _blocking:

Complete blocking of Charges
----------------------------

While the code was designed in such a way that each charge sector has a different charge, the code
should still run correctly if multiple charge sectors (for different qindex) correspond to the same charge. 
In this sense :class:`~tenpy.linalg.np_conserved.Array` can act like a sparse array class to selectively store subblocks. 
Algorithms which need a full blocking should state that explicitly in their doc-strings.
(Some functions (like `svd` and `eigh`) require complete blocking internally, but if necessary they just work on
a temporary copy returned by :meth:`~tenpy.linalg.np_conserved.as_completely_blocked`).

If you expect the tensor to be dense subject to charge constraints (as for MPS), 
it will be most efficient to fully block by charge, so that work is done on large chunks.

However, if you expect the tensor to be sparser than required by charge (as for an MPO),
it may be convenient not to completely block, which forces smaller matrices to be stored, and hence many zeroes to be dropped.
Nevertheless, the algorithms were not designed with this in mind, so it is not recommended in general.
(If you want to use it, run a benchmark to check whether it is really faster!)

If you haven't created the array yet, you can call :meth:`~tenpy.linalg.charges.LegCharge.sort` (with ``bunch=True``)
on each :class:`~tenpy.linalg.charges.LegCharge` which you want to block.
This sorts by charges and thus induces a permution of the indices, which is also returned as an 1D array ``perm``.
For consistency, you have to apply this permutation to your flat data as well. 

Alternatively, you can simply call :meth:`~tenpy.linalg.np_conserved.Array.sort_legcharge` on an existing :class:`~tenpy.linalg.np_conserved.Array`.
It calls :meth:`~tenpy.linalg.charges.LegCharge.sort` internally on the specified legs and performs the necessary
permutations directly to (a copy of) `self`. Yet, you should keep in mind, that the axes are permuted afterwards.


.. _array_storage_schema:

Internal Storage schema of npc Arrays
-------------------------------------

The actual data of the tensor is stored in ``_data``. Rather than keeping a single np.array (which would have many zeros in it),
we store only the non-zero sub blocks. So ``_data`` is a python list of `np.array`'s.
The order in which they are stored in the list is not physically meaningful, and so not guaranteed (more on this later).
So to figure out where the sub block sits in the tensor, we need the ``_qdata`` structure (on top of the LegCharges in ``legs``).

Consider a rank 3 tensor ``T``, with the first leg like::
    
    legs[0].slices = np.array([0, 1, 4, ...])
    legs[0].charges = np.array([[-2], [1], ...])

Each row of `charges` gives the charges for a `charge block` of the leg, with the actual indices of the
total tensor determined by the `slices`. 
The *qindex* simply enumerates the charge blocks of a lex.
Picking a qindex (and thus a `charge block`) from each leg, we have a subblock of the tensor.

For each (non-zero) subblock of the tensor, we put a (numpy) ndarray entry in the ``_data`` list.
Since each subblock of the tensor is specified by `rank` qindices, 
we put a corresponding entry in ``_qdata``, which is a 2D array of shape ``(#stored_blocks, rank)``.
Each row corresponds to a non-zero subblock, and there are rank columns giving the corresponding qindex for each leg.

Example: for a rank 3 tensor we might have::

    T._data = [t1, t2, t3, t4, ...]
    T._qdata = np.array([[3, 2, 1],
                         [1, 1, 1],
                         [4, 2, 2],
                         [2, 1, 2],
                         ...       ])

The third subblock has an ndarray ``t3``, and qindices ``[4 2 2]`` for the three legs.

- To find the position of ``t3`` in the actual tensor you can use :meth:`~tenpy.linalg.charges.LegCharge.get_slice`::

            T.legs[0].get_slice(4), T.legs[1].get_slice(2), T.legs[2].get_slice(2)
  
  The function ``leg.get_charges(qi)`` simply returns ``slice(leg.slices[qi], leg.slices[qi+1])``

- To find the charges of t3, we an use :meth:`~tenpy.linalg.charges.LegCharge.get_charge`::

            T.legs[0].get_charge(2), T.legs[1].get_charge(2), T.legs[2].get_charge(2)

  The function ``leg.get_charge(qi)`` simply returns ``leg.charges[qi]*leg.qconj``.

.. note ::

   Outside of `np_conserved`, you should use the API to access the entries. 
   If you really need to iterate over all blocks of an Array ``T``, try ``for (block, blockslices, charges, qindices) in T: do_something()``.

The order in which the blocks stored in ``_data``/``_qdata`` is arbitrary (although of course ``_data`` and ``_qdata`` must be in correspondence).
However, for many purposes it is useful to sort them according to some convention.  So we include a flag ``._qdata_sorted`` to the array.
So, if sorted (with :meth:`~tenpy.linalg.np_conserved.Array.isort_qdata`, the ``_qdata`` example above goes to ::

    _qdata = np.array([[1, 1, 1],
                       [3, 2, 1],
                       [2, 1, 2],
                       [4, 2, 2],
                       ...       ])

Note that `np.lexsort` chooses the right-most column to be the dominant key, a convention we follow throughout.

If ``_qdata_sorted == True``, ``_qdata`` and ``_data`` are guaranteed to be lexsorted. If ``_qdata_sorted == False``, there is no gaurantee.
If an algorithm modifies ``_qdata``, it **must** set ``_qdata_sorted = False`` (unless it gaurantees it is still sorted).
The routine :meth:`~tenpy.linalg.np_conserved.Array.sort_qdata` brings the data to sorted form.


.. _Array_element_access:

Indexing of an Array
--------------------

Although it is usually not necessary to access single entries of an :class:`~tenpy.linalg.np_conserved.Array`, you can of course do that.
In the simplest case, this is something like ``A[0, 2, 1]`` for a rank-3 Array ``A``.
However, accessing single entries is quite slow and usually not recommended. For small Arrays, it may be convenient to convert them
back to flat numpy arrays with :meth:`~tenpy.linalg.np_conserved.Array.to_ndarray`.

On top of that very basic indexing, `Array` supports slicing and some kind of advanced indexing, which is however
different from the one of numpy arrarys (described `here <http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`_).
Unlike numpy arrays, our Array class does not broadcast existing index arrays -- this would be terribly slow.
Also, `np.newaxis` is not supported, since inserting new axes requires additional information for the charges.

Instead, we allow just indexing of the legs independent of each other, of the form ``A[i0, i1, ...]``.
If all indices ``i0, i1, ...`` are integers, the single corresponding entry (of type `dtype`) is returned.

However, the individual 'indices' ``i0`` for the individual legs can also be one of what is described in the following list.
In that case, a new :class:`~tenpy.linalg.np_conserved.Array` with less data (specified by the indices) is returned.

The 'indices' can be:

- an `int`: fix the index of that axis, return array with one less dimension. See also :meth:`~tenpy.linalg.np_conserved.Array.take_slice`.
- a ``slice(None)`` or ``:``: keep the complete axis
- an ``Ellipsis`` or ``...``: shorthand for ``slice(None)`` for missing axes to fix the len
- an 1D bool `ndarray` ``mask``: apply a mask to that axis, see :meth:`~tenpy.linalg.np_conserved.Array.iproject`.
- a ``slice(start, stop, step)`` or ``start:stop:step``: keep only the indices specified by the slice. This is also implemented with `iproject`.
- an 1D int `ndarray` ``mask``: keep only the indices specified by the array. This is also implemented with `iproject`.

For slices and 1D arrays, additional permuations may be perfomed with the help of :meth:`~tenpy.linalg.np_conserved.Array.permute`.

If the number of indices is less than `rank`, the remaining axes remain free, so for a rank 4 Array ``A``, ``A[i0, i1] == A[i0, i1, ...] == A[i0, i1, :, :]``.

Note that indexing always **copies** the data -- even if `int` contains just slices, in which case numpy would return a view.
However, assigning with ``A[:, [3, 5], 3] = B`` should work as you would expect.

.. warning ::

    Due to numpy's advanced indexing, for 1D integer arrays ``a0`` and ``a1`` the following holds ::

        A[a0, a1].to_ndarray() == A.to_ndarray()[np.ix_(a0, a1)] != A.to_ndarray()[a0, a1]

    For a combination of slices and arrays, things get more complicated with numpys advanced indexing.
    In that case, a simple ``np.ix_(...)`` doesn't help any more to emulate our version of indexing.


.. _leg_pipes:

Introduction to combine_legs, split_legs and LegPipes
-----------------------------------------------------

Often, it is necessary to "combine" multiple legs into one: for example to perfom a SVD, a tensor needs to be viewed as a matrix.
For a flat array, this can be done with ``np.reshape``, e.g., if ``A`` has shape ``(10, 3, 7)`` then ``B = np.reshape(A, (30, 7))`` will
result in a (view of the) array with one less dimension, but a "larger" first leg. By default (``order='C'``), this
results in ::
    
    B[i*3 + j , k] == A[i, j, k] for i in range(10) for j in range(3) for k in range(7)

While for a np.array, also a reshaping ``(10, 3, 7) -> (2, 21, 5)`` would be allowed, it does not make sense
physically. The only sensible "reshape" operation on an :class:`~tenpy.linalg.np_conserved.Array` are

1) to **combine** multiple legs into one **leg pipe** (:class:`~tenpy.linalg.charges.LegPipe`) with  :meth:`~tenpy.linalg.np_conserved.Array.combine_legs`, or
2) to **split** a pipe of previously combined legs with :meth:`~tenpy.linalg.np_conserved.Array.split_legs`.

Each leg has a Hilbert space, and a representation of the symmetry on that Hilbert space.
Combining legs corresponds to the tensor product operation, and for abelian groups, 
the corresponding "fusion" of the representation is the simple addition of charge.

Fusion is not a lossless process, so if we ever want to split the combined leg,
we need some additional data to tell us how to reverse the tensor product.
This data is saved in the class :class:`~tenpy.linalg.charges.LegPipe`, derived from the :class:`~tenpy.linalg.charges.LegCharge` and used as new `leg`.
Details of the information contained in a LegPipe are given in the class doc string.

The rough usage idea is as follows:

1) You can call :meth:`~tenpy.linalg.np_conserved.Array.combine_legs` without supplying any LegPipes, `combine_legs` will then make them for you.

   Nevertheless, if you plan to perform the combination over and over again on sets of legs you know to be identical
   [with same charges etc, up to an overall -1 in `qconj` on all incoming and outgoing Legs]
   you might make a LegPipe anyway to save on the overhead of computing it each time.
2) In any way, the resulting Array will have a :class:`~tenpy.linalg.charges.LegPipe` as a LegCharge on the combined leg.
   Thus, it -- and all tensors inheriting the leg (e.g. the results of `svd`, `tensordot` etc.) -- will have the information
   how to split the `LegPipe` back to the original legs.
3) Once you performed the necessary operations, you can call :meth:`~tenpy.linalg.Array.split_legs`.
   This uses the information saved in the `LegPipe` to split the legs, recovering the original legs.

For a LegPipe, :meth:`~tenpy.linalg.charges.LegPipe.conj` changes ``qconj`` for the outgoing pipe *and* the incoming legs.
If you need a `LegPipe` with the same incoming ``qconj``, use :meth:`~tenpy.linalg.charges.LegPipe.outer_conj`.


Leg labeling
------------

It's convenient to name the legs of a tensor: for instance, we can name legs 0, 1, 2 to be ``'a', 'b', 'c'``: :math:`T_{i_a,i_b,i_c}`.
That way we don't have to remember the ordering! Under tensordot, we can then call ::

    U = npc.tensordot(S, T, axes = [ [...],  ['b'] ] )

without having to remember where exactly ``'b'`` is.
Obviously ``U`` should then inherit the name of its legs from the uncontracted legs of `S` and `T`.
So here is how it works:

- Labels can *only* be strings. The labels should not include the characters ``.`` or ``?``.
  Internally, the labels are stored as dict ``a.labels = {label: leg_position, ...}``. Not all legs need a label.
- To set the labels, call ::

        A.set_labels(['a', 'b', None, 'c', ... ])

  which will set up the labeling ``{'a': 0, 'b': 1, 'c': 3 ...}``.

- (Where implemented) the specification of axes can use either the labels **or** the index positions.
  For instance, the call ``tensordot(A, B, [['a', 2, 'c'], [...]])`` will interpret ``'a'`` and  ``'c'`` as labels 
  (calling :meth:`~tenpy.linalg.np_conserved.Array.get_leg_indices` to find their positions using the dict)
  and 2 as 'the 2nd leg'. That's why we require labels to be strings!
- Labels will be intelligently inherited through the various operations of `np_conserved`.
    - Under `transpose`, labels are permuted.
    - Under `tensordot`, labels are inherited from uncontracted legs. If there is a collision, both labels are dropped.
    - Under `combine_legs`, labels get concatenated with a ``.`` delimiter and sourrounded by brackets.
      Example: let ``a.labels = {'a': 1, 'b': 2, 'c': 3}``.
      Then if ``b = a.combine_legs([[0, 1], [2]])``, it will have ``b.labels = {'(a.b)': 0, '(c)': 1}``.
      If some sub-leg of a combined leg isn't named, then a ``'?#'`` label is inserted (with ``#`` the leg index), e.g., ``'a.?0.c'``.
    - Under `split_legs`, the labels are split using the delimiters (and the ``'?#'`` are dropped).
    - Under `conj`, `iconj`: take  ``'a' -> 'a*'``, ``'a*' -> 'a'``, and ``'(a.(b*.c))' -> '(a*.(b.c*))'``
    - Under `svd`, the outer labels are inherited, and inner labels can be optionally passed.
    - Under `pinv`, the labels are transposed.


See also
--------
- The module :mod:`tenpy.linalg.np_conserved` should contain all the API needed from the point of view of the algorithms.
  It contians the fundamental :class:`~tenpy.linalg.np_conserved.Array` class and functions for working with them (creating and manipulating).
- The module :mod:`tenpy.linalg.charges` contains implementations for the charge structure, for example the classes
  :class:`~tenpy.linalg.charges.ChargeInfo`, :class:`~tenpy.linalg.charges.LegCharge`, and :class:`~tenpy.linalg.charges.LegPipe`.
  As noted above, the 'public' API is imported to (and accessible from) :mod:`~tenpy.linalg.np_conserved`.

A full example code for spin-1/2
--------------------------------
Below follows a full example demonstrating the creation and contraction of Arrays.
(It's the file `a_np_conserved.py` in the examples folder of the tenpy source.)

.. literalinclude:: /../examples/a_np_conserved.py
