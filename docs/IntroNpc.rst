Introduction to np_conserved
============================

The basic idea is quickly summarized:
By inspecting the Hamiltonian, you can identify symmetries, which correspond to conserved quantities, called **charges**.
These charges divide the tensors into different sectors. This can be used to infer for example a block-diagonal structure
of certain matrices, which in turn speeds up SVD or diagonalization a lot.
Even for more general (non-square-matrix) tensors, charge conservation imposes restrictions which blocks of a tensor can
be non-zero. Only those blocks need to be saved, and e.g. tensordot can be speeded up.

Notations
---------
Lets fix the notation for this introduction and the doc-strings in :mod:`~tenpy.linalg.np_conserved`.

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
This can be done in three formats: **qflat**, as **qind** and as **qdict**.
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


.. _nonzero_entries:

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
:class:`~tenpy.linalg.charges.LegCharge` (up to different ``qconj``). This leads to the following convention:

.. topic :: Convention

   When an npc algorithm makes tensors which share a bond (either with the input tensors, as for tensordot, or amongst the output tensors, as for SVD),
   the algorithm is free, but not required, to use the **same** LegCharge for the tensors sharing the bond, without making a copy.
   Thus, if you want to modify a LegCharge, you **must** make a copy first (e.g. by using methods of LegCharge for what you want to acchive).


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

The legs ``a`` and ``b`` are physical, say with indices :math:`\uparrow = 0` and :math:`\downarrow = 1`.
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
For other states without fixed magnetization (e.g., :math:`|\uparrow \uparrow> + |\downarrow \downarrow>`)
we could not use the charge conservation.

Array creation
--------------

Making an new :class:`~tenpy.linalg.np_conserved.Array` requires both the tensor entries (data) and charge data.

The default initialization ``a = Array(...)`` creates an empty Array, where all entries are zero
(equivalent to :func:`~tenpy.linalg.np_conserved.zeros`).
(Non-zero) data can be provided either as a dense `np.array` to :meth:`~tenpy.linalg.np_conserved.Array.from_ndarray`,
or by providing a numpy function such as `np.random`, `np.ones` etc. to :meth:`~tenpy.linalg.np_conserved.Array.from_npfunc`.

In both cases, the charge data is provided by one :class:`~tenpy.linalg.charges.ChargeInfo`,
and a :class:`~tenpy.linalg.charges.LegCharge` instance for each of the legs.

.. note ::

    The charge data instances are not copied, in order to allow it to be shared between different Arrays.
    Consequently, you **must** make copies of the charge data, if you manipulate it directly.
    (However, methods like :meth:`~tenpy.linalg.charges.LegCharge.sort` do that for you.)

Of course, a new :class:`~tenpy.linalg.np_conserved.Array` can also created using the charge data from exisiting Arrays,
for examples with :meth:`~tenpy.linalg.np_conserved.Array.zeros_like` or creating a (deep or shallow) :meth:`~tenpy.linalg.np_conserved.Array.copy`.
Further, there are the higher level functions like :func:`~tenpy.linalg.np_conserved.tensordot` or :func:`~tenpy.linalg.np_conserved.svd`,
which also return new Arrays.

Further, new Arrays are created by the various functions like `tensordot` or `svd` in :mod:`~tenpy.linalg.np_conserved`.

Complete blocking of Charges
----------------------------

While the code was designed in such a way that each charge sector has a different charge, most of the code
will still run correctly if multiple charge sectors (qindices) correspond to the same charge. 
In this sense :class:`~tenpy.linalg.np_conserved.Array` acts like a sparse array class and can selectively store subblocks. 
Algorithms which need a full blocking should state that explicitly in their doc-strings.

If you expect the tensor to be dense subject to charge constraint (as for MPS), 
it will be most efficient to fully block by charge, so that work is done on large chunks.

However, if you expect the tensor to be sparser than required by charge (as for an MPO),
it may be convenient not to completely block, which forces smaller matrices to be stored, and hence many zeroes to be dropped.
Nevertheless, the algorithms were not designed with this in mind, so it is not recommended in general.

If you haven't created the array yet, you can call :meth:`~tenpy.linalg.charges.LegCharge.sort` (with ``bunch=True``)
on each :class:`~tenpy.linalg.charges.LegCharge` which you want to block.
This sorts by charges and thus induces a permution of the indices, which is also returned as an 1D array ``perm``.
For consistency, you have to apply this permutation to you flat data as well. 

Alternatively, you can simply call :meth:`~tenpy.linalg.np_conserved.Array.sort` on a existing :class:`~tenpy.linalg.np_conserved.Array`.
It calls :meth:`~tenpy.linalg.charges.LegCharge.sort` internally on the specified legs and performs the necessary
permutations directly to (a copy of) `self`. Yet, you should keep in mind, that the axes are permuted afterwards.


.. _array_storage_schema:

Internal Storage schema of npc Arrays
-------------------------------------

The actual data of the tensor is stored in ``_data``. Rather than keeping a single np.array (which would have many zeros in it),
we store only the non-zero sub blocks. So ``_data`` is a python list of `np.array`'s.
The order in which they are stored in the list is not physically meaningful, and so not guaranteed (more on this later).
So to figure out where the sub block sits in the tensor, we need the ``_qdata`` structure (on top of the LegCharges in ``legs``).

Consider a rank 3 tensor, with ``legs[0].qind`` something like::

    qind = np.array([[0, 1, -2],  # and something else for legs[1].qind and legs[2].qind
                     [1, 4,  1],
                     ...        ])

Each row of ``leg[i].qind`` is a *block* of leg[i], labeled by its *qindex* (which is just its row in ``qind``).
Picking a block from each leg, we have a subblock of the tensor.

For each non-zero subblock of the tensor, we put a np.array entry in the ``_data`` list.
Since each subblock of the tensor is specified by `rank` qindices, 
we put a corresponding entry in ``_qdata``, which is a 2D array of shape ``(#blocks, rank)``.
Each row corresponds to a non-zero subblock, and there are rank columns giving the corresponding qindices.

Example: for a rank 3 tensor we might have::

    _data = [t1, t2, t3, t4, ...]
    _qdata = np.array([[3, 2, 1],
                       [1, 1, 1],
                       [4, 2, 2],
                       [2, 1, 2],
                       ...       ])

The 'third' subblock has an nd.array ``t3``, and qindices ``[4 2 2]``.
Recall that each row of `qind` looks like ``[start, stop, charge]``. So:

- To find  t3s position in the actual tensor, we would look at the data ::

            legs[0].qind[4, 0:2], legs[1].qind[2, 0:2], legs[2].qind[2, 0:2]

- To find the charge of t3, we would look at ::

            legs[0].qind[4, 2:], legs[1].qind[2, 2:], legs[2].qind[2, 2:]

.. note ::

   Outside of `np_conserved`, you should use the API to access the entries. 
   To iterate over all blocks of an array ``A``, try ``for (block, blockslices, charges, qdat) in A: do_something()``.

The order in which the blocks stored in ``_data``/``_qdata`` is arbitrary (though of course ``_data`` and ``_qdata`` must be in correspondence).
However, for many purposes it is useful to sort them according to some convention.  So we include a flag ``._qdata_sorted`` to the array.
So, if sorted, the ``_qdata`` example above goes to ::

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

On top of that very basic indexing, `Array` supports part of the slicing and advanced indexing of numpy arrarys
described in `http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html`_.
However, our Array class does not support advanced indexing with a single index array (for `rank`>1) -- this would be terribly slow.
Also, `np.newaxis` is not supported, since inserting new axes requires additional information for the charges.

We allow only indexing of the form ``A[i0, i1, ...]``. If all indices ``i0, i1, ...`` are integers, 
the single corresponding entry (of type `dtype`) is returned.

However, the individual 'indices' ``i0`` for the individual legs can also be one of what is described in the following list.
In that case, a new :class:`~tenpy.linalg.np_conserved.Array` with less data (specified by the indices) is returned.

The 'indices' can be:

- an `int`: fix the index of that axis, return array with one less dimension. See also :meth:`~tenpy.linalg.np_conserved.Array.take_slice`.
- a ``slice(None)`` or ``:``: keep the complete axis
- an ``Ellipsis`` or ``...``: shorthand for ``slice(None)`` for missing axes to fix the len
- an 1D bool `ndarray` ``mask``: apply a mask to that axis, see :meth:`~tenpy.linalg.np_conserved.Array.iproject`.
- a ``slice(start, stop, step)`` or ``start:stop:step``: keep only the indices specified by the slice. This is also implemented with `iproject`.
- an 1D int `ndarray` ``mask``: keep only the indices specified by the array. This is also implemented with `iproject`.

If the number of indices is less than `rank`, the remaining axes remain free, so for a rank 4 Array ``A``, 
``A[i0, i1] == A[i0, i1, ...] == A[i0, i1, :, :]``.

Currently, advanced indexing is not supported for setting values.

.. warning ::

    Due to numpy's advanced indexing, for 1D integer arrays ``a0`` and ``a1`` the following holds ::

        A[a0, a1].to_ndarray() == A.to_ndarray()[np.ix_(a0, a1)] != A.to_ndarray()[a0, a1]

.. _leg_pipes:

Introduction to combine_legs, split_legs and LegPipes
-----------------------------------------------------

Unlike an np.array, the only sensible "reshape" operation on an npc.array is to combine multiple legs into one (**combine_legs**), or the reverse (**split_legs**).

Each leg has a Hilbert space, and a representation of the symmetry on that Hilbert space.
Combining legs corresponds to the tensor product operation, and for abelian groups, 
the corresponding "fusion" of the representation is the simple addition of charge.

Fusion is not a lossless process, so if we ever want to split the combined leg,
we need some additional data to tell us how to reverse the tensor product.
This data is called a **LegPipe**, which we implemented as a class :class:`~tenpy.linalg.charges.LegPipe`.
Details of the information contained in the LegPipe are given in the class doc string.

The rough usage idea is as follows:

a) If you want to combine legs, and do **not** intend to  split any of the newly formed legs back, 
   you can call :meth:`~tenpy.linalg.Array.combine_legs` without supplying any LegPipes, `combine_legs` will then make them for you.

   Nevertheless, if you plan to perform the combination over and over again on sets of legs you know to be identical (ie, same charges etc.)
   you might make a LegPipe anyway to save on the overhead of computing it each time.

b) If you want to combine legs, and for some subset of the new legs you will want to split back
   (either on the tensor in question, or progeny formed by `svd`, `tensordot`, etc.),
   you *DO* need to compute a LegPipe for the legs in questions before combining them.
   `split_legs` will then use the pipes to split the leg.

.. todo ::

   Implement LegPipes and (re-)write this chapter. By deriving LegPipe from LegCharge, we might not need to save it
   separately?

Leg labeling
------------

It's convenient to name the legs of a tensor: for instance, we can name legs 0, 1, 2 to be ``'a', 'b', 'c'``: :math:`T_{i_a,i_b,i_c}``.
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
  For instance, the call ``tensordot(A, B, [ ['a', 2, 'c'], [...]])`` will interpret ``'a'`` and  ``'c'`` as labels 
  (calling :meth:`~tenpy.linalg.np_conserved.Array.get_leg_indices` to find their positions using the dict)
  and 2 as 'the 2nd leg'. That's why we require labels to be strings!
- Labels will be intelligently inherited through the various operations of `np_conserved`.
    - Under `transpose`, labels are permuted.
    - Under `conj`, `iconj`: takes  ``'a' -> 'a*'`` and ``'a*' -> 'a'``
    - Under `tensordot`, labels are inherited from uncontracted legs. If there is a collision, both labels are dropped.
    - Under `combine_legs`, labels get concatenated with a ``.`` delimiter.  Example: let ``a.labels = {'a': 1, 'b': 2, 'c': 3}``.
      Then if `b = a.combine_legs([[0, 1], [2]])``, it will have ``b.labels = {'a.b': 0, 'c': 1}``.
      If some sub-leg of a combined leg isn't named, then a ``'?#'`` label is inserted (with ``#`` the leg index), e.g., ``'a.?0.c'``.
    - Under `split_legs`, the labels are split using the delimiters (and the ``'?#'`` are dropped).
    - Under `svd`, the outer labels are inherited, and inner labels can be optionally passed.
    - Under `pinv`, the labels are transposed


See also
--------
- The module :mod:`tenpy.linalg.np_conserved` should contain all the API needed from the point of view of the algorithms.
  It contians the fundamental :class:`~tenpy.linalg.np_conserved.Array` class and functions for working with them (creating and manipulating).
- The module :mod:`tenpy.linalg.charges` contains implementations for the charge structure, for example the classes
  :class:`~tenpy.linalg.charges.ChargeInfo`, :class:`~tenpy.linalg.charges.LegCharge`, and :class:`~tenpy.linalg.charges.LegPipe`.
  As noted above, all 'public' API is imported in :mod:`~tenpy.linalg.np_conserved`.



.. todo ::
   Full example
   Further References?!?
