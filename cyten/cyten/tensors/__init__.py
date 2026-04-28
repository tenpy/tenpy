r""".. _tensor_leg_labels:

Leg Labels
----------

TODO elaborate

The following characters have special meaning in labels and should be avoided for "single" labels:
`(`, `.`, `)`, `?`, `!` and `*`.

See :func:`is_valid_leg_label`.


.. _tensors_as_maps:

Tensors as Maps
---------------

We can view :math:`m \times n` matrices either as linear maps
:math:`\mathbb{C}^n \to \mathbb{C}^m` or as elements of the space
:math:`\mathbb{C}^n \otimes \mathbb{C}^m^*`, which is known in the context of mixed state
density matrices as "vectorization".

Similarly, we can view any tensor, i.e. elements of tensor product spaces as linear maps.
TODO elaborate.


.. _conj_and_transpose:

Conjugation and Transposition
-----------------------------

TODO should this be here or in the docstrings of the respective functions?

TODO elaborate on the differences between dagger and conj etc.

Note that only dagger is independent of partition of the legs into (co)domain.

    ==============  ====================  ====================  ============================
    tensor          domain                codomain              legs
    ==============  ====================  ====================  ============================
    A               [V1, V2]              [W1, W2]              [W1, W2, V2.dual, V1.dual]
    --------------  --------------------  --------------------  ----------------------------
    dagger(A)       [W1, W2]              [V1, V2]              [V1, V2, W2.dual, W1.dual]
    --------------  --------------------  --------------------  ----------------------------
    transpose(A)    [W2.dual, W1.dual]    [V2.dual, V1.dual]    [V2.dual, V1.dual, W1, W2]
    --------------  --------------------  --------------------  ----------------------------
    conj(A)         [V2.dual, V1.dual]    [W2.dual, W1.dual]    [W2.dual, W1.dual, V1, V2]
    ==============  ====================  ====================  ============================

Consider now a matrix ``A`` with signature ``[V] -> [W]``, i.e. with legs ``[W, V.dual]``.
The dagger ``dagger(A)`` has legs signature ``[W] -> [V]`` and legs ``[V, W.dual]``, i.e.
it can be directly contracted with ``A``.


.. _decompositions:

Tensor Decompositions
---------------------
TODO elaborate on the details of decompositions (svd, eig, qr, ...) that they have in common.
I.e. viewing tensors as linear maps, combining legs or not, mention :func:`combine_to_matrix`.

"""
# Copyright (C) TeNPy Developers, Apache license

from ._tensors import (
    ChargedTensor,
    DiagonalTensor,
    LabelledLegs,
    Mask,
    SymmetricTensor,
    Tensor,
    add_trivial_leg,
    almost_equal,
    angle,
    apply_mask,
    apply_mask_DiagonalTensor,
    bend_legs,
    check_same_legs,
    combine_legs,
    combine_to_matrix,
    complex_conj,
    compose,
    cutoff_inverse,
    dagger,
    eigh,
    enlarge_leg,
    entropy,
    exp,
    eye,
    get_same_device,
    horizontal_factorization,
    imag,
    inner,
    is_scalar,
    is_valid_leg_label,
    item,
    linear_combination,
    lq,
    move_leg,
    norm,
    on_device,
    outer,
    partial_compose,
    partial_trace,
    permute_legs,
    pinv,
    qr,
    real,
    real_if_close,
    scalar_multiply,
    scale_axis,
    split_legs,
    sqrt,
    squeeze_legs,
    stable_log,
    svd,
    svd_apply_mask,
    tdot,
    tensor,
    tensor_from_grid,
    trace,
    transpose,
    truncate_singular_values,
    truncated_svd,
    zero_like,
)
from .krylov_based import Arnoldi, KrylovBased, LanczosEvolution, LanczosGroundState, lanczos, lanczos_arpack
from .planar import (
    ContractionTree,
    PlanarDiagram,
    PlanarLinearOperator,
    planar_contraction,
    planar_partial_trace,
    planar_permute_legs,
)
from .sparse import (
    HermitianNumpyArrayLinearOperator,
    LinearOperator,
    NumpyArrayLinearOperator,
    ProjectedLinearOperator,
    ShiftedLinearOperator,
    SumLinearOperator,
    TensorLinearOperator,
    gram_schmidt,
)
