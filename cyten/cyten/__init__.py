r"""cyten library - tensor library for high-level tensor network algorithms.

Provides a tensor class with block-sparsity from symmetries with an exchangeable GPU or CPU backend.

"""
# Copyright (C) TeNPy Developers, Apache license

# note: order matters!
from . import (
    backends,
    block_backends,
    dummy_config,
    models,
    symmetries,
    tensors,
    tools,
    version,
)

# import pybind11 bindings from C++ code
# TODO do explicit imports instead of *
from ._core import *  # type: ignore
from .backends import TensorBackend, get_backend

# subpackages
from .block_backends import Block, BlockBackend, Dtype, NumpyBlockBackend, TorchBlockBackend
from .models import Coupling, Site, couplings, sites
from .symmetries import (
    AbelianGroup,
    AbelianLegPipe,
    BraidChiralityUnspecifiedError,
    BraidingStyle,
    ElementarySpace,
    FermionNumber,
    FermionParity,
    FibonacciAnyonCategory,
    FusionStyle,
    # trees
    FusionTree,
    GroupSymmetry,
    IsingAnyonCategory,
    # spaces
    Leg,
    LegPipe,
    NoSymmetry,
    ProductSymmetry,
    QuantumDoubleZNAnyonCategory,
    Sector,
    SectorArray,
    Space,
    SU2_kAnyonCategory,
    SU2Symmetry,
    SU3_3AnyonCategory,
    SUNSymmetry,
    Symmetry,
    # _symmetries.py
    SymmetryError,
    TensorProduct,
    ToricCodeCategory,
    U1Symmetry,
    ZNAnyonCategory,
    ZNAnyonCategory2,
    ZNSymmetry,
    double_semion_category,
    fermion_number,
    fermion_parity,
    fibonacci_anyon_category,
    fusion_trees,
    ising_anyon_category,
    no_symmetry,
    semion_category,
    su2_symmetry,
    toric_code_category,
    u1_symmetry,
    z2_symmetry,
    z3_symmetry,
    z4_symmetry,
    z5_symmetry,
    z6_symmetry,
    z7_symmetry,
    z8_symmetry,
    z9_symmetry,
)
from .tensors import (
    ChargedTensor,
    DiagonalTensor,
    Mask,
    PlanarDiagram,
    PlanarLinearOperator,
    SymmetricTensor,
    Tensor,
    add_trivial_leg,
    almost_equal,
    angle,
    apply_mask,
    bend_legs,
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
    horizontal_factorization,
    imag,
    inner,
    is_scalar,
    item,
    krylov_based,
    lq,
    move_leg,
    norm,
    on_device,
    outer,
    partial_trace,
    permute_legs,
    pinv,
    # planar
    planar,
    planar_contraction,
    planar_partial_trace,
    planar_permute_legs,
    qr,
    real,
    real_if_close,
    scalar_multiply,
    scale_axis,
    # sparse
    sparse,
    split_legs,
    sqrt,
    squeeze_legs,
    stable_log,
    svd,
    tdot,
    tensor,
    tensor_from_grid,
    trace,
    transpose,
    truncated_svd,
    zero_like,
)
from .version import full_version as __full_version__
from .version import version as __version__


def show_config():
    """Print information about the version of tenpy and used libraries.

    The information printed is :attr:`cyten.version.version_summary`.
    """
    print(version.version_summary)


# expose Dtypes directly
bool = Dtype.bool
float32 = Dtype.float32
complex64 = Dtype.complex64
float64 = Dtype.float64
complex128 = Dtype.complex128
