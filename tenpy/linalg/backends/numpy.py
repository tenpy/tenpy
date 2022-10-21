from __future__ import annotations
import numpy as np

from tenpy.linalg.backends.abelian import AbstractAbelianBackend
from tenpy.linalg.backends.abstract_backend import AbstractBlockBackend, Dtype, Precision, \
    BackendDtype, Block
from tenpy.linalg.backends.no_symmtery import AbstractNoSymmetryBackend
from tenpy.linalg.backends.nonabelian import AbstractNonabelianBackend
from tenpy.linalg.symmetries import AbstractSymmetry

_dtype_map = {
    Dtype(Precision.half, True): np.float16,
    Dtype(Precision.single, True): np.float32,
    Dtype(Precision.double, True): np.float64,
    Dtype(Precision.long_double, True): np.longdouble,
    Dtype(Precision.half, False): np.complex32,
    Dtype(Precision.single, False): np.complex64,
    Dtype(Precision.double, False): np.complex128,
    Dtype(Precision.long_double, False): np.clongdouble,
}

_inverse_dtype_map = {
    np.float16: Dtype(Precision.half, True),
    np.float32: Dtype(Precision.single, True),
    np.float64: Dtype(Precision.double, True),
    np.longdouble: Dtype(Precision.long_double, True),
    np.complex32: Dtype(Precision.half, False),
    np.complex64: Dtype(Precision.single, False),
    np.complex128: Dtype(Precision.double, False),
    np.clongdouble: Dtype(Precision.long_double, False),
}


class NumpyBlockBackend(AbstractBlockBackend):

    def __init__(self):
        AbstractBlockBackend.__init__(self, default_precision=Precision.single)

    def parse_dtype(self, dtype: Dtype) -> BackendDtype:
        try:
            return _dtype_map[dtype]
        except KeyError:
            raise ValueError(f'dtype {dtype} not supported for {self}.') from None

    def parse_block(self, obj, dtype: BackendDtype = None) -> Block:
        return np.asarray(obj, dtype)

    def block_is_real(self, a: Block):
        return not np.iscomplexobj(a)

    def block_tdot(self, a: Block, b: Block, a_axes: list[int], b_axes: list[int]) -> Block:
        return np.tensordot(a, b, (a_axes, b_axes))

    def block_shape(self, a: Block) -> tuple[int]:
        return np.shape(a)

    def block_item(self, a: Block):
        return a.item()

    def block_dtype(self, a: Block) -> Dtype:
        return _inverse_dtype_map[np.dtype(a)]

    def block_to_dtype(self, a: Block, dtype: BackendDtype) -> Block:
        return np.asarray(a, dtype=dtype)

    def block_copy(self, a: Block) -> Block:
        return np.copy(a)

    def _block_repr_lines(self, a: Block, indent: str, max_width: int, max_lines: int):
        # TODO i like julia style much better actually, especially for many legs
        with np.set_printoptions(linewidth=max_width - len(indent)):
            lines = [f'{indent}{line}' for line in repr(a).split('\n')]
        if len(lines) > max_lines:
            first = (max_lines - 1) // 2
            last = max_lines - 1 - first
            lines = lines[:first] + [f'{indent}...'] + lines[-last:]
        return lines


class NoSymmetryNumpyBackend(NumpyBlockBackend, AbstractNoSymmetryBackend):
    def __init__(self):
        NumpyBlockBackend.__init__(self)
        AbstractNoSymmetryBackend.__init__(self)


class AbelianNumpyBackend(NumpyBlockBackend, AbstractAbelianBackend):
    def __init__(self, symmetry: AbstractSymmetry):
        NumpyBlockBackend.__init__(self)
        AbstractAbelianBackend.__init__(self, symmetry=symmetry)


class NonabelianNumpyBackend(NumpyBlockBackend, AbstractNonabelianBackend):
    def __init__(self, symmetry: AbstractSymmetry):
        NumpyBlockBackend.__init__(self)
        AbstractNonabelianBackend.__init__(self, symmetry=symmetry)
