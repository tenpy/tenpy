from __future__ import annotations
import numpy as np

from tenpy.linalg.backends.abelian import AbstractAbelianBackend
from tenpy.linalg.backends.abstract_backend import AbstractBlockBackend, Dtype, Precision, \
    BackendDtype, BackendArray, Block
from tenpy.linalg.backends.no_symmtery import AbstractNoSymmetryBackend
from tenpy.linalg.backends.nonabelian import AbstractNonabelianBackend
from tenpy.linalg.symmetries import AbstractSymmetry


class NumpyBlockBackend(AbstractBlockBackend):

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

    def __init__(self):
        AbstractBlockBackend.__init__(self, default_precision=Precision.single)

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


class NoSymmetryNumpyBackend(NumpyBlockBackend, AbstractNoSymmetryBackend):
    def __init__(self):
        AbstractNoSymmetryBackend.__init__(self)
        NumpyBlockBackend.__init__(self)


class AbelianNumpyBackend(NumpyBlockBackend, AbstractAbelianBackend):
    def __init__(self, symmetry: AbstractSymmetry):
        AbstractAbelianBackend.__init__(self, symmetry=symmetry)
        NumpyBlockBackend.__init__(self)


class NonabelianNumpyBackend(NumpyBlockBackend, AbstractNonabelianBackend):
    def __init__(self, symmetry: AbstractSymmetry):
        AbstractNonabelianBackend.__init__(self, symmetry=symmetry)
        NumpyBlockBackend.__init__(self)
