
from ..symmetries.no_symmetry import NoSymmetryBackend

from .data_structure import DataStructure


class NumpyStructure(DataStructure):
    Block = np.ndarray
    ...

    def block_tensordot(A: np.ndarray, ....)
        return np.tensordot(A, B, ...)



class NoSymmetryNumpyBackend(NoSymmetryBackend, NumpyStructure):
    pass


class AbelianNumpyBackend(AbelianBackend, NumpyStructure):
    pass


class NonAbelianBackend(NonAbelianBackend, NumpyStructure):
    pass
