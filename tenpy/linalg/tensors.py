"""TODO

>>> sym = U1("momentum")
>>> backend = get_backend(sym, 'numpy')
>>> leg_p = VectorSpace(backend, [1, -1], conj=False)
>>> Sz = Tensor.from_dense([[1., 0.], [0, -1.]], [leg_p, leg_p.conj()], ['p', 'p*'])
>>> Sz_2 = tensordot(Sz, Sz, 'p', 'p*')
>>> Sz_2 = tensordot(Sz, Sz, ['p'], ['p*'])
>>> Id = eye(leg_p, ['p'], ['p*'])
>>> assert all_close(Sz_2, Id , eps=1.e-10)
>>> Sz_2 + 5 * Id


"""


class Tensor:
    def __init__(self, legs: List[VectorSpaces], dt...):

        self.legs = legs
        self.backend = legs[0].backend
        self.data = ...

    @classmethod
    def from_dense(self, data, legs, ...):
        ...

    @classmethod
    def change_backend(self, new_backend, ...):
        ...

    @classmethod
    def drop_charges(self, new_backend=None, ...):
        ...


# subclass? should have drop_charges etc
class DiagonalTensor(Tensor):
    ...


def zeros(self, legs, ....):
    ...


def random_uniform_tensor(self, legs, ...)
    return Tensor(...)

def random_unitary_tensor(self, legs, ....):
    return Tensor(...)



class VectorSpace:
    def __init__(self, backend, charge_list, multiplicity_list=None, conj=False, **extra_data):
        self.backend = backend
        # symmetry_group from self.backend
        ...
        self.dimension = ....
        # (initialize self._conj as well)

    def conj(self, ):
        return self._conj




class ProductSpace(VectorSpace):
    ...



def tensordot(A: Tensor, B: Tensor, contract_A, contract_B):
    check_A_B_diagonal(A, B)
    contract_A = _parse_indices(A, contract_A)
    contract_B = _parse_indices(B, contract_B)
    assert A.backend == B.backend
    new_labels = _tensordot_new_labels(A, B, contract_A, contract_B)
    new_data = backend.tensordot(A, B, contract_A, contract_B, new_labels)
    return Tensor(...)



def svd(A: Tensor, labels_A, label_B, new_label, new_conj=False,
        trunc_params=None, ...):


def eigh(A: Tensor, labels_left, labels_right):
    ...



def get_backend(symmetry_group, data_structure='numpy'):
    Cls = find_subclasse(AbstractBackend, name)
    inst = Cls(...)
    ...  # cache
    return inst



class AbstractBackend:

    Data = np.ndarray

    def __init__(self, symmetry_group: AbstractSymmetry):
        # e.g. group = U(1) x Z_2
        ...

    def svd(A: Data, B: Data, ...):
        ...
    ...

    def shape(A: Data):
        return A.shape


    def __eq__(self, other):
         return self.name == other.name and self.groups == other.groups





class JaxStructure(DataStructure):
    ...










