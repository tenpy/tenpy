

class AbstractSymmetry:
    is_abelian = True
    ChargeType = None


    def __init__(self, descripte_name: str):
        # descriptive_name = e.g. "Sz", "ky", "Sz_parity", "TotalS"

    def fusion(self, charge1, charge2):
        return iterator


class NoSymmetry(AbstractSymmetry):
    """Trivial symmetry group that doesn't do anything"""
    is_abelian = True
    ChargeType = None

    def __init__(self, descriptive_name="NoSymmetry"):

    def fusion(self, charge, charge):
        yield None


class AbelianSymmetry:
    is_abelian = True
    ChargeType = int



class U_1(AbelianSymmetry):
    ...
    def fusion(self, charge1, charge2):
        yield charge1 + charge2

class Z_N(AbelianBackend):
    ...
    def fusion(self, charge1, charge2):
        yield (charge1 + charge2) % N


class ProductSymmetry:
    ChargeType = tuple

    def __init__(self, *symmetries):
        ...




class NonAbelianSymmetry:
    is_abelian = False




