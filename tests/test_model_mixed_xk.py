from tenpy.models.mixed_xk import (MixedXKLattice, MixedXKModel, SpinlessMixedXKSquare,
                                   HubbardMixedXKSquare)
from test_model import check_general_model
import pytest


@pytest.mark.parametrize('model_class', [SpinlessMixedXKSquare, HubbardMixedXKSquare])
def test_MixedXKModel_general(model_class):
    check_general_model(model_class, dict(Lx=2, Ly=3, bc_MPS='infinite'))


# TODO: add more tests
