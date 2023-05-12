# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
import pytest
import numpy as np
from numpy.testing import assert_array_equal

from tenpy.linalg.symmetries import spaces, groups
from tenpy.linalg.backends import abelian


VectorSpace_classes = [spaces.VectorSpace, abelian.AbelianBackendVectorSpace]
ProductSpace_classes = [spaces.ProductSpace, abelian.AbelianBackendProductSpace]

symmetries = dict(
    no_symmetry=groups.no_symmetry,
    z4=groups.z4_symmetry,
    z4_named=groups.ZNSymmetry(4, 'foo'),
    z4_z5=groups.z4_symmetry * groups.z5_symmetry,
    z4_z5_named=groups.ZNSymmetry(4, 'foo') * groups.ZNSymmetry(5, 'bar'),
    # su2=groups.su2_symmetry,  # TODO (JU) : reintroduce once n symbol is implemented
)


def _get_four_sectors(symm: groups.Symmetry) -> groups.SectorArray:
    if isinstance(symm, groups.SU2Symmetry):
        res = np.arange(0, 8, 2, dtype=int)[:, None]
    elif symm.num_sectors >= 8:
        res = symm.all_sectors()[:8:2]
    elif symm.num_sectors >= 4:
        res = symm.all_sectors()[:4]
    else:
        res = np.tile(symm.all_sectors()[:, 0], 4)[:4, None]
    assert res.shape == (4, symm.sector_ind_len)
    return res


@pytest.mark.parametrize('symm', symmetries.keys())
@pytest.mark.parametrize('VectorSpace', VectorSpace_classes)
def test_vector_space(symm, VectorSpace):
    symm: groups.Symmetry = symmetries[symm]
    # TODO (JU) test real (as in "not complex") vectorspaces

    sectors = _get_four_sectors(symm)
    mults = [2, 1, 3, 5]
    s1 = VectorSpace(symmetry=symm, sectors=sectors, multiplicities=mults)
    s2 = VectorSpace.non_symmetric(dim=8)

    print('checking VectorSpace.sectors')
    assert_array_equal(s2.sectors, groups.no_symmetry.trivial_sector[None, :])
    assert_array_equal(s1.dual.sectors, symm.dual_sectors(s1.sectors))

    print('checking str and repr')
    _ = str(s1)
    _ = str(s2)
    _ = repr(s1)
    _ = repr(s2)

    print('checking duality and equality')
    assert s1 == s1
    assert s1 != s1.dual
    assert s1 != s2
    assert s1 != VectorSpace(symmetry=symm, sectors=sectors, multiplicities=[2, 1, 3, 6])
    assert s1.dual == VectorSpace(symmetry=symm, sectors=sectors, multiplicities=mults,
                                  _is_dual=True)
    assert s1.can_contract_with(s1.dual)
    assert not s1.can_contract_with(s1)
    assert not s1.can_contract_with(s2)

    print('checking is_trivial')
    assert not s1.is_trivial
    assert not s2.is_trivial
    assert VectorSpace.non_symmetric(dim=1).is_trivial
    assert VectorSpace(symmetry=symm, sectors=symm.trivial_sector[None, :]).is_trivial

    # TODO (JU) test num_parameters when ready

@pytest.mark.parametrize('symm', symmetries.keys())
@pytest.mark.parametrize('VectorSpace, ProductSpace',
                         zip(VectorSpace_classes, ProductSpace_classes))
def test_product_space(symm, VectorSpace, ProductSpace):
    symm: groups.Symmetry = symmetries[symm]
    # TODO (JU) test real (as in "not complex") vectorspaces

    sectors = _get_four_sectors(symm)
    s1 = VectorSpace(symmetry=symm, sectors=sectors, multiplicities=[2, 1, 3, 4])
    s2 = VectorSpace(symmetry=symm, sectors=sectors[:2], multiplicities=[2, 1])
    s3 = VectorSpace(symmetry=symm, sectors=sectors[::2], multiplicities=[3, 2])

    p1 = ProductSpace([s1, s2, s3])
    p2 = s1 * s2
    p3a = p2 * s3
    p3b = ProductSpace([ProductSpace([s1, s2]), s3])

    assert_array_equal(p1.sectors, p3a.sectors)
    assert_array_equal(p1.sectors, p3b.sectors)

    _ = str(p1)
    _ = str(p3a)
    _ = repr(p1)
    _ = repr(p3a)

    assert p1 == p1
    assert p1 != p3a
    assert p3a == p3b
    assert p2 == ProductSpace([s1.dual, s2.dual], _is_dual=True).dual
    for p in [p1, p2, p3a, p3b]:
        assert p.can_contract_with(p.dual)
    assert p2 == ProductSpace([s1, s2], _is_dual=True).flip_is_dual()
    assert p2.is_dual_of(ProductSpace([s1.dual, s2.dual]).flip_is_dual())
    assert p2.is_dual_of(ProductSpace([s1.dual, s2.dual], _is_dual=True))
    p1_s = p1.as_VectorSpace()
    assert isinstance(p1_s, VectorSpace)
    assert p1_s == p1
