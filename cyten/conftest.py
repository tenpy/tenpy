r"""Provide test configuration for backends etc.

Fixtures
--------

The following table summarizes the available fixtures.
There are three groups; miscellaneous independent fixtures, unconstrained fixtures
(with ``any`` in their name) and constrained fixtures (with ``compatible`` in their name).
The latter two groups are similar in terms of signature.
The unconstrained fixtures are intended if a test should be parametrized over possible
symmetry backends *or* over possible symmetries, but not both.
The constrained ("compatible") fixtures are intended if a test should be parametrized over
possible combinations of a symmetry backend *and* a symmetry it is compatible with.
They should not be mixed in any single test, as that would generate unnecessarily many tests.
Whenever applicable, the unconstrained fixtures should be preferred, since e.g. most symmetries
appear multiple times as values of ``compatible_symmetry`` (same argument for ``compatible_backend``).


=============================  ======================  ===========================================
Fixture                        Depends on / # cases    Description
=============================  ======================  ===========================================
np_random                      -                       A numpy random Generator. Use this for
                                                       reproducibility.
-----------------------------  ----------------------  -------------------------------------------
block_backend                  Generates ~2 cases      Goes over all block backends, as str
                                                       descriptions, valid for ``get_backend``.
-----------------------------  ----------------------  -------------------------------------------
any_symmetry_backend           Generates 3 cases       Goes over all symmetry backends, as str
                                                       descriptions, valid for ``get_backend``.
-----------------------------  ----------------------  -------------------------------------------
any_backend                    block_backend           Goes over all backends.
                               any_symmetry_backend
-----------------------------  ----------------------  -------------------------------------------
any_symmetry                   Generates ~10 cases     Goes over some representative symmetries.
-----------------------------  ----------------------  -------------------------------------------
make_any_sectors               any_symmetry            RNG for sectors with ``any_symmetry``.
                                                       Note that fewer than ``num`` may result.
                                                       ``make(num, sort=False)``
-----------------------------  ----------------------  -------------------------------------------
make_any_space                 any_symmetry            RNG for spaces with ``any_symmetry``.
                                                       ``make(max_sectors=5, max_mult=5, is_dual=None)``
-----------------------------  ----------------------  -------------------------------------------
make_any_block                 any_backend             RNG for blocks of ``any_backend``.
                                                       ``make(size, real=False)``
-----------------------------  ----------------------  -------------------------------------------
compatible_pairs               Generates ~20 cases     Not a public fixture, only generates
                                                       the cases. Compatible pairs are built like
                                                       combinations of ``any_symmetry_backend``
                                                       and ``any_symmetry``, constrained by
                                                       compatibility.
-----------------------------  ----------------------  -------------------------------------------
compatible_symmetry_backend    compatible_pairs        The symmetry backend of a compatible pair.
-----------------------------  ----------------------  -------------------------------------------
compatible_symmetry            compatible_pairs        The symmetry of a compatible pair.
-----------------------------  ----------------------  -------------------------------------------
compatible_backend             compatible_pairs        A backend that is compatible with
                               block_backend           ``compatible_symmetry``.
-----------------------------  ----------------------  -------------------------------------------
make_compatible_sectors        compatible_pairs        RNG for sectors with ``compatible_symmetry``.
                                                       Note that fewer than ``num`` may result.
                                                       ``make(num, sort=False)``
-----------------------------  ----------------------  -------------------------------------------
make_compatible_space          compatible_pairs        RNG for spaces with ``compatible_symmetry``.
                                                       ``make(max_sectors=5, max_mult=5, is_dual=None)``
-----------------------------  ----------------------  -------------------------------------------
make_compatible_block          compatible_backend      RNG for blocks with ``compatible_backend``.
                                                       ``make(size, real=False)``
-----------------------------  ----------------------  -------------------------------------------
make_compatible_tensor         compatible_backend      RNG for tensors with ``compatible_backend``.
                                                       Signature see below.
=============================  ======================  ===========================================

The function returned by the fixture ``make_compatible_tensor`` has the following inputs::

    codomain, domain:
        Both the domain and the codomain can be specified in the following four ways:
        1) ``None``.
            For the codomain, this means a random space if cls is DiagonalTensor or Mask.
            For SymmetricTensor or ChargedTensor, codomain may not be None
            For the domain, this means "the same as codomain" for DiagonalTensor, a random space
            that contains the codomain for Mask and an empty domain for Symmetric/Charged Tensor.
        2) an integer
            The respective number of legs is randomly generated.
        3) a list
            Each entry specifies a leg. It can already be a space. A str specifies the label
            for that leg and is otherwise equivalent to None. None means to generate a random leg
        4) a TensorProduct
            The finished (co)domain
        For Symmetric/Charged Tensor, if any legs are generated, this is done in such a way
        as to guarantee that the resulting tensor allows some blocks.
    labels: list[str | None] (default: all None)
        The labels for the resulting tensor. Note that labels can also be specified via (co)domain.
    dtype: Dtype
        The dtype for the tensor.
    device: str, optional
        The device for the tensor. Per default, use the default device of the backend.
    *
    like: Tensor, optional
        If given, the codomain, domain, labels, dtype and cls are taken to be the
        same as for `like` and th explicit arguments are ignored.
        For ChargedTensors, the same charge leg and state are initialized.
    max_blocks: int (default 5)
        The maximum number of blocks for the resulting tensor
    max_block_size: int (default 5)
        The maximum multiplicity of any sector.
    empty_ok: bool (default False)
        If an empty tensor (with no allowed blocks) is ok, or should raise.
    all_blocks: bool (default False)
        If all allowed blocks should be filled, or if some should be dropped randomly
    cls: Tensor subtype
        The type of tensor to create: SymmetricTensor, DiagonalTensor, Mask or ChargedTensor


Marks
-----
Note: a list of marks should also be maintained in ``pyproject.toml``.

- ``slow``: marks tests as slow (deselect with ``-m "not slow"``)
- ``FusionTree``: marks tests that use the FusionTreeBackend.
- ``numpy``: marks tests that use the numpy block backend.
- ``torch``: marks tests that use the torch block backend.


Deselecting invalid ChargedTensor cases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
There is a custom mark ``deselect_invalid_ChargedTensor_cases``.
It can be used as a decorator ``@pytest.mark.deselect_invalid_ChargedTensor_cases`` for test cases.
The intended use case is in a situation where a test is parametrized over multiple symmetries and
over multiple tensor types.
Some symmetries will then be incompatible with the ``ChargedTensor`` type.
Those cases should be deselected.

The decorator takes two optional keyword arguments.
``@pytest.mark.deselect_invalid_ChargedTensor_cases(get_cls: callable, get_sym: callable)``.
Both are functions, and during setup of pytest they are called as e.g. ``get_cls(kwargs)``
where kwargs are the explicit keyword arguments (e.g. parametrize keyword) of the test function.
They should return the tensor type, e.g. ``ChargedTensor`` and the symmetry instance respectively.
The default values are ``get_cls = lambda kw: kw['cls']``
and ``get_sym = lambda kw: kw['_compatible_backend_symm_pairs'][1]``.
As such, the default decorator with no arguments works in the most common design pattern for tests,
where the symmetry is determined by the fixtures (e.g. because ``make_compatible_tensor`` is used)
and the tensor cls comes from a parametrize with argname ``cls``.

"""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

import numpy as np
import pytest

import cyten as ct
from cyten import Dtype, backends, block_backends, tensors
from cyten.testing import random_block, random_ElementarySpace, random_symmetry_sectors, random_tensor

# OVERRIDE pytest routines


def pytest_addoption(parser):
    parser.addoption('--block-backends', action='store', default='numpy', help=f'Comma separated block-backend names')
    parser.addoption('--rng-seed', action='store', default=12345, type=int, help=f'The rng seed')


def pytest_collection_modifyitems(config, items):
    # deselection logic:
    removed = []
    kept = []
    for item in items:
        m = item.get_closest_marker('deselect_invalid_ChargedTensor_cases')
        if m:
            get_cls = m.kwargs.get('get_cls', lambda kw: kw['cls'])
            get_sym = m.kwargs.get('get_sym', lambda kw: kw['_compatible_backend_symm_pairs'][1])
            cls = get_cls(item.callspec.params)
            sym = get_sym(item.callspec.params)
            if cls is tensors.ChargedTensor and not tensors.ChargedTensor.supports_symmetry(sym):
                removed.append(item)
                continue
        kept.append(item)
    if removed:
        config.hook.pytest_deselected(items=removed)
        items[:] = kept


def pytest_generate_tests(metafunc):
    if 'block_backend' in metafunc.fixturenames:
        block_backends = metafunc.config.getoption('--block-backends').split(',')
        assert all(b in _block_backend_params for b in block_backends), str(block_backends)
        metafunc.parametrize('block_backend', block_backends)


# QUICK CONFIGURATION

_block_backend_params = dict(
    numpy=pytest.param('numpy', marks=pytest.mark.numpy),
    torch=pytest.param('torch', marks=pytest.mark.torch),
)
_symmetries = {
    # groups:
    'NoSymm': ct.no_symmetry,
    'U1': ct.u1_symmetry,
    'Z4_named': ct.ZNSymmetry(4, 'My_Z4_symmetry'),
    'U1xZ3': ct.ProductSymmetry([ct.u1_symmetry, ct.z3_symmetry]),
    'SU2': ct.SU2Symmetry(),
    # anyons:
    'fermion': ct.fermion_parity,
    'FibonacciAnyon': ct.fibonacci_anyon_category,
    'IsingAnyon': ct.ising_anyon_category,
    'Fib_U1': ct.fibonacci_anyon_category * ct.u1_symmetry,
}


# "UNCONSTRAINED" FIXTURES  ->  independent (mostly) of the other features. no compatibility guarantees.


@pytest.fixture
def np_random(request) -> np.random.Generator:
    return np.random.default_rng(seed=request.config.getoption('--rng-seed'))


@pytest.fixture  # values defined during `pytest_generate_tests`
def block_backend(request) -> str:
    if request.param == 'torch':
        _ = pytest.importorskip('torch', reason='torch not installed')
    return request.param


@pytest.fixture(params=['no_symmetry', 'abelian', pytest.param('fusion_tree', marks=pytest.mark.FusionTree)])
def any_symmetry_backend(request) -> str:
    return request.param


@pytest.fixture
def any_backend(block_backend, any_symmetry_backend) -> backends.TensorBackend:
    return backends.backend_factory.get_backend(any_symmetry_backend, block_backend)


@pytest.fixture(
    params=[s for s in _symmetries.values() if isinstance(s, ct.AbelianGroup)],
    ids=[k for k, s in _symmetries.items() if isinstance(s, ct.AbelianGroup)],
)
def abelian_group_symmetry(request) -> ct.Symmetry:
    return request.param


@pytest.fixture(
    params=[s for s in _symmetries.values() if s.can_be_dropped],
    ids=[k for k, s in _symmetries.items() if s.can_be_dropped],
)
def any_symmetry_that_can_be_dropped(request) -> ct.Symmetry:
    return request.param


@pytest.fixture(params=list(_symmetries.values()), ids=list(_symmetries.keys()))
def any_symmetry(request) -> ct.Symmetry:
    return request.param


@pytest.fixture
def make_any_sectors(any_symmetry, np_random):
    # if the symmetry does not have enough sectors, we return fewer!
    def make(num: int, sort: bool = False) -> ct.SectorArray:
        # return SectorArray
        return random_symmetry_sectors(any_symmetry, num, sort, np_random=np_random)

    return make


@pytest.fixture
def make_any_space(any_symmetry, np_random):
    def make(max_sectors: int = 5, max_mult: int = 5, is_dual: bool = None) -> ct.ElementarySpace:
        # return ElementarySpace
        return random_ElementarySpace(any_symmetry, max_sectors, max_mult, is_dual, np_random=np_random)

    return make


@pytest.fixture
def make_any_block(any_backend, np_random):
    def make(size: tuple[int, ...], real=False) -> block_backends.Block:
        # return Block
        return random_block(any_backend.block_backend, size, real=real, np_random=np_random)

    return make


# "COMPATIBLE" FIXTURES  ->  only go over those pairings of backend and symmetry that are compatible

# build the compatible pairs
_compatible_pairs = {'NoSymmetry': ('no_symmetry', ct.no_symmetry)}  # {id: param}
for _sym_name, _sym in _symmetries.items():
    if isinstance(_sym, ct.AbelianGroup):
        _compatible_pairs[f'AbelianBackend-{_sym_name}'] = ('abelian', _sym)
    _compatible_pairs[f'FusionTreeBackend-{_sym_name}'] = pytest.param(
        ('fusion_tree', _sym), marks=pytest.mark.FusionTree
    )


@pytest.fixture(params=list(_compatible_pairs.values()), ids=list(_compatible_pairs.keys()))
def _compatible_backend_symm_pairs(request) -> tuple[str, ct.Symmetry]:
    """Helper fixture that allows us to generate the *compatible* fixtures.

    Values are pairs (symmetry_backend: str, symmetry: Symmetry)
    """
    return request.param


@pytest.fixture
def compatible_symmetry_backend(_compatible_backend_symm_pairs) -> str:
    symmetry_backend, symmetry = _compatible_backend_symm_pairs
    return symmetry_backend


@pytest.fixture
def compatible_backend(compatible_symmetry_backend, block_backend) -> backends.TensorBackend:
    return backends.backend_factory.get_backend(compatible_symmetry_backend, block_backend)


@pytest.fixture
def compatible_symmetry(_compatible_backend_symm_pairs) -> ct.Symmetry:
    symmetry_backend, symmetry = _compatible_backend_symm_pairs
    return symmetry


@pytest.fixture
def make_compatible_sectors(compatible_symmetry, np_random):
    # if the symmetry does not have enough sectors, we return fewer!
    def make(num: int, sort: bool = False) -> ct.SectorArray:
        # returns SectorArray
        return random_symmetry_sectors(compatible_symmetry, num, sort, np_random=np_random)

    return make


@pytest.fixture
def make_compatible_space(compatible_symmetry, np_random):
    def make(
        max_sectors: int = 5, max_mult: int = 5, is_dual: bool = None, allow_basis_perm: bool = True
    ) -> ct.ElementarySpace:
        # returns ElementarySpace
        return random_ElementarySpace(
            compatible_symmetry, max_sectors, max_mult, is_dual, allow_basis_perm=allow_basis_perm, np_random=np_random
        )

    return make


@pytest.fixture
def make_compatible_block(compatible_backend, np_random):
    def make(size: tuple[int, ...], real: bool = False) -> block_backends.Block:
        # returns Block
        return random_block(compatible_backend.block_backend, size, real=real, np_random=np_random)

    return make


@pytest.fixture
def make_compatible_tensor(compatible_backend, compatible_symmetry, np_random):
    """Tensor RNG."""

    def make(
        codomain: list[ct.Space | str | None] | ct.TensorProduct | int = None,
        domain: list[ct.Space | str | None] | ct.TensorProduct | int = None,
        labels: list[str | None] = None,
        dtype: Dtype = None,
        device: str = None,
        *,
        like: tensors.Tensor = None,
        max_blocks=5,
        max_block_size=5,
        empty_ok=False,
        all_blocks=False,
        cls=tensors.SymmetricTensor,
        allow_basis_perm: bool = True,
        use_pipes: bool | float = 0.3,
    ):
        return random_tensor(
            symmetry=compatible_symmetry,
            codomain=codomain,
            domain=domain,
            labels=labels,
            dtype=dtype,
            backend=compatible_backend,
            device=device,
            like=like,
            max_blocks=max_blocks,
            max_multiplicity=max_block_size,
            empty_ok=empty_ok,
            all_blocks=all_blocks,
            cls=cls,
            allow_basis_perm=allow_basis_perm,
            use_pipes=use_pipes,
            np_random=np_random,
        )

    return make
