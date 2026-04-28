"""Assertion wrappers for testing."""

# Copyright (C) TeNPy Developers, Apache license
from ..tensors import SymmetricTensor, almost_equal


def assert_tensors_almost_equal(a: SymmetricTensor, expect: SymmetricTensor, rtol: float = 1e-12, atol: float = 1e-12):
    """Verify two tensors have the same legs and almost equal numerical entries."""
    assert a.codomain == expect.codomain
    assert a.domain == expect.domain
    assert almost_equal(a, expect, rtol=rtol, atol=atol)
