from __future__ import annotations

from tenpy.linalg.backends import AbstractBackend, Dtype
from tenpy.linalg.symmetries import VectorSpace
from tenpy.linalg.tensors import Tensor


# FIXME dont do this
def uniform(backend: AbstractBackend, legs: list[VectorSpace], labels: list[str] = None, dtype: Dtype = None
            ) -> Tensor:
    """
    Generate a sample from the probability distribution which is uniform on all tensors with the given legs
    and norm <= 1
    """
    return Tensor(backend.random_uniform(legs, dtype), backend, legs=legs, leg_labels=labels, dtype=dtype)


def gaussian(backend: AbstractBackend, legs: list[VectorSpace], sigma: float = 1., labels: list[str] = None,
             dtype: Dtype = None) -> Tensor:
    """
    Generate a sample from the probability distribution p(T) ~ exp(-.5 * (norm(T) / sigma) ** 2)
    """
    return Tensor(backend.random_gaussian(legs, dtype, sigma), backend, legs=legs, leg_labels=labels, dtype=dtype)


# TODO Haar unitary, like tenpy OLD
# TODO random isometry
