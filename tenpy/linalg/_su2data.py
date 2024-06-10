"""Compute symmetry data for SU(2)"""
# Copyright (C) TeNPy Developers, GNU GPLv3

from __future__ import annotations
import numpy as np
from functools import lru_cache

try:
    from sympy import S as sympy_S
    from sympy.physics.wigner import clebsch_gordan as sympy_cg, racah as sympy_racah
except:
    S = clebsch_gordan = racah = None


# TODO / OPTIMIZE : think about caching, pre-computing, ...
#                 : this is a quick and stupid implementation
CACHE_SIZE = 10_000


def as_j(a: int):
    """Convert sector label to sympy symbol for the spin quantum number"""
    return sympy_S(a) / 2


@lru_cache(maxsize=CACHE_SIZE)
def f_symbol(a: int, b: int, c: int, d: int, e: int, f: int) -> np.ndarray:
    """F symbols of the SU(2) group.

    Note: we need to take hashable inputs to use cache. numpy arrays are not hashable.

    Parameters
    ----------
    a, b, c, d, e, f : int
        Specifying the spin sectors. These are twice the spin quantum number, ``a == 2 * j_a``.

    Returns
    -------
    F : 4D array
        The F symbol as an array of the multiplicity indices [μ,ν,κ,λ]
    """
    # We define the F symbol as
    #  < ((j1 j2) J12, j3) J | (j1, (j2 j3) J23) J >
    # This is the Racah W symbol up to a normalization factor and up to complex conjugation.
    # The conjugation drops out, since the Racah W symbols are real.
    # j1, j2, j3 = a, b, c
    # J = d
    # J12 = f ; J23 = e
    sqrt_dim_e = np.sqrt(e + 1)  # dim = 2 * j_e + 1 = e + 1
    sqrt_dim_f = np.sqrt(f + 1)
    return sqrt_dim_e * sqrt_dim_f * racah_W(a, b, d, c, f, e) * np.ones((1, 1, 1, 1))


@lru_cache(maxsize=CACHE_SIZE)
def fusion_tensor(a: int, b: int, c: int) -> np.ndarray:
    dim_a = a + 1
    dim_b = b + 1
    dim_c = c + 1
    X = np.empty((1, dim_a, dim_b, dim_c), dtype=np.float64)
    for k_a in range(dim_a):
        for k_b in range(dim_b):
            for k_c in range(dim_c):
                X[0, k_a, k_b, k_c] = clebsch_gordan(a, k_a, b, k_b, c, k_c)
    return X


@lru_cache
def Z_iso(a: int) -> np.ndarray:
    d_a = a + 1  # 2 j_a + 1
    Z = np.zeros((d_a, d_a), dtype=float)
    for k in range(d_a):  # m == -j + k == -a/2 + k
        # OPTIMIZE can probably do this with pure numpy, no python loop...
        # Z[k, -k] = Z_{m,-m} = (-1) ** (j - m) / sqrt(2j + 1), do factor later
        # (-1) ** (j - m) == 1 - 2 * (j - m) % 2 = 1 - 2 * (a - k) % 2
        Z[k, d_a - 1 - k] = 1 - 2 * np.mod(a - k, 2)
    return Z


def clebsch_gordan(a: int, k_a: int, b: int, k_b: int, c: int, k_c: int):
    """The sectors are ``a == 2 * j_a``, and ``k_a = m_a + j_a = 0, 1, ..., 2 * j_a + 1``"""
    j_a = as_j(a)
    j_b = as_j(b)
    j_c = as_j(c)
    m_a = k_a - j_a
    m_b = k_b - j_b
    m_c = k_c - j_c
    cg = sympy_cg(j_a, j_b, j_c, m_a, m_b, m_c)
    return float(cg.doit())


def racah_W(jj1: int, jj2: int, JJ: int, jj3: int, JJ12: int, JJ23: int) -> float:
    """Inputs are sector labels, e.g. ``jj1 == 2 * j1``"""
    symbol = sympy_racah(as_j(jj1), as_j(jj2), as_j(JJ), as_j(jj3), as_j(JJ12), as_j(JJ23))
    return float(symbol.doit())
