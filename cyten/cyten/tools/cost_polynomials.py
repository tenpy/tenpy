"""Tools for expressing algorithmic costs as polynomials."""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

import itertools


class BigOMonomial:
    """A symbolic representation of an algorithmic cost as a monomial.

    A monomial is of the form ``x^a y^b z^c``, i.e. a product of integer powers.

    Attributes
    ----------
    factors : dict {str: int}
        The factor, where an entry ``{'x': n}`` represents the symbol factor ``x^n``.

    """

    def __init__(self, factors: dict[str, int]):
        self.factors = factors

    @classmethod
    def from_str(cls, mono: str):
        """Initialize from a string representation like ``'x^2 y^3'``."""
        if isinstance(mono, BigOMonomial):
            return mono
        mono = str(mono).strip()
        str_factors = mono.split(' ')
        factors = {}
        for f in str_factors:
            f = f.split('^')
            if len(f) == 1:
                dim = f[0]
                exp = 1
            elif len(f) == 2:
                dim = f[0]
                exp = int(f[1])
                assert exp > 0
            else:
                raise ValueError(f'Invalid monomial: "{mono}"')
            factors[dim] = factors.get(dim, 0) + exp
        return cls(factors=factors)

    def __add__(self, other):
        if not isinstance(other, BigOMonomial):
            return NotImplemented
        return BigOPolynomial([self, other])

    def __hash__(self):
        return hash(tuple(*self.factors.items()))

    def __mul__(self, other):
        if not isinstance(other, BigOMonomial):
            return NotImplemented
        factors = self.factors.copy()
        for s, e in other.factors.items():
            factors[s] = factors.get(s, 0) + e
        return BigOMonomial(factors)

    def __repr__(self):
        return f'<{type(self).__name__} {str(self)} >'

    def __str__(self):
        return ' '.join(f'{dim}^{exp}' for dim, exp in self.factors.items())

    def __eq__(self, other):
        if not isinstance(other, BigOMonomial):
            return NotImplemented
        for s, e in self.factors.items():
            if other.factors.get(s, 0) != e:
                return False
        for s, e in other.factors.items():
            if self.factors.get(s, 0) != e:
                return False
        return True

    def is_negligible(self, *others: BigOMonomial, relations=None):
        """If the given monomial is negligible compared to `others`, s.t. ``O(self + x) = O(x)``."""
        if relations is not None:
            raise NotImplementedError
        for o in others:
            if all(n <= o.factors.get(x, 0) for x, n in self.factors.items()):
                # can safely ignore any keys in o.factors that are not in self.factors
                return True
        return False


class BigOPolynomial:
    r"""A symbolic representation of an algorithmic cost as a monomial.

    A polynomial is a sum of :class:`BigOMonomials`\ s, i.e. it is of the form::

        x^a y^b + y^c z^d

    i.e. a sum of terms, which consist of integer powers of symbols.

    Polynomials can be added and multiplied and compared via :meth:`is_negligible`.

    Attributes
    ----------
    terms : list of BigOMonomial
        The terms such that the polynomial is their sum.

    """

    def __init__(self, terms: list[BigOMonomial] = None):
        if terms is None:
            terms = []
        self.terms = self.simplify_terms(terms)

    @staticmethod
    def simplify_terms(terms: list[BigOMonomial], relations: list[tuple[BigOMonomial, BigOMonomial]] = None):
        """Simplify a list of terms by dropping negligible terms."""
        non_negligible = []
        for t in terms:
            if not t.is_negligible(*non_negligible, relations=relations):
                non_negligible.append(t)
        return non_negligible

    @classmethod
    def from_str(cls, poly: str):
        """Initialize from a string representation like ``'x^2 y^3 + x^4'``."""
        if isinstance(poly, BigOPolynomial):
            return poly
        if isinstance(poly, BigOMonomial):
            return cls(terms=[poly])
        terms = poly.split('+')
        return cls(terms=[BigOMonomial.from_str(t.strip()) for t in terms])

    def __repr__(self):
        return f'<{type(self).__name__} {str(self)} >'

    def __str__(self):
        return ' + '.join(str(t) for t in self.terms)

    def __add__(self, other):
        if isinstance(other, str):
            other = BigOPolynomial.from_str(other)
        if isinstance(other, BigOMonomial):
            other = BigOPolynomial([other])
        if not isinstance(other, BigOPolynomial):
            return NotImplemented
        return BigOPolynomial([*self.terms, *other.terms])

    def __eq__(self, other):
        if isinstance(other, BigOMonomial):
            if len(self.terms) == 1:
                return self.terms[0] == other
            return False
        if not isinstance(other, BigOPolynomial):
            return NotImplemented
        for t in self.terms:
            # OPTIMIZE could impose a canonical order on terms, to save one loop...
            if not any(t == t2 for t2 in other.terms):
                return False
        for t2 in other.terms:
            if not any(t == t2 for t in other.terms):
                return False
        return True

    def __hash__(self):
        return hash(tuple(self.terms))

    def __mul__(self, other):
        if isinstance(other, str):
            other = BigOPolynomial.from_str(other)
        if isinstance(other, BigOMonomial):
            other = BigOPolynomial([other])
        if not isinstance(other, BigOPolynomial):
            return NotImplemented
        terms = [m1 * m2 for m1, m2 in itertools.product(self.terms, other.terms)]
        return BigOPolynomial(terms)

    __radd__ = __add__  # all allowed addition is commutative
    __rmul__ = __mul__  # all allowed multiplication is commutative

    def prod(self, *others):
        """Product of multiply symmetries"""
        if len(others) == 0:
            return self
        first, *more = others
        return (self * first).prod(*more)
