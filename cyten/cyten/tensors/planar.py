"""TODO"""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal, TypeVar, overload

import numpy as np

from ..backends import FusionTreeBackend
from ..tools import BigOPolynomial, duplicate_entries
from ._tensors import (
    CONTRACT_SYMBOL,
    LEG_SELECT_SYMBOL,
    OPEN_LEG_SYMBOL,
    LabelledLegs,
    Tensor,
    compose,
    is_valid_leg_label,
    partial_trace,
    permute_legs,
)
from .sparse import LinearOperator

NestedContainer_str = TypeVar('NestedContainer_str')


class TensorPlaceholder(LabelledLegs):
    """Placeholder for a tensor used to define :class:`PlanarDiagram` s.

    Attributes
    ----------
    labels : list of str
        The labels of the tensor (up to cyclic permutation). This means that as long as we go
        clockwise around the shape, any starting point can be chosen for the labels.
    dims : list of (str | None)
        For each of the legs, an optional symbol to represent its dimension.

    """

    def __init__(self, labels: list[str], dims: list[str] = None, cost_to_make: BigOPolynomial = None):
        assert None not in labels
        if dims is None:
            dims = [None] * len(labels)
        else:
            assert len(dims) == len(labels)
        self.dims = [BigOPolynomial.from_str(d) for d in dims]
        if cost_to_make is None:
            cost_to_make = BigOPolynomial()
        self.cost_to_make = cost_to_make
        LabelledLegs.__init__(self, labels)

    def copy(self, deep=True):
        # note: accessing the self.labels property already makes a copy of the list.
        return TensorPlaceholder(self.labels, self.dims[:], cost_to_make=self.cost_to_make)

    def __repr__(self):
        return f'TensorPlaceholder({self.labels}, dims=[{", ".join(map(str, self.dims))}])'


class PlanarDiagram:
    """Abstract representation for the contraction of multiple tensors without any braids.

    Random notes (TODO elaborate)::

        - abstract representation of the connectivity in terms of string labels
        - Use :meth:`evaluate`, or directly call the diagram (``__call__``) to actually do contractions
        - we only care about the order of legs around a tensor (counter-clockwise),
        with arbitrary starting point. I.e. no need to care about codomain / domain
        - Intended use: make instances in a module, such that they are instantiated at import-time.
        - Optimization of contraction order can be expensive in some cases.
        Intended workflow: run optimizing once during development and hard-code it.
        Fallback: run greedy optimization when the diagram is instantiated

    Parameters
    ----------
    tensors : str or {str: TensorPlaceholder}
        Specifies the tensors in the diagram, each with leg labels and a unique name.
        Syntax for string input: a comma (`,`) separated list of entries, each for one tensor.
        The entry for a tensor is its name, followed by comma separated leg labels enclosed in
        brackets. Example: ``'theta[vL, p0, p1, vR], U[p0, p1, p1*, p0*]'``.
        The same format as the attribute :attr:`tensors` (dict) is accepted as well.
    definition : str or list of (str, str, str | None, str)
        Specifies the diagram, i.e. how the `tensors` are contracted.
        Syntax for string input: a comma (`,`) separated list of instructions, each either
        a contraction or an open leg.
        Contractions are of the form ``'{tensorA}:{legA} @ {tensorB}:{legB}'``.
        Open legs are of the form ``'{tensorA}:{legA} -> {new_label}``.
        The same format as the attribute :attr:`definition` (list of tuples) is accepted as well.
    dims : {str: list of str}, optional
        Specifies a symbol for the dimension of each leg, used to show or optimize the contraction
        cost in terms of a :class:`BigOPolynomial`.
        A dictionary with pairs ``{dim: labels}`` indicating that the legs with ``labels`` have
        a dimension represented by the symbol ``dim``. If given, *all* labels in the diagram should
        be assigned to a symbol. Legs with the same label must have the same dimension.
    order : 'greedy' | 'optimal' | 'definition' | str | nested tuples of str | ContractionTree
        Specifies the contraction order, or how to determine it.
        If ``'greedy'`` (default) or ``'optimal'``, it is optimized via :meth:`optimize_order`.
        If ``'definition'``, it is taken from the order of the `definition`, with minimal extra
        optimizations (always do traces first and when contracting two tensors, contract all shared
        legs at once).
        If a single string, expect a comma separated list of instructions
        ``'{tensorA} @ {tensorB}'`` which indicate the order of pairwise contractions.
        If nested tuples of strings, interpret those strings as tensor names, and interpret
        the bracketing as the order of pairwise contractions, contracting innermost tuples first.
        The same format as the attribute :attr:`order` (``ContractionTree``) is accepted as well.

    Attributes
    ----------
    tensors : {str: TensorPlaceholder}
        The tensors in the diagram, as a dictionary from name to its placeholder, which stores
        leg labels and dims.
    definition : list of (str, str, str | None, str)
        Defines the contractions in the diagram.
        An entry ``(t1, l1, t2, l2)`` indicates to contract leg ``l1`` of ``tensors[t1]`` with
        leg ``l2`` of ``tensors[t2]``.
        An entry ``(t1, l1, None, new_l)`` indicates that leg ``l1`` of ``tensors[t1]`` is an open
        leg of the diagram and should have label ``new_l`` in the result.
    order : ContractionTree
        Specifies the order for the contractions during :meth:`evaluate`.
    open_legs : list of str
        The open legs of the diagram, up to cyclical permutation.
        This is such that the result of :meth:`evaluate` has these leg labels (up to cycl. perm.).

    Examples
    --------
    TODO

    """

    def __init__(
        self,
        tensors: str | dict[str, TensorPlaceholder],
        definition: str | list[tuple[str, str, str | None, str]],
        dims: dict[str, Sequence[str]] = None,
        order: str | NestedContainer_str | ContractionTree = 'definition',
    ):
        self.tensors = self.parse_tensors(tensors, dims)
        self.definition = self.parse_definition(definition)
        self.order = self.parse_order(order)
        self.open_legs, self.contraction_cost = self.verify_diagram()

    @property
    def tensor_names(self) -> list[str]:
        return list(self.tensors.keys())

    def add_tensor(
        self,
        tensor: str | dict[str, TensorPlaceholder],
        extra_definition: str | list[tuple[str, str, None, str]],
        extra_dims: dict[str, Sequence[str]] = None,
        order: str | NestedContainer_str | ContractionTree = 'definition',
    ) -> PlanarDiagram:
        """Create a new diagram with an additional tensor.

        TODO should we allow to reference the existing diagram as a whole, instead of its
             individual tensors?

        Parameters
        ----------
        tensor : str (or {str: TensorPlaceholder})
            Same as the parameter to :class:`PlanarDiagram`, but expect only a single tensor,
            to be added to the diagram
        extra_definition : str (or list of (str, str, str | None, str))
            Same as the parameter to :class:`PlanarDiagram`.
            Should define what each leg of the new tensor does; either contracted or open.
            The new :attr:`definition` is given by this extra definition together with the old
            definition, except for entries that correspond to legs that were open in the original
            diagram and are now contracted with the new tensor.
        extra_dims : {str: list of str}, optional
            Same as the parameter to :class:`PlanarDiagram`, but applies only to the new `tensor`.
        order : 'greedy' | 'optimal' | 'definition' | str | nested tuples of str
            Same as the parameter to :class:`PlanarDiagram`, applies to the entire new diagram.

        """
        extra_tensors = self.parse_tensors(tensor, extra_dims)
        assert len(extra_tensors) == 1
        new_name = next(iter(extra_tensors))
        if new_name in self.tensors:
            raise ValueError('There already is a tensor with that name')
        tensors = self.tensors.copy()
        tensors.update(extra_tensors)

        outdated = []  # collect indices of the old_definitions that are outdated in the new
        extra_definition = self.parse_definition(extra_definition)
        for t1, l1, t2, l2 in extra_definition:
            if t2 is None:
                continue  # new open leg: nothing to do
            if t1 == new_name and t2 == new_name:
                continue  # trace on the new tensor: nothing to do
            if t1 == new_name:
                new_tens_leg = l1
                other_tens = t2
                other_tens_leg = l2
            elif t2 == new_name:
                new_tens_leg = l2
                other_tens = t1
                other_tens_leg = l1
            else:
                raise ValueError('Invalid extra_definition. Must reference the new tensor!')
            n = self._find_open_leg_definition(other_tens, other_tens_leg)
            if n is None:
                msg = (
                    f'Invalid extra_definition. Attempted to contract '
                    f'{new_name}:{new_tens_leg} @ {other_tens}:{other_tens_leg}, but the latter '
                    f'is not an open leg of the existing diagram'
                )
                raise ValueError(msg)
            outdated.append(n)
        definition = [d for n, d in enumerate(self.definition) if n not in outdated] + extra_definition
        return PlanarDiagram(tensors=tensors, definition=definition, dims=None, order=order)

    @overload
    def evaluate(
        self, tensors: dict[str, Tensor]
    ) -> Tensor: ...  # this stub exists for type hints only, definition is below

    @overload
    def evaluate(
        self, tensors: dict[str, TensorPlaceholder]
    ) -> TensorPlaceholder: ...  # this stub exists for type hints only, definition is below

    def evaluate(self, tensors: dict[str, Tensor]) -> Tensor:
        """Do the contractions defined by the diagram for given concrete `tensors`."""
        assert tensors.keys() == self.tensors.keys(), 'Invalid tensor names (keys)'
        for name, t in tensors.items():
            ph = self.tensors[name]
            try:
                roll = ph.labels.index(t.labels[0])
            except ValueError:
                msg = f'Mismatching labels on "{name}". Expected {ph.labels} up to cyclical permutation. Got {t.labels}'
                raise ValueError(msg) from None
            expect_labels = [*ph.labels[roll:], *ph.labels[:roll]]
            if t.labels != expect_labels:
                msg = f'Mismatching labels on "{name}". Expected {expect_labels}. Got {t.labels}'
                raise ValueError(msg)

        # relabel such that labels are globally unique
        # (prepend the name of the tensor it was originally on)
        tensors = {name: t.copy().relabel({l: f'{name}:{l}' for l in t.labels}) for name, t in tensors.items()}
        traces = []
        contractions = []
        open_legs = []
        for t1, l1, t2, l2 in self.definition:
            if t2 is None:
                open_legs.append((f'{t1}:{l1}', l2))
            elif t1 == t2:
                traces.append((t1, f'{t1}:{l1}', f'{t1}:{l2}'))
            else:
                contractions.append((t1, f'{t1}:{l1}', t2, f'{t2}:{l2}'))

        self._do_traces(tensors, traces)
        self._do_contractions(tensors, contractions, self.order)
        tensor = self._extract_result(tensors, open_legs)

        return tensor

    def optimize_order(self, strategy: Literal['greedy', 'optimal']) -> ContractionTree:
        """Find the optimal contraction order for the given diagram.

        TODO make it easy to print what you need to hard-code.
        TODO allow relations like ``d < w < chi``, or ``d^2 < chi`` to simplify the polynomials.
        TODO support cost as polynomials or with concrete numbers
        """
        if strategy == 'greedy':
            # falling back on order "by definition" as a very greedy optimization as a temp solution
            return self.parse_order('definition')
        raise NotImplementedError('Optimization of contraction order is not supported yet')

    @staticmethod
    def parse_definition(
        definition: str | list[tuple[str, str, str | None, str]],
    ) -> list[tuple[str, str, str | None, str]]:
        if not isinstance(definition, str):
            for x in definition:
                assert len(x) == 4
                t1, l1, t2, l2 = x
                assert t1 == _as_valid_name(t1)
                assert l1 == _as_valid_name(l1)
                assert t2 is None or t2 == _as_valid_name(t2)
                assert l2 == _as_valid_name(l2)
            return definition

        res = []
        for i in definition.split(','):
            i = i.strip()
            if CONTRACT_SYMBOL in i:
                res.append(PlanarDiagram._parse_contract_instruction(i))
            elif OPEN_LEG_SYMBOL in definition:
                res.append(PlanarDiagram._parse_open_leg_instruction(i))
            else:
                raise ValueError(f'Invalid syntax: "{i}"')
        return res

    def parse_order(self, order: str | NestedContainer_str | ContractionTree):
        if order == 'definition':
            order = [(t1, t2) for t1, l1, t2, l2 in self.definition if t2 is not None]
            return ContractionTree.from_contraction_order(order)
        if order in ['greedy', 'optimal']:
            return self.optimize_order(strategy=order)
        if isinstance(order, str):
            contraction_order = []
            for i in order.split(','):
                parts = i.split(CONTRACT_SYMBOL)
                if len(parts) != 2:
                    raise ValueError(f'Invalid syntax for order: {i}')
                contraction_order.append((_as_valid_name(parts[0]), _as_valid_name(parts[1])))
            return ContractionTree.from_contraction_order(contraction_order)
        return ContractionTree.from_nested_containers(order)

    @staticmethod
    def parse_tensors(
        tensors: str | dict[str, TensorPlaceholder], dims: dict[str, Sequence[str]] | None
    ) -> dict[str, TensorPlaceholder]:
        """Parse the input format for the ``tensors`` arg to :class:`PlanarDiagram`."""
        if isinstance(tensors, dict):
            assert all(isinstance(key, str) for key in tensors.keys())
            assert all(isinstance(value, TensorPlaceholder) for value in tensors.values())
            if dims is not None:
                warnings.warn('dims are ignored if tensors is given as a dict')
            return tensors
        if not isinstance(tensors, str):
            raise TypeError(f'Expected dict or str. Got {type(tensors).__name__}')

        tensors = {name: legs for name, legs in _split_tensor_text(tensors)}

        if dims is None:
            leg_label_to_dim = {}
        else:
            leg_label_to_dim = {}
            for dim, labels in dims.items():
                for l in labels:
                    leg_label_to_dim[l] = dim
            all_leg_labels = [l for legs in tensors.values() for l in legs]
            defined = list(leg_label_to_dim.keys())
            undefined = [l for l in all_leg_labels if l not in defined]
            unused = [l for l in defined if l not in all_leg_labels]
            if len(undefined) > 0:
                msg = f'If dims are specified, all must be specified. Missing: {", ".join(undefined)}'
                raise ValueError(msg)
            if any(l not in defined for l in all_leg_labels):
                msg = f'The following leg labels were given in dims, but do not exist: {", ".join(unused)}'
                warnings.warn(msg, UserWarning, stacklevel=3)
        res = {}
        for name, legs in tensors.items():
            t = TensorPlaceholder(legs, [leg_label_to_dim.get(l, '?') for l in legs])
            res[name] = t
        return res

    def remove_tensor(
        self,
        name: str,
        extra_definition: str | list[tuple[str, str, None, str]] = [],
        order: str | NestedContainer_str | ContractionTree = 'greedy',
    ) -> PlanarDiagram:
        """Create a new diagram, with one tensor removed.

        Parameters
        ----------
        name : str
            The name of the tensor to be removed.
        extra_definition : str (or list of (str, str, None, str))
            Extra instructions to be added to the :attr:`definition`. Expect only for open legs.
            Same format as the `definition` parameter to :class:`PlanarDiagram`.
        order : 'greedy' | 'optimal' | 'definition' | str | nested tuples of str
            Same as the parameter to :class:`PlanarDiagram`, applies to the entire new diagram.

        """
        if name not in self.tensors:
            raise ValueError(f'Tensor does not exist: {name}')
        tensors = {n: ph for n, ph in self.tensors.items() if n != name}
        definition = []
        new_open_legs = []
        for t1, l1, t2, l2 in self.definition:
            if (t1 == name and t2 == name) or (t1 == name and t2 is None):
                # partial trace or open leg of removed tensor
                pass
            elif t1 == name:
                new_open_legs.append((t2, l2))
            elif t2 == name:
                new_open_legs.append((t1, l1))
            else:
                definition.append((t1, l1, t2, l2))
        for t1, l1, t2, l2 in self.parse_definition(extra_definition):
            if t2 is not None:
                raise ValueError('extra_definition may only contain open legs')
            if (t1, l1) in new_open_legs:
                new_open_legs.remove((t1, l1))
                definition.append((t1, l1, t2, l2))
            else:
                raise ValueError(
                    'extra_definition may only refer to legs previously contracted with the removed tensor.'
                )
        for t1, l1 in new_open_legs:
            # unspecified open legs, just keep their label
            definition.append((t1, l1, None, l1))
        return PlanarDiagram(tensors=tensors, definition=definition, dims=None, order=order)

    def verify_diagram(self) -> tuple[list[str], BigOPolynomial]:
        """Verify the definition of the diagram. Returns the :attr:`open_legs`.

        Returns
        -------
        open_legs : list of str
            The leg labels of a result of :meth:`evaluate`.
        cost : BigOPolynomial
            The cost to contract the diagram, as a polynomial in terms of the dims.

        """
        for t1, l1, t2, l2 in self.definition:
            assert t1 in self.tensors, f'No tensor with name {t1}'
            assert l1 in self.tensors[t1].labels, f'Tensor {t1} has no leg {l1}'
            if t2 is None:
                assert is_valid_leg_label(l2), f'Invalid leg label {l2}'
            else:
                assert t2 in self.tensors, f'No tensor with name {t2}'
                assert l2 in self.tensors[t2].labels, f'Tensor {t2} has no leg {l2}'

        # run the contraction with placeholders.
        # - verifies if the contractions actually are planar
        # - figures out the open_legs
        # - figures out the cost
        res = self.evaluate(self.tensors)
        return res.labels, res.cost_to_make

    def __call__(self, **tensors: Tensor):
        return self.evaluate(tensors=tensors)

    @overload
    @staticmethod
    def _do_contractions(
        tensors: dict[str, Tensor], contractions: list[tuple[str, str, str, str]], order: ContractionTree
    ) -> dict[str, Tensor]: ...  # this stub exists for type hints only, definition is below

    @overload
    @staticmethod
    def _do_contractions(
        tensors: dict[str, TensorPlaceholder], contractions: list[tuple[str, str, str, str]], order: ContractionTree
    ) -> dict[str, TensorPlaceholder]: ...  # this stub exists for type hints only, definition is below

    @staticmethod
    def _do_contractions(
        tensors: dict[str, Tensor], contractions: list[tuple[str, str, str, str]], order: ContractionTree
    ) -> dict[str, Tensor]:
        """Helper for :meth:`evaluate`. Do pairwise contractions.

        Parameters
        ----------
        tensors : {str: Tensor}
            The input tensors.
            Is modified in-place, removing used tensors and adding partial contraction results,
            until it eventually only contains one tensors, the overall result.
        contractions : list of (str, str, str, str)
            List of ``(name1, l1, name2, l2)`` indicating a contraction of leg ``l1`` on tensor
            ``name1`` with ``l2`` on ``name2``. Note that a pair ``(name1, name2)`` may appear
            multiple times if multiple legs between them are connected.
        order : ContractionTree
            The contraction order. Is explicitly copied before we modify in-place.

        """
        order = order.copy()
        while len(tensors) > 1:
            _, t_a, t_b, res_name = order.pop_contraction()
            legs_a = []
            legs_b = []
            contractions_done = []
            for n, (t1, l1, t2, l2) in enumerate(contractions):
                if (t1, t2) == (t_a, t_b):
                    legs_a.append(l1)
                    legs_b.append(l2)
                    contractions_done.append(n)
                elif (t1, t2) == (t_b, t_a):
                    legs_a.append(l2)
                    legs_b.append(l1)
                    contractions_done.append(n)

            # put contraction result as t_a, delete t_b
            tensors[res_name] = planar_contraction(tensors[t_a], tensors[t_b], legs_a, legs_b)
            tensors.pop(t_a)
            tensors.pop(t_b)
            # remove the used contractions
            contractions = [i for n, i in enumerate(contractions) if n not in contractions_done]
            # contractions involving t_a, t_b now need to reference res_name instead
            contractions = [
                (res_name if t1 in [t_a, t_b] else t1, l1, res_name if t2 in [t_a, t_b] else t2, l2)
                for t1, l1, t2, l2 in contractions
            ]
        return tensors

    @overload
    @staticmethod
    def _do_traces(
        tensors: dict[str, Tensor], traces: list[tuple[str, str, str]]
    ) -> dict[str, Tensor]: ...  # this stub exists for type hints only, definition is below

    @overload
    @staticmethod
    def _do_traces(
        tensors: dict[str, TensorPlaceholder], traces: list[tuple[str, str, str]]
    ) -> dict[str, TensorPlaceholder]: ...  # this stub exists for type hints only, definition is below

    @staticmethod
    def _do_traces(tensors: dict[str, Tensor], traces: list[tuple[str, str, str]]) -> dict[str, Tensor]:
        """Helper for :meth:`evaluate`. Do partial traces on single tensors.

        Parameters
        ----------
        tensors : {str: Tensor}
            The input tensors.
            Is modified in-place, replacing input tensors with traced tensors.
        traces : list of (str, str, str)
            List of ``(name, l1, l2)`` indicating that ``l1`` should be connected with ``l2`` on
            tensor ``name``. Note that a ``name`` may appear multiple times!

        """
        combined_traces = {}  # {name: [(l1, l2), ...]}
        for name, l1, l2 in traces:
            combined_traces[name] = combined_traces.get(name, []) + [(l1, l2)]
        for name, pairs in combined_traces.items():
            t = planar_partial_trace(tensors[name], *pairs)
            tensors[name] = t

    @overload
    @staticmethod
    def _extract_result(
        tensors: dict[str, Tensor], open_legs: list[tuple[str, str]]
    ) -> Tensor: ...  # this stub exists for type hints only, definition is below

    @overload
    @staticmethod
    def _extract_result(
        tensors: dict[str, TensorPlaceholder], open_legs: list[tuple[str, str]]
    ) -> TensorPlaceholder: ...  # this stub exists for type hints only, definition is below

    @staticmethod
    def _extract_result(tensors: dict[str, Tensor], open_legs: list[tuple[str, str]]) -> Tensor:
        """Helper for :meth:`evaluate`. Extract result from single-entry dict and relabel."""
        assert len(tensors) == 1
        tens = next(iter(tensors.values()))
        if len(open_legs) == 0:
            # result is a number
            # TODO this may change, see Issue 13 on Github
            return tens
        assert tens.labels_are(*(old for old, _ in open_legs))
        return tens.relabel({old: new for old, new in open_legs})

    @staticmethod
    def _parse_contract_instruction(i: str) -> tuple[str, str, str, str]:
        left, right, *more = i.split(CONTRACT_SYMBOL)
        left_parts = left.split(LEG_SELECT_SYMBOL)
        right_parts = right.split(LEG_SELECT_SYMBOL)
        if len(more) > 0 or len(left_parts) != 2 or len(right_parts) != 2:
            raise ValueError(str(i))
        t1 = _as_valid_name(left_parts[0])
        l1 = _as_valid_name(left_parts[1])
        t2 = _as_valid_name(right_parts[0])
        l2 = _as_valid_name(right_parts[1])
        return t1, l1, t2, l2

    @staticmethod
    def _parse_open_leg_instruction(i: str) -> tuple[str, str, None, str]:
        left, right, *more = i.split(OPEN_LEG_SYMBOL)
        left_parts = left.split(LEG_SELECT_SYMBOL)
        if len(more) > 0 or len(left_parts) != 2:
            raise ValueError(str(i))
        t1 = _as_valid_name(left_parts[0])
        l1 = _as_valid_name(left_parts[1])
        l2 = _as_valid_name(right)
        return t1, l1, None, l2

    def _find_open_leg_definition(self, name: str, leg: str) -> int | None:
        """Find an open leg in the :attr:`definition`.

        Returns
        -------
        idx : int | None
            If there is an entry ``(name, leg, None, any_new_label)`` in :attr:`definition`,
            return its index, or ``None`` if there is no such entry.

        """
        for n, (t1, l1, t2, _) in enumerate(self.definition):
            if t2 is None and t1 == name and l1 == leg:
                return n
        return None


def _as_valid_name(name: str) -> str:
    """Strip whitespace and check the name is valid as a tensor name or leg label"""
    name = str(name).strip()
    assert is_valid_leg_label(name)
    return name


def _split_tensor_text(text: str) -> list[tuple[str, list[str]]]:
    """Split up text that appears as the `tensors` input to :class:`PlanarDiagram`.

    Input is expected to be of the form::

        ', '.join(name + '[' + ', '.join(labels) + ']' for name, labels in ???)

    up to whitespace. Output is then::

        [(name, labels) for name, labels in ???]

    """
    # A[a, b, c], B[a, s, x], ...
    res = []
    done = -1  # should only consider the rest of text[done + 1:]
    for _ in range(10_000):  # should be broken, number is just to avoid infinite loop
        i = text.find('[', done + 1)  # find only in text[done:]
        j = text.find(']', done + 1)  # find only in text[done:]
        if i == -1:
            raise ValueError('Invalid syntax')
        if j == -1:
            raise ValueError('Bracket opened but not closed.')
        tensor_name = _as_valid_name(text[done + 1 : i].strip())
        legs = [_as_valid_name(l) for l in text[i + 1 : j].split(',')]
        res.append((tensor_name, legs))
        next_comma = text.find(',', j + 1)
        if next_comma == -1:
            done = j
            break
        done = next_comma
    if len(text[done + 1 :].strip()) > 0:
        raise ValueError('Invalid syntax')
    return res


class ContractionTreeNode:
    """Node in a :class:`ContractionTree`."""

    def __init__(
        self,
        parent: ContractionTreeNode | None,
        left_child: ContractionTreeNode | None,
        right_child: ContractionTreeNode | None,
        value: str | None,
    ):
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child
        self.value = value
        if (self.left_child is None) != (self.right_child is None):
            raise ValueError('Must have either none or two child nodes')

    def test_sanity(self):
        """Perform sanity checks."""
        if self.left_child is None and self.right_child is None:
            pass
        elif self.left_child is not None and self.right_child is not None:
            self.left_child.test_sanity()
            self.right_child.test_sanity()
        else:
            raise ValueError('Must have either none or two child nodes')

    @property
    def is_leaf(self):
        return self.left_child is None and self.right_child is None

    def copy(self, parent=None):
        """Implement :meth:`ContractionTree.copy` recursively."""
        if self.left_child is None:
            left_child = None
        else:
            left_child = self.left_child.copy(parent=self)
        if self.right_child is None:
            right_child = None
        else:
            right_child = self.right_child.copy(parent=self)
        return ContractionTreeNode(parent=parent, left_child=left_child, right_child=right_child, value=self.value)

    def get_leaves(self) -> tuple[list[str], int]:
        """Returns ``leaves, num_nodes_below``"""
        if self.is_leaf:
            return [self.value], 0
        leaves_L, num_L = self.left_child.get_leaves()
        leaves_R, num_R = self.right_child.get_leaves()
        return [*leaves_L, *leaves_R], 2 + num_L + num_R

    def remove_children(self) -> tuple[str, str]:
        """Remove both children and return their values."""
        assert not self.is_leaf
        a = self.left_child.value
        b = self.right_child.value
        self.left_child.parent = None
        self.right_child.parent = None
        self.left_child = None
        self.right_child = None
        return a, b

    def pop_contraction(self) -> tuple[None, str, str, str]:
        """Implement :meth:`ContractionTree.pop_contraction` recursively."""
        if self.is_leaf:
            raise ValueError('Can not pop a contraction from a single leaf')
        if not self.left_child.is_leaf:
            return self.left_child.pop_contraction()
        if not self.right_child.is_leaf:
            return self.right_child.pop_contraction()
        # remaining case: both children are leaves
        X = self.value
        a, b = self.remove_children()
        self.value = new_value = f'{a} @ {b}'
        return X, a, b, new_value

    def _str_lines(self, prefix_0: str = '', prefix: str = '') -> list[str]:
        if self.is_leaf:
            return [prefix_0 + str(self.value)]
        return [
            prefix_0 + '┓' if self.value is None else str(self.value),
            *self.left_child._str_lines(prefix_0=prefix + '┣━', prefix=prefix + '┃ '),
            *self.right_child._str_lines(prefix_0=prefix + '┗━', prefix=prefix + '  '),
        ]

    def show_whole_tree(self):
        root = self
        while root.parent is not None:
            root = root.parent
        return '\n'.join(root._str_lines())


class ContractionTree:
    """Representation of the contraction order in a :class:`PlanarDiagram` as a tree structure.

    The leaf nodes represent the tensor names in a diagram and the tree structure indicates an
    order of pairwise contractions.

    The values of non-leaf nodes currently have no meaning and are always set to ``None``,
    but may cary extra information about leg handling during a pairwise contraction in the future.
    """

    def __init__(self, root: ContractionTreeNode[str, None]):
        self.root = root

    def test_sanity(self):
        """Perform sanity checks."""
        self.root.test_sanity()

    @property
    def leaves(self) -> list[str]:
        leaves, _ = self.root.get_leaves()
        return leaves

    @property
    def num_leaves(self) -> int:
        return len(self.leaves)

    @property
    def num_nodes(self) -> int:
        _, num_nodes_below = self.root.get_leaves()
        return 1 + num_nodes_below

    @property
    def num_inner_nodes(self) -> int:
        leaves, num_nodes_below = self.root.get_leaves()
        return 1 + num_nodes_below - len(leaves)

    @classmethod
    def from_contraction_order(cls, order: list[tuple[str, str]]) -> ContractionTree:
        if len(order) == 0:
            raise ValueError('Can not be empty')
        contracted = []  # [(nested_tup, flat_list)]
        for t1, t2 in order:
            if t1 == t2:
                # partial trace
                continue
            t1_matches = [n for n, (_, lst) in enumerate(contracted) if t1 in lst]
            t2_matches = [n for n, (_, lst) in enumerate(contracted) if t2 in lst]
            if len(t1_matches) > 1 or len(t2_matches) > 1:
                raise RuntimeError  # should not happen
            if len(t1_matches) == 0 and len(t2_matches) == 0:  # dont have either tensor yet
                contracted.append(((t1, t2), [t1, t2]))
            elif len(t1_matches) == 0:  # have t2 but not t1
                n2 = t2_matches[0]
                tup2, lst2 = contracted[n2]
                contracted[n2] = ((t1, tup2), [t1, *lst2])
            elif len(t2_matches) == 0:  # have t1 but not t2
                n1 = t1_matches[0]
                tup1, lst1 = contracted[n1]
                contracted[n1] = ((tup1, t2), [*lst1, t2])
            elif t1_matches == t2_matches:  # have already "contracted" them
                pass
            else:  # already have both, but not contracted yet
                n1 = t1_matches[0]
                n2 = t2_matches[0]
                tup1, lst1 = contracted[n1]
                tup2, lst2 = contracted[n2]
                contracted[n1] = ((tup1, tup2), [*lst1, *lst2])
                contracted.pop(n2)
        assert len(contracted) == 1
        tup, _ = contracted[0]
        return cls.from_nested_containers(tup)

    @classmethod
    def from_nested_containers(cls, tree: NestedContainer_str) -> ContractionTree:
        if not isinstance(tree, tuple | list):
            return cls.from_single_node(tree)
        assert len(tree) == 2
        left = cls.from_nested_containers(tree[0])
        right = cls.from_nested_containers(tree[1])
        return left.fuse(right, value=None)

    @classmethod
    def from_single_node(cls, node: str) -> ContractionTree:
        root = ContractionTreeNode(parent=None, left_child=None, right_child=None, value=node)
        return cls(root)

    def copy(self):
        return ContractionTree(self.root.copy())

    def fuse(self, other: ContractionTree, value=None):
        r"""Fuse two trees. In-place on both trees.

        Graphically::

            |                                        value
            |                                       /     \
            |       a             b                a        b
            |      / \     ,     / \      ->      / \      / \
            |    ... ...       ... ...          ... ...  ... ...

        """
        a = self.root
        b = other.root
        root = ContractionTreeNode(parent=None, left_child=a, right_child=b, value=value)
        a.parent = root
        b.parent = root
        return ContractionTree(root)

    def pop_contraction(self) -> tuple[None, str, str]:
        r"""Replace a bottom node (where both children are leaves) with a single leaf, in-place.

        Graphically::

            |    ...              ...
            |     |                |
            |     X       ->    new_value
            |    / \
            |   a   b

        Returns
        -------
        X : None
            The value at the non-leaf node that is replaced
        a, b : str
            The values of the leaf nodes that are removed
        new_value : str
            The value of the new leaf, conventionally ``'a @ b'``.

        """
        res = self.root.pop_contraction()
        self.root.test_sanity()  # OPTIMIZE rm
        return res

    def __str__(self):
        return '\n'.join(self.root._str_lines())


class PlanarLinearOperator(LinearOperator):
    r"""Base class for :class:`LinearOperator`\ s defined in terms of :class:`PlanarDiagram`\ s.

    .. warning ::
        Instantiating :class:`PlanarDiagram`\ s may be expensive (if the order is optimized).
        Make sure to either hard-code the order, or make the diagram instance as early as possible,
        e.g. as a *class* variable of the parent class instead of during its ``__init__``.

    Parameters
    ----------
    op_diagram : PlanarDiagram
        The diagram that defines the operator (without a vector).
    matvec_diagram : PlanarDiagram
        The diagram that defines the action of the operator on a vector.
        Must have the same tensor names as the `op_diagram` in addition to a single tensor
        with `vec_name`
    op_tensors : {str : Tensor}
        The concrete tensors that define the operator, see `op_diagram`.
    vec_name : str
        The name of the "vector", i.e. the tensor that the linear operator acts on in the
        `matvec_diagram`.

    """

    def __init__(
        self, op_diagram: PlanarDiagram, matvec_diagram: PlanarDiagram, op_tensors: dict[str, Tensor], vec_name: str
    ):
        self.op_diagram = op_diagram
        self.matvec_diagram = matvec_diagram
        self.op_tensors = op_tensors
        self.vec_name = vec_name
        if {*matvec_diagram.tensor_names} != {*op_diagram.tensor_names, vec_name}:
            msg = (
                f'Inconsistent tensor names. The matvec_diagram must have the tensor names from '
                f'the op_diagram, in addition to the single name {vec_name} of the vector.'
            )
            raise ValueError(msg)

    def matvec(self, vec):
        return self.matvec_diagram.evaluate(tensors={**self.op_tensors, self.vec_name: vec})

    def to_tensor(self, **kw):
        return self.op_diagram.evaluate(tensors=self.op_tensors)


@overload
def planar_contraction(
    tensor1: Tensor,
    tensor2: Tensor,
    legs1: int | str | list[int, str],
    legs2: int | str | list[int, str],
    relabel1: dict[str, str] = {},
    relabel2: dict[str, str] = {},
) -> Tensor: ...  # this stub exists for type hints only, definition is below


@overload
def planar_contraction(
    tensor1: TensorPlaceholder,
    tensor2: TensorPlaceholder,
    legs1: int | str | list[int, str],
    legs2: int | str | list[int, str],
) -> TensorPlaceholder: ...  # this stub exists for type hints only, definition is below


def planar_contraction(
    tensor1: Tensor,
    tensor2: Tensor,
    legs1: int | str | list[int, str],
    legs2: int | str | list[int, str],
    relabel1: dict[str, str] = {},
    relabel2: dict[str, str] = {},
) -> Tensor:
    """Planar version of :func:`~cyten.tensors.tdot`.

    Here, planar means that the contraction diagram can be drawn in a plane without any braids.

    We do not make assumptions about the leg arrangement of the result.
    It is constrained by the planar requirement, but otherwise arbitrary.
    I.e. it is the leg arrangement of the result of :func:`~cyten.tensors.tdot`, up to
    braid-free :func`~cyten.tensors.permute_legs`, i.e. up to arbitrary leg bending.
    """
    legs1 = tensor1.get_leg_idcs(legs1)
    legs2 = tensor2.get_leg_idcs(legs2)
    num_contr = len(legs1)
    if len(legs2) != num_contr:
        raise ValueError('legs1 and legs2 must have the same length')

    # check if the contraction actually is planar
    # 1) check if the legs on each tensor are divided into two contiguous subsets
    contr1, open1 = parse_leg_bipartition(legs1, tensor1.num_legs)
    _, open2 = parse_leg_bipartition(legs2, tensor2.num_legs)
    # 2) check that the contracted legs connect without braids:
    #    as contr1 goes around tensor1 counter-clockwise, their connection targets must go around
    #    tensor2 clockwise
    contr2 = [legs2[legs1.index(c1)] for c1 in contr1]
    for n1, n2 in zip(contr2[:-1], contr2[1:]):
        if n2 != (n1 - 1) % tensor2.num_legs:
            raise ValueError('Not a planar contraction')

    if isinstance(tensor1, TensorPlaceholder) or isinstance(tensor2, TensorPlaceholder):
        if len(relabel1) > 0 or len(relabel2) > 0:
            raise NotImplementedError
        assert isinstance(tensor1, TensorPlaceholder) and isinstance(tensor2, TensorPlaceholder)
        labels = [tensor1.labels[n] for n in open1] + [tensor2.labels[n] for n in open2]
        dims = [tensor1.dims[n] for n in open1] + [tensor2.dims[n] for n in open2]
        contr_dims = BigOPolynomial.prod(*(tensor1.dims[n] for n in contr1))
        assert contr_dims == BigOPolynomial.prod(*(tensor2.dims[n] for n in contr2))
        cost = tensor1.cost_to_make + tensor2.cost_to_make + BigOPolynomial.prod(*dims, contr_dims)
        return TensorPlaceholder(labels, dims, cost_to_make=cost)

    # choose if we do ``compose(tensor1, tensor2)`` or ``compose(tensor2, tensor1)``:
    if all(l < tensor1.num_codomain_legs for l in contr1):
        # all contr1 are already in tensor1 codomain -> compose(tensor2, tensor1) is best
        flip_order = True
    elif all(l >= tensor1.num_codomain_legs for l in contr1):
        # all contr1 are already in tensor1 domain -> compose(tensor1, tensor2) is best
        flip_order = False
    elif all(l < tensor2.num_codomain_legs for l in contr2):
        # all contr2 are already in tensor2 codomain -> compose(tensor1, tensor2) is best
        flip_order = False
    elif all(l >= tensor2.num_codomain_legs for l in contr2):
        # all contr2 are already in tensor2 domain -> compose(tensor2, tensor1) is best
        flip_order = True
    else:
        # remaining cases: choose arbitrarily
        flip_order = False

    if flip_order:
        # planar_permute_legs takes the ordered legs; order of contr2 is reversed
        tensor1 = planar_permute_legs(tensor1, codomain=contr1, domain=open1[::-1])
        tensor2 = planar_permute_legs(tensor2, codomain=open2, domain=contr2)
        return compose(tensor2, tensor1, relabel2, relabel1)
    else:
        tensor1 = planar_permute_legs(tensor1, codomain=open1, domain=contr1[::-1])
        tensor2 = planar_permute_legs(tensor2, codomain=contr2[::-1], domain=open2[::-1])
        return compose(tensor1, tensor2, relabel1, relabel2)


@overload
def planar_partial_trace(
    tensor: Tensor, *pairs: Sequence[int, str]
) -> Tensor: ...  # this stub exists for type hints only, definition is below


@overload
def planar_partial_trace(
    tensor: TensorPlaceholder, *pairs: Sequence[int, str]
) -> TensorPlaceholder: ...  # this stub exists for type hints only, definition is below


def planar_partial_trace(tensor: Tensor, *pairs: Sequence[int, str]) -> Tensor:
    """Planar version of :func:`~cyten.tensors.partial_trace`.

    Here, planar means that the trace can be drawn as a diagram in a plane, without any braids.
    """
    # make sure it is actually planar
    pairs = [tensor.get_leg_idcs(p) for p in pairs]
    traced_legs = [l for p in pairs for l in p]
    for l1, l2 in pairs:
        # sort s.t. l1 < l2
        if l1 > l2:
            l1, l2 = l2, l1
        assert l1 != l2
        # living on a circle, there are two different regions "between" l1 and l2.
        # at least one of them may contain only traced legs
        first_half_only_traces = True
        second_half_only_traces = True
        for l in range(l1 + 1, l2):  # first half
            if l in traced_legs:
                # must connect to another leg *in the same half*, otherwise there are braids
                other_ls = [a for a, b in pairs if b == l] + [b for a, b in pairs if a == l]
                assert len(other_ls) == 1
                if not (l1 < other_ls[0] < l2):
                    raise ValueError('Not a planar trace')
            else:
                first_half_only_traces = False
        for l in [*range(l2 + 1, tensor.num_legs), *range(l1)]:  # second half
            if l in traced_legs:
                # must connect to another leg *in the same half*, otherwise there are braids
                other_ls = [a for a, b in pairs if b == l] + [b for a, b in pairs if a == l]
                assert len(other_ls) == 1
                if l1 < other_ls[0] < l2:
                    raise ValueError('Not a planar trace')
            else:
                second_half_only_traces = False
        if not (first_half_only_traces or second_half_only_traces):
            raise ValueError('Not a planar trace')

    if isinstance(tensor, TensorPlaceholder):
        contr_dims = [tensor.dims[l] for l, _ in pairs]
        assert contr_dims == [tensor.dims[l] for _, l in pairs]
        open_dims = [d for l, d in enumerate(tensor.dims) if l not in traced_legs]
        cost = tensor.cost_to_make + BigOPolynomial.prod(*open_dims, *contr_dims)
        labels = [lab for l, lab in enumerate(tensor.labels) if l not in traced_legs]
        return TensorPlaceholder(labels=labels, dims=open_dims, cost_to_make=cost)

    levels = [None] * tensor.num_legs

    # OPTIMIZE
    # fusion tree backend requires legs that are traced over to be next to each other
    if isinstance(tensor.backend, FusionTreeBackend):
        # check how many legs need to be bent up or down and choose the way with the lower number
        num_up_bends = 0
        num_down_bends = 0
        for pair in pairs:
            pair = sorted(pair)
            legs_between = list(range(pair[0] + 1, pair[1]))
            # all legs between are traced over -> trace over the right side allowed
            if all(leg in traced_legs for leg in legs_between):
                continue
            # remaining case: must trace over the left side
            num_up_bends = max(num_up_bends, pair[0] + 1)
            num_down_bends = max(num_down_bends, tensor.num_legs - pair[1])

        if num_down_bends == 0:
            assert num_up_bends == 0
        elif num_up_bends > num_down_bends:
            # bend legs down
            # number of legs to be bent twice
            if tensor.num_domain_legs > num_down_bends:
                num_legs_in_codom = 0
            else:
                num_legs_in_codom = num_down_bends - tensor.num_domain_legs
            codomain = list(range(tensor.num_legs - num_down_bends, tensor.num_legs)) + list(
                range(tensor.num_codomain_legs - num_legs_in_codom)
            )
            domain = list(reversed(range(tensor.num_codomain_legs, tensor.num_legs - num_down_bends)))
            tensor = planar_permute_legs(tensor, codomain=codomain, domain=domain)
            # update pairs
            pairs = [[(idx + num_down_bends) % tensor.num_legs for idx in pair] for pair in pairs]
        else:
            # bend legs up
            if tensor.num_codomain_legs > num_up_bends:
                num_legs_in_dom = 0
            else:
                num_legs_in_dom = num_up_bends - tensor.num_codomain_legs
            codomain = list(range(num_up_bends, tensor.num_codomain_legs))
            domain = list(reversed(range(num_up_bends))) + list(
                reversed(range(tensor.num_codomain_legs + num_legs_in_dom, tensor.num_legs))
            )
            tensor = planar_permute_legs(tensor, codomain=codomain, domain=domain)
            # update pairs
            pairs = [[(idx - num_up_bends) % tensor.num_legs for idx in pair] for pair in pairs]

        # give the traced legs levels in case they are not next to each other
        # and one leg needs to braid with another leg pair
        for i, pair in enumerate(pairs):
            levels[pair[0]] = i
            levels[pair[1]] = i

    return partial_trace(tensor, *pairs, levels=levels)


def planar_permute_legs(T: Tensor, *, codomain: list[int | str] = None, domain: list[int | str] = None):
    """Planar special case of :func:`~cyten.permute_legs`, without braids.

    It permutes the :attr:`Tensor.legs` only cyclically, and bends them to the proper codomain / domain

    A planar permutation consists only of leg bends, either to the left or right of the tensor.
    It leaves the :attr:`cyten.Tensor.legs` unchanged up to cyclical permutation.
    It is fully specified by assigning each leg to either the new codomain or the new domain.

    Parameters
    ----------
    codomain, domain : list of {str | int}
        The legs that should be in the new (co)domain, in the correct order.
        Only one of `codomain`, `domain` is required when the other can be unambiguously inferred.
        This is the case when the specified `codomain` or `domain` contains at least one leg.

    """
    # Note: parse_leg_bipartition cannot easily be used in this function due to how it interacts with empty (co)domains

    if codomain is None and domain is None:
        raise ValueError('Need to specify either codomain or domain that is non-empty')
    elif (codomain is None and len(domain) == 0) or (domain is None and len(codomain) == 0):
        raise ValueError('Specified codomain or domain is empty')

    # do this for both before potentially comparing (avoid comparing to labels)
    if domain is not None:
        domain = T.get_leg_idcs(domain)
    if codomain is not None:
        codomain = T.get_leg_idcs(codomain)

    if domain is not None and len(domain) > 0:
        expect = [(domain[-1] + i) % T.num_legs for i in range(len(domain))][::-1]
        if domain != expect:
            raise ValueError('The given domain is a non-planar permutation')
        num_codom_legs = T.num_legs - len(domain)
        codomain2 = [i % T.num_legs for i in range(domain[0] + 1, domain[0] + 1 + num_codom_legs)]
        if codomain is None:
            codomain = codomain2
        else:
            if codomain != codomain2:
                raise ValueError('The given codomain and domain are inconsistent!')
    if codomain is not None and len(codomain) > 0:
        expect = [(codomain[0] + i) % T.num_legs for i in range(len(codomain))]
        if codomain != expect:
            raise ValueError('The given codomain is a non-planar permutation')
        num_dom_legs = T.num_legs - len(codomain)
        reverse_domain = [i % T.num_legs for i in range(codomain[-1] + 1, codomain[-1] + 1 + num_dom_legs)]
        domain2 = reverse_domain[::-1]
        if domain is None:
            domain = domain2
        else:
            if domain != domain2:
                raise ValueError('The given codomain and domain are inconsistent!')

    # figure out if legs need to bend right or left of the tensor.
    codomain_staying = [n for n in range(T.num_codomain_legs) if n in codomain]
    domain_staying = [n for n in range(T.num_domain_legs) if T.num_legs - 1 - n in domain]

    # requires two bends of at least one leg
    if len(codomain_staying) > 0 and 0 in codomain and T.num_codomain_legs - 1 in codomain:
        codomain_winding = codomain.index(T.num_codomain_legs - 1) < codomain.index(0)
    else:
        codomain_winding = False
    if len(domain_staying) > 0 and T.num_codomain_legs in domain and T.num_legs - 1 in domain:
        domain_winding = domain.index(T.num_codomain_legs) < domain.index(T.num_legs - 1)
    else:
        domain_winding = False
    # one at most can be True
    assert (not codomain_winding) or (not domain_winding)

    if len(codomain_staying) == 0 and len(domain_staying) == 0 and len(codomain) > 0 and len(domain) > 0:
        # they swap places completely -> choose the direction such that we have less left bends than right bends
        if T.num_codomain_legs < T.num_domain_legs:
            bend_right = [False] * T.num_codomain_legs + [True] * T.num_domain_legs
        else:
            bend_right = [True] * T.num_codomain_legs + [False] * T.num_domain_legs

    elif codomain_winding:
        # special case where the group of legs that stay in the codomain "wraps around",
        # i.e. surrounds the ones that should go to the domain on both sides
        # three groups: stay, bend up, bend twice ("around")
        # this is an arbitrary choice of orientation "counter-clockwise"
        bend_up = T.num_codomain_legs - len(codomain_staying)
        dont_bend = codomain[-1] + 1
        bend_twice = codomain.index(T.num_codomain_legs - 1) + 1
        assert dont_bend + bend_up + bend_twice == T.num_codomain_legs
        # OPTIMIZE achieve it in a single backend function? also in a similar branch below
        # OPTIMIZE we go around counter-clockwise, clockwise could be more efficient in some cases
        res = permute_legs(T, codomain=range(dont_bend), domain=reversed(range(dont_bend, T.num_legs)), bend_right=True)
        res = permute_legs(
            res,
            codomain=[*range(dont_bend + bend_up, T.num_legs), *range(dont_bend)],
            domain=reversed(range(dont_bend, T.num_codomain_legs - bend_twice)),
            bend_right=False,
        )
        return res

    elif domain_winding:
        # special case where the group of legs that stay in the domain "wraps around",
        # i.e. surrounds the ones that should go to the codomain on both sides
        # three groups (in leg order): stay, bend down, bend twice ("around")
        # this is an arbitrary choice of orientation "counter-clockwise"
        bend_down = T.num_domain_legs - len(domain_staying)
        dont_bend = domain[0] + 1 - T.num_codomain_legs
        bend_twice = len(domain) - domain.index(T.num_legs - 1)
        assert bend_twice + bend_down + dont_bend == T.num_domain_legs
        res = permute_legs(
            T, codomain=range(dont_bend, T.num_legs), domain=reversed(range(dont_bend)), bend_right=False
        )
        res = permute_legs(
            res,
            codomain=range(bend_down),
            domain=reversed([*range(bend_down, bend_down + bend_twice), *range(bend_down + bend_twice, T.num_legs)]),
            bend_right=True,
        )
        return res

    elif len(codomain_staying) == T.num_codomain_legs and len(domain_staying) == T.num_domain_legs:
        # nothing to do
        # the number of entries in codomain_staying is only sufficient to detect this case after
        # considering the winding cases (codomain or domain could be empty)
        assert len(codomain_staying) == T.num_codomain_legs
        assert len(domain_staying) == T.num_domain_legs
        return T

    elif T.num_codomain_legs == 0:
        # split into three groups: bending down right, staying, bending down left
        # note that one of the outer groups (but not both) may be empty
        if T.num_legs - 1 in codomain:
            left_bending = codomain.index(T.num_legs - 1) + 1
        else:
            left_bending = 0
        dont_bend = len(domain_staying)
        if 0 in codomain:
            right_bending = len(codomain) - codomain.index(0)
        else:
            right_bending = 0
        assert left_bending + dont_bend + right_bending == T.num_legs
        bend_right = [True] * right_bending + [None] * dont_bend + [False] * left_bending

    elif T.num_domain_legs == 0:
        # split into three groups: bending up left, staying, bending up right
        # note that one of the outer groups (but not both) may be empty
        if 0 in domain:
            left_bending = domain.index(0) + 1
        else:
            left_bending = 0
        dont_bend = len(codomain_staying)
        if T.num_legs - 1 in domain:
            right_bending = len(domain) - domain.index(T.num_legs - 1)
        else:
            right_bending = 0
        assert left_bending + dont_bend + right_bending == T.num_legs
        bend_right = [False] * left_bending + [None] * dont_bend + [True] * right_bending

    elif len(codomain_staying) == 0:
        # codomain goes up as a whole, either right or left
        domain_bend_left = domain_staying[0]
        domain_bend_right = T.num_domain_legs - 1 - domain_staying[-1]
        assert domain_bend_left + len(domain_staying) + domain_bend_right == T.num_domain_legs
        if len(domain_staying) == T.num_domain_legs:
            # domain stays, codomain is divided and bent right and left
            num_bend_left = domain.index(T.num_legs - 1)
            bend_right = (
                [False] * num_bend_left + [True] * (T.num_codomain_legs - num_bend_left) + [None] * T.num_domain_legs
            )
        elif domain_bend_left == 0:
            # bend the codomain up to the left
            bend_right = [False] * T.num_codomain_legs + [True] * domain_bend_right + [None] * len(domain_staying)
        elif domain_bend_right == 0:
            # bend the codomain up to the right
            bend_right = [True] * T.num_codomain_legs + [None] * len(domain_staying) + [False] * domain_bend_left
        else:
            raise RuntimeError('Not planar, but that should have been detected earlier?')

    elif len(domain_staying) == 0:
        # domain goes down as a whole, either right or left
        codomain_bend_left = codomain_staying[0]
        codomain_bend_right = T.num_codomain_legs - 1 - codomain_staying[-1]
        assert codomain_bend_left + len(codomain_staying) + codomain_bend_right == T.num_codomain_legs
        if len(codomain_staying) == T.num_codomain_legs:
            num_bend_left = codomain.index(0)
            bend_right = (
                [None] * T.num_codomain_legs + [True] * (T.num_domain_legs - num_bend_left) + [False] * num_bend_left
            )
        elif codomain_bend_left == 0:
            # bend the domain down to the left
            bend_right = [None] * len(codomain_staying) + [True] * codomain_bend_right + [False] * T.num_domain_legs
        elif codomain_bend_right == 0:
            # bend the domain down to the right
            bend_right = [False] * codomain_bend_left + [None] * len(codomain_staying) + [True] * T.num_domain_legs
        else:
            raise RuntimeError('Not planar, but that should have been detected earlier?')

    else:
        codomain_bend_left = codomain_staying[0]
        codomain_bend_right = T.num_codomain_legs - 1 - codomain_staying[-1]
        domain_bend_left = domain_staying[0]
        domain_bend_right = T.num_domain_legs - 1 - domain_staying[-1]
        assert codomain_bend_left == 0 or domain_bend_left == 0
        assert codomain_bend_right == 0 or domain_bend_right == 0
        bend_right = (
            [False] * codomain_bend_left
            + [None] * len(codomain_staying)
            + [True] * codomain_bend_right
            + [True] * domain_bend_right
            + [None] * len(domain_staying)
            + [False] * domain_bend_left
        )

    return permute_legs(T, codomain=codomain, domain=domain, levels=None, bend_right=bend_right)


def parse_leg_bipartition(legs: Sequence[int], num_legs: int) -> tuple[list[int], list[int]]:
    """Parse a planar bipartition of legs into two subsets.

    We view the indices on a circle with length `num_legs`, i.e. ``0`` comes after ``num_legs - 1``.
    We verify that the ``legs`` form a single contiguous subset on that circle.
    Note that "on the circle" means that it may "wrap around", e.g. ``[7, 8, 0, 1, 2]`` is
    contiguous if ``num_legs=9``.

    Parameters
    ----------
    legs : list of int
        A subset of legs, in any order. Is explicitly checked to be contiguous on the circle.
    num_legs : int
        The total number of legs, such that we look at subsets of ``range(num_legs)``.

    Returns
    -------
    legs
        The `legs`, sorted in order around the circle.
        Note that this may include a jump, e.g. ``[7, 8, 0, 1, 2]`` is sorted if ``num_legs=9``.
    other_legs
        The complementary subset, sorted in order around the circle.
        Note that this may include a jump, e.g. ``[7, 8, 0, 1, 2]`` is sorted if ``num_legs=9``.

    """
    assert not duplicate_entries(legs)
    assert all(0 <= l < num_legs for l in legs)
    # special cases
    if len(legs) == 0:
        return [], [*range(num_legs)]
    if len(legs) == num_legs:
        return [*range(num_legs)], []

    sorted_legs = np.sort(legs)
    jumps = np.where(sorted_legs[1:] != sorted_legs[:-1] + 1)[0]
    if len(jumps) == 0:
        # legs is contiguous even on a line -> other subset wraps around the circle
        res_legs = list(sorted_legs)
        other_legs = [*range(sorted_legs[-1] + 1, num_legs), *range(sorted_legs[0])]
    elif len(jumps) == 1 and sorted_legs[0] == 0 and sorted_legs[-1] == num_legs - 1:
        # a single jump is ok, but only if the legs "wrap around", i.e. contain 0 and L-1
        # legs "wraps" around the circle -> other subset is contiguous even on the line
        last = sorted_legs[jumps[0]]
        first = sorted_legs[jumps[0] + 1]
        res_legs = [*range(first, num_legs), *range(last + 1)]
        other_legs = [*range(last + 1, first)]
    else:
        raise ValueError('Not a planar bipartition')

    return res_legs, other_legs
