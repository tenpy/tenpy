"""Defines a class describing the local physical Hilbert space.

The :class:`Site` is the prototype, read it's docstring.

.. _OperatorMiniLanguage:
TODO explain mini language


Main changes  (TODO -> changelog)
- distinction into symmetric and charged on-site operators
- Site.__init__ no longer supports directly adding operators as kwargs
  add_symmetric_operator or add_charged_operator or add_any_operator instead
- class ChargedOperator
- operators are no longer class attributes
- sites have a backend. all operators on the site need to have that backend.

"""
# Copyright (C) TeNPy Developers, GNU GPLv3


from __future__ import annotations
import numpy as np
import itertools
import copy
from typing import TypeVar, Type
from functools import partial, reduce

from ..linalg.tensors import (Tensor, SymmetricTensor, SymmetricTensor, ChargedTensor,
                              DiagonalTensor, almost_equal, angle, real_if_close, exp)
from ..linalg.backends import TensorBackend, Block
from ..linalg.symmetries import (ProductSymmetry, Symmetry, SU2Symmetry, U1Symmetry, ZNSymmetry,
                             no_symmetry, SectorArray)
from ..linalg.spaces import Space, ElementarySpace, ProductSpace
from ..tools.misc import find_subclass, make_stride
from ..tools.hdf5_io import Hdf5Exportable

__all__ = ['Site', 'GroupedSite', 'group_sites', 'set_common_symmetry', 'as_valid_operator_name',
           'split_charged_operator_symbol', 'ChargedOperator', 'SpinHalfSite', 'SpinSite',
           'FermionSite', 'SpinHalfFermionSite', 'SpinHalfHoleSite', 'BosonSite', 'ClockSite',
           'spin_half_species']


_T = TypeVar('_T')


# TODO can we improve how hc's are added?
#  - figure out what exactly hc means for the charged operators (-> use in test_sanity)
#  - implement auto-detect for charged operators
#  - manual hc is a bit ugly, op1 needs hc=False and only op2 needs hc='op1', i.e. asymmetrical


class Site(Hdf5Exportable):
    """Collects information about a single local site of a lattice.

    This class defines what the local basis states are via the :attr:`leg`, which defines the order
    of basis states (:attr:`ElementarySpace.basis_perm`) and how the symmetry acts on them
    (by assigning them to :attr:`Space.sectors`). It also provides :attr:`state_labels` for the
    basis. A `Site` instance therefore already determines which symmetry is explicitly used.
    Using the same "kind" of physical site (typically a particular subclass of `Site`),
    but using different symmetries requires *different* `Site` instances.
    
    Moreover, it stores the local operators. We distinguish two types of operators on a site:

        i.  The *symmetric* operators. These are operators that preserve the symmetry of the site,
            i.e. they can be written as a :class:`Tensor` (as opposed to the charged operators
            below, which can only be written as :class:`ChargedTensor`s).
            Applying them to a state preserves the conserved charges.

        ii. The *charged* operators. These are operators that do *not* preserve the symmetry, i.e.
            applying them to a state in a given charge sector yields a state in a *different* charge
            sector. They are commonly used as building blocks for non-local symmetry-preserving
            operators, e.g. terms like :math:`S_i^{+} S_j^{-}`. We store them
            as :class:`ChargedOperator`. Each of the charged operators comes in a left and right
            version, both of which are :class:`ChargedTensor`s with dummy legs of opposite duality.
            We only require that the left version of an operator contracted with the right version
            of its :attr:`hc_ops` partner yields the correct two-body operator (and vice versa),
            e.g. to form :math:`S_i^{+} S_j^{-}`.
            Note that not all charged operators make sense as stand-alone on-site operators,
            even after the symmetry is dropped, see :attr:`ChargedOperator.can_use_alone`.
            For example, consider a ``SpinHalfSite`` with ``conserve='Sz'``.
            The ``'Sx'`` is a charged operator with a two-dimensional dummy leg. Nevertheless,
            if we convert to dense block, thereby ignoring the symmetry, we get the expected
            operator ``[[0., 1.], [1., 0.]]``. On the other hand, ``'Svec'`` is a charged operator
            that can not be converted to a dense 2D array. It only makes sense when contracted with
            a second ``'Svec'`` over their dummy legs.

    Operators can be accessed via :meth:`get_op` or equivalently by "indexing" the site,
    ``site['O']`` is a shorthand for ``site.get_op('O')``.
    All sites define the operators ``'Id'``, the identity, and ``'JW'``, the local contribution to
    Jordan-Wigner strings, both of which are always symmetric and diagonal.

    Parameters
    ----------
    leg : :class:`~tenpy.linalg.spaces.Space`
        The Hilbert space associated with the site. Defines the basis and the symmetry.
    backend : :class:`~tenpy.linalg.backends.Backend`, optional
        The backend used to create the identity operator and possibly convert non-tensor operators
        to :class:`~tenpy.linalg.tensors.Tensor`s.
    state_labels : None | list of str
        Optionally, a label for each local basis state.
    JW : :class:`DiagonalTensor` | Block
        The local contribution to the Jordan-Wigner string. Either as tensor or as a 1D block,
        which are the diagonal entries of the operator.

    Attributes
    ----------
    leg : :class:`~tenpy.linalg.spaces.Space`
        The Hilbert space associated with the site. Defines the basis and the symmetry.
    state_labels : {str: int}
        Labels for the local basis states. Maps from label to index of the state in the basis.
    need_JW_string : set of str
        Labels of those operators (symmetric_ops *or* charged_ops) that need a Jordan-Wigner string.
        Used in :meth:`op_needs_JW` to determine whether an operator anticommutes or commutes
        with operators on other sites.
    symmetric_ops : {str: :class:`SymmetricTensor`}
        A dictionary of symmetric on-site operators. All onsite operators have labels ``'p', 'p*'``.
    charged_ops : {str: :class:`ChargedOperator`}
        A dictionary of charged on-site operators. A :class:`ChargedOperator` has a left and a
        right version, both of which are :class:`ChargedTensor`s.
    JW_exponent : :class:`DiagonalTensor`
        Exponents of the ``'JW'`` operator such that ``symmetric_ops['JW']`` is equivalent to
        ``exp(1.j * pi * JW_exponent)``.
    hc_ops : {str: str}
        Mapping from operator names (keys of `symmetric_ops` or `charged_ops_*`) to the names of
        their hermitian conjugates. Use :meth:`get_hc_op_name` to obtain entries.
    """
    
    def __init__(self, leg: Space, backend: TensorBackend = None,
                 state_labels: list[str] = None, JW: DiagonalTensor | Block = None):
        self.leg = leg
        self.state_labels = {}
        if state_labels is not None:
            for i, l in enumerate(state_labels):
                if l is not None:
                    self.state_labels[str(l)] = i
        self.symmetric_ops: dict[str, SymmetricTensor] = {}
        self.charged_ops: dict[str, ChargedOperator] = {}
        self.need_JW_string: set[str] = {'JW'}
        self.hc_ops = {}
        self.JW_exponent = None  # set by self.add_symmetric_operator('JW')
        Id = DiagonalTensor.eye(leg, backend=backend, labels=['p', 'p*'])
        self.backend = Id.backend
        self.add_symmetric_operator('Id', Id, hc='Id')
        if JW is None:
            JW = Id
        self.add_symmetric_operator('JW', JW, need_JW=True)
        # TODO ideally we should test sanity after concluding the subclasses __init__ ...
        self.test_sanity()

    def test_sanity(self):
        # check self.state_labels
        for lab, ind in self.state_labels.items():
            if not isinstance(lab, str):
                raise ValueError("wrong type of state label")
            if not 0 <= ind < self.dim:
                raise ValueError(f'index {ind} of state label "{lab}" out of bounds')
        # check self.symmetric_ops
        for name, op in self.symmetric_ops.items():
            self.check_valid_operator(op)
            # if charged operator with same name exists, they should be equivalent
            other_op = self.charged_ops.get(name, None)
            if other_op is not None:
                are_equivalent = other_op.can_use_alone  # now it suffices to check op_L
                try:
                    op_L = other_op.op_L.as_Tensor()
                except ValueError:
                    are_equivalent = False
                else:
                    are_equivalent = almost_equal(op_L, op)
                if not are_equivalent:
                    msg = ('Charged operators and symmetric operators with the same name must '
                            'be equal.')
                    raise ValueError(msg)
        # check self.charged_ops
        for name, op in self.charged_ops.items():
            self.check_valid_operator(op.op_L, test_sanity=False)
            self.check_valid_operator(op.op_R, test_sanity=False)
            op.test_sanity()
        # check self.need_JW_string
        for name in self.need_JW_string:
            assert name in self.all_op_names, name
        assert 'JW' in self.need_JW_string
        assert 'Id' not in self.need_JW_string
        # check Id, JW, JW_exponent operators
        assert almost_equal(self.Id, SymmetricTensor.from_eye([self.leg], backend=self.backend))
        assert almost_equal(exp(1.j * np.pi * self.JW_exponent), self.JW)
        # check self.hc_ops
        for name1, name2 in self.hc_ops.items():
            if name1 in self.symmetric_ops:
                assert name2 in self.symmetric_ops
                assert almost_equal(self.symmetric_ops[name1], self.symmetric_ops[name2].hconj())
            elif name1 in self.charged_ops:
                assert name2 in self.charged_ops
                # TODO what exactly does hc even mean for charged ops?
            else:
                raise ValueError('unknown name')

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.get_op(key)
        raise TypeError

    def check_valid_operator(self, op: Tensor, test_sanity: bool = True):
        """Check if `op` is a valid operator for this site. Raise if not."""
        if op.num_legs != 2:
            raise ValueError('wrong number of legs.')
        if not op.labels_are('p', 'p*'):
            raise ValueError('wrong labels')
        if op.get_legs(['p', 'p*']) != [self.leg, self.leg.dual]:
            raise ValueError('incompatible legs')
        if not op.backend == self.backend:
            raise ValueError('incompatible backend')
        if test_sanity:
            op.test_sanity()
    
    @property
    def symmetric_op_names(self) -> set[str]:
        """The names of all symmetric operators, e.g. for iteration"""
        return set(self.symmetric_ops.keys())
    
    @property
    def charged_op_names(self) -> set[str]:
        """The names of all charged operators, e.g. for iteration"""
        return set(self.charged_ops.keys())
            
    @property
    def all_op_names(self) -> set[str]:
        """The names of all operators, symmetric *and* charged, e.g. for iteration"""
        return self.symmetric_op_names.union(self.charged_op_names)

    @property
    def dim(self) -> int:
        """Dimension of the local Hilbert space."""
        return self.leg.dim

    @property
    def symmetry(self) -> Symmetry:
        """Symmetry of the local Hilbert space."""
        return self.leg.symmetry

    @property
    def Id(self) -> DiagonalTensor:
        """The identity operator"""
        return self.symmetric_ops['Id']

    @property
    def JW(self) -> DiagonalTensor:
        """The JW operator, the local contribution to Jordan-Wigner strings"""
        return self.symmetric_ops['JW']

    def as_operator(self, op: Block | Tensor, cls: Type[_T] | str = Tensor) -> _T:
        """Convert object to tensor that is a valid operator for this site.

        Parameters
        ----------
        op : tensor-like
            object to be converted
        cls : :class:`Tensor`-type | str
            The tensor class (type, not instance) to convert to, or its name.
            If one of the superclasses ``Tensor, SymmetricTensor``, we prioritize returning
            a ``DiagonalTensor``, if the data is diagonal.

        Returns
        -------
        op : :class:`Tensor`
            Converted tensor. Is an instance of `cls`.
        """
        cls = find_subclass(Tensor, cls)
        if not isinstance(op, Tensor):
            op = self.backend.as_block(op)
            if len(self.backend.block_shape(op)) == 1:
                op = DiagonalTensor.from_diag(op, self.leg, backend=self.backend, labels=['p', 'p*'])
            else:
                raise NotImplementedError  # TODO
                # op = tensor_from_block(op, legs=[self.leg, self.leg.dual], backend=self.backend,
                #                        labels=['p', 'p*'])
        if cls is Tensor:
            if isinstance(op, SymmetricTensor):
                try:
                    op = DiagonalTensor.from_tensor(op, check_offdiagonal=True)
                except ValueError:
                    pass
        elif issubclass(cls, SymmetricTensor):
            if isinstance(op, ChargedTensor):
                op = op.as_Tensor()
            assert isinstance(op, SymmetricTensor)
            if (cls is not SymmetricTensor) and isinstance(op, SymmetricTensor):
                try:
                    op = DiagonalTensor.from_tensor(op, check_offdiagonal=True)
                except ValueError:
                    # if cls is SymmetricTensor, we try converting to diagonal and ignore failure
                    if cls is DiagonalTensor:
                        raise
            if (cls is SymmetricTensor) and isinstance(op, DiagonalTensor):
                op = op.as_Tensor()
        elif cls is ChargedTensor:
            if isinstance(op, SymmetricTensor):
                op = ChargedTensor.from_tensor(op.as_Tensor())
            assert isinstance(op, ChargedTensor)
        else:
            raise ValueError(f'Unsupported tensor class "{cls}"')
        self.check_valid_operator(op)
        return op

    def add_symmetric_operator(self, name: str, op: SymmetricTensor | Block, need_JW: bool = False,
                               hc: str | bool = None, also_as_charged: bool = False
                               ) -> SymmetricTensor:
        """Add a symmetric on-site operator.

        Parameters
        ----------
        name : str
            The name of the operator. Is used as the key for :attr:`symmetric_ops`.
            Should be a valid name for the :ref:`operator mini language <OperatorMiniLanguage>`.
            Names of symmetric operators must be unique per class instance.
            A symmetric and a charged operator may share the same name *if* they are
            :func:`almost_equal`.
        op : :class:`SymmetricTensor` | Block
            A single tensor describing the operator. Legs must be the :attr:`leg` of this site with
            label ``'p'`` and its dual with label ``'p*'``.
            If not a tensor, it is converted via :func:`tensor_from_block`, using :attr:`backend`.
        need_JW : bool
            If this operator needs a Jordan-Wigner string.
            If so, `name` is added to :attr:`need_JW_string`.
        hc : None | False | str
            The name for the hermitian conjugate operator, to be used for :attr:`hc_ops`.
            By default (``None``), try to auto-determine it.
            If ``False``, disable adding entries to :attr:`hc_ops`.
        also_as_charged : bool
            Optionally, add the same operator is also as a charged operator (with "trivial" charge).

        Returns
        -------
        op : SymmetricTensor
            The operator as a tensor, i.e. the value that was added to :attr:`symmetric_ops`.

        See Also
        --------
        add_charged_operator
        """
        name = as_valid_operator_name(name)
        op = self.as_operator(op, cls=SymmetricTensor)
        if name in self.symmetric_ops:
            raise ValueError(f'symmetric operator with name "{name}" already exists')
        if name in self.charged_ops:
            other_op = self.charged_ops[name]
            assert other_op.can_use_alone  # now sufficient to check op_L
            assert other_op.op_L.almost_equal(op, allow_different_types=True)
            also_as_charged = False
        self.symmetric_ops[name] = op
        if hc is None:
            hc = self._auto_detect_hc(name, op)
        if also_as_charged:
            # note: order matters. should auto detect hc before using it here
            self.add_charged_operator(name, ChargedTensor.from_tensor(op), need_JW=need_JW, hc=hc)
        if need_JW:
            self.need_JW_string.add(name)
        if hc:
            self.hc_ops[hc] = name
            self.hc_ops[name] = hc
        if name == 'JW':
            assert isinstance(op, DiagonalTensor)
            self.JW_exponent = angle(real_if_close(op)) / np.pi
        return op

    def add_charged_operator(self, name: str, op: ChargedOperator | ChargedTensor | Block,
                             op_R: ChargedTensor | Block = None, need_JW: bool = False,
                             hc: str | bool = None) -> ChargedOperator:
        """Add a charged on-site operator.

        Parameters
        ----------
        name : str
            The name of the operator. Is used as the key for :attr:`symmetric_ops`.
            Should be a valid name for the :ref:`operator mini language <OperatorMiniLanguage>`.
            Names of charged operators must be unique per class instance.
            A symmetric and a charged operator may share the same name *if* they are
            :func:`almost_equal`.
        op : :class:`ChargedOperator` | :class:`ChargedTensor` | Block
            Together with `op_R`, specifies the :class:`ChargedOperator` to be added.
            Either a :class:`ChargedOperator` already, then `op_R` is is ignored.
            Or a :class:`ChargedTensor` or block describing the left version of the operator.
        op_R : :class:`ChargedTensor` | Block
            If `op` is the left version, the right version. Per default (and only if `op`
            if not a :class:`ChargedOperator`), we use `op_R=op.flip_dummy_leg_duality()`.
        need_JW : bool
            If this operator needs a Jordan-Wigner string.
            If so, `name` is added to :attr:`need_JW_string`.
        hc : None | False | str
            The name for the hermitian conjugate operator, to be used for :attr:`hc_ops`.
            By default (``None``), try to auto-determine it.
            If ``False``, disable adding entries to :attr:`hc_ops`.

        Returns
        -------
        op : :class:`ChargedOperator`
            The operator that was added.

        See Also
        --------
        add_symmetric_operator
        """
        name = as_valid_operator_name(name)
        if isinstance(op, ChargedOperator):
            assert op_R is None  # TODO warn instead?
        else:
            op_L = self.as_operator(op, ChargedTensor)
            if op_R is not None:
                op_R = self.as_operator(op_R, ChargedTensor)
            op = ChargedOperator(op_L, op_R)
        self.check_valid_operator(op.op_L)
        self.check_valid_operator(op.op_R)
        if name in self.charged_op_names:
            raise ValueError(f'charged operator with name "{name}" already exists.')
        if name in self.symmetric_ops:
            assert op.can_use_alone
            assert op.op_L.almost_equal(self.symmetric_ops[name], allow_different_types=True)
        self.charged_ops[name] = op
        if need_JW:
            self.need_JW_string.add(name)
        if hc is None:
            hc = self._auto_detect_hc(name, op_L)
            hc_R = self._auto_detect_hc(name, op_R)
            if hc != hc_R:
                raise RuntimeError
        if hc:
            self.hc_ops[hc] = name
            self.hc_ops[name] = hc
        return op

    def add_any_operator(self, name: str,
                         op: SymmetricTensor | ChargedOperator | ChargedTensor | Block,
                         need_JW: bool = False, hc: str | bool = None
                         ) -> SymmetricTensor | ChargedTensor:
        """Convenience wrapper for adding operators.

        The `op` may be either symmetric or charged.
        In the first case we add a symmetric operator, and in both cases we add a charged version.
        """
        if isinstance(op, ChargedOperator):
            op = self.add_charged_operator(name, op, need_JW=need_JW, hc=hc)
            return op.op_L
        op = self.as_operator(op, cls=Tensor)
        if isinstance(op, SymmetricTensor):
            return self.add_symmetric_operator(name, op, need_JW=need_JW, hc=hc,
                                               also_as_charged=True)
        else:
            op = self.add_charged_operator(name, op, need_JW=need_JW, hc=hc)
            return op.op_L

    def _auto_detect_hc(self, name: str, op: Tensor) -> str | None:
        """Automatically detect which (if any) of the existing operators is the hc of a new op

        Parameters
        ----------
        name, op
            Name and tensor of the "new" operator (not yet added).
        
        Returns
        -------
        If the hermitian conjugate was found, its name. Otherwise ``None``.
        """
        if name in self.hc_ops:
            return self.hc_ops[name]
        op_hc = op.hconj()
        if almost_equal(op_hc, op):
            return name
        if isinstance(op, SymmetricTensor):
            candidates = self.symmetric_ops
        if isinstance(op, (ChargedOperator, ChargedTensor)):
            raise NotImplementedError  # TODO what exactly does hc for charged tensors even mean?
        for other_name, other_op in candidates.items():
            # allow different types, since we might have `Tensor`s and `DiagonalTensor`s
            if almost_equal(op_hc, other_op, allow_different_types=True):
                return other_name
        return None

    def change_leg(self, new_leg: Space = None):
        """Change the :attr:`leg` of the site in-place.

        Assumes that the :attr:`state_labels` are still valid.

        Parameters
        ----------
        new_leg : :class:`Space` | None
            The new leg to be used. If ``None``, use trivial charges.
        """
        if new_leg is None:
            new_leg = ElementarySpace.from_trivial_sector(dim=self.dim, symmetry=self.symmetry)
        self.leg = new_leg
        old_symmetric_ops = self.symmetric_ops
        self.symmetric_ops = {}
        old_charged_ops = self.charged_ops
        self.charged_ops = {}
        for name, op in old_symmetric_ops.items():
            self.add_symmetric_operator(name, op.to_dense_block(['p', 'p*']), need_JW=False, hc=False)
        for name, op in old_charged_ops.items():
            # TODO what to do if charged tensor
            op = ChargedOperator(
                self.as_operator(op.op_L.to_dense_block(['p', 'p*']), cls=ChargedTensor),
                self.as_operator(op.op_R.to_dense_block(['p', 'p*']), cls=ChargedTensor),
                can_use_alone=op.can_use_alone
            )
            self.add_charged_operator(name, op, need_JW=False, hc=False)
        return self
        
    def rename_op(self, old_name: str, new_name: str):
        """Rename an added operator.

        Parameters
        ----------
        old_name : str
            The old name of the operator.
        new_name : str
            The new name of the operator.
        """
        if old_name == new_name:
            return
        if new_name in self.all_ops_names:
            raise ValueError('new_name already exists')
        hc_name = self.hc_ops.get(old_name, False)
        if hc_name == old_name:
            hc_name = new_name
        need_JW = old_name in self.need_JW_string
        symm = self.symmetric_ops.get(old_name, None)
        charged = self.charged_ops.get(old_name, None)
        self.remove_op(old_name)
        if symm is not None:
            self.add_symmetric_operator(new_name, symm, need_JW=need_JW, hc=hc_name)
        if charged is not None:
            self.add_charged_operator(new_name, charged, need_JW=need_JW, hc=hc_name)

    def remove_op(self, name: str):
        """Remove an added operator.

        Parameters
        ----------
        name : str
            The name of the operator to be removed.
        """
        hc_name = self.hc_ops.pop(name, None)
        self.hc_ops.pop(hc_name, None)
        self.symmetric_ops.pop(name)
        self.charged_ops.pop(name)
        self.need_JW_string.discard(name)

    def state_index(self, label: str | int) -> int:
        """Return index of a basis state from its label.

        Parameters
        ----------
        label : int | string
            either the index directly or a label (string) set before.

        Returns
        -------
        state_index : int
            the index of the basis state associated with the label.
        """
        if isinstance(label, int):
            return label
        try:
            return self.state_labels[label]
        except KeyError:
            raise KeyError(f'label not found: {label}') from None

    def state_indices(self, labels: list[str | int]) -> list[int]:
        """Same as :meth:`state_index`, but for multiple labels."""
        return [self.state_index(lbl) for lbl in labels]

    def get_op(self, name: str, use_left_version: bool = None) -> Tensor:
        """Obtain an on-site operator.

        TODO expand docstring
        TODO operator mini language?

        Supports left and right versions: ``site.get_op('(O*_)')`` is short-hand
        for ``site.get_op('O', use_left_version=True)`` and ``'(_*O)'`` for the right version.
        """
        if ' ' in name:
            raise NotImplementedError  # TODO redesign the operator mini language?
        if '*' in name:
            name, is_left = split_charged_operator_symbol(name)
            return self.get_op(name, use_left_version=is_left)
        if name in self.symmetric_ops:
            return self.symmetric_ops[name]
        if name in self.charged_ops:
            op = self.charged_ops[name]
            if op.can_use_alone or use_left_version is True:
                return op.op_L
            if use_left_version is False:
                return op.op_R
            msg = (f'Need to specify left or right version for charged operator "{name}". '
                   f'Use e.g. "({name}*_)" or `use_left_version=True` for the left version.')
            raise ValueError(msg)
        raise ValueError(f'Operator not found: {name}')

    def get_hc_op_name(self, name: str) -> str:
        if ' ' in name:
            raise NotImplementedError  # TODO redesign the operator mini language?
        if '*' in name:
            name, is_left = split_charged_operator_symbol(name)
            hc_name = self.get_hc_op_name(name)
            return f'({hc_name}*_)' if is_left else f'(_*{hc_name})'
        return self.hc_ops[name]
    
    def op_needs_JW(self, name: str) -> bool:
        if ' ' in name:
            raise NotImplementedError  # TODO redesign the operator mini language?
        if '*' in name:
            name, _ = split_charged_operator_symbol(name)
        return name in self.need_JW_string

    def is_valid_opname(self, name: str) -> bool:
        if ' ' in name:
            raise NotImplementedError  # TODO redesign the operator mini language?
        if '*' in name:
            try:
                name, _ = split_charged_operator_symbol(name)
            except ValueError:
                return False
        return name in self.all_ops_names
    
    def multiply_op_names(self, names: list[str]) -> str:
        raise NotImplementedError  # TODO redesign the operator mini language?

    def multiply_operators(self, operators: list[str | Tensor]) -> Tensor:
        raise NotImplementedError  # TODO redesign the operator mini language?

    def __repr__(self):
        """Debug representation of self."""
        return f'<Site, d={self.dim:d}, ops={self.all_ops_names!r}>'


class GroupedSite(Site):
    """Group two or more :class:`Site` into a larger one.

    A typical use-case is that you want a NearestNeighborModel for TEBD although you have
    next-nearest neighbor interactions: you just double your local Hilbertspace to consist of
    two original sites.
    Note that this is a 'hack' at the cost of other things (e.g., measurements of 'local'
    operators) getting more complicated/computationally expensive.

    If the individual sites indicate fermionic operators (with entries in `need_JW_string`),
    we construct the new on-site operators of `site1` to include the JW string of `site0`,
    i.e., we use the Kronecker product of ``[JW, op]`` instead of ``[Id, op]`` if necessary
    (but always ``[op, Id]``).
    In that way the onsite operators of this DoubleSite automatically fulfill the
    expected commutation relations. See also :doc:`/intro/JordanWigner`.

    TODO note about order of states and dense reps of the local operators.
         See discussion in test_double_site

    Parameters
    ----------
    sites : list of :class:`Site`
        The individual sites being grouped together. Copied before use if ``charges!='same'``.
    labels :
        Include the Kronecker product of each onsite operator `op` on ``sites[i]`` and
        identities on other sites with the name ``opname+labels[i]``.
        Similarly, set state labels for ``' '.join(state[i]+'_'+labels[i])``.
        Defaults to ``[str(i) for i in range(n_sites)]``, which for example grouping two SpinSites
        gives operators name like ``'Sz0'`` and state labels like ``'up_0 down_1'``.
    symmetry_combine : ``'same' | 'drop' | 'independent'``
        How to handle symmetries, defaults to 'same'.
        ``'same'`` means that all `sites` have the same `Symmetry`, and the total conserved charge
        is the sum of the charges on the individual `sites`.
        ``'independent'`` means that the `sites` have possibly different `Symmetry`, all of which
        are preserved independently (i.e. we assume that the symmetry of one site acts trivially
        on all the others). For ``'drop'``, we drop any symmetries.
        For more complex situations, you can call :func:`set_common_charges` beforehand and then
        use ``'same'``.

    Attributes
    ----------
    n_sites : int
        The number of sites grouped together, i.e. ``len(sites)``.
    sites : list of :class:`Site`
        The sites grouped together into self.
    labels: list of str
        The labels using which the single-site operators are added during construction.
    symmetry_combine : ``'same' | 'drop' | 'independent'``
        Same as the parameter.
    """

    def __init__(self, sites: list[Site], labels=None, symmetry_combine='same'):
        self.n_sites = n_sites = len(sites)
        self.sites = sites
        self.symmetry_combine = symmetry_combine
        assert n_sites > 0
        if labels is None:
            labels = [str(i) for i in range(n_sites)]
        self.labels = labels
        # determine new legs
        if symmetry_combine == 'same':
            legs = [site.leg for site in sites]
            res_symmetry = sites[0].leg.symmetry
        elif symmetry_combine == 'drop':
            legs = [s.leg.drop_symmetry() for s in sites]
        elif symmetry_combine == 'independent':
            legs = []
            all_symmetries = [site.leg.symmetry for site in sites]
            for i, site in enumerate(sites):
                # define the new leg to be in the trivial sector for all symmetries...
                independent_gradings = [
                    ElementarySpace.from_trivial_sector(dim=site.dim, symmetry=s)
                    for s in all_symmetries
                ]
                # ... except for "its own" symmetry
                independent_gradings[i] = site.leg
                legs.append(ElementarySpace.from_independent_symmetries(independent_gradings))
        else:
            raise ValueError("Unknown option for `symmetry_combine`: " + repr(symmetry_combine))
        # change sites to have the new legs
        if symmetry_combine != 'same':
            # copy to avoid modifying the existing sites, then change the leg
            sites = [copy.copy(s).change_leg(l) for s, l in zip(sites, legs)]
            
        # even though Site.__init__ will also set them, we need self.leg and self.backend earlier
        # to use kroneckerproduct
        self.backend = backend = sites[0].backend
        assert all(s.backend == backend for s in sites)
        self.leg = leg = ProductSpace(legs, backend=backend)
        # initialize Site
        JW_all = self.kroneckerproduct([s.JW for s in sites])
        Site.__init__(self, leg, backend=backend, state_labels=None, JW=JW_all)
        # set state labels
        dims = np.array([site.dim for site in sites])
        if leg.symmetry.is_abelian:
            perm = leg.get_basis_transformation_perm()
            strides = make_stride(dims, cstyle=True)
            for states_labels in itertools.product(*[s.state_labels.items() for s in sites]):
                # states_labels is a list of (label, index) pairs for every site
                inds = np.array([i for _, i in states_labels])
                prod_space_idx = np.sum(inds * strides)
                state_label = ' '.join(f'{lbl}_{site_lbl}' for (lbl, _), site_lbl in zip(states_labels, labels))
                self.state_labels[state_label] = perm[prod_space_idx]
        else:
            # TODO fusion is more than a permutation. labels like above make no sense.
            raise NotImplementedError
        # add operators
        Ids = [s.Id for s in sites]
        JW_Ids = Ids[:]  # in the following loop equivalent to [JW, JW, ... , Id, Id, ...]
        for i, site in enumerate(sites):
            # we modify Ids[i] and JW_Ids[i] while handling site i and restore them before
            # moving to the next site
            for name, op in site.symmetric_ops.items():
                if name == 'Id':
                    continue  # we only add a single global Id via Site.__init__
                need_JW = name in site.need_JW_string
                hc = False if name not in site.hc_ops else site.hc_ops[name] + labels[i]
                ops = JW_Ids if need_JW else Ids
                ops[i] = op
                self.add_symmetric_operator(name + labels[i], self.kroneckerproduct(ops),
                                            need_JW=need_JW, hc=hc)
            for name, op in site.charged_ops.items():
                need_JW = name in site.need_JW_string
                hc = False if name not in site.hc_ops else site.hc_ops[name] + labels[i]
                ops = JW_Ids if need_JW else Ids
                ops[i] = op.op_L
                new_op_L = self.kroneckerproduct(ops)
                ops[i] = op.op_R
                new_op_R = self.kroneckerproduct(ops)
                new_op = ChargedOperator(new_op_L, new_op_R, can_use_alone=op.can_use_alone)
                self.add_charged_operator(name + labels[i], new_op, need_JW=need_JW, hc=hc)
            Ids[i] = site.Id
            JW_Ids[i] = site.JW

    def kroneckerproduct(self, ops):
        r"""Return the Kronecker product :math:`op0 \otimes op1` of local operators.

        Parameters
        ----------
        ops : list of :class:`~tenpy.linalg.tensor.Tensor`
            One operator (or operator name) on each of the ungrouped sites.
            Each operator should have labels ``['p', 'p*']``.

        Returns
        -------
        prod : :class:`~tenpy.linalg.tensor.Tensor`
            Kronecker product :math:`ops[0] \otimes ops[1] \otimes \cdots`,
            with labels ``['p', 'p*']``.
        """
        if all(isinstance(op, DiagonalTensor) for op in ops):
            # TODO proper implementation?
            # note that block_kron is associative, order does not matter
            diag = reduce(self.backend.block_kron, (op.diag_block for op in ops))
            return DiagonalTensor.from_diag(diag, self.leg, backend=self.backend, labels=['p', 'p*'])
        for i, op in enumerate(ops):
            if isinstance(op, DiagonalTensor):
                ops[i] = op.as_Tensor()
        op = ops[0].relabel({'p': 'p0', 'p*': 'p0*'}, inplace=False)
        for i, op_i in enumerate(ops[1:], start=1):
            op = op.outer(op_i, relabel2={'p': f'p{i}', 'p*': f'p{i}*'})
        return op.combine_legs([f'p{i}' for i in range(self.n_sites)],
                               [f'p{i}*' for i in range(self.n_sites)],
                               product_spaces=[self.leg, self.leg.dual],
                               new_labels=['p', 'p*'])

    def __repr__(self):
        """Debug representation of self."""
        return f'GroupedSite({self.sites!r}, {self.labels!r}, {self.symmetry_combine!r})'


def group_sites(sites, n=2, labels=None, symmetry_combine='same'):
    """Given a list of sites, group each `n` sites together.

    Parameters
    ----------
    sites : list of :class:`Site`
        The sites to be grouped together.
    n : int
        We group each `n` consecutive sites from `sites` together in a :class:`GroupedSite`.
    labels, symmetry_combine :
        See :class:`GroupedSite`.

    Returns
    -------
    grouped_sites : list of :class:`GroupedSite`
        The grouped sites. Has length ``(len(sites)-1)//n + 1``.
    """
    grouped_sites = []
    if labels is None:
        labels = [str(i) for i in range(n)]
    for i in range(0, len(sites), n):
        group = sites[i:i + n]
        s = GroupedSite(group, labels[:len(group)], symmetry_combine)
        grouped_sites.append(s)
    return grouped_sites


def set_common_symmetry(sites: list[Site], symmetry_combine: callable | str | list = 'by_name',
                        new_symmetry: Symmetry = None):
    """Adjust the symmetries of the given sites *in place* such that they can be used together.

    Before we can contract operators (and tensors) corresponding to different :class:`Site`
    instances, we first need to define the overall symmetry, i.e., we need to merge the
    :class:`~tenpy.linalg.groups.Symmetry` of their :attr:`Site.leg`s to a single, global symmetry
    and adjust the sectors of the physical legs. That's what this function does.

    A typical place to do this would be in :meth:`tenpy.models.model.CouplingMPOModel.init_sites`.

    Parameters
    ----------
    sites : list of :class:`Site`
        The sites to be combined. The sites are modified **in place** and thus may not contain
        the same object multiple times.
    symmetry_combine : ``'by_name'`` | ``'drop'`` | ``'independent'`` | function | list
        Defines how the new common symmetry arises from the individual symmetries.

        ``'by_name'``
            Default. Any :class:`ProductSymmetry` is split into its factors. We then considers
            equal symmetries (i.e. same mathematical group and same name) to be the same symmetry
            (i.e. the sum of their charges is conserved). Unequal symmetries are considered as
            independent (i.e. their charges are conserved individually).
        
        ``'independent'``
            Consider all factors as independent, even if they have the same name.
            Exception: any unneeded :class:`NoSymmetry` instances are ignored.
        
        ``'drop'``
            Drop all symmetries.
        
        function
            A function with call structure

                new_sector: SectorArray = symmetry_combine(site_idx: int, old_sector: SectorArray)

            That specifies how the sectors of the new symmetry arise from those of the old symmetry,
            i.e. ``old_sector`` is a sector of the old symmetry on the ``site_idx``-th site.
            Using this option makes the `new_symmetry` parameter required.

        list of list of tuple
            Specifies new sectors as linear combinations of the old ones. Requires `new_symmetry`.
            Each entry of the outer list specifies a column of the resulting sectors, e.g. for one
            factor of a resulting :class:`ProductSymmetry`. The inner list goes over terms to be
            added up and its entries are tuples ``(prefactor, site_idx, old_col_idx)``, indicating
            that ``prefactor * old_sectors[:, old_col_idx]`` of the ``site_idx``-th of the `sites`
            is a term in the sum. The ``prefactor``s may be non-integer, as long as the resulting
            sectors are integer.

    new_symmetry : :class:`Symmetry`, optional
        The new symmetry. Is ignored if `symmetry_combine` is one of the pre-defined (``str``)
        options. Is required if `symmetry_combine` is a function or a list.

    TODO examples and doctests
    """
    for i, site in enumerate(sites):
        for site2 in sites[i + 1:]:
            if site2 is site:
                raise ValueError('`sites` contains the same object multiple times. Make copies!')

    if symmetry_combine == 'by_name':
        factors = []
        sites_and_slices = []  # for every factor, a list of tuples (site_idx, sector_slc)
                               # indicating that this factor appears on sites[site_idx] and its
                               # sectors are embedded at sector_slc of that sites symmetry
        for i, site in enumerate(sites):
            if isinstance(site.leg.symmetry, ProductSymmetry):
                new_factors = site.leg.symmetry.factors
                sector_slices = site.leg.symmetry.sector_slices
            else:
                new_factors = [site.leg.symmetry]
                sector_slices = [0, site.leg.symmetry.sector_ind_len]
            for n, f in enumerate(new_factors):
                slc = slice(sector_slices[n], sector_slices[n + 1])
                # Symmetry.__eq__ checks for same mathematical group *and* same descriptive_name
                if f in factors:
                    n = factors.index(f)
                    sites_and_slices[n].append((i, slc))
                else:
                    factors.append(f)
                    sites_and_slices.append([(i, slc)])
        if len(factors) == 1:
            new_symmetry = factors[0]
            new_symm_slices = [0, new_symmetry.sector_ind_len]
        else:
            new_symmetry = ProductSymmetry(factors)
            new_symm_slices = [slice(new_symmetry.sector_slices[n], new_symmetry.sector_slices[n + 1])
                               for n in range(len(factors))]
        for i, site in enumerate(sites):
            slice_tuples = []  # list of tuples (new_slice, old_slice) indicating that
                               # for this site, new_sector[new_slice] = old_sector[old_slice]
            for n in range(len(factors)):
                for j, slc in sites_and_slices[n]:
                    if i == j:
                        slice_tuples.append((new_symm_slices[n], slc))
            
            def symmetry_combine(s):
                mapped_sectors = np.tile(new_symmetry.trivial_sector[None, :], (len(s), 1))
                for mapped_slice, old_slice in slice_tuples:
                    mapped_sectors[:, mapped_slice] = s[:, old_slice]
                return mapped_sectors

            new_leg = site.leg.change_symmetry(symmetry=new_symmetry, sector_map=symmetry_combine)
            site.change_leg(new_leg)
        return
    
    if symmetry_combine == 'drop':
        assert new_symmetry is None
        for site in sites:
            site.change_leg(site.leg.drop_symmetry())
        return

    if symmetry_combine == 'independent':
        factors = []
        ind_lens = []  # basically [site.leg.symmetry.sector_ind_len for site in sites]
                       # but adjusted for the ignored no_symmetry s
        for site in sites:
            site_symmetry = site.symmetry
            if isinstance(site_symmetry, ProductSymmetry):
                new_factors = [f for f in site_symmetry.factors if f != no_symmetry]
                ind_lens.append(sum(f.sector_ind_len for f in new_factors))
                factors.extend(new_factors)
            elif site_symmetry == no_symmetry:
                ind_lens.append(0)
            else:
                ind_lens.append(site_symmetry.sector_ind_len)
                factors.append(site_symmetry)
        if len(factors) == 0:
            new_symmetry = no_symmetry
        elif len(factors) == 1:
            new_symmetry = factors[0]
        else:
            new_symmetry = ProductSymmetry(factors)

        start = 0
        for site, ind_len in zip(sites, ind_lens):

            def symmetry_combine(s):
                res = np.tile(new_symmetry.trivial_sector[None, :], (len(s), 1))
                res[:, start:start + ind_len] = s
                return res

            site.change_leg(site.leg.change_symmetry(symmetry=new_symmetry, sector_map=symmetry_combine))
            start = start + ind_len
        return

    elif isinstance(symmetry_combine, str):
        raise ValueError(f'Unknown sector_map keyword: "{symmetry_combine}"')

    if isinstance(symmetry_combine, list):
        input_symmetry_combine = symmetry_combine[:]

        def symmetry_combine(site_idx, old_sectors: SectorArray) -> SectorArray:
            cols = []
            for col_spec in input_symmetry_combine:
                col = np.zeros((len(old_sectors),), dtype=int)
                for factor, i, col_idx in col_spec:
                    assert isinstance(col_idx, int)
                    if i != site_idx:
                        continue
                    col = col + factor * old_sectors[:, col_idx]
                cols.append(np.rint(col))
                if not np.allclose(col, cols[-1]):
                    raise ValueError(f'Sectors must have integer entries. Got {col}')
            return np.stack(cols, axis=1)
        
    # can now assume that sector_map is an actual function
    # with signature (site_idx: int, sectors: SectorArray) -> SectorArray
    if new_symmetry is None:
        raise ValueError('Need to specify new_symmetry')
    for i, site in enumerate(sites):
        new_leg = site.leg.change_symmetry(symmetry=new_symmetry, sector_map=partial(symmetry_combine, i))
        site.change_leg(new_leg)


# ------------------------------------------------------------------------------
# Operator mini-language utility functions


def as_valid_operator_name(name) -> str:
    """Convert to a valid operator name.

    Operator names may not include whitespace.
    TODO are there other restrictions?
    """
    name = str(name)
    if ' ' in name:
        raise ValueError(f'Invalid operator name: "{name}"')
    return name


def split_charged_operator_symbol(name: str) -> tuple[str, bool]:
    """Helper function to parse charged operators ``"(O*_)"`` or ``"(_*O)"``.
    Returns
    -------
    name : str
    is_left : bool
    """
    assert '*' in name
    if not (name.startswith('(') and name.endswith(')')):
        raise ValueError(f'Invalid operator name: {name}')
    left, right, *more = name[1:-1].split('*')
    if more:
        raise ValueError(f'Invalid operator name: {name}')
    if left == '_':
        return right, False
    if right == '_':
        return left, True
    raise ValueError(f'Invalid operator name: {name}')


# TODO is the name to close to ChargedTensor?
class ChargedOperator:
    """Charged operators for a :class:`Site`.

    Parameters
    ----------
    op_L, op_R : :class:`ChargedTensor`
        Left and right version of the operator. See attributes below. If the right version is not
        given, it defaults to ``op_L.flip_dummy_leg_duality()``.
    can_use_alone : bool, optional
        If `op_L` or `op_R` have one-dimensional dummy legs and give the same blocks via
        :meth:`ChargedTensor.to_dense_block()`. As a consequence, it is safe to use them "alone",
        i.e. without the context of a partner on a different site

    Attributes
    ----------
    op_L : :class:`ChargedTensor`
        Left version of the operator. Has a ket-like dummy leg with ``is_dual=False``.
    op_R : :class:`ChargedTensor`
        Right version of the operator. Has a bra-like dummy leg with ``is_dual=True``.
    can_use_alone : bool
    
    """

    def __init__(self, op_L: ChargedTensor, op_R: ChargedTensor = None, can_use_alone: bool = None):
        if op_R is None:
            op_R = op_L.flip_dummy_leg_duality()
            if can_use_alone is None:
                can_use_alone = True
        elif can_use_alone is None:
            try:
                b_L = op_L.to_dense_block()
                b_R = op_R.to_dense_block()
            except ValueError:
                can_use_alone = False
            else:
                # TODO what should atol, rtol be? also in test_sanity
                can_use_alone = op_L.backend.block_allclose(b_L, b_R, atol=1e-5, rtol=1e-8)
        self.op_L = op_L
        self.op_R = op_R
        self.can_use_alone = can_use_alone

    def test_sanity(self):
        assert self.op_L.backend == self.op_R.backend
        assert self.op_L.dummy_leg.dim == self.op_R.dummy_leg.dim
        if self.can_use_alone:
            L = self.op_L.to_dense_block()
            R = self.op_R.to_dense_block()
            assert self.backend.block_allclose(L, R, atol=1e-5, rtol=1e-8)
        self.op_L.test_sanity()
        self.op_R.test_sanity()

    @property
    def backend(self) -> TensorBackend:
        return self.op_L.backend

    @property
    def dummy_leg_dim(self) -> int:
        return self.op_L.dummy_leg.dim


# ------------------------------------------------------------------------------
# The most common local sites.


class SpinHalfSite(Site):
    r"""Spin-1/2 site.

    Local states are ``up`` (0) and ``down`` (1).
    
    Possible symmetries are::

    ==============  =====================  ==================  =========================
    `conserve`      symmetry               sectors of basis    meaning of sector label
    ==============  =====================  ==================  =========================
    ``'Stot'``      ``SU2Symmetry``        ``[1, 1]``          ``2 * S``
    ``'Sz'``        ``U1Symmetry``         ``[1, -1]``         ``2 * Sz``
    ``'parity'``    ``ZNSymmetry(N=2)``    ``[1, 0]``          ``(Sz + .5) % 2``
    ``'None'``      ``NoSymmetry``         ``[0, 0]``          --
    ==============  =====================  ==================  =========================

    TODO include dipole symmetry here too?
    
    Local operators are the usual spin-1/2 operators, e.g. ``Sz = [[0.5, 0.], [0., -0.5]]``,
    ``Sx = 0.5 * Sigmax`` for the Pauli matrix `Sigmax`.

    The symmetric operators are (columns indicate `conserve`)::

    ====================  =================================  ========  ========  ========  ========
    operator              description                        Stot      Sz        parity    None
    ====================  =================================  ========  ========  ========  ========
    ``Id, JW``            Identity :math:`\mathbb{1}`        diag      diag      diag      diag
    ``Sz``                Spin component :math:`S^z`         --        diag      diag      diag
    ``Sx, Sy``            Spin components :math:`S^{x,y}`    --        --        --        tens
    ``Sp, Sm``            :math:`S^{\pm} = S^x \pm i S^y`    --        --        --        tens
    ``Sigmaz``            Pauli matrix :math:`\sigma^{z}`    --        diag      diag      diag
    ``Sigmax, Sigmay``    Pauli matrices x & y               --        --        --        tens
    ====================  =================================  ========  ========  ========  ========

    The charged operators are (columns indicate `conserve`, entries are dummy leg dimensions)::
    # TODO should we list the "symmetric dimension" or the "dense dimension", i.e. 1 or 3 for Svec?

    ====================  =================================  ========  ========  ========  ========
    operator              description                        Stot      Sz        parity    None
    ====================  =================================  ========  ========  ========  ========
    ``Sx, Sy``            Spin components :math:`S^{x,y}`    --        2         1         1
    ``Sp, Sm``            :math:`S^{\pm} = S^x \pm i S^y`    --        1         1         1
    ``Sigmax, Sigmay``    Pauli matrices x & y               --        2         1         1
    ``Svec``              The vector of spin operators       1         3         3         3
    ====================  =================================  ========  ========  ========  ========

    Parameters
    ----------
    conserve : 'Stot' | 'Sz' | 'parity' | 'None'
        Defines what is conserved, see table above.
    backend : :class:`~tenpy.linalg.backends.Backend`, optional
        The backend used to create the operators.
    """
    def __init__(self, conserve: str = 'Sz', backend: TensorBackend = None):
        # make leg
        if conserve == 'Stot':
            leg = ElementarySpace(symmetry=SU2Symmetry('Stot'), sectors=[[1]])
        elif conserve == 'Sz':
            leg = ElementarySpace.from_sectors(U1Symmetry('2*Sz'), [[1], [-1]])
        elif conserve == 'parity':
            leg = ElementarySpace.from_sectors(ZNSymmetry(2, 'parity_Sz'), [[1], [0]])
        elif conserve == 'None':
            leg = ElementarySpace.from_trivial_sector(2)
        else:
            raise ValueError(f'invalid `conserve`: {conserve}')
        # initialize Site
        self.conserve = conserve
        Site.__init__(self, leg=leg, backend=backend, state_labels=['up', 'down'])
        # further aliases for state labels
        self.state_labels['-0.5'] = self.state_labels['down']
        self.state_labels['0.5'] = self.state_labels['up']
        # operators : Svec, Sz, Sigmaz, Sp, Sm
        if conserve == 'Stot':
            # vector transforms under spin-1 irrep -> sector == [2 * J] == [2]
            dummy_leg = ElementarySpace(leg.symmetry, sectors=[[2]])
            Svec_inv = SymmetricTensor.from_block_func(
                self.backend.ones_block, backend=self.backend, legs=[leg, leg.dual, dummy_leg],
                labels=['p', 'p*', '!']
            )
            self.add_charged_operator('Svec', ChargedTensor(Svec_inv), hc='Svec')
        else:
            Sz = self.add_symmetric_operator('Sz', [.5, -.5])
            self.add_symmetric_operator('Sigmaz', 2 * Sz)
            self.add_any_operator('Sp', [[0., 1.], [0., 0.]], hc=False)  # hc='Sm' is not added yet
            self.add_any_operator('Sm', [[0., 0.], [1., 0.]], hc='Sp')
        # operators : Sx, Sy, Sigmax, Sigmay
        if conserve == 'Sz':
            # TODO add them with two-dim dummy leg.
            #      test (in particular check if e.g. (Sx @ _) and (_ @ Sx) combine as expected).
            pass
        if conserve in ['parity', 'None']:
            Sx = self.add_any_operator('Sx', [[0., 0.5], [0.5, 0.]], hc='Sx')
            Sy = self.add_any_operator('Sy', [[0., -0.5j], [+0.5j, 0.]], hc='Sy')
            self.add_any_operator('Sigmax', 2 * Sx, hc='Sigmax')
            self.add_any_operator('Sigmay', 2 * Sy, hc='Sigmay')

    def __repr__(self):
        """Debug representation of self."""
        return f'SpinHalfSite({self.conserve})'


class SpinSite(Site):
    r"""General Spin S site.

    There are `2S+1` local states range from ``down`` (0)  to ``up`` (2S+1),
    corresponding to ``Sz=-S, -S+1, ..., S-1, S``.

    ==============  =====================  ==========================  ==========================
    `conserve`      symmetry               sectors of basis            meaning of sector label
    ==============  =====================  ==========================  ==========================
    ``'Stot'``      ``SU2Symmetry``        ``[2 * S] * S``             total spin ``2 * S``
    ``'Sz'``        ``U1Symmetry``         ``[-2 * S, ..., 2 * S]``    magnetization ``2 * Sz``
    ``'parity'``    ``ZNSymmetry(N=2)``    ``[0, 1, 0, 1, ...]``       parity ``(Sz + S) % 2``
    ``'None'``      ``NoSymmetry``         ``[0] * S``                 --
    ==============  =====================  ==========================  ==========================

    Local operators are the spin-S operators,
    e.g. ``Sz = [[0.5, 0.], [0., -0.5]]``, ``Sx = [[0., 0.5], [0.5, 0.]]`` for ``S=.5``.

    The symmetric operators are (columns indicate `conserve`)::

    ====================  =================================  ========  ========  ========  ========
    operator              description                        Stot      Sz        parity    None
    ====================  =================================  ========  ========  ========  ========
    ``Id, JW``            Identity :math:`\mathbb{1}`        diag      diag      diag      diag
    ``Sz``                Spin component :math:`S^z`         --        diag      diag      diag
    ``Sx, Sy``            Spin components :math:`S^{x,y}`    --        --        --        tens
    ``Sp, Sm``            :math:`S^{\pm} = S^x \pm i S^y`    --        --        --        tens
    ====================  =================================  ========  ========  ========  ========

    The charged operators are (columns indicate `conserve`, entries are dummy leg dimensions)::

    ====================  =================================  ========  ========  ========  ========
    operator              description                        Stot      Sz        parity    None
    ====================  =================================  ========  ========  ========  ========
    ``Sx, Sy``            Spin components :math:`S^{x,y}`    --        2         2         1
    ``Sp, Sm``            :math:`S^{\pm} = S^x \pm i S^y`    --        1         1         1
    ``Svec``              The vector of spin operators       1         3         3         3
    ====================  =================================  ========  ========  ========  ========

    Note that unlike for the :class:`SpinHalfSite`, ``Sx, Sy`` have two-dimensional dummy legs
    even for ``conserve='parity'``.

    Parameters
    ----------
    S : {0.5, 1, 1.5, 2, ...}
        The 2S+1 states range from m = -S, -S+1, ... +S.
    conserve : 'Stot' | 'Sz' | 'parity' | 'None'
        Defines what is conserved, see table above.
    backend : :class:`~tenpy.linalg.backends.Backend`, optional
        The backend used to create the operators.
    """

    def __init__(self, S: float = 0.5, conserve: str = 'Sz', backend: TensorBackend = None):
        self.S = S = float(S)
        d = 2 * S + 1
        if d <= 1:
            raise ValueError("negative S?")
        if np.rint(d) != d:
            raise ValueError("S is not half-integer or integer")
        d = int(d)
        two_Sz = np.arange(1 - d, d + 1, 2)
        Sp = np.zeros([d, d], dtype=float)
        for n in range(d - 1):
            # Sp |m> = sqrt( S(S+1)-m(m+1)) |m>
            m = n - S
            Sp[n + 1, n] = np.sqrt(S * (S + 1) - m * (m + 1))
        Sm = np.transpose(Sp)  # no need to conj, Sp is real
        # make leg
        if conserve == 'Stot':
            leg = ElementarySpace(symmetry=SU2Symmetry('Stot'), sectors=[[d - 1]])
        elif conserve == 'Sz':
            leg = ElementarySpace.from_sectors(U1Symmetry('2*Sz'), two_Sz[:, None])
        elif conserve == 'parity':
            leg = ElementarySpace.from_basis(ZNSymmetry(2, 'parity_Sz'), np.arange(d)[:, None] % 2)
        elif conserve == 'None':
            leg = ElementarySpace.from_trivial_sector(d)
        else:
            raise ValueError(f'invalid `conserve`: {conserve}')
        # initialize Site
        names = [str(i) for i in np.arange(-S, S + 1, 1.)]
        Site.__init__(self, leg=leg, backend=backend, state_labels=names)
        self.state_labels['down'] = self.state_labels[names[0]]
        self.state_labels['up'] = self.state_labels[names[-1]]
        # operators : Svec, Sz, Sp, Sm
        if conserve == 'Stot':
            dummy_leg = ElementarySpace(leg.symmetry, sectors=[[2]])
            Svec_inv = SymmetricTensor.from_block_func(
                self.backend.ones_block, legs=[leg, leg.dual, dummy_leg],
                labels=['p', 'p*', '!']
            )
            self.add_charged_operator('Svec', ChargedTensor(Svec_inv), hc='Svec')
        else:
            self.add_symmetric_operator('Sz', two_Sz / 2.)
            self.add_any_operator('Sp', Sp, hc=False)  # hc='Sm', but Sm is not added yet
            self.add_any_operator('Sm', Sm, hc='Sp')
        # operators : Sx, Sy
        if conserve in ['Sz', 'parity']:
            Sp_inv = self.charged_ops['Sp'].op_L.invariant_part
            Sm_inv = self.charged_ops['Sm'].op_L.invariant_part
            # Sx_inv = concatenate([.5 * Sp_inv, .5 * Sm_inv], leg=-1)
            # Sy_inv = concatenate([-.5j * Sp_inv, .5j * Sm_inv], leg=-1)
            # self.add_charged_operator('Sx', ChargedTensor(Sx_inv), hc='Sx')
            # self.add_charged_operator('Sy', ChargedTensor(Sy_inv), hc='Sy')
            # TODO uncomment (possibly adjust) when concatenate is implemented. activate test.
        elif conserve in ['None']:
            self.add_symmetric_operator('Sx', 0.5 * (Sp + Sm), hc='Sx', also_as_charged=True)
            self.add_symmetric_operator('Sy', 0.5j * (Sm - Sp), hc='Sy', also_as_charged=True)
            # Note: For S=1/2, Sy might look wrong compared to the Pauli matrix or SpinHalfSite.
            # Don't worry, I'm 99.99% sure it's correct (J. Hauschild). Mee too (J. Unfried).
            # The reason it looks wrong is simply that this class orders the states as ['down', 'up'],
            # while the usual spin-1/2 convention is ['up', 'down'], as you can also see if you look
            # at the Sz entries...
            # (The commutation relations are checked explicitly in `tests/test_site.py`)

    def __repr__(self):
        """Debug representation of self."""
        return f'SpinSite(S={self.S}, {self.conserve})'


class FermionSite(Site):
    r"""A site with a single species of spin-less fermions.

    Local states are ``empty`` and ``full``.
    
    .. warning ::
        Using the Jordan-Wigner string (``JW``) is crucial to get correct results,
        otherwise you just describe hardcore bosons!
        Further details in :doc:`/intro/JordanWigner`.

    ==============  =====================  ==================  ==========================
    `conserve`      symmetry               sectors of basis       meaning of sector label
    ==============  =====================  ==================  ==========================
    ``'N'``         ``U1Symmetry``         ``[0, 1]``          number of fermions
    ``'parity'``    ``ZNSymmetry(N=2)``    ``[0, 1]``          parity of fermion number
    ``'None'``      ``NoSymmetry``         ``[0, 0]``          --
    ==============  =====================  ==================  ==========================

    TODO how to control if tenpy should use JW-strings or tenpy.linalg.groups.FermionParity?

    Local operators are composed of the fermionic creation ``'Cd'`` and annihilation ``'C'``
    operators. Note that the local operators do *not* include the Jordan-Wigner strings that
    make them fulfill the proper commutation relations.

    The symmetric operators are (columns indicate `conserve`)::

    ===========  =======================================  ========  ========  ========
    operator     description                              N         parity    None
    ===========  =======================================  ========  ========  ========
    ``Id``       Identity :math:`\mathbb{1}`              diag      diag      diag
    ``JW``       Sign for the Jordan-Wigner string.       diag      diag      diag
    ``C``        Annihilation operator :math:`c`          --        --        tens
    ``Cd``       Creation operator :math:`c^\dagger`      --        --        tens
    ``N``        Number operator :math:`n= c^\dagger c`   diag      diag      diag
    ``dN``       :math:`\delta n := n - filling`          diag      diag      diag
    ``dNdN``     :math:`(\delta n)^2`                     diag      diag      diag
    ===========  =======================================  ========  ========  ========

    The charged operators are (columns indicate `conserve`, entries are dummy leg dimensions)::

    ===========  =======================================  ========  ========  ========
    operator     description                              N         parity    None
    ===========  =======================================  ========  ========  ========
    ``C, Cd``    as above                                 1         1         1
    ===========  =======================================  ========  ========  ========

    Parameters
    ----------
    conserve : 'N' | 'parity' | 'None'
        Defines what is conserved, see table above.
    filling : float
        Average filling. Used to define ``dN``.
    backend : :class:`~tenpy.linalg.backends.Backend`, optional
        The backend used to create the operators.
    """

    def __init__(self, conserve: str = 'N', filling: float = 0.5, backend: TensorBackend = None):
        # make leg
        if conserve == 'N':
            leg = ElementarySpace.from_sectors(U1Symmetry('N'), [[0], [1]])
        elif conserve == 'parity':
            leg = ElementarySpace.from_sectors(ZNSymmetry(2, 'parity_N'), [[0], [1]])
        elif conserve == 'None':
            leg = ElementarySpace.from_trivial_sector(2)
        else:
            raise ValueError(f'invalid `conserve`: {conserve}')
        # initialize site
        self.filling = filling
        self.conserve = conserve
        Site.__init__(self, leg, backend=backend, state_labels=['empty', 'full'], JW=[1., -1.])
        # operators
        N_diag = np.array([0., 1.])
        self.add_symmetric_operator('N', np.diag(N_diag))
        self.add_symmetric_operator('dN', np.diag(N_diag - filling))
        self.add_symmetric_operator('dNdN', np.diag((N_diag - filling) ** 2))
        self.add_any_operator('C', [[0., 1.], [0., 0.]], need_JW=True, hc=False)
        self.add_any_operator('Cd', [[0., 0.], [1., 0.]], need_JW=True, hc='C')

    def __repr__(self):
        """Debug representation of self."""
        return f'FermionSite({self.conserve}, filling={self.filling:f})'


class SpinHalfFermionSite(Site):
    r"""A site with a single species of spinful (spin-1/2) fermions.

    Local states are:
        ``empty``  (vacuum),
        ``up``     (one spin-up electron),
        ``down``   (one spin-down electron), and
        ``full``   (both electrons)

    .. warning ::
        Using the Jordan-Wigner string (``JW``) in the correct way is crucial to get correct
        results, otherwise you just describe hardcore bosons!

    The possible symmetries factorize into the occupation number and spin sectors

    ==============  =====================  =====================  ==================================
    `conserve_N`    symmetry               sectors of basis       meaning of sector label
    ==============  =====================  =====================  ==================================
    ``'N'``         ``U1Symmetry``         ``[0, 1, 1, 2]``       total number ``N`` of fermions
    ``'parity'``    ``ZNSymmetry(N=2)``    ``[0, 1, 1, 0]``       parity ``N % 2`` of fermion number
    ``'None'``      --                     ``[0, 0, 0, 0]``       --
    ==============  =====================  =====================  ==================================

    ==============  =====================  =====================  ==============================
    `conserve_S`    symmetry               sectors of basis       meaning of sector label
    ==============  =====================  =====================  ==============================
    ``'Stot'``      ``SU2Symmetry``        ``[0, 1, 1, 0]``       total spin : ``2 * S``
    ``'Sz'``        ``U1Symmetry``         ``[0, 1, -1, 0]``      magnetization: ``2 * Sz``
    ``'parity'``    ``ZNSymmetry(N=4)``    ``[0, 1, 3, 0]``       ``(2 * Sz) % 4``
    ``'None'``      --                     ``[0, 0, 0, 0]``       --
    ==============  =====================  =====================  ==============================

    Local operators are composed of the fermionic creation operators ``Cdd, Cdu`` of up (down)
    spin electrons and corresponding annihilation operators ``Cu, Cd``.
    The spin operators are defined as :math:`S^\gamma =
    (c^\dagger_{\uparrow}, c^\dagger_{\downarrow}) \sigma^\gamma (c_{\uparrow}, c_{\downarrow})^T`,
    where :math:`\sigma^\gamma` are spin-1/2 matrices (i.e. half the pauli matrices).
    TODO its a bit unfortunate to abbreviate "down" and "dagger" with the same letter "d"...
    Note that the local operators do *not* include the Jordan-Wigner strings that
    make them fulfill the proper commutation relations.

    ===========  ===================================================================================
    operator     description
    ===========  ===================================================================================
    ``Id``       Identity :math:`\mathbb{1}`
    ``JW``       Jordan-Wigner sign :math:`(-1)^{n_{\uparrow} + n_{\downarrow}}`
    ``JWu``      Partial sign :math:`(-1)^{n_{\uparrow}}`
    ``JWd``      Partial sign :math:`(-1)^{n_{\downarrow}}`
    ``Cu``       Spin-up annihilation operator :math:`c_{\uparrow}` (up to JW string)
    ``Cdu``      Spin-up creation operator :math:`c^\dagger_{\uparrow}` (up to JW string)
    ``Cd``       Spin-down annihilation operator :math:`c_{\downarrow}` (up to JW string)
    ``Cdd``      Spin-down creation operator :math:`c^\dagger_{\downarrow}` (up to JW string)
    ``Nu``       Spin-up occupation number :math:`n_{\uparrow}= c^\dagger_{\uparrow} c_{\uparrow}`
    ``Nd``       Spin-down occ. number :math:`n_{\downarrow}= c^\dagger_{\downarrow} c_{\downarrow}`
    ``NuNd``     Product of occupations :math:`n_{\uparrow} n_{\downarrow}`
    ``Ntot``     Total occupation number :math:`n_t = n_{\uparrow} + n_{\downarrow}`
    ``dN``       Occupation imbalance :math:`\delta n = n_t - \mathtt{filling}`
    ``Sz``       Spin z-components :math:`S^z = \frac{1}{2}( n_\uparrow - n_\downarrow )`
    ``Sp, Sm``   Spin flips :math:`S^+ = c^\dagger_{\uparrow} c_{\downarrow} = (S^-)^\dagger`
    ``Sx, Sy``   Spin components such that e.g. :math:`S^{\pm} = S^{x} \pm i S^{y}`
    ``Svec``     TODO implement
    ===========  ===================================================================================

    For the availability of operators, we distinguish the following
    different cases of what is conserved:
        (a)  ``conserve_S == 'Stot'`` and any `conserve_N`
        (b)  ``conserve_S == 'Sz'`` and any `conserve_N`
        (c)  ``conserve_S == 'parity'`` and any `conserve_N`
        (d)  ``conserve_S == 'None'`` and ``conserve_N in ['N', 'parity]``
        (e)  ``conserve_S == 'None'`` and ``conserve_N == 'None'``

    The symmetric operators are (columns indicate the cases above)::

    ========================  ========  ========  ========  ========  =======
    operator                  (a)       (b)       (c)       (d)       (e)
    ========================  ========  ========  ========  ========  =======
    Id, JW, NuNd, Ntot, dN    diag      diag      diag      diag      diag
    JWu, Jwd, Nu, Nd, Sz      --        diag      diag      diag      diag
    Cu, Cdu, Cd, Cdd          --        --        --        --        tens
    Sp, Sm, Sx, Sy            --        --        --        tens      tens
    ========================  ========  ========  ========  ========  =======

    The charged operators are (columns indicate the cases above, entries are dummy leg dimensions)::

    ========================  ========  ========  ========  ========  =======
    operator                  (a)       (b)       (c)       (d)       (e)
    ========================  ========  ========  ========  ========  =======
    Cu, Cdu, Cd, Cdd          --        1         1         1         1
    Sp, Sm                    --        1         1         1         1
    Sx, Sy                    --        2         1         1         1
    Svec (TODO)               1         3         3         3         3
    ========================  ========  ========  ========  ========  =======

    Parameters
    ----------
    cons_N : ``'N' | 'parity' | 'None'``
        Whether particle number is conserved, c.f. table above.
    cons_Sz : ``'Sz' | 'parity' | 'None'``
        Whether spin is conserved, c.f. table above.
    filling : float
        Average filling. Used to define ``dN``.
    backend : :class:`~tenpy.linalg.backends.Backend`, optional
        The backend used to create the operators.
    """

    def __init__(self, conserve_N: str = 'N', conserve_S: str = 'Sz', filling: float = 1.,
                 backend: TensorBackend = None):
        # parse conserve_N
        if conserve_N == 'N':
            sectors_N = np.array([0, 1, 1, 2])
            sym_N = U1Symmetry('N')
        elif conserve_N == 'parity':
            sectors_N = np.array([0, 1, 1, 0])
            sym_N = ZNSymmetry(2, 'parity_N')
        elif conserve_N == 'None':
            sectors_N = None
            sym_N = None
        else:
            raise ValueError(f'invalid `conserve_N`: {conserve_N}')
        # parse conserve_S
        if conserve_S == 'Stot':
            # empty is a spin-0 singlet, [up, down] are a spin-1/2 doublet
            # full is spin-0. to see this consider e.g. that all spin operators S^{x,y,z} annihilate
            # |full>, thus \vect{S}^2 |full> = 0.
            sectors_S = np.array([0, 1, 1, 0])
            sym_S = SU2Symmetry('Stot')
            raise NotImplementedError  # TODO, need to special case the operator construction too..
        elif conserve_S == 'Sz':
            sectors_S = np.array([0, 1, -1, 0])
            sym_S = U1Symmetry('2*Sz')
        elif conserve_S == 'parity':
            sectors_S = np.array([0, 1, 3, 0])
            sym_S = ZNSymmetry(4, 'parity_Sz')
        elif conserve_S == 'None':
            sectors_S = None
            sym_S = None
        else:
            raise ValueError(f'invalid `conserve_S`: {conserve_S}')
        # make leg
        if sym_N is None and sym_S is None:
            leg = ElementarySpace.from_trivial_sector(4)
        elif sym_N is None:
            leg = ElementarySpace.from_basis(sym_S, sectors_S[:, None])
        elif sym_S is None:
            leg = ElementarySpace.from_basis(sym_N, sectors_N[:, None])
        else:
            leg = ElementarySpace.from_basis(sym_N * sym_S, np.stack([sectors_N, sectors_S], axis=1))
        self.conserve_N = conserve_N
        self.conserve_S = conserve_S
        self.filling = filling
        Site.__init__(self, leg=leg, backend=backend, state_labels=['empty', 'up', 'down', 'full'],
                      JW=[1., -1., -1., 1])
        # operators : NuNd, Ntot, dN
        self.add_symmetric_operator('NuNd', [0., 0., 0., 1.])
        Ntot = self.add_symmetric_operator('Ntot', [0., 1., 1., 2.])
        self.add_symmetric_operator('dN', Ntot - filling * self.Id)
        # operators : Svec, JWu, JWd, Nu, Nd, Sz, Cu, Cdu, Cd, Cdd, Sp, Sm
        if conserve_S == 'Stot':
            sector = [2]  # spin 1
            if sym_N is not None:
                sector.append(0)
            dummy_leg = ElementarySpace(leg.symmetry, sectors=[sector])
            # the only allowed blocks by charge rule for legs [p, p*, dummy] the sectors [1, 1, 2],
            # i.e. acting on the spin 1/2 doublet [up, down].
            # This means that the same construction as for the SpinHalfSite works here too.
            Svec_invariant_part = SymmetricTensor.from_block_func(
                self.backend.ones_block, backend=self.backend, legs=[leg, leg.dual, dummy_leg],
                labels=['p', 'p*', '!']
            )
            self.add_charged_operator('Svec', ChargedTensor(Svec_invariant_part), hc='Svec')
        else:
            # TODO build Svec with 3-dim dummy leg
            JWu = np.array([1., -1., 1., -1.])
            self.add_symmetric_operator('JWu', JWu)
            self.add_symmetric_operator('JWd', [1., 1., -1., -1.])
            self.add_symmetric_operator('Nu', [0., 1., 0., 1.])
            self.add_symmetric_operator('Nd', [0., 0., 1., 1.])
            self.add_symmetric_operator('Sz', [0., .5, -.5, 0.])
            Cu = np.zeros((4, 4), dtype=float)
            Cu[0, 1] = Cu[2, 3] = 1.  # up -> empty , full -> down
            self.add_any_operator('Cu', Cu, need_JW=True, hc=False)
            self.add_any_operator('Cdu', np.transpose(Cu), need_JW=True, hc='Cu')
            # For spin-down annihilation operator: include a Jordan-Wigner string JWu
            # this ensures that Cdu.Cd = - Cd.Cdu
            # c.f. the chapter on the Jordan-Wigner trafo in the userguide
            Cd_noJW = np.zeros((4, 4), dtype=float)
            Cd_noJW[0, 2] = Cd_noJW[1, 3] = 1.
            Cd = JWu[:, None] * Cd_noJW
            self.add_any_operator('Cd', Cd, need_JW=True, hc=False)
            self.add_any_operator('Cdd', np.transpose(Cd), need_JW=True, hc='Cd')
            Sp = self.add_any_operator('Sp', np.dot(np.transpose(Cu), Cd), hc=False)
            Sm = self.add_any_operator('Sm', np.dot(np.transpose(Cd), Cu), hc='Sp')
        # operators : Sx, Sy
        if conserve_S == 'Sz':
            pass  # TODO build with two-dim dummy leg
        elif conserve_S in ['parity', 'None']:
            self.add_any_operator('Sx', .5 * (Sp + Sm), hc='Sx')
            self.add_any_operator('Sy', -.5j * (Sp - Sm), hc='Sy')

    def __repr__(self):
        """Debug representation of self."""
        return f'SpinHalfFermionSite({self.conserve_N}, {self.conserve_S}, {self.filling})'


class SpinHalfHoleSite(Site):
    r"""A :class:`SpinHalfFermionSite` but restricted to empty or singly occupied sites.

    Local states are:
         ``empty``  (vacuum),
         ``up``     (one spin-up electron),
         ``down``   (one spin-down electron)

    .. warning ::
        Using the Jordan-Wigner string (``JW``) in the correct way is crucial to get correct
        results, otherwise you just describe hardcore bosons!

    The possible symmetries factorize into the occupation number and spin sectors

    ==============  =====================  =====================  ==================================
    `conserve_N`    symmetry               sectors of basis       meaning of sector label
    ==============  =====================  =====================  ==================================
    ``'N'``         ``U1Symmetry``         ``[0, 1, 1]``          total number ``N`` of fermions
    ``'parity'``    ``ZNSymmetry(N=2)``    ``[0, 1, 1]``          parity ``N % 2`` of fermion number
    ``'None'``      --                     ``[0, 0, 0]``          --
    ==============  =====================  =====================  ==================================

    ==============  =====================  =====================  ==============================
    `conserve_S`    symmetry               sectors of basis       meaning of sector label
    ==============  =====================  =====================  ==============================
    ``'Stot'``      ``SU2Symmetry``        ``[0, 1, 1]``          total spin : ``2 * S``
    ``'Sz'``        ``U1Symmetry``         ``[0, 1, -1]``         magnetization: ``2 * Sz``
    ``'parity'``    ``ZNSymmetry(N=4)``    ``[0, 1, 3]``          ``(2 * Sz) % 4``
    ``'None'``      --                     ``[0, 0, 0]``          --
    ==============  =====================  =====================  ==============================

    The local operators are the same as for the :class:`SpinHalfFermionSite`, namely
    ``Id, JW, JWu, JWd, Cu, Cdu, Cd, Cdd, Nu, Nd, Ntot, dN, Sz, Sp, Sm, Sx, Sy``.
    The definitions and availability of operators in :class:`SpinHalfFermionSite` apply here
    verbatim, see its docstring. The only difference is that ``NdNu`` is excluded since it vanishes
    when the double occupied state is excluded.

    Parameters
    ----------
    cons_N : ``'N' | 'parity' | 'None'``
        Whether particle number is conserved, c.f. table above.
    cons_Sz : ``'Sz' | 'parity' | 'None'``
        Whether spin is conserved, c.f. table above.
    filling : float
        Average filling. Used to define ``dN``.
    backend : :class:`~tenpy.linalg.backends.Backend`, optional
        The backend used to create the operators.
    """

    def __init__(self, conserve_N: str = 'N', conserve_S: str = 'Sz', filling: float = 1.,
                 backend: TensorBackend = None):
        # parse conserve_N
        if conserve_N == 'N':
            sectors_N = np.array([0, 1, 1])
            sym_N = U1Symmetry('N')
        elif conserve_N == 'parity':
            sectors_N = np.array([0, 1, 1])
            sym_N = ZNSymmetry(2, 'parity_N')
        elif conserve_N == 'None':
            sectors_N = None
            sym_N = None
        else:
            raise ValueError(f'invalid `conserve_N`: {conserve_N}')
        # parse conserve_S
        if conserve_S == 'Stot':
            sectors_S = np.array([0, 1, 1])
            sym_S = SU2Symmetry('Stot')
            raise NotImplementedError  # TODO, need to special case the operator construction too..
        elif conserve_S == 'Sz':
            sectors_S = np.array([0, 1, -1])
            sym_S = U1Symmetry('2*Sz')
        elif conserve_S == 'parity':
            sectors_S = np.array([0, 1, 3])
            sym_S = ZNSymmetry(4, 'parity_Sz')
        elif conserve_S == 'None':
            sectors_S = None
            sym_S = None
        else:
            raise ValueError(f'invalid `conserve_S`: {conserve_S}')
        # make leg
        if sym_N is None and sym_S is None:
            leg = ElementarySpace.from_trivial_sector(3)
        elif sym_N is None:
            leg = ElementarySpace.from_basis(sym_S, sectors_S[:, None])
        elif sym_S is None:
            leg = ElementarySpace.from_basis(sym_N, sectors_N[:, None])
        else:
            leg = ElementarySpace.from_basis(sym_N * sym_S, np.stack([sectors_N, sectors_S], axis=1))
        # initialize Site
        self.conserve_N = conserve_N
        self.conserve_S = conserve_S
        self.filling = filling
        Site.__init__(self, leg=leg, backend=backend, state_labels=['empty', 'up', 'down'],
                      JW=[1., -1., -1.])
        # operators : Ntot, dN
        Ntot = self.add_symmetric_operator('Ntot', [0., 1., 1.])
        self.add_symmetric_operator('dN', Ntot - filling * self.Id)
        # operators : Svec, JWu, JWd, Nu, Nd, Sz, Cu, Cdu, Cd, Cdd, Sp, Sm
        if conserve_S == 'Stot':
            sector = [2]  # spin 1
            if sym_N is not None:
                sector.append(0)
            dummy_leg = ElementarySpace(leg.symmetry, sectors=[sector])
            # the only allowed blocks by charge rule for legs [p, p*, dummy] the sectors [1, 1, 2],
            # i.e. acting on the spin 1/2 doublet [up, down].
            # This means that the same construction as for the SpinHalfSite works here too.
            Svec_inv = SymmetricTensor.from_block_func(
                self.backend.ones_block, backend=self.backend, legs=[leg, leg.dual, dummy_leg],
                labels=['p', 'p*', '!']
            )
            self.add_charged_operator('Svec', ChargedTensor(Svec_inv), hc='Svec')
        else:
            # TODO build Svec with 3-dim dummy leg
            JWu = np.array([1., -1., 1.])
            self.add_symmetric_operator('JWu', JWu)
            self.add_symmetric_operator('JWd', [1., 1., -1.])
            self.add_symmetric_operator('Nu', [0., 1., 0.])
            self.add_symmetric_operator('Nd', [0., 0., 1.])
            self.add_symmetric_operator('Sz', [0., .5, -.5])
            Cu = np.zeros((3, 3), dtype=float)
            Cu[0, 1] = 1.  # up -> empty
            self.add_any_operator('Cu', Cu, need_JW=True, hc=False)
            self.add_any_operator('Cdu', np.transpose(Cu), need_JW=True, hc='Cu')
            # For spin-down annihilation operator: include a Jordan-Wigner string JWu
            # this ensures that Cdu.Cd = - Cd.Cdu
            # c.f. the chapter on the Jordan-Wigner trafo in the userguide
            Cd_noJW = np.zeros((3, 3), dtype=float)
            Cd_noJW[0, 2] = 1.
            Cd = JWu[:, None] * Cd_noJW
            self.add_any_operator('Cd', Cd, need_JW=True, hc=False)
            self.add_any_operator('Cdd', np.transpose(Cd), need_JW=True, hc='Cd')
            Sp = self.add_any_operator('Sp', np.dot(np.transpose(Cu), Cd), hc=False)
            Sm = self.add_any_operator('Sm', np.dot(np.transpose(Cd), Cu), hc='Sp')
        # operators : Sx, Sy
        if conserve_S == 'Sz':
            pass  # TODO build with two-dim dummy leg
        elif conserve_S in ['parity', 'None']:
            self.add_any_operator('Sx', .5 * (Sp + Sm), hc='Sx')
            self.add_any_operator('Sy', -.5j * (Sp - Sm), hc='Sy')

    def __repr__(self):
        """Debug representation of self."""
        return f'SpinHalfHoleSite({self.cons_N}, {self.cons_Sz}, {self.filling})'


class BosonSite(Site):
    r"""A site with a single species of spin-less bosons.

    The "true" local Hilbert space is infinite dimensional and we need to restrict to a maximal
    occupation number `Nmax` for simulations.
    Local states are ``vac, 1, 2, ... , Nmax``.

    ==============  =====================  =======================  ================================
    `conserve`      symmetry               sectors of basis         meaning of sector label
    ==============  =====================  =======================  ================================
    ``'N'``         ``U1Symmetry``         ``[0, 1, ..., Nmax]``    total number ``N`` of bosons
    ``'parity'``    ``ZNSymmetry(N=2)``    ``[0, 1, 0, 1, ...]``    boson number parity ``N % 2``
    ``'None'``      --                     ``[0, 0, 0, 0, ...]``    --
    ==============  =====================  =======================  ================================

    The symmetric operators are (columns indicate `conserve`)::

    ============  =======================================  ========  ========  ========
    operator      description                              N         parity    None
    ============  =======================================  ========  ========  ========
    ``Id, JW``    Identity :math:`\mathbb{1}`              diag      diag      diag
    ``B``         Annihilation operator :math:`b`          --        --        tens
    ``Bd``        Creation operator :math:`b^\dagger`      --        --        tens
    ``N``         Number operator :math:`n= b^\dagger b`   diag      diag      diag
    ``NN``        :math:`n^2`                              diag      diag      diag
    ``dN``        :math:`\delta n := n - filling`          diag      diag      diag
    ``dNdN``      :math:`(\delta n)^2`                     diag      diag      diag
    ``P``         Parity :math:`(-1)^n`                    diag      diag      diag
    ============  =======================================  ========  ========  ========

    The charged operators are (columns indicate `conserve`, entries are dummy leg dimensions)::

    ============  =======================================  ========  ========  ========
    operator      description                              N         parity    None
    ============  =======================================  ========  ========  ========
    ``B, Bd``     As above                                 1         1         1
    ============  =======================================  ========  ========  ========

    Parameters
    ----------
    Nmax : int
        Cutoff defining the maximum number of bosons per site.
        The default ``Nmax=1`` describes hard-core bosons.
    conserve : 'N' | 'parity' | 'None'
        Defines what is conserved, see table above.
    filling : float
        Average filling. Used to define ``dN``.
    backend : :class:`~tenpy.linalg.backends.Backend`, optional
        The backend used to create the operators.
    """

    def __init__(self, Nmax: int = 1, conserve: str = 'N', filling: float = 0.,
                 backend: TensorBackend = None):
        assert Nmax > 0
        d = Nmax + 1
        N = np.arange(d)
        # build leg
        if conserve == 'N':
            leg = ElementarySpace.from_sectors(U1Symmetry('N'), N[:, None])
        elif conserve == 'parity':
            leg = ElementarySpace.from_sectors(ZNSymmetry(2, 'parity_N'), N[:, None] % 2)
        elif conserve == 'None':
            leg = ElementarySpace.from_trivial_sector(d)
        else:
            raise ValueError(f'invalid `conserve`: {conserve}')
        # initialize Site
        self.Nmax = Nmax
        self.conserve = conserve
        self.filling = filling
        labels = [str(n) for n in range(d)]
        Site.__init__(self, leg, backend=backend, state_labels=labels)
        self.state_labels['vac'] = self.state_labels['0']  # alias
        # operators
        self.add_symmetric_operator('N', N)
        self.add_symmetric_operator('NN', N ** 2)
        self.add_symmetric_operator('dN', N - filling)
        self.add_symmetric_operator('dNdN', (N - filling) ** 2)
        self.add_symmetric_operator('P', 1. - 2. * (N % 2))
        B = np.zeros([d, d], dtype=float)
        for n in range(1, d):
            B[n - 1, n] = np.sqrt(n)
        B = self.add_any_operator('B', B, hc=False)
        self.add_any_operator('Bd', B.hconj(), hc='B')

    def __repr__(self):
        """Debug representation of self."""
        return f'BosonSite({self.Nmax}, {self.conserve}, {self.filling})'


def spin_half_species(SpeciesSite: type[Site], conserve_N: str, conserve_S: str, **kwargs):
    """Initialize two sites of a spinless species to form a spin-1/2 doublet.

    You can use this directly in the :meth:`tenpy.models.model.CouplingMPOModel.init_sites`,
    e.g., as in the :meth:`tenpy.models.hubbard.FermiHubbardModel2.init_sites`::

        cons_N = model_params.get('cons_N', 'N')
        cons_Sz = model_params.get('cons_Sz', 'Sz')
        return spin_half_species(FermionSite, cons_N=cons_N, cons_Sz=cons_Sz)

    Parameters
    ----------
    SpeciesSite : :class:`Site` | str
        The (name of the) site class (the class itself, not an instance!) for the species;
        usually just :class:`FermionSite`.
    conserve_N : 'N' | 'parity' | 'None'
        Whether to conserve the (parity of the) total particle number ``N_up + N_down``.
    conserve_S :  'Sz' | 'parity' | 'None'
        Whether to conserve the (parity of the) total Sz spin ``N_up - N_down``.
        Using separate sites for spin up and down excludes ``'Stot'`` conservation.
        Use :class:`SpinHalfFermionSite` instead.
    **kwargs
        Keyword arguments used when initializing `SpeciesSite`.

    Returns
    -------
    sites : list of `SpeciesSite`
        Two instance of the site, one for spin up and one for down.
    species_names : list of str
        Always ``['up', 'down']``. Included such that a ``return spin_half_species(...)``
        in :meth:`~tenpy.models.model.CouplingMPOModel.init_sites` triggers the use of the
        :class:`~tenpy.models.lattice.MultiSpeciesLattice`.
    """
    SpeciesSite = find_subclass(Site, SpeciesSite)
    conserve = conserve_N if conserve_S == 'None' else 'N'
    up_site = SpeciesSite(conserve=conserve, **kwargs)
    down_site = SpeciesSite(conserve=conserve, **kwargs)

    if conserve_N == 'N':
        sym_N = U1Symmetry('N')
    elif conserve_N == 'parity':
        sym_N = ZNSymmetry(2, 'parity_N')
    elif conserve_N == 'None':
        sym_N = None
    else:
        raise ValueError(f'invalid `conserve_N`: {conserve_N}')
    
    if conserve_S == 'Sz':
        sym_S = U1Symmetry('2*Sz')
    elif conserve_S == 'parity':
        sym_S = ZNSymmetry(4, 'parity_Sz')
    elif conserve_S == 'None':
        sym_S = None
    else:
        raise ValueError(f'invalid `conserve_S`: {conserve_S}')
    
    if sym_N is None and sym_S is None:
        sym = no_symmetry
        symmetry_combine = 'drop'
    elif sym_N is None:
        sym = sym_S

        def symmetry_combine(site_idx: int, N_sectors: SectorArray) -> SectorArray:
            sign = 1 - 2 * site_idx  # +1 for up, -1 for down
            S_sectors = sign * N_sectors  # 2 * Sz = N_up - N_down
            if conserve_S == 'parity':
                S_sectors = S_sectors % 4
            return S_sectors
        
    elif sym_S is None:
        sym = sym_N
        assert up_site.symmetry == down_site.symmetry
        symmetry_combine = 'by_name'
    else:
        sym = sym_N * sym_S

        def symmetry_combine(site_idx: int, N_sectors: SectorArray) -> SectorArray:
            sign = 1 - 2 * site_idx  # +1 for up, -1 for down
            S_sectors = sign * N_sectors  # 2 * Sz = N_up - N_down
            if conserve_S == 'parity':
                S_sectors = S_sectors % 4
            if conserve_N == 'parity':
                N_sectors = N_sectors % 2
            return np.concatenate([N_sectors, S_sectors], axis=1)

    set_common_symmetry([up_site, down_site], symmetry_combine=symmetry_combine, new_symmetry=sym)
    return [up_site, down_site], ['up', 'down']


class ClockSite(Site):
    r"""Quantum clock site.

    There are ``q`` local states, with labels ``['0', '1', ..., str(q-1)]``.
    Special aliases are ``up`` (0), and if q is even ``down`` (q / 2).

    ============  =====================  ==================  =================================
    `conserve`    symmetry               sectors of basis    meaning of sector label
    ============  =====================  ==================  =================================
    ``'Z'``       ``ZNSymmetry(N=q)``    ``range(q)``        sector ``n`` has ``Z = w ** n``
    ``'None'``    ``NoSymmetry``         ``[0, ...]``        --
    ============  =====================  ==================  =================================

    Local operators are the clock operators ``Z = diag([w ** 0, w ** 1, ..., w ** (q - 1)])``
    with ``w = exp(2.j * pi / q)`` and ``X = eye(q, k=1) + eye(q, k=1-q)``, which are not hermitian.
    They are generalizations of the pauli operators and fulfill the clock algebra
    :math:`X Z = \mathtt{w} Z X` and :math:`X^q = \mathbb{1} = Z^q`.
    
    The symmetric operators are (columns indicate `conserve`)::
    
    ============  =====================================  ========  ========
    operator      description                            Z         None
    ============  =====================================  ========  ========
    ``Id, JW``    Identity :math:`\mathbb{1}`            diag      diag
    ``Z, Zhc``    Clock operator Z & its conjugate       diag      diag
    ``Zphc``      "Real part" :math:`Z + Z^\dagger`      diag      diag
    ``X, Xhc``    Clock operator X & its conjugate       --        tens
    ``Xphc``      "Real part" :math:`X + X^\dagger`      --        tens
    ============  =====================================  ========  ========

    The charged operators are (columns indicate `conserve`, entries are dummy leg dimensions)::
    
    ============  =====================================  ========  ========
    operator      description                            Z         None
    ============  =====================================  ========  ========
    ``X, Xhc``    Clock operator X & its conjugate       1         1
    ``Xphc``      "Real part" :math:`X + X^\dagger`      2         1
    ============  =====================================  ========  ========

    Parameters
    ----------
    q : int
        Number of states per site
    conserve : 'Z' | 'None'
        Defines what is conserved, see table above.
    backend : :class:`~tenpy.linalg.backends.Backend`, optional
        The backend used to create the operators.
    """
    def __init__(self, q: int, conserve: str = 'Z', backend: TensorBackend = None):
        if not (isinstance(q, int) and q > 1):
            raise ValueError(f'invalid q: {q}')
        # make leg
        if conserve == 'Z':
            leg = ElementarySpace.from_basis(ZNSymmetry(q, 'clock_phase'), np.arange(q)[:, None])
        elif conserve == 'None':
            leg = ElementarySpace.from_trivial_sector(q)
        else:
            raise ValueError(f'invalid `conserve`: {conserve}')
        # initialize Site
        names = [str(m) for m in range(q)]
        self.q = q
        self.conserve = conserve
        Site.__init__(self, leg=leg, backend=backend, state_labels=names)
        self.state_labels['up'] = self.state_labels['0']
        if q % 2 == 0:
            self.state_labels['down'] = self.state_labels[str(q // 2)]
        # operators
        self.add_symmetric_operator('Z', np.exp(2.j * np.pi * np.arange(q, dtype=np.complex128) / q))
        self.add_symmetric_operator('Zhc', np.exp(-2.j * np.pi * np.arange(q, dtype=np.complex128) / q))
        self.add_symmetric_operator('Zphc', 2. * np.cos(2. * np.pi * np.arange(q, dtype=np.complex128) / q))
        X = np.eye(q, k=1) + np.eye(q, k=1-q)
        if q == 2:
            # for q=2 we have ising spins and X is hermitian
            hc_X = 'X'
            hc_Xhc = 'Xhc'
        else:
            hc_X = False
            hc_Xhc = 'X'
        # if q==2, X is hermitian
        self.add_any_operator('X', X, hc=hc_X)
        self.add_any_operator('Xhc', X.conj().transpose(), hc=hc_Xhc)
        if conserve == 'Z':
            pass  # TODO add Xphc with two-dim dummy leg
        else:
            self.add_any_operator('Xphc', X + X.conj().transpose(), hc='Xphc')

    def __repr__(self):
        return f'ClockSite(q={self.q}, conserve={self.conserve})'
