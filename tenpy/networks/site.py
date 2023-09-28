"""Defines a class describing the local physical Hilbert space.

The :class:`Site` is the prototype, read it's docstring.

"""
# Copyright 2018-2023 TeNPy Developers, GNU GPLv3
from __future__ import annotations
import numpy as np
import itertools
import copy
from functools import partial, reduce

from ..linalg.tensors import (AbstractTensor, Tensor, SymmetricTensor, ChargedTensor,
                              DiagonalTensor, almost_equal, tensor_from_block, angle, real_if_close,
                              get_same_backend)
from ..linalg.backends import AbstractBackend, get_backend, Block
from ..linalg.matrix_operations import exp
from ..linalg.groups import (ProductSymmetry, Symmetry, SU2Symmetry, U1Symmetry, ZNSymmetry,
                             no_symmetry, SectorArray)
from ..linalg.spaces import VectorSpace, ProductSpace
from ..linalg.misc import make_stride
from ..tools.misc import inverse_permutation, find_subclass
from ..tools.hdf5_io import Hdf5Exportable

__all__ = [
    'Site',
    'GroupedSite',
    'group_sites',
    'set_common_symmetry',
    'SpinHalfSite',
    'SpinSite',
    'FermionSite',
    'SpinHalfFermionSite',
    'SpinHalfHoleSite',
    'BosonSite',
    'ClockSite',
    'spin_half_species',
]


class Site(Hdf5Exportable):
    """Collects information about a single local site of a lattice.

    This class defines what the local basis states are via the :attr:`leg`, which defines the order
    of basis states (:attr:`VectorSpace.basis_perm`) and how the symmetry acts on them (by assigning
    them to :attr:`VectorSpace.sectors`).
    A `Site` instance therefore already determines which symmetry is explicitly used.
    Using the same "kind" of phyiscal site (typically a parituclar subclass of `Site`),
    but using different symmetries requires *different* `Site` instances.
    
    Moreover, it stores the local operators. We distinguish two types of operators on a site:

        i.  The *symmetric* operators. These are operators that preserve the symmetry of the site,
            i.e. they can be written as a `Tensor` (as opposed to the general operators below, which
            can only be written as `ChargedTensor`s). Applying them to a state preserves the
            conserved charges.

        ii. The *general* operators. These are operators that do *not* preserve the symmetry, i.e.
            applying them to a state in a given charge sector yields a state in a *different* charge
            sector. Expectation values of a single one of these operators vanish by symmetry.
            They are used as building blocks for non-local operators, e.g. Hamiltonians
            or e.g. :math:`S_i^{+} S_j^{-}` whose expectation value forms a correlation function.

    .. _OperatorCategories:

    In the docstrings of :class:`Site` subclasses, we include tables which assign the local
    operators to one of the following sub-categories:

    ==========  ====================================================================================
    category    description
    ==========  ====================================================================================
    diag        A symmetric operator which is also diagonal, i.e. it is a :class:`DiagonalTensor`
    sym         A non-diagonal symmetric operator, i.e. a :class:`Tensor`
    gen         A general operator, i.e. a :class:`ChargedTensor` with a one-dimensional dummy-leg
    gen(n)      A general operator with a dummy_leg of dimension ``n > 1``.
                Always such that contracting two copies over the dummy leg gives the correct
                two-body operator, as e.g. :math:`S^x_{i} S^x_{j}` for :class:`SpinSite`.
    --          Operator not available
    ==========  ====================================================================================

    All sites define the operators ``'Id'``, the identity, and ``'JW'``, the local contribution to
    Jordan-Wigner strings, both of which are symmetric and diagonal.

    Parameters
    ----------
    leg : :class:`~tenpy.linalg.spaces.VectorSpace`
        The Hilbert space associated with the site. Defines the basis and the symmetry.
    backend : :class:`~tenpy.linalg.backends.AbstractBackend`, optional
        The backend used to create the identity operator.
    state_labels : None | list of str
        Optionally, a label for each local basis state.
    ops : {str: tensor-like}
        Pairs ``name: op`` to be given to :meth:`add_op`.
        The identity operator ``name='Id'`` should not be included, it is added automatically.
        Sites with fermionic degrees of freedom should define ``'JW'``, the local contribution to
        Jordan-Wigner strings. Otherwise (if ``'JW' not in ops``), the identity is added as `'JW'``.

    Attributes
    ----------
    leg : :class:`~tenpy.linalg.spaces.VectorSpace`
        The Hilbert space associated with the site. Defines the basis and the symmetry.
    state_labels : {str: int}
        Labels for the local basis states. Maps from label to index of the state in the basis.
    symmetric_ops_names : set of str
        Names of the symmetric operators that are defined on this site. If ``'op'`` is in
        `symmetric_ops_names`, then ``self.op`` exists and is a `SymmetricTensor`.
    general_ops_names : set of str
        Names of the general operators that are defined on this site. If ``'op'`` is in
        `symmetric_ops_names`, then ``self.op`` exists and is a `ChargedTensor`.
    need_JW_string : set of str
        Labels of those operators (symmetric_ops *or* general_ops) that need a Jordan-Wigner string.
        Used in :meth:`op_needs_JW` to determine whether an operator anticommutes or commutes
        with operators on other sites.
    *symmetric_ops : :class:`SymmetricTensor`
        Onsite operators are added directly as attributes to self.
        For example after ``self.add_op('Sz', Sz)`` you can use ``self.Sz`` for the `Sz` operator.
        All onsite operators have labels ``'p', 'p*'``.
    *general_ops : :class:`ChargedTensor`
        General (i.e. not symmetric) operators as attributes. By convention, we store ChargedTensors
        with a ket-like dummy leg.
    JW_exponent : :class:`DiagonalTensor`
        Exponents of the ``'JW'`` operator such that ``symmetric_ops['JW']`` is equivalent to
        ``exp(1.j * pi * JW_exponent)``.
    hc_ops : {str: str}
        Mapping from operator names (keys of `symmetric_ops` or `general_ops`) to the names of
        their hermitian conjugates. Use :meth:`get_hc_op_name` to obtain entries.
    """
    def __init__(self, leg: VectorSpace, backend: AbstractBackend = None,
                 state_labels: list[str] = None,
                 **site_ops: SymmetricTensor | ChargedTensor | Block):
        self.leg = leg
        self.state_labels = {}
        if state_labels is not None:
            for i, l in enumerate(state_labels):
                if l is not None:
                    self.state_labels[str(l)] = i
        self.symmetric_ops_names = set()
        self.general_ops_names = set()
        self.need_JW_string = {'JW'}
        self.hc_ops = {}
        self.JW_exponent = None  # set by self.add_op('JW')
        Id = DiagonalTensor.eye(leg, backend=backend, labels=['p', 'p*'])
        self.add_op('Id', Id, hc='Id')
        for name, op in site_ops.items():
            self.add_op(name, op)
        if 'JW' not in site_ops:
            # assume all states have even fermionic parity and include trivial `JW`.
            # allows e.g. combinations of bosonic and fermionic sites in an MPS.
            self.add_op('JW', Id, hc='JW')
        self.test_sanity()

    def test_sanity(self):
        for lab, ind in self.state_labels.items():
            if not isinstance(lab, str):
                raise ValueError("wrong type of state label")
            if not 0 <= ind < self.dim:
                raise ValueError("index of state label out of bounds")
        if any(name in self.general_ops for name in self.symmetric_ops):
            raise ValueError('Duplicate names')
        for name in self.symmetric_ops_names:
            op = getattr(self, name)
            if op.num_legs != 2:
                raise ValueError(f'op "{name}" has wrong number of legs')
            if not op.labels_are('p', 'p*'):
                raise ValueError(f'op "{name}" has wrong labels')
            if op.get_legs(['p', 'p*']) != [self.leg, self.leg.dual]:
                raise ValueError(f'op "{name}" has incompatible legs')
            op.test_sanity()
        for name in self.general_ops_names:
            op = getattr(self, name)
            assert not op.dummy_leg.is_dual
            if op.num_legs != 2:
                raise ValueError(f'op "{name}" has wrong number of legs')
            if not op.labels_are('p', 'p*'):
                raise ValueError(f'op "{name}" has wrong labels')
            if op.get_legs(['p', 'p*']) != [self.leg, self.leg.dual]:
                raise ValueError(f'op "{name}" has incompatible legs')
            op.test_sanity()
        for name in self.need_JW_string:
            assert name in self.symmetric_ops_names or name in self.general_ops_names
        assert almost_equal(exp(1.j * np.pi * self.JW_exponent), self.JW)
        for name1, name2 in self.hc_ops.items():
            if name1 in self.symmetric_ops_names:
                assert name2 in self.symmetric_ops_names
            elif name1 in self.general_ops_names:
                assert name2 in self.general_ops_names
            else:
                raise ValueError('unknown name')
            op1 = getattr(self, name1)
            op2 = getattr(self, name2)
            assert almost_equal(op1.hconj(), op2, allow_different_types=True)

    @property
    def all_ops(self) -> dict:
        """Dictionary of all operators for iteration."""
        return dict((name, getattr(self, name)) for name in sorted(self.all_ops_names))

    @property
    def all_ops_names(self) -> set:
        """Set of all operator names, symmetric and general."""
        return self.symmetric_ops_names.union(self.general_ops_names)

    @property
    def dim(self):
        """Dimension of the local Hilbert space."""
        return self.leg.dim

    @property
    def general_ops(self) -> dict:
        """Dictionary of general operators for iteration."""
        return dict((name, getattr(self, name)) for name in sorted(self.general_ops_names))

    @property
    def symmetric_ops(self) -> dict:
        """Dictionary of symmetric operators for iteration."""
        return dict((name, getattr(self, name)) for name in sorted(self.symmetric_ops_names))

    @property
    def symmetry(self):
        """Symmetry of the local Hilbert space."""
        return self.leg.symmetry

    def add_op(self, name: str, op: AbstractTensor | Block, backend: AbstractBackend = None,
               need_JW: bool = False, hc: str | bool = None):
        """Add an on-site operator.

        Parameters
        ----------
        name : str
            A valid python variable name, used to label the operator.
            The name under which `op` is added as attribute to self.
            Names of all operators must be unique per site instance.
        op : :class:`SymmetricTensor` | :class:`ChargedTensor` | Block
            A single tensor describing the operator. Legs must be the :attr:`leg` of this site with
            label ``'p'`` and its dual with label ``'p*'``.
            If not an :class:`AbstractTensor`, it is converted via :func:`tensor_from_block`.
        backend : :class:`AbstractBackend`, optional
            The backend to use if `op` is not a tensor. Is ignored if `op` is a tensor already.
            Defaults to using the backend of the identiy operator ``self.Id``.
        need_JW : bool
            If this operator needs a Jordan-Wigner string.
            If so, add `name` to :attr:`need_JW_string`.
        hc : None | False | str
            The name for the hermitian conjugate operator, to be used for :attr:`hc_ops`.
            By default (``None``), try to auto-determine it.
            If ``False``, disable adding antries to :attr:`hc_ops`.
        """
        name = str(name)
        if not name.isidentifier():
            raise ValueError(f'Not a valid identified: {name}')
        if name in self.symmetric_ops or name in self.general_ops:
            raise ValueError(f'operator with name "{name}" already exists')
        if hasattr(self, name):
            raise ValueError(f'Site already has an attribute with name {name}')
        if not isinstance(op, AbstractTensor):
            if backend is None:
                backend = self.Id.backend
            op = tensor_from_block(op, legs=[self.leg, self.leg.dual], backend=backend,
                                   labels=['p', 'p*'])
            if isinstance(op, Tensor):
                # convert to DiagonalTensor if possible
                try:
                    op = DiagonalTensor.from_tensor(op, check_offdiagonal=True)
                except ValueError:
                    pass
        if op.num_legs != 2:
            raise ValueError('wrong number of legs')
        if not op.labels_are('p', 'p*'):
            raise ValueError('wrong labels')
        if op.get_legs(['p', 'p*']) != [self.leg, self.leg.dual]:
            raise ValueError('incompatible legs')
        op.test_sanity()
        if isinstance(op, SymmetricTensor):
            self.symmetric_ops_names.add(name)
        elif isinstance(op, ChargedTensor):
            if op.dummy_leg.is_dual:
                op = op.flip_dummy_leg_duality()
            self.general_ops_names.add(name)
        else:
            raise TypeError(f'Expected SymmetricTensor or ChargedTensor. Got {type(op)}.')
        setattr(self, name, op)
        if need_JW:
            self.need_JW_string.add(name)
        if hc is None:
            hc = self._auto_detect_hc(name, op)
        if hc:
            self.hc_ops[hc] = name
            self.hc_ops[name] = hc
        if name == 'JW':
            assert isinstance(op, DiagonalTensor)
            self.JW_exponent = angle(real_if_close(op)) / np.pi

    def _auto_detect_hc(self, name: str, op: AbstractTensor) -> str | None:
        """Automatically detect which (if any) of the existing operators is the hc of a new op

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
            for other_name, other_op in self.symmetric_ops.items():
                # allow different types, since we might have `Tensor`s and `DiagonalTensor`s
                if almost_equal(op_hc, other_op, allow_different_types=True):
                    return other_name
        elif isinstance(op, ChargedTensor):
            for other_name, other_op in self.general_ops.items():
                if almost_equal(op_hc, other_op):
                    return other_name
        else:
            raise TypeError
        return None

    def change_leg(self, new_leg: VectorSpace = None):
        """Change the :attr:`leg` of the site in-place.

        Assumes that the :attr:`state_labels` are still valid.

        Parameters
        ----------
        new_leg : :class:`VectorSpace` | None
            The new leg to be used. If ``None``, use trivial charges.
        """
        if new_leg is None:
            new_leg = VectorSpace.from_trivial_sector(dim=self.dim, symmetry=self.symmetry,
                                                      is_real=self.leg.is_real)
        self.leg = new_leg
        backend = self.Id.backend
        for names_set in [self.symmetric_ops_names, self.general_ops_names]:
            for name in names_set.copy():
                names_set.remove(name)
                op = getattr(self, name).to_dense_block(['p', 'p*'])
                delattr(self, name)
                self.add_op(name, op, backend=backend, need_JW=False, hc=False)  # need_JW and hc_ops are still set
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
        old_hc_name = self.hc_ops.get(old_name, None)
        is_symmetric = old_name in self.symmetric_ops_names
        op = getattr(self, old_name)
        need_JW = old_name in self.need_JW_string
        self.remove_op(old_name)
        setattr(self, new_name, op)
        if is_symmetric:
            self.symmetric_ops_names.add(new_name)
        else:
            self.general_ops_names.add(new_name)
        if need_JW:
            self.need_JW_string.add(new_name)
        if new_name == 'JW':
            assert isinstance(op, DiagonalTensor)
            self.JW_exponent = angle(real_if_close(op)) / np.pi
        if old_hc_name is not None:
            if old_hc_name == old_name:
                self.hc_ops[new_name] = new_name
            else:
                self.hc_ops[new_name] = old_hc_name
                self.hc_ops[old_hc_name] = new_name

    def remove_op(self, name: str):
        """Remove an added operator.

        Parameters
        ----------
        name : str
            The name of the operator to be removed.
        """
        hc_name = self.hc_ops.get(name, None)
        if hc_name is not None:
            del self.hc_ops[name]
            if hc_name != name:
                del self.hc_ops[hc_name]
        self.symmetric_ops_names.discard(name)
        self.general_ops_names.discar(name)
        delattr(self, name)
        self.need_JW_string.discard(name)

    def state_index(self, label: str | int) -> int:
        """Return index of a basis state from its label.

        Parameters
        ----------
        label : int | string
            eather the index directly or a label (string) set before.

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

    def get_op(self, name) -> AbstractTensor:
        if ' ' in name:
            raise NotImplementedError  # TODO redesign the operator mini language?
        return getattr(self, name)

    def get_hc_op_name(self, name: str) -> str:
        if ' ' in name:
            raise NotImplementedError  # TODO redesign the operator mini language?
        return self.hc_ops[name]
    
    def op_needs_JW(self, name: str) -> bool:
        if ' ' in name:
            raise NotImplementedError  # TODO redesign the operator mini language?
        return name in self.need_JW_string

    def valid_opname(self, name: str) -> bool:
        if ' ' in name:
            raise NotImplementedError  # TODO redesign the operator mini language?
        return name in self.all_ops_names
    
    def multiply_op_names(self, names: list[str]) -> str:
        raise NotImplementedError  # TODO redesign the operator mini language?

    def multiply_operators(self, operators: list[str | AbstractTensor]) -> AbstractTensor:
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

    TODO note about order of states and dense reps of the local operators

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

    Id: DiagonalTensor
    JW: DiagonalTensor

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
            res_symmetry = ProductSymmetry(all_symmetries)
            for i, site in enumerate(sites):
                # define the new leg to be in the trivial sector for all symmetries...
                independent_gradings = [
                    VectorSpace.from_trivial_sector(dim=site.dim, symmetry=s) for s in all_symmetries
                ]
                # ... except for "its own" symmetry_
                independent_gradings[i] = site.leg
                legs.append(VectorSpace.from_independent_symmetries(independent_gradings, res_symmetry))
        else:
            raise ValueError("Unknown option for `symmetry_combine`: " + repr(symmetry_combine))
        # change sites to have the new legs
        if symmetry_combine != 'same':
            # copy to avoid modifiyng the existing sites
            sites = [copy.copy(s).change_leg(l) for s, l in zip(sites, legs)]
        # eventhough Site.__init__ will also set self.leg, we need it earlier to use kroneckerproduct
        backend = get_same_backend(*(s.Id for s in sites))
        self.leg = leg = ProductSpace(legs, backend=backend)
        JW_all = self.kroneckerproduct([s.JW for s in sites])
        # initialize Site , will set labels and add ops below
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
                self.state_labels[state_label] = perm[prod_space_idx]  # TODO this the right way around?
        else:
            raise NotImplementedError  # TODO fusion is more than a permutation. labels like above make no sense.
        # add remaining operators
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
                self.add_op(name + labels[i], self.kroneckerproduct(ops), need_JW=need_JW, hc=hc)
            for name, op in site.general_ops.items():
                need_JW = name in site.need_JW_string
                hc = False if name not in site.hc_ops else site.hc_ops[name] + labels[i]
                ops = JW_Ids if need_JW else Ids
                ops[i] = op
                self.add_op(name + labels[i], self.kroneckerproduct(ops), need_JW=need_JW, hc=hc)
            Ids[i] = site.symmetric_ops['Id']
            JW_Ids[i] = site.symmetric_ops['JW']
   
    def kroneckerproduct(self, ops):
        r"""Return the Kronecker product :math:`op0 \otimes op1` of local operators.

        Parameters
        ----------
        ops : list of :class:`~tenpy.linalg.tensor.AbstractTensor`
            One operator (or operator name) on each of the ungrouped sites.
            Each operator should have labels ``['p', 'p*']``.

        Returns
        -------
        prod : :class:`~tenpy.linalg.tensor.AbstractTensor`
            Kronecker product :math:`ops[0] \otimes ops[1] \otimes \cdots`,
            with labels ``['p', 'p*']``.
        """
        if all(isinstance(op, DiagonalTensor) for op in ops):
            # TODO proper implementation? e.g. in tenpy.linalg.matrix_operations?
            backend = get_same_backend(*ops)
            # note that block_kron is associative, order does not matter
            diag = reduce(backend.block_kron, (op.diag_block for op in ops))
            return DiagonalTensor.from_diag_block(diag, self.leg, backend=backend, labels=['p', 'p*'])
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
                       # but adjusted for the ignored no_symmetrys
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
# The most common local sites.


class SpinHalfSite(Site):
    r"""Spin-1/2 site.

    Local states are ``up`` (0) and ``down`` (1).

    ==============  =====================  =============  =========================
    `conserve`      symmetry               sectors        meaning of sector label
    ==============  =====================  =============  =========================
    ``'Stot'``      ``SU2Symmetry``        ``[1]``        ``2 * S``
    ``'Sz'``        ``U1Symmetry``         ``[1, -1]``    ``2 * Sz``
    ``'parity'``    ``ZNSymmetry(N=2)``    ``[1, 0]``     ``(Sz + .5) % 2``
    ``'None'``      ``NoSymmetry``         ``[0, 0]``     --
    ==============  =====================  =============  =========================

    TODO include dipole symmetry here too?
    
    Local operators are the usual spin-1/2 operators, e.g. ``Sz = [[0.5, 0.], [0., -0.5]]``,
    ``Sx = 0.5 * Sigmax`` for the Pauli matrix `Sigmax`.
    The following table lists all local operators and their :ref:`categories <OperatorCategories>`.

    ====================  =================================  ========  ========  ========  ========
    operator              description                        Stot      Sz        parity    None
    ====================  =================================  ========  ========  ========  ========
    ``Id, JW``            Identity :math:`\mathbb{1}`        diag      diag      diag      diag
    ``Sz``                Spin component :math:`S^z`         --        diag      diag      diag
    ``Sx, Sy``            Spin components :math:`S^{x,y}`    --        gen(2)    gen(2)    sym
    ``Sp, Sm``            :math:`S^{\pm} = S^x \pm i S^y`    --        gen       gen       sym
    ``Svec``              The vector of spin operators.      gen       gen(3)    gen(3)    gen(3)
    ``Sigmaz``            Pauli matrix :math:`\sigma^{z}`    --        diag      diag      diag
    ``Sigmax, Sigmay``    Pauli matrices x & y               --        gen(2)    gen(2)    sym
    ====================  =================================  ========  ========  ========  ========

    Parameters
    ----------
    conserve : 'Stot' | 'Sz' | 'parity' | 'None'
        Defines what is conserved, see table above.
    backend : :class:`~tenpy.linalg.backends.AbstractBackend`, optional
        The backend used to create the operators.
    """
    def __init__(self, conserve: str = 'Sz', backend: AbstractBackend = None):
        # make leg
        if conserve == 'Stot':
            leg = VectorSpace(symmetry=SU2Symmetry('Stot'), sectors=[[1]])
        elif conserve == 'Sz':
            leg = VectorSpace.from_sectors(U1Symmetry('2*Sz'), [[1], [-1]])
        elif conserve == 'parity':
            leg = VectorSpace.from_sectors(ZNSymmetry(2, 'parity_Sz'), [[1], [0]])
        elif conserve == 'None':
            leg = VectorSpace.from_trivial_sector(2)
        else:
            raise ValueError(f'invalid `conserve`: {conserve}')
        if backend is None:
            backend = get_backend(symmetry=leg.symmetry)
        # operators: Svec, Sz, Sp, Sm
        if conserve == 'Stot':
            # TODO test this operator!, e.g. compare Svec @ Svec vs dense expect
            dummy_leg = VectorSpace(leg.symmetry, sectors=[[2]])
            Svec_inv = Tensor.from_block_func(
                backend.ones_block, backend=backend, legs=[leg, leg.dual, dummy_leg],
                labels=['p', 'p*', '!']
            )
            ops = dict(Svec=ChargedTensor(Svec_inv))
        else:
            # TODO also add Svec, with a 3-dim dummy leg
            Sz = DiagonalTensor.from_diag_block([.5, -.5], leg, backend=backend, labels=['p', 'p*'])
            ops = dict(Sz=Sz,
                       Sp=[[0., 1.], [0., 0.]],  # == Sx + i Sy
                       Sm=[[0., 0.], [1., 0.]])  # == Sx - i Sy
        # operators : Sx, Sy
        if conserve == 'Sz':
            pass  # TODO add ChargedTensor versions of Sx, Sy with length 2 dummy legs. Then also Sigmax below.
        if conserve in ['parity', 'None']:
            ops.update(Sx=[[0., 0.5], [0.5, 0.]], Sy=[[0., -0.5j], [+0.5j, 0.]])
        # initialize
        self.conserve = conserve
        Site.__init__(self, leg=leg, backend=backend, state_labels=['up', 'down'], **ops)
        # further alias for state labels
        self.state_labels['-0.5'] = self.state_labels['down']
        self.state_labels['0.5'] = self.state_labels['up']
        # Add Pauli matrices
        if conserve in ['parity', 'None']:
            self.add_op('Sigmax', 2. * self.Sx)
            self.add_op('Sigmay', 2. * self.Sy)
        self.add_op('Sigmaz', 2. * self.Sz)

    def __repr__(self):
        """Debug representation of self."""
        return f'SpinHalfSite({self.conserve})'


class SpinSite(Site):
    r"""General Spin S site.

    There are `2S+1` local states range from ``down`` (0)  to ``up`` (2S+1),
    corresponding to ``Sz=-S, -S+1, ..., S-1, S``.

    ==============  =====================  ==========================  ==========================
    `conserve`      symmetry               sectors                     meaning of sector label
    ==============  =====================  ==========================  ==========================
    ``'Stot'``      ``SU2Symmetry``        ``[2 * S] * S``             total spin ``2 * S``
    ``'Sz'``        ``U1Symmetry``         ``[-2 * S, ..., 2 * S]``    magnetization ``2 * Sz``
    ``'parity'``    ``ZNSymmetry(N=2)``    ``[0, 1, 0, 1, ...]``       parity ``(Sz + S) % 2``
    ``'None'``      ``NoSymmetry``         ``[0] * S``                 --
    ==============  =====================  ==========================  ==========================

    Local operators are the spin-S operators,
    e.g. ``Sz = [[0.5, 0.], [0., -0.5]]``, ``Sx = [[0., 0.5], [0.5, 0.]]`` for ``S=.5``.
    The following table lists all local operators and their :ref:`categories <OperatorCategories>`.

    ===============  =====================================  ========  ========  ========  ========
    operator         description                            Stot      Sz        parity    None
    ===============  =====================================  ========  ========  ========  ========
    ``Id, JW``       Identity :math:`\mathbb{1}`            sym       sym       sym       sym
    ``Sz``           Spin component :math:`S^{z}`           --        sym       sym       sym
    ``Sx, Sy``       Spin components :math:`S^{x,y}`        --        gen(2)    gen(2)    sym
    ``Sp, Sm``       :math:`S^{\pm} = S^{x} \pm i S^{y}`    --        gen       gen       sym
    ``Svec``         The vector of spin operators.          gen       gen(3)    gen(3)    gen(3)
    ===============  =====================================  ========  ========  ========  ========

    Parameters
    ----------
    S : {0.5, 1, 1.5, 2, ...}
        The 2S+1 states range from m = -S, -S+1, ... +S.
    conserve : 'Stot' | 'Sz' | 'parity' | 'None'
        Defines what is conserved, see table above.
    backend : :class:`~tenpy.linalg.backends.AbstractBackend`, optional
        The backend used to create the operators.
    """

    def __init__(self, S: float = 0.5, conserve: str = 'Sz', backend: AbstractBackend = None):
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
            leg = VectorSpace(symmetry=SU2Symmetry('Stot'), sectors=[[d - 1]])
        elif conserve == 'Sz':
            leg = VectorSpace.from_sectors(U1Symmetry('2*Sz'), two_Sz[:, None])
        elif conserve == 'parity':
            leg = VectorSpace.from_basis(ZNSymmetry(2, 'parity_Sz'), np.arange(d)[:, None] % 2)
        elif conserve == 'None':
            leg = VectorSpace.from_trivial_sector(d)
        else:
            raise ValueError(f'invalid `conserve`: {conserve}')
        if backend is None:
            backend = get_backend(symmetry=leg.symmetry)
        # operators: Svec, Sz, Sp, Sm
        if conserve == 'Stot':
            dummy_leg = VectorSpace(leg.symmetry, sectors=[[2]])
            Svec_inv = Tensor.from_block_func(
                backend.ones_block, backend=backend, legs=[leg, leg.dual, dummy_leg],
                labels=['p', 'p*', '!']
            )
            ops = dict(Svec=ChargedTensor(Svec_inv))
        else:
            # TODO also add Svec, with a 3-dim dummy leg
            Sz = DiagonalTensor.from_diag_block(two_Sz / 2., leg, backend=backend, labels=['p', 'p*'])
            ops=dict(Sz=Sz, Sp=Sp, Sm=Sm)
        # operators: Sx, Sy
        if conserve == 'Sz':
            pass # TODO add ChargedTensor versions of Sx, Sy with length 2 dummy legs.
        if conserve in ['parity', 'None']:
            # Sp = Sx + i Sy, Sm = Sx - i Sy
            ops.update(Sx=0.5 * (Sp + Sm), Sy=0.5j * (Sm - Sp))
            # Note: For S=1/2, Sy might look wrong compared to the Pauli matrix or SpinHalfSite.
            # Don't worry, I'm 99.99% sure it's correct (J. Hauschild). Mee too (J. Unfried).
            # The reason it looks wrong is simply that this class orders the states as ['down', 'up'],
            # while the usual spin-1/2 convention is ['up', 'down'], as you can also see if you look
            # at the Sz entries...
            # (The commutation relations are checked explicitly in `tests/test_site.py`)
        names = [str(i) for i in np.arange(-S, S + 1, 1.)]
        Site.__init__(self, leg=leg, backend=backend, state_labels=names, **ops)
        self.state_labels['down'] = self.state_labels[names[0]]
        self.state_labels['up'] = self.state_labels[names[-1]]

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

    ==============  =====================  ============  ==========================
    `conserve`      symmetry               sectors       meaning of sector label
    ==============  =====================  ============  ==========================
    ``'N'``         ``U1Symmetry``         ``[0, 1]``    number of fermions
    ``'parity'``    ``ZNSymmetry(N=2)``    ``[0, 1]``    parity of fermion number
    ``'None'``      ``NoSymmetry``         ``[0, 0]``    --
    ==============  =====================  ============  ==========================

    TODO how to control if tenpy should use JW-strings or tenpy.linalg.groups.FermionParity?

    Local operators are composed of the fermionic creation ``'Cd'`` and annihilation ``'C'``
    operators. Note that the local operators do *not* include the Jordan-Wigner strings that
    make them fulfill the proper commutation relations.
    The following table lists all local operators and their :ref:`categories <OperatorCategories>`.

    ===========  =======================================  ========  ========  ========
    operator     description                              N         parity    None
    ===========  =======================================  ========  ========  ========
    ``Id``       Identity :math:`\mathbb{1}`              diag      diag      diag
    ``JW``       Sign for the Jordan-Wigner string.       diag      diag      diag
    ``C``        Annihilation operator :math:`c`          gen       gen       sym
    ``Cd``       Creation operator :math:`c^\dagger`      gen       gen       sym
    ``N``        Number operator :math:`n= c^\dagger c`   diag      diag      diag
    ``dN``       :math:`\delta n := n - filling`          diag      diag      diag
    ``dNdN``     :math:`(\delta n)^2`                     diag      diag      diag
    ===========  =======================================  ========  ========  ========

    Parameters
    ----------
    conserve : 'N' | 'parity' | 'None'
        Defines what is conserved, see table above.
    filling : float
        Average filling. Used to define ``dN``.
    backend : :class:`~tenpy.linalg.backends.AbstractBackend`, optional
        The backend used to create the operators.
    """

    def __init__(self, conserve: str = 'N', filling: float = 0.5, backend: AbstractBackend = None):
        if conserve == 'N':
            leg = VectorSpace.from_sectors(U1Symmetry('N'), [[0], [1]])
        elif conserve == 'parity':
            leg = VectorSpace.from_sectors(ZNSymmetry(2, 'parity_N'), [[0], [1]])
        elif conserve == 'None':
            leg = VectorSpace.from_trivial_sector(2)
        else:
            raise ValueError(f'invalid `conserve`: {conserve}')
        if backend is None:
            backend = get_backend(symmetry=leg.symmetry)
        JW = DiagonalTensor.from_diag_block([1., -1.], leg, backend=backend, labels=['p', 'p*'])
        C = np.array([[0., 1.], [0., 0.]])
        Cd = np.array([[0., 0.], [1., 0.]])
        N_diag = np.array([0., 1.])
        N = DiagonalTensor.from_diag_block(N_diag, leg, backend=backend, labels=['p', 'p*'])
        dN = DiagonalTensor.from_diag_block(N_diag - filling, leg, backend=backend,
                                            labels=['p', 'p*'])
        dNdN = DiagonalTensor.from_diag_block((N_diag - filling) ** 2, leg, backend=backend,
                                              labels=['p', 'p*'])
        ops = dict(JW=JW, C=C, Cd=Cd, N=N, dN=dN, dNdN=dNdN)
        self.filling = filling
        self.conserve = conserve
        Site.__init__(self, leg, backend=backend, state_labels=['empty', 'full'], **ops)
        self.need_JW_string |= set(['C', 'Cd', 'JW'])  # pipe (``|``) for sets is union

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
    TODO         TODO include spin vector operators
    ===========  ===================================================================================

    For the :ref:`categories <OperatorCategories>` of operators, we distinguish the following
    different cases of what is conserved:
        (a)  ``conserve_S == 'Stot'`` and any `conserve_N`
        (b)  ``conserve_S == 'Sz'`` and any `conserve_N`
        (c)  ``conserve_S == 'parity'`` and any `conserve_N`
        (d)  ``conserve_S == 'None'`` and `conserve_N in ['N', 'parity]`
        (e)  ``conserve_S == 'None'`` and ``conserve_N == 'None'``

    ========================  ========  ========  ========  ========  =======
    operator                  (a)       (b)       (c)       (d)       (e)
    ========================  ========  ========  ========  ========  =======
    Id, JW, NuNd, Ntot, dN    diag      diag      diag      diag      diag
    JWu, Jwd, Nu, Nd, Sz      --        diag      diag      diag      diag
    Cu, Cdu, Cd, Cdd          --        gen       gen       gen       sym
    Sp, Sm                    --        gen       gen       sym       sym
    Sx, Sy                    --        gen(2)    gen       sym       sym
    ========================  ========  ========  ========  ========  =======

    Parameters
    ----------
    cons_N : ``'N' | 'parity' | 'None'``
        Whether particle number is conserved, c.f. table above.
    cons_Sz : ``'Sz' | 'parity' | 'None'``
        Whether spin is conserved, c.f. table above.
    filling : float
        Average filling. Used to define ``dN``.
    backend : :class:`~tenpy.linalg.backends.AbstractBackend`, optional
        The backend used to create the operators.
    """

    def __init__(self, conserve_N: str = 'N', conserve_S: str = 'Sz', filling: float = 1.,
                 backend: AbstractBackend = None):
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
        # build leg
        if sym_N is None and sym_S is None:
            leg = VectorSpace.from_trivial_sector(4)
        elif sym_N is None:
            leg = VectorSpace.from_basis(sym_S, sectors_S[:, None])
        elif sym_S is None:
            leg = VectorSpace.from_basis(sym_N, sectors_N[:, None])
        else:
            leg = VectorSpace.from_basis(sym_N * sym_S, np.stack([sectors_N, sectors_S], axis=1))
        if backend is None:
            backend = get_backend(symmetry=leg.symmetry)
        # diagonal operators : Nu, Nd, Ntot, dN, NuNd, JWu, JWd, JW, Sz
        Nu = np.array([0., 1., 0., 1.])
        Nd = np.array([0., 0., 1., 1.])
        Ntot = Nu + Nd
        dN = Ntot - filling
        NuNd = Nu * Nd
        JWu = 1. - 2 * Nu  # (-1)^Nu  == [1, -1, 1, -1]
        JWd = 1. - 2 * Nd  # (-1)^Nd  == [1, 1, -1, -1]
        JW = JWu * JWd  # (-1)^{Nu+Nd} == [1, -1, -1, 1]
        Sz = .5 * (Nu - Nd)
        ops = dict(JW=JW, NuNd=NuNd, Ntot=Ntot, dN=dN)
        if conserve_S != 'Stot':
            ops.update(JWu=JWu, JWd=JWd, Nu=Nu, Nd=Nd, Sz=Sz)
        ops = {name: DiagonalTensor.from_diag_numpy(op, leg, backend=backend, labels=['p', 'p*'])
               for name, op in ops.items()}
        # sym / gen[1] operators : Cu, Cdu, Cd, Cdd, Sp, Sm
        if conserve_S != 'Stot':
            Cu = np.zeros((4, 4), dtype=float)
            Cu[0, 1] = Cu[2, 3] = 1.  # up -> emtpy , full -> down
            Cdu = np.transpose(Cu)
            # For spin-down annihilation operator: include a Jordan-Wigner string JWu
            # this ensures that Cdu.Cd = - Cd.Cdu
            # c.f. the chapter on the Jordan-Wigner trafo in the userguide
            Cd_noJW = np.zeros((4, 4), dtype=float)
            Cd_noJW[0, 2] = Cd_noJW[1, 3] = 1.
            Cd = JWu[:, None] * Cd_noJW
            Cdd = np.transpose(Cd)
            Sp = np.dot(Cdu, Cd)
            Sm = np.dot(Cdd, Cu)
            ops.update(Cu=Cu, Cdu=Cdu, Cd=Cd, Cdd=Cdd, Sp=Sp, Sm=Sm)
        # build Sx, Sy
        if conserve_S in ['parity', 'None']:
            ops.update(Sx=.5 * (Sp + Sm), Sy=-.5j * (Sp - Sm))
        elif conserve_S == 'Sz':
            pass  # TODO build gen(2) version
        # build Svec
        if conserve_S == 'Stot':
            sector = [2]  # spin 1
            # sectors [0, 0, 2] do not fulfill the charge rule, so this operator only acts
            # non-trivially on the [up, down] spin doublet where the sectors are [1, 1, 2].
            # this is what the Svec operator should do.
            # this also means that the same construction as for the SpinHalfSite works here too.
            if sym_N is not None:
                sector.append(0)
            dummy_leg = VectorSpace(leg.symmetry, sectors=[sector])
            Svec_inv = Tensor.from_block_func(
                backend.ones_block, backend=backend, legs=[leg, leg.dual, dummy_leg],
                labels=['p', 'p*', '!']
            )
            ops.update(Svec=ChargedTensor(Svec_inv))
        else:
            pass  # TODO build gen(3) version
        # initialize
        self.conserve_N = conserve_N
        self.conserve_S = conserve_S
        self.filling = filling
        states = ['empty', 'up', 'down', 'full']
        Site.__init__(self, leg=leg, backend=backend, state_labels=states, **ops)
        # specify fermionic operators
        self.need_JW_string |= set(['Cu', 'Cdu', 'Cd', 'Cdd', 'JWu', 'JWd', 'JW'])

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
    The definitions and categories of operators in :class:`SpinHalfFermionSite` apply here verbatim,
    see its docstring. The only difference is that ``NdNu`` is excluded since it vanishes when the
    double occupied state is excluded.

    Parameters
    ----------
    cons_N : ``'N' | 'parity' | 'None'``
        Whether particle number is conserved, c.f. table above.
    cons_Sz : ``'Sz' | 'parity' | 'None'``
        Whether spin is conserved, c.f. table above.
    filling : float
        Average filling. Used to define ``dN``.
    backend : :class:`~tenpy.linalg.backends.AbstractBackend`, optional
        The backend used to create the operators.
    """

    def __init__(self, conserve_N: str = 'N', conserve_S: str = 'Sz', filling: float = 1.,
                 backend: AbstractBackend = None):
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
        # build leg
        if sym_N is None and sym_S is None:
            leg = VectorSpace.from_trivial_sector(3)
        elif sym_N is None:
            leg = VectorSpace.from_basis(sym_S, sectors_S[:, None])
        elif sym_S is None:
            leg = VectorSpace.from_basis(sym_N, sectors_N[:, None])
        else:
            leg = VectorSpace.from_basis(sym_N * sym_S, np.stack([sectors_N, sectors_S], axis=1))
        if backend is None:
            backend = get_backend(symmetry=leg.symmetry)
        # diagonal operators : Nu, Nd, Ntot, dN, NuNd, JWu, JWd, JW, Sz
        Nu = np.array([0., 1., 0.])
        Nd = np.array([0., 0., 1.])
        Ntot = Nu + Nd
        dN = Ntot - filling
        JWu = 1. - 2 * Nu  # (-1)^Nu  == [1, -1, 1]
        JWd = 1. - 2 * Nd  # (-1)^Nd  == [1, 1, -1]
        JW = JWu * JWd  # (-1)^{Nu+Nd} == [1, -1, -1]
        Sz = .5 * (Nu - Nd)
        ops = dict(JW=JW, Ntot=Ntot, dN=dN)
        if conserve_S != 'Stot':
            ops.update(JWu=JWu, JWd=JWd, Nu=Nu, Nd=Nd, Sz=Sz)
        ops = {name: DiagonalTensor.from_diag_numpy(op, leg, backend=backend, labels=['p', 'p*'])
               for name, op in ops.items()}
        # sym / gen[1] operators : Cu, Cdu, Cd, Cdd, Sp, Sm
        if conserve_S != 'Stot':
            Cu = np.zeros((3, 3), dtype=float)
            Cu[0, 1] = 1.  # up -> emtpy
            Cdu = np.transpose(Cu)
            # For spin-down annihilation operator: include a Jordan-Wigner string JWu
            # this ensures that Cdu.Cd = - Cd.Cdu
            # c.f. the chapter on the Jordan-Wigner trafo in the userguide
            Cd_noJW = np.zeros((3, 3), dtype=float)
            Cd_noJW[0, 2] = 1.
            Cd = JWu[:, None] * Cd_noJW
            Cdd = np.transpose(Cd)
            Sp = np.dot(Cdu, Cd)
            Sm = np.dot(Cdd, Cu)
            ops.update(Cu=Cu, Cdu=Cdu, Cd=Cd, Cdd=Cdd, Sp=Sp, Sm=Sm)
        # build Sx, Sy
        if conserve_S in ['parity', 'None']:
            ops.update(Sx=.5 * (Sp + Sm), Sy=-.5j * (Sp - Sm))
        elif conserve_S == 'Sz':
            pass  # TODO build gen(2) version
        # build Svec
        if conserve_S == 'Stot':
            sector = [2]  # spin 1
            # sectors [0, 0, 2] do not fulfill the charge rule, so this operator only acts
            # non-trivially on the [up, down] spin doublet where the sectors are [1, 1, 2].
            # this is what the Svec operator should do.
            # this also means that the same construction as for the SpinHalfSite works here too.
            if sym_N is not None:
                sector.append(0)
            dummy_leg = VectorSpace(leg.symmetry, sectors=[sector])
            Svec_inv = Tensor.from_block_func(
                backend.ones_block, backend=backend, legs=[leg, leg.dual, dummy_leg],
                labels=['p', 'p*', '!']
            )
            ops.update(Svec=ChargedTensor(Svec_inv))
        else:
            pass  # TODO build gen(3) version
        # initialize
        self.conserve_N = conserve_N
        self.conserve_S = conserve_S
        self.filling = filling
        states = ['empty', 'up', 'down']
        Site.__init__(self, leg=leg, backend=backend, state_labels=states, **ops)
        # specify fermionic operators
        self.need_JW_string |= set(['Cu', 'Cdu', 'Cd', 'Cdd', 'JWu', 'JWd', 'JW'])

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

    The following table lists all local operators and their :ref:`categories <OperatorCategories>`.

    ============  =======================================  ========  ========  ========
    operator      description                              N         parity    None
    ============  =======================================  ========  ========  ========
    ``Id, JW``    Identity :math:`\mathbb{1}`              diag      diag      diag
    ``B``         Annihilation operator :math:`b`          gen       gen       sym
    ``Bd``        Creation operator :math:`b^\dagger`      gen       gen       sym
    ``N``         Number operator :math:`n= b^\dagger b`   diag      diag      diag
    ``NN``        :math:`n^2`                              diag      diag      diag
    ``dN``        :math:`\delta n := n - filling`          diag      diag      diag
    ``dNdN``      :math:`(\delta n)^2`                     diag      diag      diag
    ``P``         Parity :math:`(-1)^n`                    diag      diag      diag
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
    backend : :class:`~tenpy.linalg.backends.AbstractBackend`, optional
        The backend used to create the operators.
    """

    def __init__(self, Nmax: int = 1, conserve: str = 'N', filling: float = 0.,
                 backend: AbstractBackend = None):
        assert Nmax > 0
        d = Nmax + 1
        N = np.arange(d)
        # build leg
        if conserve == 'N':
            leg = VectorSpace.from_sectors(U1Symmetry('N'), N[:, None])
        elif conserve == 'parity':
            leg = VectorSpace.from_sectors(ZNSymmetry(2, 'parity_N'), N[:, None] % 2)
        elif conserve == 'None':
            leg = VectorSpace.from_trivial_sector(d)
        else:
            raise ValueError(f'invalid `conserve`: {conserve}')
        if backend is None:
            backend = get_backend(symmetry=leg.symmetry)
        # diagonal operators
        ops = dict(N=N, NN=N ** 2, dN=(N - filling), dNdN=(N - filling) ** 2, P=1. - 2. * (N % 2))
        ops = {name: DiagonalTensor.from_diag_numpy(op, leg, backend=backend, labels=['p', 'p*'])
               for name, op in ops.items()}
        # remaining ops: B, Bd
        B = np.zeros([d, d], dtype=float)
        for n in range(1, d):
            B[n - 1, n] = np.sqrt(n)
        ops.update(B=B, Bd=np.transpose(B))
        # initialize
        labels = [str(n) for n in range(d)]
        self.Nmax = Nmax
        self.conserve = conserve
        self.filling = filling
        Site.__init__(self, leg, backend=backend, state_labels=labels, **ops)
        self.state_labels['vac'] = self.state_labels['0']  # alias

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
        Using seperate sites for spin up and down excludes ``'Stot'`` conservation.
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

    ============  =====================  ==============  =================================
    `conserve`    symmetry               sectors         meaning of sector label
    ============  =====================  ==============  =================================
    ``'Z'``       ``ZNSymmetry(N=q)``    ``range(q)``    sector ``n`` has ``Z = w ** n``
    ``'None'``    ``NoSymmetry``         ``[0, ...]``    --
    ============  =====================  ==============  =================================

    Local operators are the clock operators ``Z = diag([w ** 0, w ** 1, ..., w ** (q - 1)])``
    with ``w = exp(2.j * pi / q)`` and ``X = eye(q, k=1) + eye(q, k=1-q)``, which are not hermitian.
    They are generalizations of the pauli operators and fulfill the clock algebra
    :math:`X Z = \mathtt{w} Z X` and :math:`X^q = \mathbb{1} = Z^q`.
    The following table lists all local operators and their :ref:`categories <OperatorCategories>`.

    ============  =====================================  ========  ========
    operator      description                            Z         None
    ============  =====================================  ========  ========
    ``Id, JW``    Identity :math:`\mathbb{1}`            diag      diag
    ``Z, Zhc``    Clock operator Z & its conjugate       diag      diag
    ``Zphc``      "Real part" :math:`Z + Z^\dagger`      diag      diag
    ``X, Xhc``    Clock operator X & its conjugate       gen       sym
    ``Xphc``      "Real part" :math:`X + X^\dagger`      gen(2)    sym
    ============  =====================================  ========  ========

    Parameters
    ----------
    q : int
        Number of states per site
    conserve : 'Z' | 'None'
        Defines what is conserved, see table above.
    backend : :class:`~tenpy.linalg.backends.AbstractBackend`, optional
        The backend used to create the operators.
    """
    def __init__(self, q: int, conserve: str = 'Z', backend: AbstractBackend = None):
        if not (isinstance(q, int) and q > 1):
            raise ValueError(f'invalid q: {q}')
        # build leg
        if conserve == 'Z':
            leg = VectorSpace.from_basis(ZNSymmetry(q, 'clock_phase'), np.arange(q)[:, None])
        elif conserve == 'None':
            leg = VectorSpace.from_trivial_sector(q)
        else:
            raise ValueError(f'invalid `conserve`: {conserve}')
        if backend is None:
            backend = get_backend(symmetry=leg.symmetry)
        # diagonal operators : Z, Zhc, Zphc
        Z = np.exp(2.j * np.pi * np.arange(q, dtype=np.complex128) / q)
        Zhc = Z.conj()
        Zphc = 2. * np.cos(2. * np.pi * np.arange(q, dtype=np.complex128) / q)
        ops = dict(Z=Z, Zhc=Zhc, Zphc=Zphc)
        ops = {name: DiagonalTensor.from_diag_numpy(op, leg, backend=backend, labels=['p', 'p*'])
               for name, op in ops.items()}
        # build operators X, Xhc
        X = np.eye(q, k=1) + np.eye(q, k=1-q)
        Xhc = X.conj().transpose()
        ops.update(X=X, Xhc=Xhc)
        # build Xphc
        if conserve == 'Z':
            pass  # TODO add Xphc as gen!
        else:
            ops.update(Xphc=X + Xhc)
        # initialize
        names = [str(m) for m in range(q)]
        self.q = q
        self.conserve = conserve
        Site.__init__(self, leg=leg, backend=backend, state_labels=names, **ops)
        self.state_labels['up'] = self.state_labels['0']
        if q % 2 == 0:
            self.state_labels['down'] = self.state_labels[str(q // 2)]

    def __repr__(self):
        return f'ClockSite(q={self.q}, conserve={self.conserve})'
