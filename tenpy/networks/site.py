"""Defines a class describing the local physical Hilbert space.

The :class:`Site` is the prototype, read it's docstring.

"""
# Copyright 2018-2023 TeNPy Developers, GNU GPLv3
from __future__ import annotations
import numpy as np
import itertools
import copy
import warnings
from functools import partial

from ..linalg.tensors import (AbstractTensor, Tensor, SymmetricTensor, ChargedTensor,
                              DiagonalTensor, almost_equal, tensor_from_block, angle, real_if_close,
                              get_same_backend)
from ..linalg.backends import AbstractBackend
from ..linalg.matrix_operations import exp
from ..linalg.groups import (ProductSymmetry, Symmetry, SU2Symmetry, U1Symmetry, ZNSymmetry,
                             no_symmetry)
from ..linalg.spaces import VectorSpace, ProductSpace
from ..linalg.backends import todo_get_backend, Block
from ..tools.misc import inverse_permutation, find_subclass
from ..tools.hdf5_io import Hdf5Exportable

__all__ = [
    'Site',
    'GroupedSite',
    'group_sites',
    'set_common_symmetry',
    'SpinHalfSite',
    # TODO uncomment all
    # 'SpinSite',
    # 'FermionSite',
    # 'SpinHalfFermionSite',
    # 'SpinHalfHoleSite',
    # 'BosonSite',
    # 'ClockSite',
    # 'spin_half_species',
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

    All sites define the operators ``'Id'``, the identity, and ``'JW'``, the local contribution to
    Jordan-Wigner strings, both of which are symmetric.

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
            legs = [sites[0].leg.drop_symmetry()]
            res_symmetry = legs[0].symmetry
            for site in sites[1:]:
                legs.append(site.leg.drop_symmetry(symmetry=res_symmetry))
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
        self.leg = leg = ProductSpace(legs, backend=todo_get_backend())
        JW_all = self.kroneckerproduct([s.JW for s in sites])
        # initialize Site , will set labels and add ops below
        Site.__init__(self, leg, backend=get_same_backend(*(s.Id for s in sites)),
                      state_labels=None, ops=dict(JW=JW_all))
        # set state labels
        dims = np.array([site.dim for site in sites])
        if leg.symmetry.is_abelian:
            perm = leg.get_basis_transformation_perm()
            strides = np.cumprod(dims[::-1])[::-1]  # C-style: first index has largest stride
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
            for name, (op_in, op_out) in site.general_ops.items():
                need_JW = name in site.need_JW_string
                hc = False if name not in site.hc_ops else site.hc_ops[name] + labels[i]
                ops = JW_Ids if need_JW else Ids
                ops[i] = op_in
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
        op = ops[0].relabel({'p': 'p0', 'p*': 'p0*'})
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


def set_common_symmetry(sites: list[Site], symmetry_combine: callable | str = 'by_name',
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
    symmetry_combine : ``'by_name'`` | ``'drop'`` | ``'independent'`` | function
        Defines how the new common symmetry arises from the individual symmetries.
        We split any `ProducSymmetry` from the sites into their individual factors and build the
        resulting symmetry from these factors of all sites.

        ``'by_name'``
            Default. Considers equal factors (i.e. same mathematical group and same name) to
            be the same symmetry (i.e. the sum of their charges is conserved). Different factors
            are considered as independent (i.e. their charges are conserved individually).
        ``'drop'``
            Drop all symmetries.
        ``'independent'``
            Consider all factors as independent, even if they have the same name.
        function
            A function with call structure

                new_sector: SectorArray = symmetry_combine(site_idx: int, old_sector: SectorArray)

            That specifies how the sectors of the new symmetry arise from those of the old symmetry,
            i.e. ``old_sector`` is a sector of the old symmetry on the ``site_idx``-th site.
            Using this option makes the `new_symmetry` parameter required.
    new_symmetry : :class:`Symmetry`, optional
        The new symmetry. Is ignored if `symmetry_combine` is one of the pre-defined (``str``)
        options. Is required if `symmetry_combine` is a function.

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
                sector_slices = [0, site.leg.symmetry.ind_len]
            for n, f in enumerate(new_factors):
                slc = slice(sector_slices[n], sector_slices[n + 1])
                # Symmetry.__eq__ checks for same mathematical group *and* same descriptive_name
                if f in factors:
                    n = factors.index(f)
                    sites_and_slices[n].append((i, slc))
                else:
                    factors.append(f)
                    sites_and_slices.append([(i, slc)])
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

            new_leg = site.leg.apply_sector_map(symmetry=new_symmetry, sector_map=symmetry_combine)
            site.change_leg(new_leg)
        return
    
    if symmetry_combine == 'drop':
        assert new_symmetry is None
        for site in sites:
            site.change_leg(site.leg.drop_symmetry())
        return

    if symmetry_combine == 'independent':
        new_symmetry = ProductSymmetry.from_nested_factors([site.leg.symmetry for site in sites])
        start = 0
        for i, site in enumerate(sites):
            ind_len = site.leg.symmetry.ind_len

            def symmetry_combine(s):
                res = np.tile(new_symmetry.trivial_sector[None, :], (len(s), 1))
                res[:, start:start + ind_len] = s
                return res

            site.change_leg(site.leg.change_symmetry(symmetry=new_symmetry, sector_map=symmetry_combine))
        return

    elif isinstance(symmetry_combine, str):
        raise ValueError(f'Unknown sector_map keyword: "{symmetry_combine}"')
        
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

    ==============  =================  ============  =========================
    `conserve`      symmetry           sectors       meaning of sector label
    ==============  =================  ============  =========================
    ``'SU(2)'``     SU2Symmetry        ``[1]``       2 * S
    ``'Sz'``        U1Symmetry         ``[1, -1]``   2 * Sz
    ``'parity'``    ZNSymmetry(N=2)    ``[1, 0]``    (# spin up) mod 2
    ``'None'``      NoSymmetry         ``[0, 0]``    --
    ==============  =================  ============  =========================

    TODO include dipole symmetry here too?
    
    Local operators are the usual spin-1/2 operators, e.g. ``Sz = [[0.5, 0.], [0., -0.5]]``,
    ``Sx = 0.5 * Sigmax`` for the Pauli matrix `Sigmax`.

    ====================  =====================================  =======  ====  ========  ======
    operator              description                            SU(2)    Sz    parity    None
    ====================  =====================================  =======  ====  ========  ======
    ``Id, JW``            Identity :math:`\mathbb{1}`            sym      sym   sym       sym
    ``Sx, Sy``            Spin components :math:`S^{x,y}`        --       gen!  gen!      sym
    ``Sz``                Spin component :math:`S^{z}`           --       sym   sym       sym
    ``Sigmax, Sigmay``    Pauli matrices :math:`\sigma^{x,y}`    --       gen!  gen!      sym
    ``Sigmaz``            Pauli matrix :math:`\sigma^{z}`        --       sym   sym       sym
    ``Sp, Sm``            :math:`S^{\pm} = S^{x} \pm i S^{y}`    --       gen   gen       sym
    ``Svec``              The vector of spin operators.          gen      gen!  gen!      gen!
    ====================  =====================================  =======  ====  ========  ======

    TODO where should we explain the keys:
    sym : symmetric
    gen : general with dummy_leg.dim == 1
    gen! : general with dummy_leg.dim > 1

    Parameters
    ----------
    conserve : str | None
        Defines what is conserved, see table above.
    backend : :class:`~tenpy.linalg.backends.AbstractBackend`, optional
        The backend used to create the operators.
    """
    def __init__(self, conserve: str = 'Sz', backend: AbstractBackend = None):
        assert conserve in ['SU(2)', 'Sz', 'parity', 'None']
        self.conserve = conserve
        backend = todo_get_backend()
        # make leg
        if conserve == 'SU(2)':
            leg = VectorSpace(symmetry=SU2Symmetry('SU(2)_spin'), sectors=[[1]])
        elif conserve == 'Sz':
            leg = VectorSpace.from_basis(U1Symmetry('2*Sz'), [[1], [-1]])
        elif conserve == 'parity':
            leg = VectorSpace.from_basis(ZNSymmetry(2, 'parity_Sz'), [[1], [0]])
        else:
            leg = VectorSpace.from_trivial_sector(2)
    
        if conserve == 'SU(2)':
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

        if conserve == 'Sz':
            pass  # TODO add ChargedTensor versions of Sx, Sy with length 2 dummy legs. Then also Sigmax below.
        if conserve in ['parity', 'None']:
            ops.update(Sx=[[0., 0.5], [0.5, 0.]], Sy=[[0., -0.5j], [+0.5j, 0.]])
        # Specify Hermitian conjugates
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


# TODO reintroduce the commented-out sites, implement and document like SpinHalfSite
# class SpinSite(Site):
#     r"""General Spin S site.

#     There are `2S+1` local states range from ``down`` (0)  to ``up`` (2S+1),
#     corresponding to ``Sz=-S, -S+1, ..., S-1, S``.
#     Local operators are the spin-S operators,
#     e.g. ``Sz = [[0.5, 0.], [0., -0.5]]``,
#     ``Sx = 0.5*sigma_x`` for the Pauli matrix `sigma_x`.

#     ==============  ================================================
#     operator        description
#     ==============  ================================================
#     ``Id, JW``      Identity :math:`\mathbb{1}`
#     ``Sx, Sy, Sz``  Spin components :math:`S^{x,y,z}`,
#                     equal to half the Pauli matrices.
#     ``Sp, Sm``      Spin flips :math:`S^{\pm} = S^{x} \pm i S^{y}`
#     ==============  ================================================

#     ============== ====  ============================
#     `conserve`     qmod  *excluded* onsite operators
#     ============== ====  ============================
#     ``'Sz'``       [1]   ``Sx, Sy, Sigmax, Sigmay``
#     ``'parity'``   [2]   --
#     ``'None'``     []    --
#     ============== ====  ============================

#     Parameters
#     ----------
#     conserve : str
#         Defines what is conserved, see table above.
#     sort_charge : bool
#         Whether :meth:`sort_charge` should be called at the end of initialization.
#         This is usually a good idea to reduce potential overhead when using charge conservation.
#         Note that this permutes the order of the local basis states for ``conserve='parity'``!
#         For backwards compatibility with existing data, it is not (yet) enabled by default.

#     Attributes
#     ----------
#     S : {0.5, 1, 1.5, 2, ...}
#         The 2S+1 states range from m = -S, -S+1, ... +S.
#     conserve : str
#         Defines what is conserved, see table above.
#     """

#     def __init__(self, S=0.5, conserve='Sz', sort_charge=None):
#         if not conserve:
#             conserve = 'None'
#         if conserve not in ['Sz', 'parity', 'None']:
#             raise ValueError("invalid `conserve`: " + repr(conserve))
#         self.S = S = float(S)
#         d = 2 * S + 1
#         if d <= 1:
#             raise ValueError("negative S?")
#         if np.rint(d) != d:
#             raise ValueError("S is not half-integer or integer")
#         d = int(d)
#         Sz_diag = -S + np.arange(d)
#         Sz = np.diag(Sz_diag)
#         Sp = np.zeros([d, d])
#         for n in np.arange(d - 1):
#             # Sp |m> =sqrt( S(S+1)-m(m+1)) |m+1>
#             m = n - S
#             Sp[n + 1, n] = np.sqrt(S * (S + 1) - m * (m + 1))
#         Sm = np.transpose(Sp)
#         # Sp = Sx + i Sy, Sm = Sx - i Sy
#         Sx = (Sp + Sm) * 0.5
#         Sy = (Sm - Sp) * 0.5j
#         # Note: For S=1/2, Sy might look wrong compared to the Pauli matrix or SpinHalfSite.
#         # Don't worry, I'm 99.99% sure it's correct (J. Hauschild)
#         # The reason it looks wrong is simply that this class orders the states as ['down', 'up'],
#         # while the usual spin-1/2 convention is ['up', 'down'], as you can also see if you look
#         # at the Sz entries...
#         # (The commutation relations are checked explicitly in `tests/test_site.py`)
#         ops = dict(Sp=Sp, Sm=Sm, Sz=Sz)
#         if conserve == 'Sz':
#             chinfo = npc.ChargeInfo([1], ['2*Sz'])
#             leg = npc.LegCharge.from_qflat(chinfo, np.array(2 * Sz_diag, dtype=np.int64))
#         else:
#             ops.update(Sx=Sx, Sy=Sy)
#             if conserve == 'parity':
#                 chinfo = npc.ChargeInfo([2], ['parity_Sz'])
#                 leg = npc.LegCharge.from_qflat(chinfo, np.mod(np.arange(d), 2))
#             else:
#                 leg = npc.LegCharge.from_trivial(d)
#         self.conserve = conserve
#         names = [str(i) for i in np.arange(-S, S + 1, 1.)]
#         Site.__init__(self, leg, names, sort_charge=sort_charge, **ops)
#         self.state_labels['down'] = self.state_labels[names[0]]
#         self.state_labels['up'] = self.state_labels[names[-1]]

#     def __repr__(self):
#         """Debug representation of self."""
#         return "SpinSite(S={S!s}, {c!r})".format(S=self.S, c=self.conserve)


# class FermionSite(Site):
#     r"""Create a :class:`Site` for spin-less fermions.

#     Local states are ``empty`` and ``full``.

#     .. warning ::
#         Using the Jordan-Wigner string (``JW``) is crucial to get correct results,
#         otherwise you just describe hardcore bosons!
#         Further details in :doc:`/intro/JordanWigner`.

#     ==============  ===================================================================
#     operator        description
#     ==============  ===================================================================
#     ``Id``          Identity :math:`\mathbb{1}`
#     ``JW``          Sign for the Jordan-Wigner string.
#     ``C``           Annihilation operator :math:`c` (up to 'JW'-string left of it)
#     ``Cd``          Creation operator :math:`c^\dagger` (up to 'JW'-string left of it)
#     ``N``           Number operator :math:`n= c^\dagger c`
#     ``dN``          :math:`\delta n := n - filling`
#     ``dNdN``        :math:`(\delta n)^2`
#     ==============  ===================================================================

#     ============== ====  ===============================
#     `conserve`     qmod  *exluded* onsite operators
#     ============== ====  ===============================
#     ``'N'``        [1]   --
#     ``'parity'``   [2]   --
#     ``'None'``     []    --
#     ============== ====  ===============================

#     Parameters
#     ----------
#     conserve : str
#         Defines what is conserved, see table above.
#     filling : float
#         Average filling. Used to define ``dN``.

#     Attributes
#     ----------
#     conserve : str
#         Defines what is conserved, see table above.
#     filling : float
#         Average filling. Used to define ``dN``.
#     """

#     def __init__(self, conserve='N', filling=0.5):
#         if not conserve:
#             conserve = 'None'
#         if conserve not in ['N', 'parity', 'None']:
#             raise ValueError("invalid `conserve`: " + repr(conserve))
#         JW = np.array([[1., 0.], [0., -1.]])
#         C = np.array([[0., 1.], [0., 0.]])
#         Cd = np.array([[0., 0.], [1., 0.]])
#         N = np.array([[0., 0.], [0., 1.]])
#         dN = np.array([[-filling, 0.], [0., 1. - filling]])
#         dNdN = dN**2  # (element wise power is fine since dN is diagonal)
#         ops = dict(JW=JW, C=C, Cd=Cd, N=N, dN=dN, dNdN=dNdN)
#         if conserve == 'N':
#             chinfo = npc.ChargeInfo([1], ['N'])
#             leg = npc.LegCharge.from_qflat(chinfo, [0, 1])
#         elif conserve == 'parity':
#             chinfo = npc.ChargeInfo([2], ['parity_N'])
#             leg = npc.LegCharge.from_qflat(chinfo, [0, 1])
#         else:
#             leg = npc.LegCharge.from_trivial(2)
#         self.conserve = conserve
#         self.filling = filling
#         Site.__init__(self, leg, ['empty', 'full'], sort_charge=True, **ops)
#         # specify fermionic operators
#         self.need_JW_string |= set(['C', 'Cd', 'JW'])

#     def __repr__(self):
#         """Debug representation of self."""
#         return "FermionSite({c!r}, {f:f})".format(c=self.conserve, f=self.filling)


# class SpinHalfFermionSite(Site):
#     r"""Create a :class:`Site` for spinful (spin-1/2) fermions.

#     Local states are:
#          ``empty``  (vacuum),
#          ``up``     (one spin-up electron),
#          ``down``   (one spin-down electron), and
#          ``full``   (both electrons)

#     Local operators can be built from creation operators.

#     .. warning ::
#         Using the Jordan-Wigner string (``JW``) in the correct way is crucial to get correct
#         results, otherwise you just describe hardcore bosons!

#     ==============  =============================================================================
#     operator        description
#     ==============  =============================================================================
#     ``Id``          Identity :math:`\mathbb{1}`
#     ``JW``          Sign for the Jordan-Wigner string :math:`(-1)^{n_{\uparrow}+n_{\downarrow}}`
#     ``JWu``         Partial sign for the Jordan-Wigner string :math:`(-1)^{n_{\uparrow}}`
#     ``JWd``         Partial sign for the Jordan-Wigner string :math:`(-1)^{n_{\downarrow}}`
#     ``Cu``          Annihilation operator spin-up :math:`c_{\uparrow}`
#                     (up to 'JW'-string on sites left of it).
#     ``Cdu``         Creation operator spin-up :math:`c^\dagger_{\uparrow}`
#                     (up to 'JW'-string on sites left of it).
#     ``Cd``          Annihilation operator spin-down :math:`c_{\downarrow}`
#                     (up to 'JW'-string on sites left of it).
#                     Includes ``JWu`` such that it anti-commutes onsite with ``Cu, Cdu``.
#     ``Cdd``         Creation operator spin-down :math:`c^\dagger_{\downarrow}`
#                     (up to 'JW'-string on sites left of it).
#                     Includes ``JWu`` such that it anti-commutes onsite with ``Cu, Cdu``.
#     ``Nu``          Number operator :math:`n_{\uparrow}= c^\dagger_{\uparrow} c_{\uparrow}`
#     ``Nd``          Number operator :math:`n_{\downarrow}= c^\dagger_{\downarrow} c_{\downarrow}`
#     ``NuNd``        Dotted number operators :math:`n_{\uparrow} n_{\downarrow}`
#     ``Ntot``        Total number operator :math:`n_t= n_{\uparrow} + n_{\downarrow}`
#     ``dN``          Total number operator compared to the filling :math:`\Delta n = n_t-filling`
#     ``Sx, Sy, Sz``  Spin operators :math:`S^{x,y,z}`, in particular
#                     :math:`S^z = \frac{1}{2}( n_\uparrow - n_\downarrow )`
#     ``Sp, Sm``      Spin flips :math:`S^{\pm} = S^{x} \pm i S^{y}`,
#                     e.g. :math:`S^{+} = c^\dagger_\uparrow c_\downarrow`
#     ==============  =============================================================================

#     The spin operators are defined as :math:`S^\gamma =
#     (c^\dagger_{\uparrow}, c^\dagger_{\downarrow}) \sigma^\gamma (c_{\uparrow}, c_{\downarrow})^T`,
#     where :math:`\sigma^\gamma` are spin-1/2 matrices (i.e. half the pauli matrices).

#     ============= ============= ======= =======================================
#     `cons_N`      `cons_Sz`     qmod    *excluded* onsite operators
#     ============= ============= ======= =======================================
#     ``'N'``       ``'Sz'``      [1, 1]  ``Sx, Sy``
#     ``'N'``       ``'parity'``  [1, 2]  --
#     ``'N'``       ``None``      [1]     --
#     ``'parity'``  ``'Sz'``      [2, 1]  ``Sx, Sy``
#     ``'parity'``  ``'parity'``  [2, 2]  --
#     ``'parity'``  ``None``      [2]     --
#     ``None``      ``'Sz'``      [1]     ``Sx, Sy``
#     ``None``      ``'parity'``  [2]     --
#     ``None``      ``None``      []      --
#     ============= ============= ======= =======================================

#     Parameters
#     ----------
#     cons_N : ``'N' | 'parity' | None``
#         Whether particle number is conserved, c.f. table above.
#     cons_Sz : ``'Sz' | 'parity' | None``
#         Whether spin is conserved, c.f. table above.
#     filling : float
#         Average filling. Used to define ``dN``.

#     Attributes
#     ----------
#     cons_N : ``'N' | 'parity' | None``
#         Whether particle number is conserved, c.f. table above.
#     cons_Sz : ``'Sz' | 'parity' | None``
#         Whether spin is conserved, c.f. table above.
#     filling : float
#         Average filling. Used to define ``dN``.
#     """

#     def __init__(self, cons_N='N', cons_Sz='Sz', filling=1.):
#         if not cons_N:
#             cons_N = 'None'
#         if cons_N not in ['N', 'parity', 'None']:
#             raise ValueError("invalid `cons_N`: " + repr(cons_N))
#         if not cons_Sz:
#             cons_Sz = 'None'
#         if cons_Sz not in ['Sz', 'parity', 'None']:
#             raise ValueError("invalid `cons_Sz`: " + repr(cons_Sz))
#         d = 4
#         states = ['empty', 'up', 'down', 'full']
#         # 0) Build the operators.
#         Nu_diag = np.array([0., 1., 0., 1.], dtype=np.float64)
#         Nd_diag = np.array([0., 0., 1., 1.], dtype=np.float64)
#         Nu = np.diag(Nu_diag)
#         Nd = np.diag(Nd_diag)
#         Ntot = np.diag(Nu_diag + Nd_diag)
#         dN = np.diag(Nu_diag + Nd_diag - filling)
#         NuNd = np.diag(Nu_diag * Nd_diag)
#         JWu = np.diag(1. - 2 * Nu_diag)  # (-1)^Nu
#         JWd = np.diag(1. - 2 * Nd_diag)  # (-1)^Nd
#         JW = JWu * JWd  # (-1)^{Nu+Nd}

#         Cu = np.zeros((d, d))
#         Cu[0, 1] = Cu[2, 3] = 1
#         Cdu = np.transpose(Cu)
#         # For spin-down annihilation operator: include a Jordan-Wigner string JWu
#         # this ensures that Cdu.Cd = - Cd.Cdu
#         # c.f. the chapter on the Jordan-Wigner trafo in the userguide
#         Cd_noJW = np.zeros((d, d))
#         Cd_noJW[0, 2] = Cd_noJW[1, 3] = 1
#         Cd = np.dot(JWu, Cd_noJW)  # (don't do this for spin-up...)
#         Cdd = np.transpose(Cd)

#         # spin operators are defined as  (Cdu, Cdd) S^gamma (Cu, Cd)^T,
#         # where S^gamma is the 2x2 matrix for spin-half
#         Sz = np.diag(0.5 * (Nu_diag - Nd_diag))
#         Sp = np.dot(Cdu, Cd)
#         Sm = np.dot(Cdd, Cu)
#         Sx = 0.5 * (Sp + Sm)
#         Sy = -0.5j * (Sp - Sm)

#         ops = dict(JW=JW, JWu=JWu, JWd=JWd,
#                    Cu=Cu, Cdu=Cdu, Cd=Cd, Cdd=Cdd,
#                    Nu=Nu, Nd=Nd, Ntot=Ntot, NuNd=NuNd, dN=dN,
#                    Sx=Sx, Sy=Sy, Sz=Sz, Sp=Sp, Sm=Sm)  # yapf: disable

#         # handle charges
#         qmod = []
#         qnames = []
#         charges = []
#         if cons_N == 'N':
#             qnames.append('N')
#             qmod.append(1)
#             charges.append([0, 1, 1, 2])
#         elif cons_N == 'parity':
#             qnames.append('parity_N')
#             qmod.append(2)
#             charges.append([0, 1, 1, 0])
#         if cons_Sz == 'Sz':
#             qnames.append('2*Sz')  # factor 2 s.t. Cu, Cd have well-defined charges!
#             qmod.append(1)
#             charges.append([0, 1, -1, 0])
#             del ops['Sx']
#             del ops['Sy']
#         elif cons_Sz == 'parity':
#             qnames.append('parity_Sz')  # the charge is (2*Sz) mod (2*2)
#             qmod.append(4)
#             charges.append([0, 1, 3, 0])  # == [0, 1, -1, 0] mod 4
#             # e.g. terms like `Sp_i Sp_j + hc` with Sp=Cdu Cd have charges 'N', 'parity_Sz'.
#             # The `parity_Sz` is non-trivial in this case!
#         if len(qmod) == 0:
#             leg = npc.LegCharge.from_trivial(d)
#         else:
#             if len(qmod) == 1:
#                 charges = charges[0]
#             else:  # len(charges) == 2: need to transpose
#                 charges = [[q1, q2] for q1, q2 in zip(charges[0], charges[1])]
#             chinfo = npc.ChargeInfo(qmod, qnames)
#             leg = npc.LegCharge.from_qflat(chinfo, charges)
#         self.cons_N = cons_N
#         self.cons_Sz = cons_Sz
#         self.filling = filling
#         Site.__init__(self, leg, states, sort_charge=True, **ops)
#         # specify fermionic operators
#         self.need_JW_string |= set(['Cu', 'Cdu', 'Cd', 'Cdd', 'JWu', 'JWd', 'JW'])

#     def __repr__(self):
#         """Debug representation of self."""
#         return "SpinHalfFermionSite({cN!r}, {cS!r}, {f:f})".format(cN=self.cons_N,
#                                                                    cS=self.cons_Sz,
#                                                                    f=self.filling)


# class SpinHalfHoleSite(Site):
#     r"""Create a :class:`Site` for spinful (spin-1/2) fermions, restricted to empty or singly occupied sites

#     Local states are:
#          ``empty``  (vacuum),
#          ``up``     (one spin-up electron),
#          ``down``   (one spin-down electron)

#     Local operators can be built from creation operators.

#     .. warning ::
#         Using the Jordan-Wigner string (``JW``) in the correct way is crucial to get correct
#         results, otherwise you just describe hardcore bosons!

#     ==============  =============================================================================
#     operator        description
#     ==============  =============================================================================
#     ``Id``          Identity :math:`\mathbb{1}`
#     ``JW``          Sign for the Jordan-Wigner string :math:`(-1)^{n_{\uparrow}+n_{\downarrow}}`
#     ``JWu``         Partial sign for the Jordan-Wigner string :math:`(-1)^{n_{\uparrow}}`
#     ``JWd``         Partial sign for the Jordan-Wigner string :math:`(-1)^{n_{\downarrow}}`
#     ``Cu``          Annihilation operator spin-up :math:`c_{\uparrow}`
#                     (up to 'JW'-string on sites left of it).
#     ``Cdu``         Creation operator spin-up :math:`c^\dagger_{\uparrow}`
#                     (up to 'JW'-string on sites left of it).
#     ``Cd``          Annihilation operator spin-down :math:`c_{\downarrow}`
#                     (up to 'JW'-string on sites left of it).
#                     Includes ``JWu`` such that it anti-commutes onsite with ``Cu, Cdu``.
#     ``Cdd``         Creation operator spin-down :math:`c^\dagger_{\downarrow}`
#                     (up to 'JW'-string on sites left of it).
#                     Includes ``JWu`` such that it anti-commutes onsite with ``Cu, Cdu``.
#     ``Nu``          Number operator :math:`n_{\uparrow}= c^\dagger_{\uparrow} c_{\uparrow}`
#     ``Nd``          Number operator :math:`n_{\downarrow}= c^\dagger_{\downarrow} c_{\downarrow}`
#     ``Ntot``        Total number operator :math:`n_t= n_{\uparrow} + n_{\downarrow}`
#     ``dN``          Total number operator compared to the filling :math:`\Delta n = n_t-filling`
#     ``Sx, Sy, Sz``  Spin operators :math:`S^{x,y,z}`, in particular
#                     :math:`S^z = \frac{1}{2}( n_\uparrow - n_\downarrow )`
#     ``Sp, Sm``      Spin flips :math:`S^{\pm} = S^{x} \pm i S^{y}`,
#                     e.g. :math:`S^{+} = c^\dagger_\uparrow c_\downarrow`
#     ==============  =============================================================================

#     The spin operators are defined as :math:`S^\gamma =
#     (c^\dagger_{\uparrow}, c^\dagger_{\downarrow}) \sigma^\gamma (c_{\uparrow}, c_{\downarrow})^T`,
#     where :math:`\sigma^\gamma` are spin-1/2 matrices (i.e. half the pauli matrices).

#     ============= ============= ======= =======================================
#     `cons_N`      `cons_Sz`     qmod    *excluded* onsite operators
#     ============= ============= ======= =======================================
#     ``'N'``       ``'Sz'``      [1, 1]  ``Sx, Sy``
#     ``'N'``       ``'parity'``  [1, 2]  --
#     ``'N'``       ``None``      [1]     --
#     ``'parity'``  ``'Sz'``      [2, 1]  ``Sx, Sy``
#     ``'parity'``  ``'parity'``  [2, 2]  --
#     ``'parity'``  ``None``      [2]     --
#     ``None``      ``'Sz'``      [1]     ``Sx, Sy``
#     ``None``      ``'parity'``  [2]     --
#     ``None``      ``None``      []      --
#     ============= ============= ======= =======================================

#     Parameters
#     ----------
#     cons_N : ``'N' | 'parity' | None``
#         Whether particle number is conserved, c.f. table above.
#     cons_Sz : ``'Sz' | 'parity' | None``
#         Whether spin is conserved, c.f. table above.
#     filling : float
#         Average filling. Used to define ``dN``.

#     Attributes
#     ----------
#     cons_N : ``'N' | 'parity' | None``
#         Whether particle number is conserved, c.f. table above.
#     cons_Sz : ``'Sz' | 'parity' | None``
#         Whether spin is conserved, c.f. table above.
#     filling : float
#         Average filling. Used to define ``dN``.
#     """

#     def __init__(self, cons_N='N', cons_Sz='Sz', filling=1.):
#         if not cons_N:
#             cons_N = 'None'
#         if cons_N not in ['N', 'parity', 'None']:
#             raise ValueError("invalid `cons_N`: " + repr(cons_N))
#         if not cons_Sz:
#             cons_Sz = 'None'
#         if cons_Sz not in ['Sz', 'parity', 'None']:
#             raise ValueError("invalid `cons_Sz`: " + repr(cons_Sz))
#         d = 3
#         states = ['empty', 'up', 'down']
#         # 0) Build the operators.
#         Nu_diag = np.array([0., 1., 0.], dtype=np.float64)
#         Nd_diag = np.array([0., 0., 1.], dtype=np.float64)
#         Nu = np.diag(Nu_diag)
#         Nd = np.diag(Nd_diag)
#         Ntot = np.diag(Nu_diag + Nd_diag)
#         dN = np.diag(Nu_diag + Nd_diag - filling)
#         JWu = np.diag(1. - 2 * Nu_diag)  # (-1)^Nu
#         JWd = np.diag(1. - 2 * Nd_diag)  # (-1)^Nd
#         JW = JWu * JWd  # (-1)^{Nu+Nd}

#         Cu = np.zeros((d, d))
#         Cu[0, 1] = 1
#         Cdu = np.transpose(Cu)
#         # For spin-down annihilation operator: include a Jordan-Wigner string JWu
#         # this ensures that Cdu.Cd = - Cd.Cdu
#         # c.f. the chapter on the Jordan-Wigner trafo in the userguide
#         Cd_noJW = np.zeros((d, d))
#         Cd_noJW[0, 2] = 1
#         Cd = np.dot(JWu, Cd_noJW)  # (don't do this for spin-up...)
#         Cdd = np.transpose(Cd)

#         # spin operators are defined as  (Cdu, Cdd) S^gamma (Cu, Cd)^T,
#         # where S^gamma is the 2x2 matrix for spin-half
#         Sz = np.diag(0.5 * (Nu_diag - Nd_diag))
#         Sp = np.dot(Cdu, Cd)
#         Sm = np.dot(Cdd, Cu)
#         Sx = 0.5 * (Sp + Sm)
#         Sy = -0.5j * (Sp - Sm)

#         ops = dict(JW=JW, JWu=JWu, JWd=JWd,
#                    Cu=Cu, Cdu=Cdu, Cd=Cd, Cdd=Cdd,
#                    Nu=Nu, Nd=Nd, Ntot=Ntot, dN=dN,
#                    Sx=Sx, Sy=Sy, Sz=Sz, Sp=Sp, Sm=Sm)  # yapf: disable

#         # handle charges
#         qmod = []
#         qnames = []
#         charges = []
#         if cons_N == 'N':
#             qnames.append('N')
#             qmod.append(1)
#             charges.append([0, 1, 1])
#         elif cons_N == 'parity':
#             qnames.append('parity_N')
#             qmod.append(2)
#             charges.append([0, 1, 1])
#         if cons_Sz == 'Sz':
#             qnames.append('2*Sz')  # factor 2 s.t. Cu, Cd have well-defined charges!
#             qmod.append(1)
#             charges.append([0, 1, -1])
#             del ops['Sx']
#             del ops['Sy']
#         elif cons_Sz == 'parity':
#             qnames.append('parity_Sz')  # the charge is (2*Sz) mod (2*2)
#             qmod.append(4)
#             charges.append([0, 1, 3])  # == [0, 1, -1, 0] mod 4
#             # e.g. terms like `Sp_i Sp_j + hc` with Sp=Cdu Cd have charges 'N', 'parity_Sz'.
#             # The `parity_Sz` is non-trivial in this case!
#         if len(qmod) == 0:
#             leg = npc.LegCharge.from_trivial(d)
#         else:
#             if len(qmod) == 1:
#                 charges = charges[0]
#             else:  # len(charges) == 2: need to transpose
#                 charges = [[q1, q2] for q1, q2 in zip(charges[0], charges[1])]
#             chinfo = npc.ChargeInfo(qmod, qnames)
#             leg = npc.LegCharge.from_qflat(chinfo, charges)
#         self.cons_N = cons_N
#         self.cons_Sz = cons_Sz
#         self.filling = filling
#         Site.__init__(self, leg, states, sort_charge=True, **ops)
#         # specify fermionic operators
#         self.need_JW_string |= set(['Cu', 'Cdu', 'Cd', 'Cdd', 'JWu', 'JWd', 'JW'])

#     def __repr__(self):
#         """Debug representation of self."""
#         return "SpinHalfFermionSite({cN!r}, {cS!r}, {f:f})".format(cN=self.cons_N,
#                                                                    cS=self.cons_Sz,
#                                                                    f=self.filling)


# class BosonSite(Site):
#     r"""Create a :class:`Site` for up to `Nmax` bosons.

#     Local states are ``vac, 1, 2, ... , Nc``.
#     (Exception: for parity conservation, we sort as ``vac, 2, 4, ..., 1, 3, 5, ...``.)

#     ==============  ========================================
#     operator        description
#     ==============  ========================================
#     ``Id, JW``      Identity :math:`\mathbb{1}`
#     ``B``           Annihilation operator :math:`b`
#     ``Bd``          Creation operator :math:`b^\dagger`
#     ``N``           Number operator :math:`n= b^\dagger b`
#     ``NN``          :math:`n^2`
#     ``dN``          :math:`\delta n := n - filling`
#     ``dNdN``        :math:`(\delta n)^2`
#     ``P``           Parity :math:`Id - 2 (n \mod 2)`.
#     ==============  ========================================

#     ============== ====  ==================================
#     `conserve`     qmod  *excluded* onsite operators
#     ============== ====  ==================================
#     ``'N'``        [1]   --
#     ``'parity'``   [2]   --
#     ``'None'``     []    --
#     ============== ====  ==================================

#     Parameters
#     ----------
#     Nmax : int
#         Cutoff defining the maximum number of bosons per site.
#         The default ``Nmax=1`` describes hard-core bosons.
#     conserve : str
#         Defines what is conserved, see table above.
#     filling : float
#         Average filling. Used to define ``dN``.

#     Attributes
#     ----------
#     conserve : str
#         Defines what is conserved, see table above.
#     filling : float
#         Average filling. Used to define ``dN``.
#     """

#     def __init__(self, Nmax=1, conserve='N', filling=0.):
#         if not conserve:
#             conserve = 'None'
#         if conserve not in ['N', 'parity', 'None']:
#             raise ValueError("invalid `conserve`: " + repr(conserve))
#         dim = Nmax + 1
#         states = [str(n) for n in range(0, dim)]
#         if dim < 2:
#             raise ValueError("local dimension should be larger than 1....")
#         B = np.zeros([dim, dim], dtype=np.float64)  # destruction/annihilation operator
#         for n in range(1, dim):
#             B[n - 1, n] = np.sqrt(n)
#         Bd = np.transpose(B)  # .conj() wouldn't do anything
#         # Note: np.dot(Bd, B) has numerical roundoff errors of eps~=4.4e-16.
#         Ndiag = np.arange(dim, dtype=np.float64)
#         N = np.diag(Ndiag)
#         NN = np.diag(Ndiag**2)
#         dN = np.diag(Ndiag - filling)
#         dNdN = np.diag((Ndiag - filling)**2)
#         P = np.diag(1. - 2. * np.mod(Ndiag, 2))
#         ops = dict(B=B, Bd=Bd, N=N, NN=NN, dN=dN, dNdN=dNdN, P=P)
#         if conserve == 'N':
#             chinfo = npc.ChargeInfo([1], ['N'])
#             leg = npc.LegCharge.from_qflat(chinfo, range(dim))
#         elif conserve == 'parity':
#             chinfo = npc.ChargeInfo([2], ['parity_N'])
#             leg = npc.LegCharge.from_qflat(chinfo, [i % 2 for i in range(dim)])
#         else:
#             leg = npc.LegCharge.from_trivial(dim)
#         self.Nmax = Nmax
#         self.conserve = conserve
#         self.filling = filling
#         Site.__init__(self, leg, states, sort_charge=True, **ops)
#         self.state_labels['vac'] = self.state_labels['0']  # alias

#     def __repr__(self):
#         """Debug representation of self."""
#         return "BosonSite({N:d}, {c!r}, {f:f})".format(N=self.Nmax,
#                                                        c=self.conserve,
#                                                        f=self.filling)


# def spin_half_species(SpeciesSite, cons_N, cons_Sz, **kwargs):
#     """Initialize two FermionSite to represent spin-1/2 species.

#     You can use this directly in the :meth:`tenpy.models.model.CouplingMPOModel.init_sites`,
#     e.g., as in the :meth:`tenpy.models.hubbard.FermiHubbardModel2.init_sites`::

#         cons_N = model_params.get('cons_N', 'N')
#         cons_Sz = model_params.get('cons_Sz', 'Sz')
#         return spin_half_species(FermionSite, cons_N=cons_N, cons_Sz=cons_Sz)

#     Parameters
#     ----------
#     SpeciesSite : :class:`Site` | str
#         The (name of the) site class for the species;
#         usually just :class:`FermionSite`.
#     cons_N : None | ``"N", "parity", "None"``
#         Whether to conserve the (parity of the) total particle number ``N_up + N_down``.
#     cons_Sz : None | ``"Sz", "parity", "None"``
#         Whether to conserve the (parity of the) total Sz spin ``N_up - N_down``.

#     Returns
#     -------
#     sites : list of `SpeciesSite`
#         Each one instance of the site for spin up and down.
#     species_names : list of str
#         Always ``['up', 'down']``. Included such that a ``return spin_half_species(...)``
#         in :meth:`~tenpy.models.model.CouplingMPOModel.init_sites` triggers the use of the
#         :class:`~tenpy.models.lattice.MultiSpeciesLattice`.
#     """
#     SpeciesSite = find_subclass(Site, SpeciesSite)
#     if not cons_N:
#         cons_N = 'None'
#     if cons_N not in ['N', 'parity', 'None']:
#         raise ValueError("invalid `cons_N`: " + repr(cons_N))
#     if not cons_Sz:
#         cons_Sz = 'None'
#     if cons_Sz not in ['Sz', 'parity', 'None']:
#         raise ValueError("invalid `cons_Sz`: " + repr(cons_Sz))

#     conserve = None if cons_N == 'None' and cons_Sz == 'None' else 'N'

#     up_site = SpeciesSite(conserve=conserve, **kwargs)
#     down_site = SpeciesSite(conserve=conserve, **kwargs)

#     new_charges = []
#     new_names = []
#     new_mod = []
#     if cons_N == 'N':
#         new_charges.append([(1, 0, 0), (1, 1, 0)])
#         new_names.append('N')
#         new_mod.append(1)
#     elif cons_N == 'parity':
#         new_charges.append([(1, 0, 0), (1, 1, 0)])
#         new_names.append('parity_N')
#         new_mod.append(2)
#     if cons_Sz == 'Sz':
#         new_charges.append([(1, 0, 0), (-1, 1, 0)])
#         new_names.append('2*Sz')  # factor 2 s.t. Cu, Cd have well-defined charges!
#         new_mod.append(1)
#     elif cons_Sz == 'parity':
#         new_charges.append([(1, 0, 0), (-1, 1, 0)])
#         new_names.append('2*Sz')  # factor 2 s.t. Cu, Cd have well-defined charges!
#         new_mod.append(4)
#     set_common_charges([up_site, down_site], new_charges, new_names, new_mod)
#     return [up_site, down_site], ['up', 'down']


class ClockSite(Site):
    r"""Quantum clock site.

    There are ``q`` local states, with labels ``['0', '1', ..., str(q-1)]``.
    Special aliases are ``up`` (0), and if q is even ``down`` (q / 2).

    ==============  =================  ============  =========================
    `conserve`      symmetry           sectors       meaning of sector label
    ==============  =================  ============  =========================
    ``'Z'``         ZNSymmetry(N=q)    ``range(q)``  sector n has Z = w ** n
    ``'None'``      NoSymmetry         ``[0, ...]``  --
    ==============  =================  ============  =========================

    Local operators are the clock operators ``Z = diag([w ** 0, w ** 1, ..., w ** (q - 1)])``
    with ``w = exp(2.j * pi / q)`` and ``X = eye(q, k=1) + eye(q, k=1-q)``, which are not hermitian.

    ====================  =====================================  ====  ======
    operator              description                            Z     None
    ====================  =====================================  ====  ======
    ``Id, JW``            Identity :math:`\mathbb{1}`            sym   sym
    ``Z, Zhc``            Clock operator Z & its conjugate       sym   sym
    ``Zphc``              "Real part" :math:`Z + Z^\dagger`      sym   sym
    ``X, Xhc``            Clock operator X & its conjugate       gen   sym
    ``Xphc``              "Real part" :math:`X + X^\dagger`      gen!  sym
    ====================  =====================================  ====  ======

    Parameters
    ----------
    q : int
        Number of states per site
    conserve : str
        Defines what is conserved, see table above.
    backend : :class:`~tenpy.linalg.backends.AbstractBackend`, optional
        The backend used to create the operators.
    """
    def __init__(self, q, conserve='Z', backend: AbstractBackend = None):
        if not (isinstance(q, int) and q > 1):
            raise ValueError(f'invalid q: {q}')
        self.q = q
        assert conserve in ['Z', 'None']
        self.conserve = conserve
        if conserve == 'Z':
            leg = VectorSpace.from_basis(ZNSymmetry(q, 'clock_phase'), np.arange(q)[:, None])
        else:
            leg = VectorSpace.from_trivial_sector(q)
        X = np.eye(q, k=1) + np.eye(q, k=1-q)
        Z = np.diag(np.exp(2.j * np.pi * np.arange(q, dtype=np.complex128) / q))
        Xhc = X.conj().transpose()
        Zhc = Z.conj().transpose()
        Zphc = np.diag(2. * np.cos(2. * np.pi * np.arange(q, dtype=np.complex128) / q))
        ops = dict(X=X, Z=Z, Xhc=Xhc, Zhc=Zhc, Zphc=Zphc)
        if conserve == 'Z':
            pass  # TODO add Xphc as gen!
        else:
            ops['Xphc'] = X + Xhc
        names = [str(m) for m in range(q)]
        Site.__init__(self, leg=leg, backend=backend, state_labels=names, **ops)
        self.state_labels['up'] = self.state_labels['0']
        if q % 2 == 0:
            self.state_labels['down'] = self.state_labels[str(q // 2)]

    def __repr__(self):
        return f'ClockSite(q={self.q}, conserve={self.conserve})'
