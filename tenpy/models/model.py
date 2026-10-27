"""TODO elaborate

.. _couplings:

Couplings in Models
-------------------
TODO elaborate on the system of couplings

"""
# Copyright (C) TeNPy Developers, Apache license

import itertools
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Literal

import lasyten as lasy
import numpy as np
from numpy.typing import ArrayLike

from ..networks import MPO, MPOGraph
from ..networks.mpo import partition_W
from ..networks.terms import CouplingTerms, ExponentiallyDecayingTerms, MultiCouplingTerms, OnsiteTerms, Terms
from ..tools import to_array
from .lattice import Lattice

__all__ = [
    'EvolutionModel',
    'RandomUnitaryEvolution',
    'HamiltonianModel',
    'HamiltonianMPOModel',
    'CouplingModel',
    'Model',
    'NearestNeighborModel',
]


# TODO placeholders
class Model:
    pass


class NearestNeighborModel(Model):
    pass


class MPOModel(Model):
    pass


class CouplingMPOModel(MPOModel):
    def __init__(self, params):
        self.params = params


class EvolutionModel(metaclass=ABCMeta):
    """Base class for models that define time evolution.

    A time evolution operator is defined in subclasses via :meth:`get_U_bond` and/or :meth:`get_U_MPO`.

    TODO this interface would also allow us to move time-dependent H treatment from algorithm to model

    Attributes
    ----------
    lattice: :class:`~tenpy.Lattice`
        Note that the lattice sets the symmetry and backend to be used.
    symmetry: :class:`lasyten.Symmetry`
        The symmetry for all tensors to be used.
    backend: :class:`lasyten.TensorBackend`
        The backend for all tensors to be used
    model_params: Config
        The model parameters

    """

    def __init__(self, lattice: Lattice):
        self.lattice = lattice
        self.symmetry = lattice.symmetry
        self.backend = lattice.backend
        self.bc_MPS = lattice.bc_MPS

    def test_sanity(self):
        self.lattice.test_sanity()

    @property
    def finite(self):
        return self.bc_MPS == 'finite'

    def invalidate_caches(self):
        """Invalidate any cached representations of the model, e.g. stored MPOs.

        This hook is called when the model is modified in-place.
        """
        pass

    @abstractmethod
    def get_U_bond(self, t: float, dt: float, Nsteps: int = 1) -> list[tuple[int, lasy.Tensor]]:
        """A time step as NN circuit, possibly up to approximations from discretization.

        TODO where to pass trotter options?

        Returns
        -------
        gates: list of (int, :class:`lasyten.Tensor`)
            Entries ``i, U`` indicate to apply gate ``U`` on MPS sites ``i, i + 1``.
            TODO is this a good format?

        """
        ...

    @abstractmethod
    def get_U_MPO(self, t: float, dt: float) -> MPO:
        """A time step as an MPO, possibly up to approximations from discretization."""
        ...


class RandomUnitaryEvolution(EvolutionModel):
    """TODO write docs

    Todo:
    - test coverage (properties, make sure all U are different!)
    - get rid of RandomUnitaryEvolution in tebd.py, replace with this model

    """

    def __init__(self, lattice, model_params):
        raise NotImplementedError

    def test_sanity(self):
        return super().test_sanity()


class HamiltonianModel(EvolutionModel, metaclass=ABCMeta):
    """Base class for models that define a Hamiltonian.

    The Hamiltonian is defined in subclasses via :meth:`get_H_bond` and/or :meth:`get_H_MPO`.
    """

    def __init__(self, lattice: Lattice):
        EvolutionModel.__init__(self, lattice=lattice)

    @abstractmethod
    def get_H_bond(self) -> list[lasy.Tensor | None]:
        """The Hamiltonian as a sum of bond-terms.

        Returns
        -------
        H_bond: list of :class:`lasyten.Tensor`
            The two-site bond terms such that ``H = sum_i H_bond[i]``.
            ``H_bond[i]`` acts in sites ``(i - 1, i)``.
            For finite systems, we have ``H_bond[0] = None``.
            Legs of each ``H_bond[i]`` are ``[p0, p1, p1*, p0*]``.

        """
        raise NotImplementedError(f'{type(self).__name__} does not support H_bond format.')

    @abstractmethod
    def get_H_MPO(self) -> MPO:
        """The Hamiltonian as an MPO"""
        ...

    def get_U_bond(self, t: float, dt: float, Nsteps: int = 1) -> list[tuple[int, lasy.Tensor]]:
        # TODO move code from TEBDEngine here
        #      probably want caching, but per instance and per discretization params!
        #      -> make sure to use invalidate_caches!
        raise NotImplementedError

    def get_U_MPO(self, t, dt):
        # FIXME read approximation from params?
        return self.get_H_MPO().make_U(dt, approximation='II')


class HamiltonianMPOModel(HamiltonianModel):
    """A model where the Hamiltonian is defined directly in terms of an MPO.

    This can be used when the Hamiltonian comes e.g. from compression or from other software.
    Note that some MPO features, like e.g. ``WI`` or ``WII`` exponentiation are not available for
    generic MPOs, as they need the finite state machine structure that is e.g. contructed in
    a :class:`CouplingModel`
    """

    def __init__(self, lattice: Lattice, H_MPO: MPO):
        self._H_MPO = H_MPO
        self._H_bond = None
        HamiltonianModel.__init__(self, lattice=lattice)

    def test_sanity(self):
        self._H_MPO.test_sanity()
        return super().test_sanity()

    def invalidate_caches(self):
        self._H_bond = None
        super().invalidate_caches()

    def get_H_MPO(self):
        return self._H_MPO

    def get_H_bond(self):
        if self._H_bond is None:
            self._H_bond = calc_H_bond_from_MPO(self._H_MPO)
        return self._H_bond

    def get_U_MPO(self):
        raise NotImplementedError('Can not exponentiate generic MPOs.')


class CouplingDefinition(metaclass=ABCMeta):
    """FIXME"""

    def __init__(
        self,
        backend: lasy.TensorBackend,
        symmetry: lasy.Symmetry,
        coupling_names: Sequence[str] = [],
        num_sites: dict[str, int] = {},
        hc_map: dict[str, str] = {},
    ):
        self.backend = backend
        self.symmetry = symmetry
        self.coupling_names = coupling_names = set(coupling_names)
        self.num_sites = num_sites.copy()
        self.hc_map = hc_map.copy()

    def test_sanity(self):
        assert set(self.num_sites.keys()) == set(self.hc_map.keys()) == self.coupling_names
        assert set(self.hc_map.values()).issubset(self.coupling_names)
        for op, hc in self.hc_map.items():
            assert self.hc_map[hc] == op
        # FIXME in tests: systematically check that the hc is correct

    @abstractmethod
    def decompose(self, name: str, sites: Sequence[lasy.Site]) -> list[tuple[complex, list[lasy.SymmetricTensor]]]:
        """FIXME"""
        ...

    @abstractmethod
    def permute(self, name: str, perm: Sequence[int]) -> dict[str, float | complex]: ...

    @abstractmethod
    def get_coupling(self, name: str, sites: Sequence[lasy.Site]) -> lasy.Coupling: ...

    def test_valid(self, name: str, sites: Sequence[lasy.Site]):
        """FIXME"""
        if name not in self.coupling_names:
            raise ValueError(f'Unknonwn coupling name: {name}')


class OnsiteCouplingDefinition(CouplingDefinition):
    """FIXME"""

    def __init__(self, sites: Sequence[lasy.Site], backend: lasy.TensorBackend, symmetry: lasy.Symmetry):
        self.sites = sites  # FIXME deal with duplicates?
        hc_map = {}
        for s in sites:
            for name in s.onsite_operators:
                # FIXME should add hc opname mechanism to cyten.Site!
                hc = name
                assert hc_map.get(name, hc) == hc
                hc_map[name] = hc
        num_sites = {k: 1 for k in hc_map}
        super().__init__(backend, symmetry, coupling_names=hc_map.keys(), num_sites=num_sites, hc_map=hc_map)

    def decompose(self, name, sites):
        return [(1.0, self.get_coupling(name, sites).factorization)]

    def permute(self, name, perm):
        return {name: 1.0}

    def get_coupling(self, name, sites):
        assert len(sites) == 1
        h = sites[0].get_op(name)
        return lasy.Coupling.from_tensor(h, sites, name=name)

    def test_valid(self, name, sites):
        assert len(sites) == 1
        assert name in sites[0].onsite_operators
        return super().test_valid(name, sites)


# FIXME move
# FIXME what is a good place to document the meaning? docstring? or dynamically via method?
class SpinCouplingDefinition(CouplingDefinition):
    """FIXME"""

    def __init__(self, backend: lasy.TensorBackend, symmetry: lasy.Symmetry):
        tmp = {
            'Sx.Sx': (2, 'Sx.Sx'),
            'Sy.Sy': (2, 'Sy.Sy'),
            'Sz.Sz': (2, 'Sz.Sz'),
            'S.S': (2, 'S.S'),
            'Sp.Sm': (2, 'Sm.Sp'),
            'Sm.Sp': (2, 'Sp.Sm'),
            'chiral3spin': (3, 'chiral3spin'),
        }
        num_sites = {k: num for k, (num, hc) in tmp.items()}
        hc_map = {k: hc for k, (num, hc) in tmp.items()}
        super().__init__(backend, symmetry, coupling_names=tmp.keys(), num_sites=num_sites, hc_map=hc_map)

    def get_spin_conserve(self, sites: list[lasy.Site]) -> Literal['SU(2)', 'Sz', 'parity', None]:
        if not isinstance(sites[0], lasy.models.SpinSite):
            raise NotImplementedError
        res = sites[0].conserve
        if any(s.conserve != res for s in sites[1:]):
            raise lasy.SymmetryError('Inconsistent conserve for sites!')
        return res

    def decompose(self, name, sites):
        conserve = self.get_spin_conserve(sites)
        if name in ['Sx.Sx', 'Sy.Sy', 'Sz.Sz', 'Sp.Sm', 'Sm.Sp']:
            if conserve == 'SU(2)':
                raise lasy.SymmetryError(f'Coupling not SU(2) symmetric: {name}')
            return {name: +1}
        if name == 'S.S':
            if conserve == 'SU(2)':
                return {name: +1}
            return {'Sx.Sx': 1, 'Sy.Sy': 1, 'Sz.Sz': 1}
        if name == 'chiral3spin':
            raise NotImplementedError
        raise ValueError(f'Unknown name: {name}')

    def get_coupling(self, name, sites):
        if name == 'Sx.Sx':
            return lasy.models.spin_spin_coupling(sites, Jx=1, backend=self.backend)
        if name == 'Sy.Sy':
            return lasy.models.spin_spin_coupling(sites, Jy=1, backend=self.backend)
        if name == 'Sz.Sz':
            return lasy.models.spin_spin_coupling(sites, Jz=1, backend=self.backend)
        if name == 'S.S':
            return lasy.models.spin_spin_coupling(sites, Jx=1, Jy=1, Jz=1, backend=self.backend)
        if name == 'Sp.Sm':
            raise NotImplementedError
        if name == 'Sm.Sp':
            raise NotImplementedError
        if name == 'chiral3spin':
            raise NotImplementedError
        raise ValueError(f'Unknown name: {name}')

    def permute(self, name, perm):
        if name in ['Sx.Sx', 'Sy.Sy', 'Sz.Sz', 'S.S']:
            return {name: 1}
        perm = np.asarray(perm, int)
        if np.all(perm == np.arange(len(perm))):
            return {name: 1}
        if name == 'Sm.Sp':
            return {'Sp.Sm': 1}  # trivial permutation already excluded -> must be swap
        if name == 'Sp.Sm':
            return {'Sm.Sp': 1}  # trivial permutation already excluded -> must be swap
        if name == 'chiral3spin':
            if np.all(perm == np.array([1, 2, 0])) or np.all(perm == np.array([2, 0, 1])):
                return {name: +1}
            else:
                return {name: -1}
        raise ValueError(f'Unknown name: {name}')

    def test_valid(self, name, sites):
        super().test_valid(name, sites)
        _ = self.get_spin_conserve(sites)  # <- does all checks on sites (are spin, are consistent)


class CouplingModel(HamiltonianModel):
    """A model, which defines a Hamiltonian in terms of couplings.

    TODO elaborate
    - statefulness, point to methods

    Parameters
    ----------
    lattice
        TODO
    coupling_definitions
        TODO
        Note that :class:`OnsiteCouplingDefinition` is automatically added.
    explicit_plus_hc
        TODO

    Attributes
    ----------
    explicit_plus_hc : bool
        If `True`, `self` represents the terms explicitly stored in e.g. :attr:`onsite_terms`
        and :attr:`coupling_terms` and so on, *plus* their hermitian conjugate added.
        The flag will be carried on to the MPO, which will have a reduced bond dimension if
        ``self.add_*(..., plus_hc=True)`` was used.
        Note that :meth:`add_onsite_operator`, :meth:`add_two_site_coupling`, :meth:`add_multi_site_coupling`
        and :meth:`add_exponentially_decaying_coupling` respect this flag, ensuring that the
        *represented* Hamiltonian is independent of the `explicit_plus_hc` flag.

    """

    def __init__(
        self,
        lattice: Lattice,
        coupling_definitions: Sequence[CouplingDefinition | type[CouplingDefinition]],
        explicit_plus_hc: bool = False,
    ):
        self._H_MPO: MPO | None = None
        self._L_mps = L = len(lattice.mps_sites())
        self.onsite_terms = OnsiteTerms(L)
        self.coupling_terms = CouplingTerms(L)
        self.multi_coupling_terms = MultiCouplingTerms(L)
        self.exp_decaying_terms = ExponentiallyDecayingTerms(L)
        self.explicit_plus_hc = explicit_plus_hc
        onsite_definitions = OnsiteCouplingDefinition(
            sites=lattice.unit_cell, backend=lattice.backend, symmetry=lattice.symmetry
        )
        self.coupling_definitions = [
            d if isinstance(d, CouplingDefinition) else d(backend=lattice.backend, symmetry=lattice.symmetry)
            for d in [onsite_definitions, *coupling_definitions]
        ]
        HamiltonianModel.__init__(self, lattice=lattice)

    def test_sanity(self):
        if self._H_MPO is not None:
            self._H_MPO.test_sanity()
        sites = self.lattice.mps_sites()
        self.onsite_terms.test_terms(sites, self)
        self.coupling_terms.test_terms(sites, self)
        self.multi_coupling_terms.test_terms(sites, self)
        self.exp_decaying_terms.test_terms(sites, self)

        # check coupling_definitions
        seen_names = []
        for d in self.coupling_definitions:
            d.test_sanity()
            assert all(n not in seen_names for n in d.coupling_names)
            seen_names.extend(d.coupling_names)

        return super().test_sanity()

    def coupling_num_sites(self, coupling_name: str) -> int:
        for d in self.coupling_definitions:
            if coupling_name not in d.coupling_names:
                continue
            return d.num_sites[coupling_name]
        raise ValueError(f'Unknown coupling name {coupling_name}')

    def decompose_coupling(
        self, coupling_name: str, sites: Sequence[lasy.Site]
    ) -> list[tuple[complex, list[lasy.SymmetricTensor]]]:
        for d in self.coupling_definitions:
            if coupling_name not in d.coupling_names:
                continue
            return d.decompose(coupling_name, sites)
        raise ValueError(f'Unknown coupling name {coupling_name}')

    def permute_coupling(self, coupling_name: str, perm: Sequence[int]) -> dict[str, float | complex]:
        for d in self.coupling_definitions:
            if coupling_name not in d.coupling_names:
                continue
            return d.permute(coupling_name, perm)
        raise ValueError(f'Unknown coupling name {coupling_name}')

    def _get_switch_LR(self, name: str, idcs: Sequence[int]) -> int:
        assert self.coupling_n_sites(name) == len(idcs)
        return len(idcs) - len(idcs) // 2

    def get_coupling(self, coupling_name: str, sites: Sequence[lasy.Site]) -> lasy.Coupling:
        for d in self.coupling_definitions:
            if coupling_name not in d.coupling_names:
                continue
            return d.get_coupling(coupling_name, sites)
        raise ValueError(f'Unknown coupling name {coupling_name}')

    def test_coupling(self, coupling_name: str, sites: Sequence[lasy.Site]):
        """FIXME"""
        for d in self.coupling_definitions:
            if coupling_name not in d.coupling_names:
                continue
            return d.test_valid(coupling_name, sites)
        raise ValueError(f'Unknown coupling name {coupling_name}')

    def invalidate_caches(self):
        self._H_MPO = None
        self._H_bond = None
        super().invalidate_caches()

    def get_nonzero_terms(self, tol_zero=1e-15) -> list[Terms]:
        return [
            self.onsite_terms.remove_zeros(tol_zero),
            self.coupling_terms.remove_zeros(tol_zero),
            self.multi_coupling_terms.remove_zeros(tol_zero),
            # self.exp_decaying_terms.remove_zeros(tol_zero),  # FIXME reactivate
        ]

    def get_H_MPO(self, tol_zero=1e-15):
        if self._H_MPO is None:
            terms = self.get_nonzero_terms(tol_zero=tol_zero)
            max_range = max(t.max_range() for t in terms)
            graph = MPOGraph.from_terms(
                terms,
                sites=self.lattice.mps_sites(),
                bc=self.bc_MPS,
                insert_all_id=True,
                unit_cell_width=self.lattice.mps_unit_cell_width,
            )
            # TODO optionally optimize basis here?
            self._H_MPO = graph.build_MPO(max_range=max_range, explicit_plus_hc=self.explicit_plus_hc)
        return self._H_MPO

    def get_H_bond(self):
        if self._H_bond is None:
            self._H_bond = calc_H_bond_from_MPO(self._H_MPO)
        return self._H_bond

    def add_onsite_operator(
        self,
        strength: ArrayLike,
        op: str,
        x: ArrayLike | Literal['sum_over'] = 'sum_over',
        u: int | Literal['sum_over'] = 'sum_over',
        plus_hc: bool = False,
    ):
        r"""Add one ore more onsite terms.

        Adds terms of the form

        .. math ::
            \sum_{x \in \mathtt{x}} \sum_{u \in \mathtt{u}} \lambda_x \mathtt{op}_x

        where the sums go over all lattice sites by default, but may be restricted
        by specifying `x` and `u`.

        Parameters
        ----------
        strength : array-like
            Prefactors for the terms, given in one of the following forms:

            * A single entry (0D array-like) applies to all onsite operators added.
            * If ``x='sum_over'``, ``strength`` may be a ``lattice.dim``-D array-like which is tiled
                to shape ``lattice.Ls``.
        op : str
            Specifies the operator to add. Must be a valid operator name on all sites acted on.
        x : 1D array-like | ``'sum_over'``
            Specifies for which lattice indices to add the `op`.

            * A 1D array-like requires a finite ``len(x)``-D lattice and specifies a single index.
            * ``'sum_over'`` adds an extensive sum of onsite terms. Note that `strength` can be
              used for non-uniform prefactors

        u : int | ``'sum_over'``
            Specifies which site(s) within a unit cell to add the `op` to:

            * A single ``int`` specifies a single site.
            * ``'sum_over'`` adds an onsite term to all of them. Note that `strength` applies to
              all of them the same. For different strengths per `u`, call this method multiple times.
        plus_hc : bool
            If `True`, the hermitian conjugate of the terms is added automatically.

        See Also
        --------
        add_two_site_coupling, add_multi_site_coupling

        """
        if u == 'sum_over':
            for u in range(len(self.lattice.unit_cell)):
                self.add_onsite_operator(strength, op, x=x, u=u, plus_hc=plus_hc)
            return
        u = lasy.tools.to_valid_idx(u, len(self.lattice.unit_cell))

        if self.explicit_plus_hc:
            if plus_hc:
                plus_hc = False  # explicitly add the h.c. later; don't do it here.
            else:
                strength = strength / 2  # avoid double-counting this term: add the h.c. explicitly later on

        if x == 'sum_over':
            strength = to_array(strength, self.lattice.Ls)
            # FIXME generally purge ``.lat
            for i, i_lat in zip(*self.lattice.mps_lat_idx_fix_u(u)):
                self.onsite_terms.add_onsite_term(strength[*i_lat], i, op)
        else:
            if not self.finite:
                raise ValueError('Can not add single onsite operator to infinite system.')
            x = np.asarray(x, int).reshape(-1)
            i = self.lattice.lat2mps_idx([*x, u])
            self.onsite_terms.add_onsite_term(strength, i, op)

        if plus_hc:
            hc_op = self.lattice.unit_cell[u].get_hc_op_name(op)
            self.add_onsite_operator(np.conj(strength), hc_op, x=x, u=u, plus_hc=False)

    def add_two_site_coupling(
        self,
        strength: ArrayLike,
        coupling_name: str,
        dx: ArrayLike,
        x1: ArrayLike | Literal['sum_over'] = 'sum_over',
        u1: int | Literal['sum_over_independent'] | Literal['sum_over_matching'] = 'sum_over_matching',
        u2: int | Literal['sum_over_independent'] | Literal['sum_over_matching'] = 'sum_over_matching',
        plus_hc: bool = False,
    ):
        r"""Add one or more two-site coupling terms.

        Adds terms of the form

        .. math ::
            \sum_{x_1} \sum_{u_1, u_2} \lambda_{x_1} h_{(x_1, u_1) ; (x_1 + dx, u_2)}

        where the sums go over all lattice sites ``x_1`` and all matching ``u_1 = u_2`` by default,
        but may be modified (even collapsed to a single term) by specifying `x1`, `u1`, `u2`.

        Parameters
        ----------
        strength : array-like
            Prefactors for the terms, given in one of the following forms:

            * A single entry (0D array-like) applies to all onsite operators added.
            * If ``x1='sum_over'``, ``strength`` may be a ``lattice.dim``-D array-like which is tiled
                to shape given by :meth:`Lattice.coupling_shape`, which is mostly given by the
                ``lattice.Ls`` and is reduced for open bc.
        coupling_name : str
            Specifies the operator to add by name, see :ref:`couplings`.
        dx : 1D array-like
            The translation vector from the first to the second site to act on.
        x1 : 1D array-like | ``'sum_over'``
            Specifies the position(s) for the first site of the coupling.

            * A 1D array-like requires a finite ``len(x)``-D lattice and specifies a single position.
            * ``'sum_over'`` adds an extensive sum of coupling terms. Note that `strength` can be
                used for non-uniform prefactors

        u1, u2 : int | ``'sum_over_matching'`` | ``'sum_over_independent'``
            Specifies which site(s) within a unit cell to act on.

            * ``'sum_over_matching'`` must be given for both `u1` and `u2`, specifies to add the
              term for all matching values ``u1 == u2``.
              Note that `strength` applies to all equally. Use multiple calls for `u`-dependent strength.
            * ``'sum_over_independent'`` must be given for both `u1` and `u2`, specifies to add the
              term for all possible combinations ``u1, u2``.
              Note that `strength` applies to all equally. Use multiple calls for `u`-dependent strength.
            * A single ``int`` specifies a single site.
        plus_hc : bool
            If `True`, the hermitian conjugate of the terms is added automatically.

        See Also
        --------
        add_onsite_operator, add_multi_site_coupling

        """
        if u1 == u2 == 'sum_over_independent':
            for u1 in range(len(self.lattice.unit_cell)):
                for u2 in range(len(self.lattice.unit_cell)):
                    self.add_two_site_coupling(strength, coupling_name, x1=x1, u1=u1, u2=u2, dx=dx, plus_hc=plus_hc)
            return
        elif u1 == u2 == 'sum_over_matching':
            for u in range(len(self.lattice.unit_cell)):
                self.add_two_site_coupling(strength, coupling_name, x1=x1, u1=u, u2=u, dx=dx, plus_hc=plus_hc)
            return
        elif isinstance(u1, str) or isinstance(u2, str):
            raise ValueError(
                'Invalid u1 or u2. If string, must both be the same choice of sum_over_independent or sum_over_matching'
            )
        u1 = lasy.tools.to_valid_idx(u1, len(self.lattice.unit_cell))
        u2 = lasy.tools.to_valid_idx(u2, len(self.lattice.unit_cell))

        if self.explicit_plus_hc:
            if plus_hc:
                plus_hc = False  # explicitly add the h.c. later; don't do it here.
            else:
                strength = strength / 2  # avoid double-counting this term: add the h.c. explicitly later on

        dx = np.array(dx, int).reshape([self.lattice.dim])
        if x1 == 'sum_over':
            for i, j, pref in zip(*self.lattice.possible_couplings_strengths(u1, u2, dx, strength)):
                self._add_two_site_coupling_mps(pref, i, j, coupling_name, plus_hc=plus_hc)
        else:
            if not self.finite:
                raise ValueError('Can not add single coupling to infinite system.')
            x1 = np.asarray(x1, int).reshape([self.lattice.dim])
            x2 = x1 + dx
            i = self.lattice.lat2mps_idx([*x1, u1])
            j = self.lattice.lat2mps_idx([*x2, u2])
            self._add_two_site_coupling_mps(strength, i, j, coupling_name, plus_hc=plus_hc)

    def _add_two_site_coupling_mps(
        self, pref: float | complex, i: int, j: int, coupling_name: str, plus_hc: bool = False
    ):
        if plus_hc:
            hc_name = self.get_hc_coupling_name(coupling_name)
            self._add_two_site_coupling_mps(np.conj(pref), i, j, hc_name, plus_hc=False)

        if i == j:
            raise ValueError('Two-site coupling must be on different sites!')
        if i < j:
            return self.coupling_terms.add_coupling_term(pref, i, j, coupling_name)

        for c, a in self.permute_coupling(coupling_name, [1, 0]).items():
            return self.coupling_terms.add_coupling_term(a * pref, j, i, c)
        return

    def add_multi_site_coupling(
        self,
        strength: ArrayLike,
        coupling_name: str,
        dx: ArrayLike,
        x: int | Sequence[int] | Literal['sum_over'] = 'sum_over',
        u: Sequence[int] | Literal['sum_over_independent'] | Literal['sum_over_matching'] = 'sum_over_matching',
        plus_hc: bool = False,
    ):
        r"""Add one or more multi-site coupling terms.

        Adds terms of the form

        .. math ::
            \sum_x \sum_{u_1, \dots, u_M} \lambda_x h_{(x + dx_1, u_1), \dots, (x + dx_M, u_M)}

        where the sums go over all lattice sites ``x`` and all matching ``u_1 = u_2 = ... = u_M``
        by default but may be modified (even collapsed to a single term) by specifying `x`, `u`.

        Parameters
        ----------
        strength : array-like
            Prefactors :math:`\lambda` for the terms.
            If ``x='sum_over'``, non-uniform strengths can be specified as a ``lattice.dim``-D array-like,
            which is tiled to the correct shape, see :meth:`Lattice.multi_coupling_shape`.
        coupling_name : str
            Specifies the operator to add by name, see :ref:`couplings`.
        dx : 2D array-like
            Shape ``(M, D)``. For each of the ``M`` sites acted on, a ``D = lattice.dim``
            dimensional translation vector, specifying the position of the site relative to `x`.
        x : 1D array-like | ``'sum_over'``
            Reference position for the sites acted on, `dx` is relative to this.

            * A 1D array-like requires a finite ``len(x)``-D lattice and specifies a single position.
            * ``'sum_over'`` adds an extensive sum of coupling terms. Note that `strength` can be
                used for non-uniform prefactors
        u : 1D array-like | ``'sum_over_independent'`` | ``'sum_over_matching'``
            Specifies which site(s) within a unit cell to act on.

            * ``'sum_over_matching'`` specifies to add the term for all matching values ``u1 == u2 == ... == uM``.
                Note that `strength` applies to all equally. Use multiple calls for `u`-dependent strength.
            * ``'sum_over_independent'`` specifies to add the term for all possible combinations ``u1, u2, ..., uM``.
                Note that `strength` applies to all equally. Use multiple calls for `u`-dependent strength.
            * A 1D array-like specifies a single unit-cell index for each site acted on.
        plus_hc : bool
            If `True`, the hermitian conjugate of the terms is added automatically.

        See Also
        --------
        add_onsite_operator, add_two_site_coupling

        """
        N = self.coupling_num_sites(coupling_name)
        if u == 'sum_over_independent':
            for _u in itertools.product(range(len(self.lattice.unit_cell)), repeat=N):
                self.add_multi_site_coupling(strength, coupling_name, dx, x, _u, plus_hc=plus_hc)
            return
        if u == 'sum_over_matching':
            for _u in range(len(self.lattice.unit_cell)):
                self.add_multi_site_coupling(strength, coupling_name, dx, x, [_u] * N, plus_hc=plus_hc)
            return
        u = np.array(u, int).reshape([N])
        dx = np.array(dx, int).reshape([N, self.lattice.dim])

        if self.explicit_plus_hc:
            if plus_hc:
                plus_hc = False  # explicitly add the h.c. later; don't do it here.
            else:
                strength = strength / 2  # avoid double-counting this term: add the h.c. explicitly later on

        if x == 'sum_over':
            mps_idcs, strength_vals = self.lattice.possible_multi_couplings_strengths(dx, u, strength)
            for ijkl, pref in zip(mps_idcs, strength_vals):
                self._add_multi_coupling_mps(pref, ijkl, coupling_name, plus_hc=plus_hc)
        else:
            if not self.finite:
                raise ValueError('Can not add single coupling to infinite system.')
            x = np.asarray(x, int).reshape([1, self.lattice.dim]) + dx
            lat_idcs = np.concatenate(x, u[:, None], axis=1)
            mps_idcs = self.lattice.lat2mps_idx(lat_idcs)
            self._add_multi_coupling_mps(strength, mps_idcs, coupling_name, plus_hc=plus_hc)

    def _add_multi_coupling_mps(
        self, pref: float | complex, ijkl: Sequence[int], coupling_name: str, plus_hc: bool = False
    ):
        if plus_hc:
            hc_name = self.get_hc_coupling_name(coupling_name)
            self._add_multi_coupling_mps(np.conj(pref), ijkl, hc_name, plus_hc=False)

        if lasy.tools.duplicate_entries(ijkl):
            raise ValueError('Can not act on the same site twice!')
        perm = np.argsort(ijkl)
        if np.all(perm == np.arange(len(ijkl))):
            return self.multi_coupling_terms.add_multi_coupling_term(pref, ijkl, coupling_name)
        ijkl_sorted = ijkl[perm]
        for c, a in self.permute_coupling(coupling_name, perm).items():
            self.multi_coupling_terms.add_multi_coupling_term(a * pref, ijkl_sorted, c)

    def add_exponentially_decaying_coupling(self):
        raise NotImplementedError  # TODO skipped for now

    def add_exponentially_decaying_centered_terms(self):
        raise NotImplementedError  # TODO skipped for now


def calc_H_bond_from_MPO(H_MPO: MPO, tol_zero: float = 1e-15) -> list[lasy.Tensor]:
    """Calculate the bond Hamiltonian from the MPO Hamiltonian.

    Parameters
    ----------
    H_MPO : :class:`~tenpy.MPO`
        The Hamiltonian MPO
    tol_zero : float
        Blocks with norm < `tol_zero` are considered to be zero.

    Returns
    -------
    H_bond : list of :class:`lasyten.Tensor`
        The two-site bond terms such that ``H = sum_i H_bond[i]``.
        ``H_bond[i]`` acts in sites ``(i - 1, i)``.
        For finite systems, we have ``H_bond[0] = None``.
        Legs of each ``H_bond[i]`` are ``[p0, p1, p1*, p0*]``.

    Raises
    ------
    ValueError : if the Hamiltonian contains longer-range terms.

    """
    if not H_MPO.symmetry.trivial_shift:
        raise NotImplementedError  # will need to shift tensors around

    H_onsite = []
    B_list = []
    C_list = []
    Id_list = []

    for j, W in enumerate(H_MPO._W):
        left = H_MPO.finite and j == 0
        right = H_MPO.finite and j == H_MPO.L - 1
        A, B, C, D, Id = partition_W(W, left_boundary=left, right_boundary=right)
        if A is not None and lasy.norm(A) > tol_zero:
            raise ValueError('Not a Nearest-Neighbour MPO!')
        H_onsite.append(lasy.squeeze_legs(D, ['wL', 'wR']))
        B_list.append(B)
        C_list.append(C)
        Id_list.append(Id)

    H_bond = []
    for j in range(H_MPO.L):
        if H_MPO.finite and j == 0:
            H_bond.append(None)
            continue
        i = (j - 1) % H_bond.L
        # two-body coupling is C_i @ B_j
        h = lasy.planar_contraction(
            lasy.squeeze_legs(C_list[i], 'wL'),
            lasy.squeeze_legs(B_list[j], 'wR'),
            'wR',
            'wL',
            relabel1={'p': 'p0', 'p*': 'p0*'},
            relabel2={'p': 'p1', 'p*': 'p1*'},
        )

        # add the onsite terms
        strength_i = 1 if H_MPO.finite and i == 0 else 0.5
        strength_j = 1 if H_MPO.finite and j == H_MPO.L - 1 else 0.5
        h += strength_i * lasy.outer(
            H_onsite[i], Id_list[j], relabel1={'p': 'p0', 'p*': 'p0*'}, relabel2={'p': 'p1', 'p*': 'p1*'}
        )
        h += strength_j * lasy.outer(
            Id_list[i], H_onsite[j], relabel1={'p': 'p0', 'p*': 'p0*'}, relabel2={'p': 'p1', 'p*': 'p1*'}
        )
        H_bond.append(h)

    if H_MPO.explicit_plus_hc:
        for i, h in enumerate(H_bond):
            H_bond[i] = h + h.hc

    return H_bond
