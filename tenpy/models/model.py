"""TODO elaborate

.. _couplings:

Couplings in Models
-------------------
TODO elaborate on the system of couplings


"""
# Copyright (C) TeNPy Developers, Apache license

import inspect
import itertools
import logging
from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Sequence
from typing import Literal

import lasyten as lasy
import numpy as np
from numpy.typing import ArrayLike

from ..networks import MPO, MPOGraph
from ..networks.mpo import partition_W
from ..networks.terms import CouplingTerms, ExponentiallyDecayingTerms, MultiCouplingTerms, OnsiteTerms, Terms
from ..tools import asConfig, perm_sign, to_array
from .lattice import HelicalLattice, IrregularLattice, Lattice, MultiSpeciesLattice, get_lattice

__all__ = [
    'EvolutionModel',
    'RandomUnitaryEvolution',
    'HamiltonianModel',
    'HamiltonianMPOModel',
    'CouplingModel',
    'Model',
    'NearestNeighborModel',
]

type CouplingPermuteSpec = (
    Literal['invariant', 'odd', 'swap_hc'] | Callable[[Sequence[int], dict[str, float | complex]]]
)


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

    #: logging.Logger : An instance of a logger; see :doc:`/intro/logging`. NB: class attribute.
    logger = logging.getLogger(__name__ + '.Model')

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
        # TODO read approximation from params?
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


class CouplingModel(HamiltonianModel):
    """A model, which defines a Hamiltonian in terms of couplings.

    TODO elaborate
    - statefulness, point to methods

    Parameters
    ----------
    lattice
        TODO
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
    onsite_terms
        TODO
    coupling_terms
        TODO
    multi_coupling_terms
        TODO
    exp_decaying_terms
        TODO
    couplings : dict {str: :class:`lasyten.Coupling`}
        Definition of the coupling names available in this model.
        Couplings need to be defined via :meth:`define_coupling`, by default,
        a :class:`CouplingModel` has no couplings defined.
    coupling_hc_map : dict {str: str}
        A mapping from coupling name to the name of the hc coupling.
    coupling_permute : dict {str: str | callable}
        Defines the behavior of the :attr:`couplings` under permutation of sites.
        Possible values are::

            - ``'invariant'`` indicates that the coupling does not change under any permutation
            - ``'odd'`` indicates that the coupling is invariant under even permutations and maps
              to negative itself under odd permutations
            - ``'swap_hc'`` is only allowed for two-site couplings and indicates that the coupling
              maps to its hc, given by :attr:`coupling_hc_map` under swapping the sites
            - A callable ``permute(perm: Sequence[int]) -> dict[str, float | complex]`` that realizes
              :meth:`permute_coupling` for this coupling.

    """

    tol_zero_strength = 1e-15

    def __init__(
        self,
        lattice: Lattice,
        explicit_plus_hc: bool = False,
    ):
        self._H_MPO: MPO | None = None
        self._H_bond: list[lasy.Tensor] | None = None
        self._L_mps = L = len(lattice.mps_sites())
        self.onsite_terms = OnsiteTerms(L)
        self.coupling_terms = CouplingTerms(L)
        self.multi_coupling_terms = MultiCouplingTerms(L)
        self.exp_decaying_terms = ExponentiallyDecayingTerms(L)
        self.explicit_plus_hc = explicit_plus_hc

        self.couplings: dict[str, lasy.Coupling] = {}
        self.coupling_hc_map: dict[str, str] = {}
        self.coupling_permute: dict[str, CouplingPermuteSpec] = {}

        HamiltonianModel.__init__(self, lattice=lattice)

    def test_sanity(self):
        if self._H_MPO is not None:
            self._H_MPO.test_sanity()
        sites = self.lattice.mps_sites()
        self.onsite_terms.test_terms(sites, self)
        self.coupling_terms.test_terms(sites, self)
        self.multi_coupling_terms.test_terms(sites, self)
        self.exp_decaying_terms.test_terms(sites, self)

        for name, coupling in self.couplings.items():
            assert coupling.factorization[0].symmetry == self.symmetry
            assert coupling.factorization[0].backend == self.backend
            coupling.test_sanity()

            hc_name = self.coupling_hc_map[name]
            assert self.coupling_hc_map[hc_name] == name
            hc_coupling = self.couplings[hc_name]
            assert lasy.almost_equal(coupling.to_tensor(), hc_coupling.to_tensor().hc)  # TODO is this cheap enough?
            assert coupling.num_sites == hc_coupling.num_sites
            assert (
                self.coupling_permute[name] is self.coupling_permute[hc_name]
            )  # TODO is this ok? otherwise we would need to compare action of the function on a few / all perms...

        return super().test_sanity()

    def define_coupling(self, coupling: lasy.Coupling, hc: str, permute: CouplingPermuteSpec):
        """Add a coupling to the list of named :attr:`couplings` available in the model

        Parameters
        ----------
        coupling: :class:`lasyten.Coupling`
            The new coupling. Must be named (have :attr:`lasyten.Coupling.name` set)
        hc: str
            The name for the hermitian conjugate coupling
        permute: ``'invariant'``| ``'odd'`` | ``'swap_hc'`` | callable
            Specifies how the coupling transforms under permutation of sites.
            ``'invariant'`` means it does not change under any permutation.
            ``'odd'`` means it stays invariant under even permutations and maps to negative itself under odd.
            ``'swap_hc'`` is only valid for two-site couplings and means if sites are swapped it maps to its hc.
            For the general case, specify a callable ``permute(perm: Sequence[int]) -> dict[str, float | complex]``
            that expresses the permuted coupling as a linear combination of the keys with the values as prefactors.

        """
        if coupling.name is None:
            raise ValueError('Coupling must be named')
        if coupling.name in self.couplings:
            raise ValueError('Coupling name already defined')
        assert permute in ['invariant', 'odd', 'swap_hc'] or hasattr(permute, '__call__')
        self.couplings[coupling.name] = coupling
        self.coupling_hc_map[coupling.name] = hc
        self.coupling_permute[coupling.name] = permute
        if hc in self.coupling_hc_map:
            assert self.coupling_hc_map[hc] == coupling.name
        else:
            self.couplings[hc] = coupling.dagger(new_name=hc)
            self.coupling_hc_map[hc] = coupling.name
            self.coupling_permute[hc] = permute
        return

    def permute_coupling(self, coupling_name: str, perm: Sequence[int]) -> dict[str, float | complex]:
        if not self.symmetry.has_trivial_braid:
            raise NotImplementedError  # current design does not cover this well...

        if self.coupling_permute[coupling_name] == 'invariant':
            return {coupling_name: 1}
        if self.coupling_permute[coupling_name] == 'odd':
            return {coupling_name: perm_sign(perm)}
        if self.coupling_permute[coupling_name] == 'swap_hc':
            assert self.couplings[coupling_name].num_sites == 2
            if perm_sign(perm) == +1:
                return {coupling_name: 1}
            else:
                return {self.coupling_hc_map[coupling_name]: 1}
        return self.coupling_permute(perm)

    def test_coupling(self, coupling_name: str, sites: Sequence[int]):
        """Test if the `coupling_name` is defined, and valid on the `sites`.

        Raises
        ------
        ValueError
            If not defined or not valid

        """
        mps_sites = self.lattice.mps_sites()
        sites = [mps_sites[i % len(mps_sites)] for i in sites]
        try:
            coupling = self.couplings[coupling_name]
        except KeyError:
            raise ValueError(f'Unknown coupling name {coupling_name}') from None
        for s, W in zip(sites, coupling.factorization):
            if s.leg != W.get_leg('p'):
                raise ValueError('Incompatible leg for the given site')

    def invalidate_caches(self):
        self._H_MPO = None
        self._H_bond = None
        super().invalidate_caches()

    def get_nonzero_terms(self) -> list[Terms]:
        return [
            self.onsite_terms.remove_zeros(self.tol_zero_strength),
            self.coupling_terms.remove_zeros(self.tol_zero_strength),
            self.multi_coupling_terms.remove_zeros(self.tol_zero_strength),
            # self.exp_decaying_terms.remove_zeros(self.tol_zero_strength),  # TODO reactivate
        ]

    def get_H_MPO(self):
        if self._H_MPO is None:
            terms = self.get_nonzero_terms()
            graph = MPOGraph.from_terms(
                terms,
                model=self,
                bc=self.bc_MPS,
                unit_cell_width=self.lattice.mps_unit_cell_width,
            )
            self._H_MPO = graph.build_MPO(explicit_plus_hc=self.explicit_plus_hc)
        return self._H_MPO

    def get_H_bond(self):
        if self._H_bond is None:
            sites = self.lattice.mps_sites()
            ot, ct, *others = self.get_nonzero_terms()
            H_bond = ct.to_nn_bond_tensors(sites, self)
            ot.add_to_nn_bond_tensors(H_bond, sites, finite=self.finite)
            for t in others:
                assert t.max_range() == 0
            self._H_bond = [
                None if h is None else lasy.Coupling.from_tensor(h, [sites[(j - 1) % len(sites)], sites[j]])
                for j, h in enumerate(H_bond)
            ]
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
        if np.all(np.abs(strength) < self.tol_zero_strength):
            # This allows models to "add" symmetry-forbidden terms, as long as the strength is 0
            return

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
            for i, i_lat in zip(*self.lattice.mps_lat_idx_fix_u(u)):
                self._add_onsite_operator_mps(strength[*i_lat], i, op)
        else:
            if not self.finite:
                raise ValueError('Can not add single onsite operator to infinite system.')
            x = np.asarray(x, int).reshape(-1)
            i = self.lattice.lat2mps_idx([*x, u])
            self._add_onsite_operator_mps(strength, i, op)

        if plus_hc:
            hc_op = self.lattice.unit_cell[u].get_hc_op_name(op)
            self.add_onsite_operator(np.conj(strength), hc_op, x=x, u=u, plus_hc=False)

    def _add_onsite_operator_mps(self, pref: float | complex, i: int, op: str):
        site = self.lattice.mps_sites()[i]
        site.test_op(op)
        self.onsite_terms.add_onsite_term(pref, i, op)

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
        if np.all(np.abs(strength) < self.tol_zero_strength):
            return

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
            self._add_two_site_coupling_mps(np.conj(pref), i, j, self.coupling_hc_map[coupling_name], plus_hc=False)

        self.test_coupling(coupling_name, [i, j])

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
        if np.all(np.abs(strength) < self.tol_zero_strength):
            return

        N = self.couplings[coupling_name].num_sites
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
            self._add_multi_coupling_mps(np.conj(pref), ijkl, self.coupling_hc_map[coupling_name], plus_hc=False)
        self.test_coupling(coupling_name, ijkl)
        if lasy.tools.duplicate_entries(ijkl):
            raise ValueError('Can not act on the same site twice!')
        perm = np.argsort(ijkl)
        if np.all(perm == np.arange(len(ijkl))):
            return self.multi_coupling_terms.add_multi_coupling_term(pref, ijkl, coupling_name)
        ijkl_sorted = ijkl[perm]
        for c, a in self.permute_coupling(coupling_name, perm).items():
            self.multi_coupling_terms.add_multi_coupling_term(a * pref, ijkl_sorted, c)
        return

    def add_exponentially_decaying_coupling(self):
        raise NotImplementedError  # TODO skipped for now

    def add_exponentially_decaying_centered_terms(self):
        raise NotImplementedError  # TODO skipped for now


class ModelBuilder(CouplingModel):
    """Plays the role of the old CouplingMPOModel, where you just have to override some methods

    TODO explain what *needs* to be overridden and what *can* be overriden.
    """

    #: class or str: The default lattice class or class name to be used in :meth:`init_lattice`.
    default_lattice = 'Chain'

    #: bool: If True, :meth:`init_lattice` asserts that the initialized lattice
    #: is (a subclass of) `default_lattice`
    force_default_lattice = False

    def __init__(self, model_params):
        if getattr(self, '_called_ModelBuilder_init', False):
            # If we ignore this, the same terms get added to self multiple times.
            # In the best case, this would just rescale the energy;
            # in the worst case we get the wrong Hamiltonian.
            raise ValueError('Called ModelBuilder.__init__(...) multiple times.')
            # To fix this problem, follow the instructions for subclassing in :doc:`/intro/model`.
        self.name = self.__class__.__name__
        self.options = model_params = asConfig(model_params, self.name)
        self._called_ModelBuilder_init = True
        self.manually_call_init_H = getattr(self, 'manually_call_init_H', False)
        explicit_plus_hc = model_params.get('explicit_plus_hc', False, bool)
        # 1-4) initialize lattice
        lattice = self.init_lattice(model_params)
        # 5) initialize CouplingModel
        CouplingModel.__init__(self, lattice, explicit_plus_hc=explicit_plus_hc)
        # 6) add terms of the Hamiltonian
        self.init_couplings(model_params)
        self.init_terms(model_params)
        # finally checks for misspelled parameter names
        model_params.warn_unused()
        self.H_MPO = self.get_H_MPO()  # TODO workaround -> instead should make models use the getter!

    def init_couplings(self, model_params):
        """Initialize all couplings for the given model.

        This function should define the couplings to be used in the model using :meth:`define_coupling`.
        At this point in the initialization, the :attr:`lattice` and :attr:`backend` attributes are
        already initialized and can be used to build the couplings.

        TODO elaborate
        """
        pass  # by default, do not define any couplings

    def init_lattice(self, model_params):
        """Initialize a lattice for the given model parameters.

        This function reads out the model parameter `lattice`.
        This can be a full :class:`~tenpy.models.lattice.Lattice` instance,
        in which case it is just returned without further action.
        Alternatively, the `lattice` parameter can be a string giving the name
        of one of the predefined lattices, which then gets initialized.
        Depending on the dimensionality of the lattice, this requires different model parameters.

        Parameters
        ----------
        model_params : dict
            The model parameters given to ``__init__``.

        Returns
        -------
        lat : :class:`~tenpy.models.lattice.Lattice`
            An initialized lattice.

        Options
        -------
        .. cfg:configoptions :: CouplingMPOModel

            lattice : str | :class:`Lattice`
                The name of a lattice pre-defined in TeNPy to be initialized.
                Alternatively, directly a subclass of :class:`Lattice` instead of the name.
                Alternatively, a (possibly self-defined) :class:`Lattice` instance.
                If an instance is given, none of the further options described here are read out,
                since they are already given inside the lattice instance!
            bc_MPS : str
                Boundary conditions for the MPS.
            order : str
                The order of sites within the lattice for non-trivial lattices,
                e.g, ``'default', 'snake'``, see :meth:`~tenpy.models.lattice.Lattice.ordering`.
                Only used if `lattice` is a string.
            L : int
                The length in x-direction; only read out for 1D lattices.
                For an infinite system the length of the unit cell.
            Lx, Ly : int
                The length in x- and y-direction; only read out for 2D lattices.
                For ``"infinite"`` `bc_MPS`, the system is infinite in x-direction and
                `Lx` is the number of "rings" in the infinite MPS unit cell,
                while `Ly` gives the circumference around the cylinder or width of th the rung
                for a ladder (depending on `bc_y`).
            bc_y : ``"cylinder" | "ladder" | "open" | "periodic"``
                The boundary conditions in y-direction.
                Only read out for 2D lattices.
                "cylinder" is equivalent to "periodic", "ladder" is equivalent to "open".
            bc_x : ``"open" | "periodic"``.
                Can be used to force "periodic" boundaries for the lattice,
                i.e., for the couplings in the Hamiltonian, even if the MPS is finite.
                Defaults to ``"open"`` for ``bc_MPS="finite"`` and
                ``"periodic"`` for ``bc_MPS="infinite``.
                If you are not aware of the consequences, you should probably
                *not* use "periodic" boundary conditions.
                (The MPS is still "open", so this will introduce long-range
                couplings between the first and last sites of the MPS!)
            helical: ``None`` | int
                If not ``None``, wrap the lattice into a
                :class:`~tenpy.models.lattice.HelicalLattice` with the given int as `N_unit_cells`.
            irregular_remove : ``None`` | 2D array
                If not ``None``, wrap the lattice into a
                :class:`~tenpy.models.lattice.IrregularLattice` removing the specified sites.
                To add sites, you need to overwrite the `init_lattice` method in a custom model.

        """
        lat = model_params.get('lattice', self.default_lattice)
        if isinstance(lat, str):
            LatticeClass = get_lattice(lattice_name=lat)
            lat = None
        elif inspect.isclass(lat) and issubclass(lat, Lattice):
            LatticeClass = lat
            lat = None
        elif not isinstance(lat, Lattice):
            raise ValueError("invalid type for model_params['lattice'], got " + repr(lat))
        if lat is None:  # only provided LatticeClass
            sites = self.init_sites(model_params)
            if isinstance(sites, tuple) and sites[0] is not None and not isinstance(sites[0], lasy.Site):
                species_sites, species_names = sites
                sites = None
            else:
                species_sites = None
            try:
                lat = LatticeClass.from_model_params(model_params=model_params, sites=sites)
            except Exception as e:
                msg = (
                    'Failed to initialize the lattice from model_params. Original error above. '
                    'Consider overriding init_lattice for your model?'
                )
                raise ValueError(msg) from e
            # possibly modify/generalize the already initialized lattice
            if species_sites is not None:
                lat = MultiSpeciesLattice(lat, species_sites, species_names)
            helical = model_params.get('helical_lattice', None)
            if helical is not None:
                lat = HelicalLattice(lat, helical)
            irregular_remove = model_params.get('irregular_remove', None)
            if irregular_remove is not None:
                lat = IrregularLattice(lat, remove=irregular_remove)
        # else: a lattice was already provided
        # now, `lat` is an instance of the LatticeClass
        assert isinstance(lat, Lattice)
        if self.force_default_lattice:
            DefaultLattice = self.default_lattice
            if isinstance(DefaultLattice, str):
                DefaultLattice = get_lattice(DefaultLattice)
            check_lat = lat
            if isinstance(check_lat, IrregularLattice):
                check_lat = check_lat.regular_lattice
            if isinstance(check_lat, HelicalLattice):
                check_lat = check_lat.regular_lattice
            if isinstance(check_lat, MultiSpeciesLattice):
                check_lat = check_lat.simple_lattice
            assert isinstance(check_lat, DefaultLattice), 'model sets force_default_lattice'
        return lat

    def init_sites(self, model_params) -> lasy.Site | Sequence[lasy.Site]:
        """Define the local Hilbert space and operators; needs to be implemented in subclasses.

        This function gets called by :meth:`init_lattice` to get the
        :class:`~tenpy.networks.site.Site` for the lattice unit cell.

        .. note ::
            Initializing the sites requires to define the conserved quantum numbers.
            All pre-defined sites accept ``conserve=None`` to disable using quantum numbers.
            Many models in TeNPy read out the `conserve` model parameter, which can be set
            to ``"best"`` to indicate the optimal parameters.

        If you need to initialize more than one site, the function
        :func:`tenpy.networks.site.set_common_charges` should be helpful.

        Parameters
        ----------
        model_params : dict
            The model parameters given to ``__init__``.

        Returns
        -------
        sites : (tuple of) :class:`~tenpy.networks.site.Site`
            The local sites of the lattice, defining the local basis states and operators.
        optional_species_names : not set | list of str | None
            You should usually just return the (tuple of) `sites`.
            However, you can additionally return a list `species_names` to indicate that the
            :class:`~tenpy.models.lattice.MultiSpeciesLattice` should be used.

        """
        # example:
        #     conserve = model_params.get('conserve', 'best', str)
        #     if conserve == 'best':
        #         # might check other model_params to see what's actually best possible
        #         conserve = 'Sz'
        #     return SpinHalfSite(conserve=conserve)
        # or if you want to use the MultiSpeciesLattice, e.g. for spin-full fermions:
        #     f_up, f_down = FermionSite('N'), FermionSite('N')
        #     set_common_charges([f_up, f_down], 'independent')
        #     # 'independent' for conserving N_up and N_down individually, 'same' for total N
        #     return [f_up, f_down], ["up", "down"]
        raise NotImplementedError('Subclasses should implement `init_sites`')
        # or at least redefine the lattice

    def init_terms(self, model_params):
        """Add the onsite and coupling terms to the model; subclasses should implement this."""
        pass  # Do nothing. This allows to super().init_terms(model_params) in subclasses.


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
        i = (j - 1) % H_MPO.L
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
