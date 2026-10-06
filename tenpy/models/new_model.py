import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Literal

import cyten as ct

from ..networks import MPO
from ..tools.params import Config, asConfig
from .lattice import Lattice


class EvolutionModel(metaclass=ABCMeta):
    """Base class for models that define time evolution.

    TODO this interface would also allow us to move time-dependent H treatment from algorithm to model

    Attributes
    ----------
    lattice: Lattice
    symmetry: cyten.Symmetry
    model_params: Config

    """

    def __init__(self, lattice: Lattice, model_params):
        self.lattice = lattice
        self.symmetry = lattice.symmetry  # FIXME
        self.backend = lattice.backend  # FIXME
        self.bc_MPS = lattice.bc_MPS
        self.model_params = asConfig(model_params, type(self).__name__)

    @abstractmethod
    def get_U_bond(self, t: float, dt: float, Nsteps: int = 1) -> list[tuple[int, ct.Tensor]]:
        """A time step as NN circuit, possibly up to approximations from discretization.

        FIXME where to store trotter options?

        Returns
        -------
        gates: list of (int, :class:`cyten.Tensor`)
            Entries ``i, U`` indicate to apply gate ``U`` on MPS sites ``i, i + 1``.

        """
        ...

    @abstractmethod
    def get_U_MPO(self, t: float, dt: float) -> MPO:
        """A time step as an MPO, possibly up to approximations from discretization."""
        ...


class RandomUnitaryEvolution(EvolutionModel):
    """FIXME

    Todo:
    - test coverage (properties, make sure all U are different!)
    - get rid of RandomUnitaryEvolution in tebd.py, replace with this model

    """

    def __init__(self, lattice, model_params):
        raise NotImplementedError  # FIXME -> gh tracking


class HamiltonianModel(EvolutionModel, metaclass=ABCMeta):
    """Base class for models that define a Hamiltonian."""

    def __init__(self, lattice: Lattice, model_params):
        EvolutionModel.__init__(self, lattice=lattice, model_params=model_params)

    @abstractmethod
    def get_H_bond(self) -> list[ct.Tensor | None]:
        """The Hamiltonian as a sum of bond-terms.

        Returns
        -------
        H_bond: list of :class:`cyten.Tensor`
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

    def get_U_bond(self): ...  # FIXME calc from self.get_H_bond using trotterization


class HamiltonianMPOModel(HamiltonianModel):
    """A model where the Hamiltonian is defined

    Hamiltonian is defined in terms of an MPO directly, e.g. from compression / other software.

    FIXME
    """

    def __init__(self, lattice: Lattice, H_MPO: MPO):
        self._H_MPO = H_MPO
        self._H_bond = None
        HamiltonianModel.__init__(self, lattice=lattice)

    def get_H_MPO(self):
        return self._H_MPO

    def get_H_bond(self):
        if self._H_bond is None:
            self._H_bond = self._H_MPO.to_H_bond()  # FIXME impl, use olf calc_H_bond_from_MPO and get rid of that
        return self._H_bond

    def get_U_MPO(self):
        raise NotImplementedError('Can not exponentiate generic MPOs.')


class CouplingModel(HamiltonianModel):
    """Hamiltonian is defined in terms of couplings"""

    def __init__(self, lattice: Lattice):
        self.couplings: list[Sequence[Sequence[int]], ct.Coupling] = []
        # couplings[i] = (acts_on, coupling), where act_on is a (N_sites, D+1)-array of lat idcs
        HamiltonianModel.__init__(self, lattice=lattice)

    def append_coupling(self, lat_idcs: Sequence[Sequence[int]], coupling: ct.Coupling):
        """FIXME low-level version, works on lat_idcs"""
        ...

    def add_onsite_operator(
        self,
        x: int | Sequence[int] | Literal['sum_over'],
        u: int | Literal['sum_over'],
        op: str | ct.Coupling | ct.Tensor,
        strength=1,
        plus_hc: bool = False,
    ):
        ...
        self.invalidate_caches()

    def add_two_site_operator(
        self,
        x1: int | Sequence[int] | Literal['sum_over'],
        u1: int | Literal['sum_over_independent'] | Literal['sum_over_matching'],
        u2: int | Literal['sum_over_independent'] | Literal['sum_over_matching'],
        dx: int | Sequence[int],
        op: str | ct.Coupling | ct.Tensor,
        strength=1,
        plus_hc: bool = False,
    ):
        ...
        self.invalidate_caches()

    def add_multi_site_operator(
        self,
        x: int | Sequence[int] | Literal['sum_over'],
        dx: Sequence[int | Sequence[int]],
        u: int | Sequence[int] | Literal['sum_over_independent'] | Literal['sum_over_matching'],
        op: str | ct.Coupling | ct.Tensor,
        strength=1,
        plus_hc: bool = False,
    ):
        ...
        self.invalidate_caches()

    def add_exponentially_decaying_coupling(self): ...

    def add_exponentially_decaying_centered_terms(self): ...

    def invalidate_caches(self, warn=True):
        if warn and self.H_bond is not None:
            print('dummy warning')
        self.H_bond = None
        ...  # same for all caches

    def calc_H_bond(self):
        # calculate from the full tensors of the self.couplings
        ...

    def calc_H_MPO(self):
        # build MPO graph from the self.couplings
        ...


class AutoHamiltonianModel(CouplingModel):  # FIXME better rename... ConfigHamiltonianModel
    default_lattice = 'Chain'
    force_default_lattice = False

    def __init__(self, options):
        self._AutoHamiltonianModel_init_finished = False
        self.options = options = asConfig(options, self.__class__.__name__)
        lattice = self.init_lattice(options)
        CouplingModel.__init__(self, lattice=lattice)
        self.init_terms(lattice, options)
        options.warn_unused()
        self._AutoHamiltonianModel_init_finished = True

    def init_lattice(self, options: Config) -> Lattice: ...  # like old CouplingMPOModel

    def init_site(self, options: Config) -> list[ct.Site]: ...  # like old CouplingMPOModel

    def init_terms(self, lattice: Lattice, options: Config):
        pass

    def append_coupling(self, lat_idcs, coupling):
        if self._AutoHamiltonianModel_init_finished:
            warnings.warn('Added a coupling after initialization')
        return super().append_coupling(lat_idcs, coupling)
