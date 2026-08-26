import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Literal

import cyten as ct

from ..tools.params import Config, asConfig
from .lattice import Lattice

SUM_OVER_ALL = object()


class EvolutionModel(metaclass=ABCMeta):
    def __init__(self, lattice: Lattice):
        self.lattice = lattice

    @abstractmethod
    def get_U_bond_1D(self, dt: float) -> list[ct.Tensor]: ...

    @abstractmethod
    def get_U_MPO_1D(self, dt: float) -> 'MPO': ...


class RandomUnitaryEvolution(EvolutionModel): ...


class HamiltonianModel(EvolutionModel, metaclass=ABCMeta):
    def __init__(self, lattice: Lattice):
        self.lattice = lattice

    @abstractmethod
    def get_H_bond_1D(self, dt: float) -> list[ct.Tensor]: ...

    @abstractmethod
    def get_H_MPO_1D(self, dt: float) -> 'MPO': ...

    def get_U_bond_1D(self): ...  # calc from self.get_H_bond_1D using trotterization

    def get_U_MPO_1D(self): ...  # calc from self.get_H_MPO_1D using WI / WII


class HamiltonianMPOModel(HamiltonianModel):
    """Hamiltonian is defined in terms of an MPO directly, e.g. from compression / other software.

    TODO is the convention for site order in the MPO unambiguous...?
    """

    def __init__(self, lattice: Lattice, H_MPO: 'MPO'):
        self._H_MPO_1D = H_MPO
        HamiltonianModel.__init__(self, lattice=lattice)

    def get_H_MPO_1D(self):
        return self._H_MPO_1D

    def get_H_bond_1D(self):
        raise NotImplementedError  # possible, but not really needed?


class CouplingModel(HamiltonianModel):
    """Hamiltonian is defined in terms of couplings"""

    # FIXME this is still a draft. check all functionality of v1 carefully

    _all_H_caches = ('H_MPO_1D', 'H_bond_1D')

    def __init__(self, lattice: Lattice):
        self.couplings: list[Sequence[Sequence[int]], ct.Coupling] = []
        # couplings[i] = (acts_on, coupling), where act_on is a (N_sites, D+1)-array of lat idcs
        HamiltonianModel.__init__(self, lattice=lattice)

    def append_coupling(self, lat_idcs: Sequence[Sequence[int]], coupling: ct.Coupling): ...

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
        if warn and self.H_bond_1D is not None:
            print('dummy warning')
        self.H_bond_1D = None
        ...  # same for all caches

    def calc_H_bond_1D(self):
        # calculate from the full tensors of the self.couplings
        ...

    def calc_H_MPO_1D(self):
        # build MPO graph from the self.couplings
        ...


class AutoHamiltonianModel(CouplingModel):
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
