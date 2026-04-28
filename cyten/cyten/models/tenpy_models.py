"""Should probably live in tenpy long-term.

For now, while this is the only tenpy module that we already think about modifying for v2,
it is easier to have it live in the cyten repo...

"""

# Copyright (C) TeNPy Developers, Apache license
from collections.abc import Sequence
from functools import partial
from typing import Protocol, TypeAlias

import numpy as np

from ..backends import TensorBackend
from ..tensors import SymmetricTensor
from .couplings import Coupling, gold_coupling, spin_spin_coupling
from .sites import GoldenSite, Site, SpinSite


class CouplingFactory(Protocol):
    """Defines a type (protocol) for functions that create couplings."""

    def __call__(
        self, sites: list[Site], backend: TensorBackend = None, device: str = None, name: str | None = ...
    ): ...


CouplingLike: TypeAlias = SymmetricTensor | Coupling | CouplingFactory


class CouplingModel:
    """TODO this is just a mockup

    I just sketched out an idea for how an adjusted model base class in tenpy v2 might work.
    This does not aim to be complete or functional.

    TODO it might make sense to get rid of the distinction between NNModel, MPOModel and CouplingModel
         they can all go through the Coupling framework. H_bond can be converted to coupling easily,
         so it only needs a classmethod, not a dedicated subclass. Same for MPOModel, an MPO
         essentially already is a coupling, so "model from MPO" can be captured by a classmethod
         and doesnt need its own subclass.
    """

    named_couplings: dict[str, CouplingLike]
    # TODO be very clear about the role of Coupling.name and if it is the same role or different
    #      from keys in name_couplings

    def add_coupling(
        self,
        prefactor: float | complex | Sequence[float | complex],
        coupling: str | CouplingLike,
        positions: Sequence[tuple[list[int], int]],  # one (dx, u) for each site
        name: str = None,
    ):
        """Similar role as add_coupling *and* add_multi_coupling in tenpy v1.

        The sum over lattice unit cells is implied.
        """
        # TODO maybe we want to extend pairs to groups of sites of arbitrary size?
        #      e.g. stars and plaquettes in toric code, rings on honeycomb / kagome etc
        sites = self.get_sites(positions)
        coupling = self.get_coupling(coupling, sites, name=name)
        raise NotImplementedError

    # TODO exponential couplings?

    def get_coupling(
        self,
        coupling: str | CouplingLike,
        sites: list[Site],
        name: str = None,
    ) -> Coupling:
        # TODO caching?
        backend = device = None  # TODO dummy: where should we set those?

        if isinstance(coupling, str):
            res = self.named_couplings.get(coupling, None)
            if res is None and len(sites) == 1:
                # if single-site, look up in the site
                res = sites[0].onsite_operators.get(coupling, None)
            # TODO could add more places for lookup here
            if res is None:
                raise KeyError(f'Coupling not found: {coupling}')
            coupling = res
        elif name is not None and name not in self.named_couplings:
            self.named_couplings = coupling

        if isinstance(coupling, SymmetricTensor):
            coupling = Coupling.from_tensor(coupling, sites, name=name)
        elif isinstance(coupling, Coupling):
            # TODO update name if name not None?
            pass  # TODO check it has the correct sites?
        else:
            extra_kwargs = {} if name is None else dict(name=name)
            coupling = coupling(sites, backend=backend, device=device, **extra_kwargs)

        return coupling

    def get_sites(self, positions: Sequence[tuple[list[int], int]]) -> list[Site]:
        raise NotImplementedError


# defining models:
# should be enough to override init_sites and init_terms, like in tenpy v1.
# add_coupling should be all we need. may


class TFIModel(CouplingModel):
    """spin-1/2 TFI model"""

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'best', str)
        if conserve == 'best':
            conserve = 'parity'
            self.logger.info(f'{self.name}: set conserve to {conserve}')
        elif conserve in ['parity', 'None']:
            pass
        else:
            raise ValueError
        site = SpinSite(S=0.5, conserve=conserve)
        return site

    def init_terms(self, model_params):
        option_A = True  # TODO this is only for demo, remove when merging
        if option_A:
            # option A: make a concrete Coupling instance
            #           less flexible, since you need the sites. this is easy here, since all
            #           sites are the same.
            site = self.lat.unit_cell[0]
            interaction = spin_spin_coupling([site, site], Jx=1)
        else:
            # option B: make a CouplingFactory
            interaction = partial(spin_spin_coupling, xx=1, yy=None, zz=None)

        J = np.asarray(model_params.get('J', 1.0, 'real_or_array'))
        g = np.asarray(model_params.get('g', 1.0, 'real_or_array'))
        # TODO idea: let Lattice.groups take role of pairs, but with any # of sites
        # TODO but maybe we should have attrs instead of dict entries?
        #       i.e. ``lat.all_sites`` and ``lat.nearest_neighbors`` ?
        #       or ``lat.groups.all_sites`` and ``lat.groups.nearest_neighbors``?
        for positions in self.lat.groups['all_sites']:
            self.add_coupling(-5 * g, 'Sz', positions)  # sigma_z = .5 * Sz
        for positions in self.lat.groups['nearest_neighbors']:
            self.add_coupling(-0.25 * J, interaction, positions)  # sigma_x = .5 * Sx
        return


class GoldenModel(CouplingModel):
    """TODO"""

    def init_sites(self, model_params):
        return GoldenSite(handedness=model_params.get('handedness', 'left', str))

    def init_terms(self, model_params):
        J = np.asarray(model_params.get('J', 1.0, 'real_or_array'))
        for positions in self.lat.groups['nearest_neighbors']:
            self.add_coupling(J, gold_coupling, positions)
        return


class GoldenChain(GoldenModel):
    """TODO"""

    default_lattice = 'Chain'
    force_default_lattice = True


# TODO more models. at least all from tenpy v1. also include more anyonic models
