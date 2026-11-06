"""Next-Nearest-neighbor spin-S models.

Uniform lattice of spin-S sites, coupled by next-nearest-neighbor interactions.
We have two variants implementing the same hamiltonian.
The :class:`SpinChainNNN` uses the
:class:`~tenpy.networks.site.GroupedSite` to keep it a
:class:`~tenpy.models.model.NearestNeighborModel` suitable for TEBD,
while the :class:`SpinChainNNN2` just involves longer-range couplings in the MPO.
The latter is preferable for pure DMRG calculations and avoids having to add each of the short
range couplings twice for the grouped sites.

Note that you can also get a :class:`~tenpy.models.model.NearestNeighborModel` for TEBD from the
latter by using :meth:`~tenpy.models.model.MPOModel.group_sites` and
:meth:`~tenpy.models.model.NearestNeighborModel.from_MPOModel`.
An example for such a case is given in the file ``examples/c_tebd.py``.
"""
# Copyright (C) TeNPy Developers, Apache license

import lasyten as lasy
import numpy as np

from .lattice import Chain
from .model import ModelBuilder

__all__ = ['SpinChainNNN', 'SpinChainNNN2']


class SpinChainNNN(ModelBuilder):
    r"""Spin-S sites coupled by (next-)nearest neighbor interactions on a `GroupedSite`.

    The Hamiltonian reads:

    .. math ::
        H = \sum_{\langle i,j \rangle, i < j}
                \mathtt{Jx} S^x_i S^x_j + \mathtt{Jy} S^y_i S^y_j + \mathtt{Jz} S^z_i S^z_j \\
            + \sum_{\langle \langle i,j \rangle \rangle, i< j}
                \mathtt{Jxp} S^x_i S^x_j + \mathtt{Jyp} S^y_i S^y_j + \mathtt{Jzp} S^z_i S^z_j \\
            - \sum_i
              \mathtt{hx} S^x_i + \mathtt{hy} S^y_i + \mathtt{hz} S^z_i

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbors and
    :math:`\langle \langle i,j \rangle \rangle, i < j` denotes next nearest neighbors.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`SpinChainNNN` below.

    Options
    -------
    .. cfg:config :: SpinChainNNN
        :include: CouplingMPOModel

        L : int
            Length of the chain in terms of :class:`~tenpy.networks.site.GroupedSite`,
            i.e. we have ``2*L`` spin sites.
        S : {0.5, 1, 1.5, 2, ...}
            The 2S+1 local states range from m = -S, -S+1, ... +S.
        conserve : 'best' | 'Sz' | 'parity' | None
            What should be conserved. See :class:`~tenpy.networks.Site.SpinSite`.
        Jx, Jy, Jz, Jxp, Jyp, Jzp, hx, hy, hz : float | array
            Coupling as defined for the Hamiltonian above.
            Defaults to all ``Jx=Jy=Jz=Jxp=Jyp=Jzp=1`` and zero fields ``h*=0``.
        bc_MPS : {'finite' | 'infinite'}
            MPS boundary conditions. Coupling boundary conditions are chosen appropriately.

    """

    default_lattice = Chain
    force_default_lattice = True

    def init_sites(self, model_params):
        raise NotImplementedError('grouped-sites')
        S = model_params.get('S', 0.5, 'real')
        conserve = model_params.get('conserve', 'best', str)
        if conserve == 'best':
            # check how much we can conserve
            if not model_params.any_nonzero([('Jx', 'Jy'), ('Jxp', 'Jyp'), 'hx', 'hy'], 'check Sz conservation'):
                conserve = 'Sz'
            elif not model_params.any_nonzero(['hx', 'hy'], 'check parity conservation'):
                conserve = 'parity'
            else:
                conserve = None
            self.logger.info('%s: set conserve to %s', self.name, conserve)
        spinsite = SpinSite(S, conserve, sort_charge=False)  # GroupedSite does sort charges
        site = GroupedSite([spinsite, spinsite], charges='same')
        return site

    def init_terms(self, model_params):
        Jx = model_params.get('Jx', 1.0, 'real_or_array')
        Jy = model_params.get('Jy', 1.0, 'real_or_array')
        Jz = model_params.get('Jz', 1.0, 'real_or_array')
        Jxp = model_params.get('Jxp', 1.0, 'real_or_array')
        Jyp = model_params.get('Jyp', 1.0, 'real_or_array')
        Jzp = model_params.get('Jzp', 1.0, 'real_or_array')
        hx = model_params.get('hx', 0.0, 'real_or_array')
        hy = model_params.get('hy', 0.0, 'real_or_array')
        hz = model_params.get('hz', 0.0, 'real_or_array')

        # Only valid for self.lattice being a Chain...
        self.add_onsite(-hx, 0, 'Sx0')
        self.add_onsite(-hy, 0, 'Sy0')
        self.add_onsite(-hz, 0, 'Sz0')
        self.add_onsite(-hx, 0, 'Sx1')
        self.add_onsite(-hy, 0, 'Sy1')
        self.add_onsite(-hz, 0, 'Sz1')
        # Sp = Sx + i Sy, Sm = Sx - i Sy,  Sx = (Sp+Sm)/2, Sy = (Sp-Sm)/2i
        # Sx.Sx = 0.25 ( Sp.Sm + Sm.Sp + Sp.Sp + Sm.Sm )
        # Sy.Sy = 0.25 ( Sp.Sm + Sm.Sp - Sp.Sp - Sm.Sm )
        # nearest neighbors
        self.add_onsite((Jx + Jy) / 4.0, 0, 'Sp0 Sm1', plus_hc=True)
        self.add_onsite((Jx - Jy) / 4.0, 0, 'Sp0 Sp1', plus_hc=True)
        self.add_onsite(Jz, 0, 'Sz0 Sz1')
        self.add_coupling((Jx + Jy) / 4.0, 0, 'Sp1', 0, 'Sm0', 1, plus_hc=True)
        self.add_coupling((Jx - Jy) / 4.0, 0, 'Sp1', 0, 'Sp0', 1, plus_hc=True)
        self.add_coupling(Jz, 0, 'Sz1', 0, 'Sz0', 1)
        # next nearest neighbors
        self.add_coupling((Jxp + Jyp) / 4.0, 0, 'Sp0', 0, 'Sm0', 1, plus_hc=True)
        self.add_coupling((Jxp - Jyp) / 4.0, 0, 'Sp0', 0, 'Sp0', 1, plus_hc=True)
        self.add_coupling(Jzp, 0, 'Sz0', 0, 'Sz0', 1)
        self.add_coupling((Jxp + Jyp) / 4.0, 0, 'Sp1', 0, 'Sm1', 1, plus_hc=True)
        self.add_coupling((Jxp - Jyp) / 4.0, 0, 'Sp1', 0, 'Sp1', 1, plus_hc=True)
        self.add_coupling(Jzp, 0, 'Sz1', 0, 'Sz1', 1)


class SpinChainNNN2(ModelBuilder):
    r"""Spin-S sites coupled by next-nearest neighbor interactions.

    The Hamiltonian reads:

    .. math ::
        H = \sum_{\langle i,j \rangle, i < j}
                \mathtt{Jx} S^x_i S^x_j + \mathtt{Jy} S^y_i S^y_j + \mathtt{Jz} S^z_i S^z_j \\
            + \sum_{\langle \langle i,j \rangle \rangle, i< j}
                \mathtt{Jxp} S^x_i S^x_j + \mathtt{Jyp} S^y_i S^y_j + \mathtt{Jzp} S^z_i S^z_j \\
            - \sum_i
              \mathtt{hx} S^x_i + \mathtt{hy} S^y_i + \mathtt{hz} S^z_i

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbors and
    :math:`\langle \langle i,j \rangle \rangle, i < j` denotes next nearest neighbors.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`SpinChainNNN2` below.

    Options
    -------
    .. cfg:config :: SpinChainNNN2
        :include: CouplingMPOModel

        S : {0.5, 1, 1.5, 2, ...}
            The 2S+1 local states range from m = -S, -S+1, ... +S.
        conserve : 'best' | 'SU2' | 'Sz' | 'parity' | None
            What should be conserved. See :class:`~tenpy.networks.Site.SpinSite`.
            For ``'best'``, we check the parameters what can be preserved.
        Jx, Jy, Jz, Jxp, Jyp, Jzp, hx, hy, hz : float | array
            Coupling as defined for the Hamiltonian above.

    """

    def init_sites(self, model_params):
        S = model_params.get('S', 0.5, 'real')
        conserve = model_params.get('conserve', 'best', str)
        if conserve == 'best':
            # check how much we can conserve
            if not model_params.any_nonzero(
                [('Jx', 'Jy', 'Jz'), ('Jxp', 'Jyp', 'Jzp'), 'hx', 'hy', 'hz'], 'check SU(2) conservation'
            ):
                conserve = 'SU2'
            elif not model_params.any_nonzero([('Jx', 'Jy'), ('Jxp', 'Jyp'), 'hx', 'hy'], 'check Sz conservation'):
                conserve = 'Sz'
            elif not model_params.any_nonzero(['hx', 'hy'], 'check parity conservation'):
                conserve = 'parity'
            else:
                conserve = None
            self.logger.info('%s: set conserve to %s', self.name, conserve)
        self.conserve = conserve
        return lasy.sites.SpinSite(S=S, conserve=conserve)

    def init_couplings(self, model_params):
        site, *_ = self.lattice.unit_cell

        self.define_coupling(
            lasy.models.spin_spin_coupling([site, site], Jx=1, Jy=1, Jz=1, name='S.S'),
            hc='S.S',
            permute='invariant',
        )
        if self.conserve != 'SU2':
            self.define_coupling(
                lasy.models.spin_spin_coupling([site, site], Jz=1, name='Sz.Sz'),
                hc='Sz.Sz',
                permute='invariant',
            )
            self.define_coupling(
                lasy.models.spin_spin_coupling([site, site], Jpm=1, name='Sp.Sm'),
                hc='Sm.Sp',
                permute='swap_hc',
            )
        if self.conserve not in ['SU2', 'Sz']:
            self.define_coupling(
                lasy.models.spin_spin_coupling([site, site], Jpp=1, name='Sp.Sp'),
                hc='Sm.Sm',
                permute='invariant',
            )
        return

    def init_terms(self, model_params):
        # 0) read out/set default parameters
        Jx = model_params.get('Jx', 1.0, 'real_or_array')
        Jy = model_params.get('Jy', 1.0, 'real_or_array')
        Jz = model_params.get('Jz', 1.0, 'real_or_array')
        Jxp = model_params.get('Jxp', 1.0, 'real_or_array')
        Jyp = model_params.get('Jyp', 1.0, 'real_or_array')
        Jzp = model_params.get('Jzp', 1.0, 'real_or_array')
        hx = model_params.get('hx', 0.0, 'real_or_array')
        hy = model_params.get('hy', 0.0, 'real_or_array')
        hz = model_params.get('hz', 0.0, 'real_or_array')

        self.add_onsite_operator(-hx, 'Sx')
        self.add_onsite_operator(-hy, 'Sy')
        self.add_onsite_operator(-hz, 'Sz')

        # Sp = Sx + i Sy, Sm = Sx - i Sy,  Sx = (Sp+Sm)/2, Sy = (Sp-Sm)/2i
        # Sx.Sx = 0.25 ( Sp.Sm + Sm.Sp + Sp.Sp + Sm.Sm )
        # Sy.Sy = 0.25 ( Sp.Sm + Sm.Sp - Sp.Sp - Sm.Sm )
        for u1, u2, dx in self.lattice.pairs['nearest_neighbors']:
            if np.all(Jx == Jy) and np.all(Jy == Jz):
                self.add_two_site_coupling(Jz, 'S.S', dx, u1=u1, u2=u2)
            else:
                self.add_two_site_coupling((Jx + Jy) / 4.0, 'Sp.Sm', dx, u1=u1, u2=u2, plus_hc=True)
                self.add_two_site_coupling((Jx - Jy) / 4.0, 'Sp.Sp', dx, u1=u1, u2=u2, plus_hc=True)
                self.add_two_site_coupling(Jz, 'Sz.Sz', dx, u1=u1, u2=u2)
        for u1, u2, dx in self.lattice.pairs['next_nearest_neighbors']:
            if np.all(Jxp == Jyp) and np.all(Jyp == Jzp):
                self.add_two_site_coupling(Jzp, 'S.S', dx, u1=u1, u2=u2)
            else:
                self.add_two_site_coupling((Jxp + Jyp) / 4.0, 'Sp.Sm', dx, u1=u1, u2=u2, plus_hc=True)
                self.add_two_site_coupling((Jxp - Jyp) / 4.0, 'Sp.Sp', dx, u1=u1, u2=u2, plus_hc=True)
                self.add_two_site_coupling(Jzp, 'Sz.Sz', dx, u1=u1, u2=u2)
