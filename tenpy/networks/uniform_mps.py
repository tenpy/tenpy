r"""This module contains a base class for a Uniform Matrix Product State.

This is an extension of the MPS class for tangent space
algorithms like VUMPS and TDVP (even though the current TDVP algorithm does not
use this).

A uniform MPS differs from a canonical MPS in the tensors that are stored on each site.
In a canonical MPS, we store a single tensor on each site and diagonal Schmidt
coefficients on each bond. From these, we can construct any desired form of a tensor
on each site; e.g. given B_i, we can construct A_i = S_i B_i S_{i+1}^{-1}. On every
site, we assume that $AS = SB$, which is guaranteed (up to numerical noise) after
calling canonical form. In a uniform MPS, however, we are not guaranteed that this
condition holds. Instead, we store an AL tensor (left canonical, A in MPS notation),
AR tensor (right canonical, B), and an AC tensor (one-site orthogonality center, Theta)
on each site. On each bond we store a C tensor that is not guaranteed to be diagonal.

A uniform MPS is only defined in the thermodynamic limit.

The functions in the class are mostly trivial copies of the functions from MPS that
account for the additional type of tensor structure.

"""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np
import logging
import warnings
from ..tools.misc import BetaWarning

logger = logging.getLogger(__name__)

from ..linalg import np_conserved as npc
from .mps import MPS

__all__ = ['UniformMPS']


class UniformMPS(MPS):
    r"""A Uniform Matrix Product State, only defined in the thermodynamic limit.


    Parameters
    ----------
    sites : list of :class:`~tenpy.networks.site.Site`
        Defines the local Hilbert space for each site.
    ALs : list of :class:`~tenpy.linalg.np_conserved.Array`
        The 'left-orthonormal' tensors of the MPS. Labels are ``vL, vR, p`` (in any order).
    ARs : list of :class:`~tenpy.linalg.np_conserved.Array`
        The 'right-orthonormal' tensors of the MPS. Labels are ``vL, vR, p`` (in any order).
    ACs : list of :class:`~tenpy.linalg.np_conserved.Array`
        The 'center-site' tensors of the MPS. Labels are ``vL, vR, p`` (in any order).
    Cs : list of :class:`~tenpy.linalg.np_conserved.Array`
        The center matrices on the left of site `i`, Labels are ``vL, vR`` (in any order).

    Attributes
    ----------
    sites : list of :class:`~tenpy.networks.site.Site`
        Defines the local Hilbert space for each site.
    bc : {'infinite'}
        For uniform MPS only infinite bc are allowed.
    chinfo : :class:`~tenpy.linalg.np_conserved.ChargeInfo`
        The nature of the charge.
    dtype : type
        The data type of the ``_B``.
    norm : float
        The norm of the state, i.e. ``sqrt(<psi|psi>)``.
        Ignored for (normalized) :meth:`expectation_value`, but important for :meth:`overlap`.
    grouped : int
        Number of sites grouped together, see :meth:`group_sites`.
    segment_boundaries : tuple of :class:`~tenpy.linalg.np_conserved.Array` | (None, None)
        Only defined for 'segment' `bc` if :meth:`canonical_form_finite` has been called.
        If defined, it contains the `U_L` and `V_R` that would be returned by that function.
    _AC : list of :class:`~tenpy.linalg.np_conserved.Array`
        The 'center-site' tensors of the MPS. Labels are ``vL, vR, p`` (in any order).
    _AL : list of :class:`~tenpy.linalg.np_conserved.Array`
        The 'left-orthonormal' tensors of the MPS. Labels are ``vL, vR, p`` (in any order).
    _AR : list of :class:`~tenpy.linalg.np_conserved.Array`
        The 'right-orthonormal' tensors of the MPS. Labels are ``vL, vR, p`` (in any order).
    _C : list of :class:`~tenpy.linalg.np_conserved.Array`
        The center matrices on the left of site `i`, Labels are ``vL, vR`` (in any order).
    _valid_forms : dict
        Class attribute.
        Mapping for canonical forms to a tuple ``(nuL, nuR)`` indicating that
        ``self._B[i] = s[i]**nuL -- Gamma[i] -- s[i]**nuR`` is saved.
    _valid_bc : tuple of str
        Class attribute. Possible valid boundary conditions.
    _transfermatrix_keep : int
        How many states to keep at least when diagonalizing a :class:`TransferMatrix`.
        Important if the state develops a near-degeneracy.
    _p_label, _B_labels : list of str
        Class attribute. `_p_label` defines the physical legs of the B-tensors, `_B_labels` lists
        all the labels of the B tensors. Used by methods like :meth:`get_theta` to avoid
        the necessity of re-implementations for derived classes like the
        :class:`~tenpy.networks.purification_mps.Purification_MPS` if just the number of physical
        legs changed.
    """

    # valid boundary conditions. Don't overwrite this!
    _valid_bc = ('infinite', )
    # All labels of each tensor in _C (order is used!)
    _C_labels = ['vL', 'vR']

    # Labels for other tensors are inherited from MPS.

    def __init__(self, sites, ALs, ARs, ACs, Cs, norm=1.):
        warnings.warn('UniformMPS is a new experimental feature and not as well-tested as the '
                      'rest of the library', BetaWarning, stacklevel=2)
        self.sites = list(sites)
        self.chinfo = self.sites[0].leg.chinfo
        self.dtype = dtype = np.result_type(*ALs)
        self.form = [None] * len(ARs)
        self.bc = 'infinite'  # one of ``'finite', 'infinite', 'segment'``.
        self.norm = norm
        self.grouped = 1
        self.segment_boundaries = (None, None)
        self.valid_umps = False  # Need to check that AL[n] C[n+1] = AC[n] and C[n] AR[n] = AC[n]
        self.diagonal_gauge = False  # Are all C matrices diagonal?

        # make copies of 4 types of tensors
        self._AR = [AR.astype(dtype, copy=True).itranspose(self._B_labels) for AR in ARs]
        self._AL = [AL.astype(dtype, copy=True).itranspose(self._B_labels) for AL in ALs]
        self._AC = [AC.astype(dtype, copy=True).itranspose(self._B_labels) for AC in ACs]
        self._C = [C.astype(dtype, copy=True).itranspose(self._C_labels) for C in Cs]
        # center matrix on the left of site `i`

        self._transfermatrix_keep = 1
        self.test_sanity()

    def test_sanity(self):
        """Sanity check, raises ValueErrors, if something is wrong."""
        assert self.grouped == 1
        assert self.segment_boundaries == (None, None)

        if self.bc not in self._valid_bc:
            raise ValueError("invalid boundary condition: " + repr(self.bc))
        if len(self._AL) != self.L:
            raise ValueError("wrong len of self._AL")
        if len(self._AR) != self.L:
            raise ValueError("wrong len of self._AR")
        if len(self._AC) != self.L:
            raise ValueError("wrong len of self._AC")
        if len(self._C) != self.L:
            raise ValueError("wrong len of self._C")
        assert len(self.form) == self.L

        for i, As in enumerate(zip(self._AL, self._AR, self._AC)):
            AL, AR, AC = As
            if AL.get_leg_labels() != self._B_labels:
                raise ValueError("AL has wrong labels {0!r}, expected {1!r}".format(
                    AL.get_leg_labels(), self._B_labels))
            if AR.get_leg_labels() != self._B_labels:
                raise ValueError("AR has wrong labels {0!r}, expected {1!r}".format(
                    AR.get_leg_labels(), self._B_labels))
            if AC.get_leg_labels() != self._B_labels:
                raise ValueError("AC has wrong labels {0!r}, expected {1!r}".format(
                    AC.get_leg_labels(), self._B_labels))
            AR.get_leg('vL').test_contractible(self._C[i].get_leg('vR'))
            AR.get_leg('vL').test_contractible(self._AC[(i - 1) % self.L].get_leg('vR'))
            AL.get_leg('vR').test_contractible(self._C[(i + 1) % self.L].get_leg('vL'))
            AL.get_leg('vR').test_contractible(self._AC[(i + 1) % self.L].get_leg('vL'))

        return self.test_validity()

    def test_validity(self, cutoff=1.e-8):
        """Check if AL C = AC and C AR = AC

        To have a valid MPS and take measurements, we require this to be true. This will be true after VUMPS.
        No measurements should actually be done on a UniformMPS; convert back to MPS.
        """
        err = np.empty((self.L, 3), dtype=float)
        for i in range(self.L):
            AL, AR, AC, C1, C2 = self.get_AL(i), self.get_AR(i), self.get_AC(i), self.get_C(
                i), self.get_C(i + 1)
            ALC2 = npc.tensordot(AL, C2, axes=['vR', 'vL']).itranspose(self._B_labels)
            C1AR = npc.tensordot(C1, AR, axes=['vR', 'vL']).itranspose(self._B_labels)

            err[i, 0] = npc.norm((
                ALC2 /
                npc.tensordot(ALC2, C1AR.conj(), axes=(['vL', 'p', 'vR'], ['vL*', 'p*', 'vR*']))) -
                                 C1AR)
            err[i, 1] = npc.norm(
                (ALC2 /
                 npc.tensordot(ALC2, AC.conj(), axes=(['vL', 'p', 'vR'], ['vL*', 'p*', 'vR*']))) -
                AC)
            err[i, 2] = npc.norm(
                (C1AR /
                 npc.tensordot(C1AR, AC.conj(), axes=(['vL', 'p', 'vR'], ['vL*', 'p*', 'vR*']))) -
                AC)

        self.valid_umps = np.max(err) < cutoff
        logger.info(
            f'UniformMPS is {"valid" if self.valid_umps else "invalid"} with max error {np.max(err):.5f}.'
        )
        return err

    def copy(self):
        """Returns a copy of `self`.

        The copy still shares the sites, chinfo, and LegCharges, but the values of
        the tensors are deeply copied.
        """
        cp = self.__class__(self.sites, self._AL, self._AR, self._AC, self._C, self.norm)
        cp.grouped = self.grouped
        cp._transfermatrix_keep = self._transfermatrix_keep
        cp.segment_boundaries = self.segment_boundaries
        return cp

    def save_hdf5(self, hdf5_saver, h5gr, subpath):
        """Export `self` into a HDF5 file.

        This method saves all the data it needs to reconstruct `self` with :meth:`from_hdf5`.

        Specifically, it saves
        :attr:`sites`,
        :attr:`chinfo` (under these names),
        :attr:`_AL` as ``"tensors_AL"``,
        :attr:`_AR` as ``"tensors_AR"``,
        :attr:`_AC` as ``"tensors_AC"``,
        :attr:`_C` as ``"tensors_C"``,
        Moreover, it saves :attr:`norm`, :attr:`L`, :attr:`grouped` and
        :attr:`_transfermatrix_keep` (as "transfermatrix_keep") as HDF5 attributes, as well as
        the maximum of :attr:`chi` under the name "max_bond_dimension".

        Parameters
        ----------
        hdf5_saver : :class:`~tenpy.tools.hdf5_io.Hdf5Saver`
            Instance of the saving engine.
        h5gr : :class`Group`
            HDF5 group which is supposed to represent `self`.
        subpath : str
            The `name` of `h5gr` with a ``'/'`` in the end.
        """
        hdf5_saver.save(self.sites, subpath + "sites")
        hdf5_saver.save(self._AL, subpath + "tensors_AL")
        hdf5_saver.save(self._AR, subpath + "tensors_AR")
        hdf5_saver.save(self._AC, subpath + "tensors_AC")
        hdf5_saver.save(self._C, subpath + "tensors_C")
        hdf5_saver.save(self.chinfo, subpath + "chinfo")
        segment_boundaries = getattr(self, "segment_boundaries", (None, None))
        hdf5_saver.save(self.segment_boundaries, subpath + "segment_boundaries")
        h5gr.attrs["valid_umps"] = self.valid_umps
        h5gr.attrs["norm"] = self.norm
        h5gr.attrs["grouped"] = self.grouped
        h5gr.attrs["transfermatrix_keep"] = self._transfermatrix_keep
        h5gr.attrs["L"] = self.L  # not needed for loading, but still useful metadata
        h5gr.attrs["max_bond_dimension"] = np.max(self.chi)  # same

    def to_MPS(self, cutoff=1.e-16, check_overlap=False):
        """Convert UniformMPS to MPS.

        We return the AR matrix for each site and the DIAGONAL S
        matrix to the right of each site. Thus we must make sure that the C matrices
        are converted to diagonal matrices first.

        Parameters
        ----------
        cutoff : float
            During DMRG with a mixer, `S` may be a matrix for which we need the inverse.
            This is calculated as the Penrose pseudo-inverse, which uses a cutoff for the
            singular values.
        check_overlap: bool
            Since AL C = C AR is not identically true, the MPS defined by AL and AR are not exactly the same.
            We can compute the overlap of the two to check.

        Returns
        -------
        psi : :class:`~tenpy.networks.mps.MPS`
            The right-canonical form converted from the uniform MPS.
        """

        if self.diagonal_gauge == False:
            self.to_diagonal_gauge(cutoff=cutoff, check_overlap=check_overlap)

        self.test_validity()

        MPS_B = MPS(self.sites, self._AR, self._S, bc='infinite', form='B', norm=1.)

        MPS_B.canonical_form()
        if check_overlap:
            MPS_A = MPS(self.sites, self._AL, self._S, bc='infinite', form='A', norm=1.)
            MPS_A.canonical_form()  # [TODO] should we do this? It might be expensive.
            overlap_AB = np.abs(MPS_B.overlap(MPS_A, understood_infinite=True))
            logger.info(
                f'Overlap of UniformMPS constructed from ARs with UniformMPS constructed with ALs: {overlap_AB:.10f}'
            )
            if not np.isclose(overlap_AB, 1):
                logger.warning(
                    f"overlap not close to 1: {overlap_AB:.10f}.")
        return MPS_B

    def to_diagonal_gauge(self, cutoff=1.e-16, check_overlap=False):
        """
        Convert a UniformMPS to diagonal gauge, i.e. where all of the bond matrices are diagonal.

        Parameters
        ----------
        cutoff : float
            Cutoff for the singular values.
        check_overlap: bool
            Check the overlap between the state before and after changing to diagonal gauge.
        """
        if check_overlap:
            old_uMPS = self.copy()

        self._S = []  # Empty out np.arrays on each bond.

        if self.L > 1 and cutoff > 0.0:
            logger.warning(
                "'sv_cutoff' cannot be non-zero for multi-site unit cell as this messes with the transfer matrix."
            )
            cutoff = 0.0

        for i in range(self.L):
            # For each bond matrix,
            C = self.get_C(i)
            U, VH = self._diagonal_gauge_C(C, i, cutoff)
            if i % self.L == 0:
                self.left_U = U
                self.right_U = VH

            self._diagonal_gauge_AC(U, VH, i)

        self._S.append(self._S[0])
        assert len(self._S) == self.L + 1

        self.diagonal_gauge = True

        if check_overlap:
            overlap = self.overlap(old_uMPS, understood_infinite=True)
            logger.info(f'Overlap of original UniformMPS with diagonal UniformMPS: {overlap:.10f}')

    def _diagonal_gauge_C(self, theta, i0, cutoff):
        """
        Diagonalize bond matrix theta and update ALs and ARs on sites on the boundary of the bond.
        """
        U, S, VH = npc.svd(theta,
                           cutoff=cutoff,
                           qtotal_LR=[theta.qtotal, None],
                           inner_labels=['vR', 'vL'])

        theta = npc.diag(S, VH.get_leg('vL'), labels=['vL', 'vR'])

        self.set_B(i0 - 1, npc.tensordot(self.get_B(i0 - 1, 'AL'), U, axes=(['vR'], ['vL'])), 'AL')
        self.set_B(
            i0,
            npc.tensordot(U.conj(), self.get_B(i0, 'AL'),
                          axes=(['vL*'], ['vL'])).ireplace_label('vR*', 'vL'), 'AL')

        self.set_B(i0, npc.tensordot(VH, self.get_B(i0, 'AR'), axes=(['vR'], ['vL'])), 'AR')
        self.set_B(
            i0 - 1,
            npc.tensordot(self.get_B(i0 - 1, 'AR'), VH.conj(),
                          axes=(['vR'], ['vR*'])).ireplace_label('vL*', 'vR'), 'AR')
        self._S.append(S)
        self.set_C(i0, theta)

        return U, VH

    def _diagonal_gauge_AC(self, U, VH, i0):
        """
        Given U and VH from diagonalizing the center matrix C compute the corresponding AC.
        """

        theta = self.get_B(i0, 'AC')
        theta = npc.tensordot(U.conj(), theta, axes=(['vL*'], ['vL'])).ireplace_label('vR*', 'vL')
        self.set_B(i0, theta, 'AC')

        theta = self.get_B(i0 - 1, 'AC')
        theta = npc.tensordot(theta, VH.conj(), axes=(['vR'], ['vR*'])).ireplace_label('vL*', 'vR')
        self.set_B(i0 - 1, theta, 'AC')

    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):
        """Load instance from a HDF5 file.

        This method reconstructs a class instance from the data saved with :meth:`save_hdf5`.

        Parameters
        ----------
        hdf5_loader : :class:`~tenpy.tools.hdf5_io.Hdf5Loader`
            Instance of the loading engine.
        h5gr : :class:`Group`
            HDF5 group which is represent the object to be constructed.
        subpath : str
            The `name` of `h5gr` with a ``'/'`` in the end.

        Returns
        -------
        obj : cls
            Newly generated class instance containing the required data.
        """
        obj = cls.__new__(cls)  # create class instance, no __init__() call
        hdf5_loader.memorize_load(h5gr, obj)

        obj.sites = hdf5_loader.load(subpath + "sites")
        obj._AL = hdf5_loader.load(subpath + "tensors_AL")
        obj._AR = hdf5_loader.load(subpath + "tensors_AR")
        obj._AC = hdf5_loader.load(subpath + "tensors_AC")
        obj._C = hdf5_loader.load(subpath + "tensors_C")
        obj.bc = 'infinite'
        obj.norm = hdf5_loader.get_attr(h5gr, "norm")
        obj.valid_umps = hdf5_loader.get_attr(h5gr, "valid_umps")
        obj.form = [None] * len(obj._AR)

        obj.grouped = hdf5_loader.get_attr(h5gr, "grouped")
        obj._transfermatrix_keep = hdf5_loader.get_attr(h5gr, "transfermatrix_keep")
        obj.chinfo = hdf5_loader.load(subpath + "chinfo")
        obj.dtype = np.find_common_type([B.dtype for B in obj._AR], [])
        if "segment_boundaries" in h5gr:
            obj.segment_boundaries = hdf5_loader.load(subpath + "segment_boundaries")
        else:
            obj.segment_boundaries = (None, None)
        obj.test_sanity()
        return obj

    @classmethod
    def from_MPS(cls, psi):
        """
        Convert an infinite MPS to a uniform MPS.

        Parameters
        ----------
        psi : :class:`~tenpy.networks.mps.MPS`
            Infinite MPS which we want to change to a uniform one.

        Returns
        -------
        psi : :class:`UniformMPS`
            The resulting uniform MPS.
        """
        # make copies of 4 types of tensors
        dtype = psi.dtype
        AR = [
            psi.get_B(i, form='B').astype(dtype, copy=True).itranspose(cls._B_labels)
            for i in range(psi.L)
        ]
        AC = [
            psi.get_B(i, form='Th').astype(dtype, copy=True).itranspose(cls._B_labels)
            for i in range(psi.L)
        ]
        AL = [
            psi.get_B(i, form='A').astype(dtype, copy=True).itranspose(cls._B_labels)
            for i in range(psi.L)
        ]
        C = []
        for i in range(psi.L):
            C_ = npc.diag(psi.get_SL(i), AL[i].get_leg('vL'),
                          labels=['vL', 'vR'])  # center matrix on the left of site `i`
            C.append(C_.astype(dtype, copy=True).itranspose(cls._C_labels))
        obj = cls(psi.sites, AL, AR, AC, C, psi.norm)
        obj.bc = psi.bc
        obj.grouped = psi.grouped
        obj.segment_boundaries = psi.segment_boundaries
        obj.diagonal_gauge = True
        obj.valid_umps = False  # Need to check that AL[n] C[n+1] = AC[n] and C[n] AR[n] = AC[n]

        # need to define S, since diagonal_gauge = True
        obj._S = [psi.get_SL(i).astype(dtype, copy=True) for i in range(psi.L)]

        obj._transfermatrix_keep = psi._transfermatrix_keep
        obj.test_sanity()
        return obj

    @classmethod
    def from_lat_product_state(cls, lat, p_state, **kwargs):
        raise NotImplementedError("Not valid for UniformMPS!")

    @classmethod
    def from_product_state(cls,
                           sites,
                           p_state,
                           bc='finite',
                           dtype=np.float64,
                           permute=True,
                           form='B',
                           chargeL=None):
        raise NotImplementedError("Not valid for UniformMPS.")

    @classmethod
    def from_Bflat(cls, sites, ALflat, ARflat, ACflat, Cflat, dtype=None, permute=True, legL=None):
        """Construct a matrix product state from a set of numpy arrays and singular vals.

        Parameters
        ----------
        sites : list of :class:`~tenpy.networks.site.Site`
            The sites defining the local Hilbert space.
        A{L,R,C}flat : iterable of numpy ndarrays
            The matrices defining the MPS on each site, with legs ``'p', 'vL', 'vR'``
            (physical, virtual left/right).
        Cflat : iterable of numpy ndarrays
            The matrices defining the bond matrix on each site, with legs ``'vL', 'vR'``
            (virtual left/right).
        dtype : type or string
            The data type of the array entries. Defaults to the common dtype of `Bflat`.
        permute : bool
            The :class:`~tenpy.networks.Site` might permute the local basis states if charge
            conservation gets enabled.
            If `permute` is True (default), we permute the given `Bflat` locally according to
            each site's :attr:`~tenpy.networks.Site.perm`.
            The `p_state` argument should then always be given as if `conserve=None` in the Site.
        leg_L : LegCharge | ``None``
            Leg charges at bond 0, which are purely conventional.
            If ``None``, use trivial charges.

        Returns
        -------
        mps : :class:`UniformMPS`
            An MPS with the `flat` matrices converted to npc arrays.
        """
        sites = list(sites)
        L = len(sites)
        ALflat = list(ALflat)
        ARflat = list(ARflat)
        ACflat = list(ACflat)
        Cflat = list(Cflat)
        if len(ALflat) != L:
            raise ValueError("Length of ALflat does not match number of sites.")
        if len(ARflat) != L:
            raise ValueError("Length of ARflat does not match number of sites.")
        if len(ACflat) != L:
            raise ValueError("Length of ACflat does not match number of sites.")
        if len(Cflat) != L:
            raise ValueError("Length of Cflat does not match number of sites.")
        ci = sites[0].leg.chinfo
        if legL is None:
            legL = npc.LegCharge.from_qflat(ci, [ci.make_valid(None)] * Cflat[0].shape[0])
            legL = legL.bunch()[1]
        ALs = []
        ARs = []
        ACs = []
        Cs = []
        if dtype is None:
            dtype = np.dtype(np.common_type(*ALflat))
        for i, site in enumerate(sites):
            AL = np.array(ALflat[i], dtype)
            AR = np.array(ARflat[i], dtype)
            AC = np.array(ACflat[i], dtype)
            C = np.array(Cflat[i], dtype)
            if permute:
                AL = AL[site.perm, :, :]
                AR = AR[site.perm, :, :]
                AC = AC[site.perm, :, :]

            # calculate the LegCharge of the right leg of C
            Clegs = [legL, None]
            Clegs = npc.detect_legcharge(
                C, ci, Clegs, None,
                qconj=-1)  # Even though C has no physical leg, it can have charge.
            C = npc.Array.from_ndarray(C, Clegs, dtype)
            C.iset_leg_labels(['vL', 'vR'])
            Cs.append(C)

            ARlegs = [site.leg, Clegs[-1].conj(), None]
            ARlegs = npc.detect_legcharge(AR, ci, ARlegs, None, qconj=-1)
            AR = npc.Array.from_ndarray(AR, ARlegs, dtype)
            AR.iset_leg_labels(['p', 'vL', 'vR'])
            ARs.append(AR)

            ALlegs = [site.leg, legL, None]
            ALlegs = npc.detect_legcharge(AL, ci, ALlegs, None, qconj=-1)
            AL = npc.Array.from_ndarray(AL, ALlegs, dtype)
            AL.iset_leg_labels(['p', 'vL', 'vR'])
            ALs.append(AL)

            AClegs = [site.leg, legL, None]
            AClegs = npc.detect_legcharge(AC, ci, AClegs, None, qconj=-1)
            AC = npc.Array.from_ndarray(AC, AClegs, dtype)
            AC.iset_leg_labels(['p', 'vL', 'vR'])
            ACs.append(AC)

            legL = ALlegs[-1].conj()  # prepare for next `i`

        # for an iMPS, the last leg has to match the first one.
        # so we need to gauge `qtotal` of the last tensors such that the right leg matches.
        chdiff = ALs[-1].get_leg('vR').charges[0] - AL[0].get_leg('vL').charges[0]
        ALs[-1] = ALs[-1].gauge_total_charge('vR', ci.make_valid(chdiff))
        ACs[-1] = ACs[-1].gauge_total_charge('vR', ci.make_valid(chdiff))

        chdiff = ARs[-1].get_leg('vR').charges[0] - ARs[0].get_leg('vL').charges[0]
        ARs[-1] = ARs[-1].gauge_total_charge('vR', ci.make_valid(chdiff))
        return cls(sites, ALs, ARs, ACs, Cs)

    @classmethod
    def from_full(cls,
                  sites,
                  psi,
                  form=None,
                  cutoff=1.e-16,
                  normalize=True,
                  bc='finite',
                  outer_S=None):
        raise NotImplementedError("Not valid for UniformMPS.")

    @classmethod
    def from_singlets(cls,
                      site,
                      L,
                      pairs,
                      up='up',
                      down='down',
                      lonely=[],
                      lonely_state='up',
                      bc='finite'):
        raise NotImplementedError("Not valid for UniformMPS.")

    @property
    def chi(self):
        """Dimensions of the (nontrivial) virtual bonds."""
        # s.shape[0] == len(s) for 1D numpy array, but works also for a 2D npc Array.
        return [min(C.shape) for C in self._C[self.nontrivial_bonds]]

    def get_B(self, i, form='B', copy=False, cutoff=1.e-16, label_p=None):
        """Return (view of) `B` at site `i` in canonical form.

        Parameters
        ----------
        i : int
            Index choosing the site.
        form : ``'B'/'AR' | 'A'/'AL' | 'Th'/'AC' | None`` | tuple(float, float)
            The (canonical) form of the returned B.
            For ``None``, return the matrix in 'B'-form.
        copy : bool
            Whether to return a copy even if `form` matches the current form.
        cutoff : float
            During DMRG with a mixer, `S` may be a matrix for which we need the inverse.
            This is calculated as the Penrose pseudo-inverse, which uses a cutoff for the
            singular values.
        label_p : None | str
            Ignored by default (``None``).
            Otherwise replace the physical label ``'p'`` with ``'p'+label_p'``.
            (For derived classes with more than one "physical" leg, replace all the physical leg
            labels accordingly.)

        Returns
        -------
        B : :class:`~tenpy.linalg.np_conserved.Array`
            The MPS 'matrix' `B` at site `i` with leg labels ``'vL', 'p', 'vR'``.
            May be a view of the matrix (if ``copy=False``),
            or a copy (if the form changed or ``copy=True``).
        """
        if form is None:
            return self.get_AR(i, copy=copy, label_p=label_p)
        elif form == 'A' or form == (1., 0.) or form == 'AL':
            return self.get_AL(i, copy=copy, label_p=label_p)
        elif form == 'B' or form == (0., 1.) or form == 'AR':
            return self.get_AR(i, copy=copy, label_p=label_p)
        elif form == 'Th' or form == (1., 1.) or form == 'AC':
            return self.get_AC(i, copy=copy, label_p=label_p)
        else:
            raise NotImplementedError(f"Form {form!r} is not valid for UniformMPS.")

    def get_AL(self, i, copy=False, label_p=None):
        """
        Return (view of) `AL` at site `i` in canonical form.
        """
        i = self._to_valid_index(i)
        AL = self._AL[i]
        if copy:
            AL = AL.copy()
        if label_p is not None:
            AL = self._replace_p_label(AL, label_p)
        return AL

    def get_AR(self, i, copy=False, label_p=None):
        """
        Return (view of) `AR` at site `i` in canonical form.
        """
        i = self._to_valid_index(i)
        AR = self._AR[i]
        if copy:
            AR = AR.copy()
        if label_p is not None:
            AR = self._replace_p_label(AR, label_p)
        return AR

    def get_AC(self, i, copy=False, label_p=None):
        """
        Return (view of) `AC` at site `i` in canonical form.
        """
        i = self._to_valid_index(i)
        AC = self._AC[i]
        if copy:
            AC = AC.copy()
        if label_p is not None:
            AC = self._replace_p_label(AC, label_p)
        return AC

    def get_C(self, i, copy=False):
        """Return center matrix C on the left of site `i`"""
        i = self._to_valid_index(i)
        C = self._C[i]
        if copy:
            C = C.copy()
        return C

    def set_B(self, i, B, form='B'):
        """Set tensor `B` at site `i`.

        Parameters
        ----------
        i : int
            Index choosing the site.
        B : :class:`~tenpy.linalg.np_conserved.Array`
            The 'matrix' at site `i`. No copy is made!
            Should have leg labels ``'vL', 'p', 'vR'`` (not necessarily in that order).
        form : ``'B'/'AR' | 'A'/'AL' | 'Th'/'AC'`` | tuple(float, float)
            The (canonical) form of the `B` to set.
        """
        if form == 'A' or form == (1., 0.) or form == 'AL':
            return self.set_AL(i, B)
        elif form == 'B' or form == (0., 1.) or form == 'AR':
            return self.set_AR(i, B)
        elif form == 'Th' or form == (1., 1.) or form == 'AC':
            return self.set_AC(i, B)
        else:
            raise NotImplementedError(f"Form {list(form)!r} is not valid for UniformMPS.")

    def set_AL(self, i, AL):
        """
        Set `AL` at site `i`
        """
        i = self._to_valid_index(i)
        self.dtype = np.promote_types(self.dtype, AL.dtype)
        self._AL[i] = AL.itranspose(self._B_labels)

    def set_AR(self, i, AR):
        """
        Set `AR` at site `i`
        """
        i = self._to_valid_index(i)
        self.dtype = np.promote_types(self.dtype, AR.dtype)
        self._AR[i] = AR.itranspose(self._B_labels)

    def set_AC(self, i, AC):
        """
        Set `AC` at site `i`
        """
        i = self._to_valid_index(i)
        self.dtype = np.promote_types(self.dtype, AC.dtype)
        self._AC[i] = AC.itranspose(self._B_labels)

    def set_C(self, i, C):
        """
        Set `C` left of site `i`
        """
        i = self._to_valid_index(i)
        self.dtype = np.promote_types(self.dtype, C.dtype)
        self._C[i] = C.itranspose(self._C_labels)

    def set_svd_theta(self, i, theta, trunc_par=None, update_norm=False):
        raise NotImplementedError("Not valid for UniformMPS.")

    def get_SL(self, i):
        return self.get_C(i)

    def get_SR(self, i):
        return self.get_C(i + 1)

    def set_SL(self, i, S):
        self.set_C(i, S)

    def set_SR(self, i, S):
        self.set_C(i + 1, S)

    def get_theta(self, i, n=2, cutoff=1.e-16, formL=1., formR=1.):
        """Calculates the `n`-site wavefunction on ``sites[i:i+n]``.

        Parameters
        ----------
        i : int
            Site index.
        n : int
            Number of sites. The result lives on ``sites[i:i+n]``.
        cutoff : float
            During DMRG with a mixer, `S` may be a matrix for which we need the inverse.
            This is calculated as the Penrose pseudo-inverse, which uses a cutoff for the
            singular values.
        formL : float
            Exponent for the singular values to the left. (Not used for UniformMPS)
        formR : float
            Exponent for the singular values to the right. (Not used for UniformMPS)
        Returns
        -------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            The n-site wave function with leg labels ``vL, p0, p1, .... p{n-1}, vR``.
            In Vidal's notation (with s=lambda, G=Gamma):
            ``theta = s**form_L G_i s G_{i+1} s ... G_{i+n-1} s**form_R``.
        """
        i = self._to_valid_index(i)
        if n == 1:
            return self.get_B(i, (1., 1.), True, cutoff, '0')
        elif n < 1:
            raise ValueError("n needs to be larger than 0")
        # n >= 2: contract some B's
        theta = self.get_B(i, "AC", False, cutoff, '0')  # site i in Th form
        for k in range(1, n):  # non-empty range
            j = self._to_valid_index(i + k)
            B = self.get_B(j, "AR", False, cutoff, str(k))
            theta = npc.tensordot(theta, B, axes=['vR', 'vL'])
        return theta

    def convert_form(self, new_form='B'):
        raise NotImplementedError("Not valid for UniformMPS.")

    def enlarge_mps_unit_cell(self, factor=2):
        """Repeat the unit cell for infinite uniform MPS boundary conditions; in place.

        Parameters
        ----------
        factor : int
            The new number of sites in the unit cell will be increased from `L` to ``factor*L``.
        """
        if int(factor) != factor:
            raise ValueError("`factor` should be integer!")
        if factor <= 1:
            raise ValueError("can't shrink!")
        if self.bc == 'segment':
            raise ValueError("can't enlarge segment MPS")
        self.sites = factor * self.sites
        self._AL = factor * self._AL
        self._AR = factor * self._AR
        self._AC = factor * self._AC
        self._C = factor * self._C
        self.test_sanity()

    def roll_mps_unit_cell(self, shift=1):
        """Shift the section we define as unit cell of an infinite MPS; in place.

        Suppose we have a unit cell with tensors ``[A, B, C, D]`` (repeated on both sites).
        With ``shift = 1``, the new unit cell will be ``[D, A, B, C]``,
        whereas ``shift = -1`` will give ``[B, C, D, A]``.

        Parameters
        ----------
        shift : int
            By how many sites to move the tensors to the right.
        """
        if self.finite:
            raise ValueError("makes only sense for infinite boundary conditions")
        inds = np.roll(np.arange(self.L), shift)
        self.sites = [self.sites[i] for i in inds]
        self._AL = [self._AL[i] for i in inds]
        self._AR = [self._AR[i] for i in inds]
        self._AC = [self._AC[i] for i in inds]
        self._C = [self._C[i] for i in inds]

    def spatial_inversion(self):
        """Perform a spatial inversion along the MPS.

        Exchanges the first with the last tensor and so on,
        i.e., exchange site `i` with site ``L-1 - i``.
        This is equivalent to a mirror/reflection with the bond left of L/2 (even L) or the site
        (L-1)/2 (odd L) as a fixpoint.
        For infinite MPS, the bond between MPS unit cells is another fix point.
        """
        self.sites = self.sites[::-1]
        self._AL = [
            AL.replace_labels(['vL', 'vR'], ['vR', 'vL']).transpose(self._B_labels)
            for AL in self._AL[::-1]
        ]
        self._AR = [
            AR.replace_labels(['vL', 'vR'], ['vR', 'vL']).transpose(self._B_labels)
            for AR in self._AR[::-1]
        ]
        self._AC = [
            AC.replace_labels(['vL', 'vR'], ['vR', 'vL']).transpose(self._B_labels)
            for AC in self._AC[::-1]
        ]
        self._C = [
            C.replace_labels(['vL', 'vR'], ['vR', 'vL']).transpose(self._C_labels)
            for C in self._C[::-1]
        ]
        self.test_sanity()
        return self

    def group_sites(self, n=2, grouped_sites=None):
        raise NotImplementedError("Not valid for UniformMPS.")

    def group_split(self, trunc_par=None):
        raise NotImplementedError("Not valid for UniformMPS.")

    def get_grouped_mps(self, blocklen):
        raise NotImplementedError("Not valid for UniformMPS.")

    def extract_segment(self, first, last):
        raise NotImplementedError("Not valid for UniformMPS.")

    def get_total_charge(self, only_physical_legs=False):
        """Calculate and return the `qtotal` of the whole MPS (when contracted).

        If set, the :attr:`segment_boundaries` are included (unless `only_physical_legs` is True).

        Parameters
        ----------
        only_physical_legs : bool
            For ``'finite'`` boundary conditions, the total charge can be gauged away
            by changing the LegCharge of the trivial legs on the left and right of the MPS.
            (Not possible for UniformMPS)

        Returns
        -------
        qtotal : charges
            The sum of the `qtotal` of the individual `B` tensors.
        """
        assert only_physical_legs == False, "Not possible for UniformMPS"
        # Assume self.segment_boundaries is None, None for UniformMPS
        tensors_AL = self._AL
        qtotal_AL = np.sum([AL.qtotal for AL in tensors_AL], axis=0)
        qtotal_AL = self.chinfo.make_valid(qtotal_AL)

        tensors_AR = self._AR
        qtotal_AR = np.sum([AR.qtotal for AR in tensors_AR], axis=0)
        qtotal_AR = self.chinfo.make_valid(qtotal_AR)
        qtotal_AR.test_equal(qtotal_AL)

        return qtotal_AR

    def gauge_total_charge(self, qtotal=None, vL_leg=None, vR_leg=None):
        raise NotImplementedError("Who knows if this is valid for UniformMPS?")

    def entanglement_entropy(self, n=1, bonds=None, for_matrix_S=True):
        #assert self.valid_umps
        assert for_matrix_S, "UniformMPS do not have diagonal C matrices."
        return super().entanglement_entropy(n, bonds, for_matrix_S)

    def entanglement_entropy_segment(self, segment=[0], first_site=None, n=1):
        raise NotImplementedError("Convert UniformMPS to MPS for calculations involving S.")

    def entanglement_entropy_segment2(self, segment, n=1):
        raise NotImplementedError("Convert UniformMPS to MPS for calculations involving S.")

    def entanglement_spectrum(self, by_charge=False):
        raise NotImplementedError("Convert UniformMPS to MPS for calculations involving S.")

    def get_rho_segment(self, segment):
        raise NotImplementedError("Convert UniformMPS to MPS for calculations involving S.")

    def probability_per_charge(self, bond=0):
        raise NotImplementedError("Convert UniformMPS to MPS for calculations involving S.")

    def average_charge(self, bond=0):
        raise NotImplementedError("Convert UniformMPS to MPS for calculations involving S.")

    def charge_variance(self, bond=0):
        raise NotImplementedError("Convert UniformMPS to MPS for calculations involving S.")

    def mutinf_two_site(self, max_range=None, n=1):
        raise NotImplementedError("Convert UniformMPS to MPS for calculations involving S.")

    def overlap(self, other, charge_sector=None, ignore_form=False, **kwargs):
        """Compute overlap ``<self|other>``.

        Parameters
        ----------
        other : :class:`MPS`
            An MPS with the same physical sites.
        charge_sector : None | charges | ``0``
            Selects the charge sector in which the dominant eigenvector of the TransferMatrix is.
            ``None`` stands for *all* sectors, ``0`` stands for the sector of zero charges.
            If a sector is given, it *assumes* the dominant eigenvector is in that charge sector.
        ignore_form : bool
            For UniformMPS only ``False`` is possible.
        **kwargs :
            Further keyword arguments given to :meth:`TransferMatrix.eigenvectors`;
            only used for infinite boundary conditions.

        Returns
        -------
        overlap : dtype.type
            The contraction ``<self|other> * self.norm * other.norm``
            (i.e., taking into account the :attr:`norm` of both MPS).
            For an infinite MPS, ``<self|other>`` is the overlap per unit cell, i.e.,
            the largest eigenvalue of the TransferMatrix.
        """
        assert not ignore_form, "UniformMPS have both forms. Use one."
        return super().overlap(other,
                               charge_sector=charge_sector,
                               ignore_form=ignore_form,
                               **kwargs)

    def _contract_with_LP(self, C, i):
        assert self.valid_umps
        return super()._contract_with_LP(C, i)

    def _contract_with_RP(self, C, i):
        assert self.valid_umps
        return super()._contract_with_RP(C, i)

    def sample_measurements(self,
                            first_site=0,
                            last_site=None,
                            ops=None,
                            rng=None,
                            norm_tol=1.e-12):
        assert self.valid_umps
        return super().sample_measurements(self,
                                           first_site=first_site,
                                           last_site=last_site,
                                           ops=ops,
                                           rng=rng,
                                           norm_tol=norm_tol)

    def norm_test(self, force=False):
        if not force and not self.valid_umps:
            return np.zeros((self.L, 2), dtype=float)
        else:
            return super().norm_test()

    def canonical_form(self, **kwargs):
        raise NotImplementedError("Not valid for UniformMPS.")

    def canonical_form_finite(self, renormalize=True, cutoff=0., envs_to_update=None):
        raise NotImplementedError("Not valid for UniformMPS.")

    def canonical_form_infinite(self, renormalize=True, tol_xi=1.e6):
        raise NotImplementedError("Not valid for UniformMPS.")

    def correlation_length(self, target=1, tol_ev0=1.e-8, charge_sector=0):
        assert self.valid_umps
        return super().correlation_length(self, target=target, tol_ev0=tol_ev0,
                                          charge_sector=charge_sector)

    def add(self, other, alpha, beta, cutoff=1.e-15):
        raise NotImplementedError("Not valid for UniformMPS.")

    def apply_local_op(self, i, op, unitary=None, renormalize=False, cutoff=1.e-13):
        raise NotImplementedError("Not valid for UniformMPS.")

    def apply_product_op(self, ops, unitary=None, renormalize=False):
        raise NotImplementedError("Not valid for UniformMPS.")

    def perturb(self, randomize_params=None, close_1=True, canonicalize=None):
        raise NotImplementedError("Not valid for UniformMPS.")

    def swap_sites(self, i, swap_op='auto', trunc_par=None):
        raise NotImplementedError("Not valid for UniformMPS.")

    def permute_sites(self, perm, swap_op='auto', trunc_par=None, verbose=None):
        raise NotImplementedError("Not valid for UniformMPS.")

    def compute_K(self,
                  perm,
                  swap_op='auto',
                  trunc_par=None,
                  canonicalize=1.e-6,
                  verbose=None,
                  expected_mean_k=0.):
        raise NotImplementedError("Convert UniformMPS to MPS for calculations involving S.")

    def __str__(self):
        """Some status information about the UniformMPS."""
        res = [f"UniformMPS, L={self.L:d}, bc={self.bc!r}."]
        res.append(f"chi: {self.chi}")
        res.append(f"valid: {self.valid_umps}")
        if self.L > 10:
            res.append("first two sites: " + repr(self.sites[0]) + " " + repr(self.sites[1]))
        else:
            res.append("sites: " + " ".join([repr(s) for s in self.sites]))
        return "\n".join(res)

    def compress(self, options):
        raise NotImplementedError("Not valid for UniformMPS.")

    def compress_svd(self, trunc_par):
        raise NotImplementedError("Not valid for UniformMPS.")

    def _scale_axis_B(self, B, S, form_diff, axis_B, cutoff):
        raise NotImplementedError("Not valid for UniformMPS.")

    def _canonical_form_dominant_gram_matrix(self, bond0, transpose, tol_xi, guess=None):
        raise NotImplementedError("Not valid for UniformMPS.")

    def _canonical_form_correct_right(self, i1, Gr, eps=2. * np.finfo(np.double).eps):
        raise NotImplementedError("Not valid for UniformMPS.")

    def _canonical_form_correct_left(self, i1, Gl, Wr, eps=2. * np.finfo(np.double).eps):
        raise NotImplementedError("Not valid for UniformMPS.")

    def _gauge_compatible_vL_vR(self, other):
        raise NotImplementedError("Not valid for UniformMPS.")

    def outer_virtual_legs(self):
        vL = self._AR[0].get_leg('vL')
        vR = self._AL[-1].get_leg('vR')
        return vL, vR
