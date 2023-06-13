r"""This module contains a base class for a Uniform Matrix Product State (uMPS).

This is an extension of the matrix product state (MPS) class for tangent space
algorithms like VUMPS and TDVP (even though the current TDVP algorithm does not
use this).

A uMPS differs from a canonical MPS in the tensors that are stored on each site.
In a canonical MPS, we store a single tensor on each site and diagonal Schmidt
coefficients on each bond. From these, we can construct any desired form of a tensor
on each site; e.g. given B_i, we can construct A_i = S_i B_i S_{i+1}^{-1}. On every
site, we assume that $AS = SB$, which is guaranteed (up to numerical noise) after
calling canonical form. In a uMPS, however, we are not guaranteed that this
condition holds. Instead, we store an AL tensor (left canonical, A in MPS notation),
AR tensor (right canonical, B), and an AC tensor (one-site orthogonality center, Theta)
on each site. On each bond we store a C tensor that is not guaranteed to be diagonal.

A uMPS is only defined in the thermodynamic limit.

The functions in the class are mostly trivial copies of the functions from MPS that
account for the additional type of tensor structure.

"""
# Copyright 2018-2023 TeNPy Developers, GNU GPLv3

import numpy as np
import warnings
import random
from functools import reduce
import logging
logger = logging.getLogger(__name__)

from ..linalg import np_conserved as npc
from .mps import MPS

__all__ = ['uMPS']


class uMPS(MPS):
    r"""A Uniform Matrix Product State, only defined in the thermodynamic limit

    See MPS documentation for details.
    """

    # valid boundary conditions. Don't overwrite this!
    _valid_bc = ('infinite', )
    # All labels of each tensor in _C (order is used!)
    _C_labels = ['vL', 'vR']
    # Labels for other tensors are inhereted from MPS.

    def __init__(self, sites, ALs, ARs, ACs, Cs, norm=1.):
        self.sites = list(sites)
        self.chinfo = self.sites[0].leg.chinfo
        self.dtype = dtype = np.find_common_type([AL.dtype for AL in ALs], [])
        self.form = [None] * len(ARs)
        self.bc = 'infinite'  # one of ``'finite', 'infinite', 'segment'``.
        self.norm = norm
        self.grouped = 1
        self.segment_boundaries = (None, None)
        self.valid_umps = False # Need to check that AL[n] C[n+1] = AC[n] and C[n] AR[n] = AC[n]
        self.diagonal_gauge = False # Are all C matrices diagonal?

        # make copies of 4 types of tensors
        self._AR = [AR.astype(dtype, copy=True).itranspose(self._B_labels) for AR in ARs]
        self._AL = [AL.astype(dtype, copy=True).itranspose(self._B_labels) for AL in ALs]
        self._AC = [AC.astype(dtype, copy=True).itranspose(self._B_labels) for AC in ACs]
        self._C = [C.astype(dtype, copy=True).itranspose(self._C_labels) for C in Cs] # TODO, same initialization as S?
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

        #if np.any(self._C[self.L].to_ndarray() != self._C[0].to_ndarray()): # TODO: There has got to be a better way to check equality of npc arrays.
        #    raise ValueError("uMPS with C[0] != C[L]")

        self.test_validity()

    def test_validity(self, cutoff=1.e-8):
        """Check if AL C = AC and C AR = AC

        To have a valid MPS and take measurements, we require this to be true. This will be true after VUMPS.
        No measurements should actually be done on a uMPS; convert back to MPS.
        """
        err = np.empty((self.L, 3), dtype=float)
        for i in range(self.L):
            AL, AR, AC, C1, C2 = self.get_AL(i), self.get_AR(i), self.get_AC(i), self.get_C(i), self.get_C(i+1)
            ALC2 = npc.tensordot(AL, C2, axes=['vR', 'vL']).itranspose(self._B_labels)
            C1AR = npc.tensordot(C1, AR, axes=['vR', 'vL']).itranspose(self._B_labels)

            err[i, 0] = npc.norm((ALC2 / npc.tensordot(ALC2, C1AR.conj(), axes=(['vL', 'p', 'vR'], ['vL*', 'p*', 'vR*']))) - C1AR)
            err[i, 1] = npc.norm((ALC2 / npc.tensordot(ALC2, AC.conj(), axes=(['vL', 'p', 'vR'], ['vL*', 'p*', 'vR*']))) - AC)
            err[i, 2] = npc.norm((C1AR / npc.tensordot(C1AR, AC.conj(), axes=(['vL', 'p', 'vR'], ['vL*', 'p*', 'vR*']))) - AC)

        self.valid_umps = np.max(err) < cutoff
        logger.info('uMPS is %s with max error %.5e.', 'valid' if self.valid_umps else 'invalid', np.max(err))
        return err

    def copy(self):
        # __init__ makes deep copies of 4 types of tensors.
        cp = self.__class__(self.sites, self._AL, self._AR, self._AC, self._C, self.norm)
        cp.grouped = self.grouped
        cp._transfermatrix_keep = self._transfermatrix_keep
        cp.segment_boundaries = self.segment_boundaries
        return cp

    def save_hdf5(self, hdf5_saver, h5gr, subpath):
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
        h5gr.attrs["L"] = self.L  # not needed for loading, but still usefull metadata
        h5gr.attrs["max_bond_dimension"] = np.max(self.chi)  # same

    def to_MPS(self, SV_cutoff=0., overlap=False):
        """
        Convert uMPS to MPS. We return the AR matrix for each site and the DIAGONAL S
        matrix to the right of each site. Thus we must make sure that the C matrices
        are converted to diagonal matrices first.

        Since AL C = C AR is not identically true, the MPS defined by AL and AR are not exactly the same.
        We can compute the overlap of the two to check.
        """

        if self.diagonal_gauge == False:
            self.to_diagonal_gauge(SV_cutoff=SV_cutoff, overlap=overlap)

        self.test_validity()

        MPS_A = MPS(self.sites, self._AL, self._S, bc='infinite', form='A', norm=1.)
        MPS_B = MPS(self.sites, self._AR, self._S, bc='infinite', form='B', norm=1.)

        MPS_A.canonical_form() # [TODO] should we do this? It might be expensive.
        MPS_B.canonical_form()
        if overlap:
            overlap_AB = np.abs(MPS_B.overlap(MPS_A))
            logger.info('Overlap of uMPS constructed from ARs with uMPS constructed with ALs: %.10e.', overlap_AB)
            assert np.isclose(overlap_AB, 1)
        return MPS_B

    def to_diagonal_gauge(self, SV_cutoff=0., overlap=False):
        """
        Convert a uMPS to diagonal gauge, i.e. where all of the bond matrices are diagonal.
        """
        if overlap:
            old_uMPS = self.copy()

        self._S = []    # Empty out np.arrays on each bond.

        if self.L > 1 and SV_cutoff > 0.0:
            warnings.warn("'sv_cutoff' cannot be non-zero for multi-site unit cell as this messes with the transfer matrix.")
            SV_cutoff = 0.0


        for i in range(self.L):
            # For each bond matrix,
            C = self.get_C(i)
            U, VH = self._diagonal_gauge_C(C, i, SV_cutoff)
            if i % self.L == 0:
                self.left_U = U
                self.right_U = VH

            self._diagonal_gauge_AC(U, VH, i)

        self._S.append(self._S[0])
        assert len(self._S) == self.L + 1

        self.diagonal_gauge = True

        if overlap:
            overlap = self.overlap(old_uMPS)
            logger.info('Overlap of original uMPS with diagonal uMPS: %.10e.', overlap)
            assert np.isclose(overlap, 1)

    def _diagonal_gauge_C(self, theta, i0, SV_cutoff):
        """
        Diagonalize bond matrix theta and update ALs and ARs on sites on the boundary of the bond.
        """
        U, S, VH = npc.svd(theta,
                           cutoff=SV_cutoff,
                           qtotal_LR=[theta.qtotal, None],
                           inner_labels=['vR', 'vL'])

        theta = npc.diag(S, VH.get_leg('vL'), labels=['vL', 'vR'])

        self.set_B(i0-1, npc.tensordot(self.get_B(i0-1, 'AL'), U, axes=(['vR'], ['vL'])), 'AL')
        self.set_B(i0, npc.tensordot(U.conj(), self.get_B(i0, 'AL'), axes=(['vL*'], ['vL'])).ireplace_label('vR*', 'vL'), 'AL')

        self.set_B(i0, npc.tensordot(VH, self.get_B(i0, 'AR'), axes=(['vR'], ['vL'])), 'AR')
        self.set_B(i0-1, npc.tensordot(self.get_B(i0-1, 'AR'), VH.conj(), axes=(['vR'], ['vR*'])).ireplace_label('vL*', 'vR'), 'AR')
        self._S.append(S)
        self.set_C(i0, theta)

        return U, VH

    def _diagonal_gauge_AC(self, U, VH, i0):

        theta = self.get_B(i0, 'AC')
        theta = npc.tensordot(U.conj(), theta, axes=(['vL*'], ['vL'])).ireplace_label('vR*', 'vL')
        self.set_B(i0, theta, 'AC')

        theta = self.get_B(i0-1, 'AC')
        theta = npc.tensordot(theta, VH.conj(), axes=(['vR'], ['vR*'])).ireplace_label('vL*', 'vR')
        self.set_B(i0-1, theta, 'AC')

    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):
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
        obj = cls.__new__(cls)
        obj.sites = list(psi.sites)
        obj.chinfo = psi.sites[0].leg.chinfo
        obj.dtype = psi.dtype
        obj.form = [None] * len(psi._B)
        obj.bc = psi.bc
        obj.norm = psi.norm
        obj.grouped = psi.grouped
        obj.segment_boundaries = psi.segment_boundaries
        obj.valid_umps = False # Need to check that AL[n] C[n+1] = AC[n] and C[n] AR[n] = AC[n]

        # make copies of 4 types of tensors
        obj._AR = [psi.get_B(i, form='B').astype(obj.dtype, copy=True).itranspose(obj._B_labels) for i in range(psi.L)]
        obj._AC = [psi.get_B(i, form='Th').astype(obj.dtype, copy=True).itranspose(obj._B_labels) for i in range(psi.L)]
        obj._AL = [psi.get_B(i, form='A').astype(obj.dtype, copy=True).itranspose(obj._B_labels) for i in range(psi.L)]
        obj._C = []
        # There are L+1 S matrices in an MPS, which the first and last presumably the same. For infinite MPS, the last is ignored I believe.
        # We only store L in uMPS.
        for i in range(psi.L):
            C = npc.diag(psi.get_SL(i), obj._AL[i].get_leg('vL'), labels=['vL', 'vR']) # center matrix on the left of site `i`
            obj._C.append(C.astype(obj.dtype, copy=True).itranspose(obj._C_labels))

        obj._transfermatrix_keep = psi._transfermatrix_keep
        obj.test_sanity()
        return obj

    @classmethod
    def from_lat_product_state(cls, lat, p_state, **kwargs):
        raise NotImplementedError("Not valid for UMPS!")

    @classmethod
    def from_product_state(cls,
                           sites,
                           p_state,
                           bc='finite',
                           dtype=np.float64,
                           permute=True,
                           form='B',
                           chargeL=None):
        raise NotImplementedError("Not valid for UMPS.")

    @classmethod
    def from_Bflat(cls,
                   sites,
                   ALflat,
                   ARflat,
                   ACflat,
                   Cflat,
                   dtype=None,
                   permute=True,
                   legL=None):
        """Construct a matrix product state from a set of numpy arrays `Bflat` and singular vals.

        Parameters
        ----------
        sites : list of :class:`~tenpy.networks.site.Site`
            The sites defining the local Hilbert space.
        A{L,R,C}flat : iterable of numpy ndarrays
            The matrix defining the MPS on each site, with legs ``'p', 'vL', 'vR'``
            (physical, virtual left/right).
        Cflat : iterable of numpy ndarrays
            The matrix defining the bond matrix on each site, with legs ``'vL', 'vR'``
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
        mps : :class:`MPS`
            An MPS with the matrices `Bflat` converted to npc arrays.
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
            Clegs = npc.detect_legcharge(C, ci, Clegs, None, qconj=-1) # Even though C has no physical leg, it can have charge.
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
        # so we need to gauge `qtotal` of the last `B` such that the right leg matches.
        # TODO Ask Johannes!
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
        raise NotImplementedError("Not valid for UMPS.")

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
        raise NotImplementedError("Not valid for UMPS.")

    #@property
    #def L(self):

    #@property
    #def dim(self):

    #@property
    #def finite(self):

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
        form : ``'B' | 'A' | 'C' | 'G' | 'Th' | None`` | tuple(float, float)
            The (canonical) form of the returned B.
            For ``None``, return the matrix in whatever form it is.
            If any of the tuple entry is None, also don't scale on the corresponding axis.
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

        Raises
        ------
        ValueError : if self is not in canoncial form and `form` is not None.
        """
        if form is None:
            return self.get_AR(i, copy=False, label_p=label_p)
        elif form=='A' or form==(1., 0.) or form=='AL':
            return self.get_AL(i, copy=False, label_p=label_p)
        elif form=='B' or form==(0., 1.) or form=='AR':
            return self.get_AR(i, copy=False, label_p=label_p)
        elif form=='Th' or form==(1., 1.) or form=='AC':
            return self.get_AC(i, copy=False, label_p=label_p)
        else:
            raise NotImplementedError("Form {0!r} is not valid for VUMPS.".format(form))

    #@property
    #def nontrivial_bonds(self):

    def get_AL(self, i, copy=False, label_p=None):
        i = self._to_valid_index(i)
        AL = self._AL[i]
        if copy:
            AL = AL.copy()
        if label_p is not None:
            AL = self._replace_p_label(AL, label_p)
        return AL

    def get_AR(self, i, copy=False, label_p=None):
        i = self._to_valid_index(i)
        AR = self._AR[i]
        if copy:
            AR = AR.copy()
        if label_p is not None:
            AR = self._replace_p_label(AR, label_p)
        return AR

    def get_AC(self, i, copy=False, label_p=None):
        i = self._to_valid_index(i)
        AC = self._AC[i]
        if copy:
            AC = AC.copy()
        if label_p is not None:
            AC = self._replace_p_label(AC, label_p)
        return AC

    def get_C(self, i, copy=False):
        """Return center matrix on the left of site `i`"""
        i = self._to_valid_index(i)
        C = self._C[i]
        if copy:
            C = C.copy()
        return C

    def set_B(self, i, B, form='B'):
        if form=='A' or form==(1., 0.) or form=='AL':
            return self.set_AL(i, B)
        elif form=='B' or form==(0., 1.) or form=='AR':
            return self.set_AR(i, B)
        elif form=='Th' or form==(1., 1.) or form=='AC':
            return self.set_AC(i, B)
        else:
            raise NotImplementedError("Form {0!r} is not valid for VUMPS.".format(list(form)))

    def set_AL(self, i, AL):
        i = self._to_valid_index(i)
        self.dtype = np.find_common_type([self.dtype, AL.dtype], [])
        self._AL[i] = AL.itranspose(self._B_labels)

    def set_AR(self, i, AR):
        i = self._to_valid_index(i)
        self.dtype = np.find_common_type([self.dtype, AR.dtype], [])
        self._AR[i] = AR.itranspose(self._B_labels)

    def set_AC(self, i, AC):
        i = self._to_valid_index(i)
        self.dtype = np.find_common_type([self.dtype, AC.dtype], [])
        self._AC[i] = AC.itranspose(self._B_labels)

    def set_C(self, i, C):
        i = self._to_valid_index(i)
        self.dtype = np.find_common_type([self.dtype, C.dtype], [])
        self._C[i] = C.itranspose(self._C_labels)

    def set_svd_theta(self, i, theta, trunc_par=None, update_norm=False):
        raise NotImplementedError("Not valid for UMPS.")

    def get_SL(self, i):
        return self.get_C(i)

    def get_SR(self, i):
        return self.get_C(i+1)

    def set_SL(self, i, S):
        self.set_C(i, S)

    def set_SR(self, i, S):
        self.set_C(i+1, S)

    #def get_op(self, op_list, i):

    def get_theta(self, i, n=2, cutoff=1.e-16, formL=1., formR=1.):
        """Calculates the `n`-site wavefunction on ``sites[i:i+n]``.
        Parameters
        ----------
        i : int
            Site index.
        n : int
            Number of sites. The result lives on ``sites[i:i+n]``.
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
        raise NotImplementedError("Not valid for UMPS.")

    #def increase_L(self, new_L=None):

    def enlarge_mps_unit_cell(self, factor=2):
        """Repeat the unit cell for infinite uMPS boundary conditions; in place.

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
        """Shift the section we define as unit cellof an infinite MPS; in place.

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
        raise NotImplementedError("Not valid for UMPS.")

    def group_split(self, trunc_par=None):
        raise NotImplementedError("Not valid for UMPS.")

    def get_grouped_mps(self, blocklen):
        raise NotImplementedError("Not valid for UMPS.")

    def extract_segment(self, first, last):
        raise NotImplementedError("Not valid for UMPS.")

    def get_total_charge(self, only_physical_legs=False):
        """Calculate and return the `qtotal` of the whole MPS (when contracted).

        If set, the :attr:`segment_boundaries` are included (unless `only_physical_legs` is True).

        Parameters
        ----------
        only_physical_legs : bool
            For ``'finite'`` boundary conditions, the total charge can be gauged away
            by changing the LegCharge of the trivial legs on the left and right of the MPS.
            This option allows to project out the trivial legs to get the actual "physical"
            total charge.

        Returns
        -------
        qtotal : charges
            The sum of the `qtotal` of the individual `B` tensors.
        """
        assert only_physical_legs == False
        # Assume self.segment_boundaries is None, None for UMPS
        tensors_AL = self._AL
        qtotal_AL = np.sum([AL.qtotal for AL in tensors_AL], axis=0)
        qtotal_AL = self.chinfo.make_valid(qtotal_AL)

        tensors_AR = self._AR
        qtotal_AR = np.sum([AR.qtotal for AR in tensors_AR], axis=0)
        qtotal_AR = self.chinfo.make_valid(qtotal_AR)
        qtotal_AR.test_equal(qtotal_AL)

        return qtotal_AR

    def gauge_total_charge(self, qtotal=None, vL_leg=None, vR_leg=None):
        raise NotImplementedError("Who knows if this is valid for UMPS?")

    def entanglement_entropy(self, n=1, bonds=None, for_matrix_S=True):
        #assert self.valid_umps
        assert for_matrix_S, "uMPS do not have diagonal C matrices."
        return super().entanglement_entropy(n, bonds, for_matrix_S)

    def entanglement_entropy_segment(self, segment=[0], first_site=None, n=1):
        raise NotImplementedError("Convert uMPS to MPS for calculations involving S.")

    def entanglement_entropy_segment2(self, segment, n=1):
        raise NotImplementedError("Convert uMPS to MPS for calculations involving S.")

    def entanglement_spectrum(self, by_charge=False):
        raise NotImplementedError("Convert uMPS to MPS for calculations involving S.")

    def get_rho_segment(self, segment):
        raise NotImplementedError("Convert uMPS to MPS for calculations involving S.")

    def probability_per_charge(self, bond=0):
        raise NotImplementedError("Convert uMPS to MPS for calculations involving S.")

    def average_charge(self, bond=0):
        raise NotImplementedError("Convert uMPS to MPS for calculations involving S.")

    def charge_variance(self, bond=0):
        raise NotImplementedError("Convert uMPS to MPS for calculations involving S.")

    def mutinf_two_site(self, max_range=None, n=1):
        raise NotImplementedError("Convert uMPS to MPS for calculations involving S.")

    def overlap(self, other, charge_sector=None, ignore_form=False, **kwargs):
        #assert self.valid_umps     # We want to take overlap of non-valid uMPS when doing diagonal gauge conversion.
        assert not ignore_form, "uMPS have both forms. Use one."
        return super().overlap(other, charge_sector=None, ignore_form=False, **kwargs)

    def expectation_value(self, ops, sites=None, axes=None):
        assert self.valid_umps
        return super().expectation_value(ops, sites=sites, axes=axes)

    def expectation_value_term(self, term, autoJW=True):
        assert self.valid_umps
        return super().expectation_value_term(term, autoJW=autoJW)

    #def _term_to_ops_list(self, term, autoJW=True, i_offset=0, JW_from_right=False):

    def expectation_value_multi_sites(self, operators, i0):
        assert self.valid_umps
        return super().expectation_value_multi_sites(operators, i0)

    #def _corr_ops_LP(self, operators, i0):

    #def _corr_ops_RP(self, operators, i0):

    def expectation_value_terms_sum(self, term_list, prefactors=None):
        assert self.valid_umps
        return super().expectation_value_terms_sum(term_list, prefactors)

    def correlation_function(self,
                             ops1,
                             ops2,
                             sites1=None,
                             sites2=None,
                             opstr=None,
                             str_on_first=True,
                             hermitian=False,
                             autoJW=True):
        assert self.valid_umps
        return super().correlation_function(op1,
                                            ops2,
                                            sites1=sites1,
                                            sites2=sites2,
                                            opstr=opstr,
                                            str_on_first=str_on_first,
                                            hermitian=False,
                                            autoJW=autoJW)


    def term_correlation_function_right(self,
                                        term_L,
                                        term_R,
                                        i_L=0,
                                        j_R=None,
                                        autoJW=True,
                                        opstr=None):
        assert self.valid_umps
        return super().term_correlation_function_right(term_L,
                                                       term_R,
                                                       i_L=i_L,
                                                       j_R=j_R,
                                                       autoJW=autoJW,
                                                       opstr=opstr)

    def term_correlation_function_left(self,
                                        term_L,
                                        term_R,
                                        i_L=0,
                                        j_R=None,
                                        autoJW=True,
                                        opstr=None):
        assert self.valid_umps
        return super().term_correlation_function_left(term_L,
                                                      term_R,
                                                      i_L=i_L,
                                                      j_R=j_R,
                                                      autoJW=autoJW,
                                                      opstr=opstr)

    def term_list_correlation_function_right(self,
                                             term_list_L,
                                             term_list_R,
                                             i_L=0,
                                             j_R=None,
                                             autoJW=True,
                                             opstr=None):
        assert self.valid_umps
        return super().term_list_correlation_function_right(term_list_L,
                                                            term_list_R,
                                                            i_L=i_L,
                                                            j_R=j_R,
                                                            autoJW=autoJW,
                                                            opstr=opstr)

    def sample_measurements(self,
                            first_site=0,
                            last_site=None,
                            ops=None,
                            rng=None,
                            norm_tol=1.e-12):
        assert self.valid_umps
        return super().sample_measurements(self,
                                           first_site=0,
                                           last_site=None,
                                           ops=None,
                                           rng=None,
                                           norm_tol=1.e-12)

    def norm_test(self, force=False):
        if not force and not self.valid_umps:
            return np.zeros((self.L, 2), dtype=float)
        else:
            return super().norm_test()

    def canonical_form(self, **kwargs):
        raise NotImplementedError("Not valid for UMPS.")

    def canonical_form_finite(self, renormalize=True, cutoff=0., envs_to_update=None):
        raise NotImplementedError("Not valid for UMPS.")

    def canonical_form_infinite(self, renormalize=True, tol_xi=1.e6):
        raise NotImplementedError("Not valid for UMPS.")

    def correlation_length(self, target=1, tol_ev0=1.e-8, charge_sector=0):
        assert self.valid_umps
        return super().correlation_length(self, target=1, tol_ev0=1.e-8, charge_sector=0)

    def add(self, other, alpha, beta, cutoff=1.e-15):
        raise NotImplementedError("Not valid for UMPS.")

    def apply_local_op(self, i, op, unitary=None, renormalize=False, cutoff=1.e-13):
        raise NotImplementedError("Not valid for UMPS.")

    def apply_product_op(self, ops, unitary=None, renormalize=False):
        raise NotImplementedError("Not valid for UMPS.")

    def perturb(self, randomize_params=None, close_1=True, canonicalize=None):
        raise NotImplementedError("Not valid for UMPS.")

    def swap_sites(self, i, swap_op='auto', trunc_par=None):
        raise NotImplementedError("Not valid for UMPS.")

    def permute_sites(self, perm, swap_op='auto', trunc_par=None, verbose=None):
        raise NotImplementedError("Not valid for UMPS.")

    def compute_K(self,
                  perm,
                  swap_op='auto',
                  trunc_par=None,
                  canonicalize=1.e-6,
                  verbose=None,
                  expected_mean_k=0.):
        raise NotImplementedError("Convert uMPS to MPS for calculations involving S.")

    def __str__(self):
        """Some status information about the uMPS."""
        res = ["uMPS, L={L:d}, bc={bc!r}.".format(L=self.L, bc=self.bc)]
        res.append("chi: " + str(self.chi))
        res.append("valid: " + str(self.valid_umps))
        if self.L > 10:
            res.append("first two sites: " + repr(self.sites[0]) + " " + repr(self.sites[1]))
        else:
            res.append("sites: " + " ".join([repr(s) for s in self.sites]))
        return "\n".join(res)

    def compress(self, options):
        raise NotImplementedError("Not valid for UMPS.")

    def compress_svd(self, trunc_par):
        raise NotImplementedError("Not valid for UMPS.")

    #def _to_valid_index(self, i):

    #def _parse_form(self, form):

    #def _to_valid_form(self, form):

    def _scale_axis_B(self, B, S, form_diff, axis_B, cutoff):
        raise NotImplementedError("Not valid for UMPS.")

    #def _replace_p_label(self, A, s):

    #def _get_p_label(self, s):

    #def _get_p_labels(self, ks, star=False):

    #def _expectation_value_args(self, ops, sites, axes):

    #def _correlation_function_args(self, ops1, ops2, sites1, sites2, opstr):

    #def _corr_up_diag(self, ops1, ops2, i, j_gtr, opstr, str_on_first, apply_opstr_first):

    def _canonical_form_dominant_gram_matrix(self, bond0, transpose, tol_xi, guess=None):
        raise NotImplementedError("Not valid for UMPS.")

    def _canonical_form_correct_right(self, i1, Gr, eps=2. * np.finfo(np.double).eps):
        raise NotImplementedError("Not valid for UMPS.")

    def _canonical_form_correct_left(self, i1, Gl, Wr, eps=2. * np.finfo(np.double).eps):
        raise NotImplementedError("Not valid for UMPS.")

    def _gauge_compatible_vL_vR(self, other):
        raise NotImplementedError("Not valid for UMPS.")

    def outer_virtual_legs(self):
        vL = self._AR[0].get_leg('vL')
        vR = self._AL[-1].get_leg('vR')
        return vL, vR
