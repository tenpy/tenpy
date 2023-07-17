"""Algorithms for contraction of finite PEPS diagrams, e.g. expectation values."""
# Copyright 2023 TeNPy Developers, GNU GPLv3

from ..linalg import np_conserved as npc
from ..linalg.charges import LegCharge
from ..networks.peps import PEPS, PEPO
from ..networks.mps import MPS
from ..networks.mpo import MPO
from ..networks.site import Site
from ..tools.params import asConfig, Config
import logging
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ['BulkMPO', 'TwoLayerColumn', 'ThreeLayerColumn']

class BulkMPO:
    """Like a finite MPO, but there are multiple virtual legs per bond and multiple physical leg(pairs)
    per site, and the MPO-tensors have a substructure as the contraction of multiple tensors"""

    # _virt_labels = [...]
    # _p_labels = [...]

    def __init__(self, L: int):
        self.L = L
        self.test_sanity()

    def test_sanity(self):
        # make sure class attributes are set
        assert len(self._virt_labels) > 0
        assert len(self._p_labels) > 0

    def _to_valid_index(self, n: int) -> int:
        if n < 0:
            n = n + self.L
        assert 0 <= n < self.L
        return n

    def left_contraction(self, n: int, T: npc.Array, B: npc.Array) -> npc.Array:
        r"""The following contraction

        |        
        |           |----B---- vR
        |    {*} ---T    |
        |           |----Wn--- {wR}
        |                |
        |                {p}
        
        where 
            T :: [{*}, vR, {wR}]
            B :: [vL, {p}, vR]
            {*} is any number of other legs
            {p} is self._p_labels
            {wL} is [f'{w}L' for w in self._virt_labels]
            Wn represents the MPO at site n
        """
        raise NotImplementedError('Subclasses should implement this.')

    def right_contraction(self, n: int, T: npc.Array, B: npc.Array) -> npc.Array:
        r"""The following contraction

        |        
        |    vL   ---B----|
        |            |    T--- {*}
        |    {wL} ---Wn---| 
        |            |
        |            {p}

        where 
            T :: [vL, {wL}, {*}]
            B :: [vL, {p}, vR]
            {*} is any number of other legs
            {p} is self._p_labels
            {wL} is [f'{w}L' for w in self._virt_labels]
            Wn represents the MPO at site n
        """
        raise NotImplementedError('Subclasses should implement this.')

    def sandwich_contraction(self, n: int, LP: npc.Array, B1: npc.array, B2: npc.array, RP: npc.Array):
        r"""The following contraction

        |    .----- B1 -----.
        |    |      |       |
        |    LP --- Wn --- RP
        |    |      |       |
        |    .----- B2 -----.

        and optionally also some of its derivatives (i.e. leaving out tensors).
        Note that this method does not conjugate B2!!

        where
            LP :: [v1R, {wR}, v2R]
            B1 :: [vL, {p}, vR]
            B2 :: [vL, {p}, vR]
            RP :: [v1L, {wL}, v2L]
        
        The returned results are
            v: the value of the contraction depicted above
            next_LP: ∂v/∂(RP) :: [v1R, {wR}, v2R]
            ...

        Subclasses may add more return values, so if you dont know which BulkMPO subclass you are
        dealing with, consider writing::

            v, next_LP, *_ = mpo.sandwich_contraction(...)
        """
        raise NotImplementedError('Subclasses should implement this.')

    def as_mpo(self):
        r"""Forget about possible additional structure and convert to MPO. 
        Combine multiple legs into pipes."""
        raise NotImplementedError('Subclasses should implement this.')


class TwoLayerColumn(BulkMPO):
    r"""Column (or row) of a two-layer PEPS diagram interpreted as an MPO.
    
    |                                              pk*
    |                                             /
    |           p*                     wkL --- Ket[n] --- wkR
    |           |                             / |
    |    wL --- W[n] --- wR    =            pk  |  pb*
    |           |                               | /  
    |           p                      wbL --- Bra[n] --- wbR
    |                                         /
    |                                       pb

    Note that this class does not conjugate the bra tensors!
    """
    _virt_labels = ['wb', 'wk']
    _p_labels = ['pb', 'pk']

    def __init__(self, bra_tensors, ket_tensors):
        """Note that this class does not conjugate the bra tensors!"""
        self.bra_tensors = bra_tensors  # [q*, pb, pb*, wbL, wbR]
        self.ket_tensors = ket_tensors  # [q, pk, pk*, wkL, wkR]
        BulkMPO.__init__(self, L=len(bra_tensors))

    def test_sanity(self):
        assert len(self.bra_tensors) == len(self.ket_tensors) == self.L
        BulkMPO.test_sanity(self)

    def left_contraction(self, n: int, T: npc.Array, B: npc.Array) -> npc.Array:
        # T :: [*, vR, wbR, wkR, *]  ,  B :: [vL, pb, pk, vR]
        n = self._to_valid_index(n)
        res = npc.tensordot(T, B, ['vR', 'vL'])  # [*, wkR, wbR, pb, pk, vR]
        # [*, wkR, pk, vR, q*, pb, wbR]
        res = npc.tensordot(res, self.bra_tensors[n], [['wbR', 'pb'], ['wbL', 'pb*']])
          # [*, vR, pb, wbR, pk, wkR]
        res = npc.tensordot(res, self.ket_tensors[n], [['wkR', 'pk', 'q*'], ['wkL', 'pk*', 'q']])
        return res

    def right_contraction(self, n: int, T: npc.Array, B: npc.Array) -> npc.Array:
        #  T :: [vL, wbL, wkL]  ,  B :: [vL, pb, pk, vR]
        n = self._to_valid_index(n)
        res = npc.tensordot(B, T, ['vR', 'vL'])  # [vL, pb, pk, wbL, wkL, *]
        # [q*, pb, wbL, vL, pk, wkL, *]
        res = npc.tensordot(self.bra_tensors[n], res, [['wbR', 'pb*'], ['wbL', 'pb']])
        # [pk, wkL, pb, wbL, vL, *]
        res = npc.tensordot(self.ket_tensors[n], res, [['wkR', 'pk*', 'q'], ['wkL', 'pk', 'q*']])
        return res

    def sandwich_contraction(self, n: int, LP: npc.Array, B1: npc.array, B2: npc.array, RP: npc.Array):
        # LP :: [v1R, wbR, wkR, v2R]
        # B1, B2 :: [vL, pb, pk, vR]
        # RP :: [v1L, wbL, wkL, v2L]
        n = self._to_valid_index(n)

        # build contraction of LP, B1, ket[n] and B2
        tmp = npc.tensordot(LP, B1, ['v1R', 'vL']).ireplace_label('vR', 'v1R')  # [wbR, wkR, v2R, pb, pk, v1R]
        tmp = npc.tensordot(tmp, self.ket_tensors[n], [['wkR', 'pk'], ['wkL', 'pk*']])  # [wbR, v2R, pb, v1R, q, pk, wkR]
        tmp = npc.tensordot(tmp, B2.replace_label('pb', 'pb*'), [['v2R', 'pk'], ['vL', 'pk']])
        tmp = tmp.ireplace_label('vR', 'v2R')  # [wbR, pb, v1R, q, wkR, pb*, v2R]

        # [v1R, wkR, v2R, wbR]
        next_LP = npc.tensordot(tmp, self.bra_tensors[n], [['q', 'pb*', 'pb', 'wbR'], ['q*', 'pb', 'pb*', 'wbL']])

        #  [wbR, pb, q, pb*, wbL]
        bra_grad = npc.tensordot(tmp, RP, [['v1R', 'wkR', 'v2R'], ['v1L', 'wkL', 'v2L']])
        # [wbL*, pb, q, pb*, wbR*]
        bra_grad = bra_grad.ireplace_labels(['wbR', 'wbL'], ['wbL*', 'wbR*'])
        value = npc.tensordot(bra_grad, self.bra_tensors[n], [['q', 'pb', 'pb*', 'wbL*', 'wbR*'], 
                                                              ['q*', 'pb*', 'pb', 'wbL', 'wbR']])
        # conj all legs, s.t. bra_grad can be added to bra. this need to happen after value is contracted
        bra_grad = bra_grad.iconj(complex_conj=False)  
        
        return value, next_LP, bra_grad

    def as_mpo(self):
        Ws = []
        sites = []
        for b, k in zip(self.bra_tensors, self.ket_tensors):
            W = npc.tensordot(b, k, ['q*', 'q'])
            W = W.combine_legs([['pb', 'pk'], ['pb*', 'pk*'], ['wbL', 'wkL'], ['wbR', 'wkR']],
                               qconj=[+1, -1, +1, -1])
            W = W.ireplace_labels(['(pb.pk)', '(pb*.pk*)', '(wbL.wkL)', '(wbR.wkR)'], ['p', 'p*', 'wL', 'wR'])
            Ws.append(W)
            sites.append(Site(leg=W.get_leg('p'), sort_charge=True))
        return MPO(sites=sites, Ws=Ws, bc='finite')


class ThreeLayerColumn(BulkMPO):
    r"""Column (or row) of a three-layer PEPS diagram interpreted as an MPO.
    
    |                                              pk*
    |                                             /
    |                                  wkL --- Ket[n] --- wkR
    |                                         / |
    |           p*                          pk  |  po*
    |           |                               | /  
    |    wL --- W[n] --- wR    =       woL --- Op[n]  --- woR
    |           |                             / |
    |           p                           po  |  pb*
    |                                           | /
    |                                  wbL --- Bra[n] --- wbR
    |                                         /
    |                                       pb

    Note that this class does not conjugate the bra tensors!
    """
    _virt_labels = ['wb', 'wo', 'wk']
    _p_labels = ['pb', 'po', 'pk']

    def __init__(self, bra_tensors, op_tensors, ket_tensors):
        """Note that this class does not conjugate the bra tensors!"""
        self.bra_tensors = bra_tensors  # [q*, pb, pb*, wbL, wbR]
        self.op_tensors = op_tensors  # [q, q*, po, po*, woL, woR]
        self.ket_tensors = ket_tensors  # [q, pk, pk*, wkL, wkR]
        BulkMPO.__init__(self, L=len(bra_tensors))

    def test_sanity(self):
        assert len(self.bra_tensors) == len(self.op_tensors) == len(self.ket_tensors) == self.L
        BulkMPO.test_sanity(self)

    def left_contraction(self, n: int, T: npc.Array, B: npc.Array) -> npc.Array:
        # T :: [*, vR, wbR, woR, wkR]  ,  B :: [vL, pb, po, pk, vR]
        n = self._to_valid_index(n)
        res = npc.tensordot(T, B, ['vR', 'vL'])  # [*, wbR, woR, wkR, pb, po, pk, vR]
        # [*, woR, wkR, po, pk, vR, q*, pb, wbR]
        res = npc.tensordot(res, self.bra_tensors[n], [['wbR', 'pb'], ['wbL', 'pb*']])
        # [*, wkR, pk, vR, pb, wbR, q*, po, woR]
        res = npc.tensordot(res, self.op_tensors[n], [['woR', 'po', 'q*'], ['woL', 'po*', 'q']])
        # [*, vR, pb, wbR, po, woR, pk, wkR]
        res = npc.tensordot(res, self.ket_tensors[n], [['wkR', 'pk', 'q*'], ['wkL', 'pk*', 'q']])
        return res

    def right_contraction(self, n: int, T: npc.Array, B: npc.Array) -> npc.Array:
        #  T :: [vL, wbL, woL, wkL, *]  ,  B :: [vL, pb, po, pk, vR]
        n = self._to_valid_index(n)
        res = npc.tensordot(B, T, ['vR', 'vL'])  # [vL, pb, po, pk, wbL, woL, wkL, *]
        # [q*, pb, wbL, vL, po, pk, woL, wkL, *]
        res = npc.tensordot(self.bra_tensors[n], res, [['wbR', 'pb*'], ['wbL', 'pb']])
        # [q*, po, woL, pb, wbL, vL, pk, wkL, *]
        res = npc.tensordot(self.op_tensors[n], res, [['woR', 'po*', 'q'], ['woL', 'po', 'q*']])
        # [pk, wkL, po, woL, pb, wbL, vL, *]
        res = npc.tensordot(self.ket_tensors[n], res, [['wkR', 'pk*', 'q'], ['wkL', 'pk', 'q*']])
        return res

    def sandwich_contraction(self, n: int, LP: npc.Array, B1: npc.array, B2: npc.array, RP: npc.Array):
        #  LP :: [v1R, wbR, woR, wkR, v2R]
        #  B1, B2 :: [vL, pb, po, pk, vR]
        #  RP :: [v1L, wbL, woL, wkL, v2L]
        n = self._to_valid_index(n)

        # build contraction of LP, B1, ket[n], op[n] and B2
        # [wbR, woR, wkR, v2R, pb, po, pk, v1R]
        tmp = npc.tensordot(LP, B1, ['v1R', 'vL']).ireplace_label('vR', 'v1R')
        # [wbR, woR, v2R, pb, po, v1R, q, pk, wkR]
        tmp = npc.tensordot(tmp, self.ket_tensors[n], [['wkR', 'pk'], ['wkL', 'pk*']])
        # [wbR, v2R, pb, v1R, pk, wkR, woR, po, q]
        tmp = npc.tensordot(tmp, self.op_tensors[n], [['woR', 'po', 'q'], ['woL', 'po*', 'q*']])
        tmp = npc.tensordot(tmp, B2.replace_label('pb', 'pb*'), [['v2R', 'pk', 'po'], ['vL', 'pk', 'po']])
        tmp = tmp.ireplace_label('vR', 'v2R')  # [wbR, pb, v1R, wkR, woR, q, pb*, v2R]

        # [v1R, wkR, woR, v2R, wbR]
        next_LP = npc.tensordot(tmp, self.bra_tensors[n], [['q', 'pb*', 'pb', 'wbR'], ['q*', 'pb', 'pb*', 'wbL']])

        # [wbR, pb, q, pb*, wbL]
        bra_grad = npc.tensordot(tmp, RP, [['v1R', 'wkR', 'woR', 'v2R'], ['v1L', 'wkL', 'woL', 'v2L']])
        # [wbL*, pb, q, pb*, wbR*]
        bra_grad = bra_grad.ireplace_labels(['wbR', 'wbL'], ['wbL*', 'wbR*'])
        value = npc.tensordot(bra_grad, self.bra_tensors[n], [['q', 'pb', 'pb*', 'wbL*', 'wbR*'], 
                                                              ['q*', 'pb*', 'pb', 'wbL', 'wbR']])

        # conj all legs, s.t. bra_grad can be added to bra. this need to happen after value is contracted
        bra_grad = bra_grad.iconj(complex_conj=False)  
        
        return value, next_LP, bra_grad


LEFT = 0
BOTTOM = 1
RIGHT = 2
TOP = 3


class BoundaryMPS(MPS):
    _valid_bc = ['finite']
    _valid_orientations = [LEFT, BOTTOM, RIGHT, TOP]
    _p_label = ['pb', 'pk']
    _B_labels = ['vL', 'pb', 'pk', 'vR']

    def __init__(self, orientation: int, Bs, SVs=None, form=None, norm: float = 1):
        self.orientation = orientation
        if SVs is None:
            SVs = [None] * (len(Bs) + 1)
        MPS.__init__(self, sites=None, Bs=Bs, SVs=SVs, bc='finite', form=form, norm=norm)

    def test_sanity(self):
        assert self.orientation in self._valid_orientations
        MPS.test_sanity(self)

    @classmethod
    def from_trivial(cls, orientation: int, L: int, chinfo=None, dtype=np.complex128):
        legs = cls._get_trivial_B_legs(orientation=orientation, chinfo=chinfo)
        B = npc.ones(legs, dtype=dtype, labels=cls._B_labels)
        Bs = [B.copy() for _ in range(L)]
        SVs = [np.ones([1]) for _ in range(L + 1)]
        return cls(orientation=orientation, Bs=Bs, SVs=SVs, form='B', norm=1)

    @classmethod
    def _get_trivial_B_legs(cls, orientation: int, chinfo=None):
        raise NotImplementedError('Subclasses should implement this')

    @property
    def L(self):
        return len(self._B)

    @property
    def dim(self):
        return [np.prod([B.get_leg(p).dim for p in self._p_label]) for B in self._B]

    # TODO implement h5 I/O
    # TODO raise NotImplemented on not supported methods
    

class BoundaryMPS2(BoundaryMPS):
    _p_label = ['pb', 'pk']
    _B_labels = ['vL', 'pb', 'pk', 'vR']

    @classmethod
    def _get_trivial_B_legs(cls, orientation: int, chargeinfo=None):
        vL_leg = LegCharge.from_trivial(1, chargeinfo=chargeinfo, qconj=+1)
        vR_leg = vL_leg.conj()
        if orientation in [LEFT, BOTTOM]:  # pk should be like a PEPS-leg vR or vU -> qconj=-1
            pk_leg = vR_leg
            pb_leg = vL_leg
        else:  # pk is vL or vD with qconj=+1
            pk_leg = vL_leg
            pb_leg = vR_leg
        return [vL_leg, pb_leg, pk_leg, vR_leg]
    

class BoundaryMPS3(BoundaryMPS):
    _p_label = ['pb', 'po', 'pk']
    _B_labels = ['vL', 'pb', 'po', 'pk', 'vR']

    @classmethod
    def _get_trivial_B_legs(cls, orientation: int, chargeinfo=None):
        vL_leg, pb_leg, pk_leg, vR_leg = BoundaryMPS2._get_trivial_B_legs(
            cls, orientation=orientation, chargeinfo=chargeinfo
        )
        po_leg = pk_leg
        return [vL_leg, pb_leg, po_leg, pk_leg, vR_leg]


class PepsDiagram:
    # this is for finite PEPS! 
    # (for infinite PEPS it makes no sense to consider two-layer diagram seperate from a numerator)

    # subclasses should override this with a concrete subclass of BoundaryMPS (the class itself)
    _BoundaryMPSClass = BoundaryMPS  

    def __init__(self, bra: PEPS, ket: PEPS, options=None):
        self.bra = bra
        self.ket = ket
        self.lx = lx = bra.lx
        self.ly = ly = bra.ly
        self.options = asConfig(options or {}, 'PepsContraction')  # TODO doc parameters
        self.chinfo = chinfo = bra.chinfo
        self._bmps_cache = [None] * 4
        self._bmps_cache[LEFT] = self._bmps_cache[RIGHT] = [None] * lx
        self._bmps_cache[TOP] = self._bmps_cache[BOTTOM] = [None] * ly
        self._bmps_cache[LEFT][0] = self._BoundaryMPSClass.from_trivial(LEFT, L=ly, chinfo=chinfo)
        self._bmps_cache[RIGHT][-1] = self._BoundaryMPSClass.from_trivial(RIGHT, L=ly, chinfo=chinfo)
        self._bmps_cache[BOTTOM][0] = self._BoundaryMPSClass.from_trivial(BOTTOM, L=ly, chinfo=chinfo)
        self._bmps_cache[TOP][-1] = self._BoundaryMPSClass.from_trivial(TOP, L=ly, chinfo=chinfo)
        self.test_sanity()
        
    def test_sanity(self):
        assert self.bra.bc == self.ket.bc == 'finite'
        assert self.bra.lx == self.ket.lx == self.lx
        assert self.bra.ly == self.ket.ly == self.ly
        assert self._BoundaryMPSClass is not BoundaryMPS  # subclasses need to override!
        assert len(self._bmps_cache[TOP]) == len(self._bmps_cache[BOTTOM]) == self.ly
        assert len(self._bmps_cache[LEFT]) == len(self._bmps_cache[RIGHT]) == self.lx
        assert all([
            self._bmps_cache[LEFT][0] is not None,
            self._bmps_cache[RIGHT][-1] is not None,
            self._bmps_cache[BOTTOM][0] is not None,
            self._bmps_cache[TOP][-1] is not None,
        ])

    def _lookup_bmps(self, orientation: int, n: int):
        # subclasses may change this to lookup bmps somewhere else
        # e.g. if part of an operator-diagram is like the norm diagram
        return self._bmps_cache[orientation][n]

    def get_bmps(self, orientation: int, n: int):
        if orientation in [LEFT, BOTTOM]:
            last_n = 0
            step = -1
        else:
            L = self.lx if orientation == RIGHT else self.ly
            last_n = L - 1
            step = 1

        res = self._lookup_bmps(orientation, n)
        i = n
        # go towards the boundary until we find a bMPS that is cached
        while res is None:
            i += step
            if i * step > last_n:  # if i is outside the boundaries
                raise RuntimeError  # should have hit the from_trivial bmps that are set in __init__ by now
            res = self._lookup_bmps(orientation, i)

        # go back to n and cache all results on the way
        while i * step > n * step:
            # TODO initial guess!!
            res, _ = apply_bmps(res, self.get_bmpo(orientation, i), options=self.options)
            self._bmps_cache[orientation][i] = res

        return res

    def get_bmpo(self, orientation, n: int):
        if orientation in [LEFT, RIGHT]:
            return self.get_col_bmpo(orientation, x=n)
        else:
            return self.get_row_bmpo(orientation, y=n)

    def get_col_bmpo(self, orientation: int, x: int):
        raise NotImplementedError('subclasses should implement this')

    def get_row_bmpo(self, orientation: int, y: int):
        raise NotImplementedError('subclasses should implement this')
    

class TwoLayerPepsDiagram(PepsDiagram):

    _BoundaryMPSClass = BoundaryMPS2
            
    def get_col_bmpo(self, orientation: int, x: int):
        if orientation == LEFT:
            #            ['p', 'vU',  'vL',  'vD',  'vR']
            ket_labels = ['q', 'wkR', 'pk*', 'wkL', 'pk']
            bra_labels = ['q*', 'wbR', 'pb*', 'wbL', 'pb']
        elif orientation == RIGHT:
            ket_labels = ['q', 'wkR', 'pk', 'wkL', 'pk*']
            bra_labels = ['q*', 'wbR', 'pb', 'wbL', 'pb*']
        else:
            raise ValueError
        x = self.bra._parse_x(x)
        ket_col = []
        bra_col = []
        for y in range(self.ly):
            ket_col.append(self.ket[x, y].replace_labels(['p', 'vU', 'vL', 'vD', 'vR'], ket_labels))
            bra_col.append(self.bra[x, y].conj().replace_labels(['p*', 'vU*', 'vL*', 'vD*', 'vR*'], bra_labels))
        return TwoLayerColumn(bra_tensors=bra_col, ket_tensors=ket_col)
        
    def get_row_bmpo(self, orientation: int, y: int):
        if orientation == BOTTOM:
            #            ['p', 'vU', 'vL',  'vD',  'vR']
            ket_labels = ['q', 'pk', 'wkL', 'pk*', 'wkR']
            bra_labels = ['q*', 'pb', 'wbL', 'pb*', 'wbR']
        elif orientation == TOP:
            ket_labels = ['q', 'pk*', 'wkL', 'pk', 'wkR']
            bra_labels = ['q*', 'pb*', 'wbL', 'pb', 'wbR']
        else:
            raise ValueError
        y = self.bra._parse_y(y)
        ket_row = []
        bra_row = []
        for x in range(self.lx):
            ket_row.append(self.ket[x, y].replace_labels(['p', 'vU', 'vL', 'vD', 'vR'], ket_labels))
            bra_row.append(self.bra[x, y].conj().replace_labels(['p*', 'vU*', 'vL*', 'vD*', 'vR*'], bra_labels))
        return TwoLayerColumn(bra_tensors=bra_row, ket_tensors=ket_row)


class ThreeLayerPepsDiagram(PepsDiagram):
    # this is for finite PEPS! 
    # (for infinite PEPS it makes no sense to consider two-layer diagram seperate from a numerator)

    _BoundaryMPSClass = BoundaryMPS3

    def __init__(self, bra: PEPS, op: PEPO, ket: PEPS):
        self.op = op
        PepsDiagram.__init__(self, bra=bra, ket=ket)

    def test_sanity(self):
        assert self.op.bc == 'finite'
        assert self.op.lx == self.lx
        assert self.op.ly == self.ly

    def get_col_bmpo(self, orientation: int, x: int):
        if orientation == LEFT:
            #            ['p', 'vU',  'vL',  'vD',  'vR']
            ket_labels = ['q', 'wkR', 'pk*', 'wkL', 'pk']
            op_labels = ['q', 'q*', 'woR', 'po*', 'woL', 'po']
            bra_labels = ['q*', 'wbR', 'pb*', 'wbL', 'pb']
        elif orientation == RIGHT:
            ket_labels = ['q', 'wkR', 'pk', 'wkL', 'pk*']
            op_labels = ['q', 'q*', 'woR', 'po', 'woL', 'po*']
            bra_labels = ['q*', 'wbR', 'pb', 'wbL', 'pb*']
        else:
            raise ValueError
        x = self.bra._parse_x(x)
        ket_col = []
        op_col = []
        bra_col = []
        for y in range(self.ly):
            ket_col.append(self.ket[x, y].replace_labels(['p', 'vU', 'vL', 'vD', 'vR'], ket_labels))
            op_col.append(self.op[x, y].replace_labels(['p', 'p*', 'wU', 'wL', 'wD', 'wR'], op_labels))
            bra_col.append(self.bra[x, y].conj().replace_labels(['p*', 'vU*', 'vL*', 'vD*', 'vR*'], bra_labels))
        return ThreeLayerColumn(bra_tensors=bra_col, op_tensors=op_col, ket_tensors=ket_col)

    def get_row_bmpo(self, orientation: int, y: int):
        if orientation == BOTTOM:
            #            ['p', 'vU', 'vL',  'vD',  'vR']
            ket_labels = ['q', 'pk', 'wkL', 'pk*', 'wkR']
            op_labels = ['q', 'q*', 'po', 'woL', 'po*', 'woR']
            bra_labels = ['q*', 'pb', 'wbL', 'pb*', 'wbR']
        elif orientation == TOP:
            ket_labels = ['q', 'pk*', 'wkL', 'pk', 'wkR']
            op_labels = ['q', 'q*', 'po*', 'woL', 'po', 'woR']
            bra_labels = ['q*', 'pb*', 'wbL', 'pb', 'wbR']
        else:
            raise ValueError
        y = self.bra._parse_y(y)
        ket_row = []
        op_row = []
        bra_row = []
        for x in range(self.lx):
            ket_row.append(self.ket[x, y].replace_labels(['p', 'vU', 'vL', 'vD', 'vR'], ket_labels))
            op_row.append(self.op[x, y].replace_labels(['p', 'p*', 'wU', 'wL', 'wD', 'wR'], op_labels))
            bra_row.append(self.bra[x, y].conj().replace_labels(['p*', 'vU*', 'vL*', 'vD*', 'vR*'], bra_labels))
        return ThreeLayerColumn(bra_tensors=bra_row, op_tensors=op_row, ket_tensors=ket_row)


class MpoPepsDiagram(PepsDiagram):
    def __init__(self):
        raise NotImplementedError  # FIXME


def apply_bmps(bmps: BoundaryMPS, bmpo: BulkMPO, options: Config, initial_guess: BoundaryMPS = None):
    raise NotImplementedError  # TODO
