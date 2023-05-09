"""Algorithms for contraction of finite PEPS diagrams, e.g. expectation values."""
# Copyright 2023 TeNPy Developers, GNU GPLv3

from tenpy.linalg import np_conserved as npc
from ..linalg import np_conserved as npc
from ..networks.mps import MPS
from ..networks.mpo import MPO
from ..networks.site import Site
import logging

logger = logging.getLogger(__name__)

__all__ = ['BulkMPO', 'TwoLayerColumn']

class BulkMPO:
    """Like a finite MPO, but there are multiple virtual legs per bond and multiple physical leg(pairs)
    per site, and the MPO-tensors have a substructure as the contraction of tensors"""

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
