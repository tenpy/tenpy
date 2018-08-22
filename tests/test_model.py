"""A collection of tests for (classes in) :mod:`tenpy.models.model`.
"""
# Copyright 2018 TeNPy Developers

import itertools

from tenpy.models import model, lattice
import tenpy.networks.site
import tenpy.linalg.np_conserved as npc
import test_mpo
import nose

spin_half_site = tenpy.networks.site.SpinHalfSite('Sz')

fermion_site = tenpy.networks.site.FermionSite('N')

__all__ = ["check_model_sanity", "check_general_model"]

def check_model_sanity(M, hermitian=True):
    """call M.test_sanity() for all different subclasses of M"""
    if isinstance(M, model.CouplingModel):
        if isinstance(M, model.MultiCouplingModel):
            model.MultiCouplingModel.test_sanity(M)
        else:
            model.CouplingModel.test_sanity(M)
    if isinstance(M, model.NearestNeighborModel):
        model.NearestNeighborModel.test_sanity(M)
        if hermitian:
            for i, H in enumerate(M.H_bond):
                if H is not None:
                    err = npc.norm(H - H.conj().transpose(H.get_leg_labels()))
                    if err > 1.e-14:
                        print(H)
                        raise ValueError("H on bond {i:d} not hermitian".format(i=i))
    if isinstance(M, model.MPOModel):
        model.MPOModel.test_sanity(M)
        test_mpo.check_hermitian(M.H_MPO)


def check_general_model(ModelClass, model_pars={}, check_pars={}, hermitian=True):
    """Create a model for different sets of parameters and check it's sanity.

    Parameters
    ----------
    ModelClass :
        We generate models of this class
    model_pars : dict
        Model parameters used.
    check_pars : dict
        pairs (`key`, `list of values`); we update ``model_paras[key]`` with any values of
        ``check_params[key]`` (in each possible combination!) and create a model for it.
    hermitian : bool
        If True, check that the Hamiltonian is hermitian.
    """
    for vals in itertools.product(*list(check_pars.values())):
        print("-" * 40)
        params = model_pars.copy()
        for k, v in zip(list(check_pars.keys()), vals):
            params[k] = v
        print("check_model_sanity with following parameters:")
        print(params)
        M = ModelClass(params)
        check_model_sanity(M)


def test_CouplingModel():
    spin_half_lat = lattice.Chain(5, spin_half_site, bc_MPS='finite')
    for bc in ['open', 'periodic']:
        M = model.CouplingModel(spin_half_lat, bc)
        M.add_coupling(1.2, 0, 'Sz', 0, 'Sz', 1)
        M.test_sanity()
        M.calc_H_MPO()
        if bc == 'periodic':
            with nose.tools.assert_raises(ValueError):
                M.calc_H_bond()  # should raise a ValueError
                # periodic bc but finite bc_MPS leads to a long-range coupling
        else:
            M.calc_H_bond()

def test_MultiCouplingModel_shift(Lx=3, Ly=3, shift=1):
    spin_half_square = lattice.Square(Lx, Ly, spin_half_site, bc_MPS='infinite')
    bc = ['periodic', shift]
    M = model.MultiCouplingModel(spin_half_square, bc)
    M.add_coupling(1.2, 0, 'Sz', 0, 'Sz', [1, 0])
    M.add_multi_coupling(0.8, 0, 'Sz', [(0, 'Sz', [0, 1]), (0, 'Sz', [1, 0])])
    M.test_sanity()
    H = M.calc_H_MPO()
    dims = [W.shape[0] for W in H._W]
    # check translation invariance of the MPO: at least the dimensions should fit
    # (the states are differently ordered, so the matrices differ!)
    for i in range(1, Lx):
        assert dims[:Lx] == dims[Lx:2*Lx]


def test_CouplingModel_fermions():
    for bc, bc_MPS in zip(['open', 'periodic'], ['finite', 'infinite']):
        fermion_lat = lattice.Chain(5, fermion_site, bc_MPS=bc_MPS)
        M = model.CouplingModel(fermion_lat, bc)
        M.add_coupling(1.2, 0, 'Cd', 0, 'C', 1, 'JW')
        M.add_coupling(1.2, 0, 'Cd', 0, 'C', -1, 'JW')
        M.test_sanity()
        M.calc_H_MPO()
        M.calc_H_bond()


def test_CouplingModel_explicit():
    fermion_lat_cyl = lattice.Square(1, 2, fermion_site, bc_MPS='infinite')
    M = model.CouplingModel(fermion_lat_cyl, 'periodic')
    M.add_onsite(0.125, 0, 'N')
    M.add_coupling(0.25, 0, 'Cd', 0, 'C', (0, 1), None) # auto-determine JW-string!
    M.add_coupling(0.25, 0, 'Cd', 0, 'C', (0, -1), None)
    M.add_coupling(1.5, 0, 'Cd', 0, 'C', (1, 0), None)
    M.add_coupling(1.5, 0, 'Cd', 0, 'C', (-1, 0), None)
    M.add_coupling(4., 0, 'N', 0, 'N', (2, 1), None)  # a full unit cell inbetween!
    H_mpo = M.calc_H_MPO()
    W0_new = H_mpo.get_W(0)
    W1_new = H_mpo.get_W(1)
    Id, JW, N = fermion_site.Id, fermion_site.JW, fermion_site.N
    Cd, C = fermion_site.Cd, fermion_site.C
    CdJW = Cd.matvec(JW)
    JWC = JW.matvec(C)
    # yapf: disable
    W0_ex = [[Id,   None, None, CdJW, JWC,  N,    None, None, None, N*0.125],
             [None, None, Id,   None, None, None, None, None, None, None],
             [None, None, None, None, None, None, None, None, None, N*4.0],
             [None, None, None, None, None, None, None, None, None, C*1.5],
             [None, None, None, None, None, None, None, None, None, Cd*1.5],
             [None, Id,   None, None, None, None, None, None, None, None],
             [None, None, None, None, None, None, JW,   None, None, None],
             [None, None, None, None, None, None, None, JW,   None, None],
             [None, None, None, None, None, None, None, None, Id,   None],
             [None, None, None, None, None, None, None, None, None, Id]]
    W1_ex = [[Id,  None, None, None, None, None,  CdJW, JWC,  N,    N*0.125],
             [None, Id,   None, None, None, None, None, None, None, None],
             [None, None, None, None, None, None, None, None, None, N*4.0],
             [None, None, None, JW,   None, None, None, None, None, C*0.5],
             [None, None, None, None, JW,   None, None, None, None, Cd*0.5],
             [None, None, None, None, None, Id,   None, None, None, None],
             [None, None, None, None, None, None, None, None, None, C*1.5],
             [None, None, None, None, None, None, None, None, None, Cd*1.5],
             [None, None, Id,   None, None, None, None, None, None, None],
             [None, None, None, None, None, None, None, None, None, Id]]
    # yapf: enable
    W0_ex = npc.grid_outer(W0_ex, W0_new.legs[:2])
    W1_ex = npc.grid_outer(W1_ex, W1_new.legs[:2])
    assert npc.norm(W0_new - W0_ex) == 0.  # coupling constants: no rounding errors
    assert npc.norm(W1_new - W1_ex) == 0.  # coupling constants: no rounding errors


def test_MultiCouplingModel_explicit():
    fermion_lat_cyl = lattice.Square(1, 2, fermion_site, bc_MPS='infinite')
    M = model.MultiCouplingModel(fermion_lat_cyl, 'periodic')
    # create a wired fermionic model with 3-body interactions
    M.add_onsite(0.125, 0, 'N')
    M.add_coupling(0.25, 0, 'Cd', 0, 'C', (0, 1))
    M.add_coupling(0.25, 0, 'Cd', 0, 'C', (0, -1), 'JW')
    M.add_coupling(1.5, 0, 'Cd', 0, 'C', (1, 0), 'JW')
    M.add_coupling(1.5, 0, 'Cd', 0, 'C', (-1, 0), 'JW')
    M.add_multi_coupling(4., 0, 'N', [(0, 'N', (2, 1))], 'Id')  # a full unit cell inbetween!
    # some wired mediated hopping along the diagonal
    M.add_multi_coupling(1.125, 0, 'N', other_ops=[(0, 'Cd', (0, 1)), (0, 'C', (1, 0))])
    H_mpo = M.calc_H_MPO()
    W0_new = H_mpo.get_W(0)
    W1_new = H_mpo.get_W(1)
    W2_new = H_mpo.get_W(2)
    Id, JW, N = fermion_site.Id, fermion_site.JW, fermion_site.N
    Cd, C = fermion_site.Cd, fermion_site.C
    CdJW = Cd.matvec(JW)
    JWC = JW.matvec(C)
    NJW = N.matvec(JW)
    # yapf: disable
    W0_ex = [[Id,   None, None, CdJW, None, JWC,  N,    None, None, None, N*0.125],
             [None, None, Id,   None, None, None, None, None, None, None, None],
             [None, None, None, None, None, None, None, None, None, None, N*4.0],
             [None, None, None, None, None, None, None, None, None, None, C*1.5],
             [None, None, None, None, JW,   None, None, None, None, None, None],
             [None, None, None, None, None, None, None, None, None, None, Cd*1.5],
             [None, Id,   None, None, None, None, None, None, None, None, None],
             [None, None, None, None, None, None, None, None, None, None, C*1.125],
             [None, None, None, None, None, None, None, JW,   None, None, None],
             [None, None, None, None, None, None, None, None, JW,   None, None],
             [None, None, None, None, None, None, None, None, None, Id,   None],
             [None, None, None, None, None, None, None, None, None, None, Id]]
    W1_ex = [[Id,   None, None, None, None, None, None, None, CdJW, JWC,  N,    N*0.125],
             [None, Id,   None, None, None, None, None, None, None, None, None, None],
             [None, None, None, None, None, None, None, None, None, None, None, N*4.0],
             [None, None, None, JW,   NJW,  None, None, None, None, None, None, C*0.5],
             [None, None, None, None, None, None, None, None, None, None, None, C*1.125],
             [None, None, None, None, None, JW,   None, None, None, None, None, Cd*0.5],
             [None, None, None, None, None, None, Id,   CdJW, None, None, None, None],
             [None, None, None, None, None, None, None, None, None, None, None, C*1.5],
             [None, None, None, None, None, None, None, None, None, None, None, Cd*1.5],
             [None, None, Id,   None, None, None, None, None, None, None, None, None],
             [None, None, None, None, None, None, None, None, None, None, None, Id]]
    # yapf: enable
    W0_ex = npc.grid_outer(W0_ex, W0_new.legs[:2])
    W1_ex = npc.grid_outer(W1_ex, W1_new.legs[:2])
    assert npc.norm(W0_new - W0_ex) == 0.  # coupling constants: no rounding errors
    assert npc.norm(W1_new - W1_ex) == 0.  # coupling constants: no rounding errors


if __name__ == "__main__":
    test_MultiCouplingModel_explicit()
