"""test of graph structure in :class:`tenpy.networks.mpo.MPO`
   and method `init_LP_RP_iterative` in :class:`tenpy.netoworks.mpo.MPOEnvironment`."""
# Copyright (C) TeNPy Developers, Apache license

from tenpy.linalg import np_conserved as npc
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine as dmrg_eng

from tenpy.models.tf_ising import TFIChain
from tenpy.models.lattice import Square
from tenpy.models.model import CouplingModel, MPOModel
# networks
from tenpy.networks.site import SpinHalfSite as shs
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO, MPOEnvironment, MPOTransferMatrix

# model setup
Lx, Ly = 4, 3
m_tfi = TFIChain({"conserve":"parity","g":1.5, "J":1., "L":Lx, "bc_MPS":"infinite"})
H_tfi = m_tfi.calc_H_MPO()
# general coupling model on square lattice
site_z = shs(conserve="Sz")
lat = Square(Lx, Ly, site_z, bc="periodic", bc_MPS="infinite")
m_couple = CouplingModel(lat)
# exp_decaying
m_couple.add_exponentially_decaying_coupling(1.2, 0.42, "Sigmaz", "Sigmaz", subsites = [1,4,7,10])
m_couple.add_exponentially_decaying_coupling(0.32, 0.42, "Sigmaz", "Sigmaz", subsites = [0,3,6,9])
# onsite
for site in range(11):
    m_couple.add_onsite_term(-0.2, site, "Sigmaz")
# y-direction
for s0 in range(0,12,3):
    for offset in range(3):
        i = min(s0+offset, s0+(offset+1)%3)
        j = max(s0+offset, s0+(offset+1)%3)
        m_couple.add_coupling_term(1.77, i, j, "Sp", "Sm")
        m_couple.add_coupling_term(1.77, i, j, "Sm", "Sp")
for s0 in range(0,12,4):
    for offset in range(3):
        m_couple.add_multi_coupling_term(1.16, [s0+offset,s0+offset+3,s0+offset+6,s0+offset+9], ["Sm","Sigmaz","Sigmaz","Sp"], op_string=["Id"]*3)
        m_couple.add_multi_coupling_term(1.16, [s0+offset,s0+offset+3,s0+offset+6,s0+offset+9], ["Sp","Sigmaz","Sigmaz","Sm"], op_string=["Id"]*3)
H_couple  = m_couple.calc_H_MPO()

# MPO of H^2 for TFI model
# pauli_op
s = m_tfi.lat.mps_sites()[0]
Sig_x = s.get_op("Sigmax")
Sig_y = s.get_op("Sigmay")
Sig_z = s.get_op("Sigmaz")
Sig_id = s.get_op("Id")
# TFI
grid_tfi = [[[Sig_id, Sig_x, -1.5*Sig_z, None, 1.625*Sig_id],[None, None, -Sig_x, -Sig_id, None],[None, None, Sig_id, Sig_x, -1.5*Sig_z],
             [None, None, None, None, -Sig_x], [None, None, None, None, Sig_id]] for _ in range(Lx)]
tfi_square = MPO.from_grids(m_tfi.lat.mps_sites(), grid_tfi, bc="infinite", IdL=0, IdR=4)

# compute states
psi_tfi = MPS.from_product_state(m_tfi.lat.mps_sites(), ["up"]*Lx, bc="infinite")
eng = dmrg_eng(psi_tfi, m_tfi, {"min_sweeps":19, "max_sweeps":29, "trunc_params":{"chi_max":32}})
E_tfi, psi_tfi = eng.run()
psi_couple = MPS.from_product_state(m_couple.lat.mps_sites(), ["up"]*(Lx*Ly), bc="infinite")
m_couple_MPO = MPOModel(m_couple.lat, H_couple)
eng = dmrg_eng(psi_couple, m_couple_MPO, {"min_sweeps":19, "max_sweeps":29, "trunc_params":{"chi_max":8}})
E_couple, psi_couple = eng.run()

# MPO: graph and ordering checks
# environment initialization checks
Hs = [H_tfi, H_couple, tfi_square]
states = [psi_tfi, psi_couple]
energies = [E_tfi, E_couple, E_tfi]
H_names = ["TFIChain", "CouplingModel", "TFIsquare"]
N_cycles = [2,4,3]
# CouplingModel cycle indices not intuitively clear
cycle_indices = [[0,2],None,[0,2,4]]

for j_H, H in enumerate(Hs):
    # ----- PART 1: check graph setup ------
    for sort_charges in [0,1,2]:
        if sort_charges==1:
            tperms = H.sort_legcharges()
        if sort_charges==2:
            H._reset_graph()
        if sort_charges!=1:
            H._make_graph() # implicit itranspose of W matrices for tests later
            H._order_graph()
        # check that graph is a 1 to 1 mapping of H
        chis = H.chi
        for j, W in enumerate(H._W):
            for jL in range(chis[j]):
                for jR in range(chis[j+1]):
                    op = W[jL,jR]
                    if npc.norm(op)<1e-12:
                        assert (jL,jR) not in H._graph[j], H_names[j_H]+": entry of norm zero found in graph"
                    else:
                        assert npc.norm(op-H._graph[j][(jL,jR)])<1e-12, H_names[j_H]+": _graph[{0}][({1},{2})] wrong".format(j, jL, jR)
        # environment checks
        if sort_charges!=2 and j_H<2:
            psi = states[j_H]
            E = energies[j_H]
            env_base, E_base, _ = MPOTransferMatrix.find_init_LP_RP(H, psi, 0, psi.L-1, calc_E=True)
            env = MPOEnvironment(psi, H, psi)
            env_iter, E_iter = env.init_LP_RP_iterative(which="both")
            env_base["init_RP"].itranspose(["wL","vL","vL*"])
            env_base["init_LP"].itranspose(["wR","vR","vR*"])
            assert npc.norm(env_base["init_LP"]-env_iter["init_LP"][0])<1e-8, H_names[j_H]+": LP_iterative not converged"
            assert npc.norm(env_base["init_RP"]-env_iter["init_RP"][0])<1e-8, H_names[j_H]+": RP_iterative not converged"
            assert abs(E_base[0]-E)<1e-10 and abs(E_base[1]-E)<1e-10, H_names[j_H]+": Energy of transfer matrix method doesn't match DMRG, test not possible" 
            assert abs(E_iter["init_LP"][1]-E_base[0])<1e-11 and abs(E_iter["init_RP"][1]-E_base[1])<1e-11, H_names[j_H]+": Energies of iterative environment intialization don't match"    
        # H^2 test
        if sort_charges==0 and j_H==2:
            psi = states[0]
            H_base = Hs[0]
            E = energies[j_H]
            e1 = MPOEnvironment(psi, H_base, psi)
            e2 = MPOEnvironment(psi, H, psi)
            env1, E_iter1 = e1.init_LP_RP_iterative(which="both")
            env2, E_iter2 = e2.init_LP_RP_iterative(which="both")
            assert abs(E_iter1["init_LP"][1]-E_iter2["init_LP"][1])<1e-8, H_names[j_H]+": Energies (left) computed via H and H**2 don't match"
            assert abs(E_iter1["init_RP"][1]-E_iter2["init_RP"][1])<1e-8, H_names[j_H]+": Energies (right) computed via H and H**2 don't match"
            assert abs(2*E_iter2["init_LP"][2]-4*E_iter2["init_LP"][1]**2)<1e-8, H_names[j_H]+": E and E**2 (left) from H**2 don't match"
            assert abs(2*E_iter2["init_RP"][2]-4*E_iter2["init_RP"][1]**2)<1e-8, H_names[j_H]+": E and E**2 (right) from H**2 don't match"
            # check left eigenvector explicitly
            for j in range(1,e2.L):
                e2.del_LP(j)
            e2.set_LP(0, env2['init_LP'][0],0)
            LP_prelast = e2.get_LP(e2.L-1,False)
            LP = e2._contract_LP(e2.L-1,LP_prelast)
            for _e in [LP]+env2['init_LP']:
                _e.itranspose(['wR*','vR','vR*'])
            LP_diff = LP-E_iter2['init_LP'][1]*e2.L*env2['init_LP'][1]-E_iter2['init_LP'][2]*e2.L*env2['init_LP'][2]-env2['init_LP'][0]
            assert (npc.norm(LP_diff))<1e-8, H_names[j_H]+"Left environment of H**2 is not an eigenvector"

        # enlarge unit cell
        if sort_charges==2 and j_H<2: 
            H.enlarge_mps_unit_cell(2)
            chis = H.chi
            for j, W in enumerate(H._W):
                for jL in range(chis[j]):
                    for jR in range(chis[j+1]):
                        op = W[jL,jR]
                        if npc.norm(op)<1e-12:
                            assert (jL,jR) not in H._graph[j], H_names[j_H]+": entry of norm zero found in graph after enlarge_unit_cell()"
                        else:
                            assert npc.norm(op-H._graph[j][(jL,jR)])<1e-12, H_names[j_H]+": _graph[{0}][({1},{2})] wrong after enlarge_unit_cell()".format(j, jL, jR)
        
        assert len(H._cycles)==N_cycles[j_H], H_names[j_H]+": wrong number of cycles"
        # outer permutation
        assert H._outer_permutation[0]==H.IdL[0], H_names[j_H]+": wrong IdL index in _outer_permutation"
        assert H._outer_permutation[-1]==H.IdR[-1], H_names[j_H]+": wrong IdR index in _outer_permutation"
        # IdL, IdR cycle
        for val1, val2 in zip(H._cycles[H.IdL[0]],H.IdL):
            assert val1==val2, H_names[j_H]+": H.IdL different from H._cycles[H.IdL[0]]"
        for val1, val2 in zip(H._cycles[H.IdR[-1]],H.IdR):
            assert val1==val2, H_names[j_H]+": H.IdR different from H._cycles[H.IdR[-1]]"
        # cycles
        for i0 in H._cycles:
                c = H._cycles[i0]
                assert len(c)==H.L+1, H_names[j_H]+": wrong cycle length"
                assert c[0]==c[-1], H_names[j_H]+": invalid cycle encountered"                
        if sort_charges==0 and j_H!=1: # explicit cycle check
            for j_cycle in cycle_indices[j_H]:
                assert H._cycles[j_cycle]==[j_cycle]*(H.L+1), H_names[j_H]+": _cycles[{0}] not as expected".format(j_cycle)       