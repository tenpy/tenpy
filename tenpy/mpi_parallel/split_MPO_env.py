# Copyright (C) TeNPy Developers, GNU GPLv3

import tenpy
from tenpy.mpi_parallel import helpers
from tenpy.tools import hdf5_io
from tenpy.linalg import np_conserved as npc

def split_MPO_env(filename, nodes):
    data = hdf5_io.load(filename)
    LP = data['resume_data']['init_env_data']['init_LP']
    RP = data['resume_data']['init_env_data']['init_RP']

    wR = LP.get_leg('wR')
    wL = RP.get_leg('wL')

    mpi_split_params = {method: 'block_size'}

    split_wRs = helpers.split_MPO_leg(wR, nodes, mpi_split_params)
    split_LPs = []
    for swR in  split_wRs:
        sLP = LP.copy(deep=True)
        sLP.iproject(mask=swR, axis='wR')
        split_LPs.append(sLP)

    assert LP.get_leg('wR').ind_len == [sLP.get_leg('wR').ind_len for sLP in split_LPs]


    split_wLs = helpers.split_MPO_leg(wL, nodes, mpi_split_params)
    split_RPs = []
    for swR in  split_wLs:
        sRP = RP.copy(deep=True)
        sRP.iproject(mask=swL, axis='wL')
        split_RPs.append(sRP)

    assert RP.get_leg('wL').ind_len == [sRP.get_leg('wL').ind_len for sRP in split_RPs]

    return split_LPs, split_RPs

if __name__ == "__main__":
    filename = 'state_nu_0.0_mpo_svd_1e-06_mps_chi_128.h5'
    output_filename = 'split_state_nu_0.0_mpo_svd_1e-06_mps_chi_128_mpirank'

    split_LPs, split_RPs = split_MPO_env(filename, 2)

    for i, (LP, RP) in enumerate(zip(split_LPs, split_RPs)):
        data = {}
        data['resume_data'] = {}
        data['resume_data']['init_env_data'] = {}
        data['resume_data']['init_env_data']['init_LP'] = LP
        data['resume_data']['init_env_data']['init_RP'] = RP
        hdf5_io.save(data, output_filename + str(i) + '.h5')
