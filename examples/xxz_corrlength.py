"""Calculate the correleation legnth of the Transferse field Ising model for various h_z.

.. todo :
    simplify, document
"""



import numpy as np

from tenpy.models.spins import SpinChain
from tenpy.networks.mps import MPS, TransferMatrix
from tenpy.algorithms.dmrg import run as run_DMRG
import pylab as pl


def run(Jzs):
    L = 2
    bc = 'infinite'
    model_params = dict(L=L, Jx=1., Jy=1., Jz=1., bc_MPS=bc, conserve='Sz', verbose=0)
    chi = 300
    dmrg_params = dict(
        trunc_params={'chi_max': chi,
                      'svd_min': 1.e-10,
                      'trunc_cut': None},
        update_env=20,
        start_env=20,
        max_E_err=0.0001,
        max_S_err=0.0001,
        verbose=1,
        mixer=True)  # TODO: mixer?

    M = SpinChain(model_params)
    psi = MPS.from_product_state(M.lat.mps_sites(), [1, 0] * (L // 2), bc)
    #  B = np.zeros([2, 2, 2])
    #  B[0, 0, 0] = B[1, 1, 1] = 1.
    #  psi = MPS.from_Bflat(M.lat.mps_sites(), [B]*L, bc=bc)

    np.set_printoptions(linewidth=120)
    corr_length = []
    for Jz in Jzs:
        print("-" * 80)
        print("Jz =", Jz)
        print("-" * 80)
        model_params['Jz'] = Jz
        M = SpinChain(model_params)
        #  #  psi = MPS.from_product_state(M.lat.mps_sites(), [1, 1]*(L//2), bc)
        #  B = np.zeros([2, 2, 2])
        #  B[0, 0, 0] = B[1, 1, 1] = 1.
        #  psi = MPS.from_Bflat(M.lat.mps_sites(), [B]*L, bc=bc)
        run_DMRG(psi, M, dmrg_params)
        if bc == 'infinite':
            #  T = TransferMatrix(psi, psi, charge_sector=None)
            #  E, V = T.eigenvectors(4, which='LM')
            #  chi = psi.chi[0]
            #  print V[0].to_ndarray().reshape([chi, chi])[:5, :5] * chi**0.5
            #  if len(V) > 1:
            #      print V[1].to_ndarray().reshape([chi, chi])[:5, :5] * chi**0.5
            #  print "chi:", psi.chi
            #  print "eigenvalues transfermatrix:", E
            #  print "norm_test:", psi.norm_test()
            corr_length.append(psi.correlation_length(charge_sector=0, tol_ev0=1.e-3))
            print("corr. length", corr_length[-1])
            print("corr. fct.", psi.correlation_function('Sz', 'Sz', sites1=[0], sites2=6))
            print("<Sz>", psi.expectation_value('Sz'))
        else:
            print(psi.correlation_function('Sz', 'Sz'))
    corr_length = np.array(corr_length)
    results = {
        'model_params': model_params,
        'dmrg_params': dmrg_params,
        'Jzs': Jzs,
        'corr_length': corr_length,
        'eval_transfermatrix': np.exp(-1. / corr_length)
    }
    return results


def plot(results, filename):
    corr_length = results['corr_length']
    Jzs = results['Jzs']
    pl.plot(Jzs, np.exp(-1. / corr_length))
    pl.xlabel(r'$J_z/J_x$')
    pl.ylabel(r'$t = \exp(-\frac{1}{\xi})$')
    pl.savefig(filename)
    #  pl.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = 'xxz_corrlength.pkl'
    import pickle
    import os.path
    if not os.path.exists(filename):
        results = run(list(np.arange(4.0, 1.5, -0.25)) + list(np.arange(1.5, 0.8, -0.05)))
        with open(filename, 'w') as f:
            pickle.dump(results, f)
    else:
        with open(filename) as f:
            results = pickle.load(f)
        plot(results, filename[:-4] + '.pdf')
