from tenpy.models.tf_ising import TFIChain
from tenpy.networks.purification_mps import PurificationMPS
from tenpy.algorithms.purification import PurificationTEBD, PurificationApplyMPO


def imag_tebd(L=30, beta_max=3., dt=0.05, order=2, bc="finite"):
    model_params = dict(L=L, J=1., g=1.2)
    M = TFIChain(model_params)
    psi = PurificationMPS.from_infiniteT(M.lat.mps_sites(), bc=bc)
    options = {
        'trunc_params': {
            'chi_max': 100,
            'svd_min': 1.e-8
        },
        'order': order,
        'dt': dt,
        'N_steps': 1
    }
    beta = 0.
    eng = PurificationTEBD(psi, M, options)
    Szs = [psi.expectation_value("Sz")]
    betas = [0.]
    while beta < beta_max:
        beta += 2. * dt  # factor of 2:  |psi> ~= exp^{- dt H}, but rho = |psi><psi|
        betas.append(beta)
        eng.run_imaginary(dt)  # cool down by dt
        Szs.append(psi.expectation_value("Sz"))  # and further measurements...
    return {'beta': betas, 'Sz': Szs}


def imag_apply_mpo(L=30, beta_max=3., dt=0.05, order=2, bc="finite", approx="II"):
    model_params = dict(L=L, J=1., g=1.2)
    M = TFIChain(model_params)
    psi = PurificationMPS.from_infiniteT(M.lat.mps_sites(), bc=bc)
    options = {'trunc_params': {'chi_max': 100, 'svd_min': 1.e-8}}
    beta = 0.
    if order == 1:
        Us = [M.H_MPO.make_U(-dt, approx)]
    elif order == 2:
        Us = [M.H_MPO.make_U(-d * dt, approx) for d in [0.5 + 0.5j, 0.5 - 0.5j]]
    eng = PurificationApplyMPO(psi, Us[0], options)
    Szs = [psi.expectation_value("Sz")]
    betas = [0.]
    while beta < beta_max:
        beta += 2. * dt  # factor of 2:  |psi> ~= exp^{- dt H}, but rho = |psi><psi|
        betas.append(beta)
        for U in Us:
            eng.init_env(U)  # reset environment, initialize new copy of psi
            eng.run()  # apply U to psi
        Szs.append(psi.expectation_value("Sz"))  # and further measurements...
    return {'beta': betas, 'Sz': Szs}


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    data_tebd = imag_tebd()
    data_mpo = imag_apply_mpo()

    import numpy as np
    from matplotlib.pyplot import plt

    plt.plot(data_mpo['beta'], np.sum(data_mpo['Sz'], axis=1), label='MPO')
    plt.plot(data_tebd['beta'], np.sum(data_tebd['Sz'], axis=1), label='TEBD')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'total $S^z$')
    plt.show()
