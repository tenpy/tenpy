"""An example simulating the dynamics of the Neel state under Heisenberg evolution, using TEBD."""
# Copyright (C) TeNPy Developers, GNU GPLv3
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import tenpy
import pickle
from tqdm import tqdm


model_params = dict(
    L=50,
    Jx=1, Jy=1, Jz=1,
)
model = tenpy.SpinChain(model_params)

engine_params = dict(
    order=2,
    dt=0.01,
    N_steps=5,
    trunc_params=dict(
        chi_max=None,  # to be set in loop
        svd_min=1e-10,
    )
)
dt_measure = engine_params['dt'] * engine_params['N_steps']


def main(load_existing=False, file='results.pkl'):
    if load_existing:
        with open(file, 'rb') as f:
            results = pickle.load(f)
    else:
        results = run()
        with open(file, 'wb') as f:
            pickle.dump(results, f)
    plot(results)


def run():
    tenpy.show_config()
    results = {}
    for chi in [25, 50, 100, 200]:
        print(f'chi = {chi}...')
        results[chi] = run_at_fixed_chi(chi)
    print('done')
    return results


def run_at_fixed_chi(chi):
    psi = tenpy.MPS.from_lat_product_state(model.lat, [['up'], ['down']])
    # Selects Sz=0 sector
    
    engine_params['trunc_params'].update(chi_max=chi)
    engine = tenpy.TEBDEngine(psi, model, engine_params)
    # engine = tenpy.TDVPEngine(psi, model, engine_params)
    
    t = [0]
    S = [psi.entanglement_entropy()]
    mag_z = [psi.expectation_value('Sz')]
    err = [0]
    
    for n in tqdm(range(200)):
        engine.run()
        t.append(engine.evolved_time)
        S.append(psi.entanglement_entropy())
        mag_z.append(psi.expectation_value('Sz'))
        err.append(engine.trunc_err.eps)
    
    t = np.array(t)
    S = np.array(S)
    mag_z = np.array(mag_z)
    err = np.array(err)

    return dict(t=t,
                S=S,
                mag_z=mag_z,
                imbalance=.5 * np.average(mag_z[:, ::2] - mag_z[:, 1::2], axis=1),
                err=err)


def plot(results):
    fontsize = 10
    linewidth = 5.90666  # inches
    L = model_params['L']
    mpl.rcParams.update({'font.size': fontsize})
    chis = sorted(results.keys())
    max_chi = max(chis)

    assert len(chis) == 4
    plot_styles = {
        (chi, which): dict(
            color=c,
            label=rf'$\chi={chi}$',
            ls=ls,
            lw=2 if chi == max_chi else 1.5
        )
        for chi, c, ls in zip(chis,
                            ['tab:blue', 'tab:orange', 'tab:green', 'tab:red'],
                            [':', '--', '-.', '-'])
        for which in [None, 'S', 'Imb', 'err']
    }


    aspect = .7
    fig, ((ax_mag, ax_S), (ax_Imb, ax_err)) = plt.subplots(2, 2, figsize=(linewidth, aspect * linewidth))

    # magnetization profile
    ax_mag.set_xlabel(r'$\langle S^z_i \rangle(t)$')
    ax_mag.set_xlabel(r'$t$')
    ax_mag.set_ylabel(r'Site $i$')
    ax_mag.set_yticks([0, 12, 24, 36, 49])
    ax_mag.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: str(dt_measure * int(x))))
    im = ax_mag.imshow(results[max_chi]['mag_z'].T, aspect='auto', cmap='inferno')
    # cmap candidates: viridis, inferno, coolwarm, bwr, RdBu, 
    divider = make_axes_locatable(ax_mag)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_title(r'$\langle S_i^z\rangle(t)$', fontsize=fontsize)

    # subplot: S_vN entropy
    ax_S.set_ylabel(r'$S_{\text{vN}}$')
    ax_S.set_xlabel('$t$')
    for chi in chis:
        ax_S.plot(results[chi]['t'], results[chi]['S'][:, L // 2], **plot_styles[chi, 'S'])

    # subplot: imbalance
    ax_Imb.set_ylabel(r'$\mathcal{I}$')
    ax_Imb.set_xlabel('$t$')
    for chi in chis:
        ax_Imb.plot(results[chi]['t'], results[chi]['imbalance'], **plot_styles[chi, 'Imb'])
    ax_Imb.legend()

    # subplot: truncation error
    ax_err.set_ylabel(r'$\varepsilon_\text{trunc}$')
    ax_err.set_xlabel('$t$')
    for chi in chis:
        ax_err.semilogy(results[chi]['t'], results[chi]['err'], **plot_styles[chi, 'err'])

    fig.tight_layout(pad=0.1)
    fig.savefig('heisenberg_tebd.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
