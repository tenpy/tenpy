"""An example simulating the dynamics of the Neel state under Heisenberg evolution, using TEBD."""
# Copyright (C) TeNPy Developers, GNU GPLv3
import argparse
import os
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
    dt=0.02,
    N_steps=5,
    trunc_params=dict(
        chi_max=None,  # to be set in loop
        svd_min=1e-10,
    )
)
dt_measure = engine_params['dt'] * engine_params['N_steps']


def main():
    parser = argparse.ArgumentParser(
        description='Either run simulation at fixed chi, or plot results.'
    )
    parser.add_argument('--chi', type=int, default=50, help='MPS bond dimension. Default 50.')
    parser.add_argument('--outfolder', default='./',
                        help='Folder to write results to. Default is CWD.')
    parser.add_argument('--plot', type=str, metavar='FOLDER', default=None,
                        help='Plot results in the given folder instead of running the simulation.')
    args = parser.parse_args()

    if args.plot is not None:
        plot(folder=args.plot)
        return

    results = run(chi=args.chi)
    with open(os.path.join(args.outfolder, f'heisenberg_tebd_chi_{args.chi}.pkl'), 'wb') as f:
        pickle.dump(results, f)


def run(chi: int):
    psi = tenpy.MPS.from_lat_product_state(model.lat, [['up'], ['down']])
    # Selects Sz=0 sector
    
    engine_params['trunc_params'].update(chi_max=chi)
    engine = tenpy.TEBDEngine(psi, model, engine_params)
    # engine = tenpy.TDVPEngine(psi, model, engine_params)
    
    t = [0]
    S = [psi.entanglement_entropy()]
    mag_z = [psi.expectation_value('Sz')]
    err = [0]
    
    for n in range(200):
        print(f'n={n}')
        engine.run()
        t.append(engine.evolved_time)
        S.append(psi.entanglement_entropy())
        mag_z.append(psi.expectation_value('Sz'))
        err.append(engine.trunc_err.eps)
    
    t = np.array(t)
    S = np.array(S)
    mag_z = np.array(mag_z)
    err = np.array(err)
    imbalance = .5 * np.average(mag_z[:, ::2] - mag_z[:, 1::2], axis=1),
    return dict(t=t, S=S, mag_z=mag_z, imbalance=imbalance, err=err, chi=chi)


def plot(folder):
    outfile = os.path.join(folder, 'heisenberg_tebd.pdf')
    results = {}
    for fn in os.listdir(folder):
        if not fn.startswith('heisenberg_tebd_'):
            continue
        if not fn.endswith('.pkl'):
            continue
        _, rest = fn.split('heisenberg_tebd_chi_')
        chi, _ = rest.split('.pkl')
        chi = int(chi)
        with open(os.path.join(folder, fn), 'rb') as f:
            res = pickle.load(f)
        results[chi] = res

    
    fontsize = 10
    linewidth = 5.90666  # inches
    L = model_params['L']
    mpl.rcParams.update({'font.size': fontsize})
    mpl.rcParams.update({'legend.fontsize': 8})
    mpl.rcParams.update({'legend.title_fontsize': 8})
    chis = sorted(results.keys())
    max_chi = max(chis)
    
    print(f't: {max(results[max_chi]['t'])}')

    assert len(chis) == 5
    plot_styles = {
        (chi, which): dict(
            color=c,
            label=rf'$\chi={chi}$',
            ls=ls,
            lw=2 if chi == max_chi else (1 if chi == min(chis) else 1.5)
        )
        for chi, c, ls in zip(chis,
                              ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red'],
                              ['-', ':', '--', '-.', '-'])
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
    im = ax_mag.pcolor(results[max_chi]['mag_z'].T, cmap='inferno', edgecolor='face')
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
        ax_Imb.plot(results[chi]['t'], results[chi]['imbalance'][0], **plot_styles[chi, 'Imb'])
    ax_Imb.legend()

    # subplot: truncation error
    ax_err.set_ylabel(r'$\varepsilon_\text{trunc}$')
    ax_err.set_xlabel('$t$')
    for chi in chis:
        ax_err.semilogy(results[chi]['t'], results[chi]['err'], **plot_styles[chi, 'err'])

    fig.tight_layout(pad=0.1)
    fig.savefig(outfile, bbox_inches='tight')
    print(f'saved to {outfile}')


if __name__ == '__main__':
    main()
