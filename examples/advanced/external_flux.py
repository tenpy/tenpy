"""Add external flux to a Haldane model on a cylinder to observe a charge pump.

Re-creates (parts of) figure 2 in :arxiv:`1407.6985`.
"""
# Copyright 2019 TeNPy Developers

import numpy as np

from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS

# TODO: move model in tenpy/models/
from tenpy.models.model import CouplingMPOModel
from tenpy.tools.params import get_parameter
from tenpy.networks.site import FermionSite


class FermionicHaldaneModel(CouplingMPOModel):
    def __init__(self, model_params):
        model_params.setdefault('lattice', 'Honeycomb')
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        conserve = get_parameter(model_params, 'conserve', 'N', self.name)
        filling = get_parameter(model_params, 'filling', 1 / 2., self.name)
        site = FermionSite(conserve=conserve, filling=filling)
        return site

    def init_terms(self, model_params):
        t = get_parameter(model_params, 't', -1., self.name, True)
        V = get_parameter(model_params, 'V', 1, self.name, True)
        mu = get_parameter(model_params, 'mu', 0., self.name, True)
        phi_ext = 2 * np.pi * get_parameter(model_params, 'phi_ext', 0., self.name)

        phi = np.arccos(3 * np.sqrt(3 / 43))
        t2 = (np.sqrt(129) / 36) * t * np.exp(1j * phi)
        # t2 = 0  # TODO: external parameter!

        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(mu, 0, 'N', category='mu N')
            self.add_onsite(-mu, 1, 'N', category='mu N')

        for u1, u2, dx in self.lat.nearest_neighbors:
            t_phi = self.coupling_strength_add_ext_flux(t, dx, [0, phi_ext])
            self.add_coupling(t_phi, u1, 'Cd', u2, 'C', dx, 'JW', True, category='t Cd_i C_j')
            self.add_coupling(np.conj(t_phi),
                              u2,
                              'Cd',
                              u1,
                              'C',
                              -dx,
                              'JW',
                              True,
                              category='t Cd_i C_j h.c.')  # h.c.
            self.add_coupling(V, u1, 'N', u2, 'N', dx, category='V N_i N_j')

        for u1, u2, dx in [
            (0, 0, np.array([1, 0])),
            (0, 0, np.array([0, -1])),
            (0, 0, np.array([-1, 1])),  # first triangle counter-clockwise
            (1, 1, np.array([-1, 0])),
            (1, 1, np.array([0, 1])),
            (1, 1, np.array([1, -1]))
        ]:  # second triangle counter-clockwise
            t2_phi = self.coupling_strength_add_ext_flux(t2, dx, [0, phi_ext])
            self.add_coupling(t2_phi, u1, 'Cd', u2, 'C', dx, 'JW', True, category='t2 Cd_i C_j')
            self.add_coupling(np.conj(t2_phi),
                              u2,
                              'Cd',
                              u1,
                              'C',
                              -dx,
                              'JW',
                              True,
                              category='t2 Cd_i C_j h.c.')  # h.c.


def run(phi_ext=np.linspace(0, 1.0, 11)):
    data = dict(phi_ext=phi_ext, QL=[], ent_spectrum=[])

    model_params = dict(conserve='N',
                        filling=1 / 2.,
                        Lx=1,
                        Ly=3,
                        t=-1.,
                        mu=0,
                        V=1.,
                        phi_ext=0.,
                        bc_MPS='infinite',
                        bc_y='cylinder',
                        verbose=0)
    dmrg_params = {
        'mixer': True,  # setting this to True helps to escape local minima
        'mixer_params': {
            'amplitude': 1.e-5,
            'decay': 1.2,
            'disable_after': 30
        },
        'trunc_params': {
            'svd_min': 1.e-10,
        },
        'lanczos_params': {
            'N_min': 5,
            'N_max': 20
        },
        'chi_list': {
            0: 9,
            10: 49,
            20: 200
        },
        'max_E_err': 1.e-10,
        'max_S_err': 1.e-6,
        'max_sweeps': 150,
        'verbose': 1.,
    }
    prod_state = ['full', 'empty'] * (model_params['Lx'] * model_params['Ly'] * 2 // 2)
    # TODO allow tiling of list in from_product_state...
    eng = None
    for phi in phi_ext:
        print("=" * 100)
        print("phi_ext = ", phi)
        model_params['phi_ext'] = phi
        if eng is None:  # first time in the loop
            M = FermionicHaldaneModel(model_params)
            psi = MPS.from_product_state(M.lat.mps_sites(), prod_state, bc=M.lat.bc_MPS)
            eng = dmrg.EngineCombine(psi, M, dmrg_params)
        else:
            del eng.DMRG_params['chi_list']
            M = FermionicHaldaneModel(model_params)
            eng.init_env(model=M)
        E, psi = eng.run()
        data['QL'].append(psi.average_charge(bond=0)[0])
        data['ent_spectrum'].append(psi.entanglement_spectrum(by_charge=True)[0])
    return data


def plot_model():
    model_params = dict(conserve='N',
                        filling=1 / 2.,
                        Lx=1,
                        Ly=3,
                        t=-1.,
                        mu=0,
                        V=1.,
                        phi_ext=0.1,
                        bc_MPS='infinite',
                        bc_y='cylinder',
                        verbose=0)
    M = FermionicHaldaneModel(model_params)
    import matplotlib.pyplot as plt
    plt.figure()
    ax = plt.gca()
    M.lat.plot_sites(ax)
    M.coupling_terms['t Cd_i C_j'].plot_coupling_terms(
        ax,
        M.lat,
    )
    M.coupling_terms['t2 Cd_i C_j'].plot_coupling_terms(ax,
                                                        M.lat,
                                                        text='{op_j!s} {strength_angle:.2f}',
                                                        text_pos=0.9)
    print(M.coupling_terms['t Cd_i C_j'].to_TermList())
    ax.set_aspect(1.)
    plt.show()


def plot_results(data):
    import matplotlib.pyplot as plt
    plt.figure()
    ax = plt.gca()
    ax.plot(data['phi_ext'], data['QL'], marker='o')
    ax.set_xlabel(r"$\phi / 2 \pi$")
    ax.set_ylabel(r"$ Q_L $")
    plt.savefig("external_flux_charge.pdf")

    plt.figure()
    ax = plt.gca()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    color_by_charge = {}
    for phi_ext, spectrum in zip(data['phi_ext'], data['ent_spectrum']):
        for q, s in spectrum:
            q = q[0]
            label = ""
            if q not in color_by_charge:
                label = "q={q:+d}".format(q=q)
                color_by_charge[q] = colors[len(color_by_charge) % len(colors)]
            color = color_by_charge[q]
            ax.plot(phi_ext * np.ones(s.shape),
                    s,
                    linestyle='',
                    marker='_',
                    color=color,
                    label=label)
    ax.set_xlabel(r"$\phi / 2 \pi$")
    ax.set_ylabel(r"$ \epsilon_\alpha $")
    ax.set_ylim(0., 8.)
    ax.legend(loc='upper right')
    plt.savefig("external_flux_spectrum.pdf")


if __name__ == "__main__":
    #  plot_model()
    data = run()
    plot_results(data)
