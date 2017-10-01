"""A collection of tests for (classes in) :mod:`tenpy.models.model`.

.. todo ::
    A lot more to test, e.g. conversions of the different models
"""

from __future__ import division

import itertools

from tenpy.models import model, lattice
import tenpy.networks.site
import tenpy.linalg.np_conserved as npc
import test_mpo

spin_half_site = tenpy.networks.site.SpinHalfSite('Sz')
spin_half_lat = lattice.Chain(2, spin_half_site)

fermion_site = tenpy.networks.site.FermionSite('N')
fermion_lat = lattice.Chain(5, fermion_site)


def test_CouplingModel():
    for bc in ['open', 'periodic']:
        M = model.CouplingModel(spin_half_lat, bc)
        M.add_coupling(1.2, 0, 'Sz', 0, 'Sz', 1)
        M.test_sanity()
        M = model.CouplingModel(fermion_lat, bc)
        M.add_coupling(1.2, 0, 'Cd', 0, 'C', 1, 'JW')
        M.test_sanity()


def check_model_sanity(M, hermitian=True):
    """call M.test_sanity() for all different subclasses of M"""
    if isinstance(M, model.CouplingModel):
        model.CouplingModel.test_sanity(M)
    if isinstance(M, model.NearestNeighborModel):
        model.NearestNeighborModel.test_sanity(M)
        if hermitian:
            for i, H in enumerate(M.H_bond):
                if H is not None:
                    err = npc.norm(H - H.conj().transpose(H.get_leg_labels()))
                    if err > 1.e-14:
                        print H
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
        ``check_params[key]`` (in each possible combination!) an create a model for it.
    hermitian : bool
        If True, check that the Hamiltonian is hermitian.
    """
    for vals in itertools.product(*check_pars.values()):
        print "-"*40
        params = model_pars.copy()
        for k, v in zip(check_pars.keys(), vals):
            params[k] = v
        print "check_model_sanity with following parameters:"
        print params
        M = ModelClass(params)
        check_model_sanity(M)
