"""A collection of tests for (classes in) :mod:`tenpy.models.model`.

.. todo ::
    A lot more to test, e.g. conversions of the different models
"""

from __future__ import division

import itertools

from tenpy.models import model, lattice
import tenpy.networks.site

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


def check_general_model(ModelClass, model_pars={}, check_pars={}):
    """Create a model for different sets of parameters.

    Parameters
    ----------
    ModelClass :
        We generate models of this class
    model_pars : dict
        Model parameters used.
    check_pars : dict
        pairs (`key`, `list of values`); we update ``model_paras[key]`` with any values of
        ``check_params[key]`` (in each possible combination!) an create a model for it.
    """
    M = ModelClass(model_pars.copy())  # with default parameters
    M.test_sanity()
    for vals in itertools.product(*check_pars.values()):
        params = model_pars.copy()
        for k, v in zip(check_pars.keys(), vals):
            params[k] = v
        M = ModelClass(params)
        M.test_sanity()
