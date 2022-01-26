"""Initialization of the Heisenberg model on a kagome lattice."""

from tenpy.models.spins import SpinModel

model_params = {
    "S": 0.5,  # Spin 1/2
    "lattice": "Kagome",
    "bc_MPS": "infinite",
    "bc_y": "cylinder",
    "Ly": 2,  # defines cylinder circumference
    "conserve": "Sz",  # use Sz conservation
    "Jx": 1.,
    "Jy": 1.,
    "Jz": 1.  # Heisenberg coupling
}
model = SpinModel(model_params)
