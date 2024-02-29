from pinnsim.configurations.load_component_parameters import (
    load_component_parameters,
)
from pinnsim.configurations.load_dynamic_case import load_equilibrium_case
from pinnsim.configurations.load_power_grid import load_grid_model


def test_component_parameters():
    load_component_parameters(case_name="ieee9")
    pass


def test_grid_model_constructor():
    load_grid_model(case="ieee9")
    pass


def test_dynamic_case_loader():
    load_equilibrium_case(case="ieee9")
    pass
