import torch
from pinnsim.configurations.load_network_data import (
    import_matpower_acopf_case,
    load_network_data,
)
from pinnsim.configurations.load_static_case_parameters import (
    load_static_case_parameters,
)
from pinnsim.numerics.power_flow_solvers.ac_pf_solver import ACPFSolver
from pinnsim.power_system_models import NetworkModel


def test_matpower_import():
    import_matpower_acopf_case(case_name="ieee9")


def test_case_parameter():

    load_static_case_parameters(case_name="ieee9")


def test_network_model_construction():

    parameters = load_network_data(power_system_name="ieee9")

    NetworkModel(parameters)
    # pass


def test_power_flow_computation():
    network_parameters = load_network_data(power_system_name="ieee9")
    network_model = NetworkModel(network_parameters)

    case_parameters = load_static_case_parameters(case_name="ieee9")

    pf_solver = ACPFSolver(verbose=False)

    theta, V = pf_solver.solve_power_flow_newton(
        network=network_model,
        bus_types=case_parameters["bus_types"],
        set_points=case_parameters["set_points"],
    )
    P, Q = network_model.compute_power(torch.hstack([theta, V]))

    assert torch.max(torch.abs(case_parameters["set_points"]["P"] - P)) < 1.0e-4
    assert torch.max(torch.abs(case_parameters["set_points"]["Q"] - Q)) < 1.0e-4
    pass
