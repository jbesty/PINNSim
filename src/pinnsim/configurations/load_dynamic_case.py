import torch

from pinnsim.configurations.load_power_grid import load_grid_model
from pinnsim.configurations.load_static_case_parameters import (
    load_static_case_parameters,
)
from pinnsim.numerics.power_flow_solvers.ac_pf_solver import ACPFSolver


def load_equilibrium_case(case):

    assert case in ["ieee9"]
    grid_model = load_grid_model(case=case)

    pf_solver = ACPFSolver(verbose=False)

    case_parameters = load_static_case_parameters(case_name="ieee9")
    theta, V = pf_solver.solve_power_flow_newton(
        network=grid_model.network,
        bus_types=case_parameters["bus_types"],
        set_points=case_parameters["set_points"],
    )
    P, Q = grid_model.network.compute_power(torch.hstack([theta, V]))

    set_points = [
        torch.stack([P[:, bus], Q[:, bus], theta[:, bus], V[:, bus]]).reshape((1, -1))
        for bus in grid_model.component_bus_indices
    ]
    (
        equilibrium_state_list,
        equilibrium_control_input_list,
    ) = grid_model.compute_equilibrium_values(set_points=set_points)
    return grid_model, equilibrium_state_list, equilibrium_control_input_list, theta, V
