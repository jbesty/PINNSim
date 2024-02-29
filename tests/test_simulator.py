from pinnsim.configurations.load_dynamic_case import load_equilibrium_case
from pinnsim.numerics.simulators.simulator_assimulo import SimulatorAssimulo


def test_assimulo():
    grid_model, equilibrium_state_list, equilibrium_control_input_list, theta, V = (
        load_equilibrium_case(case="ieee9")
    )

    simulator = SimulatorAssimulo(grid_model=grid_model)

    simulator.simulate_trajectory(
        time_end=0.2,
        component_state_list_initial=equilibrium_state_list,
        control_input_list=equilibrium_control_input_list,
        theta_initial=theta,
        V_initial=V,
    )
    pass


def test_assimulo_disturbed():
    grid_model, equilibrium_state_list, equilibrium_control_input_list, theta, V = (
        load_equilibrium_case(case="ieee9")
    )

    simulator = SimulatorAssimulo(grid_model=grid_model)

    equilibrium_control_input_list[0][:, 0] *= 0.5

    simulator.simulate_trajectory(
        time_end=0.2,
        component_state_list_initial=equilibrium_state_list,
        control_input_list=equilibrium_control_input_list,
        theta_initial=theta,
        V_initial=V,
    )
    pass
