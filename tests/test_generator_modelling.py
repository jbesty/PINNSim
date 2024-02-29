import torch
from pinnsim.configurations.load_generator_data import get_machine_data
from pinnsim.power_system_models.generator_model import GeneratorModel


def test_equilibrium_calculation_formats():

    generator_config = get_machine_data(seed="ieee9_2")
    generator_model = GeneratorModel(generator_config=generator_config)

    n_points = 100
    P, Q = torch.rand((n_points, 2)).split(split_size=1, dim=1)
    theta = torch.rand((n_points, 1)) * 2 * torch.pi - torch.pi
    V = torch.rand((n_points, 1)) * 0.1 + 0.95
    set_point = torch.hstack([P, Q, theta, V])
    equilibrium_state, equilibrium_control_input = generator_model.compute_equilibrium(
        set_point=set_point
    )
    assert equilibrium_state.shape[1] == generator_model.n_states
    assert equilibrium_control_input.shape[1] == generator_model.n_control_inputs


def test_equilibrium_power_calculation():

    generator_config = get_machine_data(seed="ieee9_2")
    generator_model = GeneratorModel(generator_config=generator_config)

    n_points = 100
    P, Q = torch.rand((n_points, 2)).split(split_size=1, dim=1)
    theta = torch.rand((n_points, 1)) * 2 * torch.pi - torch.pi
    V = torch.rand((n_points, 1)) * 0.1 + 0.95
    set_point = torch.hstack([P, Q, theta, V])
    equilibrium_state, equilibrium_control_input = generator_model.compute_equilibrium(
        set_point=set_point
    )
    E_q_prime, E_d_prime, delta, _ = equilibrium_state.split(split_size=1, dim=1)
    I_d, I_q = generator_model.compute_current_d_q(
        E_q_prime, E_d_prime, delta, theta, V
    )
    P_hat, Q_hat = generator_model.compute_power_injection(I_d, I_q, delta, theta, V)
    power_error = (P + 1j * Q) - (P_hat + 1j * Q_hat)

    assert torch.max(torch.abs(power_error)) < 1.0e-8


def test_equilibrium_update_equation():

    generator_config = get_machine_data(seed="ieee9_2")
    generator_model = GeneratorModel(generator_config=generator_config)

    n_points = 100
    P, Q = torch.rand((n_points, 2)).split(split_size=1, dim=1)
    theta = torch.rand((n_points, 1)) * 2 * torch.pi - torch.pi
    V = torch.rand((n_points, 1)) * 0.1 + 0.95
    set_point = torch.hstack([P, Q, theta, V])
    equilibrium_state, equilibrium_control_input = generator_model.compute_equilibrium(
        set_point=set_point
    )
    update_state = generator_model.update_function(
        time=torch.zeros((n_points, 1)),
        state=equilibrium_state,
        control_input=equilibrium_control_input,
        theta=theta,
        V=V,
    )
    assert torch.max(torch.abs(update_state)) < 1.0e-8


def test_D_Q_transform():

    generator_config = get_machine_data(seed="ieee9_2")
    generator_model = GeneratorModel(generator_config=generator_config)

    n_points = 100
    P, Q = torch.rand((n_points, 2)).split(split_size=1, dim=1)
    theta = torch.rand((n_points, 1)) * 2 * torch.pi - torch.pi
    V = torch.rand((n_points, 1)) * 0.1 + 0.95
    set_point = torch.hstack([P, Q, theta, V])
    equilibrium_state, equilibrium_control_input = generator_model.compute_equilibrium(
        set_point=set_point
    )
    E_q_prime, E_d_prime, delta, _ = equilibrium_state.split(split_size=1, dim=1)
    I_d, I_q = generator_model.compute_current_d_q(
        E_q_prime, E_d_prime, delta, theta, V
    )

    I_D, I_Q = generator_model.compute_current_D_Q(
        state=equilibrium_state, theta=theta, V=V
    )

    I_D_transformed, I_Q_transformed = generator_model.transform_currents_d_q_to_D_Q(
        I_d=I_d, I_q=I_q, delta=delta
    )
    assert torch.max(torch.abs(I_D - I_D_transformed)) < 1.0e-8
    assert torch.max(torch.abs(I_Q - I_Q_transformed)) < 1.0e-8
