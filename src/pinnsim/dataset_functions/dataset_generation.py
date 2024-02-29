import time as time_pkg

import torch

from pinnsim.configurations.load_component_set_points import (
    load_component_set_points,
)
from pinnsim.configurations.load_generator_data import get_machine_data
from pinnsim.dataset_functions.dataset_handling import save_dataset_raw
from pinnsim.numerics.predictors.predictor_ode import PredictorODE
from pinnsim.power_system_models.generator_model import GeneratorModel
from pinnsim.power_system_models.voltage_profile_polynomial import (
    VoltageProfilePolynomial,
)


def simulate_dataset(dataset):
    generator = GeneratorModel(generator_config=dataset["generator_config"])

    voltage_profile = VoltageProfilePolynomial(order_polynomial=2)

    simulator = PredictorODE(component=generator, voltage_profile=voltage_profile)
    time_extended = torch.hstack([dataset["time"] * 0.0, dataset["time"]])

    with torch.no_grad():
        state_results = torch.vstack(
            [
                simulator.predict_state(
                    time=time_extended[ii : ii + 1, :].reshape((-1, 1)),
                    state=dataset["state_initial"][ii : ii + 1, :],
                    control_input=dataset["control_input"][ii : ii + 1, :],
                    voltage_parametrisation=dataset["voltage_parametrisation"][
                        ii : ii + 1, :
                    ],
                )[1:, :]
                for ii in range(dataset["time"].shape[0])
            ]
        )

    theta, V = voltage_profile.get_voltage(
        time=dataset["time"], voltage_parametrisation=dataset["voltage_parametrisation"]
    )

    assert state_results.shape == dataset["state_initial"].shape

    dataset.update(
        {
            "state_result": state_results,
            "theta_result": theta,
            "V_result": V,
        }
    )

    return dataset


def sample_time_values(n_points, time_sampling, time_min, time_max):
    if time_sampling == "linear":
        time_values = time_min + torch.rand((n_points, 1)) * (time_max - time_min)
    elif time_sampling == "log":
        time_values_log = torch.log(torch.tensor(time_min)) + torch.rand(
            (n_points, 1)
        ) * (torch.log(torch.tensor(time_max)) - torch.log(torch.tensor(time_min)))
        time_values = torch.exp(time_values_log)
    else:
        raise NotImplementedError

    return time_values


def sample_delta_omega_variation(n_points, state_sampling, radius):
    state_samples = torch.rand((n_points, 2))
    if state_sampling == "radial":
        radius_values = state_samples[:, 0:1] * radius
        angle_values = state_samples[:, 1:2] * 2 * torch.pi - torch.pi

        delta_omega_variation = torch.hstack(
            [
                radius_values * torch.cos(angle_values),
                radius_values * torch.sin(angle_values),
            ]
        )
    elif state_sampling == "annular":
        radius_values = torch.sqrt(state_samples[:, 0:1]) * radius
        angle_values = state_samples[:, 1:2] * 2 * torch.pi - torch.pi

        delta_omega_variation = torch.hstack(
            [
                radius_values * torch.cos(angle_values),
                radius_values * torch.sin(angle_values),
            ]
        )
    elif state_sampling == "square":
        delta_omega_variation = (
            (state_samples * 2 - 1.0) * radius / 2 * torch.sqrt(torch.tensor(torch.pi))
        )
    else:
        raise Exception("Define appropriate state sampling.")

    return delta_omega_variation


def sample_voltage_variation(
    n_points, voltage_parametrisation_low, voltage_parametrisation_high
):
    assert voltage_parametrisation_low.shape[1] == voltage_parametrisation_high.shape[1]
    voltage_parametrisation_variation = voltage_parametrisation_low + torch.rand(
        size=(n_points, voltage_parametrisation_low.shape[1])
    ) * (voltage_parametrisation_high - voltage_parametrisation_low)

    return voltage_parametrisation_variation


def sample_dataset(dataset_config):
    n_points = dataset_config["n_operating_points"]

    set_points = load_component_set_points(case="ieee9")
    if dataset_config["generator_id"] == "ieee9_1":
        set_point = set_points[0].reshape((1, -1))
        control_input_adjustment = torch.tensor([[0.5, 1.0]])
    elif dataset_config["generator_id"] == "ieee9_2":
        set_point = set_points[1].reshape((1, -1))
        control_input_adjustment = torch.tensor([[1.0, 1.0]])
    elif dataset_config["generator_id"] == "ieee9_3":
        set_point = set_points[2].reshape((1, -1))
        control_input_adjustment = torch.tensor([[1.0, 1.0]])
    else:
        raise Exception

    generator_config = get_machine_data(seed=dataset_config["generator_id"])
    component = GeneratorModel(generator_config=generator_config)
    state_equilibrium, equilibrium_control_input = component.compute_equilibrium(
        set_point=set_point
    )
    torch.random.manual_seed(seed=dataset_config["sampling_seed"])

    delta_omega_variation = sample_delta_omega_variation(
        n_points=n_points,
        state_sampling=dataset_config["state_sampling"],
        radius=dataset_config["radius"],
    )
    state_variations = (
        torch.hstack([torch.zeros((n_points, 2)), delta_omega_variation])
        * component.norm_to_scale
    )

    time_values = sample_time_values(
        n_points=n_points,
        time_sampling=dataset_config["time_sampling"],
        time_min=dataset_config["time_min"],
        time_max=dataset_config["time_max"],
    )
    # "hard coded quadratic voltage profiles"
    assert dataset_config["voltage_parametrisation_low"].shape[1] == 6
    voltage_parametrisation_variation = sample_voltage_variation(
        n_points=n_points,
        voltage_parametrisation_low=dataset_config["voltage_parametrisation_low"],
        voltage_parametrisation_high=dataset_config["voltage_parametrisation_high"],
    )

    voltage_parametrisation = voltage_parametrisation_variation + set_point[
        :, 3:4
    ] @ torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    initial_conditions_theta = (
        voltage_parametrisation_variation[:, 0:1] @ component.state_to_delta.T
    )

    initial_conditions = state_equilibrium + state_variations + initial_conditions_theta

    adjusted_control_input = equilibrium_control_input * control_input_adjustment
    dataset = dict(
        {
            "generator_config": component.generator_config,
            "set_point": set_point,
            "dataset_config": dataset_config,
            "time": time_values,
            "state_initial": initial_conditions,
            "control_input": adjusted_control_input.repeat((n_points, 1)),
            "state_equilibrium": state_equilibrium.repeat((n_points, 1)),
            "voltage_parametrisation": voltage_parametrisation,
        }
    )

    return dataset


def generate_dataset(dataset_config, data_path):
    start = time_pkg.time()
    dataset = sample_dataset(dataset_config=dataset_config)
    if dataset_config["simulate_dataset"]:
        dataset = simulate_dataset(dataset=dataset)
    else:
        assert dataset_config["voltage_parametrisation_low"].shape[1] == 6
        voltage_profile = VoltageProfilePolynomial(order_polynomial=2)
        theta, V = voltage_profile.get_voltage(
            dataset["time"], dataset["voltage_parametrisation"]
        )
        dataset.update(
            {
                "state_result": torch.zeros(dataset["state_initial"].shape),
                "theta_result": theta,
                "V_result": V,
            }
        )

    save_dataset_raw(
        dataset_raw=dataset, dataset_name=dataset_config["name"], data_path=data_path
    )
    end = time_pkg.time()
    print(f"Created and saved dataset {dataset_config['name']} in {end - start:.2f} s.")
    pass
