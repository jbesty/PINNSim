import torch


def voltage_parametrisation_sampling():
    parameter_dict = dict(
        {
            "voltage_parametrisation_low": torch.tensor(
                [[-torch.pi, -0.1, -0.3, -0.4, -0.8, -0.5]]
            ),
            "voltage_parametrisation_high": torch.tensor(
                [[torch.pi, 0.35, 0.3, 0.4, 0.8, 0.5]]
            ),
        }
    )

    return parameter_dict


def define_time_sampling(time_sampling="linear"):
    if time_sampling == "linear":
        parameter_dict = dict(
            {
                "time_min": 0.0,
                "time_sampling": "linear",
            }
        )
    elif time_sampling == "log":
        parameter_dict = dict(
            {
                "time_min": 0.0001,
                "time_sampling": "log",
            }
        )
    else:
        raise Exception

    parameter_dict.update(time_max=0.5)

    return parameter_dict


def define_initial_state_sampling(state_sampling="annular"):
    assert state_sampling in ["radial", "annular", "square"]
    parameter_dict = dict({"radius": 0.25, "state_sampling": state_sampling})

    return parameter_dict


def define_dataset_config(dataset_type, generator_id, seed):
    if dataset_type == "train":
        n_points = 2500
        simulate_dataset = True
    elif dataset_type == "test":
        n_points = 4000
        simulate_dataset = True
    elif dataset_type == "collocation":
        n_points = 10000
        simulate_dataset = False
    else:
        raise Exception

    parameters_dict = dict(
        {
            "n_operating_points": n_points,
            "simulate_dataset": simulate_dataset,
            "name": f"{dataset_type}_{generator_id}",
            # "name": f"{dataset_type}_log_{generator_id}",
            # "name": f"{dataset_type}_linear_{generator_id}",
            "sampling_seed": seed,
            "generator_id": generator_id,
        }
    )
    parameters_dict.update(voltage_parametrisation_sampling())
    parameters_dict.update(define_time_sampling(time_sampling="linear"))
    # parameters_dict.update(define_time_sampling(time_sampling="log"))
    parameters_dict.update(define_initial_state_sampling(state_sampling="annular"))

    return parameters_dict
