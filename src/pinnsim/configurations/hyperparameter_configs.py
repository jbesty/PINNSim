from types import SimpleNamespace

from pinnsim import wandb_entity, wandb_project


def default_dataset_parameters():
    parameters_dict = {
        "dataset_size": {"value": 200},
        "dataset_split_seed": {"value": None},
        "dataset_collocation_size": {"value": 500},
        "dataset_collocation_split_seed": {"value": None},
        "generator_name": {"value": "ieee9_1"},
    }

    return parameters_dict


def default_core_network_parameters():
    parameters_dict = {
        "hidden_layer_size": {"value": 32},
        "n_hidden_layers": {"value": 2},
        "nn_model_init_seed": {"value": None},
    }

    return parameters_dict


def default_optimiser_parameters():
    parameters_dict = {
        "learning_rate": {"value": 1.0},
        "tolerance_change": {"value": 1e-9},
        "tolerance_grad": {"value": 1e-9},
        "history_size": {"value": 120},
        "line_search": {"value": False},
        "max_iterations": {"value": 25},
    }
    return parameters_dict


def default_scheduler_parameters():
    parameters_dict = {
        "physics_regulariser_max": {"value": 0.1},
        "physics_regulariser_epochs_to_tenfold": {"value": 10},
        "physics_regulariser_initial": {"value": 1.0e-6},
    }
    return parameters_dict


def default_workflow_parameters():
    parameters_dict = {
        "epochs": {"value": 50},
        "threads": {"value": 12},
        "run_counter": {"value": 0},
    }
    return parameters_dict


def default_hyperparameter_setup():
    parameters_dict = {}
    parameters_dict.update(default_dataset_parameters())
    parameters_dict.update(default_core_network_parameters())
    parameters_dict.update(default_optimiser_parameters())
    parameters_dict.update(default_scheduler_parameters())
    parameters_dict.update(default_workflow_parameters())

    sweep_config = {
        "program": "training_workflow.py",
        "method": "grid",
        "name": "default",
        "parameters": parameters_dict,
        "metric": {"name": "loss_validation_best", "goal": "minimize"},
    }

    return sweep_config


def convert_sweep_config_to_run_config(sweep_config):
    run_config_dict = dict(
        zip(
            list(sweep_config["parameters"].keys()),
            [subdict["value"] for subdict in sweep_config["parameters"].values()],
        )
    )
    run_config_dict.update({"project": wandb_project, "entity": wandb_entity})
    run_config = SimpleNamespace(**run_config_dict)
    return run_config


def ieee9_machines():
    sweep_config = default_hyperparameter_setup()

    sweep_config["parameters"].update(
        epochs={"value": 2000},
        dataset_size={"value": 2500},
        dataset_collocation_size={"value": 5000},
        generator_name={"values": ["ieee9_1", "ieee9_2", "ieee9_3"]},
    )
    sweep_config.update(name="pinnsim_ieee9_machines")
    return sweep_config
