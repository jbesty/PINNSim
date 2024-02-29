def check_required_config_keys(sweep_config):
    dataset_parameters = [
        "dataset_size",
        "dataset_split_seed",
        "dataset_collocation_size",
        "dataset_collocation_split_seed",
    ]
    core_network_parameters = [
        "hidden_layer_size",
        "n_hidden_layers",
        "nn_model_init_seed",
    ]

    optimiser_parameters = [
        "learning_rate",
        "tolerance_change",
        "tolerance_grad",
        "history_size",
        "line_search",
        "max_iterations",
    ]

    scheduler_parameters = [
        "physics_regulariser_max",
        "physics_regulariser_epochs_to_tenfold",
        "physics_regulariser_initial",
    ]

    workflow_parameters = [
        "epochs",
        "threads",
        "run_counter",
    ]

    required_keys = (
        dataset_parameters
        + core_network_parameters
        + optimiser_parameters
        + scheduler_parameters
        + workflow_parameters
    )

    for required_key in required_keys:
        assert (
            required_key in sweep_config["parameters"].keys()
        ), f"Please specify the hyper parameter {required_key}"

    pass
