import torch
import torch.utils.data

from pinnsim.dataset_functions.dataset_object import TrajectoryDataset
from pinnsim.learning_functions.dynamical_system_NN import DynamicalSystemResidualNN
from pinnsim.learning_functions.loss_weight_scheduler import LossWeightScheduler
from pinnsim.power_system_models.voltage_profile_polynomial import (
    VoltageProfilePolynomial,
)


def setup_nn_model(config, power_system_model, training_dataset=None):
    nn_model = DynamicalSystemResidualNN(
        hidden_layer_size=config.hidden_layer_size,
        n_hidden_layers=config.n_hidden_layers,
        pytorch_init_seed=config.nn_model_init_seed,
        voltage_profile=VoltageProfilePolynomial(order_polynomial=2),
        component=power_system_model,
        use_states=[False, False, True, True],
        use_control_inputs=[False] * 2,
        use_voltage_parametrisation=[True] * 6,
    )

    if training_dataset is not None:
        nn_model.adjust_to_dataset(dataset=training_dataset.dataset)

    return nn_model


def setup_optimiser(nn_model, config):
    optimiser = torch.optim.LBFGS(
        params=list(nn_model.parameters()),
        lr=config.learning_rate,
        tolerance_change=config.tolerance_change,
        tolerance_grad=config.tolerance_grad,
        line_search_fn="strong_wolfe" if config.line_search else None,
        history_size=config.history_size,
        max_iter=config.max_iterations,
    )
    return optimiser


def setup_schedulers(nn_model, optimiser, config):
    learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimiser, gamma=1.0
    )

    loss_weight_scheduler = LossWeightScheduler(
        nn_model=nn_model,
        max_value=config.physics_regulariser_max,
        epochs_to_tenfold=config.physics_regulariser_epochs_to_tenfold,
        initial_value=config.physics_regulariser_initial,
    )

    return learning_rate_scheduler, loss_weight_scheduler


def setup_dataset(config, data_path):
    dataset_name_training = f"train_{config.generator_name}"
    dataset_name_testing = f"test_{config.generator_name}"
    dataset_name_collocation = f"collocation_{config.generator_name}"
    dataset_full = TrajectoryDataset.from_dataset_name(
        dataset_name_training, data_path=data_path
    )

    dataset_training, dataset_validation = dataset_full.split_dataset_80_20(
        dataset_size=config.dataset_size, seed=config.dataset_split_seed
    )
    dataset_testing = TrajectoryDataset.from_dataset_name(
        dataset_name_testing, data_path=data_path
    )
    dataset_collocation_full = TrajectoryDataset.from_dataset_name(
        dataset_name_collocation, data_path=data_path
    )

    dataset_collocation = dataset_collocation_full.sample_dataset_size(
        dataset_size=config.dataset_collocation_size,
        seed=config.dataset_collocation_split_seed,
    )

    assert (
        dataset_full.component_model.generator_config
        == dataset_testing.component_model.generator_config
        == dataset_collocation.dataset.component_model.generator_config
    )

    component = dataset_full.component_model

    return (
        dataset_training,
        dataset_validation,
        dataset_testing,
        dataset_collocation,
        component,
    )
