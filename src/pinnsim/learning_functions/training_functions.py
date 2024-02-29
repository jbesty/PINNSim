import torch
import wandb


def train_epoch(
    dataset,
    dataset_collocation,
    nn_model,
    loss_function,
    optimiser,
):
    nn_model.train()
    time, state_initial, control_input, voltage_parametrisation, state_result = dataset
    (
        time_c,
        state_initial_c,
        control_input_c,
        voltage_parametrisation_c,
        _,
    ) = dataset_collocation
    loss_multiplier = torch.tensor(nn_model.epochs_total + 1)

    def closure():
        optimiser.zero_grad()

        state_prediction = nn_model.forward(
            time=time,
            state_initial=state_initial,
            control_input=control_input,
            voltage_parametrisation=voltage_parametrisation,
        )

        loss_prediction = loss_function(inputs=state_prediction, targets=state_result)

        if nn_model.physics_regulariser > 0.0:
            _, d_dt_state_prediction_c, f_state_prediction_c = nn_model.forward_lhs_rhs(
                time=time_c,
                state_initial=state_initial_c,
                control_input=control_input_c,
                voltage_parametrisation=voltage_parametrisation_c,
            )

            loss_physics = loss_function(
                inputs=d_dt_state_prediction_c, targets=f_state_prediction_c
            )
        else:
            loss_physics = torch.tensor(0.0)

        loss = (
            loss_prediction + nn_model.physics_regulariser * loss_physics
        ) * loss_multiplier
        loss.backward()

        return loss

    optimiser.step(closure)
    state_prediction = nn_model.forward(
        time=time,
        state_initial=state_initial,
        control_input=control_input,
        voltage_parametrisation=voltage_parametrisation,
    )
    loss_prediction = loss_function(inputs=state_prediction, targets=state_result)
    loss = closure()
    wandb.log(
        {
            "loss_prediction": loss_prediction,
            "loss_total": loss / loss_multiplier,
            "epoch": nn_model.epochs_total,
        },
        commit=False,
    )

    return loss / loss_multiplier, loss_prediction


def evaluate_model(
    dataset,
    nn_model,
    loss_function,
):
    nn_model.eval()

    with torch.no_grad():
        (
            time,
            state_initial,
            control_input,
            voltage_parametrisation,
            state_result,
        ) = dataset

        (
            state_prediction,
            d_dt_state_prediction,
            f_state_prediction,
        ) = nn_model.forward_lhs_rhs(
            time=time,
            state_initial=state_initial,
            control_input=control_input,
            voltage_parametrisation=voltage_parametrisation,
        )

    loss_prediction = loss_function(inputs=state_prediction, targets=state_result)
    loss_physics = loss_function(
        inputs=d_dt_state_prediction, targets=f_state_prediction
    )

    wandb.log(
        {
            "epoch": nn_model.epochs_total,
            "loss_validation": loss_prediction,
            "loss_validation_physics": loss_physics,
        },
        commit=False,
    )

    return loss_prediction
