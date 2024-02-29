import torch
import wandb


def test_model(dataset, nn_model, loss_function):
    nn_model.eval()

    with torch.no_grad():
        (
            time,
            state_initial,
            control_input,
            voltage_parametrisation,
            state_result,
        ) = dataset

        state_prediction = nn_model.predict(
            time=time,
            state_initial=state_initial,
            control_input=control_input,
            voltage_parametrisation=voltage_parametrisation,
        )

    loss_testing = loss_function(inputs=state_prediction, targets=state_result)
    error_dict = log_absolute_error_metrics(
        prediction=state_prediction,
        target=state_result,
        state_names=nn_model.component.state_names,
    )
    error_dict["loss_testing"] = loss_testing
    wandb.log(error_dict, commit=True)
    pass


def log_absolute_error_metrics(prediction, target, state_names, error_prefix=""):
    absolute_errors_mean, absolute_errors_max = compute_absolute_error_metrics(
        prediction, target
    )
    error_dict = dict()
    for state_name, ae_mean, ae_max in zip(
        state_names, absolute_errors_mean, absolute_errors_max
    ):
        error_dict[f"mean_ae_{error_prefix}{state_name}"] = ae_mean
        error_dict[f"max_ae_{error_prefix}{state_name}"] = ae_max

    return error_dict


def compute_absolute_error_metrics(prediction, target):
    absolute_errors = torch.abs(target - prediction)
    absolute_errors_mean = torch.mean(absolute_errors, dim=0)
    absolute_errors_max = torch.amax(absolute_errors, dim=0)
    return absolute_errors_mean, absolute_errors_max
