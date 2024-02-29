import os
import pathlib
import shutil
import time as time_pkg

import torch

import wandb
from pinnsim import LEARNING_DATA_PATH, learning_functions


def train(config=None):
    # Initialize a new wandb run
    # with wandb.init(config=config, mode="offline") as run:
    with wandb.init(config=config) as run:
        config = wandb.config

        torch.set_num_threads(config.threads)

        model_artifact = wandb.Artifact(f"model_{run.id}", type="model")
        model_save_path = f"{run.dir}\\model.pth"

        loss_validation_best = torch.tensor(1000.0)
        loss_training_best = torch.tensor(1000.0)
        best_epoch = 0
        training_time = 0.0

        (
            dataset_training,
            dataset_validation,
            dataset_testing,
            dataset_collocation,
            component_model,
        ) = learning_functions.setup_dataset(
            config=config, data_path=LEARNING_DATA_PATH
        )

        nn_model = learning_functions.setup_nn_model(
            config=config,
            power_system_model=component_model,
            training_dataset=dataset_training,
        )

        optimiser = learning_functions.setup_optimiser(nn_model=nn_model, config=config)

        (
            learning_rate_scheduler,
            loss_weight_scheduler,
        ) = learning_functions.setup_schedulers(
            nn_model=nn_model, optimiser=optimiser, config=config
        )

        loss_function = learning_functions.LossNormedState(
            component_model=component_model
        )

        while nn_model.epochs_total < config.epochs:
            time_epoch_train_start = time_pkg.time()
            loss, loss_training = learning_functions.train_epoch(
                dataset=dataset_training[:],
                dataset_collocation=dataset_collocation[:],
                nn_model=nn_model,
                loss_function=loss_function,
                optimiser=optimiser,
            )
            training_time += time_pkg.time() - time_epoch_train_start

            loss_validation = learning_functions.evaluate_model(
                dataset=dataset_validation[:],
                nn_model=nn_model,
                loss_function=loss_function,
            )

            if loss_validation < loss_validation_best:
                best_epoch = nn_model.epochs_total
                loss_validation_best = loss_validation.detach()
                loss_training_best = loss_training.detach()
                torch.save(nn_model.state_dict(), model_save_path)

            learning_rate_scheduler.step()
            loss_weight_scheduler()

            wandb.log(
                {
                    "epoch": nn_model.epochs_total,
                    "physics_regulariser": nn_model.physics_regulariser.detach(),
                },
                commit=True,
            )
            nn_model.epochs_total += 1

        nn_model.load_state_dict(torch.load(model_save_path))
        wandb.log(
            {
                "best_epoch": best_epoch,
                "loss_validation_best": loss_validation_best,
                "loss_training_best": loss_training_best,
                "training_time": training_time,
            },
            commit=True,
        )
        nn_model.eval()
        learning_functions.test_model(
            dataset=dataset_testing[:],
            nn_model=nn_model,
            loss_function=loss_function,
        )

        torch.save(nn_model.state_dict(), model_save_path)
        model_artifact.add_file(model_save_path)
        run.log_artifact(model_artifact)
        run.finish()

        # clean the directory
    logs_directory = pathlib.Path(run.dir).parent
    shutil.rmtree(logs_directory / "files")
    shutil.rmtree(logs_directory / "tmp")
    os.unlink(logs_directory / f"run-{run.id}.wandb")
    os.unlink(logs_directory / "logs" / "debug.log")
    os.unlink(logs_directory / "logs" / "debug-internal.log")
    return run
