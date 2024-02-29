import pathlib
import pickle
from types import SimpleNamespace

import torch
from pinnsim import DATA_PATH, NN_MODEL_DATA_PATH, wandb_entity, wandb_project
from pinnsim.configurations.load_generator_data import get_machine_data
from pinnsim.learning_functions.setup_functions import setup_nn_model
from pinnsim.power_system_models.generator_model import GeneratorModel

import wandb


def rebuild_trained_nn_model(run_id):
    api = wandb.Api()
    run = api.run(f"{wandb_entity}/{wandb_project}/{run_id}")
    config_dict = run.config

    config = SimpleNamespace(**config_dict)

    artifact = api.artifact(f"{wandb_entity}/{wandb_project}/model_{run_id}:latest")
    try:
        artifact.manifest.entries["model.pth"]
    except KeyError:
        artifact.manifest.entries["model.pth"] = artifact.manifest.entries.pop(
            "files\\model.pth"
        )
        artifact.manifest.entries["model.pth"].path = "model.pth"

    artifact_dir = pathlib.Path(artifact.download(root=DATA_PATH / "wandb_artifacts"))

    component_model = GeneratorModel(
        generator_config=get_machine_data(seed=config.generator_name)
    )
    nn_model = setup_nn_model(config=config, power_system_model=component_model)

    nn_model.load_state_dict(torch.load(artifact_dir / "model.pth"))
    nn_model.eval()

    return nn_model, config


def save_model_and_config_locally(run_id):
    model, config = rebuild_trained_nn_model(run_id=run_id)

    config_file_path = NN_MODEL_DATA_PATH / f"config_{run_id}.pkl"
    with open(config_file_path, "wb") as file_path:
        pickle.dump(config, file_path)

    nn_state_dict_path = NN_MODEL_DATA_PATH / f"nn_state_dict_{run_id}.pkl"
    torch.save(model.state_dict(), nn_state_dict_path)

    print(f"Saved config and nn_state_dict for run {run_id}!")
    pass


def load_model_from_local_files(run_id):
    config_file_path = NN_MODEL_DATA_PATH / f"config_{run_id}.pkl"

    try:
        with open(config_file_path, "rb") as file_path:
            config = pickle.load(file_path)
    except FileNotFoundError:
        save_model_and_config_locally(run_id=run_id)
        with open(config_file_path, "rb") as file_path:
            config = pickle.load(file_path)

    nn_state_dict_path = NN_MODEL_DATA_PATH / f"nn_state_dict_{run_id}.pkl"
    component_model = GeneratorModel(
        generator_config=get_machine_data(seed=config.generator_name)
    )

    nn_model = setup_nn_model(config=config, power_system_model=component_model)

    nn_model.load_state_dict(torch.load(nn_state_dict_path))
    nn_model.eval()

    return nn_model, config
