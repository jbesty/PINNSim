import pathlib

import torch

# This file contains the relevant parameters that would be either hardcoded or that are not subject to change in the
# study. Furthermore, variables of 'global' interest are stored here.

wandb_entity = "jstiasny"
wandb_project = "pinnsim_pscc"

torch.set_default_dtype(torch.float64)
f_0_Hz = torch.tensor(50.0)

PROJECT_PATH = pathlib.Path(__file__).absolute().parent.parent.parent
DATA_PATH = PROJECT_PATH / "data"
LEARNING_DATA_PATH = DATA_PATH / "learning_data"
NN_MODEL_DATA_PATH = DATA_PATH / "nn_model_data"
