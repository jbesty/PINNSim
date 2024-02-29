from pinnsim.configurations.hyperparameter_configs import (
    default_hyperparameter_setup,
)
from pinnsim.configurations.required_hyperparameters import (
    check_required_config_keys,
)


def test_default_wandb_config():
    config = default_hyperparameter_setup()
    check_required_config_keys(sweep_config=config)
