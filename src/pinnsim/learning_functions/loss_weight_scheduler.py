import torch


class LossWeightScheduler:
    def __init__(
        self,
        nn_model,
        max_value: float = 0.0,
        epochs_to_tenfold: int = 20,
        initial_value: float = 1.0e-7,
    ):
        self.max_value = torch.tensor(max_value)
        assert epochs_to_tenfold > 0
        self.epoch_factor = torch.tensor(10.0) ** (1 / epochs_to_tenfold)
        self.current_value = (
            torch.tensor(initial_value) if max_value > 0.0 else torch.tensor(0.0)
        )
        self.nn_model = nn_model

    def __call__(self):
        self.current_value = torch.minimum(
            self.current_value * self.epoch_factor, self.max_value
        )
        self.nn_model.physics_regulariser = self.current_value
