import torch
from torch.utils.data import Dataset

from .. import power_system_models
from .dataset_handling import load_dataset_raw


class TrajectoryDataset(Dataset):
    def __init__(self, dataset):
        self.component_model = power_system_models.GeneratorModel(
            generator_config=dataset["generator_config"]
        )

        self.time = dataset["time"]
        self.state_initial = dataset["state_initial"]
        self.control_input = dataset["control_input"]
        self.state_result = dataset["state_result"]
        self.state_equilibrium = dataset["state_equilibrium"]
        self.voltage_parametrisation = dataset["voltage_parametrisation"]
        self.theta_result = dataset["theta_result"]
        self.V_result = dataset["V_result"]

        self.d_dt_state_initial = self.component_model.update_function(
            time=self.time * 0.0,
            state=self.state_initial,
            control_input=self.control_input,
            theta=self.voltage_parametrisation[:, 0:1],
            V=self.voltage_parametrisation[:, 1:2],
        )
        self.d_dt_state_result = self.component_model.update_function(
            time=self.time,
            state=self.state_result,
            control_input=self.control_input,
            theta=self.theta_result,
            V=self.V_result,
        )

        self.prediction_updated = False
        self.state_prediction = torch.zeros(self.state_result.shape)
        self.d_dt_state_prediction = torch.zeros(self.state_result.shape)
        self.update_function_prediction = torch.zeros(self.state_result.shape)

        self.state_error = torch.zeros(self.state_result.shape)
        self.lhs_error = torch.zeros(self.state_result.shape)
        self.rhs_error = torch.zeros(self.state_result.shape)
        self.physics_error = torch.zeros(self.state_result.shape)

    def __len__(self):
        return self.time.shape[0]

    def __getitem__(self, idx):
        return (
            self.time[idx, :],
            self.state_initial[idx, :],
            self.control_input[idx, :],
            self.voltage_parametrisation[idx, :],
            self.state_result[idx, :],
        )

    def getitem2(self, idx):
        return (
            self.time[idx : idx + 1, :],
            self.state_initial[idx : idx + 1, :],
            self.control_input[idx : idx + 1, :],
            self.voltage_parametrisation[idx : idx + 1, :],
            self.state_result[idx : idx + 1, :],
        )

    def __repr__(self):
        return f"Dataset for {self.component_model.__repr__()}."

    def split_dataset_80_20(self, dataset_size, seed):
        if seed is None:
            seed = torch.randint(low=1000000, high=100000000, size=(1,)).item()

        assert dataset_size <= self.__len__()

        dataset_training, dataset_validation, _ = torch.utils.data.random_split(
            dataset=self,
            lengths=[
                int(dataset_size * 0.8),
                int(dataset_size * 0.2),
                self.__len__() - dataset_size,
            ],
            generator=torch.Generator().manual_seed(seed),
        )
        return dataset_training, dataset_validation

    def split_dataset_shares(self, dataset_size, seed, shares=None):
        if shares is None:
            shares = [0.8, 0.2]
        assert len(shares) == 2
        if seed is None:
            seed = torch.randint(low=1000000, high=100000000, size=(1,)).item()

        assert dataset_size <= self.__len__()

        dataset_training, dataset_validation, _ = torch.utils.data.random_split(
            dataset=self,
            lengths=[
                int(dataset_size * shares[0]),
                int(dataset_size * shares[1]),
                self.__len__() - dataset_size,
            ],
            generator=torch.Generator().manual_seed(seed),
        )
        return dataset_training, dataset_validation

    def sample_dataset_size(self, dataset_size, seed):
        if seed is None:
            seed = torch.randint(low=1000000, high=100000000, size=(1,)).item()

        assert dataset_size <= self.__len__()

        dataset_training, _ = torch.utils.data.random_split(
            dataset=self,
            lengths=[
                dataset_size,
                self.__len__() - dataset_size,
            ],
            generator=torch.Generator().manual_seed(seed),
        )
        return dataset_training

    def update_predictions_and_errors(self, state_prediction, d_dt_state_prediction):
        self.state_prediction = state_prediction.detach().clone()
        self.d_dt_state_prediction = d_dt_state_prediction.detach().clone()

        self.update_function_prediction = self.component_model.update_function(
            time=self.time,
            state=self.state_prediction,
            control_input=self.control_input,
            theta=self.theta_result,
            V=self.V_result,
        )

        self.state_error = self.state_prediction - self.state_result
        self.lhs_error = self.d_dt_state_prediction - self.d_dt_state_result
        self.rhs_error = self.update_function_prediction - self.d_dt_state_result
        self.physics_error = (
            self.d_dt_state_prediction - self.update_function_prediction
        )

        self.prediction_updated = True

    def export_dataset(self, component):
        export_dataset = dict(
            {
                "time": self.time.numpy(),
                "radius": torch.linalg.norm(
                    (self.state_initial - self.state_equilibrium)
                    * component.scale_to_norm,
                    ord=2,
                    dim=1,
                    keepdim=True,
                ).numpy(),
                "error": self.state_error.numpy(),
                "error_normed_L2": torch.linalg.norm(
                    self.state_error * component.scale_to_norm,
                    ord=2,
                    dim=1,
                    keepdim=True,
                ).numpy(),
                "error_normed_L1": torch.linalg.norm(
                    self.state_error * component.scale_to_norm,
                    ord=1,
                    dim=1,
                    keepdim=True,
                ).numpy(),
                "error_normed_LInf": torch.linalg.norm(
                    self.state_error * component.scale_to_norm,
                    ord=torch.inf,
                    dim=1,
                    keepdim=True,
                ).numpy(),
            }
        )
        return export_dataset

    @classmethod
    def from_dataset_name(cls, dataset_name, data_path):
        dataset_raw = load_dataset_raw(dataset_name=dataset_name, data_path=data_path)
        return cls(dataset=dataset_raw)
