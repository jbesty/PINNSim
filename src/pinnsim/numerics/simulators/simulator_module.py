import abc
import time as time_pkg

import torch


class Simulator:
    def __init__(self, grid_model, output_timestep_size: float = 0.001, verbosity=None):
        super(Simulator, self).__init__()

        self.grid_model = grid_model
        if verbosity is None:
            verbosity = 40
        self.verbosity = verbosity
        self.output_timestep_size = output_timestep_size
        self.output_times = 0.0
        self.rtol = 1.0e-8
        self.atol = 1.0e-8

    @property
    def output_times(self):
        return self._output_times

    @output_times.setter
    def output_times(self, value):
        self._output_times = torch.round(
            torch.arange(
                0.0,
                value + self.output_timestep_size,
                self.output_timestep_size,
            ),
            decimals=6,
        ).reshape((-1, 1))

    @abc.abstractmethod
    def __repr__(self):
        pass

    @abc.abstractmethod
    def simulate_trajectory(
        self,
        time_end,
        component_state_list_initial,
        control_input_list,
        theta_initial,
        V_initial,
        time_repeats=False,
    ):
        start_time = time_pkg.time()
        end_time = time_pkg.time()
        result_time = torch.zeros((0, 1))
        component_states_list = [
            torch.zeros((0, n_states))
            for n_states in self.grid_model.state_split_list[:-2]
        ]
        result_theta = torch.zeros((0, self.grid_model.n_buses))
        result_V = torch.zeros((0, self.grid_model.n_buses))
        solver_stats = dict(
            {
                "solver_time": end_time - start_time,
                "iteration_summaries": [],
            }
        )
        output = dict(
            {
                "time": result_time,
                "component_states_list": component_states_list,
                "theta": result_theta,
                "V": result_V,
                "control_input_list": control_input_list,
                "solver_stats": solver_stats,
            }
        )
        self.check_output(output)
        return output

    def check_output(self, output):
        assert (
            output["time"].shape[0]
            == output["component_states_list"][0].shape[0]
            == output["theta"].shape[0]
            == output["V"].shape[0]
        )
        assert output["time"].shape[1] == 1
        assert [
            component_states.shape[1]
            for component_states in output["component_states_list"]
        ] == self.grid_model.state_split_list[:-2]
        assert (
            output["theta"].shape[1]
            == output["V"].shape[1]
            == self.grid_model.network.n_buses
        )
        pass
