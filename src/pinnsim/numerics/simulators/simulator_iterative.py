from .simulator_module import Simulator

import abc
import torch
import math
import time as time_pkg


class IterativeSimulator(Simulator):
    def __init__(
        self,
        grid_model: object,
        timestep_size: float,
        newton_scheme_tolerance: float = 1.0e-8,
        newton_scheme_max_iterations: int = 20,
        verbosity=None,
    ):
        super(IterativeSimulator, self).__init__(
            grid_model=grid_model, verbosity=verbosity
        )
        self.timestep_size = timestep_size
        self.output_times = self.timestep_size
        self.newton_scheme_tolerance = newton_scheme_tolerance
        self.newton_scheme_max_iterations = newton_scheme_max_iterations

    @abc.abstractmethod
    def simulator_config(self):
        simulator_config = "Distributed simulator"
        return simulator_config

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

        time_current = 0.0
        time_results = list()
        component_states_results = list()
        theta_results = list()
        V_results = list()
        timestep_summaries = list()
        component_state_list_current = component_state_list_initial
        theta_current = theta_initial
        V_current = V_initial

        while time_current < time_end:
            timestep_results = self.simulate_timestep(
                component_state_list=component_state_list_current,
                control_input_list=control_input_list,
                theta=theta_current,
                V=V_current,
            )

            time_results.append(time_current + timestep_results["time"])
            component_states_results.append(timestep_results["component_states_list"])
            theta_results.append(timestep_results["theta"])
            V_results.append(timestep_results["V"])
            timestep_summaries.append(timestep_results["timestep_summary"])

            time_current = round(
                time_current + timestep_results["time"][-1:, 0].item(), 6
            )
            component_state_list_current = [
                component_state_list_result[-1:, :]
                for component_state_list_result in timestep_results[
                    "component_states_list"
                ]
            ]
            theta_current = timestep_results["theta"][-1:, :]
            V_current = timestep_results["V"][-1:, :]
        end_time = time_pkg.time()
        simulator_steps = math.ceil(round(time_end / self.timestep_size, 6))
        assert len(time_results) == simulator_steps
        component_states_list_results = [
            torch.vstack(
                [
                    component_state_list_result[ii]
                    for component_state_list_result in component_states_results
                ]
            )
            for ii in range(self.grid_model.n_components)
        ]

        if self.verbosity < 40:
            print(f"Simulation time: {(end_time - start_time)*1.0-9:.3f}s")

        iteration_list = [timestep["n_iterations"] for timestep in timestep_summaries]
        variable_change_list = [
            timestep["max_variable_change"] for timestep in timestep_summaries
        ]
        residual_list = [timestep["residual_norm"] for timestep in timestep_summaries]

        solver_stats = dict(
            {
                "solver": self.__repr__(),
                "solver_time": (end_time - start_time),
                "n_iterations": sum(iteration_list),
                "max_iterations": max(iteration_list),
                "mean_residual": sum(residual_list) / len(residual_list),
                "max_residual": max(residual_list),
                "mean_variable_change": sum(variable_change_list)
                / len(variable_change_list),
                "max_variable_change": max(variable_change_list),
                "time_end": time_end,
                "timestep_size": self.timestep_size,
                "newton_scheme_max_iterations": self.newton_scheme_max_iterations,
                "newton_scheme_tolerance": self.newton_scheme_tolerance,
            }
        )
        solver_stats.update(self.solver_dict_extra())

        output = dict(
            {
                "time": torch.vstack(time_results),
                "component_states_list": component_states_list_results,
                "theta": torch.vstack(theta_results),
                "V": torch.vstack(V_results),
                "control_input_list": control_input_list,
                "solver_stats": solver_stats,
            }
        )
        self.check_output(output)
        return output

    def solver_dict_extra(self):
        return dict({})

    @staticmethod
    def print_iteration_information(k_iteration, residual_norm, max_variable_update):
        print(
            f"Iteration {k_iteration:<4} | "
            f"residual norm {residual_norm:.2e} | "
            f"max variable change {max_variable_update:.2e}"
        )
        pass

    @abc.abstractmethod
    def simulate_timestep(
        self,
        component_state_list,
        control_input_list,
        theta,
        V,
    ):
        result_dict = dict(
            {
                "time": torch.zeros(self.output_times.shape),
                "component_states_list": [
                    torch.zeros((0, n_states))
                    for n_states in self.grid_model.state_split_list[:-2]
                ],
                "theta": torch.zeros(
                    (self.output_times.shape[0], self.grid_model.network.n_buses)
                ),
                "V": torch.zeros(
                    (self.output_times.shape[0], self.grid_model.network.n_buses)
                ),
                "timestep_summary": dict({}),
            }
        )
        return result_dict
