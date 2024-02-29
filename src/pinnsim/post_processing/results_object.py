import torch

import definitions


class SimulationResult:
    def __init__(self, grid_model):
        super(SimulationResult, self).__init__()

        self.state_diff_result = None
        self.state_diff_result_exact = None
        self.timestep_size = None
        self.theta_diff_result = None
        self.theta_diff_result_exact = None
        self.grid_model = grid_model
        self.slack_bus = self.grid_model.network.slack_bus_index

        self.time_result = None
        self.state_result = None
        self.theta_result = None
        self.V_result = None
        self.control_input = None

        self.state_result_exact = None
        self.theta_result_exact = None
        self.V_result_exact = None
        self.network_I_D, self.network_I_Q = None, None
        self.network_I_D_exact, self.network_I_Q_exact = None, None

        self.components_I_D, self.components_I_Q = None, None
        self.components_I_D_exact, self.components_I_Q_exact = None, None

        self.error_theta, self.error_V, self.error_states = None, None, None

        self.error_I_D, self.error_I_Q = None, None

    def process_results(self, result, result_exact, relative_angle=True):
        self.assign_result(result=result)
        self.assign_exact_result(result_exact=result_exact)
        self.compute_currents(result=True, exact=True)
        self.compute_errors(relative_angle=relative_angle)
        pass

    def assign_result(self, result):
        self.time_result = result["time"]
        self.state_result = result["component_states_list"]
        self.theta_result = result["theta"]
        self.V_result = result["V"]
        self.control_input = result["control_input_list"]
        self.timestep_size = self.time_result[1, 0] - self.time_result[0, 0]
        self.state_diff_result, self.theta_diff_result = self.compute_diff_values(
            states=self.state_result, theta=self.theta_result
        )
        pass

    def assign_exact_result(self, result_exact):
        assert self.time_result is not None

        index_list = list()
        for time_value in self.time_result[:, 0]:
            index_list += torch.where(
                torch.isclose(time_value, result_exact["time"][:, 0])
            )[0].tolist()

        assert torch.allclose(
            self.time_result, result_exact["time"][index_list, :], atol=1e-6
        )
        assert all(
            [
                torch.allclose(
                    result_exact["control_input_list"][ii],
                    self.control_input[ii],
                    atol=1e-6,
                )
                for ii in range(len(self.control_input))
            ]
        )

        self.state_result_exact = [
            state_result[index_list, :]
            for state_result in result_exact["component_states_list"]
        ]
        self.theta_result_exact = result_exact["theta"][index_list, :]
        self.V_result_exact = result_exact["V"][index_list, :]
        (
            self.state_diff_result_exact,
            self.theta_diff_result_exact,
        ) = self.compute_diff_values(
            states=self.state_result_exact, theta=self.theta_result_exact
        )

        pass

    def assign_dataset_result_exact(self, result_exact):
        self.state_result_exact = [
            state_result for state_result in result_exact["component_states_list"]
        ]
        self.theta_result_exact = result_exact["theta"]
        self.V_result_exact = result_exact["V"]
        (
            self.state_diff_result_exact,
            self.theta_diff_result_exact,
        ) = self.compute_diff_values(
            states=self.state_result_exact, theta=self.theta_result_exact
        )
        pass

    def compute_diff_values(self, states, theta):
        theta_diff = theta - theta[:, self.slack_bus : self.slack_bus + 1]
        state_diff = list()
        for ii, (component, bus) in enumerate(
            zip(self.grid_model.components, self.grid_model.component_bus_indices)
        ):
            if component.component_type == "Generator":
                state_diff.append(
                    states[ii] - theta[:, bus : bus + 1] @ component.state_to_delta.T
                )
            else:
                state_diff.append(states[ii])
        return state_diff, theta_diff

    def compute_currents(self, result=True, exact=True):
        if result:
            (
                self.network_I_D,
                self.network_I_Q,
            ) = self.grid_model.network.compute_current_injections(
                torch.hstack([self.theta_result, self.V_result])
            )
            theta_list, V_list = self.grid_model.compute_voltage_lists(
                theta=self.theta_result, V=self.V_result
            )
            (
                self.components_I_D,
                self.components_I_Q,
            ) = self.grid_model.compute_component_current_injections(
                time=self.time_result,
                state_list=self.state_result,
                control_input_list=self.control_input,
                theta_list=theta_list,
                V_list=V_list,
            )

        if exact:
            (
                self.network_I_D_exact,
                self.network_I_Q_exact,
            ) = self.grid_model.network.compute_current_injections(
                torch.hstack([self.theta_result_exact, self.V_result_exact])
            )

            theta_list_exact, V_list_exact = self.grid_model.compute_voltage_lists(
                theta=self.theta_result_exact, V=self.V_result_exact
            )
            (
                self.components_I_D,
                self.components_I_Q,
            ) = self.grid_model.compute_component_current_injections(
                time=self.time_result,
                state_list=self.state_result_exact,
                control_input_list=self.control_input,
                theta_list=theta_list_exact,
                V_list=V_list_exact,
            )

        pass

    def compute_errors(self, relative_angle=True):
        slack_bus = self.grid_model.network.slack_bus_index

        self.error_V = self.V_result - self.V_result_exact
        if relative_angle:
            self.error_theta = (
                self.theta_result - self.theta_result[:, slack_bus : slack_bus + 1]
            ) - (
                self.theta_result_exact
                - self.theta_result_exact[:, slack_bus : slack_bus + 1]
            )
        else:
            self.error_theta = self.theta_result - self.theta_result_exact

        error_states = list()
        for ii, (component, bus) in enumerate(
            zip(self.grid_model.components, self.grid_model.component_bus_indices)
        ):
            if component.component_type == "Generator" and relative_angle:
                state_results_exact = (
                    self.state_result_exact[ii]
                    - self.theta_result_exact[:, bus : bus + 1]
                    @ component.state_to_delta.T
                )
                state_results = (
                    self.state_result[ii]
                    - self.theta_result[:, bus : bus + 1] @ component.state_to_delta.T
                )
                error_states.append(state_results - state_results_exact)
            else:
                error_states.append(self.state_result[ii] - self.state_result_exact[ii])
        self.error_states = error_states

        self.error_I_D = self.network_I_D - self.components_I_D
        self.error_I_Q = self.network_I_Q - self.components_I_Q
        pass

    def integral_current_error(self):
        integral_error_I_D = torch.square(self.error_I_D).sum() * self.timestep_size
        integral_error_I_Q = torch.square(self.error_I_Q).sum() * self.timestep_size
        return integral_error_I_D, integral_error_I_Q

    def return_max_absolute_error(self):
        max_angle_error_list = list()
        max_omega_error_list = list()
        for ii, component in enumerate(self.grid_model.components):
            if component.component_type == "Generator":
                max_angle_error_list.append(self.error_states[ii][:, 2:3].abs())
                max_omega_error_list.append(self.error_states[ii][:, 3:4].abs())
        max_angle_error = torch.hstack(max_angle_error_list).max()
        max_omega_error = torch.hstack(max_omega_error_list).max()
        max_theta_error = self.error_theta.abs().max()
        max_V_error = self.error_V.abs().max()
        max_I_D_error = self.error_I_D.abs().max()
        max_I_Q_error = self.error_I_Q.abs().max()
        return (
            max_angle_error,
            max_omega_error,
            max_theta_error,
            max_V_error,
            max_I_D_error,
            max_I_Q_error,
        )

    def return_mean_absolute_error(self):
        mean_angle_error_list = list()
        mean_omega_error_list = list()
        for ii, component in enumerate(self.grid_model.components):
            if component.component_type == "Generator":
                mean_angle_error_list.append(self.error_states[ii][:, 2:3].abs())
                mean_omega_error_list.append(self.error_states[ii][:, 3:4].abs())

        mean_angle_error = torch.hstack(mean_angle_error_list).mean()
        mean_omega_error = torch.hstack(mean_omega_error_list).mean()
        mean_theta_error = self.error_theta.abs().mean()
        mean_V_error = self.error_V.abs().mean()
        mean_I_D_error = self.error_I_D.abs().mean()
        mean_I_Q_error = self.error_I_Q.abs().mean()
        return (
            mean_angle_error,
            mean_omega_error,
            mean_theta_error,
            mean_V_error,
            mean_I_D_error,
            mean_I_Q_error,
        )

    def return_max_final_error(self):
        max_angle_error = max(
            [state_error[-1, 2].abs() for state_error in self.error_states[:3]]
        )
        max_omega_error = max(
            [state_error[-1, 3].abs() for state_error in self.error_states[:3]]
        )
        max_theta_error = self.error_theta[-1, :].abs().max()
        max_V_error = self.error_V[-1, :].abs().max()
        return max_angle_error, max_omega_error, max_theta_error, max_V_error

    def print_integral_error(self):
        integral_error_I_D, integral_error_I_Q = self.integral_current_error()
        print(
            f"Error integral I_D, I_Q:                       {integral_error_I_D:.2e} | {integral_error_I_Q:.2e}"
        )
        pass

    def print_max_final_error(self):
        max_angle_error, max_omega_error, _, _ = self.return_max_final_error()
        print(
            f"Max final state error angle [deg], omega [Hz]: {max_angle_error*180/torch.pi:.2e} | {max_omega_error*definitions.f_0_Hz:.2e}"
        )
        pass

    def print_max_voltage_error(self):
        _, _, max_theta_error, max_V_error, _, _ = self.return_max_absolute_error()
        print(
            f"Max voltage error theta [deg], V [p.u.]:       {max_theta_error*180/torch.pi:.2e} | {max_V_error:.2e}"
        )
        pass

    def return_error_dict(self):
        integral_error_I_D, integral_error_I_Q = self.integral_current_error()
        (
            max_angle_error,
            max_omega_error,
            max_theta_error,
            max_V_error,
            max_I_D_error,
            max_I_Q_error,
        ) = self.return_max_absolute_error()
        (
            max_final_angle_error,
            max_final_omega_error,
            max_final_theta_error,
            max_final_V_error,
        ) = self.return_max_final_error()

        error_dict = dict(
            {
                "error_angle_max": max_angle_error.item(),
                "error_omega_max": max_omega_error.item(),
                "error_theta_max": max_theta_error.item(),
                "error_V_max": max_V_error.item(),
                "error_ID_max": max_I_D_error.item(),
                "error_IQ_max": max_I_Q_error.item(),
                "error_angle_final": max_final_angle_error.item(),
                "error_omega_final": max_final_omega_error.item(),
                "error_theta_final": max_final_theta_error.item(),
                "error_V_final": max_final_V_error.item(),
                "error_ID_integral": integral_error_I_D.item(),
                "error_IQ_integral": integral_error_I_Q.item(),
            }
        )

        return error_dict

    def return_dataset_error_dict(self):
        (
            max_angle_error,
            max_omega_error,
            max_theta_error,
            max_V_error,
            max_I_D_error,
            max_I_Q_error,
        ) = self.return_max_absolute_error()

        (
            mean_angle_error,
            mean_omega_error,
            mean_theta_error,
            mean_V_error,
            mean_I_D_error,
            mean_I_Q_error,
        ) = self.return_mean_absolute_error()

        error_dict = dict(
            {
                "error_angle_max": max_angle_error.item(),
                "error_omega_max": max_omega_error.item(),
                "error_theta_max": max_theta_error.item(),
                "error_V_max": max_V_error.item(),
                "error_ID_max": max_I_D_error.item(),
                "error_IQ_max": max_I_Q_error.item(),
                "error_angle_mean": mean_angle_error.item(),
                "error_omega_mean": mean_omega_error.item(),
                "error_theta_mean": mean_theta_error.item(),
                "error_V_mean": mean_V_error.item(),
                "error_ID_mean": mean_I_D_error.item(),
                "error_IQ_mean": mean_I_Q_error.item(),
            }
        )

        return error_dict

    def compute_value_range(self):
        variable_names = [
            "theta_diff_result_exact",
            "network_I_D",
            "network_I_Q",
            "V_result_exact",
        ]
        new_names = ["theta", "ID", "IQ", "V"]
        range_dict = dict({})
        for variable_name, new_name in zip(variable_names, new_names):
            values = self.__getattribute__(variable_name)
            value_range = (values.amax(dim=0) - values.amin(dim=0)).max()
            range_dict[f"{new_name}"] = value_range.item()

        generator_state = [
            component_type == "Generator"
            for component_type in self.grid_model.component_types
        ]

        generator_states = list()
        for ii, use_value in enumerate(generator_state):
            if use_value:
                generator_states.append(
                    self.state_diff_result_exact[ii].amax(dim=0)
                    - self.state_diff_result_exact[ii].amin(dim=0)
                )
                # generator_state_names = self.grid_model.components[ii].state_names
        generator_states_max = torch.vstack(generator_states).amax(dim=0)
        for generator_state_name, value in zip(
            ["Eqprime", "Edprime", "angle", "omega"], generator_states_max
        ):
            range_dict[f"{generator_state_name}"] = value.item()

        return range_dict
