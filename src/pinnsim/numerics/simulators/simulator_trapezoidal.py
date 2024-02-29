import torch

from .simulator_iterative import IterativeSimulator


class SimulatorTrapezoidal(IterativeSimulator):
    def __repr__(self):
        return "Trapezoidal"

    def simulator_config(self):
        simulator_config = "Trapezoidal, r = 1, s = 2"
        return simulator_config

    def simulate_timestep(self, component_state_list, control_input_list, theta, V):
        state_n_0 = torch.hstack(
            [
                torch.hstack(component_state_list).detach(),
                theta.detach().clone(),
                V.detach().clone(),
            ]
        )
        state_list_n_0 = component_state_list
        theta_n_0 = theta.detach().clone()
        V_n_0 = V.detach().clone()
        state_n_1_iterate = state_n_0.clone()[0, :]
        state_n_1_new = state_n_1_iterate.detach().clone()
        k_iteration = 0
        residual_norm = 1.0
        max_variable_change = 1.0
        while (
            max_variable_change > self.newton_scheme_tolerance
            and k_iteration < self.newton_scheme_max_iterations
        ):
            residual, jacobian = self.compute_residual_and_jacobian(
                time=torch.tensor([[self.timestep_size]]),
                state_n_1=state_n_1_iterate,
                state_list_n_0=state_list_n_0,
                control_input_list=control_input_list,
                theta_n_0=theta_n_0,
                V_n_0=V_n_0,
            )
            state_update = -torch.linalg.inv(jacobian) @ residual
            state_n_1_new = state_n_1_iterate + state_update
            state_n_1_iterate = state_n_1_new.clone()
            residual_norm = torch.linalg.norm(residual).item()
            max_variable_change = torch.max(torch.abs(state_update)).item()
            k_iteration += 1
            if self.verbosity <= 20:
                self.print_iteration_information(
                    k_iteration, residual_norm, max_variable_change
                )

        timestep_summary = dict(
            {
                "residual_norm": residual_norm,
                "max_variable_change": max_variable_change,
                "n_iterations": k_iteration,
            }
        )

        interpolation = self.interpolate_states(
            state_0=state_n_0,
            state_1=state_n_1_new.detach().reshape((1, -1)),
            control_input_list=control_input_list,
        )

        result_dict = dict(
            {
                "time": self.output_times,
                "component_states_list": interpolation[0],
                "theta": interpolation[1],
                "V": interpolation[2],
                "timestep_summary": timestep_summary,
            }
        )
        return result_dict

    def interpolate_states(self, state_0, state_1, control_input_list):
        state = state_0 + (state_1 - state_0) * torch.round(
            self.output_times / self.timestep_size, decimals=6
        )

        state_list_n_1, theta_n_1, V_n_1 = self.grid_model.split_state(
            state=state_1.reshape((1, -1))
        )
        state_list_n_0, theta_n_0, V_n_0 = self.grid_model.split_state(
            state=state_0.reshape((1, -1))
        )
        theta_list_n_0, V_list_n_0 = self.grid_model.compute_voltage_lists(
            theta=theta_n_0, V=V_n_0
        )
        f_state_list_n_0 = self.grid_model.compute_components_update_function(
            time=0.0 * torch.tensor([[1.0]]),
            state_list=state_list_n_0,
            control_input_list=control_input_list,
            theta_list=theta_list_n_0,
            V_list=V_list_n_0,
        )
        states_interpolated = [
            x_0
            + f_x_0 * self.output_times
            + (x_1 - x_0 - f_x_0 * self.timestep_size)
            * ((self.output_times / self.timestep_size) ** 2)
            for x_1, x_0, f_x_0 in zip(state_list_n_1, state_list_n_0, f_state_list_n_0)
        ]

        _, theta_interpolated, V_interpolated = self.grid_model.split_state(state)

        return states_interpolated, theta_interpolated, V_interpolated
        # return self.grid_model.split_state(state)

    def compute_residual_and_jacobian(
        self,
        time,
        state_n_1,
        state_list_n_0,
        control_input_list,
        theta_n_0,
        V_n_0,
    ):
        with torch.no_grad():
            residual = self.compute_residual(
                time=time,
                state_n_1=state_n_1,
                state_list_n_0=state_list_n_0,
                control_input_list=control_input_list,
                theta_n_0=theta_n_0,
                V_n_0=V_n_0,
            )
            jacobian = torch.autograd.functional.jacobian(
                func=lambda x: self.compute_residual(
                    time=time,
                    state_n_1=x,
                    state_list_n_0=state_list_n_0,
                    control_input_list=control_input_list,
                    theta_n_0=theta_n_0,
                    V_n_0=V_n_0,
                ),
                inputs=state_n_1,
                vectorize=True,
            )

        return residual, jacobian

    def compute_residual(
        self,
        time,
        state_n_1,
        state_list_n_0,
        control_input_list,
        theta_n_0,
        V_n_0,
    ):
        state_list_n_1, theta_n_1, V_n_1 = self.grid_model.split_state(
            state=state_n_1.reshape((1, -1))
        )
        theta_list_n_1, V_list_n_1 = self.grid_model.compute_voltage_lists(
            theta=theta_n_1, V=V_n_1
        )
        theta_n_0_list, V_n_0_list = self.grid_model.compute_voltage_lists(
            theta=theta_n_0, V=V_n_0
        )

        state_residual = self.state_balance(
            time=time,
            state_list_n_0=state_list_n_0,
            theta_n_0_list=theta_n_0_list,
            V_n_0_list=V_n_0_list,
            state_list_n_1=state_list_n_1,
            theta_list_n_1=theta_list_n_1,
            V_list_n_1=V_list_n_1,
            control_input_list=control_input_list,
        )
        power_balance_residual = self.power_balance(
            time=time,
            state_list_n_1=state_list_n_1,
            control_input_list=control_input_list,
            theta_n_1=theta_n_1,
            V_n_1=V_n_1,
        )

        return torch.hstack([state_residual, power_balance_residual]).flatten()

    def power_balance(self, time, state_list_n_1, control_input_list, theta_n_1, V_n_1):
        theta_list_n_1, V_list_n_1 = self.grid_model.compute_voltage_lists(
            theta=theta_n_1, V=V_n_1
        )
        (
            I_D_components,
            I_Q_components,
        ) = self.grid_model.compute_component_current_injections(
            time=time,
            state_list=state_list_n_1,
            control_input_list=control_input_list,
            theta_list=theta_list_n_1,
            V_list=V_list_n_1,
        )
        I_D_network, I_Q_network = self.grid_model.network.compute_current_injections(
            torch.hstack([theta_n_1, V_n_1])
        )
        return torch.hstack(
            [I_D_network - I_D_components, I_Q_network - I_Q_components]
        )

    def state_balance(
        self,
        time,
        state_list_n_0,
        theta_n_0_list,
        V_n_0_list,
        state_list_n_1,
        theta_list_n_1,
        V_list_n_1,
        control_input_list,
    ):
        f_state_list_n_1 = self.grid_model.compute_components_update_function(
            time=time,
            state_list=state_list_n_1,
            control_input_list=control_input_list,
            theta_list=theta_list_n_1,
            V_list=V_list_n_1,
        )

        f_state_list_n_0 = self.grid_model.compute_components_update_function(
            time=0.0 * time,
            state_list=state_list_n_0,
            control_input_list=control_input_list,
            theta_list=theta_n_0_list,
            V_list=V_n_0_list,
        )

        state_residual = (
            torch.hstack(state_list_n_1)
            - torch.hstack(state_list_n_0)
            - time
            / 2
            * (torch.hstack(f_state_list_n_1) + torch.hstack(f_state_list_n_0))
        )

        return state_residual

    def solver_dict_extra(self):
        extra_entries = dict(
            {
                "predictor_scheme": "Trapezoidal",
                "r": 1,
                "s": 2,
                "t0": True,
            }
        )
        return extra_entries
