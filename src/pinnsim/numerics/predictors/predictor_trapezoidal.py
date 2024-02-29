from .predictor_module import PredictorModule

import torch


class PredictorTrapezoidal(PredictorModule):
    def __init__(self, component, voltage_profile):
        super(PredictorTrapezoidal, self).__init__(
            component=component, voltage_profile=voltage_profile
        )
        self.newton_scheme_tolerance = 1.0e-10
        self.newton_scheme_max_iterations = 10

    def __repr__(self):
        return "Trap"

    def compute_residual_and_jacobian(
        self, time, state_0, state_1, control_input, theta_0, V_0, theta_1, V_1
    ):
        with torch.no_grad():
            residual = self.compute_residual(
                time=time,
                state_0=state_0,
                state_1=state_1,
                control_input=control_input,
                theta_0=theta_0,
                V_0=V_0,
                theta_1=theta_1,
                V_1=V_1,
            )
            jacobian = torch.autograd.functional.jacobian(
                func=lambda x: self.compute_residual(
                    time=time,
                    state_0=state_0,
                    state_1=x,
                    control_input=control_input,
                    theta_0=theta_0,
                    V_0=V_0,
                    theta_1=theta_1,
                    V_1=V_1,
                ),
                inputs=state_1,
                vectorize=True,
            )

        return residual, jacobian

    def compute_residual(
        self, time, state_0, state_1, control_input, theta_0, V_0, theta_1, V_1
    ):
        f_x_1 = self.component.update_function(
            time=time, state=state_1, control_input=control_input, theta=theta_1, V=V_1
        )
        f_x_0 = self.component.update_function(
            time=time * 0.0,
            state=state_0,
            control_input=control_input,
            theta=theta_0,
            V=V_0,
        )
        residual = state_1 - state_0 - time / 2 * (f_x_0 + f_x_1)
        return residual

    def predict_state(self, time, state, control_input, voltage_parametrisation):
        if not time.shape[0] == state.shape[0]:
            state = state.repeat(time.shape)

        k_iteration = 0
        max_variable_change = 1.0
        theta_1, V_1 = self.voltage_profile.get_voltage(
            time=time, voltage_parametrisation=voltage_parametrisation
        )
        theta_0, V_0 = self.voltage_profile.get_voltage(
            time=time * 0.0, voltage_parametrisation=voltage_parametrisation
        )
        state_0 = state.detach().clone()
        f_x_0 = self.component.update_function(
            time=time * 0.0,
            state=state_0,
            control_input=control_input,
            theta=theta_0,
            V=V_0,
        )
        state_1_iterate = state.detach().clone() + time * f_x_0
        while (
            max_variable_change > self.newton_scheme_tolerance
            and k_iteration < self.newton_scheme_max_iterations
        ):
            state_list = list()
            state_update_list = list()
            for (
                time_,
                state_0_,
                state_1_iterate_,
                control_input_,
                theta_0_,
                V_0_,
                theta_1_,
                V_1_,
            ) in zip(
                time,
                state_0,
                state_1_iterate,
                control_input,
                theta_0,
                V_0,
                theta_1,
                V_1,
            ):
                residual, jacobian = self.compute_residual_and_jacobian(
                    time=time_.reshape((1, 1)),
                    state_0=state_0_.reshape((1, -1)),
                    state_1=state_1_iterate_.reshape((1, -1)),
                    control_input=control_input_.reshape((1, -1)),
                    theta_0=theta_0_.reshape((1, 1)),
                    V_0=V_0_.reshape((1, 1)),
                    theta_1=theta_1_.reshape((1, 1)),
                    V_1=V_1_.reshape((1, 1)),
                )
                state_update = (-torch.linalg.inv(jacobian[0, :, 0, :]) @ residual.T).T
                state_n_1_new = state_1_iterate_ + state_update
                state_update_list.append(state_update)
                state_list.append(state_n_1_new)
            state_1_iterate = torch.vstack(state_list)
            max_variable_change = torch.max(
                torch.abs(torch.vstack(state_update_list))
            ).item()
            k_iteration += 1

        return state_1_iterate
