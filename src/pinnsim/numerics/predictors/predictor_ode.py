from .predictor_module import PredictorModule

import torch
import torchdiffeq


class PredictorODE(PredictorModule):
    def __init__(self, component, voltage_profile):
        super(PredictorODE, self).__init__(component, voltage_profile)

        self.component = component
        self.voltage_profile = voltage_profile
        self.control_input = torch.zeros((1, self.component.n_control_inputs))
        self.ode_state_split = [
            self.component.n_states,
            self.voltage_profile.n_voltage_parameters,
        ]

    def forward(self, t, y):
        state, voltage_parametrisation = y.split(split_size=self.ode_state_split)
        theta, V = self.voltage_profile.get_voltage(
            time=t.reshape((1, 1)),
            voltage_parametrisation=voltage_parametrisation.reshape((1, -1)),
        )
        state_result = self.component.update_function(
            time=t.reshape((1, 1)),
            state=state.reshape((1, -1)),
            control_input=self.control_input,
            theta=theta.reshape((1, 1)),
            V=V.reshape((1, 1)),
        )
        ode_state = torch.hstack(
            [state_result, torch.zeros(1, self.voltage_profile.n_voltage_parameters)]
        )
        return ode_state.flatten()

    def __repr__(self):
        return "ODE"

    def predict_state(self, time, state, control_input, voltage_parametrisation):
        self.control_input = control_input
        ode_output = torchdiffeq.odeint_adjoint(
            func=self.forward,
            t=time[:, 0],
            y0=torch.hstack([state, voltage_parametrisation])[0, :],
            rtol=1e-10,
            atol=1e-10,
            adjoint_params=(),
            adjoint_atol=1e-10,
            adjoint_rtol=1e-10,
        )
        state_predicted = ode_output[:, : self.component.n_states]
        return state_predicted

    def predict_current_jacobian(
        self, time, state, control_input, voltage_parametrisation
    ):
        voltage_parametrisation.requires_grad_()
        current_injections = self.predict_current(
            time=time,
            state=state,
            control_input=control_input,
            voltage_parametrisation=voltage_parametrisation,
        )

        current_jacobian = [
            torch.vstack(
                [
                    torch.autograd.grad(
                        current_injection[ii, 0],
                        voltage_parametrisation,
                        retain_graph=True,
                    )[0]
                    for ii in range(time.shape[0])
                ]
            )
            for current_injection in current_injections
        ]

        return (
            current_injections[0].detach(),
            current_injections[1].detach(),
            current_jacobian[0].detach(),
            current_jacobian[1].detach(),
        )
