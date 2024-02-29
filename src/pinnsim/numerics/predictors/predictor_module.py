import abc

import torch


class PredictorModule(torch.nn.Module):
    def __init__(self, component, voltage_profile):
        super(PredictorModule, self).__init__()

        self.component = component
        self.voltage_profile = voltage_profile

    def predict_current(self, time, state, control_input, voltage_parametrisation):
        theta, V = self.voltage_profile.get_voltage(
            time=time, voltage_parametrisation=voltage_parametrisation
        )
        state_predicted = self.predict_state(
            time=time,
            state=state,
            control_input=control_input,
            voltage_parametrisation=voltage_parametrisation,
        )
        I_D, I_Q = self.component.compute_current_D_Q(
            state=state_predicted, theta=theta, V=V, control_input=control_input
        )
        return I_D, I_Q

    @abc.abstractmethod
    def predict_state(self, time, state, control_input, voltage_parametrisation):
        return torch.zeros(state.shape)

    def predict_current_jacobian(
        self, time, state, control_input, voltage_parametrisation
    ):
        with torch.no_grad():
            current_injections = self.predict_current(
                time=time,
                state=state,
                control_input=control_input,
                voltage_parametrisation=voltage_parametrisation,
            )
            current_jacobian = torch.autograd.functional.jacobian(
                func=lambda x: self.predict_current(
                    time=time,
                    state=state,
                    control_input=control_input,
                    voltage_parametrisation=x,
                ),
                inputs=voltage_parametrisation,
                vectorize=True,
            )

        return (
            current_injections[0],
            current_injections[1],
            current_jacobian[0][:, 0, 0, :],
            current_jacobian[1][:, 0, 0, :],
        )
