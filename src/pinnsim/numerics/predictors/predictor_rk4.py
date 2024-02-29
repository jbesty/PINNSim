from .predictor_module import PredictorModule


class PredictorRK4(PredictorModule):
    def __init__(self, component, voltage_profile):
        super(PredictorRK4, self).__init__(
            component=component, voltage_profile=voltage_profile
        )

    def __repr__(self):
        return "RK4"

    def predict_state(self, time, state, control_input, voltage_parametrisation):
        theta_k1, V_k1 = self.voltage_profile.get_voltage(
            time=time * 0.0, voltage_parametrisation=voltage_parametrisation
        )
        if not time.shape[0] == state.shape[0]:
            state = state.repeat(time.shape)
        k1 = self.component.update_function(
            time=time * 0.0,
            state=state,
            control_input=control_input,
            theta=theta_k1,
            V=V_k1,
        )
        theta_k2, V_k2 = self.voltage_profile.get_voltage(
            time=time * 0.5, voltage_parametrisation=voltage_parametrisation
        )
        k2 = self.component.update_function(
            time=time * 0.5,
            state=state + k1 * time / 2,
            control_input=control_input,
            theta=theta_k2,
            V=V_k2,
        )
        theta_k3, V_k3 = self.voltage_profile.get_voltage(
            time=time * 0.5, voltage_parametrisation=voltage_parametrisation
        )
        k3 = self.component.update_function(
            time=time * 0.5,
            state=state + k2 * time / 2,
            control_input=control_input,
            theta=theta_k3,
            V=V_k3,
        )
        theta_k4, V_k4 = self.voltage_profile.get_voltage(
            time=time * 1.0, voltage_parametrisation=voltage_parametrisation
        )
        k4 = self.component.update_function(
            time=time * 1.0,
            state=state + k3 * time,
            control_input=control_input,
            theta=theta_k4,
            V=V_k4,
        )

        state_prediction = state + (k1 + 2 * k2 + 2 * k3 + k4) * time / 6
        return state_prediction
