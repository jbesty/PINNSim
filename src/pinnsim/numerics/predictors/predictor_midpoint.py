from .predictor_module import PredictorModule


class PredictorMidpoint(PredictorModule):
    def __init__(self, component, voltage_profile):
        super(PredictorMidpoint, self).__init__(
            component=component, voltage_profile=voltage_profile
        )

    def __repr__(self):
        return "Midpoint"

    def predict_state(self, time, state, control_input, voltage_parametrisation):
        theta_k1, V_k1 = self.voltage_profile.get_voltage(
            time=time * 0.0, voltage_parametrisation=voltage_parametrisation
        )
        k1 = self.component.update_function(
            time=time * 0.0,
            # state=state.repeat(time.shape),
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

        state_prediction = state + k2 * time

        return state_prediction
