from .predictor_module import PredictorModule


class PredictorStatic(PredictorModule):
    def __init__(self, component, voltage_profile):
        super(PredictorStatic, self).__init__(
            component=component, voltage_profile=voltage_profile
        )

    def __repr__(self):
        return "Static"

    def predict_state(self, time, state, control_input, voltage_parametrisation):
        return state.repeat(time.shape)
