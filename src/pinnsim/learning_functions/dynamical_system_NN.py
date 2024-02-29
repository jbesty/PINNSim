import abc

import torch.nn

from .core_neural_network import CoreNeuralNetwork


class DynamicalSystemNN(torch.nn.Module):
    def __init__(
        self,
        neurons_in_layers: list,
        component,
        voltage_profile,
        use_states: list,
        use_control_inputs: list,
        use_voltage_parametrisation: list,
        pytorch_init_seed: int = None,
    ):
        super(DynamicalSystemNN, self).__init__()

        self.NN = CoreNeuralNetwork(
            neurons_in_layers=neurons_in_layers, pytorch_init_seed=pytorch_init_seed
        )
        self.neurons_in_layers = neurons_in_layers

        self.physics_regulariser = torch.tensor(0.0)

        self.epochs_total = 0

        self.component = component
        self.voltage_profile = voltage_profile

        assert len(use_states) == self.component.n_states
        assert len(use_control_inputs) == self.component.n_control_inputs
        assert (
            len(use_voltage_parametrisation)
            == self.voltage_profile.n_voltage_parameters
        )

        self.mapping_state_input = torch.eye(self.component.n_states)[:, use_states]
        self.mapping_control_input = torch.eye(self.component.n_control_inputs)[
            :, use_control_inputs
        ]
        self.mapping_voltage_parametrisation = torch.eye(
            self.voltage_profile.n_voltage_parameters
        )[:, use_voltage_parametrisation]

        self.state_mean = torch.nn.Parameter(
            torch.zeros((1, self.component.n_states)), requires_grad=False
        )
        self.state_std = torch.nn.Parameter(
            torch.ones((1, self.component.n_states)), requires_grad=False
        )

    def __repr__(self):
        return "NN"

    def transform_state_to_norm(self, state):
        return (state - self.state_mean) / self.state_std

    def transform_state_to_scale(self, state):
        return state * self.state_std + self.state_mean

    def transform_state_to_scale_only(self, state):
        return state * self.state_std

    def transform_state_to_norm_only(self, state):
        return state / self.state_std

    @abc.abstractmethod
    def assemble_NN_input(
        self,
        time,
        state_initial=None,
        d_dt_state_initial=None,
        control_input=None,
        voltage_parametrisation=None,
    ):
        return torch.hstack(
            [
                time,
                state_initial,
                d_dt_state_initial,
                control_input,
                voltage_parametrisation,
            ]
        )

    @abc.abstractmethod
    def assemble_NN_target(
        self,
        time=None,
        state_initial=None,
        d_dt_state_initial=None,
        control_input=None,
        voltage_parametrisation=None,
        state_result=None,
    ):
        return torch.hstack([state_result])

    @abc.abstractmethod
    def assemble_prediction(
        self,
        NN_output,
        time=None,
        state_initial=None,
        d_dt_state_initial=None,
        control_input=None,
        voltage_parametrisation=None,
    ):
        return torch.hstack([NN_output])

    def forward(
        self,
        time,
        state_initial=None,
        d_dt_state_initial=None,
        control_input=None,
        voltage_parametrisation=None,
    ):
        NN_input = self.assemble_NN_input(
            time=time,
            state_initial=state_initial,
            d_dt_state_initial=d_dt_state_initial,
            control_input=control_input,
            voltage_parametrisation=voltage_parametrisation,
        )
        NN_output = self.NN.forward(NN_input)
        prediction = self.assemble_prediction(
            NN_output=NN_output,
            time=time,
            state_initial=state_initial,
            d_dt_state_initial=d_dt_state_initial,
            control_input=control_input,
            voltage_parametrisation=voltage_parametrisation,
        )
        return prediction

    def forward_dt(
        self,
        time,
        state_initial=None,
        d_dt_state_initial=None,
        control_input=None,
        voltage_parametrisation=None,
    ):
        prediction, prediction_dt = torch.autograd.functional.jvp(
            func=lambda x: self.forward(
                time=x,
                state_initial=state_initial,
                d_dt_state_initial=d_dt_state_initial,
                control_input=control_input,
                voltage_parametrisation=voltage_parametrisation,
            ),
            inputs=time,
            v=torch.ones(time.shape),
            create_graph=True,
        )
        return prediction, prediction_dt

    def predict(
        self,
        time,
        state_initial=None,
        d_dt_state_initial=None,
        control_input=None,
        voltage_parametrisation=None,
    ):
        self.eval()
        with torch.no_grad():
            prediction = self.forward(
                time=time,
                state_initial=state_initial,
                d_dt_state_initial=d_dt_state_initial,
                control_input=control_input,
                voltage_parametrisation=voltage_parametrisation,
            )
        return prediction

    def predict_dt(
        self,
        time,
        state_initial=None,
        d_dt_state_initial=None,
        control_input=None,
        voltage_parametrisation=None,
    ):
        self.eval()

        with torch.no_grad():
            prediction, prediction_dt = torch.autograd.functional.jvp(
                func=lambda x: self.forward(
                    time=x,
                    state_initial=state_initial,
                    d_dt_state_initial=d_dt_state_initial,
                    control_input=control_input,
                    voltage_parametrisation=voltage_parametrisation,
                ),
                inputs=time,
                v=torch.ones(time.shape),
                create_graph=False,
            )
        return prediction, prediction_dt

    def predict_state(
        self,
        time,
        state=None,
        d_dt_state_initial=None,
        control_input=None,
        voltage_parametrisation=None,
    ):
        return self.predict(
            time=time,
            state_initial=state,
            d_dt_state_initial=d_dt_state_initial,
            control_input=control_input,
            voltage_parametrisation=voltage_parametrisation,
        )

    def predict_current(self, time, state, control_input, voltage_parametrisation):
        theta, V = self.voltage_profile.get_voltage(
            time=time, voltage_parametrisation=voltage_parametrisation
        )
        state_predicted = self.predict_state(
            time=time,
            state=state.repeat(time.shape),
            control_input=control_input.repeat(time.shape),
            voltage_parametrisation=voltage_parametrisation.repeat(time.shape),
        )
        I_D, I_Q = self.component.compute_current_D_Q(
            state=state_predicted, theta=theta, V=V, control_input=control_input
        )
        return I_D, I_Q

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

    def forward_lhs_rhs(
        self,
        time,
        state_initial=None,
        d_dt_state_initial=None,
        control_input=None,
        voltage_parametrisation=None,
    ):
        prediction, prediction_dt = self.forward_dt(
            time=time,
            state_initial=state_initial,
            control_input=control_input,
            voltage_parametrisation=voltage_parametrisation,
        )
        theta, V = self.voltage_profile.get_voltage(
            time=time,
            voltage_parametrisation=voltage_parametrisation,
        )
        prediction_rhs = self.component.update_function(
            time=time,
            state=prediction,
            control_input=control_input,
            theta=theta,
            V=V,
        )
        return prediction, prediction_dt, prediction_rhs

    def update_state_scaling(self, dataset):
        assert torch.unique(dataset.state_equilibrium, dim=0).shape == (
            1,
            self.component.n_states,
        )

        with torch.no_grad():
            self.state_mean = torch.nn.Parameter(
                dataset.state_equilibrium[0:1, :], requires_grad=False
            )
            for ii, value in enumerate(self.component.norm_to_scale[0, :]):
                if value > torch.zeros(1):
                    self.state_std[0, ii] = torch.nn.Parameter(
                        value, requires_grad=False
                    )

        pass

    def update_input_standardisation(self, dataset):
        NN_input_data = self.assemble_NN_input(
            time=dataset.time,
            state_initial=dataset.state_initial,
            d_dt_state_initial=dataset.d_dt_state_initial,
            control_input=dataset.control_input,
            voltage_parametrisation=dataset.voltage_parametrisation,
        )

        input_std, input_mean = torch.std_mean(NN_input_data, dim=0)

        self.NN.dense_layers.input_standardisation.set_standardisation(
            mean=input_mean, standard_deviation=input_std
        )

        pass

    def update_output_standardisation(self, dataset):
        NN_target_data = self.assemble_NN_target(
            time=dataset.time,
            state_initial=dataset.state_initial,
            d_dt_state_initial=dataset.d_dt_state_initial,
            control_input=dataset.control_input,
            voltage_parametrisation=dataset.voltage_parametrisation,
            state_result=dataset.state_result,
        )

        output_std, output_mean = torch.std_mean(NN_target_data, dim=0)
        self.NN.dense_layers.output_scaling.set_scaling(
            mean=output_mean, standard_deviation=output_std
        )

        pass

    def adjust_to_dataset(self, dataset):
        self.update_state_scaling(dataset=dataset)
        self.update_input_standardisation(dataset=dataset)
        self.update_output_standardisation(dataset=dataset)
        pass


class DynamicalSystemResidualNN(DynamicalSystemNN):
    def __init__(
        self,
        hidden_layer_size: int,
        n_hidden_layers: int,
        component,
        voltage_profile,
        pytorch_init_seed: int = None,
        use_states: list = None,
        use_control_inputs: list = None,
        use_voltage_parametrisation: list = None,
    ):

        n_input_neurons = (
            1
            + sum(use_states)
            + sum(use_control_inputs)
            + sum(use_voltage_parametrisation)
        )

        neurons_in_layers = (
            [n_input_neurons]
            + [hidden_layer_size] * n_hidden_layers
            + [len(use_states)]
        )

        super(DynamicalSystemResidualNN, self).__init__(
            neurons_in_layers=neurons_in_layers,
            component=component,
            voltage_profile=voltage_profile,
            pytorch_init_seed=pytorch_init_seed,
            use_states=use_states,
            use_control_inputs=use_control_inputs,
            use_voltage_parametrisation=use_voltage_parametrisation,
        )

    def __repr__(self):
        return "NN Residual"

    def assemble_NN_input(
        self,
        time,
        state_initial=None,
        d_dt_state_initial=None,
        control_input=None,
        voltage_parametrisation=None,
    ):
        state_NN = (
            self.transform_state_to_norm(state_initial) @ self.mapping_state_input
        )
        control_input_NN = control_input @ self.mapping_control_input
        voltage_parametrisation_NN = (
            voltage_parametrisation @ self.mapping_voltage_parametrisation
        )
        return torch.hstack(
            [
                time,
                state_NN,
                control_input_NN,
                voltage_parametrisation_NN,
            ]
        )

    def assemble_NN_target(
        self,
        time=None,
        state_initial=None,
        d_dt_state_initial=None,
        control_input=None,
        voltage_parametrisation=None,
        state_result=None,
    ):
        theta_difference = self.voltage_profile.get_theta_change(
            time=time, voltage_parametrisation=voltage_parametrisation
        )

        return (
            self.transform_state_to_norm_only(
                state_result
                - state_initial
                - theta_difference @ torch.tensor([[0.0, 0.0, 1.0, 0.0]])
            )
            / time
        )

    def assemble_prediction(
        self,
        NN_output,
        time=None,
        state_initial=None,
        d_dt_state_initial=None,
        control_input=None,
        voltage_parametrisation=None,
    ):
        theta_difference = self.voltage_profile.get_theta_change(
            time=time, voltage_parametrisation=voltage_parametrisation
        )
        return (
            state_initial
            + theta_difference @ torch.tensor([[0.0, 0.0, 1.0, 0.0]])
            + time * self.transform_state_to_scale_only(NN_output)
        )
