import torch


def test_assembly_dimensions():
    # Write a test that ensures that the "assembly" of the input, target and prediction functions in the NN architecture are correct

    # time = torch.ones((1, 1))
    # state_initial = torch.ones((1, self.component.n_states))
    # d_dt_state_initial = torch.ones((1, self.component.n_states))
    # control_input = torch.ones((1, self.component.n_control_inputs))
    # voltage_parametrisation = torch.ones((1, self.voltage_profile.n_voltage_parameters))
    # state_result = torch.ones((1, self.component.n_states))
    # NN_output = torch.ones((1, self.neurons_in_layers[-1]))
    # assert (
    #     self.assemble_NN_input(
    #         time=time,
    #         state_initial=state_initial,
    #         d_dt_state_initial=d_dt_state_initial,
    #         control_input=control_input,
    #         voltage_parametrisation=voltage_parametrisation,
    #     ).shape[1]
    #     == self.neurons_in_layers[0]
    # )
    # assert (
    #     self.assemble_NN_target(
    #         time=time,
    #         state_initial=state_initial,
    #         d_dt_state_initial=d_dt_state_initial,
    #         control_input=control_input,
    #         voltage_parametrisation=voltage_parametrisation,
    #         state_result=state_result,
    #     ).shape[1]
    #     == self.neurons_in_layers[-1]
    # )
    # assert (
    #     self.assemble_prediction(
    #         time=time,
    #         state_initial=state_initial,
    #         d_dt_state_initial=d_dt_state_initial,
    #         control_input=control_input,
    #         voltage_parametrisation=voltage_parametrisation,
    #         NN_output=NN_output,
    #     ).shape[1]
    #     == self.n_states
    # )
    pass
