import torch


class NetworkModel:
    def __init__(self, network_parameters):
        self.state_map = None
        self.update_indices = None
        self.n_buses = network_parameters["n_buses"]
        self.n_states = 2 * self.n_buses
        self.Y = network_parameters["Y_bus_pu"].clone()

        # self.V_set_points = (
        #     network_parameters["V_set_points"].clone().reshape((1, self.n_buses))
        # )
        # self.P_set_points = (
        #     network_parameters["P_set_points"].clone().reshape((1, self.n_buses))
        # )
        # self.Q_set_points = (
        #     network_parameters["Q_set_points"].clone().reshape((1, self.n_buses))
        # )
        # self.theta_set_points = (
        #     network_parameters["theta_set_points"].clone().reshape((1, self.n_buses))
        # )

        # self.slack_bus_index = network_parameters["slack_bus_index"]
        # self.P_V_bus_indices = network_parameters["P_V_bus_indices"].clone()
        # self.P_Q_bus_indices = network_parameters["P_Q_bus_indices"].clone()

        # self.non_slack_bus_indices = [True] * self.n_buses
        # self.non_slack_bus_indices[self.slack_bus_index] = False
        # self.theta_map = torch.eye(self.n_buses)[:, self.non_slack_bus_indices]
        # self.compute_update_indices()
        self.bus_status = torch.ones((1, self.n_buses))
        self.network_parameters = network_parameters
        self.state_split_size = [self.n_buses, self.n_buses]

        # self.component_power_set_points = network_parameters[
        #     "component_power_set_points"
        # ]
        self.state_names = ["theta"] * self.n_buses + ["V"] * self.n_buses

    def __repr__(self):
        return self.network_parameters["name"]

    def split_voltage_state(self, voltage_state):
        assert voltage_state.shape[1] == 2 * self.n_buses
        theta, V = torch.split(
            voltage_state, split_size_or_sections=self.state_split_size, dim=1
        )
        return theta, V

    def compute_current_injections(self, voltage_state):
        theta, V = self.split_voltage_state(voltage_state)
        V_complex = (V + 1j * 0) * torch.exp(1j * theta) * self.bus_status
        I_complex = V_complex @ self.Y.T
        I_D, I_Q = I_complex.real, I_complex.imag
        return I_D, I_Q

    def compute_component_current(self, voltage_state, P, Q):
        theta, V = self.split_voltage_state(voltage_state)
        V_complex = (V + 1j * 0.0) * torch.exp(1j * theta) * self.bus_status
        I_complex = (P + 1j * Q).conj() / V_complex.conj()
        I_D, I_Q = I_complex.real, I_complex.imag
        return I_D, I_Q

    def compute_rhs(self, time, theta, V, I_D_injections, I_Q_injections):
        V_complex = (V + 1j * 0) * torch.exp(1j * theta) * self.bus_status
        I_complex = V_complex @ self.Y.T
        I_injections = I_D_injections + 1j * I_Q_injections
        current_residual = I_complex - I_injections
        return current_residual.real, current_residual.imag

    def compute_current_residual(self, voltage_state, P, Q):
        I_D_components, I_Q_components = self.compute_component_current(
            voltage_state, P, Q
        )
        I_D_network, I_Q_network = self.compute_current_injections(voltage_state)
        I_D_residual = I_D_components - I_D_network
        I_Q_residual = I_Q_components - I_Q_network
        return torch.hstack([I_D_residual, I_Q_residual])

    def compute_power(self, voltage_state):
        theta, V = self.split_voltage_state(voltage_state)
        V_complex = (V + 1j * 0) * torch.exp(1j * theta) * self.bus_status
        I_complex = V_complex @ self.Y.T
        S = V_complex * I_complex.conj()
        return S.real, S.imag

    def compute_jacobian(self, voltage_state, P, Q):
        assert voltage_state.shape[0] == 1
        jacobian = torch.autograd.functional.jacobian(
            lambda x: self.compute_current_residual(voltage_state=x, P=P, Q=Q),
            inputs=voltage_state,
        )[0, :, 0, :]
        Delta_y = self.compute_current_residual(voltage_state, P=P, Q=Q)
        return Delta_y, jacobian
