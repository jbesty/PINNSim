# Combined model
import torch.nn


class PowerGrid(torch.nn.Module):
    # combines network model with components
    # can compute rhs / lhs function
    # contains all the "injection" logic, i.e., which bus where

    def __init__(self, network, components, component_bus_indices):
        super(PowerGrid, self).__init__()

        self.network = network
        self.components = components
        self.n_components = len(self.components)
        # bus indices start with 0!!!
        self.component_bus_indices = component_bus_indices
        assert self.n_components == len(component_bus_indices)

        self.state_split_list = [
            component.n_states for component in self.components
        ] + [self.network.n_buses] * 2
        self.n_states = sum(self.state_split_list)
        self.control_input_split_list = [
            component.n_control_inputs for component in self.components
        ]
        self.n_control_inputs = sum(self.control_input_split_list)

        self.differential_variables = list()

        for component in components:
            self.differential_variables += component.differential_variable

        self.differential_variables += [False] * 2 * self.network.n_buses

        self.component_map = torch.zeros((self.n_components, self.network.n_buses))
        for ii, bus in enumerate(component_bus_indices):
            self.component_map[ii, bus] = 1.0

        self.component_types = [
            component.component_type for component in self.components
        ]
        self.state_name_list = list()
        for component in self.components:
            self.state_name_list += component.state_names
        self.state_name_list += network.state_names

        if network.n_buses == 9:
            self.case = "ieee9"
        else:
            raise Exception

    def split_state(self, state):
        states_split = state.split(split_size=self.state_split_list, dim=1)
        return states_split[:-2], states_split[-2], states_split[-1]

    @staticmethod
    def combine_state(component_state_list, theta, V):
        return torch.hstack([torch.hstack(component_state_list), theta, V])

    def split_control_input(self, control_input):
        control_input_split = control_input.split(
            split_size=self.control_input_split_list, dim=1
        )
        return control_input_split

    @staticmethod
    def combine_control_input(component_control_input_list):
        return torch.hstack(component_control_input_list)

    def assert_valid_residual_input(self, time, state, d_dt_state, control_input):
        assert (
            time.shape[0]
            == state.shape[0]
            == d_dt_state.shape[0]
            == control_input.shape[0]
        )
        assert state.shape[1] == d_dt_state.shape[1] == self.n_states
        assert control_input.shape[1] == self.n_control_inputs
        pass

    def compute_dae_lhs(self, time, d_dt_state):
        d_dt_state_list, d_dt_theta, d_dt_V = self.split_state(d_dt_state)

        state_lhs_list = self.compute_components_lhs(
            time=time, d_dt_state_list=d_dt_state_list
        )

        return self.combine_state(
            state_lhs_list, torch.zeros(d_dt_theta.shape), torch.zeros(d_dt_V.shape)
        )

    def compute_dae_rhs(self, time, state, control_input):
        state_list, theta, V = self.split_state(state=state)
        control_input_list = self.split_control_input(control_input=control_input)
        theta_list, V_list = self.compute_voltage_lists(theta=theta, V=V)

        state_rhs_list = self.compute_components_rhs(
            time=time,
            state_list=state_list,
            control_input_list=control_input_list,
            theta_list=theta_list,
            V_list=V_list,
        )

        I_D_injections, I_Q_injections = self.compute_component_current_injections(
            time=time,
            state_list=state_list,
            control_input_list=control_input_list,
            theta_list=theta_list,
            V_list=V_list,
        )

        I_D_rhs, I_Q_rhs = self.network.compute_rhs(
            time=time,
            theta=theta,
            V=V,
            I_D_injections=I_D_injections,
            I_Q_injections=I_Q_injections,
        )

        return self.combine_state(state_rhs_list, I_D_rhs, I_Q_rhs)

    def compute_dae_residual(self, time, state, d_dt_state, control_input):
        self.assert_valid_residual_input(time, state, d_dt_state, control_input)

        dae_lhs = self.compute_dae_lhs(time=time, d_dt_state=d_dt_state)
        dae_rhs = self.compute_dae_rhs(
            time=time, state=state, control_input=control_input
        )
        return dae_lhs - dae_rhs

    def compute_component_current_injections(
        self, time, state_list, control_input_list, theta_list, V_list
    ):
        current_injections = [
            component.compute_current_D_Q(
                state=state, control_input=control_input, theta=theta, V=V
            )
            for component, state, control_input, theta, V in zip(
                self.components, state_list, control_input_list, theta_list, V_list
            )
        ]

        I_D_injections = (
            torch.hstack([injection[0] for injection in current_injections])
            @ self.component_map
        )
        I_Q_injections = (
            torch.hstack([injection[1] for injection in current_injections])
            @ self.component_map
        )

        return I_D_injections, I_Q_injections

    def compute_components_rhs(
        self, time, state_list, control_input_list, theta_list, V_list
    ):
        state_rhs_list = [
            component.compute_rhs(
                time=time, state=state, control_input=control_input, theta=theta, V=V
            )
            for component, state, control_input, theta, V in zip(
                self.components, state_list, control_input_list, theta_list, V_list
            )
        ]
        return state_rhs_list

    def compute_components_update_function(
        self, time, state_list, control_input_list, theta_list, V_list
    ):
        state_update_list = [
            component.update_function(
                time=time, state=state, control_input=control_input, theta=theta, V=V
            )
            for component, state, control_input, theta, V in zip(
                self.components, state_list, control_input_list, theta_list, V_list
            )
        ]
        return state_update_list

    def compute_components_lhs(self, time, d_dt_state_list):
        state_lhs_list = [
            component.compute_lhs(time=time, d_dt_state=d_dt_state)
            for component, d_dt_state in zip(self.components, d_dt_state_list)
        ]
        return state_lhs_list

    def compute_equilibrium_values(self, set_points):
        equilibrium_values = [
            component.compute_equilibrium(set_point)
            for component, set_point in zip(self.components, set_points)
        ]

        equilibrium_state_list = [
            equilibrium_value[0] for equilibrium_value in equilibrium_values
        ]
        equilibrium_control_input_list = [
            equilibrium_value[1] for equilibrium_value in equilibrium_values
        ]

        return equilibrium_state_list, equilibrium_control_input_list

    def compute_voltage_lists(self, theta, V):
        theta_list = (theta @ self.component_map.T).split(split_size=1, dim=1)
        V_list = (V @ self.component_map.T).split(split_size=1, dim=1)
        return theta_list, V_list
