import torch

from .. import f_0_Hz


class LoadModelStatic(torch.nn.Module):
    def __init__(self):
        super(LoadModelStatic, self).__init__()

        self.n_states = 0
        self.n_control_inputs = 4

        self.f_0_Hz = torch.nn.parameter.Parameter(
            torch.tensor(f_0_Hz.item()), requires_grad=False
        )
        self.omega_s = self.f_0_Hz * 2 * torch.pi
        self.differential_variable = []
        self.lhs_factors = torch.zeros((1, 0))
        self.alpha = 0

        self.component_type = "Load"
        self.state_names = []
        self.delta_index = None
        self.state_to_delta = torch.ones((0, 1))

    def update_function(self, time, state, control_input, theta, V):
        return torch.zeros(state.shape)

    def compute_current_D_Q(self, state, control_input, theta, V):
        P_0, Q_0, theta_0, V_0 = control_input.split(split_size=1, dim=1)
        V_0_complex = (V_0 + 0j) * torch.exp(1j * theta_0)
        I_complex = (
            ((P_0 + 1j * Q_0) * torch.pow(V / V_0, self.alpha)) / V_0_complex
        ).conj()

        return I_complex.real, I_complex.imag

    def compute_equilibrium(self, set_point):
        P, Q, theta, V = set_point.split(split_size=1, dim=1)

        equilibrium_control_input = torch.hstack([P, Q, theta, V])

        equilibrium_state = torch.zeros((P.shape[0], 0))

        assert equilibrium_state.shape[1] == self.n_states
        assert equilibrium_control_input.shape[1] == self.n_control_inputs

        return equilibrium_state, equilibrium_control_input

    def compute_rhs(self, time, state, control_input, theta, V):
        return torch.zeros(state.shape)

    def compute_lhs(self, time, d_dt_state):
        return d_dt_state * self.lhs_factors
