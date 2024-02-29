import torch

from .. import f_0_Hz


class GeneratorModel(torch.nn.Module):
    def __init__(self, generator_config):
        super(GeneratorModel, self).__init__()

        self.n_states = 4
        self.n_control_inputs = 2

        self.f_0_Hz = torch.nn.parameter.Parameter(
            torch.tensor(f_0_Hz.item()), requires_grad=False
        )
        self.omega_s = self.f_0_Hz * 2 * torch.pi
        self.generator_config = generator_config

        # system parameters

        self.H = torch.nn.parameter.Parameter(
            torch.tensor(generator_config["H_s"]), requires_grad=False
        )

        self.D = torch.nn.parameter.Parameter(
            torch.tensor(generator_config["D_pu"]), requires_grad=False
        )
        self.X_d_prime = torch.nn.parameter.Parameter(
            torch.tensor(generator_config["X_d_prime_pu"]), requires_grad=False
        )
        self.X_d = torch.nn.parameter.Parameter(
            torch.tensor(generator_config["X_d_pu"]), requires_grad=False
        )
        self.X_q = torch.nn.parameter.Parameter(
            torch.tensor(generator_config["X_q_pu"]), requires_grad=False
        )
        self.X_q_prime = torch.nn.parameter.Parameter(
            torch.tensor(generator_config["X_q_prime_pu"]), requires_grad=False
        )
        self.T_d0_prime = torch.nn.parameter.Parameter(
            torch.tensor(generator_config["T_d0_prime_s"]), requires_grad=False
        )
        self.T_q0_prime = torch.nn.parameter.Parameter(
            torch.tensor(generator_config["T_q0_prime_s"]), requires_grad=False
        )
        self.R_s = torch.nn.parameter.Parameter(
            torch.tensor(generator_config["R_s_pu"]), requires_grad=False
        )

        self.lhs_factors = torch.stack(
            [self.T_d0_prime, self.T_q0_prime, torch.tensor(1.0), 2 * self.H]
        ).reshape((1, -1))
        self.differential_variable = [True] * 4

        # classical model assumption
        self.X_q = self.X_d_prime
        self.X_q_prime = self.X_d_prime

        self.Z = torch.tensor([[self.R_s, -self.X_q_prime], [self.X_d_prime, self.R_s]])
        self.Z_inverse = torch.linalg.inv(self.Z)

        # additional model related information
        self.delta_index = 2
        self.state_to_delta = torch.tensor([0.0, 0.0, 1.0, 0.0]).reshape((-1, 1))
        self.state_names = ["E_q_prime", "E_d_prime", "delta", "omega"]
        self.component_type = "Generator"

        norm_to_scale_delta = torch.tensor(generator_config["norm_to_scale_delta"])
        norm_to_scale_omega = torch.tensor(generator_config["norm_to_scale_omega"])

        self.norm_to_scale_delta = torch.nn.parameter.Parameter(
            norm_to_scale_delta, requires_grad=False
        )
        self.norm_to_scale_omega = torch.nn.parameter.Parameter(
            norm_to_scale_omega, requires_grad=False
        )

        # scaling parameters
        self.norm_to_scale = torch.tensor(
            [[0.0, 0.0, norm_to_scale_delta, norm_to_scale_omega]]
        )
        self.scale_to_norm = torch.tensor(
            [[0.0, 0.0, 1 / norm_to_scale_delta, 1 / norm_to_scale_omega]]
        )
        self.scale_states_to_print = torch.tensor(
            [[1.0, 1.0, 180 / torch.pi, self.f_0_Hz]]
        )
        self.eval()

    def __repr__(self):
        return self.generator_config["parameter_set_name"]

    def update_function(self, time, state, control_input, theta, V):
        E_q_prime, E_d_prime, delta, omega = state.split(split_size=1, dim=1)
        P_M, E_fd = control_input.split(split_size=1, dim=1)
        I_d, I_q = self.compute_current_d_q(E_q_prime, E_d_prime, delta, theta, V)
        P_e = self.compute_electric_power(E_q_prime, E_d_prime, I_d, I_q)
        d_dt_delta = omega * self.omega_s
        d_dt_omega = (P_M - P_e - self.D * omega) / (2 * self.H)
        return torch.hstack(
            [
                torch.zeros(E_q_prime.shape),
                torch.zeros(E_d_prime.shape),
                d_dt_delta,
                d_dt_omega,
            ]
        )

    def compute_rhs(self, time, state, control_input, theta, V):
        E_q_prime, E_d_prime, delta, omega = state.split(split_size=1, dim=1)
        P_M, E_fd = control_input.split(split_size=1, dim=1)
        I_d, I_q = self.compute_current_d_q(E_q_prime, E_d_prime, delta, theta, V)
        P_e = self.compute_electric_power(E_q_prime, E_d_prime, I_d, I_q)
        d_dt_delta = omega * self.omega_s
        d_dt_omega = P_M - P_e - self.D * omega
        return torch.hstack(
            [
                torch.zeros(E_q_prime.shape),
                torch.zeros(E_d_prime.shape),
                d_dt_delta,
                d_dt_omega,
            ]
        )

    def compute_lhs(self, time, d_dt_state):
        return d_dt_state * self.lhs_factors

    def compute_current_d_q(self, E_q_prime, E_d_prime, delta, theta, V):
        v1 = E_d_prime - V * torch.sin(delta - theta)
        v2 = E_q_prime - V * torch.cos(delta - theta)
        I_d = self.Z_inverse[0, 0] * v1 + self.Z_inverse[0, 1] * v2
        I_q = self.Z_inverse[1, 0] * v1 + self.Z_inverse[1, 1] * v2
        return I_d, I_q

    def compute_current_D_Q(self, state, control_input=None, theta=None, V=None):
        E_q_prime, E_d_prime, delta, _ = state.split(split_size=1, dim=1)
        I_d, I_q = self.compute_current_d_q(
            E_q_prime=E_q_prime, E_d_prime=E_d_prime, delta=delta, theta=theta, V=V
        )

        I_D = I_d * torch.cos(delta - torch.pi / 2) - I_q * torch.sin(
            delta - torch.pi / 2
        )
        I_Q = I_d * torch.sin(delta - torch.pi / 2) + I_q * torch.cos(
            delta - torch.pi / 2
        )
        return I_D, I_Q

    def compute_electric_power(self, E_q_prime, E_d_prime, I_d, I_q):
        P_e = (
            E_d_prime * I_d
            + E_q_prime * I_q
            + (self.X_q_prime - self.X_d_prime) * I_d * I_q
        )
        return P_e

    def compute_power_injection(self, I_d, I_q, delta, theta, V):
        I_D, I_Q = self.transform_currents_d_q_to_D_Q(I_d=I_d, I_q=I_q, delta=delta)
        S = V * torch.exp(1j * theta) * (I_D - 1j * I_Q)
        return S.real, S.imag

    @staticmethod
    def transform_currents_d_q_to_D_Q(I_d, I_q, delta):
        I_D_Q = (I_d + 1j * I_q) * torch.exp(1j * (delta - torch.pi / 2))
        return I_D_Q.real, I_D_Q.imag

    def compute_equilibrium(self, set_point):
        P, Q, theta, V = set_point.split(split_size=1, dim=1)
        V_complex = V * torch.exp(1j * theta)
        current_injection = (P - 1j * Q) / V_complex.conj()
        V_rotor = V_complex + (self.R_s + 1j * self.X_q) * current_injection
        delta = torch.angle(V_rotor)

        I_d_q = current_injection * torch.exp(-1j * (delta - torch.pi / 2))
        V_d_q = V_complex * torch.exp(-1j * (delta - torch.pi / 2))

        E_d_prime = (self.X_q - self.X_q_prime) * I_d_q.imag
        E_q_prime = V_d_q.imag + self.R_s * I_d_q.imag + self.X_d_prime * I_d_q.real

        E_fd = E_q_prime + (self.X_d - self.X_d_prime) * I_d_q.real
        P_M = self.compute_electric_power(
            E_q_prime, E_d_prime, I_d_q.real, I_q=I_d_q.imag
        )
        equilibrium_control_input = torch.hstack([P_M, E_fd])

        equilibrium_state = torch.hstack(
            [E_q_prime, E_d_prime, delta, torch.zeros(delta.shape)]
        )

        return equilibrium_state, equilibrium_control_input
