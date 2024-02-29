import torch


class ACPFSolver:

    def __init__(
        self, verbose=False, step_size=None, tolerance=None, max_iterations: int = None
    ):
        super(ACPFSolver, self).__init__()

        self.verbose = verbose
        if step_size is None:
            self.step_size = torch.tensor(1.0, dtype=torch.float64)

        if tolerance is None:
            self.tolerance = torch.tensor(1e-8, dtype=torch.float64)

        if max_iterations is None:
            self.max_iterations = 20

    def compute_update_indices(self, network, bus_types):
        update_indices = [True] * 2 * network.n_buses

        for bus_index, bus_type in enumerate(bus_types):
            if bus_type == 1:  # PQ bus
                pass
            elif bus_type == 2:  # PV bus
                update_indices[bus_index + network.n_buses] = False
            elif bus_type == 3:  # slack bus
                update_indices[bus_index] = False
                update_indices[bus_index + network.n_buses] = False

        state_map = torch.eye(n=2 * network.n_buses, dtype=torch.float64)[
            update_indices, :
        ]
        return update_indices, state_map

    def solve_power_flow_newton(self, network, bus_types, set_points):
        update_indices, state_map = self.compute_update_indices(
            network=network, bus_types=bus_types
        )

        # TODO: Check correctness - verify with Matpower, numerical implementation might cause issues.
        # set default parameters
        # initialise power flow (assumes flat voltage start
        iteration = 0

        theta = set_points["theta"].clone()
        V = set_points["V"].clone()
        P = set_points["P"].clone()
        Q = set_points["Q"].clone()
        assert torch.min(V) > 0.5
        x = torch.hstack([theta, V])
        Delta_x = torch.ones((len(update_indices), 1))

        while (
            torch.max(torch.abs(Delta_x)) > self.tolerance
            and iteration < self.max_iterations
        ):
            Delta_y, jacobian = network.compute_jacobian(x, P=P, Q=Q)

            Delta_x = torch.linalg.solve(
                jacobian[:, update_indices][update_indices, :].T,
                Delta_y[:, update_indices].T,
            ).T

            x -= self.step_size * Delta_x @ state_map

            iteration += 1

        if (
            torch.max(torch.abs(Delta_x)) < self.tolerance
            and iteration < self.max_iterations
        ):
            if self.verbose:
                print(f"Successful power flow in {iteration} iterations!")
                self.print_power_flow_result(x, network)
        else:
            raise Exception("Power flow not converged.")

        power_flow_solution = network.split_voltage_state(x)
        return power_flow_solution

    def print_power_flow_result(self, voltage_state, network):
        theta, V = network.split_voltage_state(voltage_state)
        P, Q = network.compute_power(voltage_state)

        print("Bus ID | theta (deg) | V (p.u.) | P (p.u.) | Q (p.u.) |")
        for bus_id, (theta_i, V_i, P_i, Q_i) in enumerate(
            zip(theta[0, :], V[0, :], P[0, :], Q[0, :])
        ):
            print(
                f"{bus_id+1:>6d} | {torch.rad2deg(theta_i):>11.3f} | {V_i:>8.3f} | {P_i:>8.3f} | {Q_i:>8.3f} |"
            )

        pass
