import time as time_pkg

import torch
from assimulo.problem import Implicit_Problem
from assimulo.solvers import IDA

from .simulator_module import Simulator


class SimulatorAssimulo(Simulator):
    def __init__(self, tolerance: float = 1.0e-8, max_order=5, **kwargs):
        super(SimulatorAssimulo, self).__init__(**kwargs)
        self.rtol = tolerance
        self.atol = tolerance
        self.max_order = max_order

        dummy_state = torch.zeros((1, self.grid_model.n_states))
        self.control_input = torch.zeros((1, self.grid_model.n_control_inputs))
        self.implicit_problem = Implicit_Problem(
            res=self.residual,
            y0=dummy_state.numpy().flatten(),
            yd0=dummy_state.numpy().flatten(),
            t0=0.0,
        )

        self.simulation_instance = IDA(self.implicit_problem)
        self.simulation_instance.maxord = self.max_order
        self.simulation_instance.atol = self.atol
        self.simulation_instance.rtol = self.rtol
        self.simulation_instance.algvar = self.grid_model.differential_variables

        self.simulation_instance.verbosity = self.verbosity

    def __repr__(self):
        return "Assimulo"

    def simulate_trajectory(
        self,
        time_end: float,
        component_state_list_initial,
        control_input_list,
        theta_initial,
        V_initial,
        time_repeats=False,
    ):
        self.output_times = time_end
        combined_current_state = self.grid_model.combine_state(
            component_state_list_initial, theta_initial, V_initial
        )
        self.control_input = self.grid_model.combine_control_input(control_input_list)
        current_state, d_dt_current_state = self.compute_consistent_state(
            time=torch.zeros((1, 1)), state=combined_current_state
        )

        self.simulation_instance.y = current_state.numpy().flatten()
        self.simulation_instance.yd = d_dt_current_state.numpy().flatten()
        self.simulation_instance.t = 0.0
        evaluation_points = self.output_times[:, 0].tolist()
        assert evaluation_points[-1] == time_end

        start_time = time_pkg.time()
        time_np, state_np, d_dt_state_np = self.simulation_instance.simulate(
            time_end, ncp=0, ncp_list=evaluation_points
        )
        end_time = time_pkg.time()
        result_time = torch.tensor(time_np).reshape((-1, 1))

        component_states_list, result_theta, result_V = self.grid_model.split_state(
            state=torch.tensor(state_np)
        )

        solver_stats = dict(
            {
                "solver": self.__repr__(),
                "solver_time": end_time - start_time,
                "atol": self.atol,
                "rtol": self.rtol,
                "n_iterations": self.simulation_instance.statistics["nsteps"],
                "time_end": time_end,
            }
        )
        output = dict(
            {
                "time": result_time,
                "component_states_list": component_states_list,
                "theta": result_theta,
                "V": result_V,
                "control_input_list": control_input_list,
                "solver_stats": solver_stats,
            }
        )
        self.check_output(output)
        return output

    def compute_consistent_state(self, time, state):
        yd0 = self.grid_model.compute_dae_rhs(
            time=time, state=state, control_input=self.control_input
        )
        self.simulation_instance.y = state.numpy().flatten()
        self.simulation_instance.yd = yd0.numpy().flatten()

        t, y, yd = self.simulation_instance.make_consistent("IDA_YA_YDP_INIT")
        return torch.tensor(y), torch.tensor(yd)

    def residual(self, time, state, d_dt_state):
        time_torch = torch.tensor(time).reshape((1, 1))
        state_torch = torch.tensor(state).reshape((1, -1))
        d_dt_state_torch = torch.tensor(d_dt_state).reshape((1, -1))
        dae_residual = self.grid_model.compute_dae_residual(
            time=time_torch,
            state=state_torch,
            d_dt_state=d_dt_state_torch,
            control_input=self.control_input,
        )

        return dae_residual.numpy().flatten()
