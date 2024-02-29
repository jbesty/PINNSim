import itertools
import operator

import torch
from pinnsim.numerics.predictors.predictor_module import PredictorModule
from pinnsim.numerics.simulators.simulator_iterative import IterativeSimulator
from pinnsim.post_processing.rebuild_nn_model import load_model_from_local_files
from pinnsim.power_system_models.voltage_profile_polynomial import (
    VoltageProfilePolynomial,
)

## The following code is not optimised, neither in terms of speed nor in terms of clean implementation,
## as the implementation aimed to closely resemble Algorithm 1 in the referenced paper.

## Furthermore, some parts of the simulator are not yet generic, so expect some hard-coded elements.
## The voltage profile order, for example, is assumed to be 2 or less.


class SimulatorDistributed(IterativeSimulator):
    def __init__(
        self,
        predictor_schemes,
        voltage_profile_order,
        n_extra_points,
        border_points=True,
        **kwargs,
    ):
        super(SimulatorDistributed, self).__init__(**kwargs)
        self.predictor_schemes = predictor_schemes
        self.voltage_profile_order = voltage_profile_order
        self.n_extra_points = n_extra_points
        self.border_points = border_points
        n_collocation_points = 1 + self.voltage_profile_order + n_extra_points
        assert n_collocation_points >= 1
        if border_points:
            if n_collocation_points == 1:
                collocation_points = [0.0]
            else:
                collocation_points = [
                    point / (n_collocation_points - 1)
                    for point in range(0, n_collocation_points)
                ]
        else:
            collocation_points = [
                point / n_collocation_points + 1 / (2 * n_collocation_points)
                for point in range(0, n_collocation_points)
            ]

        self.n_collocation_points = n_collocation_points
        assert len(collocation_points) == n_collocation_points
        self.collocation_points = torch.tensor(collocation_points).reshape((-1, 1))

        self.voltage_profiles = [
            VoltageProfilePolynomial(
                # order_polynomial=voltage_profile_order
                order_polynomial=2  # hard coded due to PINN training - entries can be 0
            )
            for _ in range(self.grid_model.network.n_buses)
        ]
        self.voltage_parameters_split_sizes = [
            voltage_profile.n_voltage_parameters
            for voltage_profile in self.voltage_profiles
        ]
        self.n_voltage_parameters = sum(self.voltage_parameters_split_sizes)
        parameter_indices = [0] + list(
            itertools.accumulate(self.voltage_parameters_split_sizes, operator.add)
        )
        self.bus_voltage_parameter_list = [
            list(range(start_index, end_index))
            for start_index, end_index in zip(
                parameter_indices[:-1], parameter_indices[1:]
            )
        ]

        self.predictors = []

        for predictor_scheme, component, bus in zip(
            predictor_schemes,
            self.grid_model.components,
            self.grid_model.component_bus_indices,
        ):
            if isinstance(predictor_scheme, str):
                nn_model, model_config = load_model_from_local_files(
                    run_id=predictor_scheme
                )
                assert (
                    model_config.generator_name
                    == component.generator_config["generator_name"]
                )
                self.predictors.append(nn_model)
            elif issubclass(predictor_scheme, PredictorModule):
                self.predictors.append(
                    predictor_scheme(
                        component=component, voltage_profile=self.voltage_profiles[bus]
                    )
                )
            else:
                raise Exception

        if voltage_profile_order == 0:
            self.used_indices = [
                True,
                True,
                False,
                False,
                False,
                False,
            ] * self.grid_model.network.n_buses
        elif voltage_profile_order == 1:
            self.used_indices = [
                True,
                True,
                True,
                True,
                False,
                False,
            ] * self.grid_model.network.n_buses
        elif voltage_profile_order == 2:
            self.used_indices = [
                True,
                True,
                True,
                True,
                True,
                True,
            ] * self.grid_model.network.n_buses
        else:
            raise NotImplementedError

        self.index_mapping = torch.eye(len(self.used_indices))[:, self.used_indices]

    def __repr__(self):
        return f"Distributed {self.predictor_schemes}"

    def simulator_config(self):
        simulator_config = f"{self.predictor_schemes}, r = {self.voltage_profile_order}, s={self.n_collocation_points}"
        if not self.predictor_schemes == "ODE":
            simulator_config += f", t_0={self.border_points}"

        return simulator_config

    def solver_dict_extra(self):
        extra_entries = dict(
            {
                "predictor_scheme": self.predictor_schemes,
                "r": self.voltage_profile_order,
                "s": self.n_collocation_points,
                "t0": self.border_points,
            }
        )
        return extra_entries

    def simulate_timestep(self, component_state_list, control_input_list, theta, V):
        voltage_parametrisation_list = self.create_initial_value_list(
            theta_initial=theta, V_initial=V
        )

        collocation_times = self.collocation_points * self.timestep_size

        (
            voltage_parametrisation_list_final,
            timestep_summary,
        ) = self.update_voltage_parametrisation(
            time=collocation_times,
            state_list=component_state_list,
            control_input_list=control_input_list,
            voltage_parametrisation_list=voltage_parametrisation_list,
        )

        with torch.no_grad():
            voltage_results = self.voltage_forward(
                time=self.output_times,
                voltage_parametrisation_list=voltage_parametrisation_list_final,
            )

            state_results = self.forward_state(
                time=self.output_times,
                state_list=component_state_list,
                control_input_list=control_input_list,
                voltage_parametrisation_list=voltage_parametrisation_list_final,
            )

        result_dict = dict(
            {
                "time": self.output_times,
                "component_states_list": state_results,
                "theta": voltage_results[0],
                "V": voltage_results[1],
                "timestep_summary": timestep_summary,
            }
        )
        return result_dict

    def update_voltage_parametrisation(
        self,
        time,
        state_list,
        control_input_list,
        voltage_parametrisation_list,
    ):
        k_iteration = 0
        max_variable_change = 1.0
        residual_norm = 1.0

        voltage_parametrisation_list_iteration = [
            element.detach() for element in voltage_parametrisation_list
        ]
        parameter_update_list = list()
        state_results_list = list()
        residual_norm_list = list()

        state_prediction = self.forward_state(
            time=torch.tensor([[self.timestep_size]]),
            state_list=state_list,
            control_input_list=control_input_list,
            voltage_parametrisation_list=voltage_parametrisation_list_iteration,
        )
        state_results_list.append(torch.hstack(state_prediction).detach().clone())

        while (
            k_iteration < self.newton_scheme_max_iterations
            and max_variable_change > self.newton_scheme_tolerance
        ):
            voltage_parametrisation_list_iteration_new = [
                element.detach().clone()
                for element in voltage_parametrisation_list_iteration
            ]
            network_currents, network_jacobian = self.compute_network_current_jacobian(
                time=time,
                voltage_parametrisation_list=voltage_parametrisation_list_iteration_new,
            )
            voltage_parametrisation_list_iteration_new = [
                element.detach().clone()
                for element in voltage_parametrisation_list_iteration
            ]
            (
                I_D_dynamic,
                I_Q_dynamic,
                I_D_dynamic_jacobians,
                I_Q_dynamic_jacobians,
            ) = self.compute_component_current_jacobian(
                time=time,
                state_list=state_list,
                control_input_list=control_input_list,
                voltage_parametrisation_list=voltage_parametrisation_list_iteration_new,
            )

            residuals = network_currents - torch.hstack(
                [
                    I_D_dynamic @ self.grid_model.component_map,
                    I_Q_dynamic @ self.grid_model.component_map,
                ]
            )
            dynamic_jacobian = torch.zeros(network_jacobian.shape)
            for bus, I_D_dynamic_jacobian, I_Q_dynamic_jacobian in zip(
                self.grid_model.component_bus_indices,
                I_D_dynamic_jacobians,
                I_Q_dynamic_jacobians,
            ):
                tensor_indices = self.bus_voltage_parameter_list[bus]
                dynamic_jacobian[:, bus, tensor_indices] += I_D_dynamic_jacobian
                dynamic_jacobian[
                    :, self.grid_model.network.n_buses + bus, tensor_indices
                ] += I_Q_dynamic_jacobian

            jacobian = network_jacobian - dynamic_jacobian
            jacobian_long = torch.hstack(
                jacobian[:, :, self.used_indices].split(split_size=1, dim=0)
            )[0, :, :]
            residual_long = torch.hstack(residuals.split(split_size=1, dim=0))[0, :]

            with torch.no_grad():
                lstsq_solution = torch.linalg.lstsq(
                    jacobian_long,
                    residual_long,
                )
                update_parameters = self.index_mapping @ lstsq_solution.solution
                voltage_parametrisation_update_list = update_parameters.split(
                    self.voltage_parameters_split_sizes
                )
                for ii in range(self.grid_model.network.n_buses):
                    voltage_parametrisation_list_iteration[
                        ii
                    ] -= voltage_parametrisation_update_list[ii]

            max_variable_change = torch.max(torch.abs(lstsq_solution.solution)).item()
            residual_norm = (
                torch.square(torch.linalg.norm(residuals)).item()
                * self.timestep_size
                / self.n_collocation_points
            )

            voltage_parametrisation_list_iteration_cloned = [
                element.detach().clone()
                for element in voltage_parametrisation_update_list
            ]
            parameter_update_list.append(voltage_parametrisation_list_iteration_cloned)
            k_iteration += 1
            if self.verbosity <= 20:
                self.print_iteration_information(
                    k_iteration, residual_norm, max_variable_change
                )

            state_prediction = self.forward_state(
                time=torch.tensor([[self.timestep_size]]),
                state_list=state_list,
                control_input_list=control_input_list,
                voltage_parametrisation_list=voltage_parametrisation_list_iteration_cloned,
            )
            state_results_list.append(torch.hstack(state_prediction).detach().clone())
            residual_norm_list.append(residual_norm)

        timestep_summary = dict(
            {
                "residual_norm": residual_norm,
                "residual_norms": residual_norm_list,
                "max_variable_change": max_variable_change,
                "n_iterations": k_iteration,
                "parameter_update_list": parameter_update_list,
                "state_results_list": state_results_list,
            }
        )

        return voltage_parametrisation_list_iteration, timestep_summary

    def split_voltage_parametrisation(self, voltage_parametrisations):
        return voltage_parametrisations.split(
            split_size=self.voltage_parameters_split_sizes, dim=1
        )

    @staticmethod
    def stack_voltage_parametrisation(voltage_parametrisation_list):
        return torch.hstack(voltage_parametrisation_list)

    def create_initial_value_list(self, theta_initial, V_initial):
        voltage_parametrisation_list = list()
        for theta, V, n_parameters in zip(
            theta_initial[0, :], V_initial[0, :], self.voltage_parameters_split_sizes
        ):
            voltage_parametrisation = torch.zeros((1, n_parameters))
            voltage_parametrisation[0, 0] = theta
            voltage_parametrisation[0, 1] = V
            voltage_parametrisation_list.append(voltage_parametrisation)

        return voltage_parametrisation_list

    def voltage_forward(self, time, voltage_parametrisation_list):
        results = [
            voltage_profile.get_voltage(
                time=time, voltage_parametrisation=voltage_parametrisation
            )
            for voltage_profile, voltage_parametrisation in zip(
                self.voltage_profiles, voltage_parametrisation_list
            )
        ]

        theta = torch.hstack([result[0] for result in results])
        V = torch.hstack([result[1] for result in results])

        return theta, V

    def compute_network_currents(self, time, voltage_parametrisations):
        theta, V = self.voltage_forward(
            time=time,
            voltage_parametrisation_list=self.split_voltage_parametrisation(
                voltage_parametrisations
            ),
        )

        V_complex = (V + 0j) * torch.exp(1j * theta)
        network_currents = V_complex @ self.grid_model.network.Y.T
        return torch.hstack([network_currents.real, network_currents.imag])

    def compute_network_current_jacobian(self, time, voltage_parametrisation_list):
        with torch.no_grad():
            voltage_parametrisation = self.stack_voltage_parametrisation(
                voltage_parametrisation_list
            )
            network_currents = self.compute_network_currents(
                time, voltage_parametrisation
            )
            network_currents_jacobian = torch.autograd.functional.jacobian(
                func=lambda x: self.compute_network_currents(time, x),
                inputs=voltage_parametrisation,
                vectorize=True,
                create_graph=False,
            )[:, :, 0, :]

        return network_currents, network_currents_jacobian

    def forward_state(
        self, time, state_list, control_input_list, voltage_parametrisation_list
    ):
        result_state_list = [
            (
                predictor.predict_state(
                    time=time,
                    state=state.repeat(time.shape),
                    control_input=control_input.repeat(time.shape),
                    voltage_parametrisation=voltage_parametrisation_list[bus].repeat(
                        time.shape
                    ),
                )
                if "NN" in predictor.__repr__()
                else predictor.predict_state(
                    time=time,
                    state=state,
                    control_input=control_input,
                    voltage_parametrisation=voltage_parametrisation_list[bus],
                )
            )
            for predictor, state, control_input, bus in zip(
                self.predictors,
                state_list,
                control_input_list,
                self.grid_model.component_bus_indices,
            )
        ]
        return result_state_list

    def compute_component_currents(
        self, time, state_list, control_input_list, voltage_parametrisation_list
    ):
        current_injections_dynamic = [
            predictor.predict_current(
                time=time,
                state=state,
                control_input=control_input,
                voltage_parametrisation=voltage_parametrisation_list[bus],
            )
            for predictor, state, control_input, bus in zip(
                self.predictors,
                state_list,
                control_input_list,
                self.grid_model.component_bus_indices,
            )
        ]
        injections_dynamic_I_D = torch.hstack(
            [injection[0] for injection in current_injections_dynamic]
        )
        injections_dynamic_I_Q = torch.hstack(
            [injection[1] for injection in current_injections_dynamic]
        )
        return injections_dynamic_I_D, injections_dynamic_I_Q

    def compute_component_current_jacobian(
        self, time, state_list, control_input_list, voltage_parametrisation_list
    ):
        current_injections_dynamic = [
            predictor.predict_current_jacobian(
                time=time,
                state=state,
                control_input=control_input,
                voltage_parametrisation=voltage_parametrisation_list[bus],
            )
            for predictor, state, control_input, bus in zip(
                self.predictors,
                state_list,
                control_input_list,
                self.grid_model.component_bus_indices,
            )
        ]
        injections_dynamic_I_D = torch.hstack(
            [injection[0] for injection in current_injections_dynamic]
        )
        injections_dynamic_I_Q = torch.hstack(
            [injection[1] for injection in current_injections_dynamic]
        )

        injections_dynamic_I_D_jacobian = [
            injection[2] for injection in current_injections_dynamic
        ]
        injections_dynamic_I_Q_jacobian = [
            injection[3] for injection in current_injections_dynamic
        ]
        return (
            injections_dynamic_I_D,
            injections_dynamic_I_Q,
            injections_dynamic_I_D_jacobian,
            injections_dynamic_I_Q_jacobian,
        )
