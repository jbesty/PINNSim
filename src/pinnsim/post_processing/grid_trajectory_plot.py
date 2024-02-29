import pinnsim
import torch
from matplotlib import pyplot as plt


class GridTrajectoryPlot:

    def __init__(self, grid_model):
        super(GridTrajectoryPlot, self).__init__()

        self.grid_model = grid_model

        self.fig, self.axs = plt.subplots(nrows=4, ncols=1, sharex="all", squeeze=True)

        self.axs[0].set_ylabel(r"$\delta_i - \theta_i$ [deg]")
        self.axs[1].set_ylabel(r"$\Delta \omega_i$ [Hz]")
        self.axs[2].set_ylabel(r"$\theta_i - \theta_1$ [deg]")
        self.axs[3].set_ylabel(r"$V_i$ [p.u.]")
        self.axs[3].set_xlabel(r"Time [s]")

    def add_results(self, results_data, line_style="-k"):
        for component_result, component_bus in zip(
            results_data["component_states_list"][:3],
            self.grid_model.component_bus_indices,
        ):
            self.axs[0].plot(
                results_data["time"][:, 0],
                torch.rad2deg(
                    component_result[:, 2] - results_data["theta"][:, component_bus]
                ),
                line_style,
            )

            self.axs[1].plot(
                results_data["time"][:, 0],
                component_result[:, 3] * pinnsim.f_0_Hz,
                line_style,
            )

        for result_series in results_data["theta"].T:
            self.axs[2].plot(
                results_data["time"][:, 0],
                torch.rad2deg(result_series - results_data["theta"][:, 0]),
                line_style,
            )

        for result_series in results_data["V"].T:
            self.axs[3].plot(results_data["time"][:, 0], result_series, line_style)

        pass

    def show_plot(self, time_end):

        self.axs[3].set_xlim([0.0, time_end])

        self.fig.show()
