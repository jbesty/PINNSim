import numpy as np
import torch

from pinnsim.configurations.load_network_data import import_matpower_acopf_case


def load_static_case_parameters(case_name, power_system_name=None):
    if power_system_name is None:
        power_system_name = case_name
    buses_mat, generators_mat, _ = import_matpower_acopf_case(case_name=case_name)
    n_buses: int = len(buses_mat)
    n_generators: int = len(generators_mat)

    baseMVA: float = 100.0

    loads = [
        dict(
            {
                "bus_id": bus,
                "P_set": -round(buses_mat["Pd"][bus] / baseMVA, ndigits=6),
                "Q_set": -round(buses_mat["Qd"][bus] / baseMVA, ndigits=6),
            }
        )
        for bus in np.where(buses_mat["Pd"].values > 0.0)[0]
    ]

    generators = [
        dict(
            {
                "bus_id": generators_mat["bus"][ii] - 1,
                "generator_id": generators_mat["bus"][ii],
                "P_set": round(generators_mat["Pg"][ii] / baseMVA, ndigits=6),
                "Q_set": round(generators_mat["Qg"][ii] / baseMVA, ndigits=6),
                "V_set": round(generators_mat["Vg"][ii], ndigits=6),
                "name": f"{power_system_name}_{ii+1}",
            }
        )
        for ii in range(n_generators)
    ]

    buses_theta_set = torch.round(
        torch.tensor(np.deg2rad(buses_mat["Va"][:].values)), decimals=6
    )

    P_set_points = torch.zeros(n_buses)
    Q_set_points = torch.zeros(n_buses)
    V_set_points = torch.ones(n_buses)
    component_power_set_points = list()
    for generator in generators:
        P_set_points[generator["bus_id"]] += generator["P_set"]
        Q_set_points[generator["bus_id"]] += generator["Q_set"]
        V_set_points[generator["bus_id"]] = generator["V_set"]
        component_power_set_points.append(torch.tensor([generator["P_set"], 0.0]))

    for load in loads:
        P_set_points[load["bus_id"]] += load["P_set"]
        Q_set_points[load["bus_id"]] += load["Q_set"]
        component_power_set_points.append(torch.tensor([load["P_set"], load["Q_set"]]))

    case_parameters = dict(
        {
            "slack_bus_index": np.where(buses_mat["type"].values == 3)[0][0],
            "P_V_bus_indices": torch.tensor(np.where(buses_mat["type"].values == 2)[0]),
            "P_Q_bus_indices": torch.tensor(np.where(buses_mat["type"].values == 1)[0]),
            "bus_types": torch.tensor(buses_mat["type"].values, dtype=torch.int),
            "set_points": dict(
                {
                    "theta": buses_theta_set.reshape((1, -1)),
                    "V": V_set_points.reshape((1, -1)),
                    "P": P_set_points.reshape((1, -1)),
                    "Q": Q_set_points.reshape((1, -1)),
                }
            ),
            "component_power_set_points": component_power_set_points,
        }
    )

    return case_parameters
