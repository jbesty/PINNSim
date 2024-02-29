import numpy as np

from .load_network_data import import_matpower_acopf_case


def load_component_parameters(case_name, power_system_name=None):
    if power_system_name is None:
        power_system_name = case_name
    buses_mat, generators_mat, _ = import_matpower_acopf_case(case_name=case_name)
    n_generators: int = len(generators_mat)

    loads = [
        dict({"bus_id": bus, "model": "StaticLoad"})
        for bus in np.where(buses_mat["Pd"].values > 0.0)[0]
    ]

    generators = [
        dict(
            {
                "bus_id": generators_mat["bus"][ii] - 1,
                "generator_id": generators_mat["bus"][ii],
                "model": "Generator",
                "name": f"{power_system_name}_{ii+1}",
            }
        )
        for ii in range(n_generators)
    ]

    components_parameters = generators + loads

    return components_parameters
