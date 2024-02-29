from pinnsim.configurations.load_component_parameters import load_component_parameters
from pinnsim.configurations.load_generator_data import get_machine_data
from pinnsim.configurations.load_network_data import load_network_data
from pinnsim.power_system_models.generator_model import GeneratorModel
from pinnsim.power_system_models.load_model_static import LoadModelStatic
from pinnsim.power_system_models.network_model import NetworkModel
from pinnsim.power_system_models.power_grid import PowerGrid


def load_grid_model(case):
    components_parameters = load_component_parameters(case_name=case)

    components = []
    component_bus_indices = []
    for component in components_parameters:
        if component["model"] == "StaticLoad":
            component_model = LoadModelStatic()
        elif component["model"] == "Generator":
            generator_parameters = get_machine_data(component["name"])
            component_model = GeneratorModel(generator_parameters)
        else:
            raise Exception()
        components.append(component_model)
        component_bus_indices.append(component["bus_id"])

    network = NetworkModel(network_parameters=load_network_data(power_system_name=case))

    grid_model = PowerGrid(
        network=network,
        components=components,
        component_bus_indices=component_bus_indices,
    )
    return grid_model
