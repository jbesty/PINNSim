import numpy as np
import pandas as pd
import torch

from pinnsim import DATA_PATH


def load_network_data(power_system_name):
    buses_mat, _, branches_mat = import_matpower_acopf_case(power_system_name)
    n_buses: int = len(buses_mat)

    baseMVA: float = 100.0

    Ybus, _, _ = compute_admittance_matrix(
        branches=branches_mat, buses=buses_mat, baseMVA=baseMVA
    )
    assert n_buses == Ybus.shape[0] == Ybus.shape[1]
    if power_system_name == "ieee9":
        display_name = "IEEE 9-bus system"
        assert n_buses == 9
    else:
        raise Exception

    network_parameters = dict(
        {
            "n_buses": n_buses,
            "Y_bus_pu": torch.complex(torch.tensor(Ybus.real), torch.tensor(Ybus.imag)),
            "name": display_name,
        }
    )

    return network_parameters


def import_matpower_acopf_case(case_name):
    available_cases = ["ieee9"]
    if case_name not in available_cases:
        raise Exception(
            f"Case not available. Choose among the following:\n {available_cases}"
        )

    bus_labels = [
        "bus",
        "type",
        "Pd",
        "Qd",
        "Gs",
        "Bs",
        "area",
        "Vm",
        "Va",
        "baseKV",
        "zone",
        "Vmax",
        "Vmin",
        "lambda_vmax",
        "lambda_vmin",
        "mu_vmax",
        "mu_vmin",
    ]
    generator_labels = [
        "bus",
        "Pg",
        "Qg",
        "Qmax",
        "Qmin",
        "Vg",
        "mBase",
        "status",
        "Pmax",
        "Pmin",
        "Pc1",
        "Pc2",
        "Qc1min",
        "Qc1max",
        "Qc2min",
        "Qc2max",
        "ramp_agc",
        "ramp_10",
        "ramp_30",
        "ramp_q",
        "apf",
        "mu_pmax",
        "mu_pmin",
        "mu_qmax",
        "mu_qmin",
    ]
    branch_labels = [
        "fbus",
        "tbus",
        "r",
        "x",
        "b",
        "rateA",
        "rateB",
        "rateC",
        "ratio",
        "angle",
        "status",
        "angmin",
        "angmax",
        "pf",
        "qf",
        "pt",
        "qt",
        "mu_sf",
        "mu_st",
        "mu_angmin",
        "mu_angmax",
    ]

    case_path = DATA_PATH / "matpower_data"

    matlab_buses = pd.read_csv(case_path / f"{case_name}_bus.csv", names=bus_labels)
    matlab_generators = pd.read_csv(
        case_path / f"{case_name}_gen.csv", names=generator_labels
    )
    matlab_branches = pd.read_csv(
        case_path / f"{case_name}_branch.csv", names=branch_labels
    )

    return matlab_buses, matlab_generators, matlab_branches


def compute_admittance_matrix(branches, buses, baseMVA):
    n_buses = len(buses["bus"].values)
    n_branches = len(branches["fbus"].values)
    # for each branch, compute the elements of the branch admittance matrix where
    #
    #      | If |   | Yff  Yft |   | Vf |
    #      |    | = |          | * |    |
    #      | It |   | Ytf  Ytt |   | Vt |
    #
    branch_status = branches["status"].values  # ones at in-service branches
    Ys = branch_status / (
        branches["r"].values + 1j * branches["x"].values
    )  # series admittance
    Bc = branch_status * branches["b"].values  # line charging susceptance
    tap = branches["ratio"].values.copy()
    tap[tap == 0] = 1  # set all zero tap ratios (lines) to 1
    tap_complex = tap * np.exp(
        1j * np.pi / 180 * branches["angle"].values
    )  # add phase shifters
    Ytt = Ys + 1j * Bc / 2
    Yff = Ytt / (tap_complex * np.conj(tap_complex))
    Yft = -Ys / np.conj(tap_complex)
    Ytf = -Ys / tap_complex

    Ysh = (
        buses["Gs"].values + 1j * buses["Bs"].values
    ) / baseMVA  # vector of shunt admittances

    Cf = np.zeros((n_branches, n_buses))
    Ct = np.zeros((n_branches, n_buses))
    for ii, (bus_from, bus_to) in enumerate(
        zip(branches["fbus"].values - 1, branches["tbus"].values - 1)
    ):
        Cf[ii, bus_from] = 1
        Ct[ii, bus_to] = 1

    # build Yf and Yt such that Yf * V is the vector of complex branch currents injected
    # at each branch's 'from' bus, and Yt is the same for the 'to' bus end

    Yf = np.diag(Yff) @ Cf + np.diag(Yft) @ Ct
    Yt = np.diag(Ytf) @ Cf + np.diag(Ytt) @ Ct

    Ybus = Cf.T @ Yf + Ct.T @ Yt + np.diag(Ysh)

    return Ybus, Yf, Yt
