import torch


class VoltageProfilePolynomial(torch.nn.Module):
    def __init__(self, order_polynomial):
        super(VoltageProfilePolynomial, self).__init__()
        self.n_voltage_parameters = (order_polynomial + 1) * 2
        voltage_parameter_mapping = torch.eye(self.n_voltage_parameters)
        self.theta_mapping = voltage_parameter_mapping[::2, :]
        self.V_mapping = voltage_parameter_mapping[1::2, :]
        self.power_series_exponents = torch.arange(
            start=0, end=order_polynomial + 1, step=1
        ).reshape((1, -1))

    def get_voltage(self, time, voltage_parametrisation):
        time_power = torch.pow(time, exponent=self.power_series_exponents)
        theta = torch.sum(
            time_power * (voltage_parametrisation @ self.theta_mapping.T),
            dim=1,
            keepdim=True,
        )
        V = torch.sum(
            time_power * (voltage_parametrisation @ self.V_mapping.T),
            dim=1,
            keepdim=True,
        )
        return theta, V

    def get_voltage_change(self, time, voltage_parametrisation):
        time_power = torch.pow(time, exponent=self.power_series_exponents)
        theta = torch.sum(
            time_power * (voltage_parametrisation @ self.theta_mapping.T),
            dim=1,
            keepdim=True,
        )
        V = torch.sum(
            time_power * (voltage_parametrisation @ self.V_mapping.T),
            dim=1,
            keepdim=True,
        )
        return (
            theta - voltage_parametrisation[:, 0:1],
            V - voltage_parametrisation[:, 1:2],
        )

    def get_theta_change(self, time, voltage_parametrisation):
        time_power = torch.pow(time, exponent=self.power_series_exponents)
        theta = torch.sum(
            time_power * (voltage_parametrisation @ self.theta_mapping.T),
            dim=1,
            keepdim=True,
        )
        return theta - voltage_parametrisation[:, 0:1]

    def get_basis_function(self, time):
        time_power = torch.pow(time, exponent=self.power_series_exponents)
        return time_power
