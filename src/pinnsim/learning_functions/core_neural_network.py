from collections import OrderedDict

import torch


class Standardise(torch.nn.Module):
    """
    Scale the input to the layer by mean and standard deviation.
    """

    def __init__(self, n_neurons):
        super(Standardise, self).__init__()
        self.mean = torch.nn.Parameter(data=torch.zeros(n_neurons), requires_grad=False)
        self.standard_deviation = torch.nn.Parameter(
            data=torch.ones(n_neurons), requires_grad=False
        )
        self.eps = 1e-8

    def forward(self, input):
        return (input - self.mean) / (self.standard_deviation + self.eps)

    def set_standardisation(self, mean, standard_deviation):
        if not len(standard_deviation.shape) == 1 or not len(mean.shape) == 1:
            raise Exception("Input statistics are not 1-D tensors.")

        if (
            not torch.nonzero(self.standard_deviation).shape[0]
            == standard_deviation.shape[0]
        ):
            raise Exception(
                "Standard deviation in standardisation contains elements equal to 0."
            )

        self.mean = torch.nn.Parameter(data=mean, requires_grad=False)
        self.standard_deviation = torch.nn.Parameter(
            data=standard_deviation, requires_grad=False
        )


class Scale(torch.nn.Module):
    """
    Scale the output of the layer by mean and standard deviation.
    """

    def __init__(self, n_neurons):
        super(Scale, self).__init__()
        self.mean = torch.nn.Parameter(data=torch.zeros(n_neurons), requires_grad=False)
        self.standard_deviation = torch.nn.Parameter(
            data=torch.ones(n_neurons), requires_grad=False
        )
        self.eps = 1e-8

    def forward(self, input):
        return self.mean + input * self.standard_deviation

    def set_scaling(self, mean, standard_deviation):
        if not len(standard_deviation.shape) == 1 or not len(mean.shape) == 1:
            raise Exception("Input statistics are not 1-D tensors.")

        if (
            not torch.nonzero(self.standard_deviation).shape[0]
            == standard_deviation.shape[0]
        ):
            raise Exception(
                "Standard deviation in state_to_scale contains elements equal to 0."
            )
        with torch.no_grad():
            self.mean = torch.nn.Parameter(data=mean, requires_grad=False)
            self.standard_deviation = torch.nn.Parameter(
                data=standard_deviation, requires_grad=False
            )


class CoreNeuralNetwork(torch.nn.Module):
    """
    A simple multi-layer perceptron network, with optional input and
    output standardisation/state_to_scale and the computation
    of output to input sensitivities.
    """

    def __init__(
        self,
        neurons_in_layers: list,
        pytorch_init_seed: int = None,
    ):
        super(CoreNeuralNetwork, self).__init__()

        if isinstance(pytorch_init_seed, int):
            torch.manual_seed(pytorch_init_seed)

        layer_dictionary = OrderedDict()
        layer_dictionary["input_standardisation"] = Standardise(neurons_in_layers[0])
        for ii, (neurons_in, neurons_out) in enumerate(
            zip(neurons_in_layers[:-2], neurons_in_layers[1:-1])
        ):
            layer_dictionary[f"dense_{ii}"] = torch.nn.Linear(
                in_features=neurons_in,
                out_features=neurons_out,
                bias=True,
                dtype=torch.float64,
            )

            layer_dictionary[f"activation_{ii}"] = torch.nn.Tanh()
            torch.nn.init.xavier_normal_(
                layer_dictionary[f"dense_{ii}"].weight,
                gain=torch.nn.init.calculate_gain("tanh"),
            )

        layer_dictionary["output_layer"] = torch.nn.Linear(
            in_features=neurons_in_layers[-2],
            out_features=neurons_in_layers[-1],
            bias=True,
            dtype=torch.float64,
        )
        torch.nn.init.xavier_normal_(layer_dictionary["output_layer"].weight, gain=1.0)

        layer_dictionary["output_scaling"] = Scale(neurons_in_layers[-1])

        self.dense_layers = torch.nn.Sequential(layer_dictionary)

    def forward(self, NN_input):
        return self.dense_layers(NN_input)
