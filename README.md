# PINNSim: A Simulator for Power System Dynamics based on Physics-Informed Neural Networks

This repository provides the implementation to the paper [PINNSim: A Simulator for Power System Dynamics based on Physics-Informed Neural Networks](https://arxiv.org/pdf/2303.10256.pdf) that was submitted to the Power System Computing Conference (PSCC) 2024.

The following provides a brief overview on how to install the `pinnsim` package and how to get started.

## Installation

To setup the python environment, we provide a `environment.yml` that can be used to create the `pinnsim_pscc` environment
```
conda env create -f environment.yml
conda activate pinnsim_pscc
```

To then install the `pinnsim` package navigate in a terminal to the folder containing this repository -- your working directory should then contain the `pyproject.toml` file. By running the following command, `pinnsim` is installed and can be accessed using ``import pinnsim``. The addition of the flag ``-e`` indicates that the repository can be modified without re-running the install command.

```
python -m pip install -e .
```

## Getting started

To illustrate the main functionalities of the package, we provide a few examples in `jupyter_notebooks`. They contain references to central files and should help to explore the repository.

The repository has three major parts
- `power_system_models` and `numerics` describe the basic modelling of the power system and its components and how to solve the resulting differential equations.
- `dataset_functions` and `learning_functions` provide the functionality that is needed to train (learn) the neural network models and generate the require datasets.
- `configurations` contains a lot of the case specific functions and setups.

The "PINNSim algorithm" (Algorithm 1 in the paper) can be found in `pinnsim.numerics.simulators.simulator_distributed`.

General settings are stored in the ``__init__.py`` file in ``src.pinnsim``. In particular, the WandB settings needed for logging the neural network training online require adjustments.

## Reference this work 

If you find this work helpful, please cite this work
```
@article{stiasny_pinnsim_2024,
	title = {{PINNSim}: {A} simulator for power system dynamics based on {Physics}-{Informed} {Neural} {Networks}},
    author = {Stiasny, Jochen and Zhang, Baosen and Chatzivasileiadis, Spyros},
    journal = {Electric Power Systems Research},
	volume = {235},
    pages = {110796},
	doi = {10.1016/j.epsr.2024.110796},
	month = oct,
	year = {2024},
}

``` 

## License
This project is made available under the MIT License.
