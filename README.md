# DualSensing
Code for "Nonlinear channel dynamics enable cell-type specific sensing and control of firing statistics"

This repository contains the code used to simulate the neuron models in the upcoming paper by Ramirez-Hincapie and O'Leary.

The aim of the paper is to show that a simple, ubiquitous signal in cells, namely intracellular calcium concentration, provides a direct readout of firing rate statistics (mean and variance). The results show this is the case in two conductance-based models: 
- the STG used to simulate intrinsically active neurons
- an adapted version of the Connor Stevens model used to simulate input-driven neurons

The repository contains the processed data, analysis and plotting scripts all written in Julia.
Raw data files are archived on Zenodo: https://doi.org/10.5281/zenodo.19813425
The file "load_format.jl" was used to process the raw data (and requires creating "simulations" folder).

The project environment can be activated with the command "> ] activate ." from inside the main folder. The package dependencies needed to run the code are specified in Project.toml.

